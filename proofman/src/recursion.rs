use borsh::{BorshSerialize, BorshDeserialize};
use libloading::{Library, Symbol};
use proofman_fields::PrimeField64;
use std::ffi::CString;
use std::fmt;
use proofman_starks_lib_c::*;
use std::path::Path;
use std::fs::File;
use std::io::Write;

use proofman_common::{
    CurveType, MpiCtx, MemoryHandlerRecursive, Proof, ProofCtx, ProofType, ProofmanResult, ProofmanError, Setup,
    SetupsVadcop, GetSizeWitnessFunc,
};

use std::os::raw::{c_void, c_char};

use proofman_util::{
    timer_start_info, timer_stop_and_log_info, timer_start_debug, timer_stop_and_log_debug,
    timer_stop_and_log_debug_net, create_buffer_fast,
};

use crate::{add_publics_circom, add_publics_aggregation};

pub type GetWitnessFunc =
    unsafe extern "C" fn(zkin: *mut u64, circom_circuit: *mut c_void, witness: *mut c_void, n_mutexes: u64) -> i64;

pub type GetWitnessFinalFunc =
    unsafe extern "C" fn(zkin: *mut c_void, dat_file: *const c_char, witness: *mut c_void, n_mutexes: u64) -> i64;

/// Joins a background FFI thread on drop, so an early `?` or panic can't detach a thread still
/// writing shared device state (the const-tree buffer) and let the next proof race it.
pub struct JoinOnDrop(Option<std::thread::JoinHandle<()>>);

impl JoinOnDrop {
    pub fn new(handle: std::thread::JoinHandle<()>) -> Self {
        Self(Some(handle))
    }

    /// Join now and return the thread's result; consumes the guard so `Drop` will not re-join.
    pub fn join_now(mut self) -> std::thread::Result<()> {
        match self.0.take() {
            Some(handle) => handle.join(),
            None => Ok(()),
        }
    }
}

impl Drop for JoinOnDrop {
    fn drop(&mut self) {
        if let Some(handle) = self.0.take() {
            let _ = handle.join();
        }
    }
}

#[derive(Debug, BorshSerialize, BorshDeserialize)]
pub struct AggProofsRegister {
    pub airgroup_id: u64,
    pub worker_indexes: Vec<usize>,
}

impl AggProofsRegister {
    pub fn new(airgroup_id: u64, worker_indexes: Vec<usize>) -> Self {
        Self { airgroup_id, worker_indexes }
    }
}

#[derive(BorshSerialize, BorshDeserialize)]
pub struct AggProofs {
    pub airgroup_id: u64,
    pub proof: Vec<u64>,
    pub worker_indexes: Vec<usize>,
}

impl AggProofs {
    pub fn new(airgroup_id: u64, proof: Vec<u64>, worker_indexes: Vec<usize>) -> Self {
        Self { airgroup_id, proof, worker_indexes }
    }
}

impl fmt::Display for AggProofs {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "AggProofs {{ airgroup_id: {}, worker_indexes: {:?} }}", self.airgroup_id, self.worker_indexes)
    }
}

impl fmt::Debug for AggProofs {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "AggProofs {{ airgroup_id: {}, worker_indexes: {:?} }}", self.airgroup_id, self.worker_indexes)
    }
}

pub fn gen_witness_recursive<F: PrimeField64>(
    pctx: &ProofCtx<F>,
    memory_handler_recursive_witness: &MemoryHandlerRecursive<F>,
    setups: &SetupsVadcop<F>,
    proof: &Proof<F>,
) -> ProofmanResult<Proof<F>> {
    let (airgroup_id, air_id) = (proof.airgroup_id, proof.air_id);

    if proof.proof_type != ProofType::Basic && proof.proof_type != ProofType::Compressor {
        return Err(ProofmanError::InvalidProof(format!(
            "Invalid proof type {:?} for airgroup_id {} air_id {}. Must be Basic or Compressor",
            proof.proof_type, airgroup_id, air_id
        )));
    }

    let has_compressor = pctx.global_info.get_air_has_compressor(airgroup_id, air_id);
    if proof.proof_type == ProofType::Basic && has_compressor {
        timer_start_debug!(
            GENERATE_COMPRESSOR_WITNESS,
            "GENERATING_COMPRESSOR_WITNESS_{} [{}:{}]",
            proof.global_idx.unwrap(),
            proof.airgroup_id,
            proof.air_id
        );
        let setup = setups.sctx_compressor.as_ref().unwrap().get_setup(airgroup_id, air_id)?;

        let publics_circom_size =
            pctx.global_info.n_publics + pctx.global_info.n_proof_values.iter().sum::<usize>() * 3 + 3;

        let mut updated_proof: Vec<u64> = vec![0; proof.proof.len() + publics_circom_size];
        updated_proof[publics_circom_size..].copy_from_slice(&proof.proof);
        add_publics_circom(&mut updated_proof, 0, pctx, None);
        let circom_witness =
            generate_witness::<F>(setup, memory_handler_recursive_witness, proof.global_idx.unwrap(), &updated_proof)?;
        timer_stop_and_log_debug_net!(
            GENERATE_COMPRESSOR_WITNESS,
            proofman_common::take_buffer_wait(circom_witness.as_ptr() as *const u8),
            "GENERATING_COMPRESSOR_WITNESS_{} [{}:{}]",
            proof.global_idx.unwrap(),
            proof.airgroup_id,
            proof.air_id
        );
        Ok(Proof::new_witness(
            ProofType::Compressor,
            airgroup_id,
            air_id,
            proof.global_idx,
            circom_witness,
            setup.n_cols as usize,
        ))
    } else {
        timer_start_debug!(
            GENERATE_RECURSIVE1_WITNESS,
            "GENERATING_RECURSIVE1_WITNESS_{} [{}:{}]",
            proof.global_idx.unwrap(),
            proof.airgroup_id,
            proof.air_id
        );
        let setup = setups.sctx_recursive1.as_ref().unwrap().get_setup(airgroup_id, air_id)?;

        let publics_circom_size =
            pctx.global_info.n_publics + pctx.global_info.n_proof_values.iter().sum::<usize>() * 3 + 3 + 4;
        let recursive2_setup = setups.sctx_recursive2.as_ref().unwrap().get_setup(airgroup_id, 0)?;

        let mut updated_proof: Vec<u64> = vec![0; proof.proof.len() + publics_circom_size];

        if proof.proof_type == ProofType::Compressor {
            let n_publics_aggregation = n_publics_aggregation(pctx, airgroup_id);
            let publics_aggregation: Vec<F> =
                proof.proof.iter().take(n_publics_aggregation).map(|&x| F::from_u64(x)).collect();
            add_publics_aggregation(&mut updated_proof, 0, &publics_aggregation, n_publics_aggregation);
            add_publics_circom(&mut updated_proof, n_publics_aggregation, pctx, Some(&recursive2_setup.verkey));
            updated_proof[(publics_circom_size + n_publics_aggregation)..]
                .copy_from_slice(&proof.proof[n_publics_aggregation..]);
        } else {
            updated_proof[publics_circom_size..].copy_from_slice(&proof.proof);
            add_publics_circom(&mut updated_proof, 0, pctx, Some(&recursive2_setup.verkey));
        }

        let circom_witness =
            generate_witness::<F>(setup, memory_handler_recursive_witness, proof.global_idx.unwrap(), &updated_proof)?;
        timer_stop_and_log_debug_net!(
            GENERATE_RECURSIVE1_WITNESS,
            proofman_common::take_buffer_wait(circom_witness.as_ptr() as *const u8),
            "GENERATING_RECURSIVE1_WITNESS_{} [{}:{}]",
            proof.global_idx.unwrap(),
            proof.airgroup_id,
            proof.air_id
        );
        Ok(Proof::new_witness(
            ProofType::Recursive1,
            airgroup_id,
            air_id,
            proof.global_idx,
            circom_witness,
            setup.n_cols as usize,
        ))
    }
}

pub fn gen_witness_aggregation<F: PrimeField64>(
    pctx: &ProofCtx<F>,
    memory_handler_recursive_witness: &MemoryHandlerRecursive<F>,
    setups: &SetupsVadcop<F>,
    proofs: &[&Proof<F>],
) -> ProofmanResult<Proof<F>> {
    timer_start_debug!(GENERATE_WITNESS_AGGREGATION);
    let arity = pctx.global_info.aggregation_arity;
    if proofs.len() != arity {
        return Err(ProofmanError::ProofmanError(format!("Aggregation expects {arity} proofs, got {}", proofs.len())));
    }

    let proof_len = proofs[0].proof.len();
    if let Some((i, p)) = proofs.iter().enumerate().find(|(_, p)| p.proof.len() != proof_len) {
        return Err(ProofmanError::ProofmanError(format!(
            "Inconsistent proof sizes: proof 0 size {proof_len}, proof {i} size {}",
            p.proof.len()
        )));
    }

    let airgroup_id = proofs[0].airgroup_id;
    if let Some((i, p)) = proofs.iter().enumerate().find(|(_, p)| p.airgroup_id != airgroup_id) {
        return Err(ProofmanError::ProofmanError(format!(
            "Inconsistent airgroup_ids: proof 0 airgroup_id {airgroup_id}, proof {i} airgroup_id {}",
            p.airgroup_id
        )));
    }

    let publics_circom_size: usize =
        pctx.global_info.n_publics + pctx.global_info.n_proof_values.iter().sum::<usize>() * 3 + 3 + 4;

    let setup_recursive2 = setups.sctx_recursive2.as_ref().unwrap().get_setup(airgroup_id, 0)?;

    let updated_proof_size = arity * proof_len + publics_circom_size;

    let mut updated_proof_recursive2: Vec<u64> = vec![0; updated_proof_size];
    for (i, p) in proofs.iter().enumerate() {
        let start = publics_circom_size + i * proof_len;
        updated_proof_recursive2[start..start + proof_len].copy_from_slice(&p.proof);
    }

    add_publics_circom(&mut updated_proof_recursive2, 0, pctx, Some(&setup_recursive2.verkey));
    let circom_witness =
        generate_witness::<F>(setup_recursive2, memory_handler_recursive_witness, 0, &updated_proof_recursive2)?;

    timer_stop_and_log_debug_net!(
        GENERATE_WITNESS_AGGREGATION,
        proofman_common::take_buffer_wait(circom_witness.as_ptr() as *const u8),
        "GENERATE_WITNESS_AGGREGATION"
    );
    Ok(Proof::new_witness(
        ProofType::Recursive2,
        airgroup_id,
        0,
        None,
        circom_witness,
        setup_recursive2.n_cols as usize,
    ))
}

pub fn n_publics_aggregation<F: PrimeField64>(pctx: &ProofCtx<F>, airgroup_id: usize) -> usize {
    let mut publics_aggregation = 0;
    publics_aggregation += 1; // circuit type
    publics_aggregation += 1; // n proofs aggregated
    publics_aggregation += 4 * pctx.global_info.agg_types[airgroup_id].len(); // agg types
    if pctx.global_info.curve != CurveType::None {
        publics_aggregation += 10; // elliptic curve hash
    } else {
        publics_aggregation += pctx.global_info.lattice_size.unwrap(); // lattice components
    }
    publics_aggregation
}

pub fn get_accumulated_challenge<F: PrimeField64>(pctx: &ProofCtx<F>, proof: &[u64]) -> Vec<u64> {
    if pctx.global_info.curve != CurveType::None {
        proof[6..16].to_vec()
    } else {
        proof[6..6 + pctx.global_info.lattice_size.unwrap()].to_vec()
    }
}

pub fn gen_recursive_proof_size<F: PrimeField64>(
    pctx: &ProofCtx<F>,
    setups: &SetupsVadcop<F>,
    witness: &Proof<F>,
) -> ProofmanResult<Proof<F>> {
    let (airgroup_id, air_id) = (witness.airgroup_id, witness.air_id);

    let setup = setups.get_setup(airgroup_id, air_id, &witness.proof_type)?;

    let mut new_proof_size = setup.proof_size;

    let publics_aggregation = n_publics_aggregation(pctx, airgroup_id);

    if witness.proof_type != ProofType::VadcopFinal && witness.proof_type != ProofType::VadcopFinalCompressed {
        new_proof_size += publics_aggregation as u64;
    } else {
        new_proof_size += 1 + setup.stark_info.n_publics;
    }

    let new_proof = create_buffer_fast(new_proof_size as usize);
    Ok(Proof::new(witness.proof_type, witness.airgroup_id, witness.air_id, witness.global_idx, new_proof))
}

/// Writes a vadcop-final proof's public section `[n_publics | publics(n_publics)]`. `publics` are
/// the circuit's OUTPUT publics (flag `is_vadcop_final_proof` at index 0), not the input publics.
fn write_vadcop_final_publics<F: PrimeField64>(proof: &mut Proof<F>, n_publics: u64, publics: &[F]) {
    proof.proof[0] = n_publics;
    for (i, p) in publics.iter().take(n_publics as usize).enumerate() {
        proof.proof[1 + i] = p.as_canonical_u64();
    }
}

#[allow(clippy::too_many_arguments)]
pub fn generate_recursive_proof<F: PrimeField64>(
    pctx: &ProofCtx<F>,
    memory_handler_recursive_witness: &MemoryHandlerRecursive<F>,
    setups: &SetupsVadcop<F>,
    witness: &mut Proof<F>,
    new_proof: &Proof<F>,
    prover_buffer: &[F],
    const_tree: &[F],
    const_pols: &[F],
    force_recursive_stream: bool,
    reserved_stream: u64,
    calculate_fixed_tree_handle: Option<JoinOnDrop>,
) -> ProofmanResult<(u64, Vec<F>)> {
    timer_start_debug!(
        GEN_RECURSIVE_PROOF,
        "GEN_RECURSIVE_PROOF_{:?} [{}:{}]",
        witness.proof_type,
        witness.airgroup_id,
        witness.air_id
    );

    let (airgroup_id, air_id, instance_id, vadcop) =
        if witness.proof_type == ProofType::VadcopFinal || witness.proof_type == ProofType::VadcopFinalCompressed {
            (0, 0, 0, false)
        } else {
            (witness.airgroup_id, witness.air_id, witness.global_idx.unwrap(), true)
        };

    // Adopt the witness buffer into a release-on-drop lease so it returns to its pool on every exit
    // path (`?`, downstream error, panic) instead of leaking when the proof is dropped on cancel.
    let circom_witness = memory_handler_recursive_witness.adopt_witness(std::mem::take(&mut witness.circom_witness));

    let setup = setups.get_setup(airgroup_id, air_id, &witness.proof_type)?;

    // Where stage 1 is built. On GPU the fill is compact and the device widens it; on CPU it is the
    // air's full cm1 width. Both lease it: aliasing the prover's own cm1 slot would need that
    // section's aux_trace offset, which only C++ `StarkInfo::setMapOffsets` computes.
    let mut trace_lease = memory_handler_recursive_witness.take_trace_lease();
    let trace_ptr = trace_lease.as_mut_ptr() as *mut u8;

    let p_setup: *mut c_void = (&setup.p_setup).into();

    let mut publics = vec![F::ZERO; setup.stark_info.n_publics as usize];

    let exec = setup.exec_data.as_ref().expect("exec_data missing on setup");
    let exec_data_ptr = exec.as_ptr() as *mut u64;
    let exec_words = exec.len() as u64;

    get_committed_pols_c(
        circom_witness.as_ptr() as *mut u8,
        exec_data_ptr,
        trace_ptr,
        publics.as_mut_ptr() as *mut u8,
        setup.size_witness.unwrap(),
        1 << (setup.stark_info.stark_struct.n_bits),
        setup.stark_info.n_publics,
        recursion_trace_stride(exec, witness.n_cols as u64, pctx.gpu),
    );
    // The hash gates map only their boundary; the rest is rebuilt from it. On GPU that happens
    // device-side inside gen_recursive_proof_c, right after the trace copy.
    if !pctx.gpu {
        expand_gate_bands_c(
            trace_ptr,
            exec_data_ptr,
            witness.n_cols as u64,
            exec_words,
            1 << setup.stark_info.stark_struct.n_bits,
        );
    }
    // Witness no longer needed; drop the lease to return it to its pool now.
    drop(circom_witness);

    let publics_aggregation = n_publics_aggregation(pctx, airgroup_id);

    let initial_idx =
        if witness.proof_type == ProofType::VadcopFinal || witness.proof_type == ProofType::VadcopFinalCompressed {
            1 + setup.stark_info.n_publics as usize
        } else {
            publics_aggregation
        };

    if witness.proof_type != ProofType::VadcopFinal && witness.proof_type != ProofType::VadcopFinalCompressed {
        add_publics_aggregation_c(
            new_proof.proof.as_ptr() as *mut u8,
            0,
            publics.as_ptr() as *mut u8,
            publics_aggregation as u64,
        );
    }
    // For VadcopFinal / VadcopFinalCompressed the caller writes the public section from the `publics`
    // returned below — the circuit's OUTPUT publics (flag at index 0), NOT `pctx.get_publics()`, which
    // holds only the flag-free INPUT publics and is one element too short.

    let (const_pols_ptr, const_tree_ptr) = if pctx.gpu {
        (std::ptr::null_mut(), std::ptr::null_mut())
    } else {
        (const_pols.as_ptr() as *mut u8, const_tree.as_ptr() as *mut u8)
    };

    if let Some(handle) = calculate_fixed_tree_handle {
        handle.join_now().map_err(|_| ProofmanError::ProofmanError("Failed to calculate fixed tree".into()))?;
    }

    // `reserved_stream`: scheduler-reserved stream, or `u64::MAX` to select internally
    // (one-off launches — outer aggregation, vadcop_final, recursers).
    let stream_id = gen_recursive_proof_c(
        p_setup,
        trace_ptr,
        prover_buffer.as_ptr() as *mut u8,
        const_pols_ptr,
        const_tree_ptr,
        publics.as_ptr() as *mut u8,
        new_proof.proof[initial_idx..].as_ptr() as *mut u64,
        "",
        airgroup_id as u64,
        air_id as u64,
        instance_id as u64,
        vadcop,
        pctx.get_device_buffers_ptr(),
        &setup.const_pols_path,
        &setup.const_pols_tree_path,
        witness.proof_type.into(),
        force_recursive_stream,
        "",
        reserved_stream, // scheduler-reserved stream, or u64::MAX for one-off internal selection
    );

    // Trace H2D is async: gate reuse on the stream's commit event so a concurrent take() can't
    // overwrite `trace` mid-copy. Must finish before the lease drops at scope exit and pools the trace.
    if pctx.gpu {
        wait_trace_h2d_done_c(pctx.get_device_buffers_ptr(), stream_id);
    }

    timer_stop_and_log_debug!(
        GEN_RECURSIVE_PROOF,
        "GEN_RECURSIVE_PROOF_{:?} [{}:{}]",
        witness.proof_type,
        witness.airgroup_id,
        witness.air_id
    );
    Ok((stream_id, publics))
}

#[allow(clippy::too_many_arguments)]
#[allow(clippy::type_complexity)]
pub fn aggregate_worker_proofs<F: PrimeField64>(
    pctx: &ProofCtx<F>,
    memory_handler_recursive_witness: &MemoryHandlerRecursive<F>,
    mpi_ctx: &MpiCtx,
    setups: &SetupsVadcop<F>,
    mut proofs: Vec<Vec<Proof<F>>>,
    prover_buffer: &[F],
    const_pols: &[F],
    const_tree: &[F],
    agg_proofs: &mut Vec<AggProofs>,
) -> ProofmanResult<()> {
    let n_processes = mpi_ctx.n_processes as usize;
    let rank = mpi_ctx.rank as usize;
    let n_airgroups = pctx.global_info.air_groups.len();
    let mut alives = vec![0; n_airgroups];
    let mut airgroup_proofs: Vec<Vec<Option<Vec<u64>>>> = Vec::with_capacity(n_airgroups);

    let mut null_proofs: Vec<Vec<u64>> = vec![Vec::new(); n_airgroups];

    let instances = pctx.dctx_get_instances();
    let mut airgroup_instances_alive = vec![vec![0; n_processes]; n_airgroups];
    for global_id in pctx.dctx_get_worker_instances().iter() {
        if let Ok(owner) = pctx.dctx_get_process_owner_instance(*global_id) {
            airgroup_instances_alive[instances[*global_id].airgroup_id][owner as usize] = 1;
        }
    }

    // Pre-process data before starting recursion loop
    for (airgroup, instances) in airgroup_instances_alive.iter().enumerate().take(n_airgroups) {
        let mut current_pos = 0;
        for (p, &alive) in instances.iter().enumerate().take(n_processes) {
            if p < rank {
                current_pos += alive;
            }
            alives[airgroup] += alive;
        }
        let setup = setups.get_setup(airgroup, 0, &ProofType::Recursive2)?;
        let publics_aggregation = n_publics_aggregation(pctx, airgroup);
        null_proofs[airgroup] = vec![0; setup.proof_size as usize + publics_aggregation];
        airgroup_proofs.push(vec![None; alives[airgroup]]);

        if !proofs[airgroup].is_empty() {
            for i in 0..proofs[airgroup].len() {
                airgroup_proofs[airgroup][current_pos + i] = Some(std::mem::take(&mut proofs[airgroup][i].proof));
            }
        } else if rank == 0 {
            airgroup_proofs[airgroup][0] = Some(vec![0; setup.proof_size as usize + publics_aggregation]);
        }
    }

    let arity = pctx.global_info.aggregation_arity;

    // agregation loop
    loop {
        mpi_ctx.barrier();
        mpi_ctx.distribute_recursive2_proofs(&alives, &mut airgroup_proofs, arity);
        let mut pending_agregations = false;
        for airgroup in 0..n_airgroups {
            //create a vector of sice indices length
            let mut alive = alives[airgroup];
            if alive > 1 {
                let n_agg_proofs = alive / arity;
                let n_remaining_proofs = alive % arity;
                for i in 0..alive.div_ceil(arity) {
                    let j = i * arity;
                    if airgroup_proofs[airgroup][j].is_none() {
                        continue;
                    }
                    if (j + arity - 1 < alive) || alive <= arity {
                        // A chunk needs at least two real proofs; the remaining
                        // slots are null-padded. At arity 2 padding never happens.
                        if airgroup_proofs[airgroup][j + 1].is_none() {
                            return Err(ProofmanError::ProofmanError("Recursive2 proof is missing".into()));
                        }

                        let chunk: Vec<Proof<F>> = (0..arity)
                            .map(|k| {
                                let slot = j + k;
                                let data = if slot < alive {
                                    airgroup_proofs[airgroup][slot]
                                        .take()
                                        .expect("chunk slot within alive must hold a proof")
                                } else {
                                    null_proofs[airgroup].clone()
                                };
                                Proof::new(ProofType::Recursive2, airgroup, 0, None, data)
                            })
                            .collect();
                        let chunk_refs: Vec<&Proof<F>> = chunk.iter().collect();

                        let mut circom_witness =
                            gen_witness_aggregation::<F>(pctx, memory_handler_recursive_witness, setups, &chunk_refs)?;
                        circom_witness.global_idx = Some(rank);

                        let recursive2_proof = match gen_recursive_proof_size::<F>(pctx, setups, &circom_witness) {
                            Ok(p) => p,
                            Err(e) => {
                                // generate_recursive_proof (which pools the witness) isn't reached;
                                // return it here instead of leaking.
                                drop(
                                    memory_handler_recursive_witness
                                        .adopt_witness(std::mem::take(&mut circom_witness.circom_witness)),
                                );
                                return Err(e);
                            }
                        };

                        let (stream_id, _) = generate_recursive_proof::<F>(
                            pctx,
                            memory_handler_recursive_witness,
                            setups,
                            &mut circom_witness,
                            &recursive2_proof,
                            prover_buffer,
                            const_tree,
                            const_pols,
                            false,
                            u64::MAX, // one-off launch: reserve stream internally
                            None,
                        )?;

                        get_stream_id_proof_c(pctx.get_device_buffers_ptr(), stream_id);

                        airgroup_proofs[airgroup][j] = Some(recursive2_proof.proof);

                        tracing::debug!("··· Recursive 2 Proof generated.");
                    }
                }
                if n_agg_proofs > 0 {
                    alive = n_agg_proofs + n_remaining_proofs;
                } else {
                    alive = 1;
                }

                //compact elements
                for i in 0..n_agg_proofs {
                    airgroup_proofs[airgroup][i] = airgroup_proofs[airgroup][i * arity].take();
                }

                for i in 0..n_remaining_proofs {
                    airgroup_proofs[airgroup][n_agg_proofs + i] =
                        airgroup_proofs[airgroup][arity * n_agg_proofs + i].take();
                }
                alives[airgroup] = alive;
                if alive > 1 {
                    pending_agregations = true;
                }
            }
        }
        if !pending_agregations {
            break;
        }
    }

    if pctx.mpi_ctx.rank == 0 {
        let worker_index = pctx.get_worker_index()?;
        for (airgroup_id, (&alive, proofs)) in alives.iter().zip(airgroup_proofs.iter_mut()).enumerate() {
            proofs.iter_mut().take(alive).filter_map(|p| p.take()).for_each(|proof| {
                agg_proofs.push(AggProofs::new(airgroup_id as u64, proof, vec![worker_index]));
            });
        }
    }

    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub fn generate_vadcop_final_proof<F: PrimeField64>(
    pctx: &ProofCtx<F>,
    memory_handler_recursive_witness: &MemoryHandlerRecursive<F>,
    setups: &SetupsVadcop<F>,
    agg_proofs: &[AggProofs],
    prover_buffer: &[F],
    const_pols: &[F],
    const_tree: &[F],
) -> ProofmanResult<Proof<F>> {
    // Phase B: the aliased recursive streams overlay the basic stream's buffer, which
    // VadcopFinal needs back. No-op when phase B is not configured.
    if pctx.gpu {
        let _ = set_phase_b_c(pctx.get_device_buffers_ptr(), 2);
    }
    timer_start_info!(GENERATE_VADCOP_FINAL_PROOF);
    let publics_circom_size =
        pctx.global_info.n_publics + pctx.global_info.n_proof_values.iter().sum::<usize>() * 3 + 3;

    let n_airgroups = pctx.global_info.air_groups.len();

    let mut updated_proof_size = publics_circom_size;

    let setup = setups.setup_vadcop_final.as_ref().unwrap();
    let p_setup: *mut c_void = (&setup.p_setup).into();

    let p_setup_addr = p_setup as usize;
    let device_buffers_addr = pctx.get_device_buffers_ptr() as usize;
    let setup_type = setup.setup_type;

    // JoinOnDrop so an early `?` below joins this const-tree thread instead of detaching it.
    let calculate_fixed_tree_handle = JoinOnDrop::new(std::thread::spawn(move || {
        calculate_const_tree_fixed_c(
            p_setup_addr as *mut c_void,
            0,
            0,
            setup_type.into(),
            device_buffers_addr as *mut c_void,
        );
    }));

    for airgroup_id in 0..n_airgroups {
        let setup = setups.get_setup(airgroup_id, 0, &ProofType::Recursive2)?;
        let publics_aggregation = n_publics_aggregation(pctx, airgroup_id);
        updated_proof_size += setup.proof_size as usize + publics_aggregation;
    }

    let mut updated_proof = vec![0; updated_proof_size];
    add_publics_circom(&mut updated_proof, 0, pctx, None);

    let mut offset = publics_circom_size;
    for airgroup_id in 0..n_airgroups {
        let setup = setups.get_setup(airgroup_id, 0, &ProofType::Recursive2)?;
        let publics_aggregation = n_publics_aggregation(pctx, airgroup_id);
        let proof_size = setup.proof_size as usize + publics_aggregation;
        if let Some(ap) = agg_proofs.iter().find(|ap| ap.airgroup_id as usize == airgroup_id) {
            if ap.proof.len() != proof_size {
                return Err(ProofmanError::ProofmanError(format!(
                    "Invalid proof size for airgroup_id {}. Expected {}, got {}",
                    airgroup_id,
                    proof_size,
                    ap.proof.len()
                )));
            }
            updated_proof[offset..offset + proof_size].copy_from_slice(&ap.proof);
        } else {
            let null_proof = vec![0; proof_size];
            updated_proof[offset..offset + proof_size].copy_from_slice(&null_proof);
        }
        offset += proof_size;
    }

    timer_start_debug!(GENERATE_VADCOP_FINAL_PROOF_WITNESS);
    let circom_witness_vadcop_final =
        generate_witness::<F>(setup, memory_handler_recursive_witness, 0, &updated_proof)?;
    timer_stop_and_log_debug!(GENERATE_VADCOP_FINAL_PROOF_WITNESS);
    let mut witness_final_proof =
        Proof::new_witness(ProofType::VadcopFinal, 0, 0, None, circom_witness_vadcop_final, setup.n_cols as usize);

    let mut final_proof = match gen_recursive_proof_size::<F>(pctx, setups, &witness_final_proof) {
        Ok(p) => p,
        Err(e) => {
            // generate_recursive_proof (which pools the witness) isn't reached; return it here instead of leaking.
            drop(
                memory_handler_recursive_witness.adopt_witness(std::mem::take(&mut witness_final_proof.circom_witness)),
            );
            return Err(e);
        }
    };
    let (stream_id, publics) = generate_recursive_proof::<F>(
        pctx,
        memory_handler_recursive_witness,
        setups,
        &mut witness_final_proof,
        &final_proof,
        prover_buffer,
        const_tree,
        const_pols,
        false,
        u64::MAX, // one-off launch: reserve stream internally
        Some(calculate_fixed_tree_handle),
    )?;
    get_stream_id_proof_c(pctx.get_device_buffers_ptr(), stream_id);

    // Write the public section from the circuit's OUTPUT publics returned by generate_recursive_proof
    // (flag at index 0), NOT `pctx.get_publics()`, which holds only the flag-free input publics.
    write_vadcop_final_publics(&mut final_proof, setup.stark_info.n_publics, &publics);

    timer_stop_and_log_info!(GENERATE_VADCOP_FINAL_PROOF);

    Ok(final_proof)
}

#[allow(clippy::too_many_arguments)]
pub fn generate_vadcop_final_compressed_proof<F: PrimeField64>(
    pctx: &ProofCtx<F>,
    memory_handler_recursive_witness: &MemoryHandlerRecursive<F>,
    setups: &SetupsVadcop<F>,
    vadcop_final_proof: &[u64],
    prover_buffer: &[F],
    const_pols: &[F],
    const_tree: &[F],
) -> ProofmanResult<Proof<F>> {
    if pctx.gpu {
        let _ = set_phase_b_c(pctx.get_device_buffers_ptr(), 2);
    }
    timer_start_info!(GENERATE_VADCOP_FINAL_COMPRESSED_PROOF);
    let setup = setups.setup_vadcop_final_compressed.as_ref().ok_or_else(|| {
        ProofmanError::InvalidConfiguration(
            "Proving key was built without the vadcop_final_compressed stage; no compressed final setup to prove with"
                .to_string(),
        )
    })?;

    let p_setup: *mut c_void = (&setup.p_setup).into();

    let p_setup_addr = p_setup as usize;
    let device_buffers_addr = pctx.get_device_buffers_ptr() as usize;
    let setup_type = setup.setup_type;

    // JoinOnDrop so an early `?` below joins this const-tree thread instead of detaching it.
    let calculate_fixed_tree_handle = JoinOnDrop::new(std::thread::spawn(move || {
        calculate_const_tree_fixed_c(
            p_setup_addr as *mut c_void,
            0,
            0,
            setup_type.into(),
            device_buffers_addr as *mut c_void,
        );
    }));

    timer_start_debug!(GENERATE_VADCOP_FINAL_COMPRESSED_PROOF_WITNESS);
    let circom_witness_vadcop_final_compressed =
        generate_witness::<F>(setup, memory_handler_recursive_witness, 0, &vadcop_final_proof[1..])?;
    timer_stop_and_log_debug!(GENERATE_VADCOP_FINAL_COMPRESSED_PROOF_WITNESS);
    let mut witness_final_proof = Proof::new_witness(
        ProofType::VadcopFinalCompressed,
        0,
        0,
        None,
        circom_witness_vadcop_final_compressed,
        setup.n_cols as usize,
    );

    let mut final_proof = match gen_recursive_proof_size::<F>(pctx, setups, &witness_final_proof) {
        Ok(p) => p,
        Err(e) => {
            // generate_recursive_proof (which pools the witness) isn't reached; return it here instead of leaking.
            drop(
                memory_handler_recursive_witness.adopt_witness(std::mem::take(&mut witness_final_proof.circom_witness)),
            );
            return Err(e);
        }
    };
    let (stream_id, publics) = generate_recursive_proof::<F>(
        pctx,
        memory_handler_recursive_witness,
        setups,
        &mut witness_final_proof,
        &final_proof,
        prover_buffer,
        const_tree,
        const_pols,
        false,
        u64::MAX, // one-off launch: reserve stream internally
        Some(calculate_fixed_tree_handle),
    )?;
    get_stream_id_proof_c(pctx.get_device_buffers_ptr(), stream_id);

    // Write the compressed proof's public section from the circuit's OUTPUT publics
    // returned by `generate_recursive_proof`, not from `pctx.get_publics()`.
    write_vadcop_final_publics(&mut final_proof, setup.stark_info.n_publics, &publics);

    timer_stop_and_log_info!(GENERATE_VADCOP_FINAL_COMPRESSED_PROOF);

    Ok(final_proof)
}

#[allow(clippy::too_many_arguments)]
pub fn generate_recursivef_proof<F: PrimeField64>(
    setup: &Setup<F>,
    memory_handler_recursive_witness: &MemoryHandlerRecursive<F>,
    vadcop_proof: &[u64],
    prover_buffer: &[F],
    const_pols: &[F],
    const_tree: &[F],
    vadcop_final_verkey: &[u64],
    prover_buffer_size: usize,
    d_buffers_recursivef: *mut c_void,
) -> ProofmanResult<*mut c_void> {
    timer_start_info!(GENERATE_RECURSIVEF);
    let p_setup: *mut c_void = (&setup.p_setup).into();

    // Cast pointers to usize to make them Send-safe for threading
    let p_setup_addr = p_setup as usize;
    let const_tree_ptr_addr = const_tree.as_ptr() as usize;
    let d_buffers_addr = d_buffers_recursivef as usize;

    // JoinOnDrop so an early `?` below joins this loader instead of detaching it.
    let load_fixed_pols_handle = JoinOnDrop::new(std::thread::spawn(move || {
        timer_start_debug!(LOAD_FIXED_POLS_RECURSIVEF);
        load_fixed_pols_recursivef_c(
            p_setup_addr as *mut c_void,
            const_tree_ptr_addr as *mut c_void,
            d_buffers_addr as *mut c_void,
        );
        timer_stop_and_log_debug!(LOAD_FIXED_POLS_RECURSIVEF);
    }));

    // Release-on-drop lease: returns to the pool on every exit path (see generate_recursive_proof).
    let mut trace = memory_handler_recursive_witness.take_trace_lease();

    let proof = &vadcop_proof[1..];
    let mut updated_proof: Vec<u64> = vec![0; proof.len() + 4];

    updated_proof[..4].copy_from_slice(&vadcop_final_verkey[..4]);

    updated_proof[4..].copy_from_slice(proof);

    timer_start_debug!(GENERATE_RECURSIVEF_WITNESS);
    let circom_witness = generate_witness::<F>(setup, memory_handler_recursive_witness, 0, &updated_proof)?;
    timer_stop_and_log_debug!(GENERATE_RECURSIVEF_WITNESS);

    let mut publics = vec![F::ZERO; setup.stark_info.n_publics as usize];

    let exec = setup.exec_data.as_ref().expect("exec_data missing on RecursiveF setup");
    let exec_data_ptr = exec.as_ptr() as *mut u64;
    let exec_words = exec.len() as u64;

    get_committed_pols_c(
        circom_witness.as_ptr() as *mut u8,
        exec_data_ptr,
        trace.as_mut_ptr() as *mut u8,
        publics.as_mut_ptr() as *mut u8,
        setup.size_witness.unwrap(),
        1 << (setup.stark_info.stark_struct.n_bits),
        setup.stark_info.n_publics,
        // Full width, NOT recursion_trace_stride: the expander below runs on the HOST here
        // unconditionally (gen_recursive_proof_final_c has no device one), and it rebuilds the
        // interiors in place, so it needs the real layout.
        setup.stark_info.map_sections_n["cm1"],
    );
    // Host-side unconditionally: gen_recursive_proof_final_c has no device expander, and
    // RecursiveF is a bn128 circuit whose exec carries no bands, so this is a no-op there.
    expand_gate_bands_c(
        trace.as_mut_ptr() as *mut u8,
        exec_data_ptr,
        setup.stark_info.map_sections_n["cm1"],
        exec_words,
        1 << setup.stark_info.stark_struct.n_bits,
    );
    memory_handler_recursive_witness.release_buffer_witness(circom_witness)?;

    timer_start_debug!(GENERATE_RECURSIVEF_PROOF);
    // prove
    let p_prove = gen_recursive_proof_final_c(
        p_setup,
        trace.as_ptr() as *mut u8,
        prover_buffer.as_ptr() as *mut u8,
        const_pols.as_ptr() as *mut u8,
        const_tree.as_ptr() as *mut u8,
        publics.as_ptr() as *mut u8,
        "",
        0,
        0,
        0,
        prover_buffer_size as u64,
        d_buffers_recursivef as *mut u8,
    );
    // `trace` (a lease) is pooled on drop at scope exit; the proof waited on the copy event, so it's safe.
    timer_stop_and_log_debug!(GENERATE_RECURSIVEF_PROOF);

    // Join the background thread (should be done by now since proof waited for copy event)
    if let Err(e) = load_fixed_pols_handle.join_now() {
        tracing::warn!("Fixed pols loading thread panicked: {:?}", e);
    }

    timer_stop_and_log_info!(GENERATE_RECURSIVEF);

    Ok(p_prove)
}

#[allow(clippy::too_many_arguments)]
pub fn generate_recurser_aggregator_proof<F: PrimeField64>(
    setup: &Setup<F>,
    memory_handler_recursive_witness: &MemoryHandlerRecursive<F>,
    proof_a: &[u64],
    proof_b: &[u64],
    free_inputs_a: &[u64],
    free_inputs_b: &[u64],
    root_c_recurser_agg: &[u64; 4],
    prover_buffer: &[F],
    const_pols: &[F],
    const_tree: &[F],
    d_buffers: *mut c_void,
    recurser_id: &str,
) -> ProofmanResult<Vec<u64>> {
    timer_start_info!(GENERATE_RECURSER_AGGREGATOR);
    let p_setup: *mut c_void = (&setup.p_setup).into();

    let p_setup_addr = p_setup as usize;
    let device_buffers_addr = d_buffers as usize;
    let setup_type = setup.setup_type;
    // JoinOnDrop so an early `?` below joins this const-tree thread instead of detaching it.
    let calc_handle = JoinOnDrop::new(std::thread::spawn(move || {
        calculate_const_tree_fixed_c(
            p_setup_addr as *mut c_void,
            0,
            0,
            setup_type.into(),
            device_buffers_addr as *mut c_void,
        );
    }));

    let mut zkin: Vec<u64> =
        Vec::with_capacity(proof_a.len() + proof_b.len() + free_inputs_a.len() + free_inputs_b.len() + 4);
    zkin.extend_from_slice(proof_a);
    zkin.extend_from_slice(proof_b);
    zkin.extend_from_slice(free_inputs_a);
    zkin.extend_from_slice(free_inputs_b);
    zkin.extend_from_slice(root_c_recurser_agg);

    timer_start_debug!(GENERATE_RECURSER_AGGREGATOR_WITNESS);
    let circom_witness = match generate_witness::<F>(setup, memory_handler_recursive_witness, 0, &zkin) {
        Ok(witness) => witness,
        Err(e) => {
            timer_stop_and_log_info!(GENERATE_RECURSER_AGGREGATOR);
            return Err(e);
        }
    };
    timer_stop_and_log_debug!(GENERATE_RECURSER_AGGREGATOR_WITNESS);

    let n_bits = setup.stark_info.stark_struct.n_bits;
    let n_publics = setup.stark_info.n_publics;
    let n_cols = setup.n_cols;
    let mut publics = vec![F::ZERO; n_publics as usize];
    let mut trace: Vec<F> = vec![F::ZERO; (1usize << n_bits) * n_cols as usize];
    let exec = setup
        .exec_data
        .as_ref()
        .ok_or_else(|| ProofmanError::InvalidSetup("recurser setup has no exec_data".into()))?;
    let exec_data_ptr = exec.as_ptr() as *mut u64;
    let exec_words = exec.len() as u64;

    get_committed_pols_c(
        circom_witness.as_ptr() as *mut u8,
        exec_data_ptr,
        trace.as_mut_ptr() as *mut u8,
        publics.as_mut_ptr() as *mut u8,
        setup.size_witness.ok_or_else(|| ProofmanError::InvalidSetup("recurser setup has no size_witness".into()))?,
        1 << n_bits,
        n_publics,
        // Compact on GPU, full width otherwise -- the same rule the device side applies. This path
        // reaches gen_recursive_proof_c, so filling wide while that reads compact hands it mapCols
        // columns' worth of an n_cols-wide row.
        recursion_trace_stride(exec, n_cols, setup.gpu),
    );
    // See generate_recursive_proof: device-side on GPU, host-side otherwise.
    if !setup.gpu {
        expand_gate_bands_c(trace.as_mut_ptr() as *mut u8, exec_data_ptr, n_cols, exec_words, 1 << n_bits);
    }
    memory_handler_recursive_witness.release_buffer_witness(circom_witness)?;

    let mut final_proof: Vec<u64> = vec![0; (1 + n_publics + setup.proof_size) as usize];
    final_proof[0] = n_publics;
    for (i, p) in publics.iter().enumerate() {
        final_proof[1 + i] = p.as_canonical_u64();
    }
    let stark_offset = (1 + n_publics) as usize;

    let (const_pols_ptr, const_tree_ptr) = if setup.gpu {
        (std::ptr::null_mut::<u8>(), std::ptr::null_mut::<u8>())
    } else {
        (const_pols.as_ptr() as *mut u8, const_tree.as_ptr() as *mut u8)
    };

    if let Err(e) = calc_handle.join_now() {
        tracing::warn!("Recurser const tree calculation thread panicked: {:?}", e);
    }

    timer_start_debug!(GENERATE_RECURSER_AGGREGATOR_PROOF);
    let stream_id = gen_recursive_proof_c(
        p_setup,
        trace.as_ptr() as *mut u8,
        prover_buffer.as_ptr() as *mut u8,
        const_pols_ptr,
        const_tree_ptr,
        publics.as_ptr() as *mut u8,
        final_proof[stark_offset..].as_mut_ptr(),
        "",
        0,
        0,
        0,
        true,
        d_buffers,
        &setup.const_pols_path,
        &setup.const_pols_tree_path,
        setup.setup_type.into(),
        false,
        recurser_id, // disambiguates recurser setups sharing (0,0,"recursive2")
        u64::MAX,    // one-off launch: reserve stream internally
    );
    get_stream_id_proof_c(d_buffers, stream_id);
    timer_stop_and_log_debug!(GENERATE_RECURSER_AGGREGATOR_PROOF);

    timer_stop_and_log_info!(GENERATE_RECURSER_AGGREGATOR);
    Ok(final_proof)
}

pub fn generate_snark_proof(
    snark_prover: *mut c_void,
    setup_path: &Path,
    proof: *mut c_void,
    prealloc_handle: std::thread::JoinHandle<()>,
    d_buffers_recursivef: *mut c_void,
) -> ProofmanResult<(Vec<u8>, Vec<u8>)> {
    let witness = generate_witness_final_snark(proof, setup_path)?;

    // Wait for GPU pre-allocation
    prealloc_handle.join().unwrap();

    timer_start_info!(CALCULATE_FINAL_PROOF);

    let mut snark_publics: Vec<u8> = vec![0; 32];
    let snark_publics_ptr = snark_publics.as_mut_ptr();

    let mut snark_proof: Vec<u8> = vec![0; 24 * 32];
    let snark_proof_ptr = snark_proof.as_mut_ptr();

    tracing::trace!("··· Generating final snark proof");
    gen_final_snark_proof_c(
        snark_prover,
        witness.as_ptr() as *mut u8,
        snark_proof_ptr,
        snark_publics_ptr,
        d_buffers_recursivef,
    );
    timer_stop_and_log_info!(CALCULATE_FINAL_PROOF);
    tracing::trace!("··· Final Snark Proof generated.");

    Ok((snark_proof, snark_publics))
}

pub fn generate_witness_final_snark(proof: *mut c_void, setup_path: &Path) -> ProofmanResult<Vec<u8>> {
    let lib_extension = if cfg!(target_os = "macos") { ".dylib" } else { ".so" };
    let rust_lib_filename = setup_path.display().to_string() + lib_extension;
    let rust_lib_path = Path::new(rust_lib_filename.as_str());

    if !rust_lib_path.exists() {
        return Err(ProofmanError::InvalidSetup(format!(
            "Rust lib dynamic library not found at path: {rust_lib_path:?}"
        )));
    }
    let library: Library = unsafe { Library::new(rust_lib_path)? };

    let dat_filename = setup_path.display().to_string() + ".dat";
    let dat_filename_str = CString::new(dat_filename.as_str()).unwrap();
    let dat_filename_ptr = dat_filename_str.as_ptr() as *mut std::os::raw::c_char;

    unsafe {
        timer_start_info!(CALCULATE_FINAL_WITNESS);

        let get_size_witness: Symbol<GetSizeWitnessFunc> = library.get(b"getSizeWitness\0")?;
        let size_witness = get_size_witness();

        let mut witness: Vec<u8> = vec![0; (size_witness * 32) as usize];
        let witness_ptr = witness.as_mut_ptr();

        let get_witness_final: Symbol<GetWitnessFinalFunc> = library.get(b"getWitness\0")?;
        let nmutex = std::cmp::min(8, rayon::current_num_threads());
        let res = get_witness_final(proof, dat_filename_ptr, witness_ptr as *mut c_void, nmutex as u64);
        if res != 0 {
            return Err(ProofmanError::InvalidProof("Error generating final witness from rust".into()));
        }
        timer_stop_and_log_info!(CALCULATE_FINAL_WITNESS);

        Ok(witness)
    }
}

/// Writes the zkin under the name `prove-air --proof` parses, when `PIL2_DUMP_ZKIN` names this
/// proof type (or `all`). Errors are logged, never returned: this is a diagnostic.
/// Row stride to fill the recursion trace with: the exec map's width when the device widens it, the
/// air's own width otherwise.
///
/// `getCommitedPols` takes this as the row stride, so handing it the map's width makes it write a
/// compact `N x map_cols` buffer with no gaps. The columns it then skips belong to a gate-band
/// expander, which rebuilds them on device -- so on the GPU path they need never cross PCIe.
///
/// The CPU path keeps the air's width: `expand_gate_bands_c` rebuilds the interiors in place and
/// needs the real layout to do it.
///
/// Every caller that can reach `gen_recursive_proof_c`'s GPU implementation must go through this,
/// because that side decides the same question from the same exec header. Filling wide while it reads
/// compact hands it `map_cols` columns' worth of an `n_cols`-wide row, and the proof then fails its
/// evaluations check with nothing pointing at the cause.
pub fn recursion_trace_stride(exec: &[u64], n_cols: u64, gpu: bool) -> u64 {
    // Mirrors `plonk2pil::{EXEC_MAGIC, EXEC_FORMAT_VERSION, EXEC_HEADER_WORDS}`, which write the
    // header, and `exec_layout` in pil2-stark/src/starkpil/exec_layout.hpp, which reads it. Spelled
    // out rather than imported: the prover does not depend on the setup crate.
    // Header: [magic|version, n_adds, map_rows, map_cols].
    const EXEC_MAGIC: u64 = 0x5058_4543_0000_0000; // "PXEC" in the high half
    const EXEC_FORMAT_VERSION: u64 = 2;
    const EXEC_HEADER_WORDS: usize = 4;
    if !gpu {
        return n_cols;
    }
    let Some(header) = exec.get(..EXEC_HEADER_WORDS) else { return n_cols };
    if header[0] != EXEC_MAGIC | EXEC_FORMAT_VERSION {
        return n_cols;
    }
    let map_cols = header[3];
    if map_cols > 0 && map_cols < n_cols {
        map_cols
    } else {
        n_cols
    }
}

fn dump_zkin_if_requested<F: PrimeField64>(setup: &Setup<F>, instance_id: usize, zkin: &[u64]) {
    let Ok(want) = std::env::var("PIL2_DUMP_ZKIN") else {
        return;
    };
    // The canonical spelling, so the dump round-trips through `ProofType::from_str`.
    let proof_type: &str = setup.setup_type.into();
    if !want.eq_ignore_ascii_case("all") && !want.eq_ignore_ascii_case(proof_type) {
        return;
    }

    let dir = std::env::var_os("TMPDIR").map(std::path::PathBuf::from).unwrap_or_else(std::env::temp_dir);
    // The instance is in the name because an air with several instances would otherwise have all
    // but the first capture dropped, and which one won would depend on scheduling.
    let path = dir.join(format!("zkin_ag{}_air{}_i{instance_id}_t{proof_type}.bin", setup.airgroup_id, setup.air_id));
    // create_new: the parallel prover would otherwise race on the same name. A leftover file from an
    // earlier run wins, so say so rather than looking like the capture happened.
    match std::fs::OpenOptions::new().write(true).create_new(true).open(&path) {
        Ok(file) => {
            let bytes: Vec<u8> = zkin.iter().flat_map(|w| w.to_le_bytes()).collect();
            let mut file = std::io::BufWriter::new(file);
            if let Err(e) = file.write_all(&bytes) {
                tracing::warn!("zkin capture to {} failed: {e}", path.display());
            } else {
                tracing::info!("Captured zkin ({} words) to {}", zkin.len(), path.display());
            }
        }
        Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => {
            tracing::warn!("zkin capture skipped: {} already exists", path.display());
        }
        Err(e) => tracing::warn!("zkin capture to {} failed: {e}", path.display()),
    }
}

fn generate_witness<F: PrimeField64>(
    setup: &Setup<F>,
    memory_handler_recursive_witness: &MemoryHandlerRecursive<F>,
    instance_id: usize,
    zkin: &[u64],
) -> ProofmanResult<Vec<F>> {
    let state = setup.circom_state.read().unwrap();
    let circom_circuit_ptr = match state.circuit {
        Some(ptr) => ptr,
        None => return Err(ProofmanError::InvalidSetup("circom_circuit is not initialized".into())),
    };

    let get_witness_fn =
        state.get_witness_fn.ok_or(ProofmanError::InvalidSetup("GetWitness function not loaded".to_string()))?;

    let nmutex = std::cmp::min(8, rayon::current_num_threads());

    // Capture the zkin feeding this recursion witness, to build test-recursive fixtures from a real
    // run. `PIL2_DUMP_ZKIN=recursive2` (or `all`) writes the first proof of each kind. Diagnostic
    // only: a dump that cannot be written must never fail the proof.
    dump_zkin_if_requested(setup, instance_id, zkin);

    // `take` blocks until a proof finishes and hands a buffer back. That wait is real time in this
    // stage but it is not witness generation, and on CPU it dominates -- so the two are timed apart:
    // the enclosing GENERATING_*_WITNESS stays end-to-end, CIRCOM_WITNESS is the circuit's own cost.
    timer_start_debug!(POOL_WAIT_WITNESS, "POOL_WAIT_WITNESS_{:?}", setup.setup_type);
    let mut witness: Vec<F> = memory_handler_recursive_witness.take_buffer_witness();
    timer_stop_and_log_debug!(POOL_WAIT_WITNESS, "POOL_WAIT_WITNESS_{:?}", setup.setup_type);

    // `getWitness` writes `size_witness` elements and `getCommitedPols` the `n_adds` after them, with
    // no bound of their own: a buffer sized for a smaller circuit is overrun silently, and the damage
    // lands on whatever the allocator put next -- another proof's buffer, most often, which then fails
    // somewhere with nothing pointing back here. The pool is sized for the largest recursive circuit,
    // so this only ever fires if that sizing and this call site have drifted apart.
    let needed = setup.size_witness.unwrap_or(0) as usize + setup.n_adds.unwrap_or(0) as usize;
    if witness.len() < needed {
        let got = witness.len();
        if let Err(e) = memory_handler_recursive_witness.release_buffer_witness(witness) {
            tracing::warn!("Failed to return witness buffer to pool: {e}");
        }
        return Err(ProofmanError::ProofmanError(format!(
            "{:?} [{}:{}] needs a {needed}-element circom witness but the pool hands out {} \
             (see SetupsVadcop::max_witness_size)",
            setup.setup_type, setup.airgroup_id, setup.air_id, got,
        )));
    }

    timer_start_debug!(CIRCOM_WITNESS, "CIRCOM_WITNESS_{:?}", setup.setup_type);
    let res: i64 = unsafe {
        get_witness_fn(
            zkin.as_ptr() as *mut u64,
            circom_circuit_ptr,
            witness.as_mut_ptr() as *mut c_void,
            nmutex as u64,
        )
    };
    timer_stop_and_log_debug!(CIRCOM_WITNESS, "CIRCOM_WITNESS_{:?}", setup.setup_type);
    drop(state);

    if res != 0 {
        let released = memory_handler_recursive_witness.release_buffer_witness(witness);
        if let Err(e) = released {
            tracing::warn!("Failed to return witness buffer to pool: {e}");
        }

        let ts = chrono::Utc::now().format("%Y%m%d_%H%M%S");
        let debug_file_path = std::path::Path::new("/tmp").join(format!(
            "proof_{instance_id}_ag{}_air{}_t{:?}_{}.bin",
            setup.airgroup_id, setup.air_id, setup.setup_type, ts
        ));
        let mut file = File::create(&debug_file_path)?;
        for word in zkin {
            file.write_all(&word.to_le_bytes())?;
        }
        file.flush()?;
        tracing::warn!("Debug proof data written to: {}", debug_file_path.display());

        return Err(ProofmanError::InvalidProof(format!(
            "Error generating witness for instance id {} [{}:{}] of type {:?}",
            instance_id, setup.airgroup_id, setup.air_id, setup.setup_type
        )));
    }

    Ok(witness)
}

pub fn get_recursive_buffer_sizes<F: PrimeField64>(
    pctx: &ProofCtx<F>,
    setups: &SetupsVadcop<F>,
) -> ProofmanResult<usize> {
    let mut max_prover_size = 0;

    for (airgroup_id, air_group) in pctx.global_info.airs.iter().enumerate() {
        for (air_id, _) in air_group.iter().enumerate() {
            if pctx.global_info.get_air_has_compressor(airgroup_id, air_id) {
                let setup_compressor = setups.sctx_compressor.as_ref().unwrap().get_setup(airgroup_id, air_id)?;
                max_prover_size = max_prover_size.max(setup_compressor.prover_buffer_size);
            }

            let setup_recursive1 = setups.sctx_recursive1.as_ref().unwrap().get_setup(airgroup_id, air_id)?;
            max_prover_size = max_prover_size.max(setup_recursive1.prover_buffer_size);
        }
    }

    let n_airgroups = pctx.global_info.air_groups.len();
    for airgroup in 0..n_airgroups {
        let setup = setups.sctx_recursive2.as_ref().unwrap().get_setup(airgroup, 0)?;
        max_prover_size = max_prover_size.max(setup.prover_buffer_size);
    }

    max_prover_size = max_prover_size
        .max(setups.setup_vadcop_final.as_ref().unwrap().prover_buffer_size)
        // Absent when the key was built without the compressed final stage, which contributes
        // nothing to the buffer it never fills.
        .max(setups.setup_vadcop_final_compressed.as_ref().map_or(0, |s| s.prover_buffer_size));

    Ok(max_prover_size as usize)
}

/// Aggregation proofs needed to reduce `n` proofs to one, and whether the last chunk needs a
/// null proof to fill it.
///
/// A short chunk needs `arity - rem` nulls, which is 0 or 1 only while `arity <= 3` — hence the
/// bool. Raising `VALID_AGGREGATION_ARITIES` past 3 means making this a count and updating the
/// `push(null_proof)` sites in proofman.rs.
#[derive(Debug)]
pub struct Recursive2Proofs {
    pub n_proofs: usize,
    pub has_remaining: bool,
}

impl Recursive2Proofs {
    pub fn new(n_proofs: usize, has_remaining: bool) -> Self {
        Self { n_proofs, has_remaining }
    }
}

pub fn total_recursive_proofs(mut n: usize, arity: usize) -> Recursive2Proofs {
    let mut total = 0;
    let mut rem = n % arity;
    while n > 1 {
        let next = n / arity;
        rem = n % arity;
        total += next;
        if next != 0 {
            n = next + rem;
        } else if rem != 1 {
            n = next;
        }
    }

    // A remainder of 2 or more needs one more aggregation, null-padded up to the arity.
    // At arity 2 the remainder is never >= 2, so this never fires.
    if rem >= 2 {
        Recursive2Proofs::new(total + 1, true)
    } else {
        Recursive2Proofs::new(total, false)
    }
}

#[cfg(test)]
mod arity_tests {
    use super::*;

    /// Model of the aggregation loop in `aggregate_recursive2_proofs`: repeatedly
    /// chunk `alive` proofs by `arity`, aggregating full chunks and any final short
    /// chunk, until one proof remains.
    fn simulate(n: usize, arity: usize) -> usize {
        if n == 0 {
            return 0;
        }
        let (mut alive, mut done) = (n, 0);
        while alive > 1 {
            let full = alive / arity;
            let rem = alive % arity;
            for i in 0..alive.div_ceil(arity) {
                let j = i * arity;
                if (j + arity - 1 < alive) || alive <= arity {
                    assert!(j + 1 < alive, "a chunk must hold at least 2 real proofs");
                    done += 1;
                }
            }
            alive = if full > 0 { full + rem } else { 1 };
        }
        done
    }

    #[test]
    fn the_formula_matches_the_loop_for_every_supported_arity() {
        // 4 is included as a pure-arithmetic check that the formula is not
        // accidentally 3-shaped. It is not a usable setup.
        for arity in [2usize, 3, 4] {
            for n in 0..200 {
                assert_eq!(total_recursive_proofs(n, arity).n_proofs, simulate(n, arity), "arity={arity} n={n}");
            }
        }
    }

    #[test]
    fn arity_three_counts_are_unchanged() {
        // Values pinned from the pre-change implementation.
        for (n, expected) in [(0usize, 0usize), (1, 0), (2, 1), (3, 1), (4, 2), (5, 2), (10, 5), (100, 50)] {
            assert_eq!(total_recursive_proofs(n, 3).n_proofs, expected, "n={n}");
        }
    }

    #[test]
    fn arity_two_is_a_binary_tree() {
        for n in 1..100 {
            assert_eq!(total_recursive_proofs(n, 2).n_proofs, n - 1, "n={n}");
        }
    }
}
