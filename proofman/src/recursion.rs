use borsh::{BorshSerialize, BorshDeserialize};
use libloading::{Library, Symbol};
use fields::PrimeField64;
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
    timer_start_info, timer_stop_and_log_info, timer_start_debug, timer_stop_and_log_debug, create_buffer_fast,
};

use crate::{add_publics_circom, add_publics_aggregation};

pub type GetWitnessFunc =
    unsafe extern "C" fn(zkin: *mut u64, circom_circuit: *mut c_void, witness: *mut c_void, n_mutexes: u64) -> i64;

pub type GetWitnessFinalFunc =
    unsafe extern "C" fn(zkin: *mut c_void, dat_file: *const c_char, witness: *mut c_void, n_mutexes: u64) -> i64;

pub const N_RECURSIVE_PROOFS_PER_AGGREGATION: usize = 3;

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
        timer_stop_and_log_debug!(
            GENERATE_COMPRESSOR_WITNESS,
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
        timer_stop_and_log_debug!(
            GENERATE_RECURSIVE1_WITNESS,
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
    proof1: &Proof<F>,
    proof2: &Proof<F>,
    proof3: &Proof<F>,
) -> ProofmanResult<Proof<F>> {
    timer_start_debug!(GENERATE_WITNESS_AGGREGATION);
    let proof_len = proof1.proof.len();
    if proof_len != proof2.proof.len() || proof_len != proof3.proof.len() {
        return Err(ProofmanError::ProofmanError(format!(
            "Inconsistent proof sizes: proof1 size {}, proof2 size {}, proof3 size {}",
            proof1.proof.len(),
            proof2.proof.len(),
            proof3.proof.len()
        )));
    }

    let airgroup_id = proof1.airgroup_id;
    if airgroup_id != proof2.airgroup_id || airgroup_id != proof3.airgroup_id {
        return Err(ProofmanError::ProofmanError(format!(
            "Inconsistent airgroup_ids: proof1 airgroup_id {}, proof2 airgroup_id {}, proof3 airgroup_id {}",
            proof1.airgroup_id, proof2.airgroup_id, proof3.airgroup_id
        )));
    }

    let publics_circom_size: usize =
        pctx.global_info.n_publics + pctx.global_info.n_proof_values.iter().sum::<usize>() * 3 + 3 + 4;

    let setup_recursive2 = setups.sctx_recursive2.as_ref().unwrap().get_setup(airgroup_id, 0)?;

    let updated_proof_size = N_RECURSIVE_PROOFS_PER_AGGREGATION * proof_len + publics_circom_size;

    let mut updated_proof_recursive2: Vec<u64> = vec![0; updated_proof_size];

    updated_proof_recursive2[publics_circom_size..(publics_circom_size + proof_len)].copy_from_slice(&proof1.proof);
    updated_proof_recursive2[publics_circom_size + proof_len..publics_circom_size + 2 * proof_len]
        .copy_from_slice(&proof2.proof);
    updated_proof_recursive2[publics_circom_size + 2 * proof_len..].copy_from_slice(&proof3.proof);

    add_publics_circom(&mut updated_proof_recursive2, 0, pctx, Some(&setup_recursive2.verkey));
    let circom_witness =
        generate_witness::<F>(setup_recursive2, memory_handler_recursive_witness, 0, &updated_proof_recursive2)?;

    timer_stop_and_log_debug!(GENERATE_WITNESS_AGGREGATION);
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
    Ok(Proof::new(witness.proof_type.clone(), witness.airgroup_id, witness.air_id, witness.global_idx, new_proof))
}

/// Writes a vadcop-final proof's public section `[n_publics | publics(n_publics)]`.
///
/// `publics` are the circuit's OUTPUT publics produced by `generate_recursive_proof`
/// (from the witness), which for vadcop_final include the `is_vadcop_final_proof` flag
/// at index 0. `gen_recursive_proof_size` sized `proof.proof` to hold `1 + n_publics`.
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
    calculate_fixed_tree_handle: Option<std::thread::JoinHandle<()>>,
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

    let setup = setups.get_setup(airgroup_id, air_id, &witness.proof_type)?;

    let mut trace = match setup.setup_type {
        ProofType::Compressor => memory_handler_recursive_witness.take_buffer_trace_compressor(),
        _ => memory_handler_recursive_witness.take_buffer_trace(),
    };

    let p_setup: *mut c_void = (&setup.p_setup).into();

    let mut publics = vec![F::ZERO; setup.stark_info.n_publics as usize];

    let exec_data_ptr = setup.exec_data.as_ref().expect("exec_data missing on setup").as_ptr() as *mut u64;

    get_committed_pols_c(
        witness.circom_witness.as_ptr() as *mut u8,
        exec_data_ptr,
        trace.as_mut_ptr() as *mut u8,
        publics.as_mut_ptr() as *mut u8,
        setup.size_witness.unwrap(),
        1 << (setup.stark_info.stark_struct.n_bits),
        setup.stark_info.n_publics,
        witness.n_cols as u64,
    );
    let circom_witness = std::mem::take(&mut witness.circom_witness);
    match setup.setup_type {
        ProofType::Compressor => memory_handler_recursive_witness.release_buffer_witness_compressor(circom_witness),
        _ => memory_handler_recursive_witness.release_buffer_witness(circom_witness),
    }?;

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
    // For VadcopFinal / VadcopFinalCompressed the proof's public section
    // (`[n_publics | publics(n_publics)]`, with the `is_vadcop_final_proof` flag at
    // index 0) is written by the caller from the `publics` returned below. These are
    // the circuit's OUTPUT publics read from the witness above (`get_committed_pols_c`),
    // NOT `pctx.get_publics()` — that buffer holds only the flag-free INPUT publics
    // (`global_info.n_publics`), so it is missing the flag and one element too short.

    let (const_pols_ptr, const_tree_ptr) = if pctx.gpu {
        (std::ptr::null_mut(), std::ptr::null_mut())
    } else {
        (const_pols.as_ptr() as *mut u8, const_tree.as_ptr() as *mut u8)
    };

    if let Some(handle) = calculate_fixed_tree_handle {
        handle.join().map_err(|_| ProofmanError::ProofmanError("Failed to calculate fixed tree".into()))?;
    }

    let stream_id = gen_recursive_proof_c(
        p_setup,
        trace.as_ptr() as *mut u8,
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
        witness.proof_type.clone().into(),
        force_recursive_stream,
        "",
    );

    // Trace H2D is async: gate buffer reuse on the stream's commit event so a
    // concurrent take() can't overwrite `trace` mid-copy (as on the main path).
    if pctx.gpu {
        wait_stream_commit_done_c(pctx.get_device_buffers_ptr(), stream_id);
    }
    match setup.setup_type {
        ProofType::Compressor => memory_handler_recursive_witness.release_buffer_trace_compressor(trace),
        _ => memory_handler_recursive_witness.release_buffer_trace(trace),
    }?;

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

    // agregation loop
    loop {
        mpi_ctx.barrier();
        mpi_ctx.distribute_recursive2_proofs(&alives, &mut airgroup_proofs);
        let mut pending_agregations = false;
        for airgroup in 0..n_airgroups {
            //create a vector of sice indices length
            let mut alive = alives[airgroup];
            if alive > 1 {
                let n_agg_proofs = alive / N_RECURSIVE_PROOFS_PER_AGGREGATION;
                let n_remaining_proofs = alive % N_RECURSIVE_PROOFS_PER_AGGREGATION;
                for i in 0..alive.div_ceil(N_RECURSIVE_PROOFS_PER_AGGREGATION) {
                    let j = i * N_RECURSIVE_PROOFS_PER_AGGREGATION;
                    if airgroup_proofs[airgroup][j].is_none() {
                        continue;
                    }
                    if (j + N_RECURSIVE_PROOFS_PER_AGGREGATION - 1 < alive)
                        || alive <= N_RECURSIVE_PROOFS_PER_AGGREGATION
                    {
                        if airgroup_proofs[airgroup][j + 1].is_none() {
                            return Err(ProofmanError::ProofmanError("Recursive2 proof is missing".into()));
                        }

                        let proof1 = Proof::new(
                            ProofType::Recursive2,
                            airgroup,
                            0,
                            None,
                            airgroup_proofs[airgroup][j].take().unwrap(),
                        );

                        let proof2 = Proof::new(
                            ProofType::Recursive2,
                            airgroup,
                            0,
                            None,
                            airgroup_proofs[airgroup][j + 1].take().unwrap(),
                        );

                        let proof_3 = if j + N_RECURSIVE_PROOFS_PER_AGGREGATION - 1 < alive {
                            airgroup_proofs[airgroup][j + N_RECURSIVE_PROOFS_PER_AGGREGATION - 1].take().unwrap()
                        } else {
                            null_proofs[airgroup].clone()
                        };

                        let proof3 = Proof::new(ProofType::Recursive2, airgroup, 0, None, proof_3);

                        let mut circom_witness = gen_witness_aggregation::<F>(
                            pctx,
                            memory_handler_recursive_witness,
                            setups,
                            &proof1,
                            &proof2,
                            &proof3,
                        )?;
                        circom_witness.global_idx = Some(rank);

                        let recursive2_proof = gen_recursive_proof_size::<F>(pctx, setups, &circom_witness)?;

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
                    airgroup_proofs[airgroup][i] =
                        airgroup_proofs[airgroup][i * N_RECURSIVE_PROOFS_PER_AGGREGATION].take();
                }

                for i in 0..n_remaining_proofs {
                    airgroup_proofs[airgroup][n_agg_proofs + i] =
                        airgroup_proofs[airgroup][N_RECURSIVE_PROOFS_PER_AGGREGATION * n_agg_proofs + i].take();
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
    timer_start_info!(GENERATE_VADCOP_FINAL_PROOF);
    let publics_circom_size =
        pctx.global_info.n_publics + pctx.global_info.n_proof_values.iter().sum::<usize>() * 3 + 3;

    let n_airgroups = pctx.global_info.air_groups.len();

    let mut updated_proof_size = publics_circom_size;

    let setup = setups.setup_vadcop_final.as_ref().unwrap();
    let p_setup: *mut c_void = (&setup.p_setup).into();

    let p_setup_addr = p_setup as usize;
    let device_buffers_addr = pctx.get_device_buffers_ptr() as usize;
    let setup_type = setup.setup_type.clone();

    let calculate_fixed_tree_handle = std::thread::spawn(move || {
        calculate_const_tree_fixed_c(
            p_setup_addr as *mut c_void,
            0,
            0,
            setup_type.into(),
            device_buffers_addr as *mut c_void,
        );
    });

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

    let mut final_proof = gen_recursive_proof_size::<F>(pctx, setups, &witness_final_proof)?;
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
        Some(calculate_fixed_tree_handle),
    )?;
    get_stream_id_proof_c(pctx.get_device_buffers_ptr(), stream_id);

    // Write the vadcop_final proof's public section (`[n_publics | publics]`, including
    // the `is_vadcop_final_proof` flag @0) from the circuit's OUTPUT publics returned by
    // `generate_recursive_proof` — NOT `pctx.get_publics()`, which holds only the
    // flag-free input publics.
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
    timer_start_info!(GENERATE_VADCOP_FINAL_COMPRESSED_PROOF);
    let setup = setups.setup_vadcop_final_compressed.as_ref().unwrap();

    let p_setup: *mut c_void = (&setup.p_setup).into();

    let p_setup_addr = p_setup as usize;
    let device_buffers_addr = pctx.get_device_buffers_ptr() as usize;
    let setup_type = setup.setup_type.clone();

    let calculate_fixed_tree_handle = std::thread::spawn(move || {
        calculate_const_tree_fixed_c(
            p_setup_addr as *mut c_void,
            0,
            0,
            setup_type.into(),
            device_buffers_addr as *mut c_void,
        );
    });

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

    let mut final_proof = gen_recursive_proof_size::<F>(pctx, setups, &witness_final_proof)?;
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

    let load_fixed_pols_handle = std::thread::spawn(move || {
        timer_start_debug!(LOAD_FIXED_POLS_RECURSIVEF);
        load_fixed_pols_recursivef_c(
            p_setup_addr as *mut c_void,
            const_tree_ptr_addr as *mut c_void,
            d_buffers_addr as *mut c_void,
        );
        timer_stop_and_log_debug!(LOAD_FIXED_POLS_RECURSIVEF);
    });

    let mut trace: Vec<F> = memory_handler_recursive_witness.take_buffer_trace();

    let proof = &vadcop_proof[1..];
    let mut updated_proof: Vec<u64> = vec![0; proof.len() + 4];

    updated_proof[..4].copy_from_slice(&vadcop_final_verkey[..4]);

    updated_proof[4..].copy_from_slice(proof);

    timer_start_debug!(GENERATE_RECURSIVEF_WITNESS);
    let circom_witness = generate_witness::<F>(setup, memory_handler_recursive_witness, 0, &updated_proof)?;
    timer_stop_and_log_debug!(GENERATE_RECURSIVEF_WITNESS);

    let mut publics = vec![F::ZERO; setup.stark_info.n_publics as usize];

    let exec_data_ptr = setup.exec_data.as_ref().expect("exec_data missing on RecursiveF setup").as_ptr() as *mut u64;

    get_committed_pols_c(
        circom_witness.as_ptr() as *mut u8,
        exec_data_ptr,
        trace.as_mut_ptr() as *mut u8,
        publics.as_mut_ptr() as *mut u8,
        setup.size_witness.unwrap(),
        1 << (setup.stark_info.stark_struct.n_bits),
        setup.stark_info.n_publics,
        setup.stark_info.map_sections_n["cm1"],
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
    memory_handler_recursive_witness.release_buffer_trace(trace)?;
    timer_stop_and_log_debug!(GENERATE_RECURSIVEF_PROOF);

    // Join the background thread (should be done by now since proof waited for copy event)
    if let Err(e) = load_fixed_pols_handle.join() {
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
    let setup_type = setup.setup_type.clone();
    let calc_handle = std::thread::spawn(move || {
        calculate_const_tree_fixed_c(
            p_setup_addr as *mut c_void,
            0,
            0,
            setup_type.into(),
            device_buffers_addr as *mut c_void,
        );
    });

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
            if let Err(p) = calc_handle.join() {
                tracing::warn!("Recurser const tree calculation thread panicked: {:?}", p);
            }
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
    let exec_data_ptr = setup
        .exec_data
        .as_ref()
        .map(|v| v.as_ptr() as *mut u64)
        .ok_or_else(|| ProofmanError::InvalidSetup("recurser setup has no exec_data".into()))?;

    get_committed_pols_c(
        circom_witness.as_ptr() as *mut u8,
        exec_data_ptr,
        trace.as_mut_ptr() as *mut u8,
        publics.as_mut_ptr() as *mut u8,
        setup.size_witness.ok_or_else(|| ProofmanError::InvalidSetup("recurser setup has no size_witness".into()))?,
        1 << n_bits,
        n_publics,
        n_cols,
    );
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

    if let Err(e) = calc_handle.join() {
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
        setup.setup_type.clone().into(),
        false,
        recurser_id, // disambiguates recurser setups sharing (0,0,"recursive2")
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

    let mut witness: Vec<F> = match setup.setup_type {
        ProofType::Compressor => memory_handler_recursive_witness.take_buffer_witness_compressor(),
        _ => memory_handler_recursive_witness.take_buffer_witness(),
    };

    let res: i64 = unsafe {
        get_witness_fn(
            zkin.as_ptr() as *mut u64,
            circom_circuit_ptr,
            witness.as_mut_ptr() as *mut c_void,
            nmutex as u64,
        )
    };
    drop(state);

    if res != 0 {
        let released = match setup.setup_type {
            ProofType::Compressor => memory_handler_recursive_witness.release_buffer_witness_compressor(witness),
            _ => memory_handler_recursive_witness.release_buffer_witness(witness),
        };
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
        .max(setups.setup_vadcop_final_compressed.as_ref().unwrap().prover_buffer_size);

    Ok(max_prover_size as usize)
}

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

pub fn total_recursive_proofs(mut n: usize) -> Recursive2Proofs {
    let mut total = 0;
    let mut rem = n % N_RECURSIVE_PROOFS_PER_AGGREGATION;
    while n > 1 {
        let next = n / N_RECURSIVE_PROOFS_PER_AGGREGATION;
        rem = n % N_RECURSIVE_PROOFS_PER_AGGREGATION;
        total += next;
        if next != 0 {
            n = next + rem;
        } else if rem != 1 {
            n = next;
        }
    }

    if rem == 2 {
        Recursive2Proofs::new(total + 1, true)
    } else {
        Recursive2Proofs::new(total, false)
    }
}
