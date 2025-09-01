use libloading::{Library, Symbol};
use fields::PrimeField64;
use std::ffi::CString;
use proofman_starks_lib_c::*;
use std::path::Path;
use num_traits::ToPrimitive;

use proofman_common::{load_const_pols, load_const_pols_tree, Proof, ProofCtx, ProofType, Setup, SetupsVadcop};

use std::os::raw::{c_void, c_char};

use proofman_util::{timer_start_info, timer_stop_and_log_info, timer_stop_and_log_trace, timer_start_trace};

use crate::{add_publics_circom, add_publics_aggregation};

type GetWitnessFunc =
    unsafe extern "C" fn(zkin: *mut u64, circom_circuit: *mut c_void, witness: *mut c_void, n_mutexes: u64);

type GetWitnessFinalFunc =
    unsafe extern "C" fn(zkin: *mut c_void, dat_file: *const c_char, witness: *mut c_void, n_mutexes: u64);

type GetSizeWitnessFunc = unsafe extern "C" fn() -> u64;

#[derive(Debug)]
pub struct MaxSizes {
    pub total_const_area: u64,
    pub max_aux_trace_area: u64,
    pub total_const_area_aggregation: u64,
}

pub fn gen_witness_recursive<F: PrimeField64>(
    pctx: &ProofCtx<F>,
    setups: &SetupsVadcop<F>,
    proof: &Proof<F>,
) -> Result<Proof<F>, Box<dyn std::error::Error>> {
    let (airgroup_id, air_id) = (proof.airgroup_id, proof.air_id);

    assert!(proof.proof_type == ProofType::Basic || proof.proof_type == ProofType::Compressor);

    let has_compressor = pctx.global_info.get_air_has_compressor(airgroup_id, air_id);
    if proof.proof_type == ProofType::Basic && has_compressor {
        timer_start_info!(
            GENERATE_COMPRESSOR_WITNESS,
            "GENERATING_COMPRESSOR_WITNESS_{} [{}:{}]",
            proof.global_idx.unwrap(),
            proof.airgroup_id,
            proof.air_id
        );
        let setup = setups.sctx_compressor.as_ref().unwrap().get_setup(airgroup_id, air_id);

        let publics_circom_size =
            pctx.global_info.n_publics + pctx.global_info.n_proof_values.iter().sum::<usize>() * 3 + 3;

        let mut updated_proof: Vec<u64> = vec![0; proof.proof.len() + publics_circom_size];
        updated_proof[publics_circom_size..].copy_from_slice(&proof.proof);
        add_publics_circom(&mut updated_proof, 0, pctx, "", false);
        let circom_witness = generate_witness::<F>(setup, &updated_proof)?;
        timer_stop_and_log_info!(
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
        timer_start_info!(
            GENERATE_RECURSIVE1_WITNESS,
            "GENERATING_RECURSIVE1_WITNESS_{} [{}:{}]",
            proof.global_idx.unwrap(),
            proof.airgroup_id,
            proof.air_id
        );
        let setup = setups.sctx_recursive1.as_ref().unwrap().get_setup(airgroup_id, air_id);

        let recursive2_verkey =
            pctx.global_info.get_air_setup_path(airgroup_id, air_id, &ProofType::Recursive2).display().to_string()
                + ".verkey.json";

        let publics_circom_size =
            pctx.global_info.n_publics + pctx.global_info.n_proof_values.iter().sum::<usize>() * 3 + 3 + 4;

        let mut updated_proof: Vec<u64> = vec![0; proof.proof.len() + publics_circom_size];

        if proof.proof_type == ProofType::Compressor {
            let n_publics_aggregation = n_publics_aggregation(pctx, airgroup_id);
            let publics_aggregation: Vec<F> =
                proof.proof.iter().take(n_publics_aggregation).map(|&x| F::from_u64(x)).collect();
            add_publics_aggregation(&mut updated_proof, 0, &publics_aggregation, n_publics_aggregation);
            add_publics_circom(&mut updated_proof, n_publics_aggregation, pctx, &recursive2_verkey, true);
            updated_proof[(publics_circom_size + n_publics_aggregation)..]
                .copy_from_slice(&proof.proof[n_publics_aggregation..]);
        } else {
            updated_proof[publics_circom_size..].copy_from_slice(&proof.proof);
            add_publics_circom(&mut updated_proof, 0, pctx, &recursive2_verkey, true);
        }

        let circom_witness = generate_witness::<F>(setup, &updated_proof)?;
        timer_stop_and_log_info!(
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
    setups: &SetupsVadcop<F>,
    proof1: &Proof<F>,
    proof2: &Proof<F>,
    proof3: &Proof<F>,
) -> Result<Proof<F>, Box<dyn std::error::Error>> {
    timer_start_info!(GENERATE_WITNESS_AGGREGATION);
    let proof_len = proof1.proof.len();
    assert!(proof_len == proof2.proof.len() && proof_len == proof3.proof.len());

    let airgroup_id = proof1.airgroup_id;
    assert!(airgroup_id == proof2.airgroup_id && airgroup_id == proof3.airgroup_id);

    let publics_circom_size: usize =
        pctx.global_info.n_publics + pctx.global_info.n_proof_values.iter().sum::<usize>() * 3 + 3 + 4;

    let setup_recursive2 = setups.sctx_recursive2.as_ref().unwrap().get_setup(airgroup_id, 0);

    let updated_proof_size = 3 * proof_len + publics_circom_size;

    let mut updated_proof_recursive2: Vec<u64> = vec![0; updated_proof_size];

    updated_proof_recursive2[publics_circom_size..(publics_circom_size + proof_len)].copy_from_slice(&proof1.proof);
    updated_proof_recursive2[publics_circom_size + proof_len..publics_circom_size + 2 * proof_len]
        .copy_from_slice(&proof2.proof);
    updated_proof_recursive2[publics_circom_size + 2 * proof_len..].copy_from_slice(&proof3.proof);

    let recursive2_verkey =
        pctx.global_info.get_air_setup_path(airgroup_id, 0, &ProofType::Recursive2).display().to_string()
            + ".verkey.json";

    add_publics_circom(&mut updated_proof_recursive2, 0, pctx, &recursive2_verkey, true);
    let circom_witness = generate_witness::<F>(setup_recursive2, &updated_proof_recursive2)?;

    timer_stop_and_log_info!(GENERATE_WITNESS_AGGREGATION);
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
    publics_aggregation += 4 * pctx.global_info.agg_types[airgroup_id].len(); // agg types
    publics_aggregation += 10; // elliptic curve hash
    publics_aggregation += 1; // n proofs aggregated
    publics_aggregation
}

pub fn gen_recursive_proof_size<F: PrimeField64>(
    pctx: &ProofCtx<F>,
    setups: &SetupsVadcop<F>,
    witness: &Proof<F>,
) -> Proof<F> {
    let (airgroup_id, air_id) = (witness.airgroup_id, witness.air_id);

    let setup = setups.get_setup(airgroup_id, air_id, &witness.proof_type);

    let mut new_proof_size = setup.proof_size;

    let publics_aggregation = n_publics_aggregation(pctx, airgroup_id);

    if witness.proof_type != ProofType::VadcopFinal {
        new_proof_size += publics_aggregation as u64;
    } else {
        new_proof_size += 1 + setup.stark_info.n_publics;
    }

    let new_proof = vec![0; new_proof_size as usize];
    Proof::new(witness.proof_type.clone(), witness.airgroup_id, witness.air_id, witness.global_idx, new_proof)
}

#[allow(clippy::too_many_arguments)]
pub fn generate_recursive_proof<F: PrimeField64>(
    pctx: &ProofCtx<F>,
    setups: &SetupsVadcop<F>,
    witness: &Proof<F>,
    new_proof: &Proof<F>,
    trace: &[F],
    prover_buffer: &[F],
    output_dir_path: &Path,
    d_buffers: *mut c_void,
    const_tree: &[F],
    const_pols: &[F],
    save_proofs: bool,
) -> u64 {
    timer_start_info!(
        GEN_RECURSIVE_PROOF,
        "GEN_RECURSIVE_PROOF_{:?} [{}:{}]",
        witness.proof_type,
        witness.airgroup_id,
        witness.air_id
    );
    let global_info_path = pctx.global_info.get_proving_key_path().join("pilout.globalInfo.json");
    let global_info_file: &str = global_info_path.to_str().unwrap();

    let (airgroup_id, air_id, instance_id, output_file_path, vadcop) = if witness.proof_type == ProofType::VadcopFinal {
        let output_file_path_ = output_dir_path.join("proofs/vadcop_final_proof.json");
        (0, 0, 0, output_file_path_, false)
    } else {
        let (airgroup_id_, air_id_) = (witness.airgroup_id, witness.air_id);
        let air_instance_name = &pctx.global_info.airs[airgroup_id_][air_id_].name;
        let output_file_path_ = if witness.proof_type == ProofType::Recursive2 {
            output_dir_path.join(format!("proofs/{:?}_{}.json", witness.proof_type, air_instance_name))
        } else {
            output_dir_path.join(format!(
                "proofs/{:?}_{}_{}.json",
                witness.proof_type,
                air_instance_name,
                witness.global_idx.unwrap()
            ))
        };
        (airgroup_id_, air_id_, witness.global_idx.unwrap(), output_file_path_, true)
    };

    let proof_file = match save_proofs {
        true => output_file_path.to_string_lossy().into_owned(),
        false => String::from(""),
    };

    let setup = setups.get_setup(airgroup_id, air_id, &witness.proof_type);
    let p_setup: *mut c_void = (&setup.p_setup).into();

    let mut publics = vec![F::ZERO; setup.stark_info.n_publics as usize];

    let exec_data_ptr = setup.exec_data.read().unwrap().as_ref().map(|v| v.as_ptr() as *mut u64).unwrap();

    get_committed_pols_c(
        witness.circom_witness.as_ptr() as *mut u8,
        exec_data_ptr,
        trace.as_ptr() as *mut u8,
        publics.as_mut_ptr() as *mut u8,
        setup.size_witness.read().unwrap().unwrap(),
        1 << (setup.stark_info.stark_struct.n_bits),
        setup.stark_info.n_publics,
        witness.n_cols as u64,
    );

    let publics_aggregation = n_publics_aggregation(pctx, airgroup_id);

    let initial_idx = if witness.proof_type == ProofType::VadcopFinal {
        1 + setup.stark_info.n_publics as usize
    } else {
        publics_aggregation
    };

    let const_pols_path = setup.setup_path.to_string_lossy().to_string() + ".const";
    let const_pols_tree_path = setup.setup_path.display().to_string() + ".consttree";
    let proof_type: &str = setup.setup_type.clone().into();

    if witness.proof_type != ProofType::VadcopFinal {
        add_publics_aggregation_c(
            new_proof.proof.as_ptr() as *mut u8,
            0,
            publics.as_ptr() as *mut u8,
            publics_aggregation as u64,
        );
    }

    let (const_pols_ptr, const_tree_ptr) = if cfg!(feature = "gpu") {
        (std::ptr::null_mut(), std::ptr::null_mut())
    } else {
        (const_pols.as_ptr() as *mut u8, const_tree.as_ptr() as *mut u8)
    };

    let stream_id = gen_recursive_proof_c(
        p_setup,
        trace.as_ptr() as *mut u8,
        prover_buffer.as_ptr() as *mut u8,
        const_pols_ptr,
        const_tree_ptr,
        publics.as_ptr() as *mut u8,
        new_proof.proof[initial_idx..].as_ptr() as *mut u64,
        &proof_file,
        global_info_file,
        airgroup_id as u64,
        air_id as u64,
        instance_id as u64,
        vadcop,
        d_buffers,
        &const_pols_path,
        &const_pols_tree_path,
        proof_type,
    );

    timer_stop_and_log_info!(
        GEN_RECURSIVE_PROOF,
        "GEN_RECURSIVE_PROOF_{:?} [{}:{}]",
        witness.proof_type,
        witness.airgroup_id,
        witness.air_id
    );
    stream_id
}

#[allow(clippy::too_many_arguments)]
pub fn aggregate_recursive2_proofs<F: PrimeField64>(
    pctx: &ProofCtx<F>,
    setups: &SetupsVadcop<F>,
    proofs: Vec<Vec<Proof<F>>>,
    trace: &[F],
    prover_buffer: &[F],
    const_pols: &[F],
    const_tree: &[F],
    output_dir_path: &Path,
    d_buffers: *mut c_void,
    save_proofs: bool,
) -> Result<Proof<F>, Box<dyn std::error::Error>> {
    let mut dctx = pctx.dctx.write().unwrap();
    let n_processes = dctx.n_processes as usize;
    let rank = dctx.rank as usize;
    let airgroup_instances_alive = &dctx.airgroup_instances_alives;
    let n_airgroups = pctx.global_info.air_groups.len();
    let mut alives = vec![0; n_airgroups];
    let mut airgroup_proofs: Vec<Vec<Option<Vec<u64>>>> = Vec::with_capacity(n_airgroups);

    let mut null_proofs: Vec<Vec<u64>> = vec![Vec::new(); n_airgroups];

    // Pre-process data before starting recursion loop
    for airgroup in 0..n_airgroups {
        let mut current_pos = 0;
        for p in 0..n_processes {
            if p < rank {
                current_pos += airgroup_instances_alive[airgroup][p];
            }
            alives[airgroup] += airgroup_instances_alive[airgroup][p];
        }
        let setup = setups.get_setup(airgroup, 0, &ProofType::Recursive2);
        let publics_aggregation = n_publics_aggregation(pctx, airgroup);
        null_proofs[airgroup] = vec![0; setup.proof_size as usize + publics_aggregation];
        airgroup_proofs.push(vec![None; alives[airgroup]]);

        if !proofs[airgroup].is_empty() {
            for i in 0..proofs[airgroup].len() {
                airgroup_proofs[airgroup][current_pos + i] = Some(proofs[airgroup][i].proof.clone());
            }
        } else if rank == 0 {
            airgroup_proofs[airgroup][0] = Some(vec![0; setup.proof_size as usize + publics_aggregation]);
        }
    }

    // agregation loop
    loop {
        dctx.barrier();
        dctx.distribute_recursive2_proofs(&alives, &mut airgroup_proofs);
        let mut pending_agregations = false;
        for airgroup in 0..n_airgroups {
            //create a vector of sice indices length
            let mut alive = alives[airgroup];
            if alive > 1 {
                let n_agg_proofs = alive / 3;
                let n_remaining_proofs = alive % 3;
                for i in 0..alive.div_ceil(3) {
                    let j = i * 3;
                    if airgroup_proofs[airgroup][j].is_none() {
                        continue;
                    }
                    if (j + 2 < alive) || alive <= 3 {
                        if airgroup_proofs[airgroup][j + 1].is_none() {
                            panic!("Recursive2 proof is missing");
                        }

                        let proof1 = Proof::new(
                            ProofType::Recursive2,
                            airgroup,
                            0,
                            None,
                            airgroup_proofs[airgroup][j].clone().unwrap(),
                        );

                        let proof2 = Proof::new(
                            ProofType::Recursive2,
                            airgroup,
                            0,
                            None,
                            airgroup_proofs[airgroup][j + 1].clone().unwrap(),
                        );

                        let proof_3 = if j + 2 < alive {
                            airgroup_proofs[airgroup][j + 2].clone().unwrap()
                        } else {
                            null_proofs[airgroup].clone()
                        };

                        let proof3 = Proof::new(ProofType::Recursive2, airgroup, 0, None, proof_3);

                        let mut circom_witness = gen_witness_aggregation::<F>(pctx, setups, &proof1, &proof2, &proof3)?;
                        circom_witness.global_idx = Some(rank);

                        let recursive2_proof = gen_recursive_proof_size::<F>(pctx, setups, &circom_witness);

                        let stream_id = generate_recursive_proof::<F>(
                            pctx,
                            setups,
                            &circom_witness,
                            &recursive2_proof,
                            trace,
                            prover_buffer,
                            output_dir_path,
                            d_buffers,
                            const_tree,
                            const_pols,
                            save_proofs,
                        );

                        get_stream_id_proof_c(d_buffers, stream_id);

                        airgroup_proofs[airgroup][j] = Some(recursive2_proof.proof);

                        tracing::info!("··· Recursive 2 Proof generated.");
                    }
                }
                if n_agg_proofs > 0 {
                    alive = n_agg_proofs + n_remaining_proofs;
                } else {
                    alive = 1;
                }

                //compact elements
                for i in 0..n_agg_proofs {
                    airgroup_proofs[airgroup][i] = airgroup_proofs[airgroup][i * 3].clone();
                }

                for i in 0..n_remaining_proofs {
                    airgroup_proofs[airgroup][n_agg_proofs + i] =
                        airgroup_proofs[airgroup][3 * n_agg_proofs + i].clone();
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

    let mut updated_proof: Vec<u64> = Vec::new();
    if dctx.rank == 0 {
        let publics_circom_size =
            pctx.global_info.n_publics + pctx.global_info.n_proof_values.iter().sum::<usize>() * 3 + 3;

        let mut updated_proof_size = publics_circom_size;
        for proofs in &airgroup_proofs {
            updated_proof_size += proofs[0].as_ref().unwrap().len();
        }

        updated_proof = vec![0; updated_proof_size];
        add_publics_circom(&mut updated_proof, 0, pctx, "", false);

        for proofs in &airgroup_proofs {
            updated_proof[publics_circom_size..].copy_from_slice(&proofs[0].clone().unwrap());
        }
    }

    Ok(Proof::new(ProofType::Recursive2, 0, 0, None, updated_proof))
}

#[allow(clippy::too_many_arguments)]
pub fn generate_vadcop_final_proof<F: PrimeField64>(
    pctx: &ProofCtx<F>,
    setups: &SetupsVadcop<F>,
    proof: &Proof<F>,
    trace: &[F],
    prover_buffer: &[F],
    output_dir_path: &Path,
    const_pols: &[F],
    const_tree: &[F],
    d_buffers: *mut c_void,
    save_proof: bool,
) -> Result<Proof<F>, Box<dyn std::error::Error>> {
    let setup = setups.setup_vadcop_final.as_ref().unwrap();
    let circom_witness_vadcop_final = generate_witness::<F>(setup, &proof.proof)?;
    let witness_final_proof =
        Proof::new_witness(ProofType::VadcopFinal, 0, 0, None, circom_witness_vadcop_final, setup.n_cols as usize);
    tracing::info!("··· Generating vadcop final proof");
    timer_start_info!(GENERATE_VADCOP_FINAL_PROOF);
    let mut final_proof = gen_recursive_proof_size::<F>(pctx, setups, &witness_final_proof);
    let stream_id = generate_recursive_proof::<F>(
        pctx,
        setups,
        &witness_final_proof,
        &final_proof,
        trace,
        prover_buffer,
        output_dir_path,
        d_buffers,
        const_tree,
        const_pols,
        save_proof,
    );
    get_stream_id_proof_c(d_buffers, stream_id);

    // Set publics for vadcop final proof
    let publics = pctx.get_publics();
    final_proof.proof[0] = setup.stark_info.n_publics;
    for p in 0..setup.stark_info.n_publics as usize {
        final_proof.proof[1 + p] = publics[p].as_canonical_u64();
    }

    tracing::info!("··· Vadcop final Proof generated.");
    timer_stop_and_log_info!(GENERATE_VADCOP_FINAL_PROOF);

    Ok(final_proof)
}

#[allow(clippy::too_many_arguments)]
pub fn generate_recursivef_proof<F: PrimeField64>(
    pctx: &ProofCtx<F>,
    setups: &SetupsVadcop<F>,
    proof: &[u64],
    trace: &[F],
    prover_buffer: &[F],
    output_dir_path: &Path,
    save_proofs: bool,
) -> Result<*mut c_void, Box<dyn std::error::Error>> {
    let global_info_path = pctx.global_info.get_proving_key_path().join("pilout.globalInfo.json");
    let global_info_file: &str = global_info_path.to_str().unwrap();

    let setup = setups.setup_recursivef.as_ref().unwrap();
    let p_setup: *mut c_void = (&setup.p_setup).into();

    let setup_path = pctx.global_info.get_setup_path("recursivef");

    let const_tree_size = setup.const_tree_size;
    let const_tree = vec![F::ZERO; const_tree_size];
    let const_pols: Vec<F> = vec![F::ZERO; setup.const_pols_size];

    load_const_pols(&setup_path, setup.const_pols_size, &const_pols);
    load_const_pols_tree(setup, &const_tree);

    let mut vadcop_final_proof: Vec<u64> = vec![0; proof.len() + pctx.global_info.n_publics];
    vadcop_final_proof[pctx.global_info.n_publics..].copy_from_slice(proof);

    let public_inputs = pctx.get_publics();
    for p in 0..pctx.global_info.n_publics {
        vadcop_final_proof[p] = (public_inputs[p].as_canonical_biguint()).to_u64().unwrap();
    }

    let circom_witness = generate_witness::<F>(setup, &vadcop_final_proof)?;

    let publics = vec![F::ZERO; setup.stark_info.n_publics as usize];

    let exec_data_ptr = setup.exec_data.read().unwrap().as_ref().map(|v| v.as_ptr() as *mut u64).unwrap();

    get_committed_pols_c(
        circom_witness.as_ptr() as *mut u8,
        exec_data_ptr,
        trace.as_ptr() as *mut u8,
        publics.as_ptr() as *mut u8,
        setup.size_witness.read().unwrap().unwrap(),
        1 << (setup.stark_info.stark_struct.n_bits),
        setup.stark_info.n_publics,
        13,
    );

    let proof_file = match save_proofs {
        true => output_dir_path.join("proofs/recursivef.json").to_string_lossy().into_owned(),
        false => String::from(""),
    };

    tracing::info!("··· Generating recursiveF proof");
    timer_start_trace!(GENERATE_RECURSIVEF_PROOF);
    // prove
    let p_prove = gen_recursive_proof_final_c(
        p_setup,
        trace.as_ptr() as *mut u8,
        prover_buffer.as_ptr() as *mut u8,
        const_pols.as_ptr() as *mut u8,
        const_tree.as_ptr() as *mut u8,
        publics.as_ptr() as *mut u8,
        &proof_file,
        global_info_file,
        0,
        0,
        0,
    );
    tracing::info!("··· RecursiveF Proof generated.");
    timer_stop_and_log_trace!(GENERATE_RECURSIVEF_PROOF);

    Ok(p_prove)
}

pub fn generate_fflonk_snark_proof<F: PrimeField64>(
    pctx: &ProofCtx<F>,
    proof: *mut c_void,
    output_dir_path: &Path,
) -> Result<(), Box<dyn std::error::Error>> {
    let setup_path = pctx.global_info.get_setup_path("final");

    let lib_extension = if cfg!(target_os = "macos") { ".dylib" } else { ".so" };
    let rust_lib_filename = setup_path.display().to_string() + lib_extension;
    let rust_lib_path = Path::new(rust_lib_filename.as_str());

    if !rust_lib_path.exists() {
        return Err(format!("Rust lib dynamic library not found at path: {rust_lib_path:?}").into());
    }
    let library: Library = unsafe { Library::new(rust_lib_path)? };

    let dat_filename = setup_path.display().to_string() + ".dat";
    let dat_filename_str = CString::new(dat_filename.as_str()).unwrap();
    let dat_filename_ptr = dat_filename_str.as_ptr() as *mut std::os::raw::c_char;

    unsafe {
        timer_start_trace!(CALCULATE_FINAL_WITNESS);

        let get_size_witness: Symbol<GetSizeWitnessFunc> = library.get(b"getSizeWitness\0")?;
        let size_witness = get_size_witness();

        let witness: Vec<u8> = vec![0; (size_witness * 32) as usize];
        let witness_ptr = witness.as_ptr() as *mut u8;

        let get_witness_final: Symbol<GetWitnessFinalFunc> = library.get(b"getWitness\0")?;

        let nmutex = rayon::current_num_threads();
        get_witness_final(proof, dat_filename_ptr, witness_ptr as *mut c_void, nmutex as u64);

        timer_stop_and_log_trace!(CALCULATE_FINAL_WITNESS);

        timer_start_trace!(CALCULATE_FINAL_PROOF);
        let proof_file = output_dir_path.join("proofs").to_string_lossy().into_owned();

        let zkey_filename = setup_path.display().to_string() + ".zkey";
        tracing::info!("··· Generating final snark proof");
        gen_final_snark_proof_c(witness_ptr, zkey_filename.as_str(), &proof_file);
        timer_stop_and_log_trace!(CALCULATE_FINAL_PROOF);
        tracing::info!("··· Final Snark Proof generated.");
    }

    Ok(())
}

fn generate_witness<F: PrimeField64>(setup: &Setup<F>, zkin: &[u64]) -> Result<Vec<F>, Box<dyn std::error::Error>> {
    let mut witness_size = setup.size_witness.read().unwrap().unwrap();
    witness_size += *setup.exec_data.read().unwrap().as_ref().unwrap().first().unwrap();

    let witness: Vec<F> = vec![F::ZERO; witness_size as usize];

    let circom_circuit_guard = setup.circom_circuit.read().unwrap();
    let circom_circuit_ptr = match *circom_circuit_guard {
        Some(ptr) => ptr,
        None => panic!("circom_circuit is not initialized"),
    };

    unsafe {
        let library_guard = setup.circom_library.read().unwrap();
        let library = library_guard.as_ref().ok_or("Circom library not loaded")?;
        let get_witness: Symbol<GetWitnessFunc> = library.get(b"getWitness\0")?;
        get_witness(zkin.as_ptr() as *mut u64, circom_circuit_ptr, witness.as_ptr() as *mut c_void, 1);
    }

    Ok(witness)
}

pub fn get_recursive_buffer_sizes<F: PrimeField64>(
    pctx: &ProofCtx<F>,
    setups: &SetupsVadcop<F>,
) -> Result<(usize, usize), Box<dyn std::error::Error>> {
    let mut max_trace = 0;
    let mut max_prover_size = 0;

    for (airgroup_id, air_group) in pctx.global_info.airs.iter().enumerate() {
        for (air_id, _) in air_group.iter().enumerate() {
            if pctx.global_info.get_air_has_compressor(airgroup_id, air_id) {
                let setup_compressor = setups.sctx_compressor.as_ref().unwrap().get_setup(airgroup_id, air_id);
                max_trace = max_trace.max(
                    setup_compressor.n_cols as usize
                        * (1 << (setup_compressor.stark_info.stark_struct.n_bits)) as usize,
                );
                max_prover_size = max_prover_size.max(setup_compressor.prover_buffer_size);
            }

            let setup_recursive1 = setups.sctx_recursive1.as_ref().unwrap().get_setup(airgroup_id, air_id);
            max_trace = max_trace.max(
                setup_recursive1.n_cols as usize * (1 << (setup_recursive1.stark_info.stark_struct.n_bits)) as usize,
            );
            max_prover_size = max_prover_size.max(setup_recursive1.prover_buffer_size);
        }
    }

    let n_airgroups = pctx.global_info.air_groups.len();
    for airgroup in 0..n_airgroups {
        let setup = setups.sctx_recursive2.as_ref().unwrap().get_setup(airgroup, 0);
        max_trace = max_trace.max(setup.n_cols as usize * (1 << (setup.stark_info.stark_struct.n_bits)) as usize);
        max_prover_size = max_prover_size.max(setup.prover_buffer_size);
    }

    if cfg!(feature = "gpu") {
        max_prover_size = 0;
    }

    Ok((max_trace, max_prover_size as usize))
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
    let mut rem = n % 3;
    while n > 1 {
        let next = n / 3;
        rem = n % 3;
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
