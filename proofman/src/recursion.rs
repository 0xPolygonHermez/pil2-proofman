use libloading::{Library, Symbol};
use p3_field::PrimeField64;
use std::ffi::CString;
use std::fs::File;
use proofman_starks_lib_c::*;
use std::path::{Path, PathBuf};
use std::io::Read;
use num_traits::ToPrimitive;

use proofman_common::{load_const_pols, load_const_pols_tree, Proof, ProofCtx, ProofType, Setup, SetupCtx, SetupsVadcop};

use std::os::raw::{c_void, c_char};

use proofman_util::{
    create_buffer_fast, create_buffer_fast_u8, timer_start_info, timer_stop_and_log_info, timer_stop_and_log_trace,
    timer_start_trace,
};

use crate::{add_publics_circom, add_publics_aggregation};

type GetWitnessFunc =
    unsafe extern "C" fn(zkin: *mut u64, dat_file: *const c_char, witness: *mut c_void, n_mutexes: u64);

type GetWitnessFinalFunc =
    unsafe extern "C" fn(zkin: *mut c_void, dat_file: *const c_char, witness: *mut c_void, n_mutexes: u64);

type GetSizeWitnessFunc = unsafe extern "C" fn() -> u64;

#[derive(Debug)]
pub struct MaxSizes {
    pub max_trace_area: u64,
    pub max_const_area: u64,
    pub max_aux_trace_area: u64,
    pub max_const_tree_size: u64,
    pub recursive: bool,
}

pub fn discover_max_sizes<F: PrimeField64>(pctx: &ProofCtx<F>, sctx: &SetupCtx<F>) -> MaxSizes {
    let max_trace_area = 0;
    let max_const_tree_size = 0;
    let max_const_area = 0;

    let mut max_aux_trace_area = 0;

    let mut update_max_values = |setup: &Setup<F>| {
        max_aux_trace_area = max_aux_trace_area.max(setup.prover_buffer_size);
    };

    let instances = pctx.dctx_get_instances();
    let my_instances = pctx.dctx_get_my_instances();

    for instance_id in my_instances {
        let (airgroup_id, air_id, _) = instances[instance_id];

        let setup = sctx.get_setup(airgroup_id, air_id);
        update_max_values(setup);
    }

    MaxSizes { max_trace_area, max_const_area, max_aux_trace_area, max_const_tree_size, recursive: false }
}

pub fn discover_max_sizes_aggregation<F: PrimeField64>(pctx: &ProofCtx<F>, setups: &SetupsVadcop<F>) -> MaxSizes {
    let mut max_trace_area = 0;
    let mut max_const_area = 0;
    let mut max_aux_trace_area = 0;
    let mut max_const_tree_size = 0;

    let mut update_max_values = |setup: &Setup<F>| {
        let n = 1 << setup.stark_info.stark_struct.n_bits;
        max_trace_area = max_trace_area.max(setup.stark_info.map_sections_n["cm1"] * n);
        max_const_area = max_const_area.max(setup.stark_info.n_constants * n);
        max_aux_trace_area = max_aux_trace_area.max(setup.prover_buffer_size);
        max_const_tree_size = max_const_tree_size.max(setup.const_tree_size as u64);
    };

    let instances = pctx.dctx_get_instances();
    let my_instances = pctx.dctx_get_my_instances();

    for instance_id in my_instances {
        let (airgroup_id, air_id, _) = instances[instance_id];

        if pctx.global_info.get_air_has_compressor(airgroup_id, air_id) {
            let setup = setups.sctx_compressor.as_ref().unwrap().get_setup(airgroup_id, air_id);
            update_max_values(setup);
        }

        let setup = setups.sctx_recursive1.as_ref().unwrap().get_setup(airgroup_id, air_id);
        update_max_values(setup);

        let setup = setups.sctx_recursive2.as_ref().unwrap().get_setup(airgroup_id, air_id);
        update_max_values(setup);
    }

    if let Some(setup) = setups.setup_vadcop_final.as_ref() {
        update_max_values(setup);
    }

    if let Some(setup) = setups.setup_recursivef.as_ref() {
        update_max_values(setup);
    }

    MaxSizes { max_trace_area, max_const_area, max_aux_trace_area, max_const_tree_size, recursive: true }
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
        timer_start_info!(GENERATE_WITNESS);
        let setup = setups.sctx_compressor.as_ref().unwrap().get_setup(airgroup_id, air_id);

        let publics_circom_size =
            pctx.global_info.n_publics + pctx.global_info.n_proof_values.iter().sum::<usize>() * 3 + 3;

        let mut updated_proof: Vec<u64> = create_buffer_fast(proof.proof.len() + publics_circom_size);
        updated_proof[publics_circom_size..].copy_from_slice(&proof.proof);
        add_publics_circom(&mut updated_proof, 0, pctx, "", false);
        let circom_witness = generate_witness::<F>(setup, &updated_proof)?;
        timer_stop_and_log_info!(GENERATE_WITNESS);
        Ok(Proof::new_witness(ProofType::Compressor, airgroup_id, air_id, proof.global_idx, circom_witness, 24))
    } else {
        timer_start_info!(GENERATE_WITNESS);
        let setup = setups.sctx_recursive1.as_ref().unwrap().get_setup(airgroup_id, air_id);

        let recursive2_verkey =
            pctx.global_info.get_air_setup_path(airgroup_id, air_id, &ProofType::Recursive2).display().to_string()
                + ".verkey.json";

        let publics_circom_size =
            pctx.global_info.n_publics + pctx.global_info.n_proof_values.iter().sum::<usize>() * 3 + 3 + 4;

        let mut updated_proof: Vec<u64> = create_buffer_fast(proof.proof.len() + publics_circom_size);

        if proof.proof_type == ProofType::Compressor {
            let n_publics_aggregation = 1 + 4 * pctx.global_info.agg_types[airgroup_id].len() + 10;
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
        timer_stop_and_log_info!(GENERATE_WITNESS);
        Ok(Proof::new_witness(ProofType::Recursive1, airgroup_id, air_id, proof.global_idx, circom_witness, 24))
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

    let mut updated_proof_recursive2: Vec<u64> = create_buffer_fast(updated_proof_size);

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
    Ok(Proof::new_witness(ProofType::Recursive2, airgroup_id, 0, None, circom_witness, 24))
}

#[allow(clippy::too_many_arguments)]
pub fn generate_recursive_proof<F: PrimeField64>(
    pctx: &ProofCtx<F>,
    setups: &SetupsVadcop<F>,
    witness: &Proof<F>,
    trace: &[F],
    prover_buffer: &[F],
    output_dir_path: &Path,
    d_buffers: *mut c_void,
    load_constants: bool,
) -> Proof<F> {
    timer_start_info!(GEN_RECURSIVE_PROOF);
    let global_info_path = pctx.global_info.get_proving_key_path().join("pilout.globalInfo.json");
    let global_info_file: &str = global_info_path.to_str().unwrap();

    let (airgroup_id, air_id, air_instance_id, output_file_path, vadcop) =
        if witness.proof_type == ProofType::VadcopFinal {
            let output_file_path_ = output_dir_path.join("proofs/vadcop_final_proof.json");
            (0, 0, 0, output_file_path_, false)
        } else {
            let (airgroup_id_, air_id_) = (witness.airgroup_id, witness.air_id);
            let air_instance_id = if witness.proof_type == ProofType::Recursive2 {
                0
            } else {
                pctx.dctx_find_air_instance_id(witness.global_idx.unwrap())
            };
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
            (airgroup_id_, air_id_, air_instance_id, output_file_path_, true)
        };

    let proof_file = match pctx.options.debug_info.save_proofs_to_file || witness.proof_type == ProofType::VadcopFinal {
        true => output_file_path.to_string_lossy().into_owned(),
        false => String::from(""),
    };

    let setup = setups.get_setup(airgroup_id, air_id, &witness.proof_type);
    let setup_path = setup.setup_path.clone();
    let p_setup: *mut c_void = (&setup.p_setup).into();

    let exec_filename = setup_path.display().to_string() + ".exec";
    let exec_filename_str = CString::new(exec_filename.as_str()).unwrap();
    let exec_filename_ptr = exec_filename_str.as_ptr() as *mut std::os::raw::c_char;

    let mut publics = vec![F::ZERO; setup.stark_info.n_publics as usize];

    get_committed_pols_c(
        witness.circom_witness.as_ptr() as *mut u8,
        exec_filename_ptr,
        trace.as_ptr() as *mut u8,
        publics.as_mut_ptr() as *mut u8,
        setup.size_witness.read().unwrap().unwrap(),
        1 << (setup.stark_info.stark_struct.n_bits),
        setup.stark_info.n_publics,
        witness.n_cols as u64,
    );

    let publics_aggregation = 1 + 4 * pctx.global_info.agg_types[airgroup_id].len() + 10;

    let mut new_proof_size = setup.proof_size;

    let add_aggregation_publics = witness.proof_type != ProofType::VadcopFinal;
    if add_aggregation_publics {
        new_proof_size += publics_aggregation as u64;
    }

    let mut new_proof = vec![0; new_proof_size as usize];

    let initial_idx = if witness.proof_type == ProofType::VadcopFinal { 0 } else { publics_aggregation };

    gen_recursive_proof_c(
        p_setup,
        trace.as_ptr() as *mut u8,
        prover_buffer.as_ptr() as *mut u8,
        setup.get_const_ptr(),
        setup.get_const_tree_ptr(),
        publics.as_ptr() as *mut u8,
        new_proof[initial_idx..].as_mut_ptr(),
        &proof_file,
        global_info_file,
        airgroup_id as u64,
        air_id as u64,
        air_instance_id as u64,
        vadcop,
        d_buffers,
        load_constants,
    );

    if add_aggregation_publics {
        add_publics_aggregation(&mut new_proof, 0, &publics, publics_aggregation);
    }

    timer_stop_and_log_info!(GEN_RECURSIVE_PROOF);
    if witness.proof_type == ProofType::Compressor || witness.proof_type == ProofType::Recursive1 {
        Proof::new(witness.proof_type.clone(), witness.airgroup_id, witness.air_id, witness.global_idx, new_proof)
    } else {
        Proof::new(witness.proof_type.clone(), witness.airgroup_id, witness.air_id, None, new_proof)
    }
}

#[allow(clippy::too_many_arguments)]
pub fn aggregate_recursive2_proofs<F: PrimeField64>(
    pctx: &ProofCtx<F>,
    setups: &SetupsVadcop<F>,
    proofs: &[Vec<Proof<F>>],
    trace: &[F],
    prover_buffer: &[F],
    output_dir_path: PathBuf,
    d_buffers: *mut c_void,
) -> Result<Proof<F>, Box<dyn std::error::Error>> {
    const MY_NAME: &str = "AggProof";

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
        let publics_aggregation = 1 + 4 * pctx.global_info.agg_types[airgroup].len() + 10;
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

                        let circom_witness = gen_witness_aggregation::<F>(pctx, setups, &proof1, &proof2, &proof3)?;

                        let recursive2_proof = generate_recursive_proof::<F>(
                            pctx,
                            setups,
                            &circom_witness,
                            trace,
                            prover_buffer,
                            &output_dir_path,
                            d_buffers,
                            true,
                        );

                        airgroup_proofs[airgroup][j] = Some(recursive2_proof.proof);

                        log::info!("{}: ··· Recursive 2 Proof generated.", MY_NAME);
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
    output_dir_path: PathBuf,
    d_buffers: *mut c_void,
) -> Result<Proof<F>, Box<dyn std::error::Error>> {
    const MY_NAME: &str = "AggProof";

    let setup = setups.setup_vadcop_final.as_ref().unwrap();
    let circom_witness_vadcop_final = generate_witness::<F>(setup, &proof.proof)?;
    let new_proof = Proof::new_witness(ProofType::VadcopFinal, 0, 0, None, circom_witness_vadcop_final, 24);
    log::info!("{}: ··· Generating vadcop final proof", MY_NAME);
    timer_start_trace!(GENERATE_VADCOP_FINAL_PROOF);
    let final_vadcop_proof = generate_recursive_proof::<F>(
        pctx,
        setups,
        &new_proof,
        trace,
        prover_buffer,
        &output_dir_path,
        d_buffers,
        true,
    );
    log::info!("{}: ··· Vadcop final Proof generated.", MY_NAME);
    timer_stop_and_log_trace!(GENERATE_VADCOP_FINAL_PROOF);

    Ok(final_vadcop_proof)
}

#[allow(clippy::too_many_arguments)]
pub fn generate_recursivef_proof<F: PrimeField64>(
    pctx: &ProofCtx<F>,
    setups: &SetupsVadcop<F>,
    proof: &[u64],
    trace: &[F],
    prover_buffer: &[F],
    output_dir_path: PathBuf,
) -> Result<*mut c_void, Box<dyn std::error::Error>> {
    const MY_NAME: &str = "RecProof";

    let global_info_path = pctx.global_info.get_proving_key_path().join("pilout.globalInfo.json");
    let global_info_file: &str = global_info_path.to_str().unwrap();

    let setup = setups.setup_recursivef.as_ref().unwrap();
    let p_setup: *mut c_void = (&setup.p_setup).into();

    let setup_path = pctx.global_info.get_setup_path("recursivef");

    let const_tree_size = setup.const_tree_size;
    let const_tree = create_buffer_fast(const_tree_size);
    let const_pols: Vec<F> = create_buffer_fast(setup.const_pols_size);

    load_const_pols(&setup_path, setup.const_pols_size, &const_pols);
    load_const_pols_tree(setup, &const_tree);

    let mut vadcop_final_proof: Vec<u64> = create_buffer_fast(proof.len() + pctx.global_info.n_publics);
    vadcop_final_proof[pctx.global_info.n_publics..].copy_from_slice(proof);

    let public_inputs = pctx.get_publics();
    for p in 0..pctx.global_info.n_publics {
        vadcop_final_proof[p] = (public_inputs[p].as_canonical_biguint()).to_u64().unwrap();
    }

    let circom_witness = generate_witness::<F>(setup, &vadcop_final_proof)?;

    let exec_filename = setup.setup_path.display().to_string() + ".exec";
    let exec_filename_str = CString::new(exec_filename.as_str()).unwrap();
    let exec_filename_ptr = exec_filename_str.as_ptr() as *mut std::os::raw::c_char;

    let publics = vec![F::ZERO; setup.stark_info.n_publics as usize];

    get_committed_pols_c(
        circom_witness.as_ptr() as *mut u8,
        exec_filename_ptr,
        trace.as_ptr() as *mut u8,
        publics.as_ptr() as *mut u8,
        setup.size_witness.read().unwrap().unwrap(),
        1 << (setup.stark_info.stark_struct.n_bits),
        setup.stark_info.n_publics,
        13,
    );

    let proof_file = match pctx.options.debug_info.save_proofs_to_file {
        true => output_dir_path.join("proofs/recursivef.json").to_string_lossy().into_owned(),
        false => String::from(""),
    };

    log::info!("{}: ··· Generating recursiveF proof", MY_NAME);
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
    log::info!("{}: ··· RecursiveF Proof generated.", MY_NAME);
    timer_stop_and_log_trace!(GENERATE_RECURSIVEF_PROOF);

    Ok(p_prove)
}

pub fn generate_fflonk_snark_proof<F: PrimeField64>(
    pctx: &ProofCtx<F>,
    proof: *mut c_void,
    output_dir_path: PathBuf,
) -> Result<(), Box<dyn std::error::Error>> {
    const MY_NAME: &str = "FinProof";

    let setup_path = pctx.global_info.get_setup_path("final");

    let rust_lib_filename = setup_path.display().to_string() + ".so";
    let rust_lib_path = Path::new(rust_lib_filename.as_str());

    if !rust_lib_path.exists() {
        return Err(format!("Rust lib dynamic library not found at path: {:?}", rust_lib_path).into());
    }
    let library: Library = unsafe { Library::new(rust_lib_path)? };

    let dat_filename = setup_path.display().to_string() + ".dat";
    let dat_filename_str = CString::new(dat_filename.as_str()).unwrap();
    let dat_filename_ptr = dat_filename_str.as_ptr() as *mut std::os::raw::c_char;

    unsafe {
        timer_start_trace!(CALCULATE_FINAL_WITNESS);

        let get_size_witness: Symbol<GetSizeWitnessFunc> = library.get(b"getSizeWitness\0")?;
        let size_witness = get_size_witness();

        let witness = create_buffer_fast_u8((size_witness * 32) as usize);
        let witness_ptr = witness.as_ptr() as *mut u8;

        let get_witness_final: Symbol<GetWitnessFinalFunc> = library.get(b"getWitness\0")?;

        let nmutex = 128;
        get_witness_final(proof, dat_filename_ptr, witness_ptr as *mut c_void, nmutex);

        timer_stop_and_log_trace!(CALCULATE_FINAL_WITNESS);

        timer_start_trace!(CALCULATE_FINAL_PROOF);
        let proof_file = output_dir_path.join("proofs").to_string_lossy().into_owned();

        let zkey_filename = setup_path.display().to_string() + ".zkey";
        log::info!("{}: ··· Generating final snark proof", MY_NAME);
        gen_final_snark_proof_c(witness_ptr, zkey_filename.as_str(), &proof_file);
        timer_stop_and_log_trace!(CALCULATE_FINAL_PROOF);
        log::info!("{}: ··· Final Snark Proof generated.", MY_NAME);
    }

    Ok(())
}

fn generate_witness<F: PrimeField64>(setup: &Setup<F>, zkin: &[u64]) -> Result<Vec<F>, Box<dyn std::error::Error>> {
    let rust_lib_filename = setup.setup_path.display().to_string() + ".so";
    let rust_lib_path = Path::new(rust_lib_filename.as_str());

    if !rust_lib_path.exists() {
        return Err(format!("Rust lib dynamic library not found at path: {:?}", rust_lib_path).into());
    }

    let library: Library = unsafe { Library::new(rust_lib_path)? };

    let dat_filename = setup.setup_path.display().to_string() + ".dat";
    let dat_filename_str = CString::new(dat_filename.as_str()).unwrap();
    let dat_filename_ptr = dat_filename_str.as_ptr() as *mut std::os::raw::c_char;

    let witness_size = get_witness_size(setup)?;

    let witness = vec![F::ZERO; witness_size];

    unsafe {
        let get_witness: Symbol<GetWitnessFunc> = library.get(b"getWitness\0")?;
        get_witness(zkin.as_ptr() as *mut u64, dat_filename_ptr, witness.as_ptr() as *mut c_void, 128);
    }

    Ok(witness)
}

pub fn get_recursive_buffer_sizes<F: PrimeField64>(
    pctx: &ProofCtx<F>,
    setups: &SetupsVadcop<F>,
) -> Result<(usize, usize), Box<dyn std::error::Error>> {
    let mut max_trace = 0;
    let mut max_prover_size = 0;
    let n_cols = 24;

    let instances = pctx.dctx_get_instances();
    let my_instances = pctx.dctx_get_my_instances();

    for instance_id in my_instances.iter() {
        let (airgroup_id, air_id, _) = instances[*instance_id];

        if pctx.global_info.get_air_has_compressor(airgroup_id, air_id) {
            let setup_compressor = setups.sctx_compressor.as_ref().unwrap().get_setup(airgroup_id, air_id);
            max_trace = max_trace.max(n_cols * (1 << (setup_compressor.stark_info.stark_struct.n_bits)) as usize);
            max_prover_size = max_prover_size.max(setup_compressor.prover_buffer_size);
        }

        let setup_recursive1 = setups.sctx_recursive1.as_ref().unwrap().get_setup(airgroup_id, air_id);
        max_trace = max_trace.max(n_cols * (1 << (setup_recursive1.stark_info.stark_struct.n_bits)) as usize);
        max_prover_size = max_prover_size.max(setup_recursive1.prover_buffer_size);
    }

    let n_airgroups = pctx.global_info.air_groups.len();
    for airgroup in 0..n_airgroups {
        let setup = setups.sctx_recursive2.as_ref().unwrap().get_setup(airgroup, 0);
        max_trace = max_trace.max(n_cols * (1 << (setup.stark_info.stark_struct.n_bits)) as usize);
        max_prover_size = max_prover_size.max(setup.prover_buffer_size);
    }

    Ok((max_trace, max_prover_size as usize))
}

fn get_witness_size<F: PrimeField64>(setup: &Setup<F>) -> Result<usize, Box<dyn std::error::Error>> {
    let exec_filename = setup.setup_path.display().to_string() + ".exec";

    let mut size_witness = setup.size_witness.read().unwrap().unwrap();

    let mut file = File::open(exec_filename)?; // Open the file

    let mut n_adds = [0u8; 8]; // Buffer for nAdds (u64 is 8 bytes)
    file.read_exact(&mut n_adds)?;
    let n_adds = u64::from_le_bytes(n_adds);

    size_witness += n_adds;

    Ok(size_witness as usize)
}

pub fn total_recursive_proofs(mut n: usize) -> usize {
    let mut total = 0;
    while n > 1 {
        let next = n / 3;
        let rem = n % 3;
        total += next;
        if next != 0 {
            n = next + rem;
        } else if rem != 1 {
            n = next;
        }
    }
    total
}
