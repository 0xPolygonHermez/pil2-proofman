use libloading::{Library, Symbol};
use p3_field::PrimeField64;
use std::ffi::CString;
use std::fs::File;
use proofman_starks_lib_c::*;
use std::path::{Path, PathBuf};
use std::io::Read;
use num_traits::ToPrimitive;
use log::info;

use colored::*;

use proofman_common::{load_const_pols, load_const_pols_tree, ProofCtx, ProofType, Setup, SetupCtx, SetupsVadcop};

use std::os::raw::{c_void, c_char};

use proofman_util::{
    create_buffer_fast, create_buffer_fast_u8, timer_start_info, timer_stop_and_log_info, timer_stop_and_log_trace,
    timer_start_trace,
};

use crate::{verify_proof, add_publics_circom, add_publics_aggregation};

type GetWitnessFunc =
    unsafe extern "C" fn(zkin: *mut u64, dat_file: *const c_char, witness: *mut c_void, n_mutexes: u64);

type GetWitnessFinalFunc =
    unsafe extern "C" fn(zkin: *mut c_void, dat_file: *const c_char, witness: *mut c_void, n_mutexes: u64);

type GetSizeWitnessFunc = unsafe extern "C" fn() -> u64;

pub struct MaxSizes {
    pub max_trace_area: u64,
    pub max_const_area: u64,
    pub max_n_publics: u64,
    pub max_aux_trace_area: u64,
    pub max_const_tree_size: u64,
}

pub fn discover_max_sizes<F: PrimeField64>(pctx: &ProofCtx<F>, sctx: &SetupCtx<F>) -> MaxSizes {
    let mut max_trace_area = 0;
    let mut max_const_area = 0;
    let mut max_n_publics = 0;
    let mut max_aux_trace_area = 0;
    let mut max_const_tree_size = 0;

    let mut update_max_values = |setup: &Setup<F>| {
        let n = 1 << setup.stark_info.stark_struct.n_bits;
        max_trace_area = max_trace_area.max(setup.stark_info.map_sections_n["cm1"] * n);
        max_const_area = max_const_area.max(setup.stark_info.n_constants * n);
        max_n_publics = max_n_publics.max(setup.stark_info.n_publics);
        max_aux_trace_area = max_aux_trace_area.max(setup.prover_buffer_size);
        max_const_tree_size = max_const_tree_size.max(get_const_tree_size_c(setup.p_setup.p_stark_info));
    };

    let instances = pctx.dctx_get_instances();
    let my_instances = pctx.dctx_get_my_instances();

    for instance_id in my_instances {
        let (airgroup_id, air_id, _) = instances[instance_id];

        let setup = sctx.get_setup(airgroup_id, air_id);
        update_max_values(setup);
    }

    max_trace_area = 0;

    MaxSizes { max_trace_area, max_const_area, max_n_publics, max_aux_trace_area, max_const_tree_size }
}

pub fn discover_max_sizes_aggregation<F: PrimeField64>(pctx: &ProofCtx<F>, setups: &SetupsVadcop<F>) -> MaxSizes {
    let mut max_trace_area = 0;
    let mut max_const_area = 0;
    let mut max_n_publics = 0;
    let mut max_aux_trace_area = 0;
    let mut max_const_tree_size = 0;

    let mut update_max_values = |setup: &Setup<F>| {
        let n = 1 << setup.stark_info.stark_struct.n_bits;
        max_trace_area = max_trace_area.max(setup.stark_info.map_sections_n["cm1"] * n);
        max_const_area = max_const_area.max(setup.stark_info.n_constants * n);
        max_n_publics = max_n_publics.max(setup.stark_info.n_publics);
        max_aux_trace_area = max_aux_trace_area.max(setup.prover_buffer_size);
        max_const_tree_size = max_const_tree_size.max(get_const_tree_size_c(setup.p_setup.p_stark_info));
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

    MaxSizes { max_trace_area, max_const_area, max_n_publics, max_aux_trace_area, max_const_tree_size }
}

#[allow(clippy::too_many_arguments)]
pub fn aggregate_proofs<F: PrimeField64>(
    name: &str,
    pctx_aggregation: &ProofCtx<F>,
    setups: &SetupsVadcop<F>,
    proofs: &[Vec<u64>],
    circom_witness: &[F],
    publics: &[F],
    trace: &[F],
    prover_buffer: &[F],
    const_pols: &[F],
    const_tree: &[F],
    output_dir_path: PathBuf,
    d_buffers: *mut c_void,
) -> Result<(), Box<dyn std::error::Error>> {
    info!("{}: ··· Generating aggregated proofs", name);

    timer_start_info!(GENERATING_AGGREGATION_PROOFS);
    pctx_aggregation.dctx.read().unwrap().barrier();
    timer_start_info!(GENERATING_RECURSIVE2_PROOFS);
    let sctx_recursive2 = &setups.sctx_recursive2;
    let recursive2_proof = generate_vadcop_recursive2_proof(
        pctx_aggregation,
        sctx_recursive2.as_ref().unwrap(),
        proofs,
        circom_witness,
        publics,
        trace,
        prover_buffer,
        const_pols,
        const_tree,
        output_dir_path.clone(),
        d_buffers,
    )?;
    timer_stop_and_log_info!(GENERATING_RECURSIVE2_PROOFS);
    info!("{}: Recursive2 proofs generated successfully", name);

    pctx_aggregation.dctx.read().unwrap().barrier();
    if pctx_aggregation.dctx_get_rank() == 0 {
        let setup_final = setups.setup_vadcop_final.as_ref().unwrap();
        timer_start_info!(GENERATING_VADCOP_FINAL_PROOF);
        let mut final_proof = generate_vadcop_final_proof(
            pctx_aggregation,
            setup_final,
            &recursive2_proof,
            circom_witness,
            publics,
            trace,
            prover_buffer,
            const_pols,
            const_tree,
            output_dir_path.clone(),
            d_buffers,
        )?;
        timer_stop_and_log_info!(GENERATING_VADCOP_FINAL_PROOF);
        info!("{}: VadcopFinal proof generated successfully", name);

        timer_stop_and_log_info!(GENERATING_AGGREGATION_PROOFS);

        if pctx_aggregation.options.final_snark {
            timer_start_info!(GENERATING_RECURSIVE_F_PROOF);
            let recursivef_proof = generate_recursivef_proof(
                pctx_aggregation,
                setups.setup_recursivef.as_ref().unwrap(),
                &final_proof,
                circom_witness,
                publics,
                trace,
                prover_buffer,
                const_pols,
                const_tree,
                output_dir_path.clone(),
            )?;
            timer_stop_and_log_info!(GENERATING_RECURSIVE_F_PROOF);

            timer_start_info!(GENERATING_FFLONK_SNARK_PROOF);
            let _ = generate_fflonk_snark_proof(pctx_aggregation, recursivef_proof, output_dir_path.clone());
            timer_stop_and_log_info!(GENERATING_FFLONK_SNARK_PROOF);
        } else {
            let setup_path = pctx_aggregation.global_info.get_setup_path("vadcop_final");
            let stark_info_path = setup_path.display().to_string() + ".starkinfo.json";
            let expressions_bin_path = setup_path.display().to_string() + ".verifier.bin";
            let verkey_path = setup_path.display().to_string() + ".verkey.json";

            timer_start_info!(VERIFYING_VADCOP_FINAL_PROOF);
            let valid_proofs = verify_proof(
                final_proof.as_mut_ptr(),
                stark_info_path,
                expressions_bin_path,
                verkey_path,
                Some(pctx_aggregation.get_publics().clone()),
                None,
                None,
            );
            timer_stop_and_log_info!(VERIFYING_VADCOP_FINAL_PROOF);
            if !valid_proofs {
                log::info!("{}: ··· {}", name, "\u{2717} Vadcop Final proof was not verified".bright_red().bold());
                return Err("Vadcop Final proof was not verified".into());
            } else {
                log::info!("{}:     {}", name, "\u{2713} Vadcop Final proof was verified".bright_green().bold());
            }
        }
    }
    pctx_aggregation.dctx_barrier();
    info!("{}: Proofs generated successfully", name);
    pctx_aggregation.dctx.read().unwrap().barrier();
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub fn generate_vadcop_recursive1_proof<F: PrimeField64>(
    pctx: &ProofCtx<F>,
    setups: &SetupsVadcop<F>,
    global_idx: usize,
    proof: &[u64],
    circom_witness: &[F],
    publics: &[F],
    trace: &[F],
    prover_buffer: &[F],
    const_pols_compressor: &[F],
    const_pols_recursive1: &[F],
    const_pols_recursive2: &[F],
    const_tree_compressor: &[F],
    const_tree_recursive1: &[F],
    const_tree_recursive2: &[F],
    recursive2_proof: &mut [u64],
    aggregate: bool,
    output_dir_path: PathBuf,
    d_buffers: *mut c_void,
) -> Result<(), Box<dyn std::error::Error>> {
    const MY_NAME: &str = "AggProof";

    let global_info_path = pctx.global_info.get_proving_key_path().join("pilout.globalInfo.json");
    let global_info_file: &str = global_info_path.to_str().unwrap();

    let (airgroup_id, air_id) = pctx.dctx_get_instance_info(global_idx);

    let air_instance_name = &pctx.global_info.airs[airgroup_id][air_id].name;
    let air_instance_id = pctx.dctx_find_air_instance_id(global_idx);

    let mut recursive_proof: Vec<u64>;

    let has_compressor = pctx.global_info.get_air_has_compressor(airgroup_id, air_id);
    if has_compressor {
        timer_start_trace!(GENERATING_COMPRESSOR_PROOF);

        let setup = setups.sctx_compressor.as_ref().unwrap().get_setup(airgroup_id, air_id);
        let p_setup: *mut c_void = (&setup.p_setup).into();

        let setup_path = pctx.global_info.get_air_setup_path(airgroup_id, air_id, &ProofType::Compressor);

        let publics_circom_size =
            pctx.global_info.n_publics + pctx.global_info.n_proof_values.iter().sum::<usize>() * 3 + 3;
        let mut updated_proof: Vec<u64> = vec![0; proof.len() + publics_circom_size];
        // Copy proof to updated proof starting at the end of the publics_circom
        updated_proof[publics_circom_size..].copy_from_slice(proof);
        add_publics_circom(&mut updated_proof, 0, pctx, "", false);
        generate_witness::<F>(circom_witness, trace, publics, &setup_path, setup, &updated_proof, 21)?;

        log::info!(
            "{}: ··· Generating compressor proof for instance {} of {}",
            MY_NAME,
            air_instance_id,
            air_instance_name
        );

        let output_file_path =
            output_dir_path.join(format!("proofs/compressor_{}_{}.json", air_instance_name, global_idx));

        let proof_file = match pctx.options.debug_info.save_proofs_to_file {
            true => output_file_path.to_string_lossy().into_owned(),
            false => String::from(""),
        };

        recursive_proof = create_buffer_fast(setup.proof_size as usize);
        gen_recursive_proof_c(
            p_setup,
            trace.as_ptr() as *mut u8,
            prover_buffer.as_ptr() as *mut u8,
            const_pols_compressor.as_ptr() as *mut u8,
            const_tree_compressor.as_ptr() as *mut u8,
            publics.as_ptr() as *mut u8,
            recursive_proof.as_mut_ptr(),
            &proof_file,
            global_info_file,
            airgroup_id as u64,
            air_id as u64,
            air_instance_id as u64,
            true,
            d_buffers,
        );

        log::info!("{}: ··· Compressor Proof generated.", MY_NAME);
        timer_stop_and_log_trace!(GENERATING_COMPRESSOR_PROOF);
    } else {
        recursive_proof = proof.to_vec();
    }

    timer_start_trace!(GENERATE_RECURSIVE1_PROOF);

    let setup = setups.sctx_recursive1.as_ref().unwrap().get_setup(airgroup_id, air_id);
    let p_setup: *mut c_void = (&setup.p_setup).into();

    let setup_path = pctx.global_info.get_air_setup_path(airgroup_id, air_id, &ProofType::Recursive1);

    let recursive2_verkey =
        pctx.global_info.get_air_setup_path(airgroup_id, air_id, &ProofType::Recursive2).display().to_string()
            + ".verkey.json";

    let publics_circom_size =
        pctx.global_info.n_publics + pctx.global_info.n_proof_values.iter().sum::<usize>() * 3 + 3 + 4;
    let publics_aggregation = 1 + 4 * pctx.global_info.agg_types[airgroup_id].len() + 10;

    let mut updated_proof_size = recursive_proof.len() + publics_circom_size;
    if has_compressor {
        updated_proof_size += publics_aggregation;
    }

    let mut updated_proof: Vec<u64> = vec![0; updated_proof_size];
    let mut initial_index = 0;
    if has_compressor {
        add_publics_aggregation(&mut updated_proof, initial_index, publics, publics_aggregation);
        initial_index += publics_aggregation;
    }
    add_publics_circom(&mut updated_proof, initial_index, pctx, &recursive2_verkey, true);

    initial_index += publics_circom_size;

    updated_proof[initial_index..].copy_from_slice(&recursive_proof);

    generate_witness::<F>(circom_witness, trace, publics, &setup_path, setup, &updated_proof, 21)?;

    log::info!(
        "{}: ··· Generating recursive1 proof for instance {} of {}",
        MY_NAME,
        air_instance_id,
        air_instance_name
    );

    let output_file_path = output_dir_path.join(format!("proofs/recursive1_{}_{}.json", air_instance_name, global_idx));

    let proof_file = match pctx.options.debug_info.save_proofs_to_file {
        true => output_file_path.to_string_lossy().into_owned(),
        false => String::from(""),
    };

    let mut recursive1_proof = create_buffer_fast(setup.proof_size as usize + publics_aggregation);
    gen_recursive_proof_c(
        p_setup,
        trace.as_ptr() as *mut u8,
        prover_buffer.as_ptr() as *mut u8,
        const_pols_recursive1.as_ptr() as *mut u8,
        const_tree_recursive1.as_ptr() as *mut u8,
        publics.as_ptr() as *mut u8,
        recursive1_proof[publics_aggregation..].as_mut_ptr(),
        &proof_file,
        global_info_file,
        airgroup_id as u64,
        air_id as u64,
        air_instance_id as u64,
        true,
        d_buffers,
    );

    log::info!("{}: ··· Recursive1 Proof generated.", MY_NAME);
    timer_stop_and_log_trace!(GENERATE_RECURSIVE1_PROOF);

    add_publics_aggregation(&mut recursive1_proof, 0, publics, publics_aggregation);

    if aggregate {
        let updated_proof_size = 2 * recursive1_proof.len() + publics_circom_size;

        let mut updated_proof_recursive2: Vec<u64> = vec![0; updated_proof_size];
        updated_proof_recursive2[publics_circom_size..(publics_circom_size + recursive1_proof.len())]
            .copy_from_slice(&recursive1_proof);
        updated_proof_recursive2[(publics_circom_size + recursive1_proof.len())..].copy_from_slice(recursive2_proof);

        add_publics_circom(&mut updated_proof_recursive2, 0, pctx, &recursive2_verkey, true);

        let setup_path = pctx.global_info.get_air_setup_path(airgroup_id, 0, &ProofType::Recursive2);
        let setup_recursive2 = setups.sctx_recursive2.as_ref().unwrap().get_setup(airgroup_id, 0);
        let p_setup_recursive2 = (&setup_recursive2.p_setup).into();

        generate_witness(circom_witness, trace, publics, &setup_path, setup_recursive2, &updated_proof_recursive2, 21)?;

        timer_start_trace!(GENERATE_RECURSIVE2_PROOF);
        let air_instance_name = &pctx.global_info.airs[airgroup_id][0].name;

        log::info!("{}: ··· Generating recursive2 proof for instances of {}", MY_NAME, air_instance_name);

        gen_recursive_proof_c(
            p_setup_recursive2,
            trace.as_ptr() as *mut u8,
            prover_buffer.as_ptr() as *mut u8,
            const_pols_recursive2.as_ptr() as *mut u8,
            const_tree_recursive2.as_ptr() as *mut u8,
            publics.as_ptr() as *mut u8,
            recursive2_proof[publics_aggregation..].as_mut_ptr(),
            "",
            global_info_file,
            airgroup_id as u64,
            0,
            0,
            true,
            d_buffers,
        );

        add_publics_aggregation(recursive2_proof, 0, publics, publics_aggregation);
        timer_stop_and_log_trace!(GENERATE_RECURSIVE2_PROOF);
        log::info!("{}: ··· Recursive2 Proof generated.", MY_NAME);
        Ok(())
    } else {
        recursive2_proof.copy_from_slice(&recursive1_proof);
        Ok(())
    }
}

#[allow(clippy::too_many_arguments)]
pub fn generate_vadcop_recursive2_proof<F: PrimeField64>(
    pctx: &ProofCtx<F>,
    sctx: &SetupCtx<F>,
    proofs: &[Vec<u64>],
    circom_witness: &[F],
    publics: &[F],
    trace: &[F],
    prover_buffer: &[F],
    const_pols: &[F],
    const_tree: &[F],
    output_dir_path: PathBuf,
    d_buffers: *mut c_void,
) -> Result<Vec<u64>, Box<dyn std::error::Error>> {
    const MY_NAME: &str = "AggProof";

    let global_info_path = pctx.global_info.get_proving_key_path().join("pilout.globalInfo.json");
    let global_info_file: &str = global_info_path.to_str().unwrap();

    let mut dctx = pctx.dctx.write().unwrap();
    let n_processes = dctx.n_processes as usize;
    let rank = dctx.rank as usize;
    let airgroup_instances_alive = &dctx.airgroup_instances_alives;
    let n_airgroups = pctx.global_info.air_groups.len();
    let mut alives = vec![0; n_airgroups];
    let mut airgroup_proofs: Vec<Vec<Option<Vec<u64>>>> = Vec::with_capacity(n_airgroups);

    // Pre-process data before starting recursion loop
    for airgroup in 0..n_airgroups {
        airgroup_proofs.push(vec![None; n_processes]);
        if airgroup_instances_alive[airgroup][rank] == 1 {
            airgroup_proofs[airgroup][rank] = Some(proofs[airgroup].clone());
        } else if rank == 0 {
            let setup = sctx.get_setup(airgroup, 0);
            let publics_aggregation = 1 + 4 * pctx.global_info.agg_types[airgroup].len() + 10;
            airgroup_proofs[airgroup][rank] = Some(vec![0; setup.proof_size as usize + publics_aggregation]);
        }
        for p in 0..n_processes {
            alives[airgroup] += airgroup_instances_alive[airgroup][p];
        }
    }

    // agregation loop
    loop {
        dctx.barrier();
        dctx.distribute_recursive2_proofs(&alives, &mut airgroup_proofs);
        let mut pending_agregations = false;
        for airgroup in 0..n_airgroups {
            let setup = sctx.get_setup(airgroup, 0);
            let p_setup: *mut c_void = (&setup.p_setup).into();
            load_const_pols_tree(setup, const_tree);
            load_const_pols(&setup.setup_path, setup.const_pols_size, const_pols);
            let publics_circom_size =
                pctx.global_info.n_publics + pctx.global_info.n_proof_values.iter().sum::<usize>() * 3 + 3 + 4;
            let publics_aggregation = 1 + 4 * pctx.global_info.agg_types[airgroup].len() + 10;

            //create a vector of sice indices length
            let mut alive = alives[airgroup];
            if alive > 1 {
                for i in 0..alive / 2 {
                    let j = i * 2;
                    if airgroup_proofs[airgroup][j].is_none() {
                        continue;
                    }
                    if j + 1 < alive {
                        if airgroup_proofs[airgroup][j + 1].is_none() {
                            panic!("Recursive2 proof is missing");
                        }

                        let proof1 = airgroup_proofs[airgroup][j].clone().unwrap();
                        let proof2 = airgroup_proofs[airgroup][j + 1].clone().unwrap();
                        let updated_proof_size = proof1.len() + proof2.len() + publics_circom_size;

                        let mut updated_proof: Vec<u64> = vec![0; updated_proof_size];
                        updated_proof[publics_circom_size..(publics_circom_size + proof1.len())]
                            .copy_from_slice(&proof1);
                        updated_proof[(publics_circom_size + proof1.len())..].copy_from_slice(&proof2);

                        let recursive2_verkey = pctx
                            .global_info
                            .get_air_setup_path(airgroup, 0, &ProofType::Recursive2)
                            .display()
                            .to_string()
                            + ".verkey.json";

                        add_publics_circom(&mut updated_proof, 0, pctx, &recursive2_verkey, true);

                        let setup_path = pctx.global_info.get_air_setup_path(airgroup, 0, &ProofType::Recursive2);

                        generate_witness(circom_witness, trace, publics, &setup_path, setup, &updated_proof, 21)?;

                        timer_start_trace!(GENERATE_RECURSIVE2_PROOF);
                        let proof_file = match pctx.options.debug_info.save_proofs_to_file {
                            true => output_dir_path
                                .join(format!(
                                    "proofs/recursive2_{}_{}_{}.json",
                                    pctx.global_info.air_groups[airgroup],
                                    j,
                                    j + 1
                                ))
                                .to_string_lossy()
                                .into_owned(),
                            false => String::from(""),
                        };

                        let air_instance_name = &pctx.global_info.airs[airgroup][0].name;

                        log::info!(
                            "{}: ··· Generating recursive2 proof for instances of {}",
                            MY_NAME,
                            air_instance_name
                        );

                        let mut recursive2_proof = create_buffer_fast(setup.proof_size as usize + publics_aggregation);
                        gen_recursive_proof_c(
                            p_setup,
                            trace.as_ptr() as *mut u8,
                            prover_buffer.as_ptr() as *mut u8,
                            const_pols.as_ptr() as *mut u8,
                            const_tree.as_ptr() as *mut u8,
                            publics.as_ptr() as *mut u8,
                            recursive2_proof[publics_aggregation..].as_mut_ptr(),
                            &proof_file,
                            global_info_file,
                            airgroup as u64,
                            0,
                            0,
                            true,
                            d_buffers,
                        );

                        add_publics_aggregation(&mut recursive2_proof, 0, publics, publics_aggregation);

                        airgroup_proofs[airgroup][j] = Some(recursive2_proof);

                        timer_stop_and_log_trace!(GENERATE_RECURSIVE2_PROOF);
                        log::info!("{}: ··· Recursive 2 Proof generated.", MY_NAME);
                    }
                }
                alive = (alive + 1) / 2;
                //compact elements
                for i in 0..alive {
                    airgroup_proofs[airgroup][i] = airgroup_proofs[airgroup][i * 2].clone();
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

    Ok(updated_proof)
}

#[allow(clippy::too_many_arguments)]
pub fn generate_vadcop_final_proof<F: PrimeField64>(
    pctx: &ProofCtx<F>,
    setup: &Setup<F>,
    proof: &[u64],
    circom_witness: &[F],
    publics: &[F],
    trace: &[F],
    prover_buffer: &[F],
    const_pols: &[F],
    const_tree: &[F],
    output_dir_path: PathBuf,
    d_buffers: *mut c_void,
) -> Result<Vec<u64>, Box<dyn std::error::Error>> {
    const MY_NAME: &str = "AggProof";

    let global_info_path = pctx.global_info.get_proving_key_path().join("pilout.globalInfo.json");
    let global_info_file: &str = global_info_path.to_str().unwrap();

    let p_setup: *mut c_void = (&setup.p_setup).into();

    let setup_path = pctx.global_info.get_setup_path("vadcop_final");

    generate_witness::<F>(circom_witness, trace, publics, &setup_path, setup, proof, 21)?;

    let proof_file = output_dir_path.join("proofs/vadcop_final_proof.json").to_string_lossy().into_owned();

    log::info!("{}: ··· Generating vadcop final proof", MY_NAME);
    timer_start_trace!(GENERATE_VADCOP_FINAL_PROOF);

    load_const_pols(&setup.setup_path, setup.const_pols_size, const_pols);
    load_const_pols_tree(setup, const_tree);

    let mut final_vadcop_proof: Vec<u64> = create_buffer_fast(setup.proof_size as usize);
    gen_recursive_proof_c(
        p_setup,
        trace.as_ptr() as *mut u8,
        prover_buffer.as_ptr() as *mut u8,
        const_pols.as_ptr() as *mut u8,
        const_tree.as_ptr() as *mut u8,
        publics.as_ptr() as *mut u8,
        final_vadcop_proof.as_mut_ptr(),
        &proof_file,
        global_info_file,
        0,
        0,
        0,
        false,
        d_buffers,
    );
    log::info!("{}: ··· Vadcop final Proof generated.", MY_NAME);
    timer_stop_and_log_trace!(GENERATE_VADCOP_FINAL_PROOF);

    Ok(final_vadcop_proof)
}

#[allow(clippy::too_many_arguments)]
pub fn generate_recursivef_proof<F: PrimeField64>(
    pctx: &ProofCtx<F>,
    setup: &Setup<F>,
    proof: &[u64],
    circom_witness: &[F],
    publics: &[F],
    trace: &[F],
    prover_buffer: &[F],
    const_pols: &[F],
    const_tree: &[F],
    output_dir_path: PathBuf,
) -> Result<*mut c_void, Box<dyn std::error::Error>> {
    const MY_NAME: &str = "RecProof";

    let global_info_path = pctx.global_info.get_proving_key_path().join("pilout.globalInfo.json");
    let global_info_file: &str = global_info_path.to_str().unwrap();

    let p_setup: *mut c_void = (&setup.p_setup).into();

    let setup_path = pctx.global_info.get_setup_path("recursivef");

    load_const_pols(&setup_path, setup.const_pols_size, const_pols);
    load_const_pols_tree(setup, const_tree);

    let mut vadcop_final_proof: Vec<u64> = create_buffer_fast(proof.len() + pctx.global_info.n_publics);
    vadcop_final_proof[pctx.global_info.n_publics..].copy_from_slice(proof);

    let public_inputs = pctx.get_publics();
    for p in 0..pctx.global_info.n_publics {
        vadcop_final_proof[p] = (public_inputs[p].as_canonical_biguint()).to_u64().unwrap();
    }

    generate_witness::<F>(circom_witness, trace, publics, &setup_path, setup, &vadcop_final_proof, 13)?;

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

fn generate_witness<F: PrimeField64>(
    witness: &[F],
    buffer: &[F],
    publics: &[F],
    setup_path: &Path,
    setup: &Setup<F>,
    zkin: &[u64],
    n_cols: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    // Load the symbol (function) from the library
    let rust_lib_filename = setup_path.display().to_string() + ".so";
    let rust_lib_path = Path::new(rust_lib_filename.as_str());

    if !rust_lib_path.exists() {
        return Err(format!("Rust lib dynamic library not found at path: {:?}", rust_lib_path).into());
    }

    let library: Library = unsafe { Library::new(rust_lib_path)? };

    let dat_filename = setup_path.display().to_string() + ".dat";
    let dat_filename_str = CString::new(dat_filename.as_str()).unwrap();
    let dat_filename_ptr = dat_filename_str.as_ptr() as *mut std::os::raw::c_char;

    let nmutex = 128;

    let exec_filename = setup_path.display().to_string() + ".exec";
    let exec_filename_str = CString::new(exec_filename.as_str()).unwrap();
    let exec_filename_ptr = exec_filename_str.as_ptr() as *mut std::os::raw::c_char;

    let size_witness = unsafe {
        let get_size_witness: Symbol<GetSizeWitnessFunc> = library.get(b"getSizeWitness\0")?;
        get_size_witness()
    };

    timer_start_trace!(CALCULATE_WITNESS);

    unsafe {
        let get_witness: Symbol<GetWitnessFunc> = library.get(b"getWitness\0")?;
        get_witness(zkin.as_ptr() as *mut u64, dat_filename_ptr, witness.as_ptr() as *mut c_void, nmutex);
    }

    get_committed_pols_c(
        witness.as_ptr() as *mut u8,
        exec_filename_ptr,
        buffer.as_ptr() as *mut u8,
        publics.as_ptr() as *mut u8,
        size_witness,
        1 << (setup.stark_info.stark_struct.n_bits),
        setup.stark_info.n_publics,
        n_cols as u64,
    );
    timer_stop_and_log_trace!(CALCULATE_WITNESS);

    Ok(())
}

pub fn get_buff_sizes<F: PrimeField64>(
    pctx: &ProofCtx<F>,
    setups: &SetupsVadcop<F>,
) -> Result<(usize, usize, usize, usize), Box<dyn std::error::Error>> {
    let mut witness_size = 0;
    let mut publics = 0;
    let mut buffer = 0;
    let mut prover_size = 0;

    let instances = pctx.dctx_get_instances();
    let my_instances = pctx.dctx_get_my_instances();

    for instance_id in my_instances.iter() {
        let (airgroup_id, air_id, _) = instances[*instance_id];

        if pctx.global_info.get_air_has_compressor(airgroup_id, air_id) {
            let setup_compressor = setups.sctx_compressor.as_ref().unwrap().get_setup(airgroup_id, air_id);
            let setup_path = pctx.global_info.get_air_setup_path(airgroup_id, air_id, &ProofType::Compressor);
            let sizes = get_size(&setup_path, setup_compressor, 21)?;
            witness_size = witness_size.max(sizes.0);
            publics = publics.max(sizes.1);
            buffer = buffer.max(sizes.2);
            prover_size = prover_size.max(setup_compressor.prover_buffer_size);
        }

        let setup_recursive1 = setups.sctx_recursive1.as_ref().unwrap().get_setup(airgroup_id, air_id);
        let setup_path = pctx.global_info.get_air_setup_path(airgroup_id, air_id, &ProofType::Recursive1);
        let sizes = get_size(&setup_path, setup_recursive1, 21)?;
        witness_size = witness_size.max(sizes.0);
        publics = publics.max(sizes.1);
        buffer = buffer.max(sizes.2);
        prover_size = prover_size.max(setup_recursive1.prover_buffer_size);
    }

    let n_airgroups = pctx.global_info.air_groups.len();
    for airgroup in 0..n_airgroups {
        let setup = setups.sctx_recursive2.as_ref().unwrap().get_setup(airgroup, 0);
        let setup_path = pctx.global_info.get_air_setup_path(airgroup, 0, &ProofType::Recursive2);
        let sizes = get_size(&setup_path, setup, 21)?;
        witness_size = witness_size.max(sizes.0);
        publics = publics.max(sizes.1);
        buffer = buffer.max(sizes.2);
        prover_size = prover_size.max(setup.prover_buffer_size);
    }

    let setup_final = setups.setup_vadcop_final.as_ref().unwrap();
    let setup_path = pctx.global_info.get_setup_path("vadcop_final");
    let sizes = get_size(&setup_path, setup_final, 21)?;
    witness_size = witness_size.max(sizes.0);
    publics = publics.max(sizes.1);
    buffer = buffer.max(sizes.2);
    prover_size = prover_size.max(setup_final.prover_buffer_size);

    if pctx.options.final_snark {
        let setup_recursivef = setups.setup_recursivef.as_ref().unwrap();
        let setup_path = pctx.global_info.get_setup_path("recursivef");
        let sizes = get_size(&setup_path, setup_recursivef, 13)?;
        witness_size = witness_size.max(sizes.0);
        publics = publics.max(sizes.1);
        buffer = buffer.max(sizes.2);
        prover_size = prover_size.max(setup_recursivef.prover_buffer_size);
    }

    Ok((witness_size, publics, buffer, prover_size as usize))
}

fn get_size<F: PrimeField64>(
    setup_path: &Path,
    setup: &Setup<F>,
    n_cols: usize,
) -> Result<(usize, usize, usize), Box<dyn std::error::Error>> {
    // Load the symbol (function) from the library
    let rust_lib_filename = setup_path.display().to_string() + ".so";
    let rust_lib_path = Path::new(rust_lib_filename.as_str());

    if !rust_lib_path.exists() {
        return Err(format!("Rust lib dynamic library not found at path: {:?}", rust_lib_path).into());
    }

    let library: Library = unsafe { Library::new(rust_lib_path)? };

    let exec_filename = setup_path.display().to_string() + ".exec";

    let mut size_witness = unsafe {
        let get_size_witness: Symbol<GetSizeWitnessFunc> = library.get(b"getSizeWitness\0")?;
        get_size_witness()
    };

    let mut file = File::open(exec_filename)?; // Open the file

    let mut n_adds = [0u8; 8]; // Buffer for nAdds (u64 is 8 bytes)
    file.read_exact(&mut n_adds)?;
    let n_adds = u64::from_le_bytes(n_adds);

    size_witness += n_adds;

    let n_publics = setup.stark_info.n_publics as usize;
    let buffer_size = n_cols * (1 << (setup.stark_info.stark_struct.n_bits)) as usize;

    Ok((size_witness as usize, n_publics, buffer_size))
}
