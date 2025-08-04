use fields::PrimeField64;
use num_traits::ToPrimitive;
use std::fs::{self, File};
use std::io::{Read, Seek, SeekFrom};
use std::{collections::HashMap, path::PathBuf};

use colored::*;

use std::error::Error;

use proofman_common::{format_bytes, ProofCtx, ProofType, Setup, SetupCtx, SetupsVadcop};
use proofman_util::DeviceBuffer;
use proofman_starks_lib_c::custom_commit_size_c;
use proofman_starks_lib_c::load_device_setup_c;

use pil_std_lib::Std;
use witness::WitnessManager;

pub fn print_summary_info<F: PrimeField64>(pctx: &ProofCtx<F>, sctx: &SetupCtx<F>) {
    let mpi_rank = pctx.dctx_get_rank();
    let n_processes = pctx.dctx_get_n_processes();

    if n_processes > 1 {
        let (average_weight, max_weight, min_weight, max_deviation) = pctx.dctx_load_balance_info();
        tracing::info!(
            "Load balance. Average: {} max: {} min: {} deviation: {}",
            average_weight,
            max_weight,
            min_weight,
            max_deviation
        );
    }

    if mpi_rank == 0 {
        print_summary(pctx, sctx, true);
    }

    if n_processes > 1 {
        print_summary(pctx, sctx, false);
    }
}

pub fn print_summary<F: PrimeField64>(pctx: &ProofCtx<F>, sctx: &SetupCtx<F>, global: bool) {
    let mut air_info = HashMap::new();

    let mut air_instances = HashMap::new();

    let instances = pctx.dctx_get_instances();
    let mut n_instances = instances.len();

    let mut print = vec![global; instances.len()];

    if !global {
        let my_instances = pctx.dctx_get_my_instances();
        for instance_id in my_instances.iter() {
            print[*instance_id] = true;
        }
        n_instances = my_instances.len();
    }

    let mut memory_tables = 0 as f64;
    for (instance_id, &instance_info) in instances.iter().enumerate() {
        let (airgroup_id, air_id, all) = (instance_info.airgroup_id, instance_info.air_id, instance_info.all);
        if !print[instance_id] {
            continue;
        }
        let air_name = pctx.global_info.airs[airgroup_id][air_id].clone().name;
        let air_group_name = pctx.global_info.air_groups[airgroup_id].clone();
        let air_instance_map = air_instances.entry(air_group_name).or_insert_with(HashMap::new);
        if !air_instance_map.contains_key(&air_name.clone()) {
            let setup = sctx.get_setup(airgroup_id, air_id);
            let n_bits = setup.stark_info.stark_struct.n_bits;
            let memory_trace = (*setup.stark_info.map_sections_n.get("cm1").unwrap()
                * (1 << (setup.stark_info.stark_struct.n_bits))) as f64
                * 8.0;
            let memory_instance = setup.prover_buffer_size as f64 * 8.0;
            let memory_fixed =
                (setup.stark_info.n_constants * (1 << (setup.stark_info.stark_struct.n_bits))) as f64 * 8.0;
            if all {
                memory_tables += memory_trace;
            }
            let total_cols: u64 = setup
                .stark_info
                .map_sections_n
                .iter()
                .filter(|(key, _)| *key != "const")
                .map(|(_, value)| *value)
                .sum();
            air_info.insert(air_name.clone(), (n_bits, total_cols, memory_fixed, memory_trace, memory_instance));
        }
        let air_instance_map_key = air_instance_map.entry(air_name).or_insert(0);
        *air_instance_map_key += 1;
    }

    let mut air_groups: Vec<_> = air_instances.keys().collect();
    air_groups.sort();

    tracing::info!("{}", "--- TOTAL PROOF INSTANCES SUMMARY ------------------------".bright_white().bold());
    tracing::info!("    ► {} Air instances found:", n_instances);
    for air_group in air_groups.clone() {
        let air_group_instances = air_instances.get(air_group).unwrap();
        let mut air_names: Vec<_> = air_group_instances.keys().collect();
        air_names.sort();

        tracing::info!("      Air Group [{}]", air_group);
        for air_name in air_names {
            let count = air_group_instances.get(air_name).unwrap();
            let (n_bits, total_cols, _, _, _) = air_info.get(air_name).unwrap();
            tracing::info!(
                "      {}",
                format!("· {count} x Air [{air_name}] ({total_cols} x 2^{n_bits})").bright_white().bold()
            );
        }
    }
    tracing::info!("{}", "--- TOTAL PROVER MEMORY USAGE ----------------------------".bright_white().bold());
    let mut max_prover_memory = 0f64;
    for air_group in air_groups {
        let air_group_instances = air_instances.get(air_group).unwrap();
        let mut air_names: Vec<_> = air_group_instances.keys().collect();
        air_names.sort();

        for air_name in air_names {
            let count = air_group_instances.get(air_name).unwrap();
            let (_, _, _, memory_trace, memory_instance) = air_info.get(air_name).unwrap();
            let gpu = cfg!(feature = "gpu");
            if gpu {
                if max_prover_memory < *memory_instance {
                    max_prover_memory = *memory_instance;
                }
                tracing::info!(
                    "      · {}: {} per each of {} instance",
                    air_name,
                    format_bytes(*memory_instance),
                    count,
                );
            } else {
                if max_prover_memory < *memory_instance + *memory_trace {
                    max_prover_memory = *memory_instance + *memory_trace;
                }
                tracing::info!(
                    "      · {}: {} + {} per each of {} instance | Total: {}",
                    air_name,
                    format_bytes(*memory_trace),
                    format_bytes(*memory_instance),
                    count,
                    format_bytes(*memory_instance + *memory_trace)
                );
            }
        }
    }
    tracing::info!("      Total memory required by proofman: {}", format_bytes(max_prover_memory));
    tracing::info!("----------------------------------------------------------");
    tracing::info!("      Extra memory tables (CPU): {}", format_bytes(memory_tables));
    tracing::info!("----------------------------------------------------------");
}

fn check_const_tree<F: PrimeField64>(
    setup: &Setup<F>,
    aggregation: bool,
    final_snark: bool,
) -> Result<(), Box<dyn Error>> {
    let const_pols_tree_path = setup.setup_path.display().to_string() + ".consttree";
    let mut flags = String::new();
    if aggregation {
        flags.push_str(" -a");
    }
    if final_snark {
        flags.push_str(" -f");
    }

    if !PathBuf::from(&const_pols_tree_path).exists() {
        let error_message = format!(
            "Error: Unable to find the constant tree at '{const_pols_tree_path}'.\n\
            Please run the following command:\n\
            \x1b[1mcargo run --bin proofman-cli check-setup --proving-key <PROVING_KEY>{flags}\x1b[0m"
        );
        return Err(error_message.into());
    }

    let error_message = format!(
        "Error: The constant tree file at '{const_pols_tree_path}' exists but is invalid or corrupted.\n\
        Please regenerate it by running:\n\
        \x1b[1mcargo run --bin proofman-cli check-setup --proving-key <PROVING_KEY>{flags}\x1b[0m"
    );

    let const_pols_tree_size = setup.const_tree_size;
    match fs::metadata(&const_pols_tree_path) {
        Ok(metadata) => {
            let actual_size = metadata.len() as usize;
            if actual_size != const_pols_tree_size * 8 {
                return Err(error_message.into());
            }
        }
        Err(err) => {
            return Err(format!("Failed to get metadata for {}: {}", setup.air_name, err).into());
        }
    }
    if setup.setup_type != ProofType::RecursiveF {
        let verkey_path = setup.setup_path.display().to_string() + ".verkey.json";

        let mut contents = String::new();
        let mut file = File::open(verkey_path).unwrap();
        let _ = file.read_to_string(&mut contents).map_err(|err| format!("Failed to read verkey path file: {err}"));
        let verkey_u64: Vec<u64> = serde_json::from_str(&contents).unwrap();

        let mut file = File::open(&const_pols_tree_path)?;
        file.seek(SeekFrom::End(-32))?; // Move to 32 bytes before the end

        let mut buffer = [0u8; 32];
        file.read_exact(&mut buffer)?;

        for (i, verkey_val) in verkey_u64.iter().enumerate() {
            let byte_range = i * 8..(i + 1) * 8;
            let value = u64::from_le_bytes(buffer[byte_range].try_into()?);
            if value != *verkey_val {
                return Err(error_message.into());
            }
        }
    }
    Ok(())
}

pub fn check_tree_paths<F: PrimeField64>(pctx: &ProofCtx<F>, sctx: &SetupCtx<F>) -> Result<(), Box<dyn Error>> {
    for (airgroup_id, air_group) in pctx.global_info.airs.iter().enumerate() {
        for (air_id, _) in air_group.iter().enumerate() {
            let setup = sctx.get_setup(airgroup_id, air_id);
            check_const_tree(setup, false, false)?;

            let n_custom_commits = setup.stark_info.custom_commits.len();

            for commit_id in 0..n_custom_commits {
                if setup.stark_info.custom_commits[commit_id].stage_widths[0] > 0 {
                    let custom_commit_file_path = pctx
                        .get_custom_commits_fixed_buffer(&setup.stark_info.custom_commits[commit_id].name, false)
                        .unwrap();

                    if !PathBuf::from(&custom_commit_file_path).exists() {
                        let error_message = format!(
                            "Error: Unable to find {} custom commit at '{}'.\n\
                            Please run the following command:\n\
                            \x1b[1mcargo run --bin proofman-cli gen-custom-commits-fixed --witness-lib <WITNESS_LIB> --proving-key <PROVING_KEY> --custom-commits <CUSTOM_COMMITS_DIR> \x1b[0m",
                            setup.stark_info.custom_commits[commit_id].name,
                            custom_commit_file_path.display(),
                        );
                        tracing::warn!(error_message);
                        return Ok(());
                    }

                    let error_message = format!(
                        "Error: The custom commit file for {} at '{}' exists but is invalid or corrupted.\n\
                        Please regenerate it by running:\n\
                        \x1b[1mcargo run --bin proofman-cli gen-custom-commits-fixed --witness-lib <WITNESS_LIB> --proving-key <PROVING_KEY> --custom-commits <CUSTOM_COMMITS_DIR> \x1b[0m",
                        setup.stark_info.custom_commits[commit_id].name,
                        custom_commit_file_path.display(),
                    );

                    let size = custom_commit_size_c((&setup.p_setup).into(), commit_id as u64) as usize;

                    match fs::metadata(custom_commit_file_path) {
                        Ok(metadata) => {
                            let actual_size = metadata.len() as usize;
                            if actual_size != (size + 4) * 8 {
                                tracing::warn!(error_message);
                                return Ok(());
                            }
                        }
                        Err(err) => {
                            tracing::warn!(
                                "Failed to get metadata for {} for custom_commit {}: {}",
                                setup.air_name,
                                setup.stark_info.custom_commits[commit_id].name,
                                err
                            );
                        }
                    }
                }
            }
        }
    }
    Ok(())
}

pub fn check_tree_paths_vadcop<F: PrimeField64>(
    pctx: &ProofCtx<F>,
    setups: &SetupsVadcop<F>,
    final_snark: bool,
) -> Result<(), Box<dyn Error>> {
    let sctx_compressor = setups.sctx_compressor.as_ref().unwrap();
    for (airgroup_id, air_group) in pctx.global_info.airs.iter().enumerate() {
        for (air_id, _) in air_group.iter().enumerate() {
            if pctx.global_info.get_air_has_compressor(airgroup_id, air_id) {
                let setup = sctx_compressor.get_setup(airgroup_id, air_id);
                check_const_tree(setup, true, false)?;
            }
        }
    }

    let sctx_recursive1 = setups.sctx_recursive1.as_ref().unwrap();
    for (airgroup_id, air_group) in pctx.global_info.airs.iter().enumerate() {
        for (air_id, _) in air_group.iter().enumerate() {
            let setup = sctx_recursive1.get_setup(airgroup_id, air_id);
            check_const_tree(setup, true, false)?;
        }
    }

    let sctx_recursive2 = setups.sctx_recursive2.as_ref().unwrap();
    let n_airgroups = pctx.global_info.air_groups.len();
    for airgroup in 0..n_airgroups {
        let setup = sctx_recursive2.get_setup(airgroup, 0);
        check_const_tree(setup, true, false)?;
    }

    let setup_vadcop_final = setups.setup_vadcop_final.as_ref().unwrap();
    check_const_tree(setup_vadcop_final, true, false)?;

    if final_snark {
        let setup_recursivef = setups.setup_recursivef.as_ref().unwrap();
        check_const_tree(setup_recursivef, true, true)?;
    }

    Ok(())
}

pub fn initialize_fixed_pols_tree<F: PrimeField64>(
    pctx: &ProofCtx<F>,
    sctx: &SetupCtx<F>,
    setups: &SetupsVadcop<F>,
    d_buffers: &DeviceBuffer,
    aggregation: bool,
) {
    let gpu = cfg!(feature = "gpu");
    if gpu {
        for (airgroup_id, air_group) in pctx.global_info.airs.iter().enumerate() {
            for (air_id, _) in air_group.iter().enumerate() {
                let setup = sctx.get_setup(airgroup_id, air_id);
                let proof_type: &str = setup.setup_type.clone().into();
                tracing::info!(airgroup_id, air_id, proof_type, "Loading expressions setup in GPU");
                load_device_setup_c(
                    airgroup_id as u64,
                    air_id as u64,
                    proof_type,
                    (&setup.p_setup).into(),
                    d_buffers.get_ptr(),
                    setup.verkey.as_ptr() as *mut u8,
                );
            }
        }
    }

    if aggregation {
        for (airgroup_id, air_group) in pctx.global_info.airs.iter().enumerate() {
            for (air_id, _) in air_group.iter().enumerate() {
                if pctx.global_info.get_air_has_compressor(airgroup_id, air_id) {
                    let setup = setups.sctx_compressor.as_ref().unwrap().get_setup(airgroup_id, air_id);
                    if gpu {
                        let proof_type: &str = setup.setup_type.clone().into();
                        tracing::info!(airgroup_id, air_id, proof_type, "Loading expressions setup in GPU");
                        load_device_setup_c(
                            airgroup_id as u64,
                            air_id as u64,
                            proof_type,
                            (&setup.p_setup).into(),
                            d_buffers.get_ptr(),
                            setup.verkey.as_ptr() as *mut u8,
                        );
                    }
                }
            }
        }

        for (airgroup_id, air_group) in pctx.global_info.airs.iter().enumerate() {
            for (air_id, _) in air_group.iter().enumerate() {
                let setup = setups.sctx_recursive1.as_ref().unwrap().get_setup(airgroup_id, air_id);
                if gpu {
                    let proof_type: &str = setup.setup_type.clone().into();
                    tracing::info!(airgroup_id, air_id, proof_type, "Loading expressions setup in GPU");
                    load_device_setup_c(
                        airgroup_id as u64,
                        air_id as u64,
                        proof_type,
                        (&setup.p_setup).into(),
                        d_buffers.get_ptr(),
                        setup.verkey.as_ptr() as *mut u8,
                    );
                }
            }
        }

        let n_airgroups = pctx.global_info.air_groups.len();
        for airgroup_id in 0..n_airgroups {
            let setup = setups.sctx_recursive2.as_ref().unwrap().get_setup(airgroup_id, 0);
            if gpu {
                let proof_type: &str = setup.setup_type.clone().into();
                tracing::info!(airgroup_id, air_id = 0, proof_type, "Loading expressions setup in GPU");
                load_device_setup_c(
                    airgroup_id as u64,
                    0_u64,
                    proof_type,
                    (&setup.p_setup).into(),
                    d_buffers.get_ptr(),
                    setup.verkey.as_ptr() as *mut u8,
                );
            }
        }

        let setup_vadcop_final = setups.setup_vadcop_final.as_ref().unwrap();
        if gpu {
            let proof_type: &str = setup_vadcop_final.setup_type.clone().into();
            tracing::info!(airgroup_id = 0, air_id = 0, proof_type, "Loading expressions setup in GPU");
            load_device_setup_c(
                0_u64,
                0_u64,
                proof_type,
                (&setup_vadcop_final.p_setup).into(),
                d_buffers.get_ptr(),
                setup_vadcop_final.verkey.as_ptr() as *mut u8,
            );
        }
    }
}

pub fn initialize_witness_circom<F: PrimeField64>(
    pctx: &ProofCtx<F>,
    setups: &SetupsVadcop<F>,
    final_snark: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    for (airgroup_id, air_group) in pctx.global_info.airs.iter().enumerate() {
        for (air_id, _) in air_group.iter().enumerate() {
            if pctx.global_info.get_air_has_compressor(airgroup_id, air_id) {
                let setup = setups.sctx_compressor.as_ref().unwrap().get_setup(airgroup_id, air_id);
                setup.set_exec_file_data()?;
                setup.set_circom_circuit()?;
            }
            let setup = setups.sctx_recursive1.as_ref().unwrap().get_setup(airgroup_id, air_id);
            setup.set_exec_file_data()?;
            setup.set_circom_circuit()?;
        }
    }

    let n_airgroups = pctx.global_info.air_groups.len();
    for airgroup in 0..n_airgroups {
        let setup = setups.sctx_recursive2.as_ref().unwrap().get_setup(airgroup, 0);
        setup.set_circom_circuit()?;
        setup.set_exec_file_data()?;
    }

    let setup_vadcop_final = setups.setup_vadcop_final.as_ref().unwrap();
    setup_vadcop_final.set_circom_circuit()?;
    setup_vadcop_final.set_exec_file_data()?;

    if final_snark {
        let setup_recursivef = setups.setup_recursivef.as_ref().unwrap();
        setup_recursivef.set_circom_circuit()?;
        setup_recursivef.set_exec_file_data()?;
    }

    Ok(())
}

pub fn add_publics_circom<F: PrimeField64>(
    proof: &mut [u64],
    initial_index: usize,
    pctx: &ProofCtx<F>,
    recursive2_verkey: &str,
    add_root_agg: bool,
) {
    let init_index = initial_index;

    let publics = pctx.get_publics();
    for p in 0..pctx.global_info.n_publics {
        proof[init_index + p] = (publics[p].as_canonical_biguint()).to_u64().unwrap();
    }

    let proof_values = pctx.get_proof_values();
    let proof_values_map = pctx.global_info.proof_values_map.as_ref().unwrap();
    let mut p = 0;
    for (idx, proof_value_map) in proof_values_map.iter().enumerate() {
        if proof_value_map.stage == 1 {
            proof[init_index + pctx.global_info.n_publics + 3 * idx] =
                (proof_values[p].as_canonical_biguint()).to_u64().unwrap();
            proof[init_index + pctx.global_info.n_publics + 3 * idx + 1] = 0;
            proof[init_index + pctx.global_info.n_publics + 3 * idx + 2] = 0;
            p += 1;
        } else {
            proof[init_index + pctx.global_info.n_publics + 3 * idx] =
                (proof_values[p].as_canonical_biguint()).to_u64().unwrap();
            proof[init_index + pctx.global_info.n_publics + 3 * idx + 1] =
                (proof_values[p + 1].as_canonical_biguint()).to_u64().unwrap();
            proof[init_index + pctx.global_info.n_publics + 3 * idx + 2] =
                (proof_values[p + 2].as_canonical_biguint()).to_u64().unwrap();
            p += 3;
        }
    }

    let global_challenge = pctx.get_global_challenge();
    proof[init_index + pctx.global_info.n_publics + 3 * proof_values_map.len()] =
        (global_challenge[0].as_canonical_biguint()).to_u64().unwrap();
    proof[init_index + pctx.global_info.n_publics + 3 * proof_values_map.len() + 1] =
        (global_challenge[1].as_canonical_biguint()).to_u64().unwrap();
    proof[init_index + pctx.global_info.n_publics + 3 * proof_values_map.len() + 2] =
        (global_challenge[2].as_canonical_biguint()).to_u64().unwrap();

    if add_root_agg {
        let mut file = File::open(recursive2_verkey).expect("Unable to open file");
        let mut json_str = String::new();
        file.read_to_string(&mut json_str).expect("Unable to read file");
        let vk: Vec<u64> = serde_json::from_str(&json_str).expect("Unable to parse json");
        for i in 0..4 {
            proof[init_index + pctx.global_info.n_publics + 3 * proof_values_map.len() + 3 + i] = vk[i];
        }
    }
}

pub fn add_publics_aggregation<F: PrimeField64>(
    proof: &mut [u64],
    initial_index: usize,
    publics: &[F],
    n_publics: usize,
) {
    for p in 0..n_publics {
        proof[initial_index + p] = (publics[p].as_canonical_biguint()).to_u64().unwrap();
    }
}

pub fn register_std<F: PrimeField64>(wcm: &WitnessManager<F>, std: &Std<F>) {
    wcm.register_component_std(std.std_prod.clone());
    wcm.register_component_std(std.std_sum.clone());
    wcm.register_component_std(std.range_check.clone());

    if std.range_check.u8air.is_some() {
        wcm.register_component_std(std.range_check.u8air.clone().unwrap());
    }

    if std.range_check.u16air.is_some() {
        wcm.register_component_std(std.range_check.u16air.clone().unwrap());
    }

    if std.range_check.specified_ranges_air.is_some() {
        wcm.register_component_std(std.range_check.specified_ranges_air.clone().unwrap());
    }
}

pub fn register_std_dev<F: PrimeField64>(
    wcm: &WitnessManager<F>,
    std: &Std<F>,
    register_u8: bool,
    register_u16: bool,
    register_specified_ranges: bool,
) {
    wcm.register_component_std(std.std_prod.clone());
    wcm.register_component_std(std.std_sum.clone());
    wcm.register_component_std(std.range_check.clone());

    if register_u8 && std.range_check.u8air.is_some() {
        wcm.register_component_std(std.range_check.u8air.clone().unwrap());
    }

    if register_u16 && std.range_check.u16air.is_some() {
        wcm.register_component_std(std.range_check.u16air.clone().unwrap());
    }

    if register_specified_ranges && std.range_check.specified_ranges_air.is_some() {
        wcm.register_component_std(std.range_check.specified_ranges_air.clone().unwrap());
    }
}
