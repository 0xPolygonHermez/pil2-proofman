use log::info;
use p3_field::PrimeField64;
use num_traits::ToPrimitive;
use std::fs::{self, File};
use std::io::{Read, Seek, SeekFrom};
use std::sync::Arc;
use std::{
    collections::{HashMap, HashSet},
    path::PathBuf,
};

use colored::*;

use std::error::Error;

use proofman_common::{format_bytes, ProofCtx, ProofType, Setup, SetupCtx, SetupsVadcop};
use proofman_util::DeviceBuffer;
use proofman_starks_lib_c::load_const_pols_gpu_c;

pub fn print_summary_info<F: PrimeField64>(name: &str, pctx: &ProofCtx<F>, sctx: &SetupCtx<F>) {
    let mpi_rank = pctx.dctx_get_rank();
    let n_processes = pctx.dctx_get_n_processes();

    if n_processes > 1 {
        let (average_weight, max_weight, min_weight, max_deviation) = pctx.dctx_load_balance_info();
        log::info!(
            "{}: Load balance. Average: {} max: {} min: {} deviation: {}",
            name,
            average_weight,
            max_weight,
            min_weight,
            max_deviation
        );
    }

    if mpi_rank == 0 {
        print_summary(name, pctx, sctx, true);
    }

    if n_processes > 1 {
        print_summary(name, pctx, sctx, false);
    }
}

pub fn print_summary<F: PrimeField64>(name: &str, pctx: &ProofCtx<F>, sctx: &SetupCtx<F>, global: bool) {
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
    for (instance_id, (airgroup_id, air_id, all)) in instances.iter().enumerate() {
        if !print[instance_id] {
            continue;
        }
        let air_name = pctx.global_info.airs[*airgroup_id][*air_id].clone().name;
        let air_group_name = pctx.global_info.air_groups[*airgroup_id].clone();
        let air_instance_map = air_instances.entry(air_group_name).or_insert_with(HashMap::new);
        if !air_instance_map.contains_key(&air_name.clone()) {
            let setup = sctx.get_setup(*airgroup_id, *air_id);
            let n_bits = setup.stark_info.stark_struct.n_bits;
            let memory_trace = (*setup.stark_info.map_sections_n.get("cm1").unwrap()
                * (1 << (setup.stark_info.stark_struct.n_bits))) as f64
                * 8.0;
            let memory_instance = setup.prover_buffer_size as f64 * 8.0;
            let memory_fixed =
                (setup.stark_info.n_constants * (1 << (setup.stark_info.stark_struct.n_bits))) as f64 * 8.0;
            if *all {
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

    info!("{}", format!("{}: --- TOTAL PROOF INSTANCES SUMMARY ------------------------", name).bright_white().bold());
    info!("{}:     ► {} Air instances found:", name, n_instances);
    for air_group in air_groups.clone() {
        let air_group_instances = air_instances.get(air_group).unwrap();
        let mut air_names: Vec<_> = air_group_instances.keys().collect();
        air_names.sort();

        info!("{}:       Air Group [{}]", name, air_group);
        for air_name in air_names {
            let count = air_group_instances.get(air_name).unwrap();
            let (n_bits, total_cols, _, _, _) = air_info.get(air_name).unwrap();
            info!(
                "{}:       {}",
                name,
                format!("· {} x Air [{}] ({} x 2^{})", count, air_name, total_cols, n_bits).bright_white().bold()
            );
        }
    }
    info!("{}: ----------------------------------------------------------", name);
    if pctx.options.verify_constraints {
        info!(
            "{}",
            format!("{}: --- TOTAL CONSTRAINT CHECKER MEMORY USAGE ----------------------------", name)
                .bright_white()
                .bold()
        );
    } else {
        info!(
            "{}",
            format!("{}: --- TOTAL PROVER MEMORY USAGE ----------------------------", name).bright_white().bold()
        );
    }
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
                info!(
                    "{}:       · {}: {} per each of {} instance",
                    name,
                    air_name,
                    format_bytes(*memory_instance),
                    count,
                );
            } else {
                if max_prover_memory < *memory_instance + *memory_trace {
                    max_prover_memory = *memory_instance + *memory_trace;
                }
                info!(
                    "{}:       · {}: {} + {} per each of {} instance | Total: {}",
                    name,
                    air_name,
                    format_bytes(*memory_trace),
                    format_bytes(*memory_instance),
                    count,
                    format_bytes(*memory_instance + *memory_trace)
                );
            }
        }
    }
    info!("{}:       Total memory required by proofman: {}", name, format_bytes(max_prover_memory));
    info!("{}: ----------------------------------------------------------", name);
    info!("{}:       Extra memory tables (CPU): {}", name, format_bytes(memory_tables));
    info!("{}: ----------------------------------------------------------", name);
}

pub fn check_paths(
    witness_lib_path: &PathBuf,
    input_data_path: &Option<PathBuf>,
    public_inputs_path: &Option<PathBuf>,
    proving_key_path: &PathBuf,
    output_dir_path: &PathBuf,
    verify_constraints: bool,
) -> Result<(), Box<dyn Error>> {
    // Check witness_lib path exists
    if !witness_lib_path.exists() {
        return Err(format!("Witness computation dynamic library not found at path: {:?}", witness_lib_path).into());
    }

    // Check input data path
    if let Some(input_data_path) = input_data_path {
        if !input_data_path.exists() {
            return Err(format!("Input data file not found at path: {:?}", input_data_path).into());
        }
    }

    // Check public_inputs_path is a folder
    if let Some(publics_path) = public_inputs_path {
        if !publics_path.exists() {
            return Err(format!("Public inputs file not found at path: {:?}", publics_path).into());
        }
    }

    check_paths2(proving_key_path, output_dir_path, verify_constraints)
}

pub fn check_paths2(
    proving_key_path: &PathBuf,
    output_dir_path: &PathBuf,
    verify_constraints: bool,
) -> Result<(), Box<dyn Error>> {
    // Check proving_key_path exists
    if !proving_key_path.exists() {
        return Err(format!("Proving key folder not found at path: {:?}", proving_key_path).into());
    }

    // Check proving_key_path is a folder
    if !proving_key_path.is_dir() {
        return Err(format!("Proving key parameter must be a folder: {:?}", proving_key_path).into());
    }

    if !verify_constraints && !output_dir_path.exists() {
        fs::create_dir_all(output_dir_path).map_err(|err| format!("Failed to create output directory: {:?}", err))?;
    }

    Ok(())
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
            "Error: Unable to find the constant tree at '{}'.\n\
            Please run the following command:\n\
            \x1b[1mcargo run --bin proofman-cli check-setup --proving-key <PROVING_KEY>{}\x1b[0m",
            const_pols_tree_path, flags
        );
        return Err(error_message.into());
    }

    let error_message = format!(
        "Error: The constant tree file at '{}' exists but is invalid or corrupted.\n\
        Please regenerate it by running:\n\
        \x1b[1mcargo run --bin proofman-cli check-setup --proving-key <PROVING_KEY>{}\x1b[0m",
        const_pols_tree_path, flags
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
        let _ = file.read_to_string(&mut contents).map_err(|err| format!("Failed to read verkey path file: {}", err));
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
    if !pctx.options.verify_constraints {
        for (airgroup_id, air_group) in pctx.global_info.airs.iter().enumerate() {
            for (air_id, _) in air_group.iter().enumerate() {
                let setup = sctx.get_setup(airgroup_id, air_id);
                check_const_tree(setup, pctx.options.aggregation, pctx.options.final_snark)?;
            }
        }
    }
    Ok(())
}

pub fn check_tree_paths_vadcop<F: PrimeField64>(
    pctx: &ProofCtx<F>,
    setups: &SetupsVadcop<F>,
) -> Result<(), Box<dyn Error>> {
    if pctx.options.aggregation {
        return Ok(());
    }
    let sctx_compressor = setups.sctx_compressor.as_ref().unwrap();
    for (airgroup_id, air_group) in pctx.global_info.airs.iter().enumerate() {
        for (air_id, _) in air_group.iter().enumerate() {
            if pctx.global_info.get_air_has_compressor(airgroup_id, air_id) {
                let setup = sctx_compressor.get_setup(airgroup_id, air_id);
                check_const_tree(setup, pctx.options.aggregation, pctx.options.final_snark)?;
            }
        }
    }

    let sctx_recursive1 = setups.sctx_recursive1.as_ref().unwrap();
    for (airgroup_id, air_group) in pctx.global_info.airs.iter().enumerate() {
        for (air_id, _) in air_group.iter().enumerate() {
            let setup = sctx_recursive1.get_setup(airgroup_id, air_id);
            check_const_tree(setup, pctx.options.aggregation, pctx.options.final_snark)?;
        }
    }

    let sctx_recursive2 = setups.sctx_recursive2.as_ref().unwrap();
    let n_airgroups = pctx.global_info.air_groups.len();
    for airgroup in 0..n_airgroups {
        let setup = sctx_recursive2.get_setup(airgroup, 0);
        check_const_tree(setup, pctx.options.aggregation, pctx.options.final_snark)?;
    }

    let setup_vadcop_final = setups.setup_vadcop_final.as_ref().unwrap();
    check_const_tree(setup_vadcop_final, pctx.options.aggregation, pctx.options.final_snark)?;

    if pctx.options.final_snark {
        let setup_recursivef = setups.setup_recursivef.as_ref().unwrap();
        check_const_tree(setup_recursivef, pctx.options.aggregation, pctx.options.final_snark)?;
    }

    Ok(())
}

pub fn initialize_fixed_pols_tree<F: PrimeField64>(
    pctx: &ProofCtx<F>,
    sctx: &SetupCtx<F>,
    setups: &SetupsVadcop<F>,
    d_buffers: Arc<DeviceBuffer>,
) {
    let gpu = cfg!(feature = "gpu");
    if gpu && pctx.options.preallocate {
        let mut offset = 0;
        for (airgroup_id, air_group) in pctx.global_info.airs.iter().enumerate() {
            for (air_id, _) in air_group.iter().enumerate() {
                println!("Loading const pols in GPU for airgroup {} and air {}", airgroup_id, air_id);
                let setup = sctx.get_setup(airgroup_id, air_id);
                let const_pols_path = setup.setup_path.to_string_lossy().to_string() + ".const";
                let const_pols_tree_path = setup.setup_path.display().to_string() + ".consttree";
                let proof_type: &str = setup.setup_type.clone().into();
                println!("PROOF TYPE: {}", proof_type);
                load_const_pols_gpu_c(
                    airgroup_id as u64,
                    air_id as u64,
                    offset,
                    d_buffers.get_ptr(),
                    &const_pols_path,
                    setup.const_pols_size as u64,
                    &const_pols_tree_path,
                    setup.const_tree_size as u64,
                    proof_type,
                );
                offset += setup.const_pols_size as u64;
                offset += setup.const_tree_size as u64;
            }
        }
    }

    if pctx.options.aggregation {
        for (airgroup_id, air_group) in pctx.global_info.airs.iter().enumerate() {
            for (air_id, _) in air_group.iter().enumerate() {
                if pctx.global_info.get_air_has_compressor(airgroup_id, air_id) {
                    let setup = setups.sctx_compressor.as_ref().unwrap().get_setup(airgroup_id, air_id);
                    setup.load_const_pols();
                    setup.load_const_pols_tree();
                }
            }
        }

        for (airgroup_id, air_group) in pctx.global_info.airs.iter().enumerate() {
            for (air_id, _) in air_group.iter().enumerate() {
                let setup = setups.sctx_recursive1.as_ref().unwrap().get_setup(airgroup_id, air_id);
                setup.load_const_pols();
                setup.load_const_pols_tree();
            }
        }

        let n_airgroups = pctx.global_info.air_groups.len();
        for airgroup in 0..n_airgroups {
            let setup = setups.sctx_recursive2.as_ref().unwrap().get_setup(airgroup, 0);
            setup.load_const_pols();
            setup.load_const_pols_tree();
        }

        let setup_vadcop_final = setups.setup_vadcop_final.as_ref().unwrap();
        setup_vadcop_final.load_const_pols();
        setup_vadcop_final.load_const_pols_tree();

        if pctx.options.final_snark {
            let setup_recursivef = setups.setup_recursivef.as_ref().unwrap();
            setup_recursivef.load_const_pols();
            setup_recursivef.load_const_pols_tree();
        }
    }
}

pub fn initialize_size_witness<F: PrimeField64>(
    pctx: &ProofCtx<F>,
    setups: &SetupsVadcop<F>,
) -> Result<(), Box<dyn std::error::Error>> {
    let instances = pctx.dctx_get_instances();
    let my_instances = pctx.dctx_get_my_instances();

    let mut airs = Vec::new();
    let mut seen = HashSet::new();

    for instance_id in my_instances.iter() {
        let (airgroup_id, air_id, _) = instances[*instance_id];
        if seen.insert((airgroup_id, air_id)) {
            airs.push((airgroup_id, air_id));
        }
    }

    for &(airgroup_id, air_id) in airs.iter() {
        if pctx.global_info.get_air_has_compressor(airgroup_id, air_id) {
            let setup = setups.sctx_compressor.as_ref().unwrap().get_setup(airgroup_id, air_id);
            setup.set_size_witness()?;
        }
        let setup = setups.sctx_recursive1.as_ref().unwrap().get_setup(airgroup_id, air_id);
        setup.set_size_witness()?;
    }

    let n_airgroups = pctx.global_info.air_groups.len();
    for airgroup in 0..n_airgroups {
        let setup = setups.sctx_recursive2.as_ref().unwrap().get_setup(airgroup, 0);
        setup.set_size_witness()?;
    }

    let setup_vadcop_final = setups.setup_vadcop_final.as_ref().unwrap();
    setup_vadcop_final.set_size_witness()?;

    if pctx.options.final_snark {
        let setup_recursivef = setups.setup_recursivef.as_ref().unwrap();
        setup_recursivef.set_size_witness()?;
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
