use log::info;
use p3_field::PrimeField;
use std::fs;

use std::{collections::HashMap, path::PathBuf};

use colored::*;

use std::error::Error;

use proofman_common::{format_bytes, ProofCtx, SetupsVadcop};

pub fn print_summary_info<F: PrimeField>(name: &str, pctx: &ProofCtx<F>, setups: &SetupsVadcop<F>) {
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
        print_summary(name, pctx, setups, true);
    }

    if n_processes > 1 {
        print_summary(name, pctx, setups, false);
    }
}

pub fn print_summary<F: PrimeField>(name: &str, pctx: &ProofCtx<F>, setups: &SetupsVadcop<F>, global: bool) {
    let mut air_info = HashMap::new();

    let mut air_instances = HashMap::new();

    let instances = pctx.dctx_get_instances();

    let mut print = vec![global; instances.len()];

    if !global {
        let my_instances = pctx.dctx_get_my_instances();
        for instance_id in my_instances.iter() {
            print[*instance_id] = true;
        }
    }

    for (instance_id, (airgroup_id, air_id, _)) in instances.iter().enumerate() {
        if !print[instance_id] {
            continue;
        }
        let air_name = pctx.global_info.airs[*airgroup_id][*air_id].clone().name;
        let air_group_name = pctx.global_info.air_groups[*airgroup_id].clone();
        let air_instance_map = air_instances.entry(air_group_name).or_insert_with(HashMap::new);
        if !air_instance_map.contains_key(&air_name.clone()) {
            let setup = setups.sctx.get_setup(*airgroup_id, *air_id);
            let n_bits = setup.stark_info.stark_struct.n_bits;
            let memory_instance = setup.prover_buffer_size as f64 * 8.0;
            let mut memory_fixed =
                (setup.stark_info.n_constants * (1 << (setup.stark_info.stark_struct.n_bits))) as f64;
            if !pctx.options.verify_constraints {
                memory_fixed += (setup.stark_info.n_constants * (1 << (setup.stark_info.stark_struct.n_bits_ext))
                    + (1 << (setup.stark_info.stark_struct.n_bits_ext))
                    + ((2 * (1 << (setup.stark_info.stark_struct.n_bits_ext)) - 1) * 4))
                    as f64;
            }
            memory_fixed *= 8.0;
            let mut memory_fixed_aggregation = 0f64;
            if pctx.options.aggregation {
                if pctx.global_info.get_air_has_compressor(*airgroup_id, *air_id) {
                    let setup_compressor = setups.sctx_compressor.as_ref().unwrap().get_setup(*airgroup_id, *air_id);
                    memory_fixed_aggregation += (setup_compressor.stark_info.n_constants
                        * (1 << (setup_compressor.stark_info.stark_struct.n_bits))
                        + setup_compressor.stark_info.n_constants
                            * (1 << (setup_compressor.stark_info.stark_struct.n_bits_ext))
                        + (1 << (setup_compressor.stark_info.stark_struct.n_bits_ext))
                        + ((2 * (1 << (setup_compressor.stark_info.stark_struct.n_bits_ext)) - 1) * 4))
                        as f64
                        * 8.0;
                }

                let setup_recursive1 = setups.sctx_recursive1.as_ref().unwrap().get_setup(*airgroup_id, *air_id);
                memory_fixed_aggregation += (setup_recursive1.stark_info.n_constants
                    * (1 << (setup_recursive1.stark_info.stark_struct.n_bits))
                    + setup_recursive1.stark_info.n_constants
                        * (1 << (setup_recursive1.stark_info.stark_struct.n_bits_ext))
                    + (1 << (setup_recursive1.stark_info.stark_struct.n_bits_ext))
                    + ((2 * (1 << (setup_recursive1.stark_info.stark_struct.n_bits_ext)) - 1) * 4))
                    as f64
                    * 8.0;
            }

            let memory_helpers = setup.stark_info.get_buff_helper_size() as f64 * 8.0;
            let total_cols: u64 = setup
                .stark_info
                .map_sections_n
                .iter()
                .filter(|(key, _)| *key != "const")
                .map(|(_, value)| *value)
                .sum();
            air_info.insert(
                air_name.clone(),
                (n_bits, total_cols, memory_fixed, memory_fixed_aggregation, memory_helpers, memory_instance),
            );
        }
        let air_instance_map_key = air_instance_map.entry(air_name).or_insert(0);
        *air_instance_map_key += 1;
    }

    let mut air_groups: Vec<_> = air_instances.keys().collect();
    air_groups.sort();

    info!("{}", format!("{}: --- TOTAL PROOF INSTANCES SUMMARY ------------------------", name).bright_white().bold());
    info!("{}:     ► {} Air instances found:", name, instances.len());
    for air_group in air_groups.clone() {
        let air_group_instances = air_instances.get(air_group).unwrap();
        let mut air_names: Vec<_> = air_group_instances.keys().collect();
        air_names.sort();

        info!("{}:       Air Group [{}]", name, air_group);
        for air_name in air_names {
            let count = air_group_instances.get(air_name).unwrap();
            let (n_bits, total_cols, _, _, _, _) = air_info.get(air_name).unwrap();
            info!(
                "{}:       {}",
                name,
                format!("· {} x Air [{}] ({} x 2^{})", count, air_name, total_cols, n_bits).bright_white().bold()
            );
        }
    }
    info!("{}: ----------------------------------------------------------", name);
    info!("{}", format!("{}: --- TOTAL SETUP MEMORY USAGE ----------------------------", name).bright_white().bold());
    let mut total_memory = 0f64;
    for air_group in air_groups.clone() {
        let air_group_instances = air_instances.get(air_group).unwrap();
        let mut air_names: Vec<_> = air_group_instances.keys().collect();
        air_names.sort();

        for air_name in air_names {
            let (_, _, memory_fixed, memory_fixed_aggregation, _, _) = air_info.get(air_name).unwrap();
            total_memory += memory_fixed;

            if !pctx.options.aggregation {
                info!("{}:       {}", name, format!("· {}: {} fixed cols", air_name, format_bytes(*memory_fixed),));
            } else {
                total_memory += memory_fixed_aggregation;
                info!(
                    "{}:       {}",
                    name,
                    format!(
                        "· {}: {} fixed cols | {} fixed cols aggregation | Total: {}",
                        air_name,
                        format_bytes(*memory_fixed),
                        format_bytes(*memory_fixed_aggregation),
                        format_bytes(*memory_fixed + *memory_fixed_aggregation),
                    )
                );
            }
        }
        info!(
            "{}:       {}",
            name,
            format!("Total setup memory required: {}", format_bytes(total_memory)).bright_white().bold()
        );
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
    let mut total_memory = 0f64;
    let mut memory_helper_size = 0f64;
    for air_group in air_groups {
        let air_group_instances = air_instances.get(air_group).unwrap();
        let mut air_names: Vec<_> = air_group_instances.keys().collect();
        air_names.sort();

        for air_name in air_names {
            let count = air_group_instances.get(air_name).unwrap();
            let (_, _, _, _, memory_helper_instance_size, memory_instance) = air_info.get(air_name).unwrap();
            let total_memory_instance = memory_instance * *count as f64;
            total_memory += total_memory_instance;
            if *memory_helper_instance_size > memory_helper_size {
                memory_helper_size = *memory_helper_instance_size;
            }
            info!(
                "{}:       {}",
                name,
                format!(
                    "· {}: {} per each of {} instance | Total {}",
                    air_name,
                    format_bytes(*memory_instance),
                    count,
                    format_bytes(total_memory_instance)
                )
            );
        }
    }
    total_memory += memory_helper_size;
    info!("{}:       {}", name, format!("Extra helper memory: {}", format_bytes(memory_helper_size)));
    info!(
        "{}:       {}",
        name,
        format!("Total prover memory required: {}", format_bytes(total_memory)).bright_white().bold()
    );
    info!("{}: ----------------------------------------------------------", name);
}

pub fn check_paths(
    witness_lib_path: &PathBuf,
    rom_path: &Option<PathBuf>,
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

    // Check rom_path path exists
    if let Some(rom_path) = rom_path {
        if !rom_path.exists() {
            return Err(format!("ROM file not found at path: {:?}", rom_path).into());
        }
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
