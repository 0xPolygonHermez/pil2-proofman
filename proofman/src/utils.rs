use fields::PrimeField64;
use num_traits::ToPrimitive;
use std::fs::{self, File};
use std::io::{Read, Seek, SeekFrom};
use std::{
    collections::HashMap,
    path::{Path, PathBuf},
};
use std::os::raw::c_void;

use colored::*;

use proofman_common::{
    format_bytes, MpiCtx, ParamsGPU, ProofCtx, ProofType, ProofmanError, ProofmanResult, Setup, SetupCtx, SetupsVadcop,
};
use proofman_starks_lib_c::load_device_const_pols_c;
use proofman_starks_lib_c::load_device_setup_c;
use proofman_starks_lib_c::get_unified_buffer_gpu_c;
use proofman_starks_lib_c::verify_root_bn128_from_tree_c;
use proofman_starks_lib_c::pack_const_pols_c;
use proofman_starks_lib_c::{
    calculate_const_tree_c, calculate_const_tree_bn128_c, write_const_tree_c, write_const_tree_bn128_c,
    prepare_blocks_c, tile_const_pols_c, load_const_pols_c,
};
use proofman_util::create_buffer_fast;
use proofman_common::{PackedInfo, VerboseMode, GlobalInfo};

use pil_std_lib::Std;
use witness::WitnessManager;

pub fn print_summary_info<F: PrimeField64>(
    pctx: &ProofCtx<F>,
    sctx: &SetupCtx<F>,
    mpi_ctx: &MpiCtx,
    packed_info: &HashMap<(usize, usize), PackedInfo>,
    verbose_mode: VerboseMode,
) -> ProofmanResult<String> {
    let summary = print_summary(pctx, sctx, packed_info, true, verbose_mode, mpi_ctx.rank == 0)?;

    if mpi_ctx.n_processes > 1 {
        let (average_weight, max_weight, min_weight, max_deviation) = pctx.dctx_load_balance_info_process();
        tracing::info!(
            "Load balance. Average: {} max: {} min: {} deviation: {}",
            average_weight,
            max_weight,
            min_weight,
            max_deviation
        );

        let _ = print_summary(pctx, sctx, packed_info, false, verbose_mode, true)?;
    }
    Ok(summary)
}

pub fn print_summary<F: PrimeField64>(
    pctx: &ProofCtx<F>,
    sctx: &SetupCtx<F>,
    packed_info: &HashMap<(usize, usize), PackedInfo>,
    global: bool,
    verbose_mode: VerboseMode,
    print_output: bool,
) -> ProofmanResult<String> {
    let mut summary_info = String::new();

    let mut air_info = HashMap::new();

    let mut air_instances = HashMap::new();

    let instances = pctx.dctx_get_instances();
    let mut n_instances = instances.len();

    let mut print = vec![global; instances.len()];

    if !global {
        let my_instances = pctx.dctx_get_process_instances();
        for instance_id in my_instances.iter() {
            print[*instance_id] = true;
        }
        n_instances = my_instances.len();
    }

    let max_prover_memory = sctx.max_prover_buffer_size as f64 * 8.0;

    let mut memory_tables = 0 as f64;
    for (instance_id, &instance_info) in instances.iter().enumerate() {
        let (airgroup_id, air_id, is_table) = (instance_info.airgroup_id, instance_info.air_id, instance_info.table);
        if !print[instance_id] {
            continue;
        }
        let air_name = pctx.global_info.airs[airgroup_id][air_id].clone().name;
        let air_group_name = pctx.global_info.air_groups[airgroup_id].clone();
        let air_instance_map = air_instances.entry(air_group_name).or_insert_with(HashMap::new);
        if !air_instance_map.contains_key(&air_name.clone()) {
            let setup = sctx.get_setup(airgroup_id, air_id)?;
            let n_bits = setup.stark_info.stark_struct.n_bits;
            let memory_trace = if cfg!(feature = "gpu") && cfg!(feature = "packed") {
                let num_packed_words = packed_info.get(&(airgroup_id, air_id)).map(|info| info.num_packed_words);
                if let Some(num_packed_words) = num_packed_words {
                    (num_packed_words * (1 << setup.stark_info.stark_struct.n_bits)) as f64 * 8.0
                } else {
                    (setup.stark_info.map_sections_n["cm1"] * (1 << setup.stark_info.stark_struct.n_bits)) as f64 * 8.0
                }
            } else {
                (setup.stark_info.map_sections_n["cm1"] * (1 << (setup.stark_info.stark_struct.n_bits))) as f64 * 8.0
            };
            let memory_instance = setup.prover_buffer_size as f64 * 8.0;
            let memory_fixed =
                (setup.stark_info.n_constants * (1 << (setup.stark_info.stark_struct.n_bits))) as f64 * 8.0;
            if is_table {
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

    if verbose_mode != VerboseMode::Info {
        if print_output {
            tracing::info!("{}", "--- TOTAL PROOF INSTANCES SUMMARY ------------------------".bright_white().bold());
            tracing::info!("    ► {} Air instances found:", n_instances);
        }
        for air_group in &air_groups {
            let air_group_instances = air_instances.get(*air_group).unwrap();
            let mut air_names: Vec<_> = air_group_instances.keys().collect();
            air_names.sort();

            if print_output {
                tracing::info!("      Air Group [{}]", air_group);
            }
            for air_name in air_names {
                let count = air_group_instances.get(air_name).unwrap();
                let (n_bits, total_cols, _, _, _) = air_info.get(air_name).unwrap();
                if print_output {
                    tracing::info!(
                        "      {}",
                        format!("· {count} x Air [{air_name}] ({total_cols} x 2^{n_bits})").bright_white().bold()
                    );
                }
            }
        }
        if print_output {
            tracing::info!("{}", "--- TOTAL PROVER MEMORY USAGE ----------------------------".bright_white().bold());
        }
        for air_group in &air_groups {
            let air_group_instances = air_instances.get(*air_group).unwrap();
            let mut air_names: Vec<_> = air_group_instances.keys().collect();
            air_names.sort();

            for air_name in air_names {
                let count = air_group_instances.get(air_name).unwrap();
                let (_, _, _, memory_trace, memory_instance) = air_info.get(air_name).unwrap();
                let gpu = cfg!(feature = "gpu");
                if print_output {
                    if gpu {
                        tracing::info!(
                            "      · {}: {} GPU per each of {} instance | Witness CPU: {}",
                            air_name,
                            format_bytes(*memory_instance),
                            count,
                            format_bytes(*memory_trace),
                        );
                    } else {
                        tracing::info!(
                            "      · {}: {} per each of {} instance | Witness : {}",
                            air_name,
                            format_bytes(*memory_instance),
                            count,
                            format_bytes(*memory_trace),
                        );
                    }
                }
            }
        }
        if print_output {
            tracing::info!("      Total memory required by proofman: {}", format_bytes(max_prover_memory));
            tracing::info!("----------------------------------------------------------");
            tracing::info!("      Extra memory tables (CPU): {}", format_bytes(memory_tables));
            tracing::info!("----------------------------------------------------------");
        }
    } else {
        if print_output {
            tracing::info!("{}", "--- PROOF INSTANCES SUMMARY ---".bright_white().bold());
        }

        for air_group in &air_groups {
            let air_group_instances = air_instances.get(*air_group).unwrap();
            let mut air_names: Vec<_> = air_group_instances.keys().collect();
            air_names.sort();

            let mut summary: Vec<String> = air_names
                .iter()
                .map(|air_name| {
                    let count = air_group_instances.get(*air_name).unwrap();
                    format!("{air_name}: {count}")
                })
                .collect();

            summary.push(format!("Total {} instances: {}", if global { "global" } else { "local" }, n_instances));

            if print_output {
                tracing::info!("{} | {}", air_group.bright_white().bold(), summary.join(" | "));
            }

            if global {
                summary_info = summary.join(" | ");
            }
        }

        if print_output {
            tracing::info!("{}", "--------------------------------".bright_white().bold());
        }
    }

    Ok(summary_info)
}

pub fn needs_const_tree_regeneration<F: PrimeField64>(setup: &Setup<F>) -> ProofmanResult<bool> {
    let const_pols_tree_path = &setup.const_pols_tree_path;
    let const_pols_tree_size = setup.const_tree_size;

    // Check if file exists
    if !PathBuf::from(&const_pols_tree_path).exists() {
        return Ok(true);
    }

    // Check file size
    match fs::metadata(const_pols_tree_path) {
        Ok(metadata) => {
            let actual_size = metadata.len() as usize;
            if actual_size != const_pols_tree_size * 8 {
                return Ok(true);
            }
        }
        Err(_) => return Ok(true),
    }

    // Validate the tree content
    let mut file = File::open(const_pols_tree_path)?;
    file.seek(SeekFrom::End(-32))?;

    let mut buffer = [0u8; 32];
    file.read_exact(&mut buffer)?;

    if setup.setup_type != ProofType::RecursiveF {
        let verkey_path = setup.verkey_file.clone();
        let mut contents = String::new();
        let mut file = File::open(verkey_path).unwrap();
        let _ = file.read_to_string(&mut contents).map_err(|err| format!("Failed to read verkey path file: {err}"));
        let verkey_u64: Vec<u64> = serde_json::from_str(&contents).unwrap();

        for (i, verkey_val) in verkey_u64.iter().enumerate() {
            let byte_range = i * 8..(i + 1) * 8;
            let value = u64::from_le_bytes(buffer[byte_range].try_into()?);
            if value != *verkey_val {
                return Ok(true);
            }
        }
    } else {
        let verkey_path = setup.verkey_file.clone();
        let mut contents = String::new();
        let mut file = File::open(verkey_path).unwrap();
        let _ = file.read_to_string(&mut contents).map_err(|err| format!("Failed to read verkey path file: {err}"));

        let verkey_str: String = serde_json::from_str(&contents)
            .map_err(|err| ProofmanError::InvalidSetup(format!("Failed to parse verkey as string: {}", err)))?;

        let is_valid = verify_root_bn128_from_tree_c(const_pols_tree_path, &verkey_str);
        if !is_valid {
            return Ok(true);
        }
    }

    Ok(false)
}

pub fn check_const_tree<F: PrimeField64>(setup: &Setup<F>, d_buffers: &Option<*mut c_void>) -> ProofmanResult<()> {
    let const_pols_tree_path = &setup.const_pols_tree_path;
    let const_pols_tree_size = setup.const_tree_size;

    let mut needs_regeneration = false;
    let mut validation_failed = false;

    // Check if file exists and has correct size
    if PathBuf::from(&const_pols_tree_path).exists() {
        match fs::metadata(const_pols_tree_path) {
            Ok(metadata) => {
                let actual_size = metadata.len() as usize;
                if actual_size != const_pols_tree_size * 8 {
                    tracing::trace!(
                        "Constant tree file '{}' has incorrect size ({} bytes, expected {} bytes). Regenerating...",
                        const_pols_tree_path,
                        actual_size,
                        const_pols_tree_size * 8
                    );
                    needs_regeneration = true;
                } else {
                    // Validate the tree content
                    let mut file = File::open(const_pols_tree_path)?;
                    file.seek(SeekFrom::End(-32))?;

                    let mut buffer = [0u8; 32];
                    file.read_exact(&mut buffer)?;

                    if setup.setup_type != ProofType::RecursiveF {
                        let verkey_path = setup.verkey_file.clone();
                        let mut contents = String::new();
                        let mut file = File::open(verkey_path).unwrap();
                        let _ = file
                            .read_to_string(&mut contents)
                            .map_err(|err| format!("Failed to read verkey path file: {err}"));
                        let verkey_u64: Vec<u64> = serde_json::from_str(&contents).unwrap();

                        for (i, verkey_val) in verkey_u64.iter().enumerate() {
                            let byte_range = i * 8..(i + 1) * 8;
                            let value = u64::from_le_bytes(buffer[byte_range].try_into()?);
                            if value != *verkey_val {
                                validation_failed = true;
                                break;
                            }
                        }
                    } else {
                        let verkey_path = setup.verkey_file.clone();
                        let mut contents = String::new();
                        let mut file = File::open(verkey_path).unwrap();
                        let _ = file
                            .read_to_string(&mut contents)
                            .map_err(|err| format!("Failed to read verkey path file: {err}"));

                        let verkey_str: String = serde_json::from_str(&contents).map_err(|err| {
                            ProofmanError::InvalidSetup(format!("Failed to parse verkey as string: {}", err))
                        })?;

                        let is_valid = verify_root_bn128_from_tree_c(const_pols_tree_path, &verkey_str);
                        if !is_valid {
                            validation_failed = true;
                        }
                    }

                    if validation_failed {
                        tracing::trace!(
                            "Constant tree file '{}' validation failed. Regenerating...",
                            const_pols_tree_path
                        );
                        needs_regeneration = true;
                    }
                }
            }
            Err(err) => {
                return Err(ProofmanError::InvalidSetup(format!(
                    "Failed to get metadata for {}: {}",
                    setup.air_name, err
                )));
            }
        }
    } else {
        tracing::trace!("Constant tree file '{}' does not exist. Generating...", const_pols_tree_path);
        needs_regeneration = true;
    }

    // Regenerate the const tree if needed
    if needs_regeneration {
        let const_pols_size = (setup.stark_info.n_constants * (1 << setup.stark_info.stark_struct.n_bits)) as usize;
        let mut const_pols: Vec<F> = create_buffer_fast(const_pols_size);
        let const_pols_path = setup.setup_path.display().to_string() + ".const";
        load_const_pols_c(const_pols.as_ptr() as *mut u8, const_pols_path.as_str(), const_pols.len() as u64 * 8);

        let const_tree: Vec<F> = create_buffer_fast(const_pols_tree_size);
        let p_stark_info = setup.p_setup.p_stark_info;

        let unified_buffer_gpu =
            if let Some(d_buffers) = d_buffers { get_unified_buffer_gpu_c(*d_buffers) } else { std::ptr::null_mut() };

        if setup.stark_info.stark_struct.verification_hash_type == "GL" {
            if cfg!(feature = "gpu") {
                prepare_blocks_c(
                    const_pols.as_mut_ptr() as *mut u64,
                    1 << setup.stark_info.stark_struct.n_bits,
                    setup.stark_info.n_constants,
                    unified_buffer_gpu,
                );
                calculate_const_tree_c(
                    p_stark_info,
                    const_pols.as_ptr() as *mut u8,
                    const_tree.as_ptr() as *mut u8,
                    unified_buffer_gpu,
                );
                write_const_tree_c(p_stark_info, const_tree.as_ptr() as *mut u8, const_pols_tree_path.as_str());
            } else {
                calculate_const_tree_c(
                    p_stark_info,
                    const_pols.as_ptr() as *mut u8,
                    const_tree.as_ptr() as *mut u8,
                    unified_buffer_gpu,
                );
                write_const_tree_c(p_stark_info, const_tree.as_ptr() as *mut u8, const_pols_tree_path.as_str());
            }
        } else {
            // BN128 case (RecursiveF)
            calculate_const_tree_bn128_c(p_stark_info, const_pols.as_ptr() as *mut u8, const_tree.as_ptr() as *mut u8);

            // For RecursiveF, we need to write to CPU path first
            let const_pols_tree_path_cpu = setup.setup_path.display().to_string() + ".consttree";
            write_const_tree_bn128_c(p_stark_info, const_tree.as_ptr() as *mut u8, const_pols_tree_path_cpu.as_str());

            // For GPU, use tile_const_pols_c to create both GPU const pols and GPU const tree
            if cfg!(feature = "gpu") {
                tile_const_pols_c(
                    p_stark_info,
                    const_pols.as_ptr() as *mut u8,
                    setup.const_pols_path.as_str(),
                    const_tree.as_ptr() as *mut u8,
                    const_pols_tree_path.as_str(),
                    unified_buffer_gpu,
                );
            }
        }

        tracing::trace!("Successfully generated constant tree file '{}'", const_pols_tree_path);
    }

    Ok(())
}

pub fn needs_const_pols_gpu_regeneration<F: PrimeField64>(setup: &Setup<F>) -> ProofmanResult<bool> {
    if !cfg!(feature = "gpu") {
        return Ok(false);
    }

    let n_constants = setup.stark_info.n_constants as usize;
    let n_rows = 1usize << setup.stark_info.stark_struct.n_bits as usize;

    // Check if file exists
    if !PathBuf::from(&setup.const_pols_path).exists() {
        return Ok(true);
    }

    // Check file size
    let mut file = File::open(&setup.const_pols_path)?;
    let mut words_per_row_bytes = [0u8; 8];
    file.read_exact(&mut words_per_row_bytes)?;
    let words_per_row = u64::from_le_bytes(words_per_row_bytes);

    let expected_size = 8 + (n_constants * 8) + (n_rows * words_per_row as usize * 8);

    match fs::metadata(&setup.const_pols_path) {
        Ok(metadata) => {
            let actual_size = metadata.len() as usize;
            if actual_size != expected_size {
                return Ok(true);
            }
        }
        Err(_) => return Ok(true),
    }

    Ok(false)
}

fn check_const_pols_gpu<F: PrimeField64>(setup: &Setup<F>) -> ProofmanResult<()> {
    if !cfg!(feature = "gpu") {
        return Ok(());
    }

    let n_constants = setup.stark_info.n_constants as usize;
    let n_rows = 1usize << setup.stark_info.stark_struct.n_bits as usize;

    let mut needs_regeneration = false;
    let expected_size;

    // Check if file exists and has correct size
    if PathBuf::from(&setup.const_pols_path).exists() {
        let mut file = File::open(&setup.const_pols_path)?;
        let mut words_per_row_bytes = [0u8; 8];
        file.read_exact(&mut words_per_row_bytes)?;
        let words_per_row = u64::from_le_bytes(words_per_row_bytes);

        // Calculate expected size
        expected_size = 8 + (n_constants * 8) + (n_rows * words_per_row as usize * 8);

        match fs::metadata(&setup.const_pols_path) {
            Ok(metadata) => {
                let actual_size = metadata.len() as usize;
                if actual_size != expected_size {
                    tracing::trace!(
                        "GPU constant polynomials file '{}' has incorrect size ({} bytes, expected {} bytes). Regenerating...",
                        setup.const_pols_path, actual_size, expected_size
                    );
                    needs_regeneration = true;
                }
            }
            Err(err) => {
                return Err(ProofmanError::InvalidSetup(format!(
                    "Failed to get metadata for GPU const pols {}: {}",
                    setup.air_name, err
                )));
            }
        }
    } else {
        tracing::trace!("GPU constant polynomials file '{}' does not exist. Generating...", setup.const_pols_path);
        needs_regeneration = true;
    }

    if needs_regeneration {
        let const_pols_size = (setup.stark_info.n_constants * (1 << setup.stark_info.stark_struct.n_bits)) as usize;
        let const_pols: Vec<F> = create_buffer_fast(const_pols_size);
        let const_pols_path = setup.setup_path.display().to_string() + ".const";

        load_const_pols_c(const_pols.as_ptr() as *mut u8, const_pols_path.as_str(), const_pols.len() as u64 * 8);

        pack_const_pols_c(setup.p_setup.p_stark_info, const_pols.as_ptr() as *mut u8, setup.const_pols_path.as_str());

        tracing::trace!("Successfully generated GPU constant polynomials file '{}'", setup.const_pols_path);
    }

    Ok(())
}

pub fn needs_regeneration_fixed<F: PrimeField64>(
    pctx: &ProofCtx<F>,
    sctx: &SetupCtx<F>,
) -> ProofmanResult<(bool, bool)> {
    let mut needs_const_regen = false;
    let mut needs_tree_regen = false;

    for (airgroup_id, air_group) in pctx.global_info.airs.iter().enumerate() {
        for (air_id, _) in air_group.iter().enumerate() {
            let setup = sctx.get_setup(airgroup_id, air_id)?;
            if needs_const_pols_gpu_regeneration(setup)? {
                needs_const_regen = true;
                tracing::debug!("GPU const pols regeneration needed for [{}:{}]", airgroup_id, air_id);
            }
            if needs_const_tree_regeneration(setup)? {
                needs_tree_regen = true;
                tracing::debug!("Const tree regeneration needed for [{}:{}]", airgroup_id, air_id);
            }
        }
    }

    Ok((needs_const_regen, needs_tree_regen))
}

pub fn check_const_paths<F: PrimeField64>(pctx: &ProofCtx<F>, sctx: &SetupCtx<F>) -> ProofmanResult<()> {
    for (airgroup_id, air_group) in pctx.global_info.airs.iter().enumerate() {
        for (air_id, _) in air_group.iter().enumerate() {
            let setup = sctx.get_setup(airgroup_id, air_id)?;
            check_const_pols_gpu(setup)?;
        }
    }
    Ok(())
}

pub fn needs_regeneration_vadcop_fixed<F: PrimeField64>(
    pctx: &ProofCtx<F>,
    setups: &SetupsVadcop<F>,
) -> ProofmanResult<(bool, bool)> {
    let mut needs_const_regen = false;
    let mut needs_tree_regen = false;

    let sctx_compressor = setups.sctx_compressor.as_ref().unwrap();
    for (airgroup_id, air_group) in pctx.global_info.airs.iter().enumerate() {
        for (air_id, _) in air_group.iter().enumerate() {
            if pctx.global_info.get_air_has_compressor(airgroup_id, air_id) {
                let setup = sctx_compressor.get_setup(airgroup_id, air_id)?;
                if needs_const_pols_gpu_regeneration(setup)? {
                    needs_const_regen = true;
                    tracing::debug!(
                        "Vadcop compressor const pols regeneration needed for [{}:{}]",
                        airgroup_id,
                        air_id
                    );
                }
                if needs_const_tree_regeneration(setup)? {
                    needs_tree_regen = true;
                    tracing::debug!("Vadcop compressor tree regeneration needed for [{}:{}]", airgroup_id, air_id);
                }
            }
        }
    }

    let sctx_recursive1 = setups.sctx_recursive1.as_ref().unwrap();
    for (airgroup_id, air_group) in pctx.global_info.airs.iter().enumerate() {
        for (air_id, _) in air_group.iter().enumerate() {
            let setup = sctx_recursive1.get_setup(airgroup_id, air_id)?;
            if needs_const_pols_gpu_regeneration(setup)? {
                needs_const_regen = true;
                tracing::debug!("Vadcop recursive1 const pols regeneration needed for [{}:{}]", airgroup_id, air_id);
            }
            if needs_const_tree_regeneration(setup)? {
                needs_tree_regen = true;
                tracing::debug!("Vadcop recursive1 tree regeneration needed for [{}:{}]", airgroup_id, air_id);
            }
        }
    }

    let sctx_recursive2 = setups.sctx_recursive2.as_ref().unwrap();
    let n_airgroups = pctx.global_info.air_groups.len();
    for airgroup in 0..n_airgroups {
        let setup = sctx_recursive2.get_setup(airgroup, 0)?;
        if needs_const_pols_gpu_regeneration(setup)? {
            needs_const_regen = true;
            tracing::debug!("Vadcop recursive2 const pols regeneration needed for airgroup {}", airgroup);
        }
        if needs_const_tree_regeneration(setup)? {
            needs_tree_regen = true;
            tracing::debug!("Vadcop recursive2 tree regeneration needed for airgroup {}", airgroup);
        }
    }

    let setup_vadcop_final = setups.setup_vadcop_final.as_ref().unwrap();
    if needs_const_pols_gpu_regeneration(setup_vadcop_final)? {
        needs_const_regen = true;
        tracing::debug!("Vadcop final const pols regeneration needed");
    }
    if needs_const_tree_regeneration(setup_vadcop_final)? {
        needs_tree_regen = true;
        tracing::debug!("Vadcop final tree regeneration needed");
    }

    let setup_vadcop_final_compressed = setups.setup_vadcop_final_compressed.as_ref().unwrap();
    if needs_const_pols_gpu_regeneration(setup_vadcop_final_compressed)? {
        needs_const_regen = true;
        tracing::debug!("Vadcop final compressed const pols regeneration needed");
    }
    if needs_const_tree_regeneration(setup_vadcop_final_compressed)? {
        needs_tree_regen = true;
        tracing::debug!("Vadcop final compressed tree regeneration needed");
    }

    Ok((needs_const_regen, needs_tree_regen))
}

pub fn check_const_paths_vadcop<F: PrimeField64>(pctx: &ProofCtx<F>, setups: &SetupsVadcop<F>) -> ProofmanResult<()> {
    let sctx_compressor = setups.sctx_compressor.as_ref().unwrap();
    for (airgroup_id, air_group) in pctx.global_info.airs.iter().enumerate() {
        for (air_id, _) in air_group.iter().enumerate() {
            if pctx.global_info.get_air_has_compressor(airgroup_id, air_id) {
                let setup = sctx_compressor.get_setup(airgroup_id, air_id)?;
                check_const_pols_gpu(setup)?;
            }
        }
    }

    let sctx_recursive1 = setups.sctx_recursive1.as_ref().unwrap();
    for (airgroup_id, air_group) in pctx.global_info.airs.iter().enumerate() {
        for (air_id, _) in air_group.iter().enumerate() {
            let setup = sctx_recursive1.get_setup(airgroup_id, air_id)?;
            check_const_pols_gpu(setup)?;
        }
    }

    let sctx_recursive2 = setups.sctx_recursive2.as_ref().unwrap();
    let n_airgroups = pctx.global_info.air_groups.len();
    for airgroup in 0..n_airgroups {
        let setup = sctx_recursive2.get_setup(airgroup, 0)?;
        check_const_pols_gpu(setup)?;
    }

    let setup_vadcop_final = setups.setup_vadcop_final.as_ref().unwrap();
    check_const_pols_gpu(setup_vadcop_final)?;

    let setup_vadcop_final_compressed = setups.setup_vadcop_final_compressed.as_ref().unwrap();
    check_const_pols_gpu(setup_vadcop_final_compressed)?;
    Ok(())
}

pub fn check_tree_paths<F: PrimeField64>(pctx: &ProofCtx<F>, sctx: &SetupCtx<F>) -> ProofmanResult<()> {
    for (airgroup_id, air_group) in pctx.global_info.airs.iter().enumerate() {
        for (air_id, _) in air_group.iter().enumerate() {
            let setup = sctx.get_setup(airgroup_id, air_id)?;
            let d_buffers = if cfg!(feature = "gpu") { Some(pctx.get_device_buffers_ptr()) } else { None };
            check_const_tree(setup, &d_buffers)?;
        }
    }
    Ok(())
}

pub fn check_tree_paths_vadcop<F: PrimeField64>(pctx: &ProofCtx<F>, setups: &SetupsVadcop<F>) -> ProofmanResult<()> {
    let d_buffers = if cfg!(feature = "gpu") { Some(pctx.get_device_buffers_ptr()) } else { None };
    let sctx_compressor = setups.sctx_compressor.as_ref().unwrap();
    for (airgroup_id, air_group) in pctx.global_info.airs.iter().enumerate() {
        for (air_id, _) in air_group.iter().enumerate() {
            if pctx.global_info.get_air_has_compressor(airgroup_id, air_id) {
                let setup = sctx_compressor.get_setup(airgroup_id, air_id)?;
                check_const_tree(setup, &d_buffers)?;
            }
        }
    }

    let sctx_recursive1 = setups.sctx_recursive1.as_ref().unwrap();
    for (airgroup_id, air_group) in pctx.global_info.airs.iter().enumerate() {
        for (air_id, _) in air_group.iter().enumerate() {
            let setup = sctx_recursive1.get_setup(airgroup_id, air_id)?;
            check_const_tree(setup, &d_buffers)?;
        }
    }

    let sctx_recursive2 = setups.sctx_recursive2.as_ref().unwrap();
    let n_airgroups = pctx.global_info.air_groups.len();
    for airgroup in 0..n_airgroups {
        let setup = sctx_recursive2.get_setup(airgroup, 0)?;
        check_const_tree(setup, &d_buffers)?;
    }

    let setup_vadcop_final = setups.setup_vadcop_final.as_ref().unwrap();
    check_const_tree(setup_vadcop_final, &d_buffers)?;

    let setup_vadcop_final_compressed = setups.setup_vadcop_final_compressed.as_ref().unwrap();
    check_const_tree(setup_vadcop_final_compressed, &d_buffers)?;

    Ok(())
}

pub fn calculate_max_witness_trace_size<F: PrimeField64>(
    pctx: &ProofCtx<F>,
    sctx: &SetupCtx<F>,
    packed_info: &HashMap<(usize, usize), PackedInfo>,
    gpu_params: &ParamsGPU,
) -> ProofmanResult<usize> {
    let mut max_witness_trace_size = 0;
    for (airgroup_id, air_group) in pctx.global_info.airs.iter().enumerate() {
        for (air_id, _) in air_group.iter().enumerate() {
            let setup = sctx.get_setup(airgroup_id, air_id)?;
            let n = 1 << setup.stark_info.stark_struct.n_bits;
            let num_packed_words =
                packed_info.get(&(airgroup_id, air_id)).map(|info| info.num_packed_words).unwrap_or(0);
            let is_packed =
                cfg!(feature = "gpu") && cfg!(feature = "packed") && gpu_params.pack_trace && num_packed_words > 0;
            let trace_size = if !is_packed {
                let n_cols = setup.stark_info.map_sections_n["cm1"];
                n * n_cols
            } else {
                n * num_packed_words
            };

            max_witness_trace_size = max_witness_trace_size.max(trace_size as usize);
        }
    }
    Ok(max_witness_trace_size)
}

pub fn load_device_setups<F: PrimeField64>(
    pctx: &ProofCtx<F>,
    sctx: &SetupCtx<F>,
    setups: &SetupsVadcop<F>,
    aggregation: bool,
    packed_info: &HashMap<(usize, usize), PackedInfo>,
) -> ProofmanResult<()> {
    let d_buffers = pctx.get_device_buffers_ptr();
    for (airgroup_id, air_group) in pctx.global_info.airs.iter().enumerate() {
        for (air_id, _) in air_group.iter().enumerate() {
            let setup = sctx.get_setup(airgroup_id, air_id)?;
            let proof_type: &str = setup.setup_type.clone().into();
            if cfg!(feature = "gpu") {
                tracing::debug!(airgroup_id, air_id, proof_type, "Loading expressions setup in GPU");
            }
            let packed_info_air =
                packed_info.get(&(airgroup_id, air_id)).cloned().unwrap_or_else(|| PackedInfo::new(false, 0, vec![]));
            load_device_setup_c(
                airgroup_id as u64,
                air_id as u64,
                proof_type,
                (&setup.p_setup).into(),
                d_buffers,
                setup.verkey.as_ptr() as *mut u8,
                packed_info_air.as_ffi().get_ptr(),
            );
        }
    }

    if aggregation {
        for (airgroup_id, air_group) in pctx.global_info.airs.iter().enumerate() {
            for (air_id, _) in air_group.iter().enumerate() {
                if pctx.global_info.get_air_has_compressor(airgroup_id, air_id) {
                    let setup = setups.sctx_compressor.as_ref().unwrap().get_setup(airgroup_id, air_id)?;
                    let proof_type: &str = setup.setup_type.clone().into();
                    if cfg!(feature = "gpu") {
                        tracing::debug!(airgroup_id, air_id, proof_type, "Loading expressions setup in GPU");
                    }
                    load_device_setup_c(
                        airgroup_id as u64,
                        air_id as u64,
                        proof_type,
                        (&setup.p_setup).into(),
                        d_buffers,
                        setup.verkey.as_ptr() as *mut u8,
                        std::ptr::null_mut(),
                    );
                }
            }
        }

        for (airgroup_id, air_group) in pctx.global_info.airs.iter().enumerate() {
            for (air_id, _) in air_group.iter().enumerate() {
                let setup = setups.sctx_recursive1.as_ref().unwrap().get_setup(airgroup_id, air_id)?;
                let proof_type: &str = setup.setup_type.clone().into();
                if cfg!(feature = "gpu") {
                    tracing::debug!(airgroup_id, air_id, proof_type, "Loading expressions setup in GPU");
                }
                load_device_setup_c(
                    airgroup_id as u64,
                    air_id as u64,
                    proof_type,
                    (&setup.p_setup).into(),
                    d_buffers,
                    setup.verkey.as_ptr() as *mut u8,
                    std::ptr::null_mut(),
                );
            }
        }

        let n_airgroups = pctx.global_info.air_groups.len();
        for airgroup_id in 0..n_airgroups {
            let setup = setups.sctx_recursive2.as_ref().unwrap().get_setup(airgroup_id, 0)?;
            let proof_type: &str = setup.setup_type.clone().into();
            if cfg!(feature = "gpu") {
                tracing::debug!(airgroup_id, air_id = 0, proof_type, "Loading expressions setup in GPU");
            }
            load_device_setup_c(
                airgroup_id as u64,
                0_u64,
                proof_type,
                (&setup.p_setup).into(),
                d_buffers,
                setup.verkey.as_ptr() as *mut u8,
                std::ptr::null_mut(),
            );
        }

        let setup_vadcop_final = setups.setup_vadcop_final.as_ref().unwrap();
        let proof_type: &str = setup_vadcop_final.setup_type.clone().into();
        if cfg!(feature = "gpu") {
            tracing::debug!(airgroup_id = 0, air_id = 0, proof_type, "Loading expressions setup in GPU");
        }
        load_device_setup_c(
            0_u64,
            0_u64,
            proof_type,
            (&setup_vadcop_final.p_setup).into(),
            d_buffers,
            setup_vadcop_final.verkey.as_ptr() as *mut u8,
            std::ptr::null_mut(),
        );

        let setup_vadcop_final_compressed = setups.setup_vadcop_final_compressed.as_ref().unwrap();
        let proof_type: &str = setup_vadcop_final_compressed.setup_type.clone().into();
        if cfg!(feature = "gpu") {
            tracing::debug!(airgroup_id = 0, air_id = 0, proof_type, "Loading expressions setup in GPU");
        }
        load_device_setup_c(
            0_u64,
            0_u64,
            proof_type,
            (&setup_vadcop_final_compressed.p_setup).into(),
            d_buffers,
            setup_vadcop_final_compressed.verkey.as_ptr() as *mut u8,
            std::ptr::null_mut(),
        );
    }
    Ok(())
}

pub fn load_device_const_pols<F: PrimeField64>(
    pctx: &ProofCtx<F>,
    sctx: &SetupCtx<F>,
    setups: &SetupsVadcop<F>,
    verify_constraints: bool,
    aggregation: bool,
    only_first_gpu: bool,
) -> ProofmanResult<()> {
    let d_buffers = pctx.get_device_buffers_ptr();

    // Phase 2: Load all constant polynomials
    let mut offset = 0;
    for (airgroup_id, air_group) in pctx.global_info.airs.iter().enumerate() {
        for (air_id, _) in air_group.iter().enumerate() {
            let setup = sctx.get_setup(airgroup_id, air_id)?;
            let proof_type: &str = setup.setup_type.clone().into();
            if cfg!(feature = "gpu") {
                let const_pols_path = &setup.const_pols_path;
                tracing::debug!(airgroup_id, air_id, proof_type, "Loading const pols in GPU");
                let load_tree = setup.preallocate && !verify_constraints;
                let tree_path = match load_tree {
                    true => &setup.const_pols_tree_path,
                    false => "",
                };
                load_device_const_pols_c(
                    airgroup_id as u64,
                    air_id as u64,
                    offset,
                    d_buffers,
                    const_pols_path,
                    setup.const_pols_size_packed as u64,
                    tree_path,
                    setup.const_tree_size as u64,
                    proof_type,
                    only_first_gpu,
                );
                offset += setup.const_pols_size_packed as u64;
                if load_tree {
                    offset += setup.const_tree_size as u64;
                }
            }
        }
    }

    let mut _offset_aggregation = 0;
    if aggregation {
        for (airgroup_id, air_group) in pctx.global_info.airs.iter().enumerate() {
            for (air_id, _) in air_group.iter().enumerate() {
                if pctx.global_info.get_air_has_compressor(airgroup_id, air_id) {
                    let setup = setups.sctx_compressor.as_ref().unwrap().get_setup(airgroup_id, air_id)?;
                    let proof_type: &str = setup.setup_type.clone().into();
                    if cfg!(feature = "gpu") {
                        let const_pols_path = &setup.const_pols_path;
                        tracing::debug!(airgroup_id, air_id, proof_type, "Loading const pols in GPU");
                        let load_tree = setup.preallocate && !verify_constraints;
                        let tree_path = match load_tree {
                            true => &setup.const_pols_tree_path,
                            false => "",
                        };
                        load_device_const_pols_c(
                            airgroup_id as u64,
                            air_id as u64,
                            _offset_aggregation,
                            d_buffers,
                            const_pols_path,
                            setup.const_pols_size_packed as u64,
                            tree_path,
                            setup.const_tree_size as u64,
                            proof_type,
                            only_first_gpu,
                        );
                        _offset_aggregation += setup.const_pols_size_packed as u64;
                        if load_tree {
                            _offset_aggregation += setup.const_tree_size as u64;
                        }
                    }
                }
            }
        }

        for (airgroup_id, air_group) in pctx.global_info.airs.iter().enumerate() {
            for (air_id, _) in air_group.iter().enumerate() {
                let setup = setups.sctx_recursive1.as_ref().unwrap().get_setup(airgroup_id, air_id)?;
                let proof_type: &str = setup.setup_type.clone().into();
                if cfg!(feature = "gpu") {
                    let const_pols_path = &setup.const_pols_path;
                    tracing::debug!(airgroup_id, air_id, proof_type, "Loading const pols in GPU");
                    let load_tree = setup.preallocate && !verify_constraints;
                    let tree_path = match load_tree {
                        true => &setup.const_pols_tree_path,
                        false => "",
                    };
                    load_device_const_pols_c(
                        airgroup_id as u64,
                        air_id as u64,
                        _offset_aggregation,
                        d_buffers,
                        const_pols_path,
                        setup.const_pols_size_packed as u64,
                        tree_path,
                        setup.const_tree_size as u64,
                        proof_type,
                        only_first_gpu,
                    );
                    _offset_aggregation += setup.const_pols_size_packed as u64;
                    if load_tree {
                        _offset_aggregation += setup.const_tree_size as u64;
                    }
                }
            }
        }

        let n_airgroups = pctx.global_info.air_groups.len();
        for airgroup_id in 0..n_airgroups {
            let setup = setups.sctx_recursive2.as_ref().unwrap().get_setup(airgroup_id, 0)?;
            let proof_type: &str = setup.setup_type.clone().into();
            if cfg!(feature = "gpu") {
                let const_pols_path = &setup.const_pols_path;
                tracing::debug!(airgroup_id, air_id = 0, proof_type, "Loading const pols in GPU");
                let load_tree = setup.preallocate && !verify_constraints;
                let tree_path = match load_tree {
                    true => &setup.const_pols_tree_path,
                    false => "",
                };
                load_device_const_pols_c(
                    airgroup_id as u64,
                    0_u64,
                    _offset_aggregation,
                    d_buffers,
                    const_pols_path,
                    setup.const_pols_size_packed as u64,
                    tree_path,
                    setup.const_tree_size as u64,
                    proof_type,
                    only_first_gpu,
                );
                _offset_aggregation += setup.const_pols_size_packed as u64;
                if load_tree {
                    _offset_aggregation += setup.const_tree_size as u64;
                }
            }
        }

        let setup_vadcop_final = setups.setup_vadcop_final.as_ref().unwrap();
        let proof_type: &str = setup_vadcop_final.setup_type.clone().into();
        if cfg!(feature = "gpu") {
            let const_pols_path = &setup_vadcop_final.const_pols_path;
            tracing::debug!(airgroup_id = 0, air_id = 0, proof_type, "Loading const pols in GPU");
            let load_tree = setup_vadcop_final.preallocate && !verify_constraints;
            let tree_path = match load_tree {
                true => &setup_vadcop_final.const_pols_tree_path,
                false => "",
            };
            load_device_const_pols_c(
                0_u64,
                0_u64,
                _offset_aggregation,
                d_buffers,
                const_pols_path,
                setup_vadcop_final.const_pols_size_packed as u64,
                tree_path,
                setup_vadcop_final.const_tree_size as u64,
                proof_type,
                only_first_gpu,
            );
            _offset_aggregation += setup_vadcop_final.const_pols_size_packed as u64;
            if load_tree {
                _offset_aggregation += setup_vadcop_final.const_tree_size as u64;
            }
        }

        let setup_vadcop_final_compressed = setups.setup_vadcop_final_compressed.as_ref().unwrap();
        let proof_type: &str = setup_vadcop_final_compressed.setup_type.clone().into();
        if cfg!(feature = "gpu") {
            let const_pols_path = &setup_vadcop_final_compressed.const_pols_path;
            tracing::debug!(airgroup_id = 0, air_id = 0, proof_type, "Loading const pols in GPU");
            let load_tree = setup_vadcop_final_compressed.preallocate && !verify_constraints;
            let tree_path = match load_tree {
                true => &setup_vadcop_final_compressed.const_pols_tree_path,
                false => "",
            };
            load_device_const_pols_c(
                0_u64,
                0_u64,
                _offset_aggregation,
                d_buffers,
                const_pols_path,
                setup_vadcop_final_compressed.const_pols_size_packed as u64,
                tree_path,
                setup_vadcop_final_compressed.const_tree_size as u64,
                proof_type,
                only_first_gpu,
            );
            _offset_aggregation += setup_vadcop_final_compressed.const_pols_size_packed as u64;
            if load_tree {
                _offset_aggregation += setup_vadcop_final_compressed.const_tree_size as u64;
            }
        }
    }
    Ok(())
}

pub fn initialize_witness_circom<F: PrimeField64>(pctx: &ProofCtx<F>, setups: &SetupsVadcop<F>) -> ProofmanResult<()> {
    for (airgroup_id, air_group) in pctx.global_info.airs.iter().enumerate() {
        for (air_id, _) in air_group.iter().enumerate() {
            if pctx.global_info.get_air_has_compressor(airgroup_id, air_id) {
                let setup = setups.sctx_compressor.as_ref().unwrap().get_setup(airgroup_id, air_id)?;
                setup.set_exec_file_data()?;
                setup.set_circom_circuit()?;
            }
            let setup = setups.sctx_recursive1.as_ref().unwrap().get_setup(airgroup_id, air_id)?;
            setup.set_exec_file_data()?;
            setup.set_circom_circuit()?;
        }
    }

    let n_airgroups = pctx.global_info.air_groups.len();
    for airgroup in 0..n_airgroups {
        let setup = setups.sctx_recursive2.as_ref().unwrap().get_setup(airgroup, 0)?;
        setup.set_circom_circuit()?;
        setup.set_exec_file_data()?;
    }

    let setup_vadcop_final = setups.setup_vadcop_final.as_ref().unwrap();
    setup_vadcop_final.set_circom_circuit()?;
    setup_vadcop_final.set_exec_file_data()?;

    let setup_vadcop_final_compressed = setups.setup_vadcop_final_compressed.as_ref().unwrap();
    setup_vadcop_final_compressed.set_circom_circuit()?;
    setup_vadcop_final_compressed.set_exec_file_data()?;

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
    wcm.register_component_std(std.prod_bus.clone());
    wcm.register_component_std(std.sum_bus.clone());
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

    wcm.register_component_std(std.virtual_table.clone());
    if std.virtual_table.virtual_table_airs.is_some() {
        for air in std.virtual_table.virtual_table_airs.clone().unwrap() {
            wcm.register_component_std(air);
        }
    }
}

pub fn register_std_dev<F: PrimeField64>(
    wcm: &WitnessManager<F>,
    std: &Std<F>,
    register_u8: bool,
    register_u16: bool,
    register_specified_ranges: bool,
) {
    wcm.register_component_std(std.prod_bus.clone());
    wcm.register_component_std(std.sum_bus.clone());
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

    wcm.register_component_std(std.virtual_table.clone());
}

pub fn print_roots<F: PrimeField64>(pctx: &ProofCtx<F>, roots_contributions: &[[F; 4]]) {
    let instances = pctx.dctx_get_instances();
    for (instance_id, &instance_info) in instances.iter().enumerate() {
        let (airgroup_id, air_id) = (instance_info.airgroup_id, instance_info.air_id);
        let contribution = roots_contributions[instance_id];
        tracing::info!(
            "Contribution for instance id {} [{}:{}] is: {:?}",
            instance_id,
            airgroup_id,
            air_id,
            contribution,
        );
    }
}

pub fn get_vadcop_final_proof_vkey(proving_key_path: &Path, compressed: bool) -> ProofmanResult<Vec<u8>> {
    let global_info = GlobalInfo::new(proving_key_path)?;
    let setup_path = match compressed {
        true => global_info.get_setup_path("vadcop_final_compressed"),
        false => global_info.get_setup_path("vadcop_final"),
    };

    let verkey_file = setup_path.display().to_string() + ".verkey.bin";

    let mut file = File::open(&verkey_file)
        .map_err(|e| ProofmanError::InvalidSetup(format!("Failed to open verkey file '{}': {}", verkey_file, e)))?;

    let mut contents = Vec::new();
    file.read_to_end(&mut contents)
        .map_err(|e| ProofmanError::InvalidSetup(format!("Failed to read verkey file '{}': {}", verkey_file, e)))?;

    Ok(contents)
}

pub fn deterministic_shuffle<T>(slice: &mut [T], seed: u64) {
    let len = slice.len();
    if len <= 1 {
        return;
    }

    const A: u64 = 1103515245;
    const C: u64 = 12345;
    const M: u64 = 1 << 31;

    let mut state = seed;

    for i in (1..len).rev() {
        state = state.wrapping_mul(A).wrapping_add(C) % M;
        let j = (state as usize) % (i + 1);
        slice.swap(i, j);
    }
}
