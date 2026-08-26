use proofman_fields::PrimeField64;
use std::fs::{self, File};
use std::io::{Read, Seek, SeekFrom};
use std::{
    collections::HashMap,
    path::{Path, PathBuf},
};
use std::os::raw::c_void;

use colored::*;

use proofman_common::{
    format_bytes, FixedGroup, MpiCtx, ProofCtx, ProofType, ProofmanError, ProofmanResult, Setup, SetupCtx, SetupsVadcop,
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

use pil2_std_lib::Std;
use proofman_witness::WitnessManager;

pub fn print_summary_info<F: PrimeField64>(
    pctx: &ProofCtx<F>,
    sctx: &SetupCtx<F>,
    mpi_ctx: &MpiCtx,
    packed_info: &HashMap<(usize, usize), PackedInfo>,
    verbose_mode: VerboseMode,
) -> ProofmanResult<String> {
    let n_partitions = pctx.get_n_partitions();
    let summary = print_summary(
        pctx,
        sctx,
        packed_info,
        true,
        mpi_ctx.rank,
        mpi_ctx.n_processes,
        verbose_mode,
        n_partitions == 1 && mpi_ctx.rank == 0,
    )?;

    let _ = print_summary(
        pctx,
        sctx,
        packed_info,
        false,
        mpi_ctx.rank,
        mpi_ctx.n_processes,
        verbose_mode,
        n_partitions != 1 || mpi_ctx.n_processes > 1,
    )?;

    if mpi_ctx.n_processes > 1 {
        let (average_weight, max_weight, min_weight, max_deviation) = pctx.dctx_load_balance_info_process();
        tracing::info!(
            "Load balance (processes). Average: {average_weight} max: {max_weight} min: {min_weight} deviation: {max_deviation:.2}%"
        );
    }

    Ok(summary)
}

#[allow(clippy::too_many_arguments)]
pub fn print_summary<F: PrimeField64>(
    pctx: &ProofCtx<F>,
    sctx: &SetupCtx<F>,
    packed_info: &HashMap<(usize, usize), PackedInfo>,
    global: bool,
    rank: i32,
    n_processes: i32,
    verbose_mode: VerboseMode,
    print_output: bool,
) -> ProofmanResult<String> {
    let mut summary_info = String::new();

    let mut air_info = HashMap::new();

    let mut air_instances = HashMap::new();

    let instances = pctx.dctx_get_instances();

    let (print, n_instances) = if global {
        let n = instances.len();
        (vec![true; n], n)
    } else {
        let my_instances = pctx.dctx_get_process_instances();
        let n = my_instances.len();
        let mut print = vec![false; instances.len()];
        for id in my_instances {
            print[id] = true;
        }
        (print, n)
    };

    let max_prover_memory = sctx.max_prover_buffer_size as f64 * 8.0;

    let mut memory_tables = 0 as f64;
    let mut total_weight: u64 = 0;
    for (instance_id, &instance_info) in instances.iter().enumerate() {
        let (airgroup_id, air_id, is_table) = (instance_info.airgroup_id, instance_info.air_id, instance_info.table);
        if !print[instance_id] {
            continue;
        }
        if !global {
            total_weight += pctx.get_weight(airgroup_id, air_id);
        }
        let air_name = pctx.global_info.airs[airgroup_id][air_id].clone().name;
        let air_group_name = pctx.global_info.air_groups[airgroup_id].clone();
        let air_instance_map = air_instances.entry(air_group_name).or_insert_with(HashMap::new);
        if !air_instance_map.contains_key(&air_name.clone()) {
            let setup = sctx.get_setup(airgroup_id, air_id)?;
            let n_bits = setup.stark_info.stark_struct.n_bits;
            let num_packed_words = packed_info.get(&(airgroup_id, air_id)).map(|info| info.num_packed_words);
            let memory_trace = if let Some(num_packed_words) = num_packed_words {
                (num_packed_words * (1 << setup.stark_info.stark_struct.n_bits)) as f64 * 8.0
            } else {
                (setup.stark_info.map_sections_n["cm1"] * (1 << setup.stark_info.stark_struct.n_bits)) as f64 * 8.0
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
            let header = if global {
                "--- TOTAL PROOF INSTANCES SUMMARY ------------------------".to_string()
            } else {
                format!("--- PROOF INSTANCES SUMMARY [Process {rank}/{n_processes}] -------")
            };
            tracing::info!("{}", header.bright_white().bold());
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
                if print_output {
                    if pctx.gpu {
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
            if !global {
                tracing::info!("      Total weight [Process {rank}/{n_processes}]: {total_weight}");
                tracing::info!("----------------------------------------------------------");
            }
        }
    } else {
        if print_output {
            let header = if global {
                "--- PROOF INSTANCES SUMMARY ---".to_string()
            } else {
                format!("--- PROOF INSTANCES SUMMARY [Process {rank}/{n_processes}] ---")
            };
            tracing::info!("{}", header.bright_white().bold());
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

            let scope = if global { "global" } else { "process" };
            summary.push(format!("Total {scope} instances: {n_instances}"));
            if !global {
                summary.push(format!("Total weight: {total_weight}"));
            }

            if print_output {
                tracing::info!("{} | {}", air_group.bright_white().bold(), summary.join(" | "));
            }

            if global {
                summary_info = summary.join(" | ");
            }
        }

        if print_output {
            if !global {
                tracing::info!("Total weight [Process {rank}/{n_processes}]: {total_weight}");
            }
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

    if setup.gpu && setup.setup_type == ProofType::RecursiveF && !needs_regeneration {
        let expected_const_gpu_size = setup.const_pols_size * 8;
        let const_gpu_valid = PathBuf::from(&setup.const_pols_path).exists()
            && fs::metadata(&setup.const_pols_path)
                .map(|m| m.len() as usize == expected_const_gpu_size)
                .unwrap_or(false);
        if !const_gpu_valid {
            needs_regeneration = true;
        }
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
            if setup.gpu {
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
            if setup.gpu {
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

/// Is `.const_gpu` *provably* older than the `.const` it was packed from?
///
/// The size check cannot tell: the expected size is pure AIR geometry, so changed
/// fixed-polynomial *values* leave it byte-identical. `.consttree` checks its root
/// against the verkey instead; a flat repack has no root. Unreadable metadata answers
/// "no": a GPU key needs no `.const`, and the repack this would trigger reads it.
fn const_gpu_older_than_source(const_path: &Path, const_gpu_path: &Path) -> bool {
    let modified = |p: &Path| fs::metadata(p).and_then(|m| m.modified()).ok();
    match (modified(const_path), modified(const_gpu_path)) {
        (Some(src), Some(derived)) => src > derived,
        _ => false,
    }
}

pub fn needs_const_pols_gpu_regeneration<F: PrimeField64>(setup: &Setup<F>) -> ProofmanResult<bool> {
    if !setup.gpu {
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

    let source = setup.setup_path.display().to_string() + ".const";
    if const_gpu_older_than_source(Path::new(&source), Path::new(&setup.const_pols_path)) {
        return Ok(true);
    }

    Ok(false)
}

pub fn check_const_pols_gpu<F: PrimeField64>(setup: &Setup<F>) -> ProofmanResult<()> {
    if !setup.gpu {
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

    let source = setup.setup_path.display().to_string() + ".const";
    if !needs_regeneration && const_gpu_older_than_source(Path::new(&source), Path::new(&setup.const_pols_path)) {
        tracing::trace!(
            "GPU constant polynomials file '{}' is older than '{}'. Regenerating...",
            setup.const_pols_path,
            source
        );
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
                if setup.gpu {
                    needs_const_regen = true;
                    tracing::debug!(
                        "GPU const pols regeneration also needed for [{}:{}] due to tree regeneration",
                        airgroup_id,
                        air_id
                    );
                }
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
                    if setup.gpu {
                        needs_const_regen = true;
                        tracing::debug!("Vadcop compressor const pols regeneration also needed for [{}:{}] due to tree regeneration", airgroup_id, air_id);
                    }
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
                if setup.gpu {
                    needs_const_regen = true;
                    tracing::debug!(
                        "Vadcop recursive1 const pols regeneration also needed for [{}:{}] due to tree regeneration",
                        airgroup_id,
                        air_id
                    );
                }
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
            if setup.gpu {
                needs_const_regen = true;
                tracing::debug!(
                    "Vadcop recursive2 const pols regeneration also needed for airgroup {} due to tree regeneration",
                    airgroup
                );
            }
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
        if setup_vadcop_final.gpu {
            needs_const_regen = true;
            tracing::debug!("Vadcop final const pols regeneration also needed due to tree regeneration");
        }
    }

    let setup_vadcop_final_compressed = setups.setup_vadcop_final_compressed.as_ref().unwrap();
    if needs_const_pols_gpu_regeneration(setup_vadcop_final_compressed)? {
        needs_const_regen = true;
        tracing::debug!("Vadcop final compressed const pols regeneration needed");
    }
    if needs_const_tree_regeneration(setup_vadcop_final_compressed)? {
        needs_tree_regen = true;
        tracing::debug!("Vadcop final compressed tree regeneration needed");
        if setup_vadcop_final_compressed.gpu {
            needs_const_regen = true;
            tracing::debug!("Vadcop final compressed const pols regeneration also needed due to tree regeneration");
        }
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
            let d_buffers = if pctx.gpu { Some(pctx.get_device_buffers_ptr()) } else { None };
            check_const_tree(setup, &d_buffers)?;
        }
    }
    Ok(())
}

pub fn check_tree_paths_vadcop<F: PrimeField64>(pctx: &ProofCtx<F>, setups: &SetupsVadcop<F>) -> ProofmanResult<()> {
    let d_buffers = if pctx.gpu { Some(pctx.get_device_buffers_ptr()) } else { None };
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
) -> ProofmanResult<(usize, usize)> {
    let mut max_witness_trace_size = 0;
    let mut max_witness_trace_size_packed = 0;
    for (airgroup_id, air_group) in pctx.global_info.airs.iter().enumerate() {
        for (air_id, _) in air_group.iter().enumerate() {
            let setup = sctx.get_setup(airgroup_id, air_id)?;
            let n = 1 << setup.stark_info.stark_struct.n_bits;
            let num_packed_words =
                packed_info.get(&(airgroup_id, air_id)).map(|info| info.num_packed_words).unwrap_or(0);
            let n_cols = setup.stark_info.map_sections_n["cm1"];

            let trace_size = n * n_cols;
            let trace_size_packed = if num_packed_words == 0 { n * n_cols } else { n * num_packed_words };

            max_witness_trace_size = max_witness_trace_size.max(trace_size as usize);
            max_witness_trace_size_packed = max_witness_trace_size_packed.max(trace_size_packed as usize);
        }
    }
    Ok((max_witness_trace_size, max_witness_trace_size_packed))
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
            let proof_type: &str = setup.setup_type.into();
            if setup.gpu {
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
                    let proof_type: &str = setup.setup_type.into();
                    if setup.gpu {
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
                let proof_type: &str = setup.setup_type.into();
                if setup.gpu {
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
            let proof_type: &str = setup.setup_type.into();
            if setup.gpu {
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
        let proof_type: &str = setup_vadcop_final.setup_type.into();
        if setup_vadcop_final.gpu {
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
        let proof_type: &str = setup_vadcop_final_compressed.setup_type.into();
        if setup_vadcop_final_compressed.gpu {
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

/// Uploads one air's packed const pols -- plus its tree when the group is preallocated --
/// at `*offset`, then advances past the slot. Only a group owner transfers anything; later
/// members are just pointed at its offset. `slots` is per const buffer. Must stay in lockstep
/// with the sizing in `SetupRepository::new`, which walks the same airs in the same order.
#[allow(clippy::too_many_arguments)]
fn load_const_pols_slot<F: PrimeField64>(
    d_buffers: *mut c_void,
    setup: &Setup<F>,
    group: FixedGroup,
    airgroup_id: usize,
    air_id: usize,
    verify_constraints: bool,
    only_first_gpu: bool,
    slots: &mut HashMap<(usize, usize), u64>,
    offset: &mut u64,
) {
    let proof_type: &str = setup.setup_type.into();
    let load_tree = group.load_tree && !verify_constraints;
    let tree_path = match load_tree {
        true => setup.const_pols_tree_path.as_str(),
        false => "",
    };

    let shared_slot = slots.get(&group.owner).copied();
    let slot_offset = shared_slot.unwrap_or(*offset);
    tracing::debug!(
        airgroup_id,
        air_id,
        proof_type,
        slot_offset,
        shared = shared_slot.is_some(),
        "Loading const pols in GPU"
    );
    load_device_const_pols_c(
        airgroup_id as u64,
        air_id as u64,
        slot_offset,
        d_buffers,
        &setup.const_pols_path,
        setup.const_pols_size_packed as u64,
        tree_path,
        setup.const_tree_size as u64,
        proof_type,
        only_first_gpu,
        shared_slot.is_some(),
    );

    if shared_slot.is_none() {
        slots.insert(group.owner, slot_offset);
        *offset += setup.const_pols_size_packed as u64;
        if load_tree {
            *offset += setup.const_tree_size as u64;
        }
    }
}

/// Defaults to a group of its own for setups the repository did not fingerprint: the
/// standalone vadcop_final setups, and any air without a verkey.
fn fixed_group_or_own<F: PrimeField64>(
    sctx: Option<&SetupCtx<F>>,
    setup: &Setup<F>,
    airgroup_id: usize,
    air_id: usize,
) -> FixedGroup {
    sctx.and_then(|s| s.get_fixed_group(airgroup_id, air_id))
        .unwrap_or(FixedGroup { owner: (airgroup_id, air_id), load_tree: setup.preallocate })
}

pub fn load_device_const_pols<F: PrimeField64>(
    pctx: &ProofCtx<F>,
    sctx: &SetupCtx<F>,
    setups: &SetupsVadcop<F>,
    verify_constraints: bool,
    aggregation: bool,
    only_first_gpu: bool,
) -> ProofmanResult<u64> {
    let d_buffers = pctx.get_device_buffers_ptr();

    // Phase 2: Load all constant polynomials. One slot map per const buffer -- basic lives in
    // d_constPols, everything below in d_constPolsAggregation, and their offsets are unrelated.
    let mut offset = 0;
    let mut basic_slots: HashMap<(usize, usize), u64> = HashMap::new();
    for (airgroup_id, air_group) in pctx.global_info.airs.iter().enumerate() {
        for (air_id, _) in air_group.iter().enumerate() {
            let setup = sctx.get_setup(airgroup_id, air_id)?;
            if setup.gpu {
                let group = fixed_group_or_own(Some(sctx), setup, airgroup_id, air_id);
                load_const_pols_slot(
                    d_buffers,
                    setup,
                    group,
                    airgroup_id,
                    air_id,
                    verify_constraints,
                    only_first_gpu,
                    &mut basic_slots,
                    &mut offset,
                );
            }
        }
    }

    let mut offset_aggregation = 0;
    if aggregation {
        let mut compressor_slots: HashMap<(usize, usize), u64> = HashMap::new();
        let mut recursive1_slots: HashMap<(usize, usize), u64> = HashMap::new();
        let mut recursive2_slots: HashMap<(usize, usize), u64> = HashMap::new();

        for (airgroup_id, air_group) in pctx.global_info.airs.iter().enumerate() {
            for (air_id, _) in air_group.iter().enumerate() {
                if pctx.global_info.get_air_has_compressor(airgroup_id, air_id) {
                    let sctx_compressor = setups.sctx_compressor.as_ref().unwrap();
                    let setup = sctx_compressor.get_setup(airgroup_id, air_id)?;
                    if setup.gpu {
                        let group = fixed_group_or_own(Some(sctx_compressor), setup, airgroup_id, air_id);
                        load_const_pols_slot(
                            d_buffers,
                            setup,
                            group,
                            airgroup_id,
                            air_id,
                            verify_constraints,
                            only_first_gpu,
                            &mut compressor_slots,
                            &mut offset_aggregation,
                        );
                    }
                }
            }
        }

        for (airgroup_id, air_group) in pctx.global_info.airs.iter().enumerate() {
            for (air_id, _) in air_group.iter().enumerate() {
                let sctx_recursive1 = setups.sctx_recursive1.as_ref().unwrap();
                let setup = sctx_recursive1.get_setup(airgroup_id, air_id)?;
                if setup.gpu {
                    let group = fixed_group_or_own(Some(sctx_recursive1), setup, airgroup_id, air_id);
                    load_const_pols_slot(
                        d_buffers,
                        setup,
                        group,
                        airgroup_id,
                        air_id,
                        verify_constraints,
                        only_first_gpu,
                        &mut recursive1_slots,
                        &mut offset_aggregation,
                    );
                }
            }
        }

        let n_airgroups = pctx.global_info.air_groups.len();
        for airgroup_id in 0..n_airgroups {
            let sctx_recursive2 = setups.sctx_recursive2.as_ref().unwrap();
            let setup = sctx_recursive2.get_setup(airgroup_id, 0)?;
            if setup.gpu {
                let group = fixed_group_or_own(Some(sctx_recursive2), setup, airgroup_id, 0);
                load_const_pols_slot(
                    d_buffers,
                    setup,
                    group,
                    airgroup_id,
                    0,
                    verify_constraints,
                    only_first_gpu,
                    &mut recursive2_slots,
                    &mut offset_aggregation,
                );
            }
        }

        // The two finals are single setups, not repository entries: each is its own group.
        let mut final_slots: HashMap<(usize, usize), u64> = HashMap::new();

        let setup_vadcop_final = setups.setup_vadcop_final.as_ref().unwrap();
        if setup_vadcop_final.gpu {
            let group = fixed_group_or_own(None::<&SetupCtx<F>>, setup_vadcop_final, 0, 0);
            load_const_pols_slot(
                d_buffers,
                setup_vadcop_final,
                group,
                0,
                0,
                verify_constraints,
                only_first_gpu,
                &mut final_slots,
                &mut offset_aggregation,
            );
        }

        let setup_vadcop_final_compressed = setups.setup_vadcop_final_compressed.as_ref().unwrap();
        if setup_vadcop_final_compressed.gpu {
            // Distinct key from vadcop_final above: both report (0, 0) but are separate slots.
            let group = FixedGroup { owner: (0, 1), load_tree: setup_vadcop_final_compressed.preallocate };
            load_const_pols_slot(
                d_buffers,
                setup_vadcop_final_compressed,
                group,
                0,
                0,
                verify_constraints,
                only_first_gpu,
                &mut final_slots,
                &mut offset_aggregation,
            );
        }
    }
    Ok(offset_aggregation)
}

pub fn add_publics_circom<F: PrimeField64>(
    proof: &mut [u64],
    initial_index: usize,
    pctx: &ProofCtx<F>,
    root_agg_verkey: Option<&[F]>,
) {
    let init_index = initial_index;

    let publics = pctx.public_inputs.values.read().unwrap();
    for p in 0..pctx.global_info.n_publics {
        proof[init_index + p] = publics[p].as_canonical_u64();
    }

    let proof_values = pctx.proof_values.values.read().unwrap();
    let proof_values_map = pctx.global_info.proof_values_map.as_ref().unwrap();
    let mut p = 0;
    for (idx, proof_value_map) in proof_values_map.iter().enumerate() {
        if proof_value_map.stage == 1 {
            proof[init_index + pctx.global_info.n_publics + 3 * idx] = proof_values[p].as_canonical_u64();
            proof[init_index + pctx.global_info.n_publics + 3 * idx + 1] = 0;
            proof[init_index + pctx.global_info.n_publics + 3 * idx + 2] = 0;
            p += 1;
        } else {
            proof[init_index + pctx.global_info.n_publics + 3 * idx] = proof_values[p].as_canonical_u64();
            proof[init_index + pctx.global_info.n_publics + 3 * idx + 1] = proof_values[p + 1].as_canonical_u64();
            proof[init_index + pctx.global_info.n_publics + 3 * idx + 2] = proof_values[p + 2].as_canonical_u64();
            p += 3;
        }
    }

    let global_challenge = pctx.global_challenge.values.read().unwrap();
    proof[init_index + pctx.global_info.n_publics + 3 * proof_values_map.len()] =
        global_challenge[0].as_canonical_u64();
    proof[init_index + pctx.global_info.n_publics + 3 * proof_values_map.len() + 1] =
        global_challenge[1].as_canonical_u64();
    proof[init_index + pctx.global_info.n_publics + 3 * proof_values_map.len() + 2] =
        global_challenge[2].as_canonical_u64();

    if let Some(vk) = root_agg_verkey {
        for i in 0..4 {
            proof[init_index + pctx.global_info.n_publics + 3 * proof_values_map.len() + 3 + i] =
                vk[i].as_canonical_u64();
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
        proof[initial_index + p] = publics[p].as_canonical_u64();
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

pub fn get_vadcop_final_proof_vkey(proving_key_path: &Path, compressed: bool) -> ProofmanResult<Vec<u64>> {
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

    if !contents.len().is_multiple_of(8) {
        return Err(ProofmanError::InvalidSetup(format!(
            "Verkey file '{}' has length {} which is not a multiple of 8",
            verkey_file,
            contents.len()
        )));
    }

    Ok(contents.as_chunks::<8>().0.iter().map(|c| u64::from_le_bytes(*c)).collect())
}

#[cfg(test)]
mod const_gpu_staleness_tests {
    use super::const_gpu_older_than_source;
    use std::fs;
    use std::path::PathBuf;

    fn scratch(tag: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("const_gpu_stale_{tag}"));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        dir
    }

    /// What a size check cannot see: `.const` rewritten to the same length with
    /// different values -- changed fixed pols over unchanged AIR geometry.
    #[test]
    fn same_size_newer_source_is_stale() {
        let d = scratch("same_size");
        let src = d.join("a.const");
        let derived = d.join("a.const_gpu");
        fs::write(&src, vec![1u8; 4096]).unwrap();
        fs::write(&derived, vec![9u8; 4096]).unwrap();
        assert!(!const_gpu_older_than_source(&src, &derived), "freshly packed must not look stale");

        std::thread::sleep(std::time::Duration::from_millis(20));
        fs::write(&src, vec![2u8; 4096]).unwrap();
        assert_eq!(
            fs::metadata(&src).unwrap().len(),
            4096,
            "length must be unchanged for this to be the interesting case"
        );
        assert!(const_gpu_older_than_source(&src, &derived), "same-size source rewrite must be detected");
    }

    /// A GPU proving key is complete without `.const`, and the repack a "stale" answer
    /// triggers loads `.const` -- so an absent source must never report stale.
    #[test]
    fn missing_source_is_not_stale() {
        let d = scratch("missing");
        let src = d.join("a.const");
        let derived = d.join("a.const_gpu");
        fs::write(&derived, vec![9u8; 4096]).unwrap();
        assert!(!const_gpu_older_than_source(&src, &derived), "source missing: size check alone decides");
        assert!(!const_gpu_older_than_source(&src, &d.join("absent.const_gpu")), "neither present");
    }
}
