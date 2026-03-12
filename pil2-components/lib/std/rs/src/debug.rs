use std::{
    fs::{self, File},
    io::{self, Write},
    path::{Path, PathBuf},
};

use rayon::prelude::*;
use rustc_hash::FxHashMap;

use proofman_common::SetupCtx;

use colored::Colorize;
use fields::PrimeField64;
use proofman_common::{ProofCtx, ProofmanError, ProofmanResult};
use proofman_hints::{
    get_hint_ids_by_name, format_hint_field_output_vec, HintFieldOutput, HintFieldValue, HintFieldValuesVec,
    HintFieldOptions,
};
use proofman_util::{timer_start_info, timer_stop_and_log_info};

use crate::{
    get_global_hint_field_constant_a_as_string, get_global_hint_field_constant_as_string,
    get_hint_field_constant_a_as_string, get_hint_field_constant_as_string,
};

#[derive(Clone)]
pub struct HintMetadata<F: PrimeField64> {
    pub hint: u64,
    pub hint_id: usize,
    pub busid: HintFieldValue<F>,
    pub type_piop: u64,
    pub num_reps: HintFieldValue<F>,
    pub expressions: HintFieldValuesVec<F>,
    pub name_piop: String,
    pub name_exprs: Vec<String>,
    pub deg_expr: F,
    pub deg_mul: F,
}

pub type DebugData = FxHashMap<u64, FxHashMap<u64, BusValue>>; // opid -> val -> SharedData
pub type DebugDataInfo = FxHashMap<u64, FxHashMap<u64, BusValueInfo>>;

#[derive(Debug)]
pub struct BusValue {
    shared_data: SharedData, // Data shared across all airgroups, airs, and instances
}

#[derive(Debug)]
pub struct BusValueInfo {
    local_data: LocalBusMap, // Data grouped by: airgroup_id -> air_id -> AirData -> instance_id -> InstanceData
    global_data: Option<GlobalAirGroupData>,
}

#[derive(Debug)]
struct SharedData {
    vals: String,
    num_proves: u64,
    num_assumes: u64,
}

#[derive(Default, Debug)]
pub struct GlobalAirGroupData(u32);

impl GlobalAirGroupData {
    fn new(airgroup_id: u8, hint_id: u16, is_prod: bool) -> Self {
        let prod_flag = if is_prod { 1u32 } else { 0u32 };
        let val = ((airgroup_id as u32) << 24) | ((hint_id as u32) << 8) | prod_flag;
        GlobalAirGroupData(val)
    }

    fn unpack(&self) -> (u8, u16, bool) {
        let airgroup_id = (self.0 >> 24) as u8;
        let hint_id = ((self.0 >> 8) & 0xFFFF) as u16;
        let is_prod = (self.0 & 0xFF) != 0;
        (airgroup_id, hint_id, is_prod)
    }
}

#[derive(Debug, Copy, Clone, Hash, Eq, PartialEq)]
pub struct LocalKey(u64);

impl LocalKey {
    fn new(airgroup_id: u8, air_id: u8, instance_id: u16, hint_id: u16, is_prod: bool) -> Self {
        let prod_flag = if is_prod { 1u64 } else { 0u64 };
        let val = ((airgroup_id as u64) << 56)
            | ((air_id as u64) << 48)
            | ((instance_id as u64) << 32)
            | ((hint_id as u64) << 16)
            | prod_flag;
        LocalKey(val)
    }

    fn unpack(self) -> (u8, u8, u16, u16, bool) {
        let airgroup_id = (self.0 >> 56) as u8;
        let air_id = ((self.0 >> 48) & 0xFF) as u8;
        let instance_id = ((self.0 >> 32) & 0xFFFF) as u16;
        let hint_id = ((self.0 >> 16) & 0xFFFF) as u16;
        let is_prod = (self.0 & 1) != 0;
        (airgroup_id, air_id, instance_id, hint_id, is_prod)
    }
}

#[derive(Debug, Default)]
struct LocalBusData {
    row_proves: Vec<usize>,
    row_assumes: Vec<usize>,
}

type LocalBusMap = FxHashMap<LocalKey, LocalBusData>;

/// Handle global debug data updates (shared across all instances)
#[allow(clippy::too_many_arguments)]
pub fn update_global_debug_data<F: PrimeField64>(
    debug_data: &mut DebugData,
    debug_data_info: &mut DebugDataInfo,
    hint_id: usize,
    opid: u64,
    norm_vals: &[HintFieldOutput<F>],
    hash: u64,
    airgroup_id: usize,
    is_proves: bool,
    times: u64,
    is_prod: bool,
) -> ProofmanResult<()> {
    let bus_opid = debug_data.entry(opid).or_default();
    let bus_val = bus_opid.entry(hash).or_insert_with(|| BusValue {
        shared_data: SharedData {
            vals: format_hint_field_output_vec(norm_vals).to_string(),
            num_proves: 0,
            num_assumes: 0,
        },
    });

    let bus_info_opid = debug_data_info.entry(opid).or_default();
    let bus_info_val = bus_info_opid
        .entry(hash)
        .or_insert_with(|| BusValueInfo { local_data: FxHashMap::default(), global_data: None });

    // Skip if already processed
    if bus_info_val.global_data.is_some() {
        return Ok(());
    }

    // Store global data for this airgroup
    let global_info_data = GlobalAirGroupData::new(airgroup_id as u8, hint_id as u16, is_prod);
    bus_info_val.global_data = Some(global_info_data);

    if is_proves {
        bus_val.shared_data.num_proves += times;
    } else {
        if times != 1 {
            return Err(ProofmanError::StdError(format!(
                "The selector value is invalid: expected 1, but received {times:?}."
            )));
        }
        bus_val.shared_data.num_assumes += times;
    }
    Ok(())
}

/// Handle local debug data updates (specific to airgroup/air/instance)
#[allow(clippy::too_many_arguments)]
pub fn update_local_debug_data<F: PrimeField64>(
    debug_data: &mut DebugData,
    debug_data_info: &mut DebugDataInfo,
    hint_id: usize,
    opid: u64,
    norm_vals: &[HintFieldOutput<F>],
    hash: u64,
    airgroup_id: usize,
    air_id: usize,
    instance_id: usize,
    row: usize,
    is_proves: bool,
    times: u64,
    is_prod: bool,
    store_row_info: bool,
) -> ProofmanResult<()> {
    let bus_val = debug_data.entry(opid).or_default().entry(hash).or_insert_with(|| BusValue {
        shared_data: SharedData {
            vals: format_hint_field_output_vec(norm_vals).to_string(),
            num_proves: 0,
            num_assumes: 0,
        },
    });

    if is_proves {
        bus_val.shared_data.num_proves += times;
    } else {
        bus_val.shared_data.num_assumes += times;
    }

    if store_row_info {
        let key = LocalKey::new(airgroup_id as u8, air_id as u8, instance_id as u16, hint_id as u16, is_prod);

        let bus_info_opid = debug_data_info.entry(opid).or_default();
        let bus_info_val = bus_info_opid
            .entry(hash)
            .or_insert_with(|| BusValueInfo { local_data: FxHashMap::default(), global_data: None });

        let local = bus_info_val
            .local_data
            .entry(key)
            .or_insert_with(|| LocalBusData { row_proves: vec![], row_assumes: vec![] });

        if is_proves {
            local.row_proves.push(row);
        } else {
            local.row_assumes.push(row);
        }
    }

    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub fn update_debug_data<F: PrimeField64>(
    debug_data: &mut DebugData,
    debug_data_info: &mut DebugDataInfo,
    hint_id: usize,
    opid: u64,
    norm_vals: &[HintFieldOutput<F>],
    hash: u64,
    airgroup_id: usize,
    air_id: Option<usize>,
    instance_id: Option<usize>,
    row: usize,
    is_proves: bool,
    times: u64,
    is_global: bool,
    is_prod: bool,
    store_row_info_: bool,
    debug_hashes: &[u64],
) -> ProofmanResult<()> {
    if !debug_hashes.is_empty() && !debug_hashes.contains(&hash) {
        return Ok(());
    }

    let store_row_info = store_row_info_ || !debug_hashes.is_empty();

    if is_global {
        update_global_debug_data(
            debug_data,
            debug_data_info,
            hint_id,
            opid,
            norm_vals,
            hash,
            airgroup_id,
            is_proves,
            times,
            is_prod,
        )
    } else {
        update_local_debug_data(
            debug_data,
            debug_data_info,
            hint_id,
            opid,
            norm_vals,
            hash,
            airgroup_id,
            air_id.unwrap(),
            instance_id.unwrap(),
            row,
            is_proves,
            times,
            is_prod,
            store_row_info,
        )
    }
}

pub fn print_debug_info<F: PrimeField64>(
    pctx: &ProofCtx<F>,
    sctx: &SetupCtx<F>,
    max_values_to_print: usize,
    print_to_file: bool,
    debug_data: &mut DebugData,
    debug_data_info: &mut DebugDataInfo,
) -> ProofmanResult<()> {
    timer_start_info!(PRINT_DEBUG_INFO);
    let mut file_path = PathBuf::new();
    let mut output: Box<dyn Write> = Box::new(io::stdout());
    let mut there_are_errors = false;

    // Parallel pre-filtering: collect only mismatched opids
    let mismatched_opids: Vec<_> = debug_data
        .par_iter()
        .filter_map(|(opid, bus)| {
            let has_mismatch = bus.iter().any(|(_, v)| v.shared_data.num_proves != v.shared_data.num_assumes);
            if has_mismatch {
                Some(*opid)
            } else {
                None
            }
        })
        .collect();

    // Early exit if no errors
    if mismatched_opids.is_empty() {
        tracing::info!("··· {}", "\u{2713} All bus values match.".bright_green().bold());
        timer_stop_and_log_info!(PRINT_DEBUG_INFO);
        return Ok(());
    }

    // Process mismatched opids serially for ordered output, consuming entries as we go
    for opid in mismatched_opids {
        let bus = debug_data.remove(&opid).unwrap();
        let bus_info = debug_data_info.remove(&opid);

        if !there_are_errors {
            // Print to a file if requested
            if print_to_file {
                let tmp_dir = Path::new("tmp");
                if !tmp_dir.exists() {
                    match fs::create_dir_all(tmp_dir) {
                        Ok(_) => tracing::info!("Debug   : Created directory: {:?}", tmp_dir),
                        Err(e) => {
                            eprintln!("Failed to create directory {tmp_dir:?}: {e}");
                            std::process::exit(1);
                        }
                    }
                }

                file_path = tmp_dir.join("debug.log");

                match File::create(&file_path) {
                    Ok(file) => {
                        output = Box::new(file);
                    }
                    Err(e) => {
                        eprintln!("Failed to create log file at {file_path:?}: {e}");
                        std::process::exit(1);
                    }
                }
            }

            let file_msg =
                if print_to_file { format!(" Check the {file_path:?} file for more details.") } else { "".to_string() };
            tracing::error!("Some bus values do not match.{}", file_msg);

            // Set the flag to avoid printing the error message multiple times
            there_are_errors = true;
        }
        writeln!(output, "\t► Mismatched bus values for opid {opid}:").expect("Write error");

        let (overassumed_values, overproven_values): (Vec<_>, Vec<_>) = bus
            .into_par_iter()
            .filter(|(_, v)| v.shared_data.num_proves != v.shared_data.num_assumes)
            .partition(|(_, v)| v.shared_data.num_proves < v.shared_data.num_assumes);

        let len_overassumed = overassumed_values.len();
        let len_overproven = overproven_values.len();

        if len_overassumed > 0 {
            writeln!(output, "\t  ⁃ There are {len_overassumed} unmatching values thrown as 'assume':")
                .expect("Write error");
        }

        for (i, (val, data)) in overassumed_values.iter().enumerate() {
            if i == max_values_to_print {
                writeln!(output, "\t      ...").expect("Write error");
                break;
            }
            let shared_data = &data.shared_data;
            let bus_data = bus_info.as_ref().and_then(|info| info.get(val));
            print_diffs(pctx, sctx, max_values_to_print, shared_data, bus_data, false, *val, &mut output)?;
        }

        if len_overassumed > 0 {
            writeln!(output).expect("Write error");
        }

        if len_overproven > 0 {
            writeln!(output, "\t  ⁃ There are {len_overproven} unmatching values thrown as 'prove':")
                .expect("Write error");
        }

        for (i, (val, data)) in overproven_values.iter().enumerate() {
            if i == max_values_to_print {
                writeln!(output, "\t      ...").expect("Write error");
                break;
            }

            let shared_data = &data.shared_data;
            let bus_data = bus_info.as_ref().and_then(|info| info.get(val));
            print_diffs(pctx, sctx, max_values_to_print, shared_data, bus_data, true, *val, &mut output)?;
        }

        if len_overproven > 0 {
            writeln!(output).expect("Write error");
        }
    }

    /// Parses decimal value string and formats as hexadecimal and binary
    /// Handles both simple values "1,2,3" and extended field format "[1,2,3]"
    fn parse_and_format_values(val_str: &str) -> (String, String) {
        let mut hex_parts = Vec::new();
        let mut bin_parts = Vec::new();

        // Check if it's an extended field (starts with '[')
        if val_str.starts_with('[') && val_str.ends_with(']') {
            // Extended field format: "[val1,val2,...]"
            let inner = &val_str[1..val_str.len() - 1];
            let parts: Vec<&str> = inner.split(',').collect();

            for part in parts {
                if let Ok(num) = part.trim().parse::<u64>() {
                    hex_parts.push(format!("0x{:x}", num));
                    bin_parts.push(format!("0b{:b}", num));
                } else {
                    // If parsing fails, keep the original
                    hex_parts.push(part.to_string());
                    bin_parts.push(part.to_string());
                }
            }

            (format!("[{}]", hex_parts.join(",")), format!("[{}]", bin_parts.join(",")))
        } else {
            // Simple format: "val1,val2,..."
            let parts: Vec<&str> = val_str.split(',').collect();

            for part in parts {
                if let Ok(num) = part.trim().parse::<u64>() {
                    hex_parts.push(format!("0x{:x}", num));
                    bin_parts.push(format!("0b{:b}", num));
                } else {
                    // If parsing fails, keep the original
                    hex_parts.push(part.to_string());
                    bin_parts.push(part.to_string());
                }
            }

            (hex_parts.join(","), bin_parts.join(","))
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn print_diffs<F: PrimeField64>(
        pctx: &ProofCtx<F>,
        sctx: &SetupCtx<F>,
        max_values_to_print: usize,
        shared_data: &SharedData,
        bus_data: Option<&BusValueInfo>,
        proves: bool,
        hash: u64,
        output: &mut dyn Write,
    ) -> ProofmanResult<()> {
        let val = &shared_data.vals;
        let num_assumes = shared_data.num_assumes;
        let num_proves = shared_data.num_proves;

        let num = if proves { num_proves } else { num_assumes };
        let num_str = if num != 1 { "times" } else { "time" };

        // Parse and format values in different bases
        let (vals_hex, _) = parse_and_format_values(val);

        writeln!(output, "\t    ==================================================").expect("Write error");
        writeln!(output, "\t    • Value (decimal): [{}]", val).expect("Write error");
        writeln!(output, "\t      Value (hex):     [{}]", vals_hex).expect("Write error");
        writeln!(output, "\t      Hash:            0x{:016x}", hash).expect("Write error");
        writeln!(output, "\t      Appears {} {} across the following:", num, num_str).expect("Write error");

        // Print global data first (if it exists)
        if let Some(bus_info) = bus_data {
            if let Some(global_data) = &bus_info.global_data {
                let gprod_debug_data_global = get_hint_ids_by_name(sctx.get_global_bin(), "gprod_debug_data_global");
                let gsum_debug_data_global = get_hint_ids_by_name(sctx.get_global_bin(), "gsum_debug_data_global");

                let (airgroup_id, hint_id, is_prod) = global_data.unpack();
                let airgroup_name = pctx.global_info.get_air_group_name(airgroup_id as usize);
                writeln!(output, "\t        - Airgroup: {airgroup_name} (id: {airgroup_id})").expect("Write error");

                let name_piop = match is_prod {
                    true => get_global_hint_field_constant_as_string(
                        sctx,
                        gprod_debug_data_global[1 + hint_id as usize],
                        "name_piop",
                    )?,
                    false => get_global_hint_field_constant_as_string(
                        sctx,
                        gsum_debug_data_global[1 + hint_id as usize],
                        "name_piop",
                    )?,
                };

                let name_exprs = match is_prod {
                    true => get_global_hint_field_constant_a_as_string(
                        sctx,
                        gprod_debug_data_global[1 + hint_id as usize],
                        "name_exprs",
                    )?,
                    false => get_global_hint_field_constant_a_as_string(
                        sctx,
                        gsum_debug_data_global[1 + hint_id as usize],
                        "name_exprs",
                    )?,
                };

                writeln!(output, "\t          PIOP: {}", name_piop).expect("Write error");
                writeln!(output, "\t          Expression: {:?}", name_exprs).expect("Write error");
                writeln!(output, "\t          Num: 1").expect("Write error");
            }

            // Print local data
            if !bus_info.local_data.is_empty() {
                // Parallel collection and organization of rows
                let mut organized_rows: Vec<(usize, usize, usize, usize, bool, Vec<usize>)> = bus_info
                    .local_data
                    .par_iter()
                    .filter_map(|(key, meta_data)| {
                        let row = if proves { &meta_data.row_proves } else { &meta_data.row_assumes };
                        if row.is_empty() {
                            None
                        } else {
                            let (airgroup_id, air_id, instance_id, hint_id, is_prod) = key.unpack();
                            Some((
                                airgroup_id as usize,
                                air_id as usize,
                                instance_id as usize,
                                hint_id as usize,
                                is_prod,
                                row.clone(),
                            ))
                        }
                    })
                    .collect();

                // Sort rows by airgroup_id, air_id, and instance_id
                organized_rows.sort_by(|a, b| a.0.cmp(&b.0).then(a.1.cmp(&b.1)).then(a.2.cmp(&b.2)));

                // Print grouped rows
                for (airgroup_id, air_id, instance_id, hint_id, is_prod, mut rows) in organized_rows {
                    let airgroup_name = pctx.global_info.get_air_group_name(airgroup_id);
                    let air_name = pctx.global_info.get_air_name(airgroup_id, air_id);

                    let setup = sctx.get_setup(airgroup_id, air_id)?;
                    let p_expressions_bin = setup.p_setup.p_expressions_bin;

                    let debug_data_hints_prod = get_hint_ids_by_name(p_expressions_bin, "gprod_debug_data");

                    let debug_data_hints_sum = get_hint_ids_by_name(p_expressions_bin, "gsum_debug_data");

                    let hint = match is_prod {
                        true => debug_data_hints_prod[hint_id],
                        false => debug_data_hints_sum[hint_id],
                    };

                    let piop_name = get_hint_field_constant_as_string(
                        pctx,
                        setup,
                        airgroup_id,
                        air_id,
                        hint as usize,
                        "name_piop",
                        HintFieldOptions::default(),
                    )?;

                    let expr_name = get_hint_field_constant_a_as_string(
                        pctx,
                        setup,
                        airgroup_id,
                        air_id,
                        hint as usize,
                        "name_exprs",
                        HintFieldOptions::default(),
                    )?;

                    rows.sort_unstable();
                    let rows_display =
                        rows.iter().take(max_values_to_print).map(|x| x.to_string()).collect::<Vec<_>>().join(",");

                    let truncated = rows.len() > max_values_to_print;
                    writeln!(output, "\t        - Airgroup: {airgroup_name} (id: {airgroup_id})").expect("Write error");
                    writeln!(output, "\t          Air: {air_name} (id: {air_id})").expect("Write error");

                    writeln!(output, "\t          PIOP: {piop_name}").expect("Write error");
                    writeln!(output, "\t          Expression: {expr_name:?}").expect("Write error");

                    writeln!(
                        output,
                        "\t          Instance ID: {} | Hint ID: {} | Num: {} | Rows: [{}{}]",
                        instance_id,
                        hint_id,
                        rows.len(),
                        rows_display,
                        if truncated { ",..." } else { "" }
                    )
                    .expect("Write error");
                }
            }
        }

        writeln!(output, "\t    --------------------------------------------------").expect("Write error");
        let diff = if proves { num_proves - num_assumes } else { num_assumes - num_proves };
        writeln!(
        output,
        "\t    Total Num Assumes: {num_assumes}.\n\t    Total Num Proves: {num_proves}.\n\t    Total Unmatched: {diff}."
    )
        .expect("Write error");
        writeln!(output, "\t    ==================================================\n").expect("Write error");

        Ok(())
    }

    timer_stop_and_log_info!(PRINT_DEBUG_INFO);

    Ok(())
}
