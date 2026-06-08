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
use proofman_common::{
    find_bucket_rule, BucketRule, BusBucket, BusSection, BusValueGlobalOrigin, BusValueLocalOrigin, BusValueMismatch,
    Classifier, ProofCtx, ProofmanResult,
};
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

pub type DebugData = FxHashMap<u64, FxHashMap<u64, FxHashMap<u64, BusValue>>>; // opid -> bucket_key -> val -> SharedData
pub type DebugDataInfo = FxHashMap<u64, FxHashMap<u64, FxHashMap<u64, BusValueInfo>>>;

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
    bucket_key: u64,
    norm_vals: &[HintFieldOutput<F>],
    hash: u64,
    airgroup_id: usize,
    is_proves: bool,
    times: u64,
    is_prod: bool,
) -> ProofmanResult<()> {
    let bus_opid = debug_data.entry(opid).or_default();
    let bus_bucket = bus_opid.entry(bucket_key).or_default();
    let bus_val = bus_bucket.entry(hash).or_insert_with(|| BusValue {
        shared_data: SharedData {
            vals: format_hint_field_output_vec(norm_vals).to_string(),
            num_proves: 0,
            num_assumes: 0,
        },
    });

    let bus_info_opid = debug_data_info.entry(opid).or_default();
    let bus_info_bucket = bus_info_opid.entry(bucket_key).or_default();
    let bus_info_val = bus_info_bucket
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
    bucket_key: u64,
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
    let bus_val =
        debug_data.entry(opid).or_default().entry(bucket_key).or_default().entry(hash).or_insert_with(|| BusValue {
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
        let bus_info_bucket = bus_info_opid.entry(bucket_key).or_default();
        let bus_info_val = bus_info_bucket
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
    bucket_key: u64,
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
            bucket_key,
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
            bucket_key,
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

#[allow(clippy::too_many_arguments)]
pub fn print_debug_info<F: PrimeField64>(
    pctx: &ProofCtx<F>,
    sctx: &SetupCtx<F>,
    max_values_to_print: usize,
    print_to_file: bool,
    output_file_path: &Path,
    debug_data: &mut DebugData,
    debug_data_info: &mut DebugDataInfo,
    is_prod: bool,
) -> ProofmanResult<()> {
    let label = if is_prod { "std_prod" } else { "std_sum" };
    timer_start_info!(PRINT_DEBUG_INFO);
    let mut file_path = PathBuf::new();
    let mut output: Box<dyn Write> = Box::new(io::stdout());
    let mut there_are_errors = false;

    let group_by = pctx.debug_info.read().unwrap().bus_mode.group_by.clone();

    // Parallel pre-filtering: collect only mismatched opids
    let mismatched_opids: Vec<_> = debug_data
        .par_iter()
        .filter_map(|(opid, buckets)| {
            let has_mismatch = buckets
                .iter()
                .any(|(_, bus)| bus.iter().any(|(_, v)| v.shared_data.num_proves != v.shared_data.num_assumes));
            if has_mismatch {
                Some(*opid)
            } else {
                None
            }
        })
        .collect();

    // Early exit if no errors
    if mismatched_opids.is_empty() {
        tracing::info!("··· {}", format!("\u{2713} [{label}] All bus values match.").bright_green().bold());
        timer_stop_and_log_info!(PRINT_DEBUG_INFO);
        return Ok(());
    }

    // Process mismatched opids serially for ordered output, consuming entries as we go
    for opid in mismatched_opids {
        let opid_buckets = debug_data.remove(&opid).unwrap();
        let opid_info_buckets = debug_data_info.remove(&opid);
        let opid_rule = find_bucket_rule(&group_by, opid);

        if !there_are_errors {
            // Print to a file if requested
            if print_to_file {
                if let Some(parent) = output_file_path.parent() {
                    if !parent.as_os_str().is_empty() && !parent.exists() {
                        match fs::create_dir_all(parent) {
                            Ok(_) => tracing::info!("Debug   : Created directory: {:?}", parent),
                            Err(e) => {
                                eprintln!("Failed to create directory {parent:?}: {e}");
                                std::process::exit(1);
                            }
                        }
                    }
                }

                file_path = output_file_path.to_path_buf();

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

            // Two-line format: header in bold red so it stands out at a glance,
            // detail (file pointer) on its own line below.
            tracing::error!("··· {}", format!("\u{2717} [{label}] Some bus values do not match.").bright_red().bold());
            if print_to_file {
                tracing::error!("··· {}", format!("Check the {file_path:?} file for more details.").bright_red());
            }

            // Set the flag to avoid printing the error message multiple times
            there_are_errors = true;
        }

        // Build the per-opid body into a buffer for file/stdout AND assemble a structured
        // BusSection for the in-memory report — both come from the same data, no double work.
        let mut opid_buf: Vec<u8> = Vec::new();
        let mut num_overassumed_total = 0usize;
        let mut num_overproven_total = 0usize;
        let mut section_buckets: Vec<BusBucket> = Vec::new();

        writeln!(opid_buf, "\t► Mismatched bus values for opid {opid}:").expect("Write error");

        // Iterate over buckets within the opid. For unbucketed opids, only key 0 is present.
        let mut bucket_keys: Vec<u64> = opid_buckets.keys().copied().collect();
        bucket_keys.sort_unstable();

        let mut opid_buckets = opid_buckets;
        let mut opid_info_buckets = opid_info_buckets;

        for bucket_key in bucket_keys {
            let bus = opid_buckets.remove(&bucket_key).unwrap();
            let bus_info = opid_info_buckets.as_mut().and_then(|m| m.remove(&bucket_key));

            // Skip buckets with no mismatch.
            let has_mismatch = bus.values().any(|v| v.shared_data.num_proves != v.shared_data.num_assumes);
            if !has_mismatch {
                continue;
            }

            let bucket_label = opid_rule.map(|rule| format_bucket_desc(rule, bucket_key));
            if let Some(label) = &bucket_label {
                writeln!(opid_buf, "\t  ◆ Bucket: {}", label).expect("Write error");
            }

            let (overassumed_values, overproven_values): (Vec<_>, Vec<_>) = bus
                .into_par_iter()
                .filter(|(_, v)| v.shared_data.num_proves != v.shared_data.num_assumes)
                .partition(|(_, v)| v.shared_data.num_proves < v.shared_data.num_assumes);

            let len_overassumed = overassumed_values.len();
            let len_overproven = overproven_values.len();
            num_overassumed_total += len_overassumed;
            num_overproven_total += len_overproven;

            let mut bucket_overassumed: Vec<BusValueMismatch> = Vec::with_capacity(len_overassumed);
            let mut bucket_overproven: Vec<BusValueMismatch> = Vec::with_capacity(len_overproven);

            if len_overassumed > 0 {
                writeln!(opid_buf, "\t  ⁃ There are {len_overassumed} unmatching values thrown as 'assume':")
                    .expect("Write error");
            }

            for (i, (val, data)) in overassumed_values.iter().enumerate() {
                let shared_data = &data.shared_data;
                let bus_data = bus_info.as_ref().and_then(|info| info.get(val));
                let mismatch = build_value_mismatch(pctx, sctx, shared_data, bus_data, false, *val)?;
                if i < max_values_to_print {
                    write_value_mismatch(&mismatch, false, max_values_to_print, &mut opid_buf);
                } else if i == max_values_to_print {
                    writeln!(opid_buf, "\t      ...").expect("Write error");
                }
                bucket_overassumed.push(mismatch);
            }

            if len_overassumed > 0 {
                writeln!(opid_buf).expect("Write error");
            }

            if len_overproven > 0 {
                writeln!(opid_buf, "\t  ⁃ There are {len_overproven} unmatching values thrown as 'prove':")
                    .expect("Write error");
            }

            for (i, (val, data)) in overproven_values.iter().enumerate() {
                let shared_data = &data.shared_data;
                let bus_data = bus_info.as_ref().and_then(|info| info.get(val));
                let mismatch = build_value_mismatch(pctx, sctx, shared_data, bus_data, true, *val)?;
                if i < max_values_to_print {
                    write_value_mismatch(&mismatch, true, max_values_to_print, &mut opid_buf);
                } else if i == max_values_to_print {
                    writeln!(opid_buf, "\t      ...").expect("Write error");
                }
                bucket_overproven.push(mismatch);
            }

            if len_overproven > 0 {
                writeln!(opid_buf).expect("Write error");
            }

            section_buckets.push(BusBucket {
                bucket_key,
                bucket_label,
                overassumed: bucket_overassumed,
                overproven: bucket_overproven,
            });
        }

        // Tee: write the assembled text to the file/stdout sink and push the structured
        // section into the report. Same data, two consumers.
        output.write_all(&opid_buf).expect("Write error");
        pctx.debug_report.write().unwrap().push(BusSection {
            opid,
            mismatched: true,
            num_overassumed: num_overassumed_total,
            num_overproven: num_overproven_total,
            buckets: section_buckets,
        });
    }

    fn format_bucket_desc(rule: &BucketRule, bucket_key: u64) -> String {
        match &rule.classifier {
            Classifier::Value { .. } => format!("col[{}] = 0x{bucket_key:x}", rule.column),
            Classifier::Range { ranges, .. } => {
                let idx = bucket_key as usize;
                let r = &ranges[idx];
                let lo = r.min.map(|v| format!("0x{v:x}")).unwrap_or_else(|| "-∞".to_string());
                let hi = r.max.map(|v| format!("0x{v:x}")).unwrap_or_else(|| "+∞".to_string());
                format!("col[{}] in [{lo}, {hi})", rule.column)
            }
            Classifier::Prefix { prefixes, .. } => {
                let idx = bucket_key as usize;
                if idx < prefixes.len() {
                    let p = &prefixes[idx];
                    format!("col[{}] starts with 0x{:x} ({} bits)", rule.column, p.value, p.bits)
                } else {
                    format!("col[{}] matches no prefix", rule.column)
                }
            }
            Classifier::Step { start, stop, step, .. } => {
                let oor = crate::step_oor_index(*start, *stop, *step);
                if bucket_key == oor {
                    format!("col[{}] outside [0x{start:x}, 0x{stop:x})", rule.column)
                } else {
                    let lo = start + bucket_key * step;
                    let hi = lo.saturating_add(*step).min(*stop);
                    format!("col[{}] in [0x{lo:x}, 0x{hi:x})", rule.column)
                }
            }
        }
    }

    fn build_value_mismatch<F: PrimeField64>(
        pctx: &ProofCtx<F>,
        sctx: &SetupCtx<F>,
        shared_data: &SharedData,
        bus_data: Option<&BusValueInfo>,
        proves: bool,
        hash: u64,
    ) -> ProofmanResult<BusValueMismatch> {
        // SharedData stores the value tuple as a comma-separated decimal String to keep
        // its in-memory footprint small (one record per occurrence, many occurrences).
        // Parse to Vec<u64> here, at the report boundary — once per mismatched value.
        let vals: Vec<u64> = shared_data.vals.split(',').filter_map(|p| p.trim().parse::<u64>().ok()).collect();

        let mut global_origin: Option<BusValueGlobalOrigin> = None;
        let mut local_origins: Vec<BusValueLocalOrigin> = Vec::new();

        if let Some(bus_info) = bus_data {
            if let Some(global_data) = &bus_info.global_data {
                let gprod_debug_data_global = get_hint_ids_by_name(sctx.get_global_bin(), "gprod_debug_data_global");
                let gsum_debug_data_global = get_hint_ids_by_name(sctx.get_global_bin(), "gsum_debug_data_global");

                let (airgroup_id, hint_id, is_prod) = global_data.unpack();
                let airgroup_name = pctx.global_info.get_air_group_name(airgroup_id as usize).to_string();

                let hint = if is_prod {
                    gprod_debug_data_global[1 + hint_id as usize]
                } else {
                    gsum_debug_data_global[1 + hint_id as usize]
                };
                let piop_name = get_global_hint_field_constant_as_string(sctx, hint, "name_piop")?;
                let expression_names = get_global_hint_field_constant_a_as_string(sctx, hint, "name_exprs")?;

                global_origin = Some(BusValueGlobalOrigin {
                    airgroup_id: airgroup_id as usize,
                    airgroup_name,
                    piop_name,
                    expression_names,
                    is_prod,
                });
            }

            if !bus_info.local_data.is_empty() {
                let mut organized: Vec<(usize, usize, usize, usize, bool, Vec<usize>)> = bus_info
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

                organized.sort_by(|a, b| a.0.cmp(&b.0).then(a.1.cmp(&b.1)).then(a.2.cmp(&b.2)));

                for (airgroup_id, air_id, instance_id, hint_id, is_prod, mut rows) in organized {
                    let airgroup_name = pctx.global_info.get_air_group_name(airgroup_id).to_string();
                    let air_name = pctx.global_info.get_air_name(airgroup_id, air_id).to_string();

                    let setup = sctx.get_setup(airgroup_id, air_id)?;
                    let p_expressions_bin = setup.p_setup.p_expressions_bin;
                    let debug_data_hints_prod = get_hint_ids_by_name(p_expressions_bin, "gprod_debug_data");
                    let debug_data_hints_sum = get_hint_ids_by_name(p_expressions_bin, "gsum_debug_data");

                    let hint = if is_prod { debug_data_hints_prod[hint_id] } else { debug_data_hints_sum[hint_id] };

                    let piop_name = get_hint_field_constant_as_string(
                        pctx,
                        setup,
                        airgroup_id,
                        air_id,
                        hint as usize,
                        "name_piop",
                        HintFieldOptions::default(),
                    )?;
                    let expression_names = get_hint_field_constant_a_as_string(
                        pctx,
                        setup,
                        airgroup_id,
                        air_id,
                        hint as usize,
                        "name_exprs",
                        HintFieldOptions::default(),
                    )?;

                    rows.sort_unstable();
                    local_origins.push(BusValueLocalOrigin {
                        airgroup_id,
                        airgroup_name,
                        air_id,
                        air_name,
                        instance_id,
                        hint_id,
                        piop_name,
                        expression_names,
                        is_prod,
                        rows,
                    });
                }
            }
        }

        Ok(BusValueMismatch {
            vals,
            hash,
            num_assumes: shared_data.num_assumes,
            num_proves: shared_data.num_proves,
            global_origin,
            local_origins,
        })
    }

    fn write_value_mismatch(
        mismatch: &BusValueMismatch,
        proves: bool,
        max_values_to_print: usize,
        output: &mut dyn Write,
    ) {
        let num = if proves { mismatch.num_proves } else { mismatch.num_assumes };
        let num_str = if num != 1 { "times" } else { "time" };

        let vals_decimal = mismatch.vals.iter().map(|v| v.to_string()).collect::<Vec<_>>().join(",");
        let vals_hex = mismatch.vals.iter().map(|v| format!("0x{:x}", v)).collect::<Vec<_>>().join(",");
        writeln!(output, "\t    ==================================================").expect("Write error");
        writeln!(output, "\t    • Value (decimal): [{}]", vals_decimal).expect("Write error");
        writeln!(output, "\t      Value (hex):     [{}]", vals_hex).expect("Write error");
        writeln!(output, "\t      Hash:            0x{:016x}", mismatch.hash).expect("Write error");
        writeln!(output, "\t      Appears {} {} across the following:", num, num_str).expect("Write error");

        if let Some(g) = &mismatch.global_origin {
            writeln!(output, "\t        - Airgroup: {} (id: {})", g.airgroup_name, g.airgroup_id).expect("Write error");
            writeln!(output, "\t          PIOP: {}", g.piop_name).expect("Write error");
            writeln!(output, "\t          Expression: {:?}", g.expression_names).expect("Write error");
            writeln!(output, "\t          Num: 1").expect("Write error");
        }

        for origin in &mismatch.local_origins {
            let rows_display =
                origin.rows.iter().take(max_values_to_print).map(|x| x.to_string()).collect::<Vec<_>>().join(",");
            let truncated = origin.rows.len() > max_values_to_print;
            writeln!(output, "\t        - Airgroup: {} (id: {})", origin.airgroup_name, origin.airgroup_id)
                .expect("Write error");
            writeln!(output, "\t          Air: {} (id: {})", origin.air_name, origin.air_id).expect("Write error");
            writeln!(output, "\t          PIOP: {}", origin.piop_name).expect("Write error");
            writeln!(output, "\t          Expression: {:?}", origin.expression_names).expect("Write error");
            writeln!(
                output,
                "\t          Instance ID: {} | Hint ID: {} | Num: {} | Rows: [{}{}]",
                origin.instance_id,
                origin.hint_id,
                origin.rows.len(),
                rows_display,
                if truncated { ",..." } else { "" }
            )
            .expect("Write error");
        }

        writeln!(output, "\t    --------------------------------------------------").expect("Write error");
        let diff = if proves {
            mismatch.num_proves - mismatch.num_assumes
        } else {
            mismatch.num_assumes - mismatch.num_proves
        };
        writeln!(
            output,
            "\t    Total Num Assumes: {}.\n\t    Total Num Proves: {}.\n\t    Total Unmatched: {diff}.",
            mismatch.num_assumes, mismatch.num_proves
        )
        .expect("Write error");
        writeln!(output, "\t    ==================================================\n").expect("Write error");
    }

    timer_stop_and_log_info!(PRINT_DEBUG_INFO);

    Ok(())
}
