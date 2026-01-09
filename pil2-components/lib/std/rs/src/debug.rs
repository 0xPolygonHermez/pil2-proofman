use std::{
    fs::{self, File},
    io::{self, Write},
    path::{Path, PathBuf},
};

use rustc_hash::FxHashMap;

use proofman_common::SetupCtx;

use colored::Colorize;
use fields::PrimeField64;
use proofman_common::{ProofCtx, ProofmanError, ProofmanResult};
use proofman_hints::{
    get_hint_ids_by_name, format_hint_field_output_vec, HintFieldOutput, HintFieldValue, HintFieldValuesVec,
    HintFieldOptions,
};

use crate::{
    get_global_hint_field_constant_a_as_string, get_global_hint_field_constant_as_string,
    get_hint_field_constant_a_as_string, get_hint_field_constant_as_string,
};

use crate::normalize_vals;

pub type DebugData<F> = FxHashMap<F, FxHashMap<Vec<HintFieldOutput<F>>, BusValue<F>>>; // opid -> val -> BusValue

#[derive(Clone)]
pub struct HintMetadata<F: PrimeField64> {
    pub hint: u64,
    pub hint_id: usize,
    pub busid: HintFieldValue<F>,
    pub type_piop: u64,
    pub num_reps: HintFieldValue<F>,
    pub expressions: HintFieldValuesVec<F>,
    pub deg_expr: F,
    pub deg_mul: F,
}

#[derive(Debug)]
pub struct BusValue<F> {
    shared_data: SharedData<F>,     // Data shared across all airgroups, airs, and instances
    local_data: AirGroupMap,        // Data grouped by: airgroup_id -> air_id -> AirData -> instance_id -> InstanceData
    global_data: GlobalAirGroupMap, // Data grouped by: airgroup_id -> AirGroupData (for global operations)
}

#[derive(Debug)]
struct SharedData<F> {
    direct_was_called: bool,
    num_proves: F,
    num_assumes: F,
}

type AirGroupMap = FxHashMap<usize, AirMap>;
type AirMap = FxHashMap<usize, AirData>;

#[derive(Debug)]
struct AirData {
    hint_id: usize,
    is_prod: bool,
    instances: InstanceMap,
}

type InstanceMap = FxHashMap<usize, InstanceData>;

#[derive(Debug)]
struct InstanceData {
    row_proves: Vec<usize>,
    row_assumes: Vec<usize>,
}

type GlobalAirGroupMap = FxHashMap<usize, AirGroupData>;

#[derive(Debug)]
struct AirGroupData {
    hint_id: usize,
    is_prod: bool,
}

/// Handle global debug data updates (shared across all instances)
#[allow(clippy::too_many_arguments)]
pub fn update_global_debug_data<F: PrimeField64>(
    debug_data: &mut DebugData<F>,
    hint_id: usize,
    opid: F,
    vals: Vec<HintFieldOutput<F>>,
    airgroup_id: usize,
    is_proves: bool,
    times: F,
    is_prod: bool,
) -> ProofmanResult<()> {
    let bus_opid = debug_data.entry(opid).or_default();
    let norm_vals = normalize_vals(&vals);
    let bus_val = bus_opid.entry(norm_vals).or_insert_with(|| BusValue {
        shared_data: SharedData { direct_was_called: false, num_proves: F::ZERO, num_assumes: F::ZERO },
        local_data: AirGroupMap::default(),
        global_data: GlobalAirGroupMap::default(),
    });

    // Skip if already processed
    if bus_val.shared_data.direct_was_called {
        return Ok(());
    }

    bus_val.shared_data.direct_was_called = true;

    // Store global data for this airgroup
    bus_val.global_data.entry(airgroup_id).or_insert_with(|| AirGroupData { hint_id, is_prod });

    if is_proves {
        bus_val.shared_data.num_proves += times;
    } else {
        if !times.is_one() {
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
    debug_data: &mut DebugData<F>,
    hint_id: usize,
    opid: F,
    vals: Vec<HintFieldOutput<F>>,
    airgroup_id: usize,
    air_id: usize,
    instance_id: usize,
    row: usize,
    is_proves: bool,
    times: F,
    is_prod: bool,
) -> ProofmanResult<()> {
    let bus_opid = debug_data.entry(opid).or_default();
    let norm_vals = normalize_vals(&vals);
    let bus_val = bus_opid.entry(norm_vals).or_insert_with(|| BusValue {
        shared_data: SharedData { direct_was_called: false, num_proves: F::ZERO, num_assumes: F::ZERO },
        local_data: AirGroupMap::default(),
        global_data: GlobalAirGroupMap::default(),
    });

    let local_data = bus_val
        .local_data
        .entry(airgroup_id)
        .or_default()
        .entry(air_id)
        .or_insert_with(|| AirData { hint_id, is_prod, instances: InstanceMap::default() })
        .instances
        .entry(instance_id)
        .or_insert_with(|| InstanceData { row_proves: Vec::new(), row_assumes: Vec::new() });

    // Update shared counters
    if is_proves {
        bus_val.shared_data.num_proves += times;
        local_data.row_proves.push(row);
    } else {
        bus_val.shared_data.num_assumes += times;
        local_data.row_assumes.push(row);
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub fn update_debug_data<F: PrimeField64>(
    debug_data: &mut DebugData<F>,
    hint_id: usize,
    opid: F,
    vals: Vec<HintFieldOutput<F>>,
    airgroup_id: usize,
    air_id: Option<usize>,
    instance_id: Option<usize>,
    row: usize,
    is_proves: bool,
    times: F,
    is_global: bool,
    is_prod: bool,
) -> ProofmanResult<()> {
    if is_global {
        update_global_debug_data(debug_data, hint_id, opid, vals, airgroup_id, is_proves, times, is_prod)
    } else {
        update_local_debug_data(
            debug_data,
            hint_id,
            opid,
            vals,
            airgroup_id,
            air_id.unwrap(),
            instance_id.unwrap(),
            row,
            is_proves,
            times,
            is_prod,
        )
    }
}

pub fn print_debug_info<F: PrimeField64>(
    pctx: &ProofCtx<F>,
    sctx: &SetupCtx<F>,
    max_values_to_print: usize,
    print_to_file: bool,
    debug_data: &mut DebugData<F>,
) -> ProofmanResult<()> {
    let mut file_path = PathBuf::new();
    let mut output: Box<dyn Write> = Box::new(io::stdout());
    let mut there_are_errors = false;
    for (opid, bus) in debug_data.iter_mut() {
        if bus.iter().any(|(_, v)| v.shared_data.num_proves != v.shared_data.num_assumes) {
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

                let file_msg = if print_to_file {
                    format!(" Check the {file_path:?} file for more details.")
                } else {
                    "".to_string()
                };
                tracing::error!("Some bus values do not match.{}", file_msg);

                // Set the flag to avoid printing the error message multiple times
                there_are_errors = true;
            }
            writeln!(output, "\t► Mismatched bus values for opid {opid}:").expect("Write error");
        } else {
            continue;
        }

        // TODO: Sort unmatching values by the row
        let mut overassumed_values: Vec<(&Vec<HintFieldOutput<F>>, &mut BusValue<F>)> =
            bus.iter_mut().filter(|(_, v)| v.shared_data.num_proves < v.shared_data.num_assumes).collect();
        let len_overassumed = overassumed_values.len();

        if len_overassumed > 0 {
            writeln!(output, "\t  ⁃ There are {len_overassumed} unmatching values thrown as 'assume':")
                .expect("Write error");
        }

        for (i, (val, data)) in overassumed_values.iter_mut().enumerate() {
            if i == max_values_to_print {
                writeln!(output, "\t      ...").expect("Write error");
                break;
            }
            let shared_data = &data.shared_data;
            let local_data = &mut data.local_data;
            let global_data = &data.global_data;
            print_diffs(
                pctx,
                sctx,
                val,
                max_values_to_print,
                shared_data,
                local_data,
                global_data,
                false,
                &mut output,
            )?;
        }

        if len_overassumed > 0 {
            writeln!(output).expect("Write error");
        }

        // TODO: Sort unmatching values by the row
        let mut overproven_values: Vec<(&Vec<HintFieldOutput<F>>, &mut BusValue<F>)> =
            bus.iter_mut().filter(|(_, v)| v.shared_data.num_proves > v.shared_data.num_assumes).collect();
        let len_overproven = overproven_values.len();

        if len_overproven > 0 {
            writeln!(output, "\t  ⁃ There are {len_overproven} unmatching values thrown as 'prove':")
                .expect("Write error");
        }

        for (i, (val, data)) in overproven_values.iter_mut().enumerate() {
            if i == max_values_to_print {
                writeln!(output, "\t      ...").expect("Write error");
                break;
            }

            let shared_data = &data.shared_data;
            let local_data = &mut data.local_data;
            let global_data = &data.global_data;
            print_diffs(pctx, sctx, val, max_values_to_print, shared_data, local_data, global_data, true, &mut output)?;
        }

        if len_overproven > 0 {
            writeln!(output).expect("Write error");
        }
    }

    if !there_are_errors {
        tracing::info!("··· {}", "\u{2713} All bus values match.".bright_green().bold());
    }

    #[allow(clippy::too_many_arguments)]
    fn print_diffs<F: PrimeField64>(
        pctx: &ProofCtx<F>,
        sctx: &SetupCtx<F>,
        val: &[HintFieldOutput<F>],
        max_values_to_print: usize,
        shared_data: &SharedData<F>,
        local_data: &mut AirGroupMap,
        global_data: &GlobalAirGroupMap,
        proves: bool,
        output: &mut dyn Write,
    ) -> ProofmanResult<()> {
        let num_assumes = shared_data.num_assumes;
        let num_proves = shared_data.num_proves;

        let num = if proves { num_proves } else { num_assumes };
        let num_str = if num.is_one() { "time" } else { "times" };

        writeln!(output, "\t    ==================================================").expect("Write error");
        writeln!(
            output,
            "\t    • Value:\n\t        {}\n\t      Appears {} {} across the following:",
            format_hint_field_output_vec(val),
            num,
            num_str,
        )
        .expect("Write error");

        // Print global data first
        let gprod_debug_data_global = get_hint_ids_by_name(sctx.get_global_bin(), "gprod_debug_data_global");

        let gsum_debug_data_global = get_hint_ids_by_name(sctx.get_global_bin(), "gsum_debug_data_global");

        for (airgroup_id, airgroup_data) in global_data.iter() {
            let airgroup_name = pctx.global_info.get_air_group_name(*airgroup_id);
            writeln!(output, "\t        - Airgroup: {airgroup_name} (id: {airgroup_id})").expect("Write error");
            let name_piop = match airgroup_data.is_prod {
                true => get_global_hint_field_constant_as_string(
                    sctx,
                    gprod_debug_data_global[1 + airgroup_data.hint_id],
                    "name_piop",
                )?,
                false => get_global_hint_field_constant_as_string(
                    sctx,
                    gsum_debug_data_global[1 + airgroup_data.hint_id],
                    "name_piop",
                )?,
            };

            let name_exprs = match airgroup_data.is_prod {
                true => get_global_hint_field_constant_a_as_string(
                    sctx,
                    gprod_debug_data_global[1 + airgroup_data.hint_id],
                    "name_exprs",
                )?,
                false => get_global_hint_field_constant_a_as_string(
                    sctx,
                    gsum_debug_data_global[1 + airgroup_data.hint_id],
                    "name_exprs",
                )?,
            };

            writeln!(output, "\t          PIOP: {}", name_piop).expect("Write error");
            writeln!(output, "\t          Expression: {:?}", name_exprs).expect("Write error");
            writeln!(output, "\t          Num: 1").expect("Write error");
        }

        // Print local data next

        // Collect and organize rows
        let mut organized_rows = Vec::new();
        for (airgroup_id, air_id_map) in local_data.iter_mut() {
            for (air_id, air_data) in air_id_map.iter_mut() {
                for (instance_id, meta_data) in air_data.instances.iter_mut() {
                    let rows = {
                        let rows = if proves { &meta_data.row_proves } else { &meta_data.row_assumes };
                        if rows.is_empty() {
                            continue;
                        }
                        rows.clone()
                    };
                    organized_rows.push((*airgroup_id, *air_id, *instance_id, rows));
                }
            }
        }

        // Sort rows by airgroup_id, air_id, and instance_id
        organized_rows.sort_by(|a, b| a.0.cmp(&b.0).then(a.1.cmp(&b.1)).then(a.2.cmp(&b.2)));

        // Print grouped rows
        for (airgroup_id, air_id, instance_id, mut rows) in organized_rows {
            let airgroup_name = pctx.global_info.get_air_group_name(airgroup_id);
            let air_name = pctx.global_info.get_air_name(airgroup_id, air_id);

            let setup = sctx.get_setup(airgroup_id, air_id)?;
            let p_expressions_bin = setup.p_setup.p_expressions_bin;

            let debug_data_hints_prod = get_hint_ids_by_name(p_expressions_bin, "gprod_debug_data");

            let debug_data_hints_sum = get_hint_ids_by_name(p_expressions_bin, "gsum_debug_data");

            let is_prod = local_data.get(&airgroup_id).unwrap().get(&air_id).unwrap().is_prod;

            let hint_id = local_data.get(&airgroup_id).unwrap().get(&air_id).unwrap().hint_id;

            let hint = match is_prod {
                true => debug_data_hints_prod[hint_id],
                false => debug_data_hints_sum[hint_id],
            };

            let piop_name = get_hint_field_constant_as_string(
                sctx,
                airgroup_id,
                air_id,
                hint as usize,
                "name_piop",
                HintFieldOptions::default(),
            )?;

            let expr_name = get_hint_field_constant_a_as_string(
                sctx,
                airgroup_id,
                air_id,
                hint as usize,
                "name_exprs",
                HintFieldOptions::default(),
            )?;

            rows.sort();
            let rows_display =
                rows.iter().map(|x| x.to_string()).take(max_values_to_print).collect::<Vec<_>>().join(",");

            let truncated = rows.len() > max_values_to_print;
            writeln!(output, "\t        - Airgroup: {airgroup_name} (id: {airgroup_id})").expect("Write error");
            writeln!(output, "\t          Air: {air_name} (id: {air_id})").expect("Write error");

            writeln!(output, "\t          PIOP: {piop_name}").expect("Write error");
            writeln!(output, "\t          Expression: {expr_name:?}").expect("Write error");

            writeln!(
                output,
                "\t          Instance ID: {} | Num: {} | Rows: [{}{}]",
                instance_id,
                rows.len(),
                rows_display,
                if truncated { ",..." } else { "" }
            )
            .expect("Write error");
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

    Ok(())
}
