use std::{
    collections::HashMap,
    fs::{self, File},
    hash::{DefaultHasher, Hasher, Hash},
    io::{self, Write},
    path::{Path, PathBuf},
};

use fields::PrimeField64;
use proofman_common::ProofCtx;
use proofman_hints::{format_vec, HintFieldOutput};

use num_bigint::BigUint;
use num_traits::Zero;

use colored::*;

pub type DebugData<F> = HashMap<F, HashMap<Vec<HintFieldOutput<F>>, BusValue<F>>>; // opid -> val -> BusValue

pub type DebugDataFast<F> = HashMap<F, SharedDataFast>; // opid -> sharedDataFast

#[derive(Debug)]
pub struct BusValue<F> {
    shared_data: SharedData<F>, // Data shared across all airgroups, airs, and instances
    grouped_data: AirGroupMap,  // Data grouped by: airgroup_id -> air_id -> instance_id -> InstanceData
}

#[derive(Debug)]
struct SharedData<F> {
    direct_was_called: bool,
    num_proves: F,
    num_assumes: F,
}

#[derive(Clone, Debug)]
pub struct SharedDataFast {
    pub num_proves: BigUint,
    pub num_assumes: BigUint,
    pub num_proves_global: Vec<BigUint>,
    pub num_assumes_global: Vec<BigUint>,
}

type AirGroupMap = HashMap<usize, AirMap>;
type AirMap = HashMap<usize, AirData>;

#[derive(Debug)]
struct AirData {
    name_piop: String,
    name_expr: Vec<String>,
    instances: InstanceMap,
}

type InstanceMap = HashMap<usize, InstanceData>;

#[derive(Debug)]
struct InstanceData {
    row_proves: Vec<usize>,
    row_assumes: Vec<usize>,
}

#[allow(clippy::too_many_arguments)]
pub fn update_debug_data_fast<F: PrimeField64>(
    debug_data_fast: &mut DebugDataFast<F>,
    opid: F,
    val: Vec<HintFieldOutput<F>>,
    proves: bool,
    times: F,
    is_global: bool,
) {
    let bus_opid_times = debug_data_fast.entry(opid).or_insert_with(|| SharedDataFast {
        num_assumes_global: Vec::new(),
        num_proves_global: Vec::new(),
        num_proves: BigUint::zero(),
        num_assumes: BigUint::zero(),
    });

    let mut values = Vec::new();
    for value in val.iter() {
        match value {
            HintFieldOutput::Field(f) => values.push(*f),
            HintFieldOutput::FieldExtended(ef) => {
                values.push(ef.value[0]);
                values.push(ef.value[1]);
                values.push(ef.value[2]);
            }
        }
    }

    let mut hasher = DefaultHasher::new();
    values.hash(&mut hasher);

    let hash_value = BigUint::from(hasher.finish());

    if is_global {
        if proves {
            // Check if bus op id times num proves global contains value
            if bus_opid_times.num_proves_global.contains(&hash_value) {
                return;
            }
            bus_opid_times.num_proves_global.push(hash_value * times.as_canonical_biguint());
        } else {
            if bus_opid_times.num_assumes_global.contains(&hash_value) {
                return;
            }
            bus_opid_times.num_assumes_global.push(hash_value);
        }
    } else if proves {
        bus_opid_times.num_proves += hash_value * times.as_canonical_biguint();
    } else {
        assert!(times.is_one(), "The selector value is invalid: expected 1, but received {:?}.", times);
        bus_opid_times.num_assumes += hash_value;
    }
}

#[allow(clippy::too_many_arguments)]
pub fn update_debug_data<F: PrimeField64>(
    debug_data: &mut DebugData<F>,
    name_piop: &str,
    name_expr: &[String],
    opid: F,
    val: Vec<HintFieldOutput<F>>,
    airgroup_id: usize,
    air_id: usize,
    instance_id: usize,
    row: usize,
    proves: bool,
    times: F,
    is_global: bool,
) {
    let bus_opid = debug_data.entry(opid).or_default();

    let bus_val = bus_opid.entry(val).or_insert_with(|| BusValue {
        shared_data: SharedData { direct_was_called: false, num_proves: F::ZERO, num_assumes: F::ZERO },
        grouped_data: AirGroupMap::new(),
    });

    let grouped_data = bus_val
        .grouped_data
        .entry(airgroup_id)
        .or_default()
        .entry(air_id)
        .or_insert_with(|| AirData {
            name_piop: name_piop.to_owned(),
            name_expr: name_expr.to_owned(),
            instances: InstanceMap::new(),
        })
        .instances
        .entry(instance_id)
        .or_insert_with(|| InstanceData { row_proves: Vec::new(), row_assumes: Vec::new() });

    // If the value is global but it was already processed, skip it
    if is_global {
        if bus_val.shared_data.direct_was_called {
            return;
        }
        bus_val.shared_data.direct_was_called = true;
    }

    if proves {
        bus_val.shared_data.num_proves += times;
        grouped_data.row_proves.push(row);
    } else {
        assert!(times.is_one(), "The selector value is invalid: expected 1, but received {:?}.", times);
        bus_val.shared_data.num_assumes += times;
        grouped_data.row_assumes.push(row);
    }
}

pub fn check_invalid_opids<F: PrimeField64>(_pctx: &ProofCtx<F>, debugs_data_fasts: &mut [DebugDataFast<F>]) -> Vec<F> {
    let mut debug_data_fast = HashMap::new();

    let mut global_assumes = Vec::new();
    let mut global_proves = Vec::new();
    for map in debugs_data_fasts {
        for (opid, bus) in map.iter() {
            if debug_data_fast.contains_key(opid) {
                let bus_fast: &mut SharedDataFast = debug_data_fast.get_mut(opid).unwrap();
                for assume_global in bus.num_assumes_global.iter() {
                    if global_assumes.contains(assume_global) {
                        continue;
                    }
                    global_assumes.push(assume_global.clone());
                    bus_fast.num_assumes += assume_global;
                }
                for prove_global in bus.num_proves_global.iter() {
                    if global_proves.contains(prove_global) {
                        continue;
                    }
                    global_proves.push(prove_global.clone());
                    bus_fast.num_proves += prove_global;
                }

                bus_fast.num_proves += bus.num_proves.clone();
                bus_fast.num_assumes += bus.num_assumes.clone();
            } else {
                debug_data_fast.insert(*opid, bus.clone());
            }
        }
    }

    // TODO: SINCRONIZATION IN DISTRIBUTED MODE

    let mut invalid_opids = Vec::new();

    // Check if there are any invalid opids

    for (opid, bus) in debug_data_fast.iter_mut() {
        if bus.num_proves != bus.num_assumes {
            invalid_opids.push(*opid);
        }
    }

    if !invalid_opids.is_empty() {
        tracing::error!(
            "··· {}",
            format!("\u{2717} The following opids does not match {:?}", invalid_opids).bright_red().bold()
        );
    } else {
        tracing::info!("··· {}", "\u{2713} All bus values match.".bright_green().bold());
    }

    invalid_opids
}
pub fn print_debug_info<F: PrimeField64>(
    pctx: &ProofCtx<F>,
    max_values_to_print: usize,
    print_to_file: bool,
    debug_data: &mut DebugData<F>,
) {
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
                                eprintln!("Failed to create directory {:?}: {}", tmp_dir, e);
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
                            eprintln!("Failed to create log file at {:?}: {}", file_path, e);
                            std::process::exit(1);
                        }
                    }
                }

                let file_msg = if print_to_file {
                    format!(" Check the {:?} file for more details.", file_path)
                } else {
                    "".to_string()
                };
                tracing::error!("Some bus values do not match.{}", file_msg);

                // Set the flag to avoid printing the error message multiple times
                there_are_errors = true;
            }
            writeln!(output, "\t► Mismatched bus values for opid {}:", opid).expect("Write error");
        } else {
            continue;
        }

        // TODO: Sort unmatching values by the row
        let mut overassumed_values: Vec<(&Vec<HintFieldOutput<F>>, &mut BusValue<F>)> =
            bus.iter_mut().filter(|(_, v)| v.shared_data.num_proves < v.shared_data.num_assumes).collect();
        let len_overassumed = overassumed_values.len();

        if len_overassumed > 0 {
            writeln!(output, "\t  ⁃ There are {} unmatching values thrown as 'assume':", len_overassumed)
                .expect("Write error");
        }

        for (i, (val, data)) in overassumed_values.iter_mut().enumerate() {
            if i == max_values_to_print {
                writeln!(output, "\t      ...").expect("Write error");
                break;
            }
            let shared_data = &data.shared_data;
            let grouped_data = &mut data.grouped_data;
            print_diffs(pctx, val, max_values_to_print, shared_data, grouped_data, false, &mut output);
        }

        if len_overassumed > 0 {
            writeln!(output).expect("Write error");
        }

        // TODO: Sort unmatching values by the row
        let mut overproven_values: Vec<(&Vec<HintFieldOutput<F>>, &mut BusValue<F>)> =
            bus.iter_mut().filter(|(_, v)| v.shared_data.num_proves > v.shared_data.num_assumes).collect();
        let len_overproven = overproven_values.len();

        if len_overproven > 0 {
            writeln!(output, "\t  ⁃ There are {} unmatching values thrown as 'prove':", len_overproven)
                .expect("Write error");
        }

        for (i, (val, data)) in overproven_values.iter_mut().enumerate() {
            if i == max_values_to_print {
                writeln!(output, "\t      ...").expect("Write error");
                break;
            }

            let shared_data = &data.shared_data;
            let grouped_data = &mut data.grouped_data;
            print_diffs(pctx, val, max_values_to_print, shared_data, grouped_data, true, &mut output);
        }

        if len_overproven > 0 {
            writeln!(output).expect("Write error");
        }
    }

    if !there_are_errors {
        tracing::info!("··· {}", "\u{2713} All bus values match.".bright_green().bold());
    }

    fn print_diffs<F: PrimeField64>(
        pctx: &ProofCtx<F>,
        val: &[HintFieldOutput<F>],
        max_values_to_print: usize,
        shared_data: &SharedData<F>,
        grouped_data: &mut AirGroupMap,
        proves: bool,
        output: &mut dyn Write,
    ) {
        let num_assumes = shared_data.num_assumes;
        let num_proves = shared_data.num_proves;

        let num = if proves { num_proves } else { num_assumes };
        let num_str = if num.is_one() { "time" } else { "times" };

        writeln!(output, "\t    ==================================================").expect("Write error");
        writeln!(
            output,
            "\t    • Value:\n\t        {}\n\t      Appears {} {} across the following:",
            format_vec(val),
            num,
            num_str,
        )
        .expect("Write error");

        // Collect and organize rows
        let mut organized_rows = Vec::new();
        for (airgroup_id, air_id_map) in grouped_data.iter_mut() {
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
            let piop_name = &grouped_data.get(&airgroup_id).unwrap().get(&air_id).unwrap().name_piop;
            let expr_name = &grouped_data.get(&airgroup_id).unwrap().get(&air_id).unwrap().name_expr;

            rows.sort();
            let rows_display =
                rows.iter().map(|x| x.to_string()).take(max_values_to_print).collect::<Vec<_>>().join(",");

            let truncated = rows.len() > max_values_to_print;
            writeln!(output, "\t        - Airgroup: {} (id: {})", airgroup_name, airgroup_id).expect("Write error");
            writeln!(output, "\t          Air: {} (id: {})", air_name, air_id).expect("Write error");

            writeln!(output, "\t          PIOP: {}", piop_name).expect("Write error");
            writeln!(output, "\t          Expression: {:?}", expr_name).expect("Write error");

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
            "\t    Total Num Assumes: {}.\n\t    Total Num Proves: {}.\n\t    Total Unmatched: {}.",
            num_assumes, num_proves, diff
        )
        .expect("Write error");
        writeln!(output, "\t    ==================================================\n").expect("Write error");
    }
}
