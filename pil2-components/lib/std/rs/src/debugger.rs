use std::{collections::HashMap, sync::Mutex};

use proofman_hints::HintFieldOutput;
use proofman_hints::format_vec;

use num_traits::ToPrimitive;
use p3_field::PrimeField;

pub struct DebugData<F: PrimeField> {
    pub bus_values: Mutex<BusValues<F>>,
    pub opid_metadata: Mutex<OpIdMetadata>,
}

pub struct BusValue<F: PrimeField> {
    pub num_proves: F,
    pub num_assumes: F,
    pub row_proves: Vec<usize>,
    pub row_assumes: Vec<usize>,
}

// TODO: I should differenciate also in airgroup_id and air_id: (opid, airgroup_id, air_id). Run Lookup example with errors
type BusValues<F> = HashMap<u64, HashMap<Vec<HintFieldOutput<F>>, BusValue<F>>>; // opid ->   val    -> BusValue
type OpIdMetadata = HashMap<u64, Vec<(String, String, String, bool, Vec<String>)>>; // opid -> Vec(airgroup_name, air_name, piop_name, is_prove, expr_names)

pub fn check_bus_values<F: PrimeField>(
    name: &'static str,
    max_values_to_print: usize,
    debug_data: &Option<DebugData<F>>,
) {
    let mut there_are_errors = false;
    let debug_data = debug_data.as_ref().expect("Debug data missing");
    let opid_metadata = debug_data.opid_metadata.lock().expect("Opid metadata missing");
    let mut bus_values = debug_data.bus_values.lock().expect("Bus values missing");
    for (opid, bus) in bus_values.iter_mut() {
        let metadata = opid_metadata.get(opid).expect("Metadata missing");
        if bus.iter().any(|(_, v)| v.num_proves != v.num_assumes) {
            if !there_are_errors {
                there_are_errors = true;
                log::error!("{}: Some bus values do not match.", name);
            }
            println!("\t► Mismatched bus values for Opid #{}:", opid);
        } else {
            continue;
        }

        let mut unmatching_values_assume: Vec<(&Vec<HintFieldOutput<F>>, &mut BusValue<F>)> =
            bus.iter_mut().filter(|(_, v)| v.num_proves < v.num_assumes).collect();
        let num_errors_assume = unmatching_values_assume.len();

        let assume_expr: Vec<_> = metadata.iter().filter(|(_, _, _, is_prove, _)| !is_prove).collect();
        if num_errors_assume > 0 {
            println!("\t  ⁃ There are {} unmatching values thrown as 'assume' in:", num_errors_assume);
            for expr in assume_expr {
                println!(
                    "\t     (Air Group, Air, Argument, Expression) = ({}, {}, {}, {:?})",
                    expr.0, expr.1, expr.2, expr.4
                );
            }
        }

        for (i, (val, data)) in unmatching_values_assume.iter_mut().enumerate() {
            let num_proves = data.num_proves;
            let num_assumes = data.num_assumes;
            let diff = num_assumes - num_proves;
            let diff_usize = diff.as_canonical_biguint().to_usize().expect("Cannot convert to usize");
            let row_assumes = &mut data.row_assumes;

            row_assumes.sort();
            let row_assumes = if max_values_to_print < diff_usize {
                row_assumes[..max_values_to_print].to_vec()
            } else {
                row_assumes[..diff_usize].to_vec()
            };
            let row_assumes = row_assumes.iter().map(|x| x.to_string()).collect::<Vec<_>>().join(",");

            let name_str = if row_assumes.len() == 1 {
                format!("at row {}.", row_assumes)
            } else if max_values_to_print < row_assumes.len() {
                format!("at rows {},...", row_assumes)
            } else {
                format!("at rows {}.", row_assumes)
            };
            let diff_str = if diff.is_one() { "time" } else { "times" };
            println!(
                "\t    • Value:\n\t        {}\n\t      Appears {} {} {}\n\t      Num Assumes: {}.\n\t      Num Proves: {}.",
                format_vec(val),
                diff,
                diff_str,
                name_str,
                num_assumes,
                num_proves
            );

            if i == max_values_to_print {
                println!("\t      ...");
                break;
            }
        }

        if num_errors_assume > 0 {
            println!();
        }

        let mut unmatching_values_prove: Vec<(&Vec<HintFieldOutput<F>>, &mut BusValue<F>)> =
            bus.iter_mut().filter(|(_, v)| v.num_proves > v.num_assumes).collect();
        let num_errors_prove = unmatching_values_prove.len();

        let prove_expr: Vec<_> = metadata.iter().filter(|(_, _, _, is_prove, _)| *is_prove).collect();
        if num_errors_prove > 0 {
            println!("\t  ⁃ There are {} unmatching values thrown as 'prove' in:", num_errors_assume);
            for expr in prove_expr {
                println!(
                    "\t     (Air Group, Air, Argument, Expression) = ({}, {}, {}, {:?})",
                    expr.0, expr.1, expr.2, expr.4
                );
            }
        }

        for (i, (val, data)) in unmatching_values_prove.iter_mut().enumerate() {
            let num_proves = data.num_proves;
            let num_assumes = data.num_assumes;
            let diff = num_proves - num_assumes;
            let diff_usize = diff.as_canonical_biguint().to_usize().expect("Cannot convert to usize");
            let row_proves = &mut data.row_proves;

            row_proves.sort();
            let row_proves = if max_values_to_print < diff_usize {
                row_proves[..max_values_to_print].to_vec()
            } else {
                row_proves[..diff_usize].to_vec()
            };
            let row_proves = row_proves.iter().map(|x| x.to_string()).collect::<Vec<_>>().join(",");

            let name_str = if row_proves.len() == 1 {
                format!("at row {}.", row_proves)
            } else if max_values_to_print < row_proves.len() {
                format!("at rows {},...", row_proves)
            } else {
                format!("at rows {}.", row_proves)
            };
            let diff_str = if diff.is_one() { "time" } else { "times" };
            println!(
                "\t    • Value:\n\t        {}\n\t      Appears {} {} {}\n\t      Num Assumes: {}.\n\t      Num Proves: {}.",
                format_vec(val),
                diff,
                diff_str,
                name_str,
                num_assumes,
                num_proves
            );

            if i == max_values_to_print {
                println!("\t      ...");
                break;
            }
        }

        println!();
    }
}
