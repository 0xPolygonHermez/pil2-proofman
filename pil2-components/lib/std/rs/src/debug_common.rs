use std::sync::RwLock;

use rayon::prelude::*;

use rustc_hash::FxHashMap;
use fields::PrimeField64;

use proofman_common::{DebugInfo, ProofCtx, ProofmanError, ProofmanResult, SetupCtx, store_rows_info_air};
use proofman_hints::{
    get_hint_field, get_hint_field_a, get_hint_field_gc, get_hint_field_gc_a, get_hint_ids_by_name, HintFieldOptions,
    HintFieldValue, HintFieldValuesVec,
};

use crate::{
    check_invalid_opids, get_global_hint_field, get_global_hint_field_constant_as, get_hint_field_constant_as,
    get_hint_field_constant_as_string, get_hint_field_constant_a_as_string, get_hint_field_constant_as_field,
    get_row_field_value, print_debug_info, update_debug_data, update_debug_data_fast, DebugData, DebugDataFast,
    DebugDataFastGlobal, DebugDataInfo, HintMetadata, hash_vals, normalize_vals,
};

#[allow(clippy::too_many_arguments)]
pub fn extract_global_hint_fields<F: PrimeField64>(
    pctx: &ProofCtx<F>,
    sctx: &SetupCtx<F>,
    debug_data: &mut DebugData,
    debug_data_info: &mut DebugDataInfo,
    debug_data_fast: &mut DebugDataFast,
    debug_data_fast_global: &mut DebugDataFastGlobal,
    fast_mode: bool,
    is_prod: bool,
    debug_hashes: &[u64],
) -> ProofmanResult<()> {
    let debug_data_name = if is_prod { "gprod_debug_data_global" } else { "gsum_debug_data_global" };
    let debug_data_hints = get_hint_ids_by_name(sctx.get_global_bin(), debug_data_name);
    if !debug_data_hints.is_empty() {
        let num_global_hints =
            get_global_hint_field_constant_as::<usize, F>(sctx, debug_data_hints[0], "num_global_hints")?;
        for i in 0..num_global_hints {
            let airgroup_id =
                get_global_hint_field_constant_as::<usize, F>(sctx, debug_data_hints[1 + i], "airgroup_id")?;
            let type_piop = get_global_hint_field_constant_as::<u64, F>(sctx, debug_data_hints[1 + i], "type_piop")?;
            if ![0, 1, 2].contains(&type_piop) {
                return Err(ProofmanError::StdError(format!("Invalid type_piop: {type_piop}")));
            }

            let opid = get_global_hint_field(sctx, debug_data_hints[1 + i], "busid")?;
            let opid = opid.as_canonical_u64();

            // If opids are specified, then only update the bus if the opid is in the list
            if !pctx.debug_info.read().unwrap().std_mode.opids.is_empty()
                && !pctx.debug_info.read().unwrap().std_mode.opids.contains(&opid)
            {
                continue;
            }

            let num_reps = get_hint_field_gc(pctx, sctx, debug_data_hints[1 + i], "num_reps", false)?;

            // If the number of repetitions is zero, continue
            let mut num_reps = get_row_field_value(&num_reps, 0, "num_reps")?;
            if num_reps.is_zero() {
                continue;
            }

            // If the type_piop is free and the num_reps is minus_one, simply flip the num_reps
            if type_piop == 2 {
                if num_reps == F::NEG_ONE {
                    num_reps = -num_reps;
                } else if num_reps != F::ONE {
                    return Err(ProofmanError::StdError(format!(
                        "The number of repetitions in a free piop can only be {{-1, 0, 1}}, received: {num_reps}"
                    )));
                }
            }

            let expressions = get_hint_field_gc_a(pctx, sctx, debug_data_hints[1 + i], "expressions", false)?;
            let is_proves = type_piop == 1;
            let expr = expressions.get(0);
            let norm_vals = normalize_vals(&expr);
            let hash = hash_vals(norm_vals);
            if fast_mode {
                update_debug_data_fast(
                    debug_data_fast,
                    debug_data_fast_global,
                    opid,
                    hash,
                    is_proves,
                    num_reps.as_canonical_u64(),
                    true,
                )?;
            } else {
                update_debug_data(
                    debug_data,
                    debug_data_info,
                    i,
                    opid,
                    norm_vals,
                    hash,
                    airgroup_id,
                    None,
                    None,
                    0,
                    is_proves,
                    num_reps.as_canonical_u64(),
                    true,
                    is_prod,
                    true,
                    debug_hashes,
                )?;
            }
        }
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub fn extract_hint_fields<F: PrimeField64>(
    pctx: &ProofCtx<F>,
    sctx: &SetupCtx<F>,
    instance_id: usize,
    debug_data: &mut DebugData,
    debug_data_info: &mut DebugDataInfo,
    debug_data_fast: &RwLock<DebugDataFast>,
    debug_data_fast_global: &RwLock<DebugDataFastGlobal>,
    fast_mode: bool,
    is_prod: bool,
    debug_hashes: &[u64],
) -> ProofmanResult<()> {
    let (airgroup_id, air_id) = pctx.dctx_get_instance_info(instance_id)?;
    let air_instance_id = pctx.dctx_find_air_instance_id(instance_id)?;

    let setup = sctx.get_setup(airgroup_id, air_id)?;
    let p_expressions_bin = setup.p_setup.p_expressions_bin;

    let debug_data_name = if is_prod { "gprod_debug_data" } else { "gsum_debug_data" };

    let debug_data_hints = get_hint_ids_by_name(p_expressions_bin, debug_data_name);

    let num_rows = pctx.global_info.airs[airgroup_id][air_id].num_rows;

    let hint_metadatas: Result<Vec<_>, ProofmanError> = debug_data_hints
        .iter()
        .enumerate()
        .map(|(i, &hint)| {
            let busid = get_hint_field(pctx, setup, instance_id, hint as usize, "busid", HintFieldOptions::default())?;

            let type_piop = get_hint_field_constant_as::<u64, F>(
                pctx,
                setup,
                airgroup_id,
                air_id,
                hint as usize,
                "type_piop",
                HintFieldOptions::default(),
            )?;
            if ![0, 1, 2].contains(&type_piop) {
                return Err(ProofmanError::StdError(format!("Invalid type_piop: {type_piop}")));
            }

            let num_reps =
                get_hint_field(pctx, setup, instance_id, hint as usize, "num_reps", HintFieldOptions::default())?;

            let deg_expr = get_hint_field_constant_as_field(
                pctx,
                setup,
                airgroup_id,
                air_id,
                hint as usize,
                "deg_expr",
                HintFieldOptions::default(),
            )?;

            let deg_mul = get_hint_field_constant_as_field(
                pctx,
                setup,
                airgroup_id,
                air_id,
                hint as usize,
                "deg_sel",
                HintFieldOptions::default(),
            )?;

            let name_piop = get_hint_field_constant_as_string(
                pctx,
                setup,
                airgroup_id,
                air_id,
                hint as usize,
                "name_piop",
                HintFieldOptions::default(),
            )?;

            let name_exprs = get_hint_field_constant_a_as_string(
                pctx,
                setup,
                airgroup_id,
                air_id,
                hint as usize,
                "name_exprs",
                HintFieldOptions::default(),
            )?;

            let expressions =
                get_hint_field_a(pctx, setup, instance_id, hint as usize, "expressions", HintFieldOptions::default())?;

            Ok(HintMetadata {
                hint,
                hint_id: i,
                busid,
                type_piop,
                num_reps,
                expressions,
                deg_expr,
                deg_mul,
                name_piop,
                name_exprs,
            })
        })
        .collect();

    let hint_metadatas = hint_metadatas?;

    let opids = &pctx.debug_info.read().unwrap().std_mode.opids;

    if fast_mode {
        // Process hints in chunks of to reuse pre-allocated HashMaps
        hint_metadatas.par_iter().try_for_each(|hint_metadata| -> ProofmanResult<()> {
            // Directly acquire write lock and work with it
            let mut local_debug_data = DebugDataFast::new();
            let mut local_debug_data_global = DebugDataFastGlobal::new();

            // If both the expression and the mul are of degree zero, then simply update the bus once
            if hint_metadata.deg_expr.is_zero() && hint_metadata.deg_mul.is_zero() {
                update_bus_fast(
                    opids,
                    &hint_metadata.busid,
                    hint_metadata.type_piop,
                    &hint_metadata.num_reps,
                    &hint_metadata.expressions,
                    0,
                    &mut local_debug_data,
                    &mut local_debug_data_global,
                    false,
                )?;
            }
            // Otherwise, update the bus for each row
            else {
                for j in 0..num_rows {
                    update_bus_fast(
                        opids,
                        &hint_metadata.busid,
                        hint_metadata.type_piop,
                        &hint_metadata.num_reps,
                        &hint_metadata.expressions,
                        j,
                        &mut local_debug_data,
                        &mut local_debug_data_global,
                        false,
                    )?;
                }
            }

            let mut shared = debug_data_fast.write().unwrap();
            local_debug_data.merge_into(&mut shared);
            let mut shared_global = debug_data_fast_global.write().unwrap();
            for (opid, hashes) in local_debug_data_global.into_iter() {
                shared_global.entry(opid).or_default().extend(hashes);
            }
            Ok(())
        })?;
    } else {
        let store_row_info = store_rows_info_air(pctx, airgroup_id, air_id, instance_id);
        for hint_metadata in hint_metadatas.iter() {
            // If both the expresion and the mul are of degree zero, then simply update the bus once
            if hint_metadata.deg_expr.is_zero() && hint_metadata.deg_mul.is_zero() {
                update_bus(
                    opids,
                    hint_metadata.hint_id,
                    airgroup_id,
                    air_id,
                    air_instance_id,
                    &hint_metadata.busid,
                    hint_metadata.type_piop,
                    &hint_metadata.num_reps,
                    &hint_metadata.expressions,
                    0,
                    debug_data,
                    debug_data_info,
                    false,
                    is_prod,
                    store_row_info,
                    debug_hashes,
                )?;
            }
            // Otherwise, update the bus for each row
            else {
                for j in 0..num_rows {
                    update_bus(
                        opids,
                        hint_metadata.hint_id,
                        airgroup_id,
                        air_id,
                        air_instance_id,
                        &hint_metadata.busid,
                        hint_metadata.type_piop,
                        &hint_metadata.num_reps,
                        &hint_metadata.expressions,
                        j,
                        debug_data,
                        debug_data_info,
                        false,
                        is_prod,
                        store_row_info,
                        debug_hashes,
                    )?;
                }
            }
        }
    }

    std::thread::spawn(move || {
        drop(hint_metadatas);
    });
    Ok(())
}

#[allow(clippy::too_many_arguments)]
#[inline]
pub fn update_bus<F: PrimeField64>(
    op_ids: &[u64],
    hint_id: usize,
    airgroup_id: usize,
    air_id: usize,
    instance_id: usize,
    busid: &HintFieldValue<F>,
    type_piop: u64,
    num_reps: &HintFieldValue<F>,
    expressions: &HintFieldValuesVec<F>,
    row: usize,
    debug_data: &mut DebugData,
    debug_data_info: &mut DebugDataInfo,
    is_global: bool,
    is_prod: bool,
    store_row_info: bool,
    debug_hashes: &[u64],
) -> ProofmanResult<()> {
    let opid = get_row_field_value(busid, row, "busid")?;
    if !op_ids.is_empty() && !op_ids.contains(&opid.as_canonical_u64()) {
        return Ok(());
    }

    let mut num_reps = get_row_field_value(num_reps, row, "num_reps")?;
    if num_reps.is_zero() {
        return Ok(());
    }

    let is_proves = match type_piop {
        0 => false,
        1 => true,
        2 => {
            if num_reps == F::NEG_ONE {
                // If the type is free and the num_reps is minus_one, simply flip the num_reps
                num_reps = -num_reps;
                false
            } else if num_reps == F::ONE {
                true
            } else {
                return Err(ProofmanError::StdError(format!(
                    "The number of repetitions in a free piop can only be {{-1, 0, 1}}, received: {num_reps}"
                )));
            }
        }
        _ => unreachable!(),
    };

    let expr = expressions.get(row);
    let norm_vals = normalize_vals(&expr);
    let hash = hash_vals(norm_vals);

    update_debug_data(
        debug_data,
        debug_data_info,
        hint_id,
        opid.as_canonical_u64(),
        norm_vals,
        hash,
        airgroup_id,
        Some(air_id),
        Some(instance_id),
        row,
        is_proves,
        num_reps.as_canonical_u64(),
        is_global,
        is_prod,
        store_row_info,
        debug_hashes,
    )
}

#[allow(clippy::too_many_arguments)]
fn update_bus_fast<F: PrimeField64>(
    op_ids: &[u64],
    busid: &HintFieldValue<F>,
    type_piop: u64,
    num_reps: &HintFieldValue<F>,
    expressions: &HintFieldValuesVec<F>,
    row: usize,
    debug_data_fast: &mut DebugDataFast,
    debug_data_fast_global: &mut DebugDataFastGlobal,
    is_global: bool,
) -> ProofmanResult<()> {
    let opid = get_row_field_value(busid, row, "busid")?;
    if !op_ids.is_empty() && !op_ids.contains(&opid.as_canonical_u64()) {
        return Ok(());
    }

    let mut num_reps = get_row_field_value(num_reps, row, "num_reps")?;
    if num_reps.is_zero() {
        return Ok(());
    }

    let is_proves = match type_piop {
        0 => false,
        1 => true,
        2 => {
            if num_reps == F::NEG_ONE {
                // If the type is free and the num_reps is minus_one, simply flip the num_reps
                num_reps = -num_reps;
                false
            } else if num_reps == F::ONE {
                true
            } else {
                return Err(ProofmanError::StdError(format!(
                    "The number of repetitions in a free piop can only be {{-1, 0, 1}}, received: {num_reps}"
                )));
            }
        }
        _ => unreachable!(),
    };

    let expr = expressions.get(row);
    let norm_vals = normalize_vals(&expr);
    let hash = hash_vals(norm_vals);

    update_debug_data_fast(
        debug_data_fast,
        debug_data_fast_global,
        opid.as_canonical_u64(),
        hash,
        is_proves,
        num_reps.as_canonical_u64(),
        is_global,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn print_std_debug_info<F: PrimeField64>(
    pctx: &ProofCtx<F>,
    sctx: &SetupCtx<F>,
    debug_data: &RwLock<DebugData>,
    debug_data_info: &RwLock<DebugDataInfo>,
    debug_data_fast: &RwLock<DebugDataFast>,
    debug_data_fast_global: &RwLock<DebugDataFastGlobal>,
    debug_info: &DebugInfo,
    is_prod: bool,
    debug_hashes: &[u64],
) -> ProofmanResult<()> {
    let fast_mode = debug_info.std_mode.fast_mode;

    if fast_mode {
        let mut debug_data_fast = debug_data_fast.write().unwrap();
        let mut debug_data_fast_global = debug_data_fast_global.write().unwrap();

        // Perform the global hint update
        extract_global_hint_fields(
            pctx,
            sctx,
            &mut FxHashMap::default(),
            &mut FxHashMap::default(),
            &mut debug_data_fast,
            &mut debug_data_fast_global,
            fast_mode,
            is_prod,
            debug_hashes,
        )?;

        check_invalid_opids(pctx, std::mem::take(&mut *debug_data_fast));
    } else {
        let mut debug_data_t = debug_data.write().unwrap();
        let mut debug_data_info_t = debug_data_info.write().unwrap();
        extract_global_hint_fields(
            pctx,
            sctx,
            &mut debug_data_t,
            &mut debug_data_info_t,
            &mut DebugDataFast::new(),
            &mut DebugDataFastGlobal::new(),
            fast_mode,
            is_prod,
            debug_hashes,
        )?;

        let max_values_to_print = debug_info.std_mode.n_vals;
        let print_to_file = debug_info.std_mode.print_to_file;
        print_debug_info(pctx, sctx, max_values_to_print, print_to_file, &mut debug_data_t, &mut debug_data_info_t)?;
    }

    Ok(())
}
