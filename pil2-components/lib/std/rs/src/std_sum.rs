use std::{collections::HashMap, sync::Arc};

use rayon::prelude::*;

use num_traits::ToPrimitive;
use p3_field::PrimeField;

use proofman_util::{timer_start_info, timer_stop_and_log_info};
use witness::WitnessComponent;
use proofman_common::{AirInstance, ProofCtx, SetupCtx};
use proofman_hints::{
    get_hint_field_gc_constant_a, get_hint_field, get_hint_field_a, acc_mul_hint_fields, update_airgroupvalue,
    get_hint_ids_by_name, mul_hint_fields, HintFieldOptions, HintFieldOutput, HintFieldValue, HintFieldValuesVec,
};

use crate::{
    check_invalid_opids, extract_field_element_as_usize, get_global_hint_field_constant_as,
    get_hint_field_constant_a_as_string, get_hint_field_constant_as_field, get_hint_field_constant_as_string,
    get_row_field_value, print_debug_info, update_debug_data, update_debug_data_fast, AirComponent, DebugData,
    DebugDataFast, SharedDataFast,
};

pub struct StdSum<F: PrimeField> {
    stage_wc: Option<u32>,
    _phantom: std::marker::PhantomData<F>,
}

impl<F: PrimeField> AirComponent<F> for StdSum<F> {
    const MY_NAME: &'static str = "STD Sum ";

    fn new(
        _pctx: Arc<ProofCtx<F>>,
        sctx: Arc<SetupCtx<F>>,
        _airgroup_id: Option<usize>,
        _air_id: Option<usize>,
    ) -> Arc<Self> {
        // Retrieve the std_sum_users hint ID
        let std_sum_users_id = get_hint_ids_by_name(sctx.get_global_bin(), "std_sum_users");

        // Initialize std_sum with the extracted data
        Arc::new(Self {
            stage_wc: match std_sum_users_id.is_empty() {
                true => None,
                false => {
                    // Get the "stage_wc" hint
                    let stage_wc =
                        get_global_hint_field_constant_as::<u32, F>(sctx.clone(), std_sum_users_id[0], "stage_wc");
                    Some(stage_wc)
                }
            },
            _phantom: std::marker::PhantomData,
        })
    }
}

impl<F: PrimeField> StdSum<F> {
    #[allow(clippy::too_many_arguments)]
    fn debug_mode(
        &self,
        pctx: &ProofCtx<F>,
        sctx: &SetupCtx<F>,
        air_instance: &AirInstance<F>,
        air_instance_id: usize,
        num_rows: usize,
        debug_data_hints: Vec<u64>,
        debug_data: &mut DebugData<F>,
        debug_data_fast: &mut DebugDataFast<F>,
        fast_mode: bool,
    ) {
        let airgroup_id = air_instance.airgroup_id;
        let air_id = air_instance.air_id;

        // Process each debug hint
        for &hint in debug_data_hints.iter() {
            // Extract hint fields
            let name_piop = get_hint_field_constant_as_string::<F>(
                sctx,
                airgroup_id,
                air_id,
                hint as usize,
                "name_piop",
                HintFieldOptions::default(),
            );

            let name_expr = get_hint_field_constant_a_as_string::<F>(
                sctx,
                airgroup_id,
                air_id,
                hint as usize,
                "name_expr",
                HintFieldOptions::default(),
            );

            let busid =
                get_hint_field::<F>(sctx, pctx, air_instance, hint as usize, "busid", HintFieldOptions::default());

            let is_global = get_hint_field_constant_as_field::<F>(
                sctx,
                airgroup_id,
                air_id,
                hint as usize,
                "is_global",
                HintFieldOptions::default(),
            );

            let proves =
                get_hint_field::<F>(sctx, pctx, air_instance, hint as usize, "proves", HintFieldOptions::default());

            let mul =
                get_hint_field::<F>(sctx, pctx, air_instance, hint as usize, "selector", HintFieldOptions::default());

            let expressions = get_hint_field_a::<F>(
                sctx,
                pctx,
                air_instance,
                hint as usize,
                "expressions",
                HintFieldOptions::default(),
            );

            let deg_expr = get_hint_field_constant_as_field::<F>(
                sctx,
                airgroup_id,
                air_id,
                hint as usize,
                "deg_expr",
                HintFieldOptions::default(),
            );

            let deg_mul = get_hint_field_constant_as_field::<F>(
                sctx,
                airgroup_id,
                air_id,
                hint as usize,
                "deg_sel",
                HintFieldOptions::default(),
            );

            // If both the expresion and the mul are of degree zero, then simply update the bus once
            if deg_expr.is_zero() && deg_mul.is_zero() {
                // In this case, the busid must be a field element
                let opid = match busid {
                    HintFieldValue::Field(opid) => {
                        // If opids are specified, then only update the bus if the opid is in the list
                        let opids = &pctx.options.debug_info.std_mode.opids;
                        if !opids.is_empty()
                            && !opids.contains(&opid.as_canonical_biguint().to_u64().expect("Cannot convert to u64"))
                        {
                            continue;
                        }
                        opid
                    }
                    _ => panic!("busid must be a field element"),
                };

                update_bus(
                    &name_piop,
                    &name_expr,
                    airgroup_id,
                    air_id,
                    air_instance_id,
                    opid,
                    &proves,
                    &mul,
                    &expressions,
                    0,
                    debug_data,
                    debug_data_fast,
                    is_global.is_one(),
                    fast_mode,
                );
            }
            // Otherwise, update the bus for each row
            else {
                for j in 0..num_rows {
                    // Get the opid for this row
                    let opid = match busid.get(j) {
                        HintFieldOutput::Field(opid) => {
                            // If opids are specified, then only update the bus if the opid is in the list
                            let opids = &pctx.options.debug_info.std_mode.opids;
                            if !opids.is_empty()
                                && !opids
                                    .contains(&opid.as_canonical_biguint().to_u64().expect("Cannot convert to u64"))
                            {
                                continue;
                            }

                            opid
                        }
                        _ => panic!("busid must be a field element"),
                    };

                    update_bus(
                        &name_piop,
                        &name_expr,
                        airgroup_id,
                        air_id,
                        air_instance_id,
                        opid,
                        &proves,
                        &mul,
                        &expressions,
                        j,
                        debug_data,
                        debug_data_fast,
                        false,
                        fast_mode,
                    );
                }
            }
        }

        #[allow(clippy::too_many_arguments)]
        fn update_bus<F: PrimeField>(
            name_piop: &str,
            name_expr: &[String],
            airgroup_id: usize,
            air_id: usize,
            instance_id: usize,
            opid: F,
            proves: &HintFieldValue<F>,
            mul: &HintFieldValue<F>,
            expressions: &HintFieldValuesVec<F>,
            row: usize,
            debug_data: &mut DebugData<F>,
            debug_data_fast: &mut DebugDataFast<F>,
            is_global: bool,
            fast_mode: bool,
        ) {
            let mut mul = get_row_field_value(mul, row, "mul");
            if mul.is_zero() {
                return;
            }

            let proves = match get_row_field_value(proves, row, "proves") {
                p if p.is_zero() || p == F::neg_one() => {
                    // If it's an "assume", negate its value
                    if p == F::neg_one() {
                        mul = -mul;
                    }
                    false
                }
                p if p.is_one() => true,
                _ => panic!("Proves hint must be either 0, 1, or -1"),
            };

            if fast_mode {
                update_debug_data_fast(debug_data_fast, opid, expressions.get(row), proves, mul, is_global);
            } else {
                update_debug_data(
                    debug_data,
                    name_piop,
                    name_expr,
                    opid,
                    expressions.get(row),
                    airgroup_id,
                    air_id,
                    instance_id,
                    row,
                    proves,
                    mul,
                    is_global,
                );
            }
        }
    }
}

impl<F: PrimeField> WitnessComponent<F> for StdSum<F> {
    fn calculate_witness(&self, stage: u32, pctx: Arc<ProofCtx<F>>, sctx: Arc<SetupCtx<F>>) {
        let stage_wc = self.stage_wc.as_ref();
        if stage_wc.is_none() {
            return;
        }

        if stage == *stage_wc.unwrap() {
            // Get the number of sum check users and their airgroup and air IDs
            let std_sum_users = get_hint_ids_by_name(sctx.get_global_bin(), "std_sum_users")[0];

            let num_users = get_global_hint_field_constant_as::<usize, F>(sctx.clone(), std_sum_users, "num_users");
            let airgroup_ids = get_hint_field_gc_constant_a::<F>(sctx.clone(), std_sum_users, "airgroup_ids", false);
            let air_ids = get_hint_field_gc_constant_a::<F>(sctx.clone(), std_sum_users, "air_ids", false);

            // Process each sum check user
            for i in 0..num_users {
                let airgroup_id = extract_field_element_as_usize(&airgroup_ids.values[i], "airgroup_id");
                let air_id = extract_field_element_as_usize(&air_ids.values[i], "air_id");

                // Get all air instances ids for this airgroup and air_id
                let global_instance_ids = pctx.air_instance_repo.find_air_instances(airgroup_id, air_id);
                for global_instance_id in global_instance_ids {
                    // Retrieve all air instances
                    let air_instances = &mut pctx.air_instance_repo.air_instances.write().unwrap();
                    let air_instance = air_instances.get_mut(&global_instance_id).unwrap();

                    if !air_instance.prover_initialized {
                        continue;
                    }

                    // Get the air associated with the air_instance
                    let airgroup_id = air_instance.airgroup_id;
                    let air_id = air_instance.air_id;
                    let air_name = &pctx.global_info.airs[airgroup_id][air_id].name;

                    let setup = sctx.get_setup(airgroup_id, air_id);
                    let p_expressions_bin = setup.p_setup.p_expressions_bin;

                    log::debug!("{}: ··· Computing witness for AIR '{}' at stage {}", Self::MY_NAME, air_name, stage);

                    let im_hints = get_hint_ids_by_name(p_expressions_bin, "im_col");
                    let im_airval_hints = get_hint_ids_by_name(p_expressions_bin, "im_airval");
                    let gsum_hints = get_hint_ids_by_name(p_expressions_bin, "gsum_col");

                    let im_total_hints: Vec<u64> = im_hints.iter().chain(im_airval_hints.iter()).cloned().collect();

                    let n_im_total_hints = im_total_hints.len();

                    if !im_total_hints.is_empty() {
                        mul_hint_fields::<F>(
                            &sctx,
                            &pctx,
                            air_instance,
                            im_total_hints.len() as u64,
                            im_total_hints,
                            vec!["reference"; n_im_total_hints],
                            vec!["numerator"; n_im_total_hints],
                            vec![HintFieldOptions::default(); n_im_total_hints],
                            vec!["denominator"; n_im_total_hints],
                            vec![HintFieldOptions::inverse(); n_im_total_hints],
                        );
                    }

                    // We know that at most one product hint exists
                    let gsum_hint = if gsum_hints.len() > 1 {
                        panic!("Multiple product hints found for AIR '{}'", air_name);
                    } else {
                        gsum_hints[0] as usize
                    };

                    // This call accumulates "expression" into "reference" expression and stores its last value to "result"
                    // Alternatively, this could be done using get_hint_field and set_hint_field methods and doing the accumulation in Rust,
                    acc_mul_hint_fields::<F>(
                        &sctx,
                        &pctx,
                        air_instance,
                        gsum_hint,
                        "reference",
                        "result",
                        "numerator_air",
                        "denominator_air",
                        HintFieldOptions::default(),
                        HintFieldOptions::inverse(),
                        true,
                    );

                    update_airgroupvalue::<F>(
                        &sctx,
                        &pctx,
                        air_instance,
                        gsum_hint,
                        "result",
                        "numerator_direct",
                        "denominator_direct",
                        HintFieldOptions::default(),
                        HintFieldOptions::inverse(),
                        true,
                    );
                }
            }

            // TODO: Process each direct update to the bus
            // when airgroup hints are available
        }
    }

    fn debug(&self, pctx: Arc<ProofCtx<F>>, sctx: Arc<SetupCtx<F>>) {
        timer_start_info!(DEBUG_MODE_SUM);
        let std_sum_users_vec = get_hint_ids_by_name(sctx.get_global_bin(), "std_sum_users");

        if !std_sum_users_vec.is_empty() {
            let std_sum_users = std_sum_users_vec[0];

            let num_users = get_global_hint_field_constant_as::<usize, F>(sctx.clone(), std_sum_users, "num_users");
            let airgroup_ids = get_hint_field_gc_constant_a::<F>(sctx.clone(), std_sum_users, "airgroup_ids", false);
            let air_ids = get_hint_field_gc_constant_a::<F>(sctx.clone(), std_sum_users, "air_ids", false);

            let fast_mode = pctx.options.debug_info.std_mode.fast_mode;

            let mut debug_data = HashMap::new();

            let mut debugs_data_fasts: Vec<HashMap<F, SharedDataFast>> = Vec::new();

            let mut global_instance_ids = Vec::new();

            for i in 0..num_users {
                let airgroup_id = extract_field_element_as_usize(&airgroup_ids.values[i], "airgroup_id");
                let air_id = extract_field_element_as_usize(&air_ids.values[i], "air_id");

                // Get all air instances ids for this airgroup and air_id
                let global_ids = pctx.air_instance_repo.find_air_instances(airgroup_id, air_id);

                for global_instance_id in global_ids {
                    // Retrieve all air instances
                    let air_instances = &mut pctx.air_instance_repo.air_instances.read().unwrap();
                    let air_instance = air_instances.get(&global_instance_id).unwrap();

                    if air_instance.prover_initialized {
                        global_instance_ids.push(global_instance_id);
                    }
                }
            }

            if fast_mode {
                // Process each sum check user
                debugs_data_fasts = global_instance_ids
                    .par_iter()
                    .map(|&global_instance_id| {
                        let mut local_debug_data_fast = HashMap::new();

                        // Retrieve all air instances
                        let air_instances = &mut pctx.air_instance_repo.air_instances.read().unwrap();

                        let air_instance = air_instances.get(&global_instance_id).unwrap();
                        let air_instance_id = pctx.dctx_find_air_instance_id(global_instance_id);
                        let air_name = &pctx.global_info.airs[air_instance.airgroup_id][air_instance.air_id].name;

                        log::debug!(
                            "{}: ··· Checking debug mode fast for instance_id {} of {}",
                            Self::MY_NAME,
                            air_instance_id,
                            air_name
                        );

                        // Get the air associated with the air_instance
                        let airgroup_id = air_instance.airgroup_id;
                        let air_id = air_instance.air_id;

                        let setup = sctx.get_setup(airgroup_id, air_id);
                        let p_expressions_bin = setup.p_setup.p_expressions_bin;

                        let num_rows = pctx.global_info.airs[airgroup_id][air_id].num_rows;

                        let debug_data_hints = get_hint_ids_by_name(p_expressions_bin, "gsum_debug_data");

                        self.debug_mode(
                            &pctx,
                            &sctx,
                            air_instance,
                            air_instance_id,
                            num_rows,
                            debug_data_hints.clone(),
                            &mut HashMap::new(),
                            &mut local_debug_data_fast,
                            true,
                        );

                        local_debug_data_fast
                    })
                    .collect();
            } else {
                // Process each sum check user
                for global_instance_id in global_instance_ids {
                    // Retrieve all air instances
                    let air_instances = &mut pctx.air_instance_repo.air_instances.read().unwrap();
                    let air_instance = air_instances.get(&global_instance_id).unwrap();
                    let air_instance_id = pctx.dctx_find_air_instance_id(global_instance_id);
                    let air_name = &pctx.global_info.airs[air_instance.airgroup_id][air_instance.air_id].name;

                    log::debug!(
                        "{}: ··· Checking debug mode for instance_id {} of {}",
                        Self::MY_NAME,
                        air_instance_id,
                        air_name
                    );

                    // Get the air associated with the air_instance
                    let airgroup_id = air_instance.airgroup_id;
                    let air_id = air_instance.air_id;

                    let setup = sctx.get_setup(airgroup_id, air_id);
                    let p_expressions_bin = setup.p_setup.p_expressions_bin;

                    let num_rows = pctx.global_info.airs[airgroup_id][air_id].num_rows;

                    let debug_data_hints = get_hint_ids_by_name(p_expressions_bin, "gsum_debug_data");

                    self.debug_mode(
                        &pctx,
                        &sctx,
                        air_instance,
                        air_instance_id,
                        num_rows,
                        debug_data_hints.clone(),
                        &mut debug_data,
                        &mut HashMap::new(),
                        false,
                    );
                }
            }

            if fast_mode {
                check_invalid_opids(&pctx, Self::MY_NAME, &mut debugs_data_fasts);
            } else {
                let max_values_to_print = pctx.options.debug_info.std_mode.n_vals;
                let print_to_file = pctx.options.debug_info.std_mode.print_to_file;
                print_debug_info(&pctx, Self::MY_NAME, max_values_to_print, print_to_file, &mut debug_data);
            }
        }
        timer_stop_and_log_info!(DEBUG_MODE_SUM);
    }
}
