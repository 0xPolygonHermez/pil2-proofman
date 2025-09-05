use std::{
    collections::HashMap,
    sync::{Arc, RwLock},
};

use rayon::prelude::*;

use fields::PrimeField64;

use proofman_util::{timer_start_info, timer_stop_and_log_info};
use witness::WitnessComponent;
use proofman_common::{skip_prover_instance, BufferPool, DebugInfo, ModeName, ProofCtx, SetupCtx};
use proofman_hints::{
    acc_mul_hint_fields, get_hint_field, get_hint_field_a, get_hint_field_gc, get_hint_field_gc_a,
    get_hint_ids_by_name, mul_hint_fields, update_airgroupvalue, HintFieldOptions, HintFieldOutput, HintFieldValue,
    HintFieldValuesVec,
};

use crate::{
    check_invalid_opids, get_global_hint_field, get_global_hint_field_constant_a_as,
    get_global_hint_field_constant_a_as_string, get_global_hint_field_constant_as,
    get_global_hint_field_constant_as_string, get_hint_field_constant_a_as_string, get_hint_field_constant_as,
    get_hint_field_constant_as_field, get_hint_field_constant_as_string, get_row_field_value, print_debug_info,
    update_debug_data, update_debug_data_fast, DebugData, DebugDataFast, SharedDataFast, STD_MODE_DEFAULT,
    STD_MODE_ONE_INSTANCE,
};

pub struct StdSum<F: PrimeField64> {
    num_users: usize,
    std_mode: Vec<usize>,
    airgroup_ids: Vec<usize>,
    air_ids: Vec<usize>,
    debug_data: RwLock<DebugData<F>>,
    debug_data_fast: RwLock<Vec<DebugDataFast<F>>>,
}

impl<F: PrimeField64> StdSum<F> {
    pub fn new(sctx: &Arc<SetupCtx<F>>) -> Arc<Self> {
        // Get the sum check global data related to its users
        let std_sum_users = get_hint_ids_by_name(sctx.get_global_bin(), "std_sum_users");

        let Some(&std_sum_users) = std_sum_users.first() else {
            return Arc::new(Self {
                num_users: 0,
                std_mode: Vec::new(),
                airgroup_ids: Vec::new(),
                air_ids: Vec::new(),
                debug_data: RwLock::new(HashMap::new()),
                debug_data_fast: RwLock::new(Vec::new()),
            });
        };

        let num_users = get_global_hint_field_constant_as::<usize, F>(sctx, std_sum_users, "num_users");
        let std_mode = get_global_hint_field_constant_a_as::<usize, F>(sctx, std_sum_users, "std_mode");
        let airgroup_ids = get_global_hint_field_constant_a_as::<usize, F>(sctx, std_sum_users, "airgroup_ids");
        let air_ids = get_global_hint_field_constant_a_as::<usize, F>(sctx, std_sum_users, "air_ids");

        Arc::new(Self {
            num_users,
            std_mode,
            airgroup_ids,
            air_ids,
            debug_data: RwLock::new(HashMap::new()),
            debug_data_fast: RwLock::new(Vec::new()),
        })
    }
}

impl<F: PrimeField64> WitnessComponent<F> for StdSum<F> {
    fn calculate_witness(
        &self,
        stage: u32,
        pctx: Arc<ProofCtx<F>>,
        sctx: Arc<SetupCtx<F>>,
        instance_ids: &[usize],
        _n_cores: usize,
        _buffer_pool: &dyn BufferPool<F>,
    ) {
        if stage == 2 {
            let instances = pctx.dctx_get_instances();
            // Process each sum check user
            for i in 0..self.num_users {
                let airgroup_id = self.airgroup_ids[i];
                let air_id = self.air_ids[i];

                for instance_id in instance_ids.iter() {
                    if instances[*instance_id].airgroup_id != airgroup_id
                        || instances[*instance_id].air_id != air_id
                        || skip_prover_instance(&pctx, *instance_id).0
                    {
                        continue;
                    }

                    // Get the air associated with the air_instance
                    let air_name = &pctx.global_info.airs[airgroup_id][air_id].name;

                    let setup = sctx.get_setup(airgroup_id, air_id);
                    let p_expressions_bin = setup.p_setup.p_expressions_bin;

                    let im_hints = get_hint_ids_by_name(p_expressions_bin, "im_col");
                    let im_airval_hints = get_hint_ids_by_name(p_expressions_bin, "im_airval");
                    let gsum_hints = get_hint_ids_by_name(p_expressions_bin, "gsum_col");

                    let im_total_hints: Vec<u64> = im_hints.iter().chain(im_airval_hints.iter()).cloned().collect();

                    let n_im_total_hints = im_total_hints.len();

                    if !im_total_hints.is_empty() {
                        mul_hint_fields(
                            &sctx,
                            &pctx,
                            *instance_id,
                            im_total_hints.len() as u64,
                            im_total_hints,
                            vec!["reference"; n_im_total_hints],
                            vec!["numerator"; n_im_total_hints],
                            vec![HintFieldOptions::default(); n_im_total_hints],
                            vec!["denominator"; n_im_total_hints],
                            vec![HintFieldOptions::inverse(); n_im_total_hints],
                        );
                    }

                    // We know that at most one sumuct hint exists
                    let gsum_hint = if gsum_hints.len() > 1 {
                        panic!("Multiple sumuct hints found for AIR '{air_name}'");
                    } else {
                        gsum_hints[0] as usize
                    };

                    let std_mode = self.std_mode[i];
                    let result = match std_mode {
                        STD_MODE_DEFAULT => Some("result"),
                        STD_MODE_ONE_INSTANCE => None,
                        _ => panic!("Invalid std_mode: {std_mode}"),
                    };
                    // This call accumulates "expression" into "reference" expression and stores its last value to "result"
                    // Alternatively, this could be done using get_hint_field and set_hint_field methods and doing the accumulation in Rust
                    acc_mul_hint_fields(
                        &sctx,
                        &pctx,
                        *instance_id,
                        gsum_hint,
                        "reference",
                        result,
                        "numerator_air",
                        "denominator_air",
                        HintFieldOptions::default(),
                        HintFieldOptions::inverse(),
                        true,
                    );

                    update_airgroupvalue(
                        &sctx,
                        &pctx,
                        *instance_id,
                        gsum_hint,
                        result,
                        "numerator_direct",
                        "denominator_direct",
                        HintFieldOptions::default(),
                        HintFieldOptions::inverse(),
                        true,
                    );
                }
            }
        }
    }

    fn debug(&self, pctx: Arc<ProofCtx<F>>, sctx: Arc<SetupCtx<F>>, instance_ids: &[usize]) {
        timer_start_info!(DEBUG_MODE_SUM);
        if self.num_users > 0 {
            let instances = pctx.dctx_get_instances();
            let my_instances = pctx.dctx_get_process_instances();
            let mut global_instance_ids = Vec::new();
            for i in 0..self.num_users {
                let airgroup_id = self.airgroup_ids[i];
                let air_id = self.air_ids[i];

                // Get all air instances ids for this airgroup and air_id
                for instance_id in my_instances.iter() {
                    if instances[*instance_id].airgroup_id == airgroup_id
                        && instances[*instance_id].air_id == air_id
                        && instance_ids.contains(instance_id)
                        && !skip_prover_instance(&pctx, *instance_id).0
                    {
                        global_instance_ids.push(instance_id);
                    }
                }
            }

            let fast_mode = pctx.debug_info.read().unwrap().std_mode.fast_mode;
            if fast_mode {
                // Process each sum check user
                let debugs_data_fasts: Vec<HashMap<F, SharedDataFast>> = global_instance_ids
                    .par_iter()
                    .map(|&global_instance_id| {
                        if !instance_ids.contains(global_instance_id) {
                            return HashMap::new();
                        }

                        let mut local_debug_data_fast = HashMap::new();

                        Self::extract_hint_fields(
                            &pctx,
                            &sctx,
                            *global_instance_id,
                            &mut HashMap::new(),
                            &mut local_debug_data_fast,
                            true,
                        );

                        local_debug_data_fast
                    })
                    .collect();

                for debug_data_fast in debugs_data_fasts.iter() {
                    self.debug_data_fast.write().unwrap().push(debug_data_fast.clone());
                }
            } else {
                // Process each sum check user
                let mut debug_data = self.debug_data.write().unwrap();
                for global_instance_id in global_instance_ids {
                    Self::extract_hint_fields(
                        &pctx,
                        &sctx,
                        *global_instance_id,
                        &mut debug_data,
                        &mut HashMap::new(),
                        false,
                    );
                }
            }
        }
        timer_stop_and_log_info!(DEBUG_MODE_SUM);
    }

    fn end(&self, pctx: Arc<ProofCtx<F>>, sctx: Arc<SetupCtx<F>>, debug_info: &DebugInfo) {
        if debug_info.std_mode.name == ModeName::Debug || !debug_info.debug_instances.is_empty() {
            let fast_mode = debug_info.std_mode.fast_mode;

            // Perform the global hint update
            if fast_mode {
                let mut local_debug_data_fast = HashMap::new();
                Self::extract_global_hint_fields(&pctx, &sctx, &mut HashMap::new(), &mut local_debug_data_fast, true);
                self.debug_data_fast.write().unwrap().push(local_debug_data_fast);
            } else {
                let mut debug_data = self.debug_data.write().unwrap();
                Self::extract_global_hint_fields(&pctx, &sctx, &mut debug_data, &mut HashMap::new(), false);
            }

            // At the end, check all the debug data
            if fast_mode {
                let mut debug_data_fast = self.debug_data_fast.write().unwrap();
                check_invalid_opids(&pctx, &mut debug_data_fast);
            } else {
                let mut debug_data = self.debug_data.write().unwrap();
                let max_values_to_print = debug_info.std_mode.n_vals;
                let print_to_file = debug_info.std_mode.print_to_file;
                print_debug_info(&pctx, max_values_to_print, print_to_file, &mut debug_data);
            }
        }
    }
}

impl<F: PrimeField64> StdSum<F> {
    const SUM_TYPE_ASSUMES: u64 = 0;
    const SUM_TYPE_PROVES: u64 = 1;
    const SUM_TYPE_FREE: u64 = 2;

    fn extract_global_hint_fields(
        pctx: &ProofCtx<F>,
        sctx: &SetupCtx<F>,
        debug_data: &mut DebugData<F>,
        debug_data_fast: &mut DebugDataFast<F>,
        fast_mode: bool,
    ) {
        let gsum_debug_data = get_hint_ids_by_name(sctx.get_global_bin(), "gsum_debug_data_global");
        if !gsum_debug_data.is_empty() {
            let num_global_hints =
                get_global_hint_field_constant_as::<usize, F>(sctx, gsum_debug_data[0], "num_global_hints");
            for i in 0..num_global_hints {
                let airgroup_id =
                    get_global_hint_field_constant_as::<usize, F>(sctx, gsum_debug_data[1 + i], "airgroup_id");
                let name_piop = get_global_hint_field_constant_as_string(sctx, gsum_debug_data[1 + i], "name_piop");
                let type_piop = get_global_hint_field_constant_as::<u64, F>(sctx, gsum_debug_data[1 + i], "type_piop");
                assert!(
                    type_piop == Self::SUM_TYPE_ASSUMES
                        || type_piop == Self::SUM_TYPE_PROVES
                        || type_piop == Self::SUM_TYPE_FREE,
                    "Invalid type_piop: {type_piop}"
                );

                let opid = get_global_hint_field(sctx, gsum_debug_data[1 + i], "busid");

                // If opids are specified, then only update the bus if the opid is in the list
                if !pctx.debug_info.read().unwrap().std_mode.opids.is_empty()
                    && !pctx.debug_info.read().unwrap().std_mode.opids.contains(&opid.as_canonical_u64())
                {
                    continue;
                }

                let num_reps = get_hint_field_gc(pctx, sctx, gsum_debug_data[1 + i], "num_reps", false);

                // If the number of repetitions is zero, continue
                let mut num_reps = get_row_field_value(&num_reps, 0, "num_reps");
                if num_reps.is_zero() {
                    continue;
                }

                // If the type_piop is free and the num_reps is minus_one, simply flip the num_reps
                if type_piop == Self::SUM_TYPE_FREE {
                    if num_reps == F::NEG_ONE {
                        num_reps = -num_reps;
                    } else if num_reps != F::ONE {
                        panic!(
                            "The number of repetitions in a free piop can only be {{-1, 0, 1}}, received: {num_reps}"
                        );
                    }
                }

                let name_exprs = get_global_hint_field_constant_a_as_string(sctx, gsum_debug_data[1 + i], "name_exprs");

                let expressions = get_hint_field_gc_a(pctx, sctx, gsum_debug_data[1 + i], "expressions", false);
                let is_proves = type_piop == Self::SUM_TYPE_PROVES;
                if fast_mode {
                    update_debug_data_fast(debug_data_fast, opid, expressions.get(0), is_proves, num_reps, true);
                } else {
                    update_debug_data(
                        debug_data,
                        &name_piop,
                        &name_exprs,
                        opid,
                        expressions.get(0),
                        airgroup_id,
                        None,
                        None,
                        0,
                        is_proves,
                        num_reps,
                        true,
                    );
                }
            }
        }
    }

    fn extract_hint_fields(
        pctx: &ProofCtx<F>,
        sctx: &SetupCtx<F>,
        instance_id: usize,
        debug_data: &mut DebugData<F>,
        debug_data_fast: &mut DebugDataFast<F>,
        fast_mode: bool,
    ) {
        let (airgroup_id, air_id) = pctx.dctx_get_instance_info(instance_id);
        let air_instance_id = pctx.dctx_find_air_instance_id(instance_id);

        let setup = sctx.get_setup(airgroup_id, air_id);
        let p_expressions_bin = setup.p_setup.p_expressions_bin;

        let debug_data_hints = get_hint_ids_by_name(p_expressions_bin, "gsum_debug_data");

        let num_rows = pctx.global_info.airs[airgroup_id][air_id].num_rows;

        // Process each debug hint
        for &hint in debug_data_hints.iter() {
            // Extract hint fields
            let name_piop = get_hint_field_constant_as_string(
                sctx,
                airgroup_id,
                air_id,
                hint as usize,
                "name_piop",
                HintFieldOptions::default(),
            );

            let name_exprs = get_hint_field_constant_a_as_string(
                sctx,
                airgroup_id,
                air_id,
                hint as usize,
                "name_exprs",
                HintFieldOptions::default(),
            );

            let busid = get_hint_field(sctx, pctx, instance_id, hint as usize, "busid", HintFieldOptions::default());

            let type_piop = get_hint_field_constant_as::<u64, F>(
                sctx,
                airgroup_id,
                air_id,
                hint as usize,
                "type_piop",
                HintFieldOptions::default(),
            );
            assert!(
                type_piop == Self::SUM_TYPE_ASSUMES
                    || type_piop == Self::SUM_TYPE_PROVES
                    || type_piop == Self::SUM_TYPE_FREE,
                "Invalid type_piop: {type_piop}"
            );

            let num_reps =
                get_hint_field(sctx, pctx, instance_id, hint as usize, "num_reps", HintFieldOptions::default());

            let expressions =
                get_hint_field_a(sctx, pctx, instance_id, hint as usize, "expressions", HintFieldOptions::default());

            let deg_expr = get_hint_field_constant_as_field(
                sctx,
                airgroup_id,
                air_id,
                hint as usize,
                "deg_expr",
                HintFieldOptions::default(),
            );

            let deg_mul = get_hint_field_constant_as_field(
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
                        let opids = &pctx.debug_info.read().unwrap().std_mode.opids;
                        if !opids.is_empty() && !opids.contains(&opid.as_canonical_u64()) {
                            continue;
                        }
                        opid
                    }
                    _ => panic!("busid must be a field element"),
                };

                Self::update_bus(
                    &name_piop,
                    &name_exprs,
                    airgroup_id,
                    air_id,
                    air_instance_id,
                    opid,
                    type_piop,
                    &num_reps,
                    &expressions,
                    0,
                    debug_data,
                    debug_data_fast,
                    false,
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
                            let opids = &pctx.debug_info.read().unwrap().std_mode.opids;
                            if !opids.is_empty() && !opids.contains(&opid.as_canonical_u64()) {
                                continue;
                            }

                            opid
                        }
                        _ => panic!("busid must be a field element"),
                    };

                    Self::update_bus(
                        &name_piop,
                        &name_exprs,
                        airgroup_id,
                        air_id,
                        air_instance_id,
                        opid,
                        type_piop,
                        &num_reps,
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
    }

    #[allow(clippy::too_many_arguments)]
    fn update_bus(
        name_piop: &str,
        name_exprs: &[String],
        airgroup_id: usize,
        air_id: usize,
        instance_id: usize,
        opid: F,
        type_piop: u64,
        num_reps: &HintFieldValue<F>,
        expressions: &HintFieldValuesVec<F>,
        row: usize,
        debug_data: &mut DebugData<F>,
        debug_data_fast: &mut DebugDataFast<F>,
        is_global: bool,
        fast_mode: bool,
    ) {
        let mut num_reps = get_row_field_value(num_reps, row, "num_reps");
        if num_reps.is_zero() {
            return;
        }

        let is_proves = match type_piop {
            Self::SUM_TYPE_ASSUMES => false,
            Self::SUM_TYPE_PROVES => true,
            Self::SUM_TYPE_FREE => {
                if num_reps == F::NEG_ONE {
                    // If the type is free and the num_reps is minus_one, simply flip the num_reps
                    num_reps = -num_reps;
                    false
                } else if num_reps == F::ONE {
                    true
                } else {
                    panic!("The number of repetitions in a free piop can only be {{-1, 0, 1}}, received: {num_reps}");
                }
            }
            _ => unreachable!(),
        };
        if fast_mode {
            update_debug_data_fast(debug_data_fast, opid, expressions.get(row), is_proves, num_reps, is_global);
        } else {
            update_debug_data(
                debug_data,
                name_piop,
                name_exprs,
                opid,
                expressions.get(row),
                airgroup_id,
                Some(air_id),
                Some(instance_id),
                row,
                is_proves,
                num_reps,
                is_global,
            );
        }
    }
}
