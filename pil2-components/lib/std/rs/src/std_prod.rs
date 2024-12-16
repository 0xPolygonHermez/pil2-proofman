use std::{
    collections::HashMap,
    sync::{Arc, Mutex},
};

use p3_field::PrimeField;

use proofman::{get_hint_field_gc_constant_a, WitnessComponent, WitnessManager};
use proofman_common::{AirInstance, ExecutionCtx, ModeName, ProofCtx, SetupCtx, StdMode};
use proofman_hints::{
    get_hint_field, get_hint_field_a, get_hint_field_constant, get_hint_field_constant_a, acc_mul_hint_fields,
    update_airgroupvalue, get_hint_ids_by_name, HintFieldOptions, HintFieldValue, HintFieldValuesVec,
};

use crate::{
    print_debug_info, update_debug_data, DebugData, get_global_hint_field_constant_as, get_hint_field_constant_as_field,
    get_row_field_value, extract_field_element_as_usize,
};

pub struct StdProd<F: PrimeField> {
    mode: StdMode,
    stage_wc: Option<Mutex<u32>>,
    debug_data: Option<DebugData<F>>,
}

impl<F: PrimeField> StdProd<F> {
    const MY_NAME: &'static str = "STD Prod";

    pub fn new(mode: StdMode, wcm: Arc<WitnessManager<F>>) {
        let sctx = wcm.get_sctx();

        // Retrieve the std_prod_users hint ID
        let std_prod_users_id = get_hint_ids_by_name(sctx.get_global_bin(), "std_prod_users");

        // Initialize std_prod with the extracted data
        let std_prod = Arc::new(Self {
            mode: mode.clone(),
            stage_wc: match std_prod_users_id.is_empty() {
                true => None,
                false => {
                    // Get the "stage_wc" hint
                    let stage_wc = get_global_hint_field_constant_as::<u32, F>(sctx.clone(), std_prod_users_id[0], "stage_wc");
                    Some(Mutex::new(stage_wc))
                }
            },
            debug_data: if mode.name == ModeName::Debug { Some(Mutex::new(HashMap::new())) } else { None },
        });

        // Register the component
        wcm.register_component(std_prod.clone(), None, None);
    }

    fn debug(
        &self,
        pctx: &ProofCtx<F>,
        sctx: &SetupCtx,
        air_instance: &mut AirInstance<F>,
        num_rows: usize,
        debug_data_hints: Vec<u64>,
    ) {
        let debug_data = self.debug_data.as_ref().expect("Debug data missing");
        let airgroup_id = air_instance.airgroup_id;
        let air_id = air_instance.air_id;
        let instance_id = air_instance.air_instance_id.unwrap_or_default();

        // Process each debug hint
        for &hint in debug_data_hints.iter() {
            // Extract hint fields
            let _name_piop = get_hint_field_constant::<F>(
                sctx,
                airgroup_id,
                air_id,
                hint as usize,
                "name_piop",
                HintFieldOptions::default(),
            );

            let _name_expr = get_hint_field_constant_a::<F>(
                sctx,
                airgroup_id,
                air_id,
                hint as usize,
                "name_expr",
                HintFieldOptions::default(),
            );

            let opid = get_hint_field_constant_as_field::<F>(
                sctx,
                airgroup_id,
                air_id,
                hint as usize,
                "busid",
                HintFieldOptions::default(),
            );

            let is_global = get_hint_field_constant_as_field::<F>(
                sctx,
                airgroup_id,
                air_id,
                hint as usize,
                "is_global",
                HintFieldOptions::default(),
            );

            let proves = get_hint_field_constant_as_field::<F>(
                sctx,
                airgroup_id,
                air_id,
                hint as usize,
                "proves",
                HintFieldOptions::default(),
            );
            let proves = if proves.is_zero() {
                false
            } else if proves.is_one() {
                true
            } else {
                log::error!("Proves hint must be either 0 or 1");
                panic!();
            };

            let selector: HintFieldValue<F> =
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

            let deg_sel = get_hint_field_constant_as_field::<F>(
                sctx,
                airgroup_id,
                air_id,
                hint as usize,
                "deg_sel",
                HintFieldOptions::default(),
            );

            // If both the expresion and the mul are of degree zero, then simply update the bus once
            if deg_expr.is_zero() && deg_sel.is_zero() {
                update_bus(
                    airgroup_id,
                    air_id,
                    instance_id,
                    opid,
                    proves,
                    &selector,
                    &expressions,
                    0,
                    debug_data,
                    is_global.is_one(),
                );
            }
            // Otherwise, update the bus for each row
            else {
                for j in 0..num_rows {
                    update_bus(
                        airgroup_id,
                        air_id,
                        instance_id,
                        opid,
                        proves,
                        &selector,
                        &expressions,
                        j,
                        debug_data,
                        false,
                    );
                }
            }

            #[allow(clippy::too_many_arguments)]
            fn update_bus<F: PrimeField>(
                airgroup_id: usize,
                air_id: usize,
                instance_id: usize,
                opid: F,
                proves: bool,
                selector: &HintFieldValue<F>,
                expressions: &HintFieldValuesVec<F>,
                row: usize,
                debug_data: &DebugData<F>,
                is_global: bool,
            ) {
                let selector = get_row_field_value(selector, row, "sel");
                if selector.is_zero() {
                    return;
                }

                update_debug_data(
                    debug_data,
                    opid,
                    expressions.get(row),
                    airgroup_id,
                    air_id,
                    instance_id,
                    row,
                    proves,
                    F::one(),
                    is_global,
                );
            }
        }
    }
}

impl<F: PrimeField> WitnessComponent<F> for StdProd<F> {
    fn calculate_witness(
        &self,
        stage: u32,
        _air_instance: Option<usize>,
        pctx: Arc<ProofCtx<F>>,
        _ectx: Arc<ExecutionCtx>,
        sctx: Arc<SetupCtx>,
    ) {
        let stage_wc = self.stage_wc.as_ref();
        if stage_wc.is_none() {
            return;
        }

        if stage == *stage_wc.unwrap().lock().unwrap() {
            // Get the number of product check users and their airgroup and air IDs
            let std_prod_users = get_hint_ids_by_name(sctx.get_global_bin(), "std_prod_users")[0];

            let num_users = get_global_hint_field_constant_as::<usize, F>(sctx.clone(), std_prod_users, "num_users");
            let airgroup_ids = get_hint_field_gc_constant_a::<F>(sctx.clone(), std_prod_users, "airgroup_ids", false);
            let air_ids = get_hint_field_gc_constant_a::<F>(sctx.clone(), std_prod_users, "air_ids", false);

            // Process each product check user
            for i in 0..num_users {
                let airgroup_id = extract_field_element_as_usize(&airgroup_ids.values[i], "airgroup_id");
                let air_id = extract_field_element_as_usize(&air_ids.values[i], "air_id");

                // Get all air instances ids for this airgroup and air_id
                let air_instance_ids = pctx.air_instance_repo.find_air_instances(airgroup_id, air_id);
                for air_instance_id in air_instance_ids {
                    // Retrieve all air instances
                    let air_instaces = &mut pctx.air_instance_repo.air_instances.write().unwrap();
                    let air_instance = &mut air_instaces[air_instance_id];

                    // Get the AIR associated with the air_instance
                    let air = pctx.pilout.get_air(airgroup_id, air_id);
                    let air_name = air.name().unwrap_or("unknown");
                    log::debug!("{}: ··· Computing witness for AIR '{}' at stage {}", Self::MY_NAME, air_name, stage);

                    // Setup and process AIR instance
                    let num_rows = air.num_rows();
                    let setup = sctx.get_setup(airgroup_id, air_id);
                    let p_expressions_bin = setup.p_setup.p_expressions_bin;

                    let gprod_hints = get_hint_ids_by_name(p_expressions_bin, "gprod_col");
                    let debug_data_hints = get_hint_ids_by_name(p_expressions_bin, "gprod_debug_data");

                    // Debugging, if enabled
                    if self.mode.name == ModeName::Debug {
                        self.debug(&pctx, &sctx, air_instance, num_rows, debug_data_hints.clone());
                    }

                    // We know that at most one gprod hint exists
                    if gprod_hints.len() > 1 {
                        panic!("Multiple gprod hints found for AIR '{}'", air_name);
                    }

                    // Process the gprod hints
                    if let Some(&gprod_hint) = gprod_hints.first() {
                        // This call calculates "numerator" / "denominator" and accumulates it into "reference". Its last value is stored into "result"
                        // Alternatively, this could be done using get_hint_field and set_hint_field methods and calculating the operations in Rust,
                        let (pol_id, _) = acc_mul_hint_fields::<F>(
                            &sctx,
                            &pctx,
                            air_instance,
                            gprod_hint as usize,
                            "reference",
                            "result",
                            "numerator_air",
                            "denominator_air",
                            HintFieldOptions::default(),
                            HintFieldOptions::inverse(),
                            false,
                        );
                        air_instance.set_commit_calculated(pol_id as usize);

                        let airgroupvalue_id = update_airgroupvalue::<F>(
                            &sctx,
                            &pctx,
                            air_instance,
                            gprod_hint as usize,
                            "result",
                            "numerator_direct",
                            "denominator_direct",
                            HintFieldOptions::default(),
                            HintFieldOptions::inverse(),
                            false,
                        );
                        air_instance.set_airgroupvalue_calculated(airgroupvalue_id as usize);
                    }
                }
            }
        }
    }

    fn end_proof(&self) {
        // Print debug info if in debug mode
        if self.mode.name == ModeName::Debug {
            let name = Self::MY_NAME;
            let max_values_to_print = self.mode.n_vals;
            let print_to_file = self.mode.print_to_file;
            let debug_data = self.debug_data.as_ref().expect("Debug data missing");
            print_debug_info(name, max_values_to_print, print_to_file, debug_data);
        }
    }
}
