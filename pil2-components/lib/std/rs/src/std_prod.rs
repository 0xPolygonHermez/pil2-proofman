use std::{
    collections::HashMap,
    sync::{Arc, Mutex},
};

use num_traits::ToPrimitive;
use p3_field::PrimeField;

use proofman::{WitnessComponent, WitnessManager};
use proofman_common::{AirInstance, ModeName, ProofCtx, SetupCtx, StdMode};
use proofman_hints::{
    get_hint_field, get_hint_field_a, get_hint_field_constant, get_hint_field_constant_a, get_hint_ids_by_name,
    update_airgroupvalue, acc_mul_hint_fields, HintFieldOptions, HintFieldOutput, HintFieldValue, HintFieldValuesVec,
};

use crate::{print_debug_info, update_debug_data, DebugData, Decider};

type ProdAirsItem = (usize, usize, Vec<u64>, Vec<u64>); // (airgroup_id, air_id, gprod_hints, debug_hints_data, debug_hints)

pub struct StdProd<F: PrimeField> {
    mode: StdMode,
    prod_airs: Mutex<Vec<ProdAirsItem>>,
    debug_data: Option<DebugData<F>>,
}

impl<F: PrimeField> Decider<F> for StdProd<F> {
    fn decide(&self, sctx: Arc<SetupCtx>) {
        // Scan the pilout for airs that have prod-related hints

        for (airgroup_id, air_id) in sctx.get_setups_list() {
            let setup = sctx.get_setup(airgroup_id, air_id);
            let p_expressions_bin = setup.p_setup.p_expressions_bin;

            let gprod_hints = get_hint_ids_by_name(p_expressions_bin, "gprod_col");
            let debug_hints_data = get_hint_ids_by_name(p_expressions_bin, "gprod_member_data");
            if !gprod_hints.is_empty() {
                // Save the air for latter witness computation
                self.prod_airs.lock().unwrap().push((airgroup_id, air_id, gprod_hints, debug_hints_data));
            }
        }
    }
}

impl<F: PrimeField> StdProd<F> {
    const MY_NAME: &'static str = "STD Prod";

    pub fn new(mode: StdMode, wcm: Arc<WitnessManager<F>>) -> Arc<Self> {
        let std_prod = Arc::new(Self {
            mode: mode.clone(),
            prod_airs: Mutex::new(Vec::new()),
            debug_data: if mode.name == ModeName::Debug { Some(Mutex::new(HashMap::new())) } else { None },
        });

        wcm.register_proxy_component(std_prod.clone());

        std_prod
    }

    fn debug(
        &self,
        pctx: &ProofCtx<F>,
        sctx: &SetupCtx,
        air_instance: &mut AirInstance<F>,
        num_rows: usize,
        debug_hints_data: Vec<u64>,
    ) {
        let debug_data = self.debug_data.as_ref().expect("Debug data missing");
        let airgroup_id = air_instance.airgroup_id;
        let air_id = air_instance.air_id;
        let instance_id = air_instance.air_instance_id.unwrap_or_default();

        for hint in debug_hints_data.iter() {
            let _name_piop = get_hint_field_constant::<F>(
                sctx,
                airgroup_id,
                air_id,
                *hint as usize,
                "name_piop",
                HintFieldOptions::default(),
            );

            let _name_expr = get_hint_field_constant_a::<F>(
                sctx,
                airgroup_id,
                air_id,
                *hint as usize,
                "name_expr",
                HintFieldOptions::default(),
            );

            let opid =
                get_hint_field::<F>(sctx, pctx, air_instance, *hint as usize, "opid", HintFieldOptions::default());
            let opid = if let HintFieldValue::Field(opid) = opid {
                if !self.mode.opids.is_empty()
                    && !self.mode.opids.contains(&opid.as_canonical_biguint().to_u64().expect("Cannot convert to u64"))
                {
                    continue;
                }

                opid
            } else {
                panic!("sumid must be a field element");
            };

            let HintFieldValue::Field(is_global) = get_hint_field_constant::<F>(
                sctx,
                airgroup_id,
                air_id,
                *hint as usize,
                "is_global",
                HintFieldOptions::default(),
            ) else {
                log::error!("is_global hint must be a field element");
                panic!();
            };

            let HintFieldValue::Field(proves) = get_hint_field_constant::<F>(
                sctx,
                airgroup_id,
                air_id,
                *hint as usize,
                "proves",
                HintFieldOptions::default(),
            ) else {
                log::error!("proves hint must be a field element");
                panic!();
            };
            let proves = if proves.is_zero() {
                false
            } else if proves.is_one() {
                true
            } else {
                log::error!("Proves hint must be either 0 or 1");
                panic!();
            };

            let selector =
                get_hint_field::<F>(sctx, pctx, air_instance, *hint as usize, "selector", HintFieldOptions::default());

            let expressions = get_hint_field_a::<F>(
                sctx,
                pctx,
                air_instance,
                *hint as usize,
                "expressions",
                HintFieldOptions::default(),
            );

            let HintFieldValue::Field(deg_expr) = get_hint_field_constant::<F>(
                sctx,
                airgroup_id,
                air_id,
                *hint as usize,
                "deg_expr",
                HintFieldOptions::default(),
            ) else {
                log::error!("deg_expr hint must be a field element");
                panic!();
            };

            let HintFieldValue::Field(deg_sel) = get_hint_field_constant::<F>(
                sctx,
                airgroup_id,
                air_id,
                *hint as usize,
                "deg_sel",
                HintFieldOptions::default(),
            ) else {
                log::error!("deg_sel hint must be a field element");
                panic!();
            };

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
            } else {
                // Otherwise, update the bus for each row
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
                let sel = if let HintFieldOutput::Field(selector) = selector.get(row) {
                    if !selector.is_zero() && !selector.is_one() {
                        log::error!("Selector must be either 0 or 1");
                        panic!();
                    }
                    selector.is_one()
                } else {
                    log::error!("Selector must be a field element");
                    panic!();
                };

                if sel {
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
}

impl<F: PrimeField> WitnessComponent<F> for StdProd<F> {
    fn start_proof(&self, _pctx: Arc<ProofCtx<F>>, sctx: Arc<SetupCtx>) {
        self.decide(sctx);
    }

    fn calculate_witness(&self, stage: u32, _air_instance: Option<usize>, pctx: Arc<ProofCtx<F>>, sctx: Arc<SetupCtx>) {
        if stage == 2 {
            let prod_airs = self.prod_airs.lock().unwrap();

            for (airgroup_id, air_id, gprod_hints, debug_hints_data) in prod_airs.iter() {
                let air_instance_ids = pctx.air_instance_repo.find_air_instances(*airgroup_id, *air_id);

                for air_instance_id in air_instance_ids {
                    let air_instances_vec = &mut pctx.air_instance_repo.air_instances.write().unwrap();
                    let air_instance = &mut air_instances_vec[air_instance_id];

                    // Get the air associated with the air_instance
                    let airgroup_id = air_instance.airgroup_id;
                    let air_id = air_instance.air_id;
                    let air_name = &pctx.global_info.airs[airgroup_id][air_id].name;

                    log::debug!("{}: ··· Computing witness for AIR '{}' at stage {}", Self::MY_NAME, air_name, stage);

                    let num_rows = pctx.global_info.airs[airgroup_id][air_id].num_rows;

                    if self.mode.name == ModeName::Debug {
                        self.debug(&pctx, &sctx, air_instance, num_rows, debug_hints_data.clone());
                    }

                    // We know that at most one product hint exists
                    let gprod_hint = if gprod_hints.len() > 1 {
                        panic!("Multiple product hints found for AIR '{}'", air_name);
                    } else {
                        gprod_hints[0] as usize
                    };

                    // This call calculates "numerator" / "denominator" and accumulates it into "reference". Its last value is stored into "result"
                    // Alternatively, this could be done using get_hint_field and set_hint_field methods and calculating the operations in Rust,
                    acc_mul_hint_fields::<F>(
                        &sctx,
                        &pctx,
                        air_instance,
                        gprod_hint,
                        "reference",
                        "result",
                        "numerator_air",
                        "denominator_air",
                        HintFieldOptions::default(),
                        HintFieldOptions::inverse(),
                        false,
                    );

                    update_airgroupvalue::<F>(
                        &sctx,
                        &pctx,
                        air_instance,
                        gprod_hint,
                        "result",
                        "numerator_direct",
                        "denominator_direct",
                        HintFieldOptions::default(),
                        HintFieldOptions::inverse(),
                        false,
                    );
                }
            }
        }
    }

    fn end_proof(&self) {
        if self.mode.name == ModeName::Debug {
            let name = Self::MY_NAME;
            let max_values_to_print = self.mode.n_vals;
            let print_to_file = self.mode.print_to_file;
            let debug_data = self.debug_data.as_ref().expect("Debug data missing");
            print_debug_info(name, max_values_to_print, print_to_file, debug_data);
        }
    }
}
