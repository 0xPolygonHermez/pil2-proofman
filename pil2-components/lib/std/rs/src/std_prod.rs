use core::panic;
use std::{
    collections::HashMap,
    sync::{Arc, Mutex},
};

use num_traits::ToPrimitive;
use p3_field::PrimeField;

use proofman::{WitnessComponent, WitnessManager};
use proofman_common::{AirInstance, ExecutionCtx, ProofCtx, SetupCtx};
use proofman_hints::{
    get_hint_field, get_hint_field_a, get_hint_ids_by_name, set_hint_field, set_hint_field_val, HintFieldOptions,
    HintFieldOutput, HintFieldValue,
};

use crate::{check_bus_values, BusValue, DebugData, Decider, ModeName, StdMode};

type ProdAirsItem = (usize, usize, Vec<u64>, Vec<u64>);

pub struct StdProd<F: PrimeField> {
    mode: StdMode,
    prod_airs: Mutex<Vec<ProdAirsItem>>, // (airgroup_id, air_id, gprod_hints, debug_hints_data, debug_hints)
    debug_data: Option<DebugData<F>>,
}

impl<F: PrimeField> Decider<F> for StdProd<F> {
    fn decide(&self, sctx: Arc<SetupCtx>, pctx: Arc<ProofCtx<F>>) {
        // Scan the pilout for airs that have prod-related hints
        for airgroup in pctx.pilout.air_groups() {
            for air in airgroup.airs() {
                let airgroup_id = air.airgroup_id;
                let air_id = air.air_id;

                let setup = sctx.get_partial_setup(airgroup_id, air_id).expect("REASON");
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
}

impl<F: PrimeField> StdProd<F> {
    const MY_NAME: &'static str = "STD Prod";

    pub fn new(mode: StdMode, wcm: Arc<WitnessManager<F>>) -> Arc<Self> {
        let std_prod = Arc::new(Self {
            mode: mode.clone(),
            prod_airs: Mutex::new(Vec::new()),
            debug_data: if mode.name == ModeName::Debug {
                Some(DebugData { bus_values: Mutex::new(HashMap::new()), opid_metadata: Mutex::new(HashMap::new()) })
            } else {
                None
            },
        });

        wcm.register_component(std_prod.clone(), None, None);

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
        let air_group = pctx.pilout.get_air_group(air_instance.airgroup_id);
        let air = pctx.pilout.get_air(air_instance.airgroup_id, air_instance.air_id);
        let air_group_name = air_group.name().unwrap_or("Unknown air group").to_string();
        let air_name = air.name().unwrap_or("Unknown air").to_string();

        let debug_data = self.debug_data.as_ref().expect("Debug data missing");
        let mut opid_metadata = debug_data.opid_metadata.lock().expect("Opid metadata missing");
        for hint in debug_hints_data.iter() {
            let opid = get_hint_field::<F>(
                sctx,
                &pctx.public_inputs,
                &pctx.challenges,
                air_instance,
                *hint as usize,
                "opid",
                HintFieldOptions::default(),
            );
            let opid = if let HintFieldValue::Field(opid) = opid {
                if let Some(mode_opids) = &self.mode.opids {
                    if !mode_opids.contains(&opid.as_canonical_biguint().to_u64().expect("Cannot convert to u64")) {
                        continue;
                    }
                }
                opid.as_canonical_biguint().to_u64().expect("Cannot convert to u64")
            } else {
                panic!("sumid must be a field element");
            };

            let proves = get_hint_field::<F>(
                sctx,
                &pctx.public_inputs,
                &pctx.challenges,
                air_instance,
                *hint as usize,
                "proves",
                HintFieldOptions::default(),
            );
            let proves = if let HintFieldValue::Field(proves) = proves {
                if !proves.is_zero() && !proves.is_one() {
                    log::error!("Proves hint must be either 0 or 1");
                    panic!();
                }
                proves.is_one()
            } else {
                log::error!("Proves hint must be a field element");
                panic!();
            };

            // Store the name of the PIOP
            let name = get_hint_field::<F>(
                sctx,
                &pctx.public_inputs,
                &pctx.challenges,
                air_instance,
                *hint as usize,
                "name_piop",
                HintFieldOptions::default(),
            );
            match name {
                HintFieldValue::String(name) => {
                    opid_metadata.entry(opid).or_default().push((
                        air_group_name.clone(),
                        air_name.clone(),
                        name.clone(),
                        proves,
                        Vec::new(),
                    ));
                }
                _ => {
                    log::error!("Name hint must be a string");
                    panic!("Name hint must be a string");
                }
            };

            // Store the names of the expressions
            let names = get_hint_field_a::<F>(
                sctx,
                &pctx.public_inputs,
                &pctx.challenges,
                air_instance,
                *hint as usize,
                "names",
                HintFieldOptions::default(),
            );
            names.values.iter().for_each(|name| match name {
                HintFieldValue::String(name) => {
                    let metadata = opid_metadata.get_mut(&opid).expect("Metadata missing");
                    metadata.last_mut().expect("Last metadata missing").4.push(name.clone());
                }
                _ => {
                    log::error!("Names hint must be a string");
                    panic!("Names hint must be a string");
                }
            });

            let selector = get_hint_field::<F>(
                sctx,
                &pctx.public_inputs,
                &pctx.challenges,
                air_instance,
                *hint as usize,
                "selector",
                HintFieldOptions::default(),
            );

            let expressions = get_hint_field_a::<F>(
                sctx,
                &pctx.public_inputs,
                &pctx.challenges,
                air_instance,
                *hint as usize,
                "references",
                HintFieldOptions::default(),
            );

            for j in 0..num_rows {
                let sel = if let HintFieldOutput::Field(selector) = selector.get(j) {
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
                    self.update_bus_vals(num_rows, opid, expressions.get(j), j, proves);
                }
            }
        }
    }

    fn update_bus_vals(&self, num_rows: usize, opid: u64, val: Vec<HintFieldOutput<F>>, row: usize, is_num: bool) {
        let debug_data = self.debug_data.as_ref().expect("Debug data missing");
        let mut bus_values = debug_data.bus_values.lock().expect("Bus values missing");

        let bus_opid = bus_values.entry(opid).or_default();

        let bus_val = bus_opid.entry(val).or_insert_with(|| BusValue {
            num_proves: F::zero(),
            num_assumes: F::zero(),
            row_proves: Vec::with_capacity(num_rows),
            row_assumes: Vec::with_capacity(num_rows),
        });

        if is_num {
            bus_val.num_proves += F::one();
            bus_val.row_proves.push(row);
        } else {
            bus_val.num_assumes += F::one();
            bus_val.row_assumes.push(row);
        }
    }
}

impl<F: PrimeField> WitnessComponent<F> for StdProd<F> {
    fn start_proof(&self, pctx: Arc<ProofCtx<F>>, _ectx: Arc<ExecutionCtx>, sctx: Arc<SetupCtx>) {
        self.decide(sctx, pctx);
    }

    fn calculate_witness(
        &self,
        stage: u32,
        _air_instance: Option<usize>,
        pctx: Arc<ProofCtx<F>>,
        _ectx: Arc<ExecutionCtx>,
        sctx: Arc<SetupCtx>,
    ) {
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
                    let air = pctx.pilout.get_air(airgroup_id, air_id);
                    let air_name = air.name().unwrap_or("unknown");

                    log::debug!("{}: ··· Computing witness for AIR '{}' at stage {}", Self::MY_NAME, air_name, stage);

                    let num_rows = air.num_rows();

                    if self.mode.name == ModeName::Debug {
                        self.debug(&pctx, &sctx, air_instance, num_rows, debug_hints_data.clone());
                    }

                    // We know that at most one product hint exists
                    let gprod_hint = if gprod_hints.len() > 1 {
                        panic!("Multiple product hints found for AIR '{}'", air.name().unwrap_or("unknown"));
                    } else {
                        gprod_hints[0] as usize
                    };

                    // Use the hint to populate the gprod column
                    let mut gprod = get_hint_field::<F>(
                        &sctx,
                        &pctx.public_inputs,
                        &pctx.challenges,
                        air_instance,
                        gprod_hint,
                        "reference",
                        HintFieldOptions::dest(),
                    );
                    let num = get_hint_field::<F>(
                        &sctx,
                        &pctx.public_inputs,
                        &pctx.challenges,
                        air_instance,
                        gprod_hint,
                        "numerator",
                        HintFieldOptions::default(),
                    );
                    let den = get_hint_field::<F>(
                        &sctx,
                        &pctx.public_inputs,
                        &pctx.challenges,
                        air_instance,
                        gprod_hint,
                        "denominator",
                        HintFieldOptions::default(),
                    );

                    gprod.set(0, num.get(0) / den.get(0));
                    for i in 1..num_rows {
                        gprod.set(i, gprod.get(i - 1) * (num.get(i) / den.get(i)));
                    }

                    // set the computed gprod column and its associated airgroup_val
                    set_hint_field(&sctx, air_instance, gprod_hint as u64, "reference", &gprod);
                    set_hint_field_val(&sctx, air_instance, gprod_hint as u64, "result", gprod.get(num_rows - 1));
                }
            }
        }
    }

    fn end_proof(&self) {
        if self.mode.name == ModeName::Debug {
            check_bus_values(Self::MY_NAME, self.mode.vals_to_print, &self.debug_data);
        }
    }
}
