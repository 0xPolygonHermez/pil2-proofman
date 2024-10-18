use std::{
    collections::HashMap,
    fmt::Debug,
    sync::{Arc, Mutex},
};

use num_traits::ToPrimitive;
use p3_field::PrimeField;
use rayon::prelude::*;

use log::debug;

use proofman::{WitnessComponent, WitnessManager};
use proofman_common::{AirInstance, ExecutionCtx, ProofCtx, SetupCtx};
use proofman_hints::{
    get_hint_field, get_hint_field_a, get_hint_ids_by_name, set_hint_field, set_hint_field_val, HintFieldOptions,
    HintFieldOutput, HintFieldValue,
};

use crate::{check_bus_values, BusValue, DebugData, Decider, ModeName, StdMode};

type SumAirsItem = (usize, usize, Vec<u64>, Vec<u64>, Vec<u64>);

pub struct StdSum<F: PrimeField> {
    mode: StdMode,
    sum_airs: Mutex<Vec<SumAirsItem>>, // (airgroup_id, air_id, gsum_hints, im_hints, debug_hints_data, debug_hints)
    debug_data: Option<DebugData<F>>,
}

impl<F: PrimeField> Decider<F> for StdSum<F> {
    fn decide(&self, sctx: Arc<SetupCtx>, pctx: Arc<ProofCtx<F>>) {
        // Scan the pilout for airs that have sum-related hints
        let air_groups = pctx.pilout.air_groups();
        let mut sum_airs_guard = self.sum_airs.lock().unwrap();
        air_groups.iter().for_each(|air_group| {
            let airs = air_group.airs();
            airs.iter().for_each(|air| {
                let airgroup_id = air.airgroup_id;
                let air_id = air.air_id;

                let setup = sctx.get_partial_setup(airgroup_id, air_id).expect("REASON");
                let p_expressions_bin = setup.p_setup.p_expressions_bin;

                let im_hints = get_hint_ids_by_name(p_expressions_bin, "im_col");
                let gsum_hints = get_hint_ids_by_name(p_expressions_bin, "gsum_col");
                let debug_hints_data = get_hint_ids_by_name(p_expressions_bin, "gsum_member_data");
                if !gsum_hints.is_empty() {
                    // Save the air for latter witness computation
                    sum_airs_guard.push((airgroup_id, air_id, im_hints, gsum_hints, debug_hints_data));
                }
            });
        });
    }
}

impl<F: Copy + Debug + PrimeField> StdSum<F> {
    const MY_NAME: &'static str = "STD Sum ";

    pub fn new(mode: StdMode, wcm: Arc<WitnessManager<F>>) -> Arc<Self> {
        let std_sum = Arc::new(Self {
            mode: mode.clone(),
            sum_airs: Mutex::new(Vec::new()),
            debug_data: if mode.name == ModeName::Debug {
                Some(DebugData { bus_values: Mutex::new(HashMap::new()), opid_metadata: Mutex::new(HashMap::new()) })
            } else {
                None
            },
        });

        wcm.register_component(std_sum.clone(), None, None);

        std_sum
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
            let opids = get_hint_field_a::<F>(
                sctx,
                &pctx.public_inputs,
                &pctx.challenges,
                air_instance,
                *hint as usize,
                "opids",
                HintFieldOptions::default(),
            );
            let mut selected_opids = opids
                .values
                .iter()
                .map(|opid| match opid {
                    HintFieldValue::Field(opid) => opid.as_canonical_biguint().to_u64().expect("Cannot convert to u64"),
                    _ => {
                        log::error!("Opid hint must be a field element");
                        panic!("Opid hint must be a field element");
                    }
                })
                .collect::<Vec<_>>();

            // If none of the opids is selected, skip
            let mode_opids = &self.mode.opids;
            if let Some(mode_opids) = mode_opids {
                selected_opids.retain(|opid| mode_opids.contains(opid));
            }
            if selected_opids.is_empty() {
                continue;
            }

            let proves = get_hint_field::<F>(
                sctx,
                &pctx.public_inputs,
                &pctx.challenges,
                air_instance,
                *hint as usize,
                "proves",
                HintFieldOptions::default(),
            );
            let proves = match proves {
                HintFieldValue::Field(proves) => {
                    assert!(proves.is_zero() || proves.is_one(), "Proves hint must be either 0 or 1");
                    proves.is_one()
                }
                _ => {
                    log::error!("Proves hint must be a field element");
                    panic!("Proves hint must be a field element");
                }
            };

            for opid in selected_opids {
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
            }

            let sumid = get_hint_field::<F>(
                sctx,
                &pctx.public_inputs,
                &pctx.challenges,
                air_instance,
                *hint as usize,
                "sumid",
                HintFieldOptions::default(),
            );

            let mul = get_hint_field::<F>(
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
                let mul = match mul.get(j) {
                    HintFieldOutput::Field(mul) => mul,
                    _ => panic!("mul must be a field element"),
                };

                if !mul.is_zero() {
                    let sumid = match sumid.get(j) {
                        HintFieldOutput::Field(sumid) => {
                            sumid.as_canonical_biguint().to_u64().expect("Cannot convert to u64")
                        }
                        _ => panic!("sumid must be a field element"),
                    };

                    self.update_bus_vals(sumid, expressions.get(j), j, proves, mul);
                }
            }
        }
    }

    fn update_bus_vals(&self, sumid: u64, val: Vec<HintFieldOutput<F>>, row: usize, proves: bool, times: F) {
        let debug_data = self.debug_data.as_ref().expect("Debug data missing");
        let mut bus_values = debug_data.bus_values.lock().expect("Bus values missing");

        let bus_sumid = bus_values.entry(sumid).or_default();

        let bus_val = bus_sumid.entry(val.clone()).or_insert_with(|| BusValue {
            num_proves: F::zero(),
            num_assumes: F::zero(),
            row_proves: Vec::new(),
            row_assumes: Vec::new(),
        });

        if proves {
            bus_val.num_proves += times;
            bus_val.row_proves.push(row);
        } else {
            assert!(times.is_one());
            bus_val.num_assumes += times;
            bus_val.row_assumes.push(row);
        }
    }
}

impl<F: PrimeField> WitnessComponent<F> for StdSum<F> {
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
            let sum_airs = self.sum_airs.lock().unwrap();

            for (airgroup_id, air_id, im_hints, gsum_hints, debug_hints_data) in sum_airs.iter() {
                let air_instance_ids = pctx.air_instance_repo.find_air_instances(*airgroup_id, *air_id);

                for air_instance_id in air_instance_ids {
                    let air_instaces_vec = &mut pctx.air_instance_repo.air_instances.write().unwrap();

                    let air_instance = &mut air_instaces_vec[air_instance_id];

                    // Get the air associated with the air_instance
                    let airgroup_id = air_instance.airgroup_id;
                    let air_id = air_instance.air_id;
                    let air = pctx.pilout.get_air(airgroup_id, air_id);
                    let air_name = air.name().unwrap_or("unknown");

                    debug!("{}: ··· Computing witness for AIR '{}' at stage {}", Self::MY_NAME, air_name, stage);

                    let num_rows = air.num_rows();

                    if self.mode.name == ModeName::Debug {
                        self.debug(&pctx, &sctx, air_instance, num_rows, debug_hints_data.clone());
                    }

                    // Populate the im columns
                    for hint in im_hints {
                        let mut im = get_hint_field::<F>(
                            &sctx,
                            &pctx.public_inputs,
                            &pctx.challenges,
                            air_instance,
                            *hint as usize,
                            "reference",
                            HintFieldOptions::dest(),
                        );
                        let num = get_hint_field::<F>(
                            &sctx,
                            &pctx.public_inputs,
                            &pctx.challenges,
                            air_instance,
                            *hint as usize,
                            "numerator",
                            HintFieldOptions::default(),
                        );
                        let den = get_hint_field::<F>(
                            &sctx,
                            &pctx.public_inputs,
                            &pctx.challenges,
                            air_instance,
                            *hint as usize,
                            "denominator",
                            HintFieldOptions::inverse(),
                        );

                        // Apply a map&reduce strategy to compute the division
                        // TODO! Explore how to do it in only one step
                        // Step 1: Compute the division in parallel
                        let results: Vec<HintFieldOutput<F>> =
                            (0..num_rows).into_par_iter().map(|i| num.get(i) * den.get(i)).collect(); // Collect results into a vector
                                                                                                      // Step 2: Store the results in 'im'
                        for (i, &value) in results.iter().enumerate() {
                            im.set(i, value);
                        }
                        set_hint_field(&sctx, air_instance, *hint, "reference", &im);
                    }

                    // We know that at most one product hint exists
                    let gsum_hint = if gsum_hints.len() > 1 {
                        panic!("Multiple product hints found for AIR '{}'", air.name().unwrap_or("unknown"));
                    } else {
                        gsum_hints[0] as usize
                    };

                    // Use the hint to populate the gsum column
                    let mut gsum = get_hint_field::<F>(
                        &sctx,
                        &pctx.public_inputs,
                        &pctx.challenges,
                        air_instance,
                        gsum_hint,
                        "reference",
                        HintFieldOptions::dest(),
                    );
                    let expr = get_hint_field::<F>(
                        &sctx,
                        &pctx.public_inputs,
                        &pctx.challenges,
                        air_instance,
                        gsum_hint,
                        "expression",
                        HintFieldOptions::default(),
                    );

                    gsum.set(0, expr.get(0));
                    for i in 1..num_rows {
                        // TODO: We should perform the following division in batch using div_lib
                        gsum.set(i, gsum.get(i - 1) + expr.get(i));
                    }

                    // set the computed gsum column and its associated airgroup_val
                    set_hint_field(&sctx, air_instance, gsum_hint as u64, "reference", &gsum);
                    set_hint_field_val(&sctx, air_instance, gsum_hint as u64, "result", gsum.get(num_rows - 1));
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
