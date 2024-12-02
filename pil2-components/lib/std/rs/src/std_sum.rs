use std::{
    collections::HashMap,
    sync::{Arc, Mutex},
};

use num_traits::ToPrimitive;
use p3_field::PrimeField;

use proofman::{WitnessComponent, WitnessManager};
use proofman_common::{AirInstance, ExecutionCtx, ProofCtx, SetupCtx, StdMode, ModeName};
use proofman_hints::{
    acc_mul_hint_fields, get_hint_field, get_hint_field_a, get_hint_ids_by_name, mul_hint_fields, HintFieldOptions,
    HintFieldOutput,
};

use crate::{print_debug_info, update_debug_data, DebugData, Decider};

type SumAirsItem = (usize, usize, Vec<u64>, Vec<u64>, Vec<u64>); // (airgroup_id, air_id, gsum_hints, im_hints, debug_hints_data, debug_hints)

pub struct StdSum<F: PrimeField> {
    mode: StdMode,
    sum_airs: Mutex<Vec<SumAirsItem>>,
    debug_data: Option<DebugData<F>>,
}

impl<F: PrimeField> Decider<F> for StdSum<F> {
    fn decide(&self, sctx: Arc<SetupCtx<F>>, pctx: Arc<ProofCtx<F>>) {
        // Scan the pilout for airs that have sum-related hints
        let air_groups = pctx.pilout.air_groups();
        let mut sum_airs_guard = self.sum_airs.lock().unwrap();
        air_groups.iter().for_each(|air_group| {
            let airs = air_group.airs();
            airs.iter().for_each(|air| {
                let airgroup_id = air.airgroup_id;
                let air_id = air.air_id;

                let setup = sctx.get_setup(airgroup_id, air_id);
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

impl<F: PrimeField> StdSum<F> {
    const MY_NAME: &'static str = "STD Sum ";

    pub fn new(mode: StdMode, wcm: Arc<WitnessManager<F>>) -> Arc<Self> {
        let std_sum = Arc::new(Self {
            mode: mode.clone(),
            sum_airs: Mutex::new(Vec::new()),
            debug_data: if mode.name == ModeName::Debug { Some(Mutex::new(HashMap::new())) } else { None },
        });

        wcm.register_component(std_sum.clone(), None, None);

        std_sum
    }

    fn debug(
        &self,
        pctx: &ProofCtx<F>,
        sctx: &SetupCtx<F>,
        air_instance: &mut AirInstance<F>,
        num_rows: usize,
        debug_hints_data: Vec<u64>,
    ) {
        for hint in debug_hints_data.iter() {
            // let _name =
            //     get_hint_field::<F>(sctx, pctx, air_instance, *hint as usize, "name_piop", HintFieldOptions::default());

            let sumid =
                get_hint_field::<F>(sctx, pctx, air_instance, *hint as usize, "sumid", HintFieldOptions::default());
            if let HintFieldOutput::Field(sumid) = sumid.get(0) {
                if let Some(opids) = &self.mode.opids {
                    if !opids.contains(&sumid.as_canonical_biguint().to_u64().expect("Cannot convert to u64")) {
                        continue;
                    }
                }
            } else {
                panic!("sumid must be a field element");
            };

            let proves =
                get_hint_field::<F>(sctx, pctx, air_instance, *hint as usize, "proves", HintFieldOptions::default());

            let mul =
                get_hint_field::<F>(sctx, pctx, air_instance, *hint as usize, "selector", HintFieldOptions::default());

            let expressions = get_hint_field_a::<F>(
                sctx,
                pctx,
                air_instance,
                *hint as usize,
                "references",
                HintFieldOptions::default(),
            );

            // let _names = get_hint_field_a::<F>(
            //     sctx,
            //     &pctx.public_inputs,
            //     &pctx.challenges,
            //     air_instance,
            //     *hint as usize,
            //     "names",
            //     HintFieldOptions::default(),
            // );

            (0..num_rows).for_each(|j| {
                let mut mul = match mul.get(j) {
                    HintFieldOutput::Field(mul) => mul,
                    _ => panic!("mul must be a field element"),
                };

                if !mul.is_zero() {
                    let sumid = match sumid.get(j) {
                        HintFieldOutput::Field(sumid) => sumid,
                        _ => panic!("sumid must be a field element"),
                    };

                    let proves = match proves.get(j) {
                        HintFieldOutput::Field(proves) => match proves {
                            p if p.is_zero() || p == -F::one() => {
                                // If it's an assume, then negate its value
                                if p == -F::one() {
                                    mul = -mul;
                                }
                                false
                            }
                            p if p.is_one() => true,
                            _ => panic!("Proves hint must be either 0, 1, or -1"),
                        },
                        _ => panic!("Proves hint must be a field element"),
                    };

                    let debug_data = self.debug_data.as_ref().expect("Debug data missing");
                    let airgroup_id = air_instance.airgroup_id;
                    let air_id = air_instance.air_id;
                    let instance_id = air_instance.air_instance_id.unwrap_or_default();
                    update_debug_data(
                        debug_data,
                        sumid,
                        expressions.get(j),
                        airgroup_id,
                        air_id,
                        instance_id,
                        j,
                        proves,
                        mul,
                    );
                }
            });
        }
    }
}

impl<F: PrimeField> WitnessComponent<F> for StdSum<F> {
    fn start_proof(&self, pctx: Arc<ProofCtx<F>>, _ectx: Arc<ExecutionCtx<F>>, sctx: Arc<SetupCtx<F>>) {
        self.decide(sctx, pctx);
    }

    fn calculate_witness(
        &self,
        stage: u32,
        _air_instance: Option<usize>,
        pctx: Arc<ProofCtx<F>>,
        _ectx: Arc<ExecutionCtx<F>>,
        sctx: Arc<SetupCtx<F>>,
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

                    log::debug!("{}: ··· Computing witness for AIR '{}' at stage {}", Self::MY_NAME, air_name, stage);

                    let num_rows = air.num_rows();

                    if self.mode.name == ModeName::Debug {
                        self.debug(&pctx, &sctx, air_instance, num_rows, debug_hints_data.clone());
                    }

                    // Populate the im columns
                    for hint in im_hints {
                        let id = mul_hint_fields::<F>(
                            &sctx,
                            &pctx,
                            air_instance,
                            *hint as usize,
                            "reference",
                            "numerator",
                            HintFieldOptions::default(),
                            "denominator",
                            HintFieldOptions::inverse(),
                        );

                        air_instance.set_commit_calculated(id as usize);
                    }

                    // We know that at most one product hint exists
                    let gsum_hint = if gsum_hints.len() > 1 {
                        panic!("Multiple product hints found for AIR '{}'", air.name().unwrap_or("unknown"));
                    } else {
                        gsum_hints[0] as usize
                    };

                    // This call accumulates "expression" into "reference" expression and stores its last value to "result"
                    // Alternatively, this could be done using get_hint_field and set_hint_field methods and doing the accumulation in Rust,
                    // TODO: GENERALIZE CALLS
                    let (pol_id, airgroupvalue_id) = acc_mul_hint_fields::<F>(
                        &sctx,
                        &pctx,
                        air_instance,
                        gsum_hint,
                        "reference",
                        "result",
                        "numerator",
                        "denominator",
                        HintFieldOptions::default(),
                        HintFieldOptions::inverse(),
                        true,
                    );

                    air_instance.set_commit_calculated(pol_id as usize);
                    air_instance.set_airgroupvalue_calculated(airgroupvalue_id as usize);
                }
            }
        }
    }

    fn end_proof(&self) {
        if self.mode.name == ModeName::Debug {
            let name = Self::MY_NAME;
            let max_values_to_print = self.mode.n_vals;
            let debug_data = self.debug_data.as_ref().expect("Debug data missing");
            print_debug_info(name, max_values_to_print, debug_data);
        }
    }
}
