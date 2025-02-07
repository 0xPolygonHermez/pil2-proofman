use std::sync::{Arc, Mutex, RwLock};

use proofman_common::{AirInstance, FromTrace, ProofCtx, SetupCtx};
use witness::WitnessComponent;
use pil_std_lib::Std;
use p3_field::{AbstractField, PrimeField64};
use num_bigint::BigInt;
use rayon::prelude::*;
use crate::{BuildPublicValues, FibonacciSquareTrace, ModuleAirValues, ModuleTrace};

pub struct Module<F: PrimeField64> {
    inputs: Mutex<Vec<(u64, u64)>>,
    instance_ids: RwLock<Vec<usize>>,
    std_lib: Arc<Std<F>>,
}

impl<F: PrimeField64 + AbstractField + Clone + Copy + Default + 'static> Module<F> {
    const MY_NAME: &'static str = "ModuleSM";

    pub fn new(std_lib: Arc<Std<F>>) -> Arc<Self> {
        Arc::new(Module { inputs: Mutex::new(Vec::new()), std_lib, instance_ids: RwLock::new(Vec::new()) })
    }

    pub fn set_inputs(&self, inputs: Vec<(u64, u64)>) {
        *self.inputs.lock().unwrap() = inputs;
    }
}

impl<F: PrimeField64 + AbstractField + Copy> WitnessComponent<F> for Module<F> {
    fn execute(&self, pctx: Arc<ProofCtx<F>>) -> Vec<usize> {
        let mut instance_ids = Vec::new();
        let num_instances = FibonacciSquareTrace::<usize>::NUM_ROWS / ModuleTrace::<usize>::NUM_ROWS;
        for _ in 0..num_instances {
            instance_ids.push(pctx.add_instance(ModuleTrace::<usize>::AIRGROUP_ID, ModuleTrace::<usize>::AIR_ID));
        }
        *self.instance_ids.write().unwrap() = instance_ids.clone();
        instance_ids
    }

    fn calculate_witness(&self, stage: u32, pctx: Arc<ProofCtx<F>>, _sctx: Arc<SetupCtx<F>>, instance_ids: &[usize]) {
        if stage == 1 {
            log::debug!("{} ··· Starting witness computation stage {}", Self::MY_NAME, 1);
            let publics = BuildPublicValues::from_vec_guard(pctx.get_publics());
            let module = F::as_canonical_u64(&publics.module);

            //range_check(colu: mod - x_mod, min: 1, max: 2**8-1);
            let range = self.std_lib.get_range(BigInt::from(1), BigInt::from((1 << 8) - 1), None);

            let inputs = self.inputs.lock().unwrap();

            let num_rows = ModuleTrace::<F>::NUM_ROWS;

            let num_instances = self.instance_ids.read().unwrap().len();
            for j in 0..num_instances {
                let instance_id = self.instance_ids.read().unwrap()[j];
                if !instance_ids.contains(&instance_id) {
                    continue;
                }
                let mut x_mods = Vec::new();

                let mut trace = ModuleTrace::new();

                let start = j * num_rows;
                let end = ((j + 1) * num_rows).min(inputs.len());

                let inputs_slice = inputs[start..end].to_vec();

                for (i, input) in inputs_slice.iter().enumerate() {
                    let x = input.0;
                    let q = x / module;
                    let x_mod = input.1;

                    trace[i].x = F::from_canonical_u64(x);
                    trace[i].q = F::from_canonical_u64(q);
                    trace[i].x_mod = F::from_canonical_u64(x_mod);
                    x_mods.push(x_mod);
                }

                for i in inputs_slice.len()..num_rows {
                    trace[i].x = F::zero();
                    trace[i].q = F::zero();
                    trace[i].x_mod = F::zero();
                }

                let mut air_values = ModuleAirValues::<F>::new();
                air_values.last_segment = F::from_bool(j == num_instances - 1);

                x_mods.par_iter().for_each(|x_mod| {
                    self.std_lib.range_check(F::from_canonical_u64(module - x_mod), F::one(), range);
                });

                // Trivial range check for the remaining rows
                for _ in inputs_slice.len()..trace.num_rows() {
                    self.std_lib.range_check(F::from_canonical_u64(module), F::one(), range);
                }

                let air_instance =
                    AirInstance::new_from_trace(FromTrace::new(&mut trace).with_air_values(&mut air_values));

                pctx.add_air_instance(air_instance, instance_id);
            }
        }
    }
}
