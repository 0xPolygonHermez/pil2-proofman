use std::sync::{Arc, Mutex, RwLock};

use proofman_common::{AirInstance, FromTrace, ProofCtx, SetupCtx};
use witness::{WitnessComponent, execute};
use pil_std_lib::Std;
use fields::PrimeField64;
use rayon::prelude::*;
use crate::{BuildPublicValues, FibonacciSquareTrace, ModuleAirValues, ModuleTrace};

pub struct Module<F: PrimeField64> {
    inputs: Mutex<Vec<u64>>,
    instance_ids: RwLock<Vec<usize>>,
    std_lib: Arc<Std<F>>,
}

impl<F: PrimeField64> Module<F> {
    pub fn new(std_lib: Arc<Std<F>>) -> Arc<Self> {
        Arc::new(Module { inputs: Mutex::new(Vec::new()), std_lib, instance_ids: RwLock::new(Vec::new()) })
    }

    pub fn set_inputs(&self, inputs: Vec<u64>) {
        *self.inputs.lock().unwrap() = inputs;
    }
}

impl<F: PrimeField64> WitnessComponent<F> for Module<F> {
    execute!(ModuleTrace, FibonacciSquareTrace::<usize>::NUM_ROWS / ModuleTrace::<usize>::NUM_ROWS);

    fn calculate_witness(&self, stage: u32, pctx: Arc<ProofCtx<F>>, _sctx: Arc<SetupCtx<F>>, instance_ids: &[usize]) {
        if stage == 1 {
            tracing::debug!("··· Starting witness computation stage 1");
            let publics = BuildPublicValues::from_vec_guard(pctx.get_publics());
            let module = F::as_canonical_u64(&publics.module);

            //range_check(colu: mod - x_mod, min: 1, max: 2**8-1);
            let range = self.std_lib.get_range(1, (1 << 8) - 1, None);

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
                    let x = *input;
                    let q = x / module;
                    let x_mod = x % module;

                    trace[i].x = F::from_u64(x);
                    trace[i].q = F::from_u64(q);
                    trace[i].x_mod = F::from_u64(x_mod);
                    x_mods.push(x_mod);
                }

                for i in inputs_slice.len()..num_rows {
                    trace[i].x = F::ZERO;
                    trace[i].q = F::ZERO;
                    trace[i].x_mod = F::ZERO;
                }

                let mut air_values = ModuleAirValues::<F>::new();
                air_values.last_segment = F::from_bool(j == num_instances - 1);

                x_mods.par_iter().for_each(|x_mod| {
                    self.std_lib.range_check((module - x_mod) as i64, 1, range);
                });

                // Trivial range check for the remaining rows
                for _ in inputs_slice.len()..trace.num_rows() {
                    self.std_lib.range_check(module as i64, 1, range);
                }

                let air_instance =
                    AirInstance::new_from_trace(FromTrace::new(&mut trace).with_air_values(&mut air_values));

                pctx.add_air_instance(air_instance, instance_id);
            }
        }
    }
}
