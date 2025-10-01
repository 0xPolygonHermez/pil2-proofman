use std::sync::{Arc, RwLock};

use proofman_common::{BufferPool, AirInstance, FromTrace, ProofCtx, SetupCtx};
use witness::{WitnessComponent, execute};
use pil_std_lib::Std;
use fields::PrimeField64;
use rayon::prelude::*;
use crate::{BuildPublicValues, FibonacciSquareTrace, ModuleAirValues, ModuleTrace};

pub struct Module<F: PrimeField64> {
    fibonacci_rows: u64,
    instance_ids: RwLock<Vec<usize>>,
    std_lib: Arc<Std<F>>,
}

impl<F: PrimeField64> Module<F> {
    pub fn new(fibonacci_rows: u64, std_lib: Arc<Std<F>>) -> Arc<Self> {
        Arc::new(Module { fibonacci_rows, std_lib, instance_ids: RwLock::new(Vec::new()) })
    }
}

impl<F: PrimeField64> WitnessComponent<F> for Module<F> {
    execute!(ModuleTrace, FibonacciSquareTrace::<F>::NUM_ROWS / ModuleTrace::<F>::NUM_ROWS);

    fn calculate_witness(
        &self,
        stage: u32,
        pctx: Arc<ProofCtx<F>>,
        _sctx: Arc<SetupCtx<F>>,
        instance_ids: &[usize],
        _n_cores: usize,
        buffer_pool: &dyn BufferPool<F>,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        if stage == 1 {
            tracing::debug!("··· Starting witness computation stage 1");
            let publics = BuildPublicValues::from_vec_guard(pctx.get_publics());
            let module = F::as_canonical_u64(&publics.module);
            let mut a = F::as_canonical_u64(&publics.in1);
            let mut b = F::as_canonical_u64(&publics.in2);

            //range_check(colu: mod - x_mod, min: 1, max: 2**8-1);
            let range = self.std_lib.get_range_id(1, (1 << 8) - 1, None)?;

            let mut modules = Vec::new();
            for _ in 1..self.fibonacci_rows {
                let tmp = b;
                let result = (a.pow(2) + b.pow(2)) % module;
                modules.push(a.pow(2) + b.pow(2));
                (a, b) = (tmp, result);
            }

            let num_rows = ModuleTrace::<F>::NUM_ROWS;

            let num_instances = self.instance_ids.read().unwrap().len();
            for j in 0..num_instances {
                let instance_id = self.instance_ids.read().unwrap()[j];
                if !instance_ids.contains(&instance_id) {
                    continue;
                }
                let mut x_mods = Vec::new();

                let mut trace = ModuleTrace::new_from_vec(buffer_pool.take_buffer());

                let start = j * num_rows;
                let end = ((j + 1) * num_rows).min(modules.len());

                let modules_slice = modules[start..end].to_vec();

                for (i, input) in modules_slice.iter().enumerate() {
                    let x = *input;
                    let q = x / module;
                    let x_mod = x % module;

                    trace[i].x = F::from_u64(x);
                    trace[i].q = F::from_u64(q);
                    trace[i].x_mod = F::from_u64(x_mod);
                    x_mods.push(x_mod);
                }

                for i in modules_slice.len()..num_rows {
                    trace[i].x = F::ZERO;
                    trace[i].q = F::ZERO;
                    trace[i].x_mod = F::ZERO;
                }

                let mut air_values = ModuleAirValues::<F>::new();
                air_values.last_segment = F::from_bool(j == num_instances - 1);

                x_mods.par_iter().for_each(|x_mod| {
                    self.std_lib.range_check(range, (module - x_mod) as i64, 1);
                });

                // Trivial range check for the remaining rows
                for _ in modules_slice.len()..trace.num_rows() {
                    self.std_lib.range_check(range, module as i64, 1);
                }

                let air_instance =
                    AirInstance::new_from_trace(FromTrace::new(&mut trace).with_air_values(&mut air_values));

                pctx.add_air_instance(air_instance, instance_id);
            }
        }
        Ok(())
    }
}
