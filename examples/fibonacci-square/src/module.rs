use std::sync::{Arc, Mutex};

use proofman_common::{add_air_instance, FromTrace, AirInstance, ProofCtx};
use witness::WitnessComponent;
use pil_std_lib::Std;
use p3_field::{AbstractField, PrimeField64};
use num_bigint::BigInt;

use crate::{BuildPublicValues, ModuleTrace};

pub struct Module<F: PrimeField64> {
    inputs: Mutex<Vec<(u64, u64)>>,
    std_lib: Arc<Std<F>>,
}

impl<F: PrimeField64 + AbstractField + Clone + Copy + Default + 'static> Module<F> {
    const MY_NAME: &'static str = "ModuleSM";

    pub fn new(std_lib: Arc<Std<F>>) -> Arc<Self> {
        Arc::new(Module { inputs: Mutex::new(Vec::new()), std_lib })
    }

    pub fn calculate_module(&self, x: u64, module: u64) -> u64 {
        let x_mod = x % module;

        let mut inputs = self.inputs.lock().unwrap();

        inputs.push((x, x_mod));

        x_mod
    }
}

impl<F: PrimeField64 + AbstractField + Copy> WitnessComponent<F> for Module<F> {
    fn execute(&self, pctx: Arc<ProofCtx<F>>) {
        log::debug!("{} ··· Starting witness computation stage {}", Self::MY_NAME, 1);

        let publics = BuildPublicValues::from_vec_guard(pctx.get_publics());
        let module = F::as_canonical_u64(&publics.module);

        let mut trace = ModuleTrace::new_zeroes();

        //range_check(colu: mod - x_mod, min: 1, max: 2**8-1);
        let range = self.std_lib.get_range(BigInt::from(1), BigInt::from((1 << 8) - 1), None);

        let inputs = self.inputs.lock().unwrap();

        let inputs_len = inputs.len();

        let mut x_mods = Vec::new();

        for (i, input) in inputs.iter().enumerate() {
            let x = input.0;
            let q = x / module;
            let x_mod = input.1;

            trace[i].x = F::from_canonical_u64(x);
            trace[i].q = F::from_canonical_u64(q);
            trace[i].x_mod = F::from_canonical_u64(x_mod);
            x_mods.push(x_mod);
        }

        let air_instance = AirInstance::new_from_trace(FromTrace::new(&mut trace));
        let is_mine = add_air_instance::<F>(air_instance, pctx.clone());
        if is_mine.is_some() {
            for x_mod in x_mods.iter() {
                self.std_lib.range_check(F::from_canonical_u64(module - x_mod), F::one(), range);
            }

            // Trivial range check for the remaining rows
            for _ in inputs_len..trace.num_rows() {
                self.std_lib.range_check(F::from_canonical_u64(module), F::one(), range);
            }
        }
    }
}
