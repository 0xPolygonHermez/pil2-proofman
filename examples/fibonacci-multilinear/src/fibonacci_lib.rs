use fields::{Goldilocks, PrimeField64};
use proofman_common::{load_from_json, ProofmanResult};
use proofman::register_std;
use witness::{witness_library, WitnessLibrary, WitnessManager};
use pil_std_lib::Std;

use crate::{Fibonacci, FibonacciTrace, BuildPublicValues, BuildPublics, Module};

witness_library!(WitnessLib, Goldilocks);

impl<F: PrimeField64> WitnessLibrary<F> for WitnessLib {
    fn register_witness(&mut self, wcm: &WitnessManager<F>) -> ProofmanResult<()> {
        let std_lib = Std::new(wcm.get_pctx(), wcm.get_sctx(), true)?;
        let module = Module::new(FibonacciTrace::<F>::NUM_ROWS as u64, std_lib.clone());
        let fibonacci = Fibonacci::new();

        register_std(wcm, &std_lib);

        wcm.register_component(fibonacci.clone());
        wcm.register_component(module.clone());

        let public_inputs: BuildPublics = load_from_json(&wcm.get_public_inputs_path());

        let mut publics = BuildPublicValues::from_vec_guard(wcm.get_pctx().get_publics());
        publics.module = F::from_u64(public_inputs.module);
        publics.in1 = F::from_u64(public_inputs.in1);
        publics.in2 = F::from_u64(public_inputs.in2);

        // out = b after N-1 steps of (a, b) -> (b, a + b), reduced mod m
        let m = public_inputs.module;
        let mut a = public_inputs.in1;
        let mut b = public_inputs.in2;
        for _ in 1..FibonacciTrace::<F>::NUM_ROWS {
            let tmp = b;
            let result = if m == 0 { 0 } else { (a + b) % m };
            (a, b) = (tmp, result);
        }
        publics.out = F::from_u64(b as u64);

        Ok(())
    }
}
