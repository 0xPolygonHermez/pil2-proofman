use proofman_common::load_from_json;
use witness::{witness_library, WitnessLibrary, WitnessManager};
use pil_std_lib::Std;
use fields::PrimeField64;
use fields::Goldilocks;
use proofman::register_std;

use crate::{BuildPublics, BuildPublicValues, BuildProofValues, FibonacciSquare, Module, FibonacciSquareTrace};

witness_library!(WitnessLib, Goldilocks);

impl<F: PrimeField64> WitnessLibrary<F> for WitnessLib {
    fn register_witness(&mut self, wcm: &WitnessManager<F>) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let std_lib = Std::new(wcm.get_pctx(), wcm.get_sctx(), true)?;
        let module = Module::new(FibonacciSquareTrace::<F>::NUM_ROWS as u64, std_lib.clone());
        let fibonacci = FibonacciSquare::new();

        register_std(wcm, &std_lib);

        wcm.register_component(fibonacci.clone());
        wcm.register_component(module.clone());

        let public_inputs: BuildPublics = load_from_json(&wcm.get_public_inputs_path());

        let mut publics = BuildPublicValues::from_vec_guard(wcm.get_pctx().get_publics());

        publics.module = F::from_u64(public_inputs.module);
        publics.in1 = F::from_u64(public_inputs.in1);
        publics.in2 = F::from_u64(public_inputs.in2);

        let mut a = public_inputs.in1;
        let mut b = public_inputs.in2;
        for _ in 1..FibonacciSquareTrace::<F>::NUM_ROWS {
            let tmp = b;
            let result = if public_inputs.module == 0 { 0 } else { (a.pow(2) + b.pow(2)) % public_inputs.module };
            (a, b) = (tmp, result);
        }

        publics.out = F::from_u64(b);

        let mut proof_values = BuildProofValues::from_vec_guard(wcm.get_pctx().get_proof_values());
        proof_values.value1 = F::from_u64(5);
        proof_values.value2 = F::from_u64(125);
        Ok(())
    }
}
