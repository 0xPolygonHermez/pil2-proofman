use std::sync::Arc;
use proofman_common::load_from_json;
use witness::{witness_library, WitnessLibrary, WitnessManager};
use pil_std_lib::Std;
use p3_field::PrimeField64;
use p3_goldilocks::Goldilocks;

use crate::{BuildPublics, BuildPublicValues, FibonacciSquare, Module, FibonacciSquareTrace};

witness_library!(WitnessLib, Goldilocks);

impl<F: PrimeField64> WitnessLibrary<F> for WitnessLib {
    fn register_witness(&mut self, wcm: Arc<WitnessManager<F>>) {
        let std_lib = Std::new(wcm.clone());
        let module = Module::new(FibonacciSquareTrace::<usize>::NUM_ROWS as u64, std_lib.clone());
        let fibonacci = FibonacciSquare::new();

        wcm.register_component(fibonacci.clone());
        wcm.register_component(module.clone());

        let public_inputs: BuildPublics = load_from_json(&wcm.get_public_inputs_path());

        let mut publics = BuildPublicValues::from_vec_guard(wcm.get_pctx().get_publics());

        publics.module = F::from_u64(public_inputs.module);
        publics.in1 = F::from_u64(public_inputs.in1);
        publics.in2 = F::from_u64(public_inputs.in2);
    }
}
