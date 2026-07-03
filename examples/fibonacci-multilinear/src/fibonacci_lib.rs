use fields::{Goldilocks, PrimeField64};
use proofman_common::{load_from_json, ProofmanResult};
use witness::{witness_library, WitnessLibrary, WitnessManager};

use crate::{FibonacciML, FibonacciMLTrace, FibonacciPublicValues, FibonacciPublics};

witness_library!(WitnessLib, Goldilocks);

impl<F: PrimeField64> WitnessLibrary<F> for WitnessLib {
    fn register_witness(&mut self, wcm: &WitnessManager<F>) -> ProofmanResult<()> {
        let fibonacci = FibonacciML::new();
        wcm.register_component(fibonacci.clone());

        let public_inputs: FibonacciPublics = load_from_json(&wcm.get_public_inputs_path());

        let mut publics = FibonacciPublicValues::from_vec_guard(wcm.get_pctx().get_publics());
        publics.in1 = F::from_u64(public_inputs.in1);
        publics.in2 = F::from_u64(public_inputs.in2);

        // out = b after N-1 steps of (a, b) -> (b, a + b), reduced mod p.
        let p = Goldilocks::ORDER_U64 as u128;
        let mut a = public_inputs.in1 as u128;
        let mut b = public_inputs.in2 as u128;
        for _ in 1..FibonacciMLTrace::<F>::NUM_ROWS {
            (a, b) = (b, (a + b) % p);
        }
        publics.out = F::from_u64(b as u64);

        Ok(())
    }
}
