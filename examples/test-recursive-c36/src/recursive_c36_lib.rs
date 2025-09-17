use witness::{witness_library, WitnessLibrary, WitnessManager};
use pil_std_lib::Std;
use fields::PrimeField64;
use fields::Goldilocks;
use proofman::register_std;

use crate::RecursiveC36;

witness_library!(WitnessLib, Goldilocks);

impl<F: PrimeField64> WitnessLibrary<F> for WitnessLib {
    fn register_witness(&mut self, wcm: &WitnessManager<F>) {
        let std_lib = Std::new(wcm.get_pctx(), wcm.get_sctx(), false, vec![]);
        let recursive_c36 = RecursiveC36::new();

        register_std(wcm, &std_lib);

        wcm.register_component(recursive_c36.clone());
    }
}
