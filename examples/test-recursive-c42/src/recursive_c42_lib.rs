use proofman_common::ProofmanResult;
use witness::{witness_library, WitnessLibrary, WitnessManager};
use pil_std_lib::Std;
use fields::PrimeField64;
use fields::Goldilocks;
use proofman::register_std;

use crate::RecursiveC42;

witness_library!(WitnessLib, Goldilocks);

impl<F: PrimeField64> WitnessLibrary<F> for WitnessLib {
    fn register_witness(&mut self, wcm: &WitnessManager<F>) -> ProofmanResult<()> {
        let std_lib = Std::new(wcm.get_pctx(), wcm.get_sctx(), false)?;
        let recursive_c42 = RecursiveC42::new();

        register_std(wcm, &std_lib);

        wcm.register_component(recursive_c42.clone());
        Ok(())
    }
}
