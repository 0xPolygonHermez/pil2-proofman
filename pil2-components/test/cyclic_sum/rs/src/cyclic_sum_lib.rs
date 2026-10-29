use pil2_std_lib::Std;
use proofman_common::ProofmanResult;
use proofman_witness::{witness_library, WitnessLibrary, WitnessManager};

use proofman_fields::PrimeField64;
use proofman_fields::Goldilocks;

use crate::{CyclicSumDirect, CyclicSumQ0};
use proofman::register_std;

witness_library!(WitnessLib, Goldilocks);

impl<F: PrimeField64> WitnessLibrary<F> for WitnessLib {
    fn register_witness(&mut self, wcm: &WitnessManager<F>) -> ProofmanResult<()> {
        let std = Std::new(wcm.get_pctx(), wcm.get_sctx(), false)?;
        register_std(wcm, &std);
        wcm.register_component(CyclicSumQ0::new());
        wcm.register_component(CyclicSumDirect::new());
        Ok(())
    }
}
