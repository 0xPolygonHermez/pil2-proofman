use proofman_common::ProofmanResult;
use proofman_witness::{witness_library, WitnessLibrary, WitnessManager};
use pil_std_lib::Std;
use proofman_fields::PrimeField64;
use proofman_fields::Goldilocks;
use proofman::register_std;

use crate::Compressor;

witness_library!(WitnessLib, Goldilocks);

impl<F: PrimeField64> WitnessLibrary<F> for WitnessLib {
    fn register_witness(&mut self, wcm: &WitnessManager<F>) -> ProofmanResult<()> {
        let std_lib = Std::new(wcm.get_pctx(), wcm.get_sctx(), false)?;
        let compressor = Compressor::new();

        register_std(wcm, &std_lib);

        wcm.register_component(compressor.clone());
        Ok(())
    }
}
