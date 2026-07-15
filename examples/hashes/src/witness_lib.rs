use proofman_common::ProofmanResult;
use proofman::register_std;
use witness::{witness_library, WitnessLibrary, WitnessManager};
use pil_std_lib::Std;
use fields::PrimeField64;
use fields::Goldilocks;

use crate::blake2b::Blake2bAir;
use crate::blake3::Blake3Air;
use crate::sha2::Sha2Air;

witness_library!(WitnessLib, Goldilocks);

impl<F: PrimeField64> WitnessLibrary<F> for WitnessLib {
    fn register_witness(&mut self, wcm: &WitnessManager<F>) -> ProofmanResult<()> {
        let std_lib = Std::new(wcm.get_pctx(), wcm.get_sctx(), true)?;
        register_std(wcm, &std_lib);

        let sha2_air = Sha2Air::new::<F>();
        wcm.register_component(sha2_air);

        let blake2b_air = Blake2bAir::new::<F>();
        wcm.register_component(blake2b_air);

        let blake3_air = Blake3Air::new::<F>();
        wcm.register_component(blake3_air);

        Ok(())
    }
}
