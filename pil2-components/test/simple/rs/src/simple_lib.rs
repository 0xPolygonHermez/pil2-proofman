use pil_std_lib::Std;
use proofman_common::ProofmanResult;
use witness::{witness_library, WitnessLibrary, WitnessManager};

use fields::PrimeField64;
use fields::Goldilocks;
use rand::{rng, Rng};

use crate::{SimpleLeft, SimpleRight};
use proofman::register_std;

witness_library!(WitnessLib, Goldilocks);

impl<F: PrimeField64> WitnessLibrary<F> for WitnessLib {
    fn register_witness(&mut self, wcm: &WitnessManager<F>) -> ProofmanResult<()> {
        let seed = if cfg!(feature = "debug") { 0 } else { rng().random::<u64>() };

        let std_lib = Std::new(wcm.get_pctx(), wcm.get_sctx(), false)?;
        let simple_left = SimpleLeft::new(std_lib.clone());
        let simple_right = SimpleRight::new();

        register_std(wcm, &std_lib);

        wcm.register_component(simple_left.clone());
        wcm.register_component(simple_right.clone());

        simple_left.set_seed(seed);
        Ok(())
    }
}
