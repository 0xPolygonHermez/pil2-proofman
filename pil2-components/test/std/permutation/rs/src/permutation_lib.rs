use pil_std_lib::Std;
use witness::{witness_library, WitnessLibrary, WitnessManager};

use fields::PrimeField64;
use fields::Goldilocks;
use rand::distr::{StandardUniform, Distribution};

use crate::{Permutation1_6, Permutation1_7, Permutation1_8, Permutation2};
use proofman::register_std;

witness_library!(WitnessLib, Goldilocks);

impl<F: PrimeField64> WitnessLibrary<F> for WitnessLib
where
    StandardUniform: Distribution<F>,
{
    fn register_witness(&mut self, wcm: &WitnessManager<F>) {
        let std = Std::new(wcm.get_pctx(), wcm.get_sctx(), false);
        register_std(wcm, &std);
        wcm.register_component(Permutation1_6::new());
        wcm.register_component(Permutation1_7::new());
        wcm.register_component(Permutation1_8::new());
        wcm.register_component(Permutation2::new());
    }
}
