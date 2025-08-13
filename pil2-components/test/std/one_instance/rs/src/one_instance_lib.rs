use pil_std_lib::Std;
use witness::{witness_library, WitnessLibrary, WitnessManager};

use fields::PrimeField64;
use fields::Goldilocks;
use rand::distr::{StandardUniform, Distribution};

use crate::{AirProd, AirSum};
use proofman::register_std;

witness_library!(WitnessLib, Goldilocks);

impl<F: PrimeField64> WitnessLibrary<F> for WitnessLib
where
    StandardUniform: Distribution<F>,
{
    fn register_witness(&mut self, wcm: &WitnessManager<F>) {
        let std = Std::new(wcm.get_pctx(), wcm.get_sctx());
        register_std(wcm, &std);
        wcm.register_component(AirProd::new());
        wcm.register_component(AirSum::new());
    }
}
