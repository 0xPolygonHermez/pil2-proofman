use std::sync::Arc;
use pil_std_lib::Std;
use witness::{witness_library, WitnessLibrary, WitnessManager};

use p3_field::PrimeField64;
use p3_goldilocks::Goldilocks;
use rand::{
    distr::{StandardUniform, Distribution},
    Rng,
};

use crate::{SimpleLeft, SimpleRight};

witness_library!(WitnessLib, Goldilocks);

impl<F: PrimeField64> WitnessLibrary<F> for WitnessLib
where
    StandardUniform: Distribution<F>,
{
    fn register_witness(&mut self, wcm: Arc<WitnessManager<F>>) {
        let seed = if cfg!(feature = "debug") { 0 } else { rand::rng().random::<u64>() };

        let std_lib = Std::new(wcm.clone());
        let simple_left = SimpleLeft::new(std_lib.clone());
        let simple_right = SimpleRight::new();

        wcm.register_component(simple_left.clone());
        wcm.register_component(simple_right.clone());

        simple_left.set_seed(seed);
    }
}
