use std::sync::Arc;

use pil_std_lib::Std;
use witness::{witness_library, WitnessLibrary, WitnessManager};

use p3_field::PrimeField64;
use p3_goldilocks::Goldilocks;
use rand::distr::{StandardUniform, Distribution};

use crate::{ProdBus, BothBuses, SumBus};

witness_library!(WitnessLib, Goldilocks);

impl<F: PrimeField64> WitnessLibrary<F> for WitnessLib
where
    StandardUniform: Distribution<F>,
{
    fn register_witness(&mut self, wcm: Arc<WitnessManager<F>>) {
        Std::new(wcm.clone());
        let prod_bus = ProdBus::new();
        let sum_bus = SumBus::new();
        let both_buses = BothBuses::new();

        wcm.register_component(prod_bus.clone());
        wcm.register_component(sum_bus.clone());
        wcm.register_component(both_buses.clone());
    }
}
