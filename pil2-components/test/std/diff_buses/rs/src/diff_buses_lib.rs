use std::sync::Arc;

use pil_std_lib::Std;
use witness::{witness_library, WitnessLibrary, WitnessManager};

use fields::PrimeField64;
use fields::Goldilocks;
use rand::distr::{StandardUniform, Distribution};

use crate::{ProdBus, BothBuses, SumBus};

witness_library!(WitnessLib, Goldilocks);

impl<F: PrimeField64> WitnessLibrary<F> for WitnessLib
where
    StandardUniform: Distribution<F>,
{
    fn register_witness(&mut self, wcm: Arc<WitnessManager<F>>) {
        Std::new(wcm.get_pctx(), wcm.get_sctx());
        let prod_bus = ProdBus::new();
        let sum_bus = SumBus::new();
        let both_buses = BothBuses::new();

        wcm.register_component(prod_bus.clone());
        wcm.register_component(sum_bus.clone());
        wcm.register_component(both_buses.clone());
    }
}
