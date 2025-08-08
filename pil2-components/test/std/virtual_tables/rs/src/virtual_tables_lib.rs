use pil_std_lib::Std;
use witness::{witness_library, WitnessLibrary, WitnessManager};

use fields::PrimeField64;
use fields::Goldilocks;
use rand::{
    distr::{StandardUniform, Distribution},
    Rng,
};

use crate::{Component1, Component2, Component3, Component4, Component5, Component6 /*Component7, Table7*/, Component8};
use proofman::register_std;

witness_library!(WitnessLib, Goldilocks);

impl<F: PrimeField64> WitnessLibrary<F> for WitnessLib
where
    StandardUniform: Distribution<F>,
{
    fn register_witness(&mut self, wcm: &WitnessManager<F>) {
        let seed = if cfg!(feature = "debug") { 0 } else { rand::rng().random::<u64>() };

        let std = Std::new(wcm.get_pctx(), wcm.get_sctx());
        register_std(wcm, &std);

        let component1 = Component1::new(std.clone());
        let component2 = Component2::new(std.clone());
        let component3 = Component3::new(std.clone());
        let component4 = Component4::new(std.clone());
        let component5 = Component5::new(std.clone());
        let component6 = Component6::new(std.clone());
        // let table7 = Table7::new();
        // let component7 = Component7::new(table7.clone());
        let component8 = Component8::new(std.clone());
        component1.set_seed(seed);
        component2.set_seed(seed);
        component3.set_seed(seed);
        component4.set_seed(seed);
        component5.set_seed(seed);
        component6.set_seed(seed);
        // component7.set_seed(seed);
        component8.set_seed(seed);

        wcm.register_component(component1);
        wcm.register_component(component2);
        wcm.register_component(component3);
        wcm.register_component(component4);
        wcm.register_component(component5);
        wcm.register_component(component6);
        // wcm.register_component(table7);
        // wcm.register_component(component7);
        wcm.register_component(component8);
    }
}
