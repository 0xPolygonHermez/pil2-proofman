use std::sync::Arc;

use pil_std_lib::Std;
use witness::{witness_library, WitnessLibrary, WitnessManager};

use p3_field::PrimeField64;
use p3_goldilocks::Goldilocks;
use rand::{
    distr::{StandardUniform, Distribution},
    Rng
};

use crate::{
    RangeCheckMix, RangeCheckDynamic1, RangeCheckDynamic2, MultiRangeCheck1, MultiRangeCheck2, RangeCheck1,
    RangeCheck2, RangeCheck3, RangeCheck4,
};

witness_library!(WitnessLib, Goldilocks);

impl<F: PrimeField64> WitnessLibrary<F> for WitnessLib
where
    StandardUniform: Distribution<F>,
{
    fn register_witness(&mut self, wcm: Arc<WitnessManager<F>>) {
        let seed = if cfg!(feature = "debug") { 0 } else { rand::rng().random::<u64>() };

        let std_lib = Std::new(wcm.clone());
        let range_check1 = RangeCheck1::new(std_lib.clone());
        let range_check2 = RangeCheck2::new(std_lib.clone());
        let range_check3 = RangeCheck3::new(std_lib.clone());
        let range_check4 = RangeCheck4::new(std_lib.clone());
        let multi_range_check1 = MultiRangeCheck1::new(std_lib.clone());
        let multi_range_check2 = MultiRangeCheck2::new(std_lib.clone());
        let range_check_dynamic1 = RangeCheckDynamic1::new(std_lib.clone());
        let range_check_dynamic2 = RangeCheckDynamic2::new(std_lib.clone());
        let range_check_mix = RangeCheckMix::new(std_lib.clone());

        range_check1.set_seed(seed);
        range_check2.set_seed(seed);
        range_check3.set_seed(seed);
        range_check4.set_seed(seed);
        multi_range_check1.set_seed(seed);
        multi_range_check2.set_seed(seed);
        range_check_dynamic1.set_seed(seed);
        range_check_dynamic2.set_seed(seed);
        range_check_mix.set_seed(seed);

        wcm.register_component(range_check1.clone());
        wcm.register_component(range_check2.clone());
        wcm.register_component(range_check3.clone());
        wcm.register_component(range_check4.clone());
        wcm.register_component(multi_range_check1.clone());
        wcm.register_component(multi_range_check2.clone());
        wcm.register_component(range_check_dynamic1.clone());
        wcm.register_component(range_check_dynamic2.clone());
        wcm.register_component(range_check_mix.clone());
    }
}
