use pil2_std_lib::Std;
use proofman_common::ProofmanResult;
use proofman_witness::{witness_library, WitnessLibrary, WitnessManager};

use proofman_fields::PrimeField64;
use proofman_fields::Goldilocks;
use rand::{rng, RngExt};

use proofman::register_std;

use crate::{
    RangeCheckMix, RangeCheckDynamic1, RangeCheckDynamic2, MultiRangeCheck1, MultiRangeCheck2, RangeCheck1,
    RangeCheck2, RangeCheck3, RangeCheck4,
};

witness_library!(WitnessLib, Goldilocks);

impl<F: PrimeField64> WitnessLibrary<F> for WitnessLib {
    fn register_witness(&mut self, wcm: &WitnessManager<F>) -> ProofmanResult<()> {
        let seed = if cfg!(feature = "debug") { 0 } else { rng().random::<u64>() };

        let std_lib = Std::new(wcm.get_pctx(), wcm.get_sctx(), false)?;
        let range_check1 = RangeCheck1::new();
        let range_check2 = RangeCheck2::new();
        let range_check3 = RangeCheck3::new();
        let range_check4 = RangeCheck4::new();
        let multi_range_check1 = MultiRangeCheck1::new();
        let multi_range_check2 = MultiRangeCheck2::new();
        let range_check_dynamic1 = RangeCheckDynamic1::new();
        let range_check_dynamic2 = RangeCheckDynamic2::new();
        let range_check_mix = RangeCheckMix::new();

        register_std(wcm, &std_lib);
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
        Ok(())
    }
}
