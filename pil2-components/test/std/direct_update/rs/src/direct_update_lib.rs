use std::sync::Arc;

use pil_std_lib::Std;
use witness::{witness_library, WitnessLibrary, WitnessManager};

use fields::PrimeField64;
use fields::Goldilocks;
use rand::{
    distr::{StandardUniform, Distribution},
};
use proofman::register_std;

use crate::{DirectUpdateProdLocal, DirectUpdateProdGlobal, DirectUpdateSumLocal, DirectUpdateSumGlobal};

witness_library!(WitnessLib, Goldilocks);

impl<F: PrimeField64> WitnessLibrary<F> for WitnessLib
where
    StandardUniform: Distribution<F>,
{
    fn register_witness(&mut self, wcm: Arc<WitnessManager<F>>) {
        // let seed = if cfg!(feature = "debug") { 0 } else { rand::rng().random::<u64>() };
        let seed = 0;

        let std = Std::new(wcm.get_pctx(), wcm.get_sctx());
        let direct_update_prod_local = DirectUpdateProdLocal::new();
        let direct_update_prod_global = DirectUpdateProdGlobal::new();
        let direct_update_sum_local = DirectUpdateSumLocal::new();
        let direct_update_sum_global = DirectUpdateSumGlobal::new();

        register_std(&wcm, &std);
        direct_update_prod_local.set_seed(seed);
        direct_update_prod_global.set_seed(seed);
        direct_update_sum_local.set_seed(seed);
        direct_update_sum_global.set_seed(seed);

        wcm.register_component(direct_update_prod_local.clone());
        wcm.register_component(direct_update_prod_global.clone());
        wcm.register_component(direct_update_sum_local.clone());
        wcm.register_component(direct_update_sum_global.clone());
    }
}
