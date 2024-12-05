use std::sync::Arc;

use pil_std_lib::Std;
use proofman::{WitnessComponent, WitnessManager};
use proofman_common::{AirInstance, ExecutionCtx, ProofCtx, SetupCtx};

use num_bigint::BigInt;
use p3_field::PrimeField;
use rand::{distributions::Standard, prelude::Distribution, Rng};

use crate::{RangeCheck2Trace, RANGE_CHECK_2_AIRGROUP_ID, RANGE_CHECK_2_AIR_IDS};

pub struct RangeCheck2<F: PrimeField> {
    std_lib: Arc<Std<F>>,
}

impl<F: PrimeField> RangeCheck2<F>
where
    Standard: Distribution<F>,
{
    const MY_NAME: &'static str = "RngChck2";

    pub fn new(wcm: Arc<WitnessManager<F>>, std_lib: Arc<Std<F>>) -> Arc<Self> {
        let range_check1 = Arc::new(Self { std_lib });

        wcm.register_component(range_check1.clone(), Some(RANGE_CHECK_2_AIRGROUP_ID), Some(RANGE_CHECK_2_AIR_IDS));

        // Register dependency relations
        range_check1.std_lib.register_predecessor();

        range_check1
    }

    pub fn execute(&self, pctx: Arc<ProofCtx<F>>, ectx: Arc<ExecutionCtx>, sctx: Arc<SetupCtx>) {
        let mut rng = rand::thread_rng();
        let mut trace = RangeCheck2Trace::new_zeroes();
        let num_rows = trace.num_rows();

        log::debug!("{} ··· Starting witness computation stage {}", Self::MY_NAME, 1);

        let range1 = self.std_lib.get_range(BigInt::from(0), BigInt::from((1 << 8) - 1), Some(false));
        let range2 = self.std_lib.get_range(BigInt::from(0), BigInt::from((1 << 9) - 1), Some(false));
        let range3 = self.std_lib.get_range(BigInt::from(0), BigInt::from((1 << 10) - 1), Some(false));

        for i in 0..num_rows {
            trace[i].b1 = F::from_canonical_u16(rng.gen_range(0..=(1 << 8) - 1));
            trace[i].b2 = F::from_canonical_u16(rng.gen_range(0..=(1 << 9) - 1));
            trace[i].b3 = F::from_canonical_u16(rng.gen_range(0..=(1 << 10) - 1));

            self.std_lib.range_check(trace[i].b1, F::one(), range1);
            self.std_lib.range_check(trace[i].b2, F::one(), range2);
            self.std_lib.range_check(trace[i].b3, F::one(), range3);
        }

        AirInstance::from_trace(pctx.clone(), ectx.clone(), sctx.clone(), None, &mut trace);
    }
}

impl<F: PrimeField> WitnessComponent<F> for RangeCheck2<F>
where
    Standard: Distribution<F>,
{
    fn calculate_witness(
        &self,
        _stage: u32,
        _air_instance_id: Option<usize>,
        pctx: Arc<ProofCtx<F>>,
        _ectx: Arc<ExecutionCtx>,
        _sctx: Arc<SetupCtx>,
    ) {
        self.std_lib.unregister_predecessor(pctx, None);
    }
}
