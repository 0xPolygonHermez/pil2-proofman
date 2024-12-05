use std::sync::Arc;

use pil_std_lib::Std;
use proofman::{WitnessComponent, WitnessManager};
use proofman_common::{AirInstance, ExecutionCtx, ProofCtx, SetupCtx};

use num_bigint::BigInt;
use p3_field::PrimeField;
use rand::{distributions::Standard, prelude::Distribution, Rng};

use crate::{MultiRangeCheck2Trace, MULTI_RANGE_CHECK_2_AIRGROUP_ID, MULTI_RANGE_CHECK_2_AIR_IDS};

pub struct MultiRangeCheck2<F: PrimeField> {
    std_lib: Arc<Std<F>>,
}

impl<F: PrimeField> MultiRangeCheck2<F>
where
    Standard: Distribution<F>,
{
    const MY_NAME: &'static str = "MtRngCh2";

    pub fn new(wcm: Arc<WitnessManager<F>>, std_lib: Arc<Std<F>>) -> Arc<Self> {
        let multi_range_check2 = Arc::new(Self { std_lib });

        wcm.register_component(
            multi_range_check2.clone(),
            Some(MULTI_RANGE_CHECK_2_AIRGROUP_ID),
            Some(MULTI_RANGE_CHECK_2_AIR_IDS),
        );

        // Register dependency relations
        multi_range_check2.std_lib.register_predecessor();

        multi_range_check2
    }

    pub fn execute(&self, pctx: Arc<ProofCtx<F>>, ectx: Arc<ExecutionCtx>, sctx: Arc<SetupCtx>) {
        let mut rng = rand::thread_rng();

        let mut trace = MultiRangeCheck2Trace::new_zeroes();
        let num_rows = trace.num_rows();

        log::debug!("{} ··· Starting witness computation stage {}", Self::MY_NAME, 1);

        let range1 = self.std_lib.get_range(BigInt::from(1 << 5), BigInt::from((1 << 8) - 1), Some(false));
        let range2 = self.std_lib.get_range(BigInt::from(1 << 8), BigInt::from((1 << 9) - 1), Some(false));
        let range3 = self.std_lib.get_range(BigInt::from(0), BigInt::from((1 << 7) - 1), Some(false));
        let range4 = self.std_lib.get_range(BigInt::from(0), BigInt::from((1 << 4) - 1), Some(false));

        for i in 0..num_rows {
            let selected1 = rng.gen_bool(0.5);
            let range_selector1 = rng.gen_bool(0.5);
            trace[i].sel[0] = F::from_bool(selected1);
            trace[i].range_sel[0] = F::from_bool(range_selector1);

            let selected2 = rng.gen_bool(0.5);
            let range_selector2 = rng.gen_bool(0.5);
            trace[i].sel[1] = F::from_bool(selected2);
            trace[i].range_sel[1] = F::from_bool(range_selector2);

            if selected1 {
                if range_selector1 {
                    trace[i].a[0] = F::from_canonical_u16(rng.gen_range((1 << 5)..=(1 << 8) - 1));

                    self.std_lib.range_check(trace[i].a[0], F::one(), range1);
                } else {
                    trace[i].a[0] = F::from_canonical_u16(rng.gen_range((1 << 8)..=(1 << 9) - 1));

                    self.std_lib.range_check(trace[i].a[0], F::one(), range2);
                }
            }

            if selected2 {
                if range_selector2 {
                    trace[i].a[1] = F::from_canonical_u16(rng.gen_range(0..=(1 << 7) - 1));

                    self.std_lib.range_check(trace[i].a[1], F::one(), range3);
                } else {
                    trace[i].a[1] = F::from_canonical_u16(rng.gen_range(0..=(1 << 4) - 1));

                    self.std_lib.range_check(trace[i].a[1], F::one(), range4);
                }
            }
        }

        AirInstance::from_trace(pctx.clone(), ectx.clone(), sctx.clone(), None, &mut trace);
    }
}

impl<F: PrimeField> WitnessComponent<F> for MultiRangeCheck2<F>
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
