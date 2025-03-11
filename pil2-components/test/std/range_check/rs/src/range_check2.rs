use std::sync::Arc;

use witness::{WitnessComponent, execute, define_wc_with_std};

use proofman_common::{FromTrace, AirInstance, ProofCtx, SetupCtx};

use p3_field::PrimeField64;
use rand::{distributions::Standard, prelude::Distribution, Rng, SeedableRng, rngs::StdRng};

use crate::RangeCheck2Trace;

define_wc_with_std!(RangeCheck2, "RngChck2");

impl<F: PrimeField64> WitnessComponent<F> for RangeCheck2<F>
where
    Standard: Distribution<F>,
{
    execute!(RangeCheck2Trace, 1);

    fn calculate_witness(&self, stage: u32, pctx: Arc<ProofCtx<F>>, _sctx: Arc<SetupCtx<F>>, instance_ids: &[usize]) {
        if stage == 1 {
            let mut rng = StdRng::seed_from_u64(self.seed.load(Ordering::Relaxed));
            let mut trace = RangeCheck2Trace::new();
            let num_rows = trace.num_rows();

            log::debug!("{} ··· Starting witness computation stage {}", Self::MY_NAME, 1);

            let range1 = self.std_lib.get_range(0, (1 << 8) - 1, Some(false));
            let range2 = self.std_lib.get_range(0, (1 << 9) - 1, Some(false));
            let range3 = self.std_lib.get_range(0, (1 << 10) - 1, Some(false));

            for i in 0..num_rows {
                let val1 = rng.gen_range(0..=(1 << 8) - 1);
                let val2 = rng.gen_range(0..=(1 << 9) - 1);
                let val3 = rng.gen_range(0..=(1 << 10) - 1);
                trace[i].b1 = F::from_canonical_u16(val1);
                trace[i].b2 = F::from_canonical_u16(val2);
                trace[i].b3 = F::from_canonical_u16(val3);

                self.std_lib.range_check(val1 as i64, 1, range1);
                self.std_lib.range_check(val2 as i64, 1, range2);
                self.std_lib.range_check(val3 as i64, 1, range3);
            }

            let air_instance = AirInstance::new_from_trace(FromTrace::new(&mut trace));
            pctx.add_air_instance(air_instance, instance_ids[0]);
        }
    }
}
