use std::sync::Arc;

use witness::{WitnessComponent, execute, define_wc_with_std};

use proofman_common::{FromTrace, AirInstance, ProofCtx, SetupCtx};

use p3_field::PrimeField64;
use rand::{
    distr::{StandardUniform, Distribution},
    Rng, SeedableRng,
    rngs::StdRng,
};

use crate::MultiRangeCheck2Trace;

define_wc_with_std!(MultiRangeCheck2, "MtRngCh2");

impl<F: PrimeField64> WitnessComponent<F> for MultiRangeCheck2<F>
where
    StandardUniform: Distribution<F>,
{
    execute!(MultiRangeCheck2Trace, 1);

    fn calculate_witness(
        &self,
        stage: u32,
        pctx: Arc<ProofCtx<F>>,
        _sctx: Arc<SetupCtx<F>>,
        instance_ids: &[usize],
        _core_id: usize,
        _n_cores: usize,
    ) {
        if stage == 1 {
            let mut rng = StdRng::seed_from_u64(self.seed.load(Ordering::Relaxed));

            let mut trace = MultiRangeCheck2Trace::new();
            let num_rows = trace.num_rows();

            log::debug!("{} ··· Starting witness computation stage {}", Self::MY_NAME, 1);

            let range1 = self.std_lib.get_range(1 << 5, (1 << 8) - 1, Some(false));
            let range2 = self.std_lib.get_range(1 << 8, (1 << 9) - 1, Some(false));
            let range3 = self.std_lib.get_range(0, (1 << 7) - 1, Some(false));
            let range4 = self.std_lib.get_range(0, (1 << 4) - 1, Some(false));

            for i in 0..num_rows {
                trace[i].a[0] = F::ZERO;
                trace[i].a[1] = F::ZERO;

                let selected1 = rng.random::<bool>();
                let range_selector1 = rng.random::<bool>();
                trace[i].sel[0] = F::from_bool(selected1);
                trace[i].range_sel[0] = F::from_bool(range_selector1);

                let selected2 = rng.random::<bool>();
                let range_selector2 = rng.random::<bool>();
                trace[i].sel[1] = F::from_bool(selected2);
                trace[i].range_sel[1] = F::from_bool(range_selector2);

                if selected1 {
                    if range_selector1 {
                        let val = rng.random_range((1 << 5)..=(1 << 8) - 1);
                        trace[i].a[0] = F::from_u16(val);

                        self.std_lib.range_check(val as i64, 1, range1);
                    } else {
                        let val = rng.random_range((1 << 8)..=(1 << 9) - 1);
                        trace[i].a[0] = F::from_u16(val);

                        self.std_lib.range_check(val as i64, 1, range2);
                    }
                }

                if selected2 {
                    if range_selector2 {
                        let val = rng.random_range(0..=(1 << 7) - 1);
                        trace[i].a[1] = F::from_u16(val);

                        self.std_lib.range_check(val as i64, 1, range3);
                    } else {
                        let val = rng.random_range(0..=(1 << 4) - 1);
                        trace[i].a[1] = F::from_u16(val);

                        self.std_lib.range_check(val as i64, 1, range4);
                    }
                }
            }

            let air_instance = AirInstance::new_from_trace(FromTrace::new(&mut trace));
            pctx.add_air_instance(air_instance, instance_ids[0]);
        }
    }
}
