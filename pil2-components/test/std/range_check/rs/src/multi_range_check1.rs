use std::sync::Arc;

use witness::{define_wc_with_std, execute, WitnessComponent};

use proofman_common::{BufferPool, FromTrace, AirInstance, ProofCtx, SetupCtx};

use fields::PrimeField64;
use rand::{Rng, SeedableRng, rngs::StdRng};

use crate::MultiRangeCheck1Trace;

define_wc_with_std!(MultiRangeCheck1, "MtRngCh1");

impl<F: PrimeField64> WitnessComponent<F> for MultiRangeCheck1<F> {
    execute!(MultiRangeCheck1Trace, 1);

    fn calculate_witness(
        &self,
        stage: u32,
        pctx: Arc<ProofCtx<F>>,
        _sctx: Arc<SetupCtx<F>>,
        instance_ids: &[usize],
        _n_cores: usize,
        buffer_pool: &dyn BufferPool<F>,
    ) {
        if stage == 1 {
            let mut rng = StdRng::seed_from_u64(self.seed.load(Ordering::Relaxed));

            let mut trace = MultiRangeCheck1Trace::new_from_vec(buffer_pool.take_buffer());
            let num_rows = trace.num_rows();

            tracing::debug!("··· Starting witness computation stage {}", 1);

            let range1 = self.std_lib.get_range_id(0, (1 << 7) - 1, Some(false));
            let range2 = self.std_lib.get_range_id(0, (1 << 8) - 1, Some(false));
            let range3 = self.std_lib.get_range_id(0, (1 << 6) - 1, Some(false));
            let range4 = self.std_lib.get_range_id(1 << 5, (1 << 8) - 1, Some(false));
            let range5 = self.std_lib.get_range_id(1 << 8, (1 << 9) - 1, Some(false));

            for i in 0..num_rows {
                let selected1 = rng.random::<bool>();
                let range_selector1 = rng.random::<bool>();
                trace[i].sel[0] = F::from_bool(selected1);
                trace[i].range_sel[0] = F::from_bool(range_selector1);

                let selected2 = rng.random::<bool>();
                let range_selector2 = rng.random::<bool>();
                trace[i].sel[1] = F::from_bool(selected2);
                trace[i].range_sel[1] = F::from_bool(range_selector2);

                let selected3 = rng.random::<bool>();
                let range_selector3 = rng.random::<bool>();
                trace[i].sel[2] = F::from_bool(selected3);
                trace[i].range_sel[2] = F::from_bool(range_selector3);

                trace[i].a[0] = F::ZERO;
                trace[i].a[1] = F::ZERO;
                trace[i].a[2] = F::ZERO;

                if selected1 {
                    if range_selector1 {
                        let val = rng.random_range(0..=(1 << 7) - 1);
                        trace[i].a[0] = F::from_u16(val);

                        self.std_lib.range_check(range1, val as i64, 1);
                    } else {
                        let val = rng.random_range(0..=(1 << 8) - 1);
                        trace[i].a[0] = F::from_u16(val);

                        self.std_lib.range_check(range2, val as i64, 1);
                    }
                }

                if selected2 {
                    if range_selector2 {
                        let val = rng.random_range(0..=(1 << 7) - 1);
                        trace[i].a[1] = F::from_u16(val);

                        self.std_lib.range_check(range1, val as i64, 1);
                    } else {
                        let val = rng.random_range(0..=(1 << 6) - 1);
                        trace[i].a[1] = F::from_u16(val);

                        self.std_lib.range_check(range3, val as i64, 1);
                    }
                }

                if selected3 {
                    if range_selector3 {
                        let val = rng.random_range((1 << 5)..=(1 << 8) - 1);
                        trace[i].a[2] = F::from_u16(val);

                        self.std_lib.range_check(range4, val as i64, 1);
                    } else {
                        let val = rng.random_range((1 << 8)..=(1 << 9) - 1);
                        trace[i].a[2] = F::from_u16(val);

                        self.std_lib.range_check(range5, val as i64, 1);
                    }
                }
            }

            let air_instance = AirInstance::new_from_trace(FromTrace::new(&mut trace));
            pctx.add_air_instance(air_instance, instance_ids[0]);
        }
    }
}
