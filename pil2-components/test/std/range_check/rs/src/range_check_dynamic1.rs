use std::sync::Arc;

use witness::{WitnessComponent, execute, define_wc_with_std};

use proofman_common::{BufferPool, FromTrace, AirInstance, ProofCtx, SetupCtx};

use fields::PrimeField64;
use rand::{
    distr::{StandardUniform, Distribution},
    Rng, SeedableRng,
    rngs::StdRng,
};

use crate::RangeCheckDynamic1Trace;

define_wc_with_std!(RangeCheckDynamic1, "RngChDy1");

impl<F: PrimeField64> WitnessComponent<F> for RangeCheckDynamic1<F>
where
    StandardUniform: Distribution<F>,
{
    execute!(RangeCheckDynamic1Trace, 1);

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

            let mut trace = RangeCheckDynamic1Trace::new_from_vec(buffer_pool.take_buffer());
            let num_rows = trace.num_rows();

            tracing::debug!("··· Starting witness computation stage {}", 1);

            let range7 = self.std_lib.get_range_id(0, (1 << 7) - 1, Some(false));
            let range8 = self.std_lib.get_range_id(0, (1 << 8) - 1, Some(false));
            let range16 = self.std_lib.get_range_id(0, (1 << 16) - 1, Some(false));
            let range17 = self.std_lib.get_range_id(0, (1 << 17) - 1, Some(false));

            for i in 0..num_rows {
                let range = rng.random_range(0..=3);

                match range {
                    0 => {
                        trace[i].sel_7 = F::ONE;
                        trace[i].sel_8 = F::ZERO;
                        trace[i].sel_16 = F::ZERO;
                        trace[i].sel_17 = F::ZERO;
                        let val = rng.random_range(0..=(1 << 7) - 1);
                        trace[i].colu = F::from_u16(val);

                        self.std_lib.range_check(range7, val as i64, 1);
                    }
                    1 => {
                        trace[i].sel_7 = F::ZERO;
                        trace[i].sel_8 = F::ONE;
                        trace[i].sel_16 = F::ZERO;
                        trace[i].sel_17 = F::ZERO;
                        let val = rng.random_range(0..=(1 << 8) - 1);
                        trace[i].colu = F::from_u16(val);

                        self.std_lib.range_check(range8, val as i64, 1);
                    }
                    2 => {
                        trace[i].sel_7 = F::ZERO;
                        trace[i].sel_8 = F::ZERO;
                        trace[i].sel_16 = F::ONE;
                        trace[i].sel_17 = F::ZERO;
                        let val = rng.random_range(0..=(1 << 16) - 1);
                        trace[i].colu = F::from_u32(val);

                        self.std_lib.range_check(range16, val as i64, 1);
                    }
                    3 => {
                        trace[i].sel_7 = F::ZERO;
                        trace[i].sel_8 = F::ZERO;
                        trace[i].sel_16 = F::ZERO;
                        trace[i].sel_17 = F::ONE;
                        let val = rng.random_range(0..=(1 << 17) - 1);
                        trace[i].colu = F::from_u32(val);

                        self.std_lib.range_check(range17, val as i64, 1);
                    }
                    _ => panic!("Invalid range"),
                }
            }

            let air_instance = AirInstance::new_from_trace(FromTrace::new(&mut trace));
            pctx.add_air_instance(air_instance, instance_ids[0]);
        }
    }
}
