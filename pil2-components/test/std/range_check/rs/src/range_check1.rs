use std::sync::Arc;

use witness::{WitnessComponent, execute, define_wc_with_std};

use proofman_common::{BufferPool, FromTrace, AirInstance, ProofCtx, SetupCtx, ProofmanResult};

use fields::PrimeField64;
use rand::{Rng, SeedableRng, rngs::StdRng};

use crate::RangeCheck1Trace;

define_wc_with_std!(RangeCheck1, "RngChck1");

impl<F: PrimeField64> WitnessComponent<F> for RangeCheck1<F> {
    execute!(RangeCheck1Trace, 1);

    fn calculate_witness(
        &self,
        stage: u32,
        pctx: Arc<ProofCtx<F>>,
        _sctx: Arc<SetupCtx<F>>,
        instance_ids: &[usize],
        _n_cores: usize,
        buffer_pool: &dyn BufferPool<F>,
    ) -> ProofmanResult<()> {
        if stage == 1 {
            let mut rng = StdRng::seed_from_u64(self.seed.load(Ordering::Relaxed));

            let mut trace = RangeCheck1Trace::new_from_vec(buffer_pool.take_buffer());
            let num_rows = trace.num_rows();

            tracing::debug!("··· Starting witness computation stage {}", 1);

            let range1 = self.std_lib.get_range_id(0, (1 << 8) - 1, Some(false))?;
            let range2 = self.std_lib.get_range_id(0, (1 << 4) - 1, Some(false))?;
            let range3 = self.std_lib.get_range_id(60, (1 << 16) - 1, Some(false))?;
            let range4 = self.std_lib.get_range_id(8228, 17400, Some(false))?;

            for i in 0..num_rows {
                trace[i].a1 = F::ZERO;
                trace[i].a2 = F::ZERO;
                trace[i].a3 = F::ZERO;
                trace[i].a4 = F::ZERO;
                trace[i].a5 = F::ZERO;

                let selected1 = rng.random::<bool>();
                trace[i].sel1 = F::from_bool(selected1);

                let selected2 = rng.random::<bool>();
                trace[i].sel2 = F::from_bool(selected2);

                let selected3 = rng.random::<bool>();
                trace[i].sel3 = F::from_bool(selected3);

                if selected1 {
                    let val1 = rng.random_range(0..=(1 << 8) - 1);
                    let val2 = rng.random_range(60..=(1 << 16) - 1);
                    trace[i].a1 = F::from_u16(val1);
                    trace[i].a3 = F::from_u32(val2);

                    self.std_lib.range_check(range1, val1 as i64, 1);
                    self.std_lib.range_check(range3, val2 as i64, 1);
                }

                if selected2 {
                    let val1 = rng.random_range(0..=(1 << 4) - 1);
                    let val2 = rng.random_range(8228..=17400);
                    trace[i].a2 = F::from_u8(val1);
                    trace[i].a4 = F::from_u16(val2);

                    self.std_lib.range_check(range2, val1 as i64, 1);
                    self.std_lib.range_check(range4, val2 as i64, 1);
                }

                if selected3 {
                    let val = rng.random_range(0..=(1 << 8) - 1);
                    trace[i].a5 = F::from_u16(val);

                    self.std_lib.range_check(range1, val as i64, 1);
                }
            }

            let air_instance = AirInstance::new_from_trace(FromTrace::new(&mut trace));
            pctx.add_air_instance(air_instance, instance_ids[0]);
        }
        Ok(())
    }
}
