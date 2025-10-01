use std::sync::Arc;

use witness::{WitnessComponent, execute, define_wc_with_std};

use proofman_common::{BufferPool, FromTrace, AirInstance, ProofCtx, SetupCtx};

use fields::PrimeField64;
use rand::{Rng, SeedableRng, rngs::StdRng};

use crate::RangeCheck3Trace;

define_wc_with_std!(RangeCheck3, "RngChck3");

impl<F: PrimeField64> WitnessComponent<F> for RangeCheck3<F> {
    execute!(RangeCheck3Trace, 1);

    fn calculate_witness(
        &self,
        stage: u32,
        pctx: Arc<ProofCtx<F>>,
        _sctx: Arc<SetupCtx<F>>,
        instance_ids: &[usize],
        _n_cores: usize,
        buffer_pool: &dyn BufferPool<F>,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        if stage == 1 {
            let mut rng = StdRng::seed_from_u64(self.seed.load(Ordering::Relaxed));
            let mut trace = RangeCheck3Trace::new_from_vec(buffer_pool.take_buffer());
            let num_rows = trace.num_rows();

            tracing::debug!("··· Starting witness computation stage {}", 1);

            let range1 = self.std_lib.get_range_id(0, (1 << 4) - 1, Some(false))?;
            let range2 = self.std_lib.get_range_id(0, (1 << 8) - 1, Some(false))?;

            for i in 0..num_rows {
                let val1 = rng.random_range(0..=(1 << 4) - 1);
                let val2 = rng.random_range(0..=(1 << 8) - 1);
                trace[i].c1 = F::from_u16(val1);
                trace[i].c2 = F::from_u16(val2);

                self.std_lib.range_check(range1, val1 as i64, 1);
                self.std_lib.range_check(range2, val2 as i64, 1);
            }

            let air_instance = AirInstance::new_from_trace(FromTrace::new(&mut trace));
            pctx.add_air_instance(air_instance, instance_ids[0]);
        }
        Ok(())
    }
}
