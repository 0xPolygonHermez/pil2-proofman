use std::sync::Arc;

use proofman_witness::{WitnessComponent, execute, define_wc};

use proofman_common::{AirInstance, FromTrace, ProofCtx, ProofmanResult, SetupCtx};
use proofman_common::BufferPool;

use proofman_fields::PrimeField64;
use rand::{SeedableRng, rngs::StdRng, RngExt};

use crate::RangeCheck2Trace;

define_wc!(RangeCheck2, "RngChck2");

impl<F: PrimeField64> WitnessComponent<F> for RangeCheck2 {
    execute!(RangeCheck2Trace, 1);

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
            let mut trace = RangeCheck2Trace::new_from_vec(buffer_pool.take_buffer())?;
            let num_rows = trace.num_rows();

            tracing::debug!("··· Starting witness computation stage {}", 1);

            for i in 0..num_rows {
                let val1 = rng.random_range(0..=(1 << 8) - 1);
                let val2 = rng.random_range(0..=(1 << 9) - 1);
                let val3 = rng.random_range(0..=(1 << 10) - 1);
                trace[i].b1 = F::from_u16(val1);
                trace[i].b2 = F::from_u16(val2);
                trace[i].b3 = F::from_u16(val3);
            }

            let air_instance = AirInstance::new_from_trace(FromTrace::new(&mut trace));
            pctx.add_air_instance(air_instance, instance_ids[0]);
        }
        Ok(())
    }
}
