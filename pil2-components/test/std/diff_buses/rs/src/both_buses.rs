use std::sync::Arc;

use witness::{WitnessComponent, execute, define_wc};
use proofman_common::{BufferPool, FromTrace, AirInstance, ProofCtx, SetupCtx, ProofmanResult};

use fields::PrimeField64;
use rand::{rng, Rng, SeedableRng, rngs::StdRng};

use crate::BothBusesTrace;

define_wc!(BothBuses, "BothBus ");

impl<F: PrimeField64> WitnessComponent<F> for BothBuses {
    execute!(BothBusesTrace, 1);

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
            let seed = if cfg!(feature = "debug") { 0 } else { rng().random::<u64>() };
            let mut rng = StdRng::seed_from_u64(seed);

            let mut trace = BothBusesTrace::new_from_vec(buffer_pool.take_buffer());
            let num_rows = trace.num_rows();

            tracing::debug!("··· Starting witness computation stage {}", 1);

            for i in 0..num_rows {
                trace[i].a = F::from_u64(rng.random_range(0..=(1 << 63) - 1));
                trace[i].b = trace[i].a;

                trace[i].c = F::from_usize(i);
                trace[i].d = trace[i].c;
            }

            let air_instance = AirInstance::new_from_trace(FromTrace::new(&mut trace));
            pctx.add_air_instance(air_instance, instance_ids[0]);
        }
        Ok(())
    }
}
