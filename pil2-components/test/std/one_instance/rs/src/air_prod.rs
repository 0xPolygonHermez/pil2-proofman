use std::sync::Arc;

use witness::{WitnessComponent, execute, define_wc};
use proofman_common::{BufferPool, FromTrace, AirInstance, ProofCtx, SetupCtx};

use fields::PrimeField64;
use rand::{rng, rngs::StdRng, Rng, SeedableRng};

use crate::AirProdTrace;

define_wc!(AirProd, "AirProd ");

impl<F: PrimeField64> WitnessComponent<F> for AirProd {
    execute!(AirProdTrace, 1);

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
            let seed = if cfg!(feature = "debug") { 0 } else { rng().random::<u64>() };
            let mut rng = StdRng::seed_from_u64(seed);

            let mut trace = AirProdTrace::new_from_vec(buffer_pool.take_buffer());
            let num_rows = trace.num_rows();

            tracing::debug!("··· Starting witness computation stage {}", 1);

            for i in 0..num_rows {
                let r = F::from_u64(rng.random_range(0..=(1 << 63) - 1));
                trace[i].a = r;
                trace[i].b = r;
                trace[i].c = F::ZERO;
            }

            let air_instance = AirInstance::new_from_trace(FromTrace::new(&mut trace));
            pctx.add_air_instance(air_instance, instance_ids[0]);
        }
    }
}
