use std::sync::Arc;

use witness::{WitnessComponent, execute, define_wc};
use proofman_common::{BufferPool, FromTrace, AirInstance, ProofCtx, SetupCtx};

use fields::PrimeField64;
use rand::{Rng, SeedableRng, rngs::StdRng, rng};

use crate::Lookup1Trace;

define_wc!(Lookup1, "Lookup_1");

impl<F: PrimeField64> WitnessComponent<F> for Lookup1 {
    execute!(Lookup1Trace, 1);

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
            let seed = if cfg!(feature = "debug") { 0 } else { rng().random::<u64>() };
            let mut rng = StdRng::seed_from_u64(seed);

            let mut trace = Lookup1Trace::new_from_vec(buffer_pool.take_buffer());
            let num_rows = trace.num_rows();

            tracing::debug!("··· Starting witness computation stage {}", 1);

            let num_lookups = trace[0].sel.len();

            for i in 0..num_rows {
                let val = F::from_u64(rng.random_range(0..=(1 << 63) - 1));
                let mut n_sel = 0;
                for j in 0..num_lookups {
                    trace[i].f[j] = val;
                    let selected = rng.random::<bool>();
                    trace[i].sel[j] = F::from_bool(selected);
                    if selected {
                        n_sel += 1;
                    }
                }
                trace[i].t = val;
                trace[i].mul = F::from_usize(n_sel);
            }

            let air_instance = AirInstance::new_from_trace(FromTrace::new(&mut trace));
            pctx.add_air_instance(air_instance, instance_ids[0]);
        }
        Ok(())
    }
}
