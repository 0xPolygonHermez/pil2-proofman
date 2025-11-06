use std::sync::Arc;

use witness::{WitnessComponent, execute, define_wc};
use proofman_common::{BufferPool, FromTrace, AirInstance, ProofCtx, SetupCtx, ProofmanResult};

use fields::PrimeField64;
use rand::{Rng, SeedableRng, rngs::StdRng, rng};

use crate::Lookup2_12Trace;

define_wc!(Lookup2_12, "Lkup2_12");

impl<F: PrimeField64> WitnessComponent<F> for Lookup2_12 {
    execute!(Lookup2_12Trace, 1);

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

            let mut trace = Lookup2_12Trace::new_from_vec(buffer_pool.take_buffer());
            let num_rows = trace.num_rows();

            tracing::debug!("··· Starting witness computation stage {}", 1);

            // TODO: Add the ability to send inputs to lookup3
            //       and consequently add random selectors

            for i in 0..num_rows {
                // Inner lookups
                trace[i].a1 = F::from_u64(rng.random_range(0..=(1 << 63) - 1));
                trace[i].b1 = F::from_u64(rng.random_range(0..=(1 << 63) - 1));
                trace[i].c1 = trace[i].a1;
                trace[i].d1 = trace[i].b1;

                trace[i].a3 = F::from_u64(rng.random_range(0..=(1 << 63) - 1));
                trace[i].b3 = F::from_u64(rng.random_range(0..=(1 << 63) - 1));
                trace[i].c2 = trace[i].a3;
                trace[i].d2 = trace[i].b3;
                let selected = rng.random::<bool>();
                trace[i].sel1 = F::from_bool(selected);
                if selected {
                    trace[i].mul = trace[i].sel1;
                } else {
                    trace[i].mul = F::ZERO;
                }

                // Outer lookups
                trace[i].a2 = F::from_usize(i);
                trace[i].b2 = F::from_usize(i);

                trace[i].a4 = F::from_usize(i);
                trace[i].b4 = F::from_usize(i);
                trace[i].sel2 = F::from_bool(true);
            }

            let air_instance = AirInstance::new_from_trace(FromTrace::new(&mut trace));
            pctx.add_air_instance(air_instance, instance_ids[0]);
        }
        Ok(())
    }
}
