use std::sync::Arc;

use witness::{WitnessComponent, execute, define_wc};
use proofman_common::{BufferPool, FromTrace, AirInstance, ProofCtx, SetupCtx};

use fields::PrimeField64;
use rand::{rng, rngs::StdRng, seq::SliceRandom, Rng, SeedableRng};

use crate::Permutation1_6Trace;

define_wc!(Permutation1_6, "Perm1_6 ");

impl<F: PrimeField64> WitnessComponent<F> for Permutation1_6 {
    execute!(Permutation1_6Trace, 2);

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

            let mut trace = Permutation1_6Trace::new_from_vec(buffer_pool.take_buffer());
            let num_rows = trace.num_rows();

            tracing::debug!("··· Starting witness computation stage {}", 1);

            // TODO: Add the ability to send inputs to permutation2
            //       and consequently add random selectors

            // Assumes
            for i in 0..num_rows {
                trace[i].a1 = F::from_u64(rng.random_range(0..=(1 << 63) - 1));
                trace[i].b1 = F::from_u64(rng.random_range(0..=(1 << 63) - 1));

                trace[i].a2 = F::from_u8(200);
                trace[i].b2 = F::from_u8(201);

                trace[i].a3 = F::from_u64(rng.random_range(0..=(1 << 63) - 1));
                trace[i].b3 = F::from_u64(rng.random_range(0..=(1 << 63) - 1));

                trace[i].a4 = F::from_u8(100);
                trace[i].b4 = F::from_u8(101);

                trace[i].sel1 = F::from_bool(rng.random_bool(0.5));
                trace[i].sel3 = F::ONE;
            }

            let mut indices: Vec<usize> = (0..num_rows).collect();
            indices.shuffle(&mut rng);

            // Proves
            for i in 0..num_rows {
                // We take a random permutation of the indices to show that the permutation check is passing
                trace[i].c1 = trace[indices[i]].a1;
                trace[i].d1 = trace[indices[i]].b1;

                trace[i].c2 = trace[indices[i]].a3;
                trace[i].d2 = trace[indices[i]].b3;

                trace[i].sel2 = trace[indices[i]].sel1;
            }

            let air_instance = AirInstance::new_from_trace(FromTrace::new(&mut trace));
            pctx.add_air_instance(air_instance, instance_ids[0]);
        }
        Ok(())
    }
}
