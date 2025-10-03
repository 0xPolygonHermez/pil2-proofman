use std::sync::Arc;

use witness::{WitnessComponent, execute, define_wc};
use proofman_common::{BufferPool, FromTrace, AirInstance, ProofCtx, SetupCtx, ProofmanResult};
use fields::PrimeField64;
use rand::{rng, Rng, SeedableRng, rngs::StdRng};

use crate::Permutation1_7Trace;

define_wc!(Permutation1_7, "Perm1_7 ");

impl<F: PrimeField64> WitnessComponent<F> for Permutation1_7 {
    execute!(Permutation1_7Trace, 1);

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
            let mut trace = Permutation1_7Trace::new_from_vec(buffer_pool.take_buffer());
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

                trace[i].sel1 = F::ONE;
                trace[i].sel3 = F::ONE; // F::from_u8(rng.random_range(0..=1));
            }

            // TODO: Add the permutation of indexes

            // Proves
            for i in 0..num_rows {
                let index = num_rows - i - 1;
                // let mut index = rng.random_range(0..num_rows);
                trace[i].c1 = trace[index].a1;
                trace[i].d1 = trace[index].b1;

                // index = rng.random_range(0..num_rows);
                trace[i].c2 = trace[index].a3;
                trace[i].d2 = trace[index].b3;

                trace[i].sel2 = trace[i].sel1;
            }

            let air_instance = AirInstance::new_from_trace(FromTrace::new(&mut trace));
            pctx.add_air_instance(air_instance, instance_ids[0]);
        }
        Ok(())
    }
}
