use std::sync::Arc;

use witness::{WitnessComponent, execute, define_wc};
use proofman_common::{FromTrace, AirInstance, ProofCtx, SetupCtx};

use p3_field::PrimeField64;
use rand::{
    distr::{StandardUniform, Distribution},
    Rng, SeedableRng,
    rngs::StdRng,
    seq::SliceRandom,
};

use crate::SimpleLeftTrace;

define_wc!(SimpleLeft, "SimLeft ");

impl<F: PrimeField64 + Copy> WitnessComponent<F> for SimpleLeft
where
    StandardUniform: Distribution<F>,
{
    execute!(SimpleLeftTrace, 1);

    fn calculate_witness(&self, stage: u32, pctx: Arc<ProofCtx<F>>, _sctx: Arc<SetupCtx<F>>, instance_ids: &[usize]) {
        if stage == 1 {
            let seed = if cfg!(feature = "debug") { 0 } else { rand::rng().random::<u64>() };
            let mut rng = StdRng::seed_from_u64(seed);

            let mut trace = SimpleLeftTrace::new();
            let num_rows = trace.num_rows();

            log::debug!("{} ··· Starting witness computation stage {}", Self::MY_NAME, 1);

            // Assumes
            for i in 0..num_rows {
                trace[i].a = F::from_u64(rng.random_range(0..=(1 << 63) - 1));
                trace[i].b = F::from_u64(rng.random_range(0..=(1 << 63) - 1));

                trace[i].e = F::from_u8(200);
                trace[i].f = F::from_u8(201);

                trace[i].g = F::from_usize(i);
                trace[i].h = F::from_usize(num_rows - i - 1);
            }

            let mut indices: Vec<usize> = (0..num_rows).collect();
            indices.shuffle(&mut rng);

            // Proves
            for i in 0..num_rows {
                // We take a random permutation of the indices to show that the permutation check is passing
                trace[i].c = trace[indices[i]].a;
                trace[i].d = trace[indices[i]].b;
            }

            let air_instance = AirInstance::new_from_trace(FromTrace::new(&mut trace));
            pctx.add_air_instance(air_instance, instance_ids[0]);
        }
    }
}
