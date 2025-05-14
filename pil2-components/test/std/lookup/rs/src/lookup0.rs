use std::sync::Arc;

use fields::PrimeField64;
use witness::{WitnessComponent, execute, define_wc};
use proofman_common::{FromTrace, AirInstance, ProofCtx, SetupCtx};

use rand::{
    distr::{StandardUniform, Distribution},
    Rng, SeedableRng,
    rngs::StdRng,
};

use crate::Lookup0Trace;

define_wc!(Lookup0, "Lookup_0");

impl<F: PrimeField64> WitnessComponent<F> for Lookup0
where
    StandardUniform: Distribution<F>,
{
    execute!(Lookup0Trace, 1);

    fn calculate_witness(&self, stage: u32, pctx: Arc<ProofCtx<F>>, _sctx: Arc<SetupCtx<F>>, instance_ids: &[usize]) {
        if stage == 1 {
            let seed = if cfg!(feature = "debug") { 0 } else { rand::rng().random::<u64>() };
            let mut rng = StdRng::seed_from_u64(seed);

            let mut trace = Lookup0Trace::new();
            let num_rows = trace.num_rows();

            log::debug!("{} ··· Starting witness computation stage {}", Self::MY_NAME, 1);

            let num_lookups = trace[0].sel.len();

            for j in 0..num_lookups {
                for i in 0..num_rows {
                    // Assumes
                    trace[i].f[2 * j] = rng.random();
                    trace[i].f[2 * j + 1] = rng.random();
                    let selected = rng.random::<bool>();
                    trace[i].sel[j] = F::from_bool(selected);

                    // Proves
                    trace[i].t[2 * j] = trace[i].f[2 * j];
                    trace[i].t[2 * j + 1] = trace[i].f[2 * j + 1];
                    if selected {
                        trace[i].mul[j] = F::ONE;
                    } else {
                        trace[i].mul[j] = F::ZERO;
                    }
                }
            }

            let air_instance = AirInstance::new_from_trace(FromTrace::new(&mut trace));
            pctx.add_air_instance(air_instance, instance_ids[0]);
        }
    }
}
