use std::sync::Arc;

use witness::{WitnessComponent, execute, define_wc};
use proofman_common::{FromTrace, AirInstance, ProofCtx, SetupCtx};

use fields::PrimeField64;
use rand::{
    distr::{StandardUniform, Distribution},
    Rng, SeedableRng,
    rngs::StdRng,
};

use crate::Lookup2_15Trace;

define_wc!(Lookup2_15, "Lkup2_15");

impl<F: PrimeField64> WitnessComponent<F> for Lookup2_15
where
    StandardUniform: Distribution<F>,
{
    execute!(Lookup2_15Trace, 1);

    fn calculate_witness(&self, stage: u32, pctx: Arc<ProofCtx<F>>, _sctx: Arc<SetupCtx<F>>, instance_ids: &[usize]) {
        if stage == 1 {
            let seed = if cfg!(feature = "debug") { 0 } else { rand::rng().random::<u64>() };
            let mut rng = StdRng::seed_from_u64(seed);

            let mut trace = Lookup2_15Trace::new();
            let num_rows = trace.num_rows();

            tracing::debug!("··· Starting witness computation stage {}", 1);

            // TODO: Add the ability to send inputs to lookup3
            //       and consequently add random selectors

            for i in 0..num_rows {
                // Inner lookups
                trace[i].a1 = rng.random();
                trace[i].b1 = rng.random();
                trace[i].c1 = trace[i].a1;
                trace[i].d1 = trace[i].b1;

                trace[i].a3 = rng.random();
                trace[i].b3 = rng.random();
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
                trace[i].a2 = F::from_usize(i % (1 << 14));
                trace[i].b2 = F::from_usize(i % (1 << 14));

                trace[i].a4 = F::from_usize(i % (1 << 14));
                trace[i].b4 = F::from_usize(i % (1 << 14));
                trace[i].sel2 = F::from_bool(true);
            }

            let air_instance = AirInstance::new_from_trace(FromTrace::new(&mut trace));
            pctx.add_air_instance(air_instance, instance_ids[0]);
        }
    }
}
