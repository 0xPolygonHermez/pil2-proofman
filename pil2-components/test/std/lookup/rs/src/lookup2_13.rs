use std::sync::Arc;

use fields::PrimeField64;
use witness::{WitnessComponent, execute, define_wc};
use proofman_common::{FromTrace, AirInstance, ProofCtx, SetupCtx};

use rand::{
    distr::{StandardUniform, Distribution},
    Rng, SeedableRng,
    rngs::StdRng,
};

use crate::Lookup2_13Trace;

define_wc!(Lookup2_13, "Lkup2_13");

impl<F: PrimeField64> WitnessComponent<F> for Lookup2_13
where
    StandardUniform: Distribution<F>,
{
    execute!(Lookup2_13Trace, 1);

    fn calculate_witness(&self, stage: u32, pctx: Arc<ProofCtx<F>>, _sctx: Arc<SetupCtx<F>>, instance_ids: &[usize]) {
        if stage == 1 {
            let seed = if cfg!(feature = "debug") { 0 } else { rand::rng().random::<u64>() };
            let mut rng = StdRng::seed_from_u64(seed);

            let mut trace = Lookup2_13Trace::new();
            let num_rows = trace.num_rows();

            // TODO: Add the ability to send inputs to lookup3
            //       and consequently add random selectors

            log::debug!("{} ··· Starting witness computation stage {}", Self::MY_NAME, 1);

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
                trace[i].a2 = F::from_usize(i);
                trace[i].b2 = F::from_usize(i);

                trace[i].a4 = F::from_usize(i);
                trace[i].b4 = F::from_usize(i);
                trace[i].sel2 = F::from_bool(true);
            }

            let air_instance = AirInstance::new_from_trace(FromTrace::new(&mut trace));
            pctx.add_air_instance(air_instance, instance_ids[0]);
        }
    }
}
