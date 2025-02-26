use std::sync::Arc;

use witness::{WitnessComponent, execute, define_wc};
use proofman_common::{FromTrace, AirInstance, ProofCtx, SetupCtx};

use p3_field::PrimeField;
use rand::{distributions::Standard, prelude::Distribution, Rng, SeedableRng, rngs::StdRng};

use crate::BothBusesTrace;

define_wc!(BothBuses, "BothBus ");

impl<F: PrimeField> WitnessComponent<F> for BothBuses
where
    Standard: Distribution<F>,
{
    execute!(BothBusesTrace, 1);

    fn calculate_witness(&self, stage: u32, pctx: Arc<ProofCtx<F>>, _sctx: Arc<SetupCtx<F>>, instance_ids: &[usize]) {
        if stage == 1 {
            let seed = if cfg!(feature = "debug") { 0 } else { rand::thread_rng().gen::<u64>() };
            let mut rng = StdRng::seed_from_u64(seed);

            let mut trace = BothBusesTrace::new();
            let num_rows = trace.num_rows();

            log::debug!("{}: ··· Starting witness computation stage {}", Self::MY_NAME, 1);

            for i in 0..num_rows {
                trace[i].a = F::from_canonical_u64(rng.gen_range(0..=(1 << 63) - 1));
                trace[i].b = trace[i].a;

                trace[i].c = F::from_canonical_usize(i);
                trace[i].d = trace[i].c;
            }

            let air_instance = AirInstance::new_from_trace(FromTrace::new(&mut trace));
            pctx.add_air_instance(air_instance, instance_ids[0]);
        }
    }
}
