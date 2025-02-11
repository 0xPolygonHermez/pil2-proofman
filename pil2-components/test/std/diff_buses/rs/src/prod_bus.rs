use std::sync::Arc;

use witness::WitnessComponent;
use proofman_common::{add_air_instance, FromTrace, AirInstance, ProofCtx};

use p3_field::PrimeField;
use rand::{distributions::Standard, prelude::Distribution, Rng, SeedableRng, rngs::StdRng};

use crate::ProdBusTrace;

pub struct ProdBus;

impl ProdBus {
    const MY_NAME: &'static str = "ProdBus ";

    pub fn new() -> Arc<Self> {
        Arc::new(Self)
    }
}

impl<F: PrimeField> WitnessComponent<F> for ProdBus
where
    Standard: Distribution<F>,
{
    fn execute(&self, pctx: Arc<ProofCtx<F>>) {
        let seed = if cfg!(feature = "debug") {
            0
        } else {
            rand::thread_rng().gen::<u64>()
        };
        let mut rng = StdRng::seed_from_u64(seed);

        let mut trace = ProdBusTrace::new();
        let num_rows = trace.num_rows();

        log::debug!("{}: ··· Starting witness computation stage {}", Self::MY_NAME, 1);

        for i in 0..num_rows {
            trace[i].a = F::from_canonical_u64(rng.gen_range(0..=(1 << 63) - 1));
            trace[i].b = trace[i].a;
        }

        let air_instance = AirInstance::new_from_trace(FromTrace::new(&mut trace));
        add_air_instance::<F>(air_instance, pctx.clone());
    }
}
