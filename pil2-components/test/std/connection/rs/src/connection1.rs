use std::sync::{Arc, RwLock};

use witness::WitnessComponent;
use proofman_common::{FromTrace, AirInstance, ProofCtx, SetupCtx};

use p3_field::PrimeField;
use rand::{distributions::Standard, prelude::Distribution, Rng, SeedableRng, rngs::StdRng};

use crate::Connection1Trace;

pub struct Connection1 {
    instance_ids: RwLock<Vec<usize>>,
}

impl Connection1 {
    const MY_NAME: &'static str = "Connct_1";

    pub fn new() -> Arc<Self> {
        Arc::new(Self { instance_ids: RwLock::new(Vec::new()) })
    }
}

impl<F: PrimeField + Copy> WitnessComponent<F> for Connection1
where
    Standard: Distribution<F>,
{
    fn execute(&self, pctx: Arc<ProofCtx<F>>) -> Vec<usize> {
        let global_ids =
            vec![pctx.add_instance(Connection1Trace::<usize>::AIRGROUP_ID, Connection1Trace::<usize>::AIR_ID)];
        self.instance_ids.write().unwrap().push(global_ids[0]);
        global_ids
    }

    fn calculate_witness(&self, stage: u32, pctx: Arc<ProofCtx<F>>, _sctx: Arc<SetupCtx<F>>, instance_ids: &[usize]) {
        if stage == 1 {
            let seed = if cfg!(feature = "debug") { 0 } else { rand::thread_rng().gen::<u64>() };
            let mut rng = StdRng::seed_from_u64(seed);

            let mut trace = Connection1Trace::new();
            let num_rows = trace.num_rows();

            log::debug!("{} ··· Starting witness computation stage {}", Self::MY_NAME, 1);

            for i in 0..num_rows {
                trace[i].a = rng.gen();
                trace[i].b = rng.gen();
                trace[i].c = rng.gen();
            }

            let air_instance = AirInstance::new_from_trace(FromTrace::new(&mut trace));
            pctx.add_air_instance(air_instance, instance_ids[0]);
        }
    }
}
