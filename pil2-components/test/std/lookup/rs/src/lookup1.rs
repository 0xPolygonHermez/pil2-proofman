use std::sync::{Arc, RwLock};

use witness::WitnessComponent;
use proofman_common::{FromTrace, AirInstance, ProofCtx, SetupCtx};

use p3_field::PrimeField;
use rand::{distributions::Standard, prelude::Distribution, Rng, SeedableRng, rngs::StdRng};

use crate::Lookup1Trace;

pub struct Lookup1 {
    instance_ids: RwLock<Vec<usize>>,
}

impl Lookup1 {
    const MY_NAME: &'static str = "Lookup_1";

    pub fn new() -> Arc<Self> {
        Arc::new(Self { instance_ids: RwLock::new(Vec::new()) })
    }
}

impl<F: PrimeField + Copy> WitnessComponent<F> for Lookup1
where
    Standard: Distribution<F>,
{
    fn execute(&self, pctx: Arc<ProofCtx<F>>) -> Vec<usize> {
        let global_ids = vec![pctx.add_instance(Lookup1Trace::<usize>::AIRGROUP_ID, Lookup1Trace::<usize>::AIR_ID)];
        self.instance_ids.write().unwrap().push(global_ids[0]);
        global_ids
    }

    fn calculate_witness(&self, stage: u32, pctx: Arc<ProofCtx<F>>, _sctx: Arc<SetupCtx<F>>, instance_ids: &[usize]) {
        if stage == 1 {
            let seed = if cfg!(feature = "debug") { 0 } else { rand::thread_rng().gen::<u64>() };
            let mut rng = StdRng::seed_from_u64(seed);

            let mut trace = Lookup1Trace::new();
            let num_rows = trace.num_rows();

            log::debug!("{} ··· Starting witness computation stage {}", Self::MY_NAME, 1);

            let num_lookups = trace[0].sel.len();

            for i in 0..num_rows {
                let val = rng.gen();
                let mut n_sel = 0;
                for j in 0..num_lookups {
                    trace[i].f[j] = val;
                    let selected = rng.gen::<bool>();
                    trace[i].sel[j] = F::from_bool(selected);
                    if selected {
                        n_sel += 1;
                    }
                }
                trace[i].t = val;
                trace[i].mul = F::from_canonical_usize(n_sel);
            }

            let air_instance = AirInstance::new_from_trace(FromTrace::new(&mut trace));
            pctx.add_air_instance(air_instance, instance_ids[0]);
        }
    }
}
