use std::sync::{Arc, RwLock};

use witness::WitnessComponent;
use proofman_common::{FromTrace, AirInstance, ProofCtx, SetupCtx};

use p3_field::PrimeField;
use rand::{distributions::Standard, prelude::Distribution, Rng, SeedableRng, rngs::StdRng};

use crate::Lookup2_15Trace;

pub struct Lookup2_15 {
    instance_ids: RwLock<Vec<usize>>,
}

impl Lookup2_15 {
    const MY_NAME: &'static str = "Lkup2_15";

    pub fn new() -> Arc<Self> {
        Arc::new(Self { instance_ids: RwLock::new(Vec::new()) })
    }
}

impl<F: PrimeField + Copy> WitnessComponent<F> for Lookup2_15
where
    Standard: Distribution<F>,
{
    fn execute(&self, pctx: Arc<ProofCtx<F>>) -> Vec<usize> {
        let global_ids =
            vec![pctx.add_instance(Lookup2_15Trace::<usize>::AIRGROUP_ID, Lookup2_15Trace::<usize>::AIR_ID)];
        self.instance_ids.write().unwrap().push(global_ids[0]);
        global_ids
    }

    fn calculate_witness(&self, stage: u32, pctx: Arc<ProofCtx<F>>, _sctx: Arc<SetupCtx<F>>, instance_ids: &[usize]) {
        if stage == 1 {
            let seed = if cfg!(feature = "debug") { 0 } else { rand::thread_rng().gen::<u64>() };
            let mut rng = StdRng::seed_from_u64(seed);

            let mut trace = Lookup2_15Trace::new();
            let num_rows = trace.num_rows();

            log::debug!("{} ··· Starting witness computation stage {}", Self::MY_NAME, 1);

            // TODO: Add the ability to send inputs to lookup3
            //       and consequently add random selectors

            for i in 0..num_rows {
                // Inner lookups
                trace[i].a1 = rng.gen();
                trace[i].b1 = rng.gen();
                trace[i].c1 = trace[i].a1;
                trace[i].d1 = trace[i].b1;

                trace[i].a3 = rng.gen();
                trace[i].b3 = rng.gen();
                trace[i].c2 = trace[i].a3;
                trace[i].d2 = trace[i].b3;
                let selected = rng.gen::<bool>();
                trace[i].sel1 = F::from_bool(selected);
                if selected {
                    trace[i].mul = trace[i].sel1;
                } else {
                    trace[i].mul = F::zero();
                }

                // Outer lookups
                trace[i].a2 = F::from_canonical_usize(i % (1 << 14));
                trace[i].b2 = F::from_canonical_usize(i % (1 << 14));

                trace[i].a4 = F::from_canonical_usize(i % (1 << 14));
                trace[i].b4 = F::from_canonical_usize(i % (1 << 14));
                trace[i].sel2 = F::from_bool(true);
            }

            let air_instance = AirInstance::new_from_trace(FromTrace::new(&mut trace));
            pctx.add_air_instance(air_instance, instance_ids[0]);
        }
    }
}
