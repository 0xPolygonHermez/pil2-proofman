use std::sync::{Arc, RwLock};

use witness::WitnessComponent;
use proofman_common::{FromTrace, AirInstance, ProofCtx, SetupCtx};

use p3_field::PrimeField;
use rand::{distributions::Standard, prelude::Distribution, seq::SliceRandom, Rng, SeedableRng, rngs::StdRng};

use crate::Permutation1_6Trace;

pub struct Permutation1_6 {
    instance_ids: RwLock<Vec<usize>>,
}

impl Permutation1_6 {
    const MY_NAME: &'static str = "Perm1_6 ";

    pub fn new() -> Arc<Self> {
        Arc::new(Self { instance_ids: RwLock::new(Vec::new()) })
    }
}
impl<F: PrimeField + Copy> WitnessComponent<F> for Permutation1_6
where
    Standard: Distribution<F>,
{
    fn execute(&self, pctx: Arc<ProofCtx<F>>) -> Vec<usize> {
        let global_ids = vec![
            pctx.add_instance(Permutation1_6Trace::<usize>::AIRGROUP_ID, Permutation1_6Trace::<usize>::AIR_ID),
            pctx.add_instance(Permutation1_6Trace::<usize>::AIRGROUP_ID, Permutation1_6Trace::<usize>::AIR_ID),
        ];
        self.instance_ids.write().unwrap().push(global_ids[0]);
        self.instance_ids.write().unwrap().push(global_ids[1]);
        global_ids
    }

    fn calculate_witness(&self, stage: u32, pctx: Arc<ProofCtx<F>>, _sctx: Arc<SetupCtx<F>>, instance_ids: &[usize]) {
        if stage == 1 {
            let seed = if cfg!(feature = "debug") { 0 } else { rand::thread_rng().gen::<u64>() };
            let mut rng = StdRng::seed_from_u64(seed);

            let mut trace = Permutation1_6Trace::new();
            let num_rows = trace.num_rows();

            log::debug!("{} ··· Starting witness computation stage {}", Self::MY_NAME, 1);

            // TODO: Add the ability to send inputs to permutation2
            //       and consequently add random selectors

            // Assumes
            for i in 0..num_rows {
                trace[i].a1 = rng.gen();
                trace[i].b1 = rng.gen();

                trace[i].a2 = F::from_canonical_u8(200);
                trace[i].b2 = F::from_canonical_u8(201);

                trace[i].a3 = rng.gen();
                trace[i].b3 = rng.gen();

                trace[i].a4 = F::from_canonical_u8(100);
                trace[i].b4 = F::from_canonical_u8(101);

                trace[i].sel1 = F::from_bool(rng.gen_bool(0.5));
                trace[i].sel3 = F::one();
            }

            let mut indices: Vec<usize> = (0..num_rows).collect();
            indices.shuffle(&mut rng);

            // Proves
            for i in 0..num_rows {
                // We take a random permutation of the indices to show that the permutation check is passing
                trace[i].c1 = trace[indices[i]].a1;
                trace[i].d1 = trace[indices[i]].b1;

                trace[i].c2 = trace[indices[i]].a3;
                trace[i].d2 = trace[indices[i]].b3;

                trace[i].sel2 = trace[indices[i]].sel1;
            }

            let air_instance = AirInstance::new_from_trace(FromTrace::new(&mut trace));
            pctx.add_air_instance(air_instance, instance_ids[0]);
        }
    }
}
