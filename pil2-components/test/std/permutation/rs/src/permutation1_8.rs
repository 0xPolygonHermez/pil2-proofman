use std::sync::{Arc, RwLock};

use witness::WitnessComponent;
use proofman_common::{FromTrace, AirInstance, ProofCtx, SetupCtx};

use p3_field::PrimeField;
use rand::{distributions::Standard, prelude::Distribution, Rng, SeedableRng, rngs::StdRng};

use crate::Permutation1_8Trace;

pub struct Permutation1_8 {
    instance_ids: RwLock<Vec<usize>>,
}

impl Permutation1_8 {
    const MY_NAME: &'static str = "Perm1_8 ";

    pub fn new() -> Arc<Self> {
        Arc::new(Self { instance_ids: RwLock::new(Vec::new()) })
    }
}

impl<F: PrimeField + Copy> WitnessComponent<F> for Permutation1_8
where
    Standard: Distribution<F>,
{
    fn execute(&self, pctx: Arc<ProofCtx<F>>) -> Vec<usize> {
        let global_ids =
            vec![pctx.add_instance(Permutation1_8Trace::<usize>::AIRGROUP_ID, Permutation1_8Trace::<usize>::AIR_ID)];
        self.instance_ids.write().unwrap().push(global_ids[0]);
        global_ids
    }

    fn calculate_witness(&self, stage: u32, pctx: Arc<ProofCtx<F>>, _sctx: Arc<SetupCtx<F>>, instance_ids: &[usize]) {
        if stage == 1 {
            let seed = if cfg!(feature = "debug") { 0 } else { rand::thread_rng().gen::<u64>() };
            let mut rng = StdRng::seed_from_u64(seed);
            let mut trace = Permutation1_8Trace::new();
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

                trace[i].sel1 = F::one();
                trace[i].sel3 = F::one(); // F::from_canonical_u8(rng.gen_range(0..=1));
            }

            // TODO: Add the permutation of indexes

            // Proves
            for i in 0..num_rows {
                let index = num_rows - i - 1;
                // let mut index = rng.gen_range(0..num_rows);
                trace[i].c1 = trace[index].a1;
                trace[i].d1 = trace[index].b1;

                // index = rng.gen_range(0..num_rows);
                trace[i].c2 = trace[index].a3;
                trace[i].d2 = trace[index].b3;

                trace[i].sel2 = trace[i].sel1;
            }

            let air_instance = AirInstance::new_from_trace(FromTrace::new(&mut trace));
            pctx.add_air_instance(air_instance, instance_ids[0]);
        }
    }
}
