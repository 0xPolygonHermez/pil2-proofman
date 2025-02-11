use std::sync::{Arc, RwLock};

use witness::WitnessComponent;
use proofman_common::{FromTrace, AirInstance, ProofCtx, SetupCtx};

use p3_field::PrimeField;

use crate::Permutation2_6Trace;

pub struct Permutation2 {
    instance_ids: RwLock<Vec<usize>>,
}

impl Permutation2 {
    const MY_NAME: &'static str = "Perm2   ";

    pub fn new() -> Arc<Self> {
        Arc::new(Self { instance_ids: RwLock::new(Vec::new()) })
    }
}

impl<F: PrimeField + Copy> WitnessComponent<F> for Permutation2 {
    fn execute(&self, pctx: Arc<ProofCtx<F>>) -> Vec<usize> {
        let global_ids =
            vec![pctx.add_instance(Permutation2_6Trace::<usize>::AIRGROUP_ID, Permutation2_6Trace::<usize>::AIR_ID)];
        self.instance_ids.write().unwrap().push(global_ids[0]);
        global_ids
    }

    fn calculate_witness(&self, stage: u32, pctx: Arc<ProofCtx<F>>, _sctx: Arc<SetupCtx<F>>, instance_ids: &[usize]) {
        if stage == 1 {
            let mut trace = Permutation2_6Trace::new();
            let num_rows = trace.num_rows();

            log::debug!("{} ··· Starting witness computation stage {}", Self::MY_NAME, 1);

            // Note: Here it is assumed that num_rows of permutation2 is equal to
            //       the sum of num_rows of each variant of permutation1.
            //       Ohterwise, the permutation check cannot be satisfied.
            // Proves
            for i in 0..num_rows {
                trace[i].c1 = F::from_canonical_u8(200);
                trace[i].d1 = F::from_canonical_u8(201);

                trace[i].c2 = F::from_canonical_u8(100);
                trace[i].d2 = F::from_canonical_u8(101);

                trace[i].sel = F::from_bool(true);
            }

            let air_instance = AirInstance::new_from_trace(FromTrace::new(&mut trace));
            pctx.add_air_instance(air_instance, instance_ids[0]);
        }
    }
}
