use std::sync::{Arc, RwLock};

use witness::WitnessComponent;
use proofman_common::{FromTrace, AirInstance, ProofCtx, SetupCtx};

use rand::{distributions::Standard, prelude::Distribution};

use p3_field::PrimeField;

use crate::Lookup3Trace;

pub struct Lookup3 {
    instance_ids: RwLock<Vec<usize>>,
}

impl Lookup3 {
    const MY_NAME: &'static str = "Lookup_3";

    pub fn new() -> Arc<Self> {
        Arc::new(Self { instance_ids: RwLock::new(Vec::new()) })
    }
}

impl<F: PrimeField + Copy> WitnessComponent<F> for Lookup3
where
    Standard: Distribution<F>,
{
    fn execute(&self, pctx: Arc<ProofCtx<F>>) -> Vec<usize> {
        let global_ids = vec![pctx.add_instance(Lookup3Trace::<usize>::AIRGROUP_ID, Lookup3Trace::<usize>::AIR_ID)];
        self.instance_ids.write().unwrap().push(global_ids[0]);
        global_ids
    }

    fn calculate_witness(&self, stage: u32, pctx: Arc<ProofCtx<F>>, _sctx: Arc<SetupCtx<F>>, instance_ids: &[usize]) {
        if stage == 1 {
            // For simplicity, add a single instance of each air
            let mut trace = Lookup3Trace::new();
            let num_rows = trace.num_rows();

            log::debug!("{} ··· Starting witness computation stage {}", Self::MY_NAME, 1);

            for i in 0..num_rows {
                trace[i].c1 = F::from_canonical_usize(i);
                trace[i].d1 = F::from_canonical_usize(i);
                if i < (1 << 12) {
                    trace[i].mul1 = F::from_canonical_usize(4);
                } else if i < (1 << 13) {
                    trace[i].mul1 = F::from_canonical_usize(3);
                } else {
                    trace[i].mul1 = F::from_canonical_usize(2);
                }

                trace[i].c2 = F::from_canonical_usize(i);
                trace[i].d2 = F::from_canonical_usize(i);
                if i < (1 << 12) {
                    trace[i].mul2 = F::from_canonical_usize(4);
                } else if i < (1 << 13) {
                    trace[i].mul2 = F::from_canonical_usize(3);
                } else {
                    trace[i].mul2 = F::from_canonical_usize(2);
                }
            }

            let air_instance = AirInstance::new_from_trace(FromTrace::new(&mut trace));
            pctx.add_air_instance(air_instance, instance_ids[0]);
        }
    }
}
