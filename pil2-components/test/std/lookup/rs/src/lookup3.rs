use std::sync::Arc;

use witness::{WitnessComponent, execute, define_wc};
use proofman_common::{FromTrace, AirInstance, ProofCtx, SetupCtx};

use rand::{distributions::Standard, prelude::Distribution};

use p3_field::PrimeField;

use crate::Lookup3Trace;

define_wc!(Lookup3, "Lkup3");

impl<F: PrimeField + Copy> WitnessComponent<F> for Lookup3
where
    Standard: Distribution<F>,
{
    execute!(Lookup3Trace, 1);

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
