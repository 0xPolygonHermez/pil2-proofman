use std::sync::Arc;

use witness::{WitnessComponent, execute, define_wc};
use proofman_common::{FromTrace, AirInstance, ProofCtx, SetupCtx};

use p3_field::PrimeField64;
use rand::{distributions::Standard, prelude::Distribution};

use crate::SimpleRightTrace;

define_wc!(SimpleRight, "SimRight");

impl<F: PrimeField64 + Copy> WitnessComponent<F> for SimpleRight
where
    Standard: Distribution<F>,
{
    execute!(SimpleRightTrace, 1);


    fn calculate_witness(&self, stage: u32, pctx: Arc<ProofCtx<F>>, _sctx: Arc<SetupCtx<F>>, instance_ids: &[usize]) {
        if stage == 1 {
            let mut trace = SimpleRightTrace::new();
            let num_rows = trace.num_rows();

            log::debug!("{} ··· Starting witness computation stage {}", Self::MY_NAME, 1);

            // Proves
            for i in 0..num_rows {
                trace[i].a = F::from_canonical_u8(200);
                trace[i].b = F::from_canonical_u8(201);

                trace[i].c = F::from_canonical_usize(i);
                trace[i].d = F::from_canonical_usize(num_rows - i - 1);

                trace[i].mul = F::from_canonical_usize(1);
            }

            let air_instance = AirInstance::new_from_trace(FromTrace::new(&mut trace));
            pctx.add_air_instance(air_instance, instance_ids[0]);
        }
    }
}
