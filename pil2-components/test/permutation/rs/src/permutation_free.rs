use std::sync::Arc;

use proofman_witness::{WitnessComponent, execute, define_wc};
use proofman_common::{BufferPool, FromTrace, AirInstance, ProofCtx, SetupCtx, ProofmanResult};

use proofman_fields::PrimeField64;

use crate::PermutationFreeTrace;

define_wc!(PermutationFree, "PermFree");

impl<F: PrimeField64> WitnessComponent<F> for PermutationFree {
    execute!(PermutationFreeTrace, 1);

    fn calculate_witness(
        &self,
        stage: u32,
        pctx: Arc<ProofCtx<F>>,
        _sctx: Arc<SetupCtx<F>>,
        instance_ids: &[usize],
        _n_cores: usize,
        buffer_pool: &dyn BufferPool<F>,
    ) -> ProofmanResult<()> {
        if stage == 1 {
            let mut trace = PermutationFreeTrace::new_from_vec(buffer_pool.take_buffer())?;
            let num_rows = trace.num_rows();
            let half = num_rows / 2;

            // sel = 1 on the first half, -1 on the second (the same pairs, reversed), 0 on every 4th row
            for i in 0..half {
                let j = num_rows - 1 - i;
                let (a, b) = (F::from_usize(3 * i + 1), F::from_usize(7 * i + 2));
                trace[i].a = a;
                trace[i].b = b;
                trace[j].a = a;
                trace[j].b = b;
                let sel = if i % 4 == 3 { F::ZERO } else { F::ONE };
                trace[i].sel = sel;
                trace[j].sel = -sel;
            }

            let air_instance = AirInstance::new_from_trace(FromTrace::new(&mut trace));
            pctx.add_air_instance(air_instance, instance_ids[0]);
        }
        Ok(())
    }
}
