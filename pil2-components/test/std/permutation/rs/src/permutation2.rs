use std::sync::Arc;

use witness::{WitnessComponent, execute, define_wc};
use proofman_common::{BufferPool, FromTrace, AirInstance, ProofCtx, SetupCtx};

use fields::PrimeField64;

use crate::Permutation2_6Trace;

define_wc!(Permutation2, "Perm2   ");

impl<F: PrimeField64> WitnessComponent<F> for Permutation2 {
    execute!(Permutation2_6Trace, 1);

    fn calculate_witness(
        &self,
        stage: u32,
        pctx: Arc<ProofCtx<F>>,
        _sctx: Arc<SetupCtx<F>>,
        instance_ids: &[usize],
        _n_cores: usize,
        buffer_pool: &dyn BufferPool<F>,
    ) {
        if stage == 1 {
            let mut trace = Permutation2_6Trace::new_from_vec(buffer_pool.take_buffer());
            let num_rows = trace.num_rows();

            tracing::debug!("··· Starting witness computation stage {}", 1);

            // Note: Here it is assumed that num_rows of permutation2 is equal to
            //       the sum of num_rows of each variant of permutation1.
            //       Ohterwise, the permutation check cannot be satisfied.
            // Proves
            for i in 0..num_rows {
                trace[i].c1 = F::from_u8(200);
                trace[i].d1 = F::from_u8(201);

                trace[i].c2 = F::from_u8(100);
                trace[i].d2 = F::from_u8(101);

                trace[i].sel = F::from_bool(true);
            }

            let air_instance = AirInstance::new_from_trace(FromTrace::new(&mut trace));
            pctx.add_air_instance(air_instance, instance_ids[0]);
        }
    }
}
