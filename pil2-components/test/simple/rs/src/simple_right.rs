use std::sync::Arc;

use witness::{WitnessComponent, execute, define_wc};
use proofman_common::{BufferPool, FromTrace, AirInstance, ProofCtx, SetupCtx};

use fields::PrimeField64;

use crate::SimpleRightTrace;

define_wc!(SimpleRight, "SimRight");

impl<F: PrimeField64> WitnessComponent<F> for SimpleRight {
    execute!(SimpleRightTrace, 1);

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
            let mut trace = SimpleRightTrace::new_from_vec(buffer_pool.take_buffer());
            let num_rows = trace.num_rows();

            tracing::debug!("··· Starting witness computation stage {}", 1);

            // Proves
            for i in 0..num_rows {
                trace[i].a = F::from_u8(200);
                trace[i].b = F::from_u8(201);

                trace[i].c = F::from_usize(i);
                trace[i].d = F::from_usize(num_rows - i - 1);

                trace[i].mul = F::from_usize(1);
            }

            let air_instance = AirInstance::new_from_trace(FromTrace::new(&mut trace));
            pctx.add_air_instance(air_instance, instance_ids[0]);
        }
    }
}
