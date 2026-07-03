use std::sync::{Arc, RwLock};

use fields::PrimeField64;
use proofman_common::{AirInstance, BufferPool, FromTrace, ProofCtx, ProofmanResult, SetupCtx};
use witness::WitnessComponent;

use crate::{FibonacciMLTrace, FibonacciPublicValues};

pub struct FibonacciML {
    instance_ids: RwLock<Vec<usize>>,
}

impl FibonacciML {
    pub fn new() -> Arc<Self> {
        Arc::new(Self { instance_ids: RwLock::new(Vec::new()) })
    }
}

impl<F: PrimeField64> WitnessComponent<F> for FibonacciML {
    fn execute(
        &self,
        pctx: Arc<ProofCtx<F>>,
        _sctx: Arc<SetupCtx<F>>,
        global_ids: &RwLock<Vec<usize>>,
    ) -> ProofmanResult<()> {
        let global_id = pctx.add_instance(FibonacciMLTrace::<F>::AIRGROUP_ID, FibonacciMLTrace::<F>::AIR_ID)?;
        *self.instance_ids.write().unwrap() = vec![global_id];
        global_ids.write().unwrap().push(global_id);
        Ok(())
    }

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
            let instance_id = instance_ids[0];
            tracing::debug!("··· Starting FibonacciML witness computation stage 1");

            let publics = FibonacciPublicValues::from_vec_guard(pctx.get_publics());
            let mut trace = FibonacciMLTrace::new_from_vec_zeroes(buffer_pool.take_buffer())?;

            trace[0].a = publics.in1;
            trace[0].b = publics.in2;
            for i in 1..trace.num_rows() {
                trace[i].a = trace[i - 1].b;
                trace[i].b = trace[i - 1].a + trace[i - 1].b;
            }

            let air_instance = AirInstance::new_from_trace(FromTrace::new(&mut trace));
            pctx.add_air_instance(air_instance, instance_id);
        }
        Ok(())
    }
}
