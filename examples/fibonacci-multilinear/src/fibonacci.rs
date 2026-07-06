use std::sync::{Arc, RwLock};

use fields::PrimeField64;
use proofman_common::{AirInstance, BufferPool, FromTrace, ProofCtx, ProofmanResult, SetupCtx};
use witness::WitnessComponent;

use crate::{FibonacciTrace, BuildPublicValues};

pub struct Fibonacci {
    instance_ids: RwLock<Vec<usize>>,
}

impl Fibonacci {
    pub fn new() -> Arc<Self> {
        Arc::new(Self { instance_ids: RwLock::new(Vec::new()) })
    }
}

impl<F: PrimeField64> WitnessComponent<F> for Fibonacci {
    fn execute(
        &self,
        pctx: Arc<ProofCtx<F>>,
        _sctx: Arc<SetupCtx<F>>,
        global_ids: &RwLock<Vec<usize>>,
    ) -> ProofmanResult<()> {
        let global_id = pctx.add_instance(FibonacciTrace::<F>::AIRGROUP_ID, FibonacciTrace::<F>::AIR_ID)?;
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
            tracing::debug!("··· Starting Fibonacci witness computation stage 1");

            let publics = BuildPublicValues::from_vec_guard(pctx.get_publics());

            let module = F::as_canonical_u64(&publics.module);
            let mut a = F::as_canonical_u64(&publics.in1);
            let mut b = F::as_canonical_u64(&publics.in2);

            let mut trace = FibonacciTrace::new_from_vec_zeroes(buffer_pool.take_buffer())?;

            trace[0].a = publics.in1;
            trace[0].b = publics.in2;
            for i in 1..trace.num_rows() {
                let tmp = b;
                let result = (a + b) % module;
                (a, b) = (tmp, result);

                trace[i].a = F::from_u64(a);
                trace[i].b = F::from_u64(b);
            }

            let air_instance = AirInstance::new_from_trace(FromTrace::new(&mut trace));
            pctx.add_air_instance(air_instance, instance_id);
        }
        Ok(())
    }
}
