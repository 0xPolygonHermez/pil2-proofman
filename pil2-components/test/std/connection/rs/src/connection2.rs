use std::sync::Arc;

use proofman::{WitnessComponent, WitnessManager};
use proofman_common::{add_air_instance, FromTrace, AirInstance, ExecutionCtx, ProofCtx, SetupCtx};

use p3_field::PrimeField;
use rand::{distributions::Standard, prelude::Distribution, Rng};

use crate::Connection2Trace;

pub struct Connection2<F> {
    _phantom: std::marker::PhantomData<F>,
}

impl<F: PrimeField + Copy> Connection2<F>
where
    Standard: Distribution<F>,
{
    const MY_NAME: &'static str = "Connct_2";

    pub fn new(wcm: Arc<WitnessManager<F>>) -> Arc<Self> {
        let connection2 = Arc::new(Self { _phantom: std::marker::PhantomData });

        let airgroup_id = Connection2Trace::<F>::AIRGROUP_ID;
        let air_id = Connection2Trace::<F>::AIR_ID;

        wcm.register_component(connection2.clone(), airgroup_id, air_id);

        connection2
    }

    pub fn execute(&self, pctx: Arc<ProofCtx<F>>, ectx: Arc<ExecutionCtx>, _sctx: Arc<SetupCtx>) {
        let mut rng = rand::thread_rng();

        let mut trace = Connection2Trace::new_zeroes();
        let num_rows = trace.num_rows();

        log::debug!("{} ··· Starting witness computation stage {}", Self::MY_NAME, 1);

        for i in 0..num_rows {
            trace[i].a = rng.gen();
            trace[i].b = rng.gen();
            trace[i].c = rng.gen();
        }

        trace[0].a = trace[1].a;

        let air_instance = AirInstance::new_from_trace( FromTrace::new(&mut trace));
        add_air_instance::<F>(air_instance, ectx, pctx.clone());
    }
}

impl<F: PrimeField + Copy> WitnessComponent<F> for Connection2<F>
where
    Standard: Distribution<F>,
{
    fn calculate_witness(
        &self,
        _stage: u32,
        _air_instance_id: Option<usize>,
        _pctx: Arc<ProofCtx<F>>,
        _ectx: Arc<ExecutionCtx>,
        _sctx: Arc<SetupCtx>,
    ) {
    }
}
