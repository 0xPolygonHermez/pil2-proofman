use std::sync::Arc;

use proofman::{WitnessComponent, WitnessManager};
use proofman_common::{add_air_instance, FromTrace, AirInstance, ExecutionCtx, ProofCtx, SetupCtx};

use p3_field::PrimeField;
use rand::{distributions::Standard, prelude::Distribution, Rng};

use crate::Connection1Trace;

pub struct Connection1<F> {
    _phantom: std::marker::PhantomData<F>,
}

impl<F: PrimeField + Copy> Connection1<F>
where
    Standard: Distribution<F>,
{
    const MY_NAME: &'static str = "Connct_1";

    pub fn new(wcm: Arc<WitnessManager<F>>) -> Arc<Self> {
        let connection1 = Arc::new(Self { _phantom: std::marker::PhantomData });

        let airgroup_id = Connection1Trace::<F>::AIRGROUP_ID;
        let air_id = Connection1Trace::<F>::AIR_ID;

        wcm.register_component(connection1.clone(), airgroup_id, air_id);

        connection1
    }

    pub fn execute(&self, pctx: Arc<ProofCtx<F>>, ectx: Arc<ExecutionCtx>, sctx: Arc<SetupCtx>) {
        let mut rng = rand::thread_rng();

        let mut trace = Connection1Trace::new_zeroes();
        let num_rows = trace.num_rows();

        log::debug!("{} ··· Starting witness computation stage {}", Self::MY_NAME, 1);

        for i in 0..num_rows {
            trace[i].a = rng.gen();
            trace[i].b = rng.gen();
            trace[i].c = rng.gen();
        }

        let air_instance = AirInstance::new_from_trace(sctx.clone(), FromTrace::new(&mut trace));
        add_air_instance::<F>(air_instance, ectx, pctx.clone());
    }
}

impl<F: PrimeField + Copy> WitnessComponent<F> for Connection1<F>
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
