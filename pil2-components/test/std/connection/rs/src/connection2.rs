use std::sync::Arc;

use proofman::{WitnessComponent, WitnessManager};
use proofman_common::{AirInstance, ExecutionCtx, ProofCtx, SetupCtx};

use p3_field::PrimeField;
use rand::{distributions::Standard, prelude::Distribution, Rng};

use crate::{Connection2Trace, CONNECTION_2_AIR_IDS, CONNECTION_AIRGROUP_ID};

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

        wcm.register_component(connection2.clone(), Some(CONNECTION_AIRGROUP_ID), Some(CONNECTION_2_AIR_IDS));

        connection2
    }

    pub fn execute(&self, pctx: Arc<ProofCtx<F>>, ectx: Arc<ExecutionCtx>, sctx: Arc<SetupCtx>) {
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

        AirInstance::from_trace(pctx.clone(), ectx.clone(), sctx.clone(), None, &mut trace, None, None);
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
