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

        let num_rows = pctx.global_info.airs[CONNECTION_AIRGROUP_ID][CONNECTION_2_AIR_IDS[0]].num_rows;
        let air = pctx.pilout.get_air(CONNECTION_AIRGROUP_ID, CONNECTION_2_AIR_IDS[0]);
        let mut trace = Connection2Trace::new_zeroes(num_rows);

        log::debug!(
            "{}: ··· Witness computation for AIR '{}' at stage 1",
            Self::MY_NAME,
            air.name().unwrap_or("unknown"),
        );

        for i in 0..num_rows {
            trace[i].a = rng.gen();
            trace[i].b = rng.gen();
            trace[i].c = rng.gen();
        }

        trace[0].a = trace[1].a;

        let air_instance = AirInstance::new(
            sctx.clone(),
            CONNECTION_AIRGROUP_ID,
            CONNECTION_2_AIR_IDS[0],
            None,
            trace.buffer.unwrap(),
        );
        let (is_myne, gid) =
            ectx.dctx.write().unwrap().add_instance(CONNECTION_AIRGROUP_ID, CONNECTION_2_AIR_IDS[0], 1);
        if is_myne {
            pctx.air_instance_repo.add_air_instance(air_instance, Some(gid));
        }
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
