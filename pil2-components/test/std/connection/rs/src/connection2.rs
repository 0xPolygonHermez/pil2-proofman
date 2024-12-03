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
        let num_rows = pctx.global_info.airs[CONNECTION_AIRGROUP_ID][CONNECTION_2_AIR_IDS[0]].num_rows;
        let trace = Connection2Trace::new(num_rows);

        let air_instance =
            AirInstance::new(sctx.clone(), CONNECTION_AIRGROUP_ID, CONNECTION_2_AIR_IDS[0], None, trace.buffer.unwrap());
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
        stage: u32,
        air_instance_id: Option<usize>,
        pctx: Arc<ProofCtx<F>>,
        _ectx: Arc<ExecutionCtx>,
        _sctx: Arc<SetupCtx>,
    ) {
        let mut rng = rand::thread_rng();

        let air_instances_vec = &mut pctx.air_instance_repo.air_instances.write().unwrap();
        let air_instance = &mut air_instances_vec[air_instance_id.unwrap()];
        let air = pctx.pilout.get_air(air_instance.airgroup_id, air_instance.air_id);

        log::debug!(
            "{}: ··· Witness computation for AIR '{}' at stage {}",
            Self::MY_NAME,
            air.name().unwrap_or("unknown"),
            stage
        );

        if stage == 1 {
            let buffer = &mut air_instance.trace;

            let num_rows = pctx.pilout.get_air(CONNECTION_AIRGROUP_ID, CONNECTION_2_AIR_IDS[0]).num_rows();
            let mut trace = Connection2Trace::map_buffer(buffer.as_mut_slice(), num_rows, 0).unwrap();

            for i in 0..num_rows {
                trace[i].a = rng.gen();
                trace[i].b = rng.gen();
                trace[i].c = rng.gen();
            }

            trace[0].a = trace[1].a;
        }
    }
}
