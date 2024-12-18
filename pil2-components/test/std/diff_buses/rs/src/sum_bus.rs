use std::sync::Arc;

use proofman::{WitnessComponent, WitnessManager};
use proofman_common::{AirInstance, ExecutionCtx, ProofCtx, SetupCtx};

use p3_field::PrimeField;

use crate::{SumBusTrace, BUSES_AIRGROUP_ID, SUM_BUS_AIR_IDS};

pub struct SumBus<F: PrimeField> {
    _phantom: std::marker::PhantomData<F>,
}

impl<F: PrimeField> SumBus<F> {
    const MY_NAME: &'static str = "SumBus";

    pub fn new(wcm: Arc<WitnessManager<F>>) -> Arc<Self> {
        let sum_bus = Arc::new(Self { _phantom: std::marker::PhantomData });

        wcm.register_component(sum_bus.clone(), Some(BUSES_AIRGROUP_ID), Some(SUM_BUS_AIR_IDS));

        sum_bus
    }

    pub fn execute(&self, pctx: Arc<ProofCtx<F>>, ectx: Arc<ExecutionCtx>, sctx: Arc<SetupCtx>) {
        let num_rows = pctx.global_info.airs[BUSES_AIRGROUP_ID][SUM_BUS_AIR_IDS[0]].num_rows;
        let trace = SumBusTrace::new(num_rows);

        let air_instance =
            AirInstance::new(sctx.clone(), BUSES_AIRGROUP_ID, SUM_BUS_AIR_IDS[0], None, trace.buffer.unwrap());
        let (is_myne, gid) = ectx.dctx.write().unwrap().add_instance(BUSES_AIRGROUP_ID, SUM_BUS_AIR_IDS[0], 1);
        if is_myne {
            pctx.air_instance_repo.add_air_instance(air_instance, Some(gid));
        }
    }
}

impl<F: PrimeField> WitnessComponent<F> for SumBus<F> {
    fn calculate_witness(
        &self,
        stage: u32,
        air_instance_id: Option<usize>,
        pctx: Arc<ProofCtx<F>>,
        _ectx: Arc<ExecutionCtx>,
        _sctx: Arc<SetupCtx>,
    ) {
        log::debug!("{}: ··· Witness computation for AIR '{}' at stage {}", Self::MY_NAME, "SumBus", stage);

        if stage == 1 {
            let air_instances_vec = &mut pctx.air_instance_repo.air_instances.write().unwrap();
            let air_instance = &mut air_instances_vec[air_instance_id.unwrap()];
            let buffer = &mut air_instance.trace;
            let num_rows = pctx.pilout.get_air(BUSES_AIRGROUP_ID, SUM_BUS_AIR_IDS[0]).num_rows();

            let mut trace = SumBusTrace::map_buffer(buffer.as_mut_slice(), num_rows, 0).unwrap();

            for i in 0..num_rows {
                trace[i].a = F::from_canonical_usize(i);
                trace[i].b = trace[i].a;
            }
        }
    }
}
