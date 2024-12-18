use std::sync::Arc;

use proofman::{WitnessComponent, WitnessManager};
use proofman_common::{AirInstance, ExecutionCtx, ProofCtx, SetupCtx};

use p3_field::PrimeField;
use rand::{distributions::Standard, prelude::Distribution, Rng};

use crate::{BothBusesTrace, BUSES_AIRGROUP_ID, BOTH_BUSES_AIR_IDS};

pub struct BothBuses<F: PrimeField> {
    _phantom: std::marker::PhantomData<F>,
}

impl<F: PrimeField> BothBuses<F>
where
    Standard: Distribution<F>,
{
    const MY_NAME: &'static str = "BothBuses";

    pub fn new(wcm: Arc<WitnessManager<F>>) -> Arc<Self> {
        let both_buses = Arc::new(Self { _phantom: std::marker::PhantomData });

        wcm.register_component(both_buses.clone(), Some(BUSES_AIRGROUP_ID), Some(BOTH_BUSES_AIR_IDS));

        both_buses
    }

    pub fn execute(&self, pctx: Arc<ProofCtx<F>>, ectx: Arc<ExecutionCtx>, sctx: Arc<SetupCtx>) {
        let num_rows = pctx.global_info.airs[BUSES_AIRGROUP_ID][BOTH_BUSES_AIR_IDS[0]].num_rows;
        let trace = BothBusesTrace::new(num_rows);

        let air_instance =
            AirInstance::new(sctx.clone(), BUSES_AIRGROUP_ID, BOTH_BUSES_AIR_IDS[0], None, trace.buffer.unwrap());
        let (is_myne, gid) = ectx.dctx.write().unwrap().add_instance(BUSES_AIRGROUP_ID, BOTH_BUSES_AIR_IDS[0], 1);
        if is_myne {
            pctx.air_instance_repo.add_air_instance(air_instance, Some(gid));
        }
    }
}

impl<F: PrimeField> WitnessComponent<F> for BothBuses<F>
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
        log::debug!("{}: ··· Witness computation for AIR '{}' at stage {}", Self::MY_NAME, "BothBuses", stage);

        if stage == 1 {
            let air_instances_vec = &mut pctx.air_instance_repo.air_instances.write().unwrap();
            let air_instance = &mut air_instances_vec[air_instance_id.unwrap()];
            let buffer = &mut air_instance.trace;
            let num_rows = pctx.pilout.get_air(BUSES_AIRGROUP_ID, BOTH_BUSES_AIR_IDS[0]).num_rows();

            let mut trace = BothBusesTrace::map_buffer(buffer.as_mut_slice(), num_rows, 0).unwrap();

            for i in 0..num_rows {
                trace[i].a = F::from_canonical_u64(rng.gen_range(0..=(1 << 63) - 1));
                trace[i].b = trace[i].a;

                trace[i].c = F::from_canonical_usize(i);
                trace[i].d = trace[i].c;
            }
        }
    }
}
