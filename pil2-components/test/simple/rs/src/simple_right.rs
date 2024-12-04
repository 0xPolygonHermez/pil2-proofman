use std::sync::Arc;

use proofman::{WitnessComponent, WitnessManager};
use proofman_common::{AirInstance, ExecutionCtx, ProofCtx, SetupCtx};

use p3_field::PrimeField;
use rand::{distributions::Standard, prelude::Distribution};

use crate::{SimpleRightTrace, SIMPLE_AIRGROUP_ID, SIMPLE_RIGHT_AIR_IDS};

pub struct SimpleRight<F> {
    _phantom: std::marker::PhantomData<F>,
}

impl<F: PrimeField + Copy> SimpleRight<F>
where
    Standard: Distribution<F>,
{
    const MY_NAME: &'static str = "SimRight";

    pub fn new(wcm: Arc<WitnessManager<F>>) -> Arc<Self> {
        let simple_right = Arc::new(Self { _phantom: std::marker::PhantomData });

        wcm.register_component(simple_right.clone(), Some(SIMPLE_AIRGROUP_ID), Some(SIMPLE_RIGHT_AIR_IDS));

        simple_right
    }

    pub fn execute(&self, pctx: Arc<ProofCtx<F>>, ectx: Arc<ExecutionCtx>, sctx: Arc<SetupCtx>) {
        let num_rows = pctx.global_info.airs[SIMPLE_AIRGROUP_ID][SIMPLE_RIGHT_AIR_IDS[0]].num_rows;
        let air = pctx.pilout.get_air(SIMPLE_AIRGROUP_ID, SIMPLE_RIGHT_AIR_IDS[0]);
        let mut trace = SimpleRightTrace::new(num_rows);

        log::debug!(
            "{}: ··· Computing witness computation for AIR '{}' at stage 1",
            Self::MY_NAME,
            air.name().unwrap_or("unknown"),
        );

        // Proves
        for i in 0..num_rows {
            trace[i].a = F::from_canonical_u8(200);
            trace[i].b = F::from_canonical_u8(201);

            trace[i].c = F::from_canonical_usize(i);
            trace[i].d = F::from_canonical_usize(num_rows - i - 1);

            trace[i].mul = F::from_canonical_usize(1);
        }

        let air_instance =
            AirInstance::new(sctx.clone(), SIMPLE_AIRGROUP_ID, SIMPLE_RIGHT_AIR_IDS[0], None, trace.buffer.unwrap());
        let (is_myne, gid) = ectx.dctx.write().unwrap().add_instance(SIMPLE_AIRGROUP_ID, SIMPLE_RIGHT_AIR_IDS[0], 1);
        if is_myne {
            pctx.air_instance_repo.add_air_instance(air_instance, Some(gid));
        }
    }
}

impl<F: PrimeField + Copy> WitnessComponent<F> for SimpleRight<F>
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
