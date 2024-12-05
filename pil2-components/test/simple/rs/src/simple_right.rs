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
        let mut trace = SimpleRightTrace::new();
        let num_rows = trace.num_rows();

        log::debug!("{} ··· Starting witness computation stage {}", Self::MY_NAME, 1);

        // Proves
        for i in 0..num_rows {
            trace[i].a = F::from_canonical_u8(200);
            trace[i].b = F::from_canonical_u8(201);

            trace[i].c = F::from_canonical_usize(i);
            trace[i].d = F::from_canonical_usize(num_rows - i - 1);

            trace[i].mul = F::from_canonical_usize(1);
        }

        AirInstance::from_trace(pctx.clone(), ectx.clone(), sctx.clone(), None, &mut trace, None, None);
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
