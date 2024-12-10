use std::sync::Arc;

use proofman::{WitnessComponent, WitnessManager};
use proofman_common::{add_air_instance, FromTrace, AirInstance, ExecutionCtx, ProofCtx, SetupCtx};

use p3_field::PrimeField;
use rand::{distributions::Standard, prelude::Distribution};

use crate::SimpleRightTrace;

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

        let airgroup_id = SimpleRightTrace::<F>::AIRGROUP_ID;
        let air_id = SimpleRightTrace::<F>::AIR_ID;

        wcm.register_component(simple_right.clone(), airgroup_id, air_id);

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

        let air_instance = AirInstance::new_from_trace(sctx.clone(), FromTrace::new(&mut trace));
        add_air_instance::<F>(air_instance, ectx, pctx.clone());
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
