use std::sync::Arc;

use proofman::{WitnessComponent, WitnessManager};
use proofman_common::{add_air_instance, FromTrace, AirInstance, ExecutionCtx, ProofCtx, SetupCtx};

use p3_field::PrimeField;
use rand::{distributions::Standard, prelude::Distribution, seq::SliceRandom};

use crate::SimpleLeftTrace;

pub struct SimpleLeft<F> {
    _phantom: std::marker::PhantomData<F>,
}

impl<F: PrimeField + Copy> SimpleLeft<F>
where
    Standard: Distribution<F>,
{
    const MY_NAME: &'static str = "SimLeft ";

    pub fn new(wcm: Arc<WitnessManager<F>>) -> Arc<Self> {
        let simple_left = Arc::new(Self { _phantom: std::marker::PhantomData });

        wcm.register_component(simple_left.clone(), SimpleLeftTrace::<F>::AIRGROUP_ID, SimpleLeftTrace::<F>::AIR_ID);

        simple_left
    }

    pub fn execute(&self, pctx: Arc<ProofCtx<F>>, ectx: Arc<ExecutionCtx>, sctx: Arc<SetupCtx>) {
        let mut rng = rand::thread_rng();

        let mut trace = SimpleLeftTrace::new();
        let num_rows = trace.num_rows();

        log::debug!("{} ··· Starting witness computation stage {}", Self::MY_NAME, 1);

        // Assumes
        for i in 0..num_rows {
            trace[i].a = F::from_canonical_usize(i);
            trace[i].b = F::from_canonical_usize(i);

            trace[i].e = F::from_canonical_u8(200);
            trace[i].f = F::from_canonical_u8(201);

            trace[i].g = F::from_canonical_usize(i);
            trace[i].h = F::from_canonical_usize(num_rows - i - 1);
        }

        let mut indices: Vec<usize> = (0..num_rows).collect();
        indices.shuffle(&mut rng);

        // Proves
        for i in 0..num_rows {
            // We take a random permutation of the indices to show that the permutation check is passing
            trace[i].c = trace[indices[i]].a;
            trace[i].d = trace[indices[i]].b;
        }

        let air_instance = AirInstance::new_from_trace(sctx.clone(), FromTrace::new(&mut trace));
        add_air_instance::<F>(air_instance, ectx, pctx.clone());
    }
}

impl<F: PrimeField + Copy> WitnessComponent<F> for SimpleLeft<F>
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
