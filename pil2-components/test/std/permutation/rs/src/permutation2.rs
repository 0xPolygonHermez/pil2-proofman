use std::sync::Arc;

use proofman::{WitnessComponent, WitnessManager};
use proofman_common::{AirInstance, ExecutionCtx, ProofCtx, SetupCtx};

use p3_field::PrimeField;

use crate::Permutation2_6Trace;

pub struct Permutation2<F> {
    _phantom: std::marker::PhantomData<F>,
}

impl<F: PrimeField + Copy> Permutation2<F> {
    const MY_NAME: &'static str = "Perm2   ";

    pub fn new(wcm: Arc<WitnessManager<F>>) -> Arc<Self> {
        let permutation2 = Arc::new(Self { _phantom: std::marker::PhantomData });
        
        let airgroup_id = Permutation2_6Trace::<F>::get_airgroup_id();
        let air_id = Permutation2_6Trace::<F>::get_air_id();

        wcm.register_component(permutation2.clone(), airgroup_id, air_id);

        permutation2
    }

    pub fn execute(&self, pctx: Arc<ProofCtx<F>>, ectx: Arc<ExecutionCtx>, sctx: Arc<SetupCtx>) {
        let mut trace = Permutation2_6Trace::new();
        let num_rows = trace.num_rows();

        log::debug!("{} ··· Starting witness computation stage {}", Self::MY_NAME, 1);

        // Note: Here it is assumed that num_rows of permutation2 is equal to
        //       the sum of num_rows of each variant of permutation1.
        //       Ohterwise, the permutation check cannot be satisfied.
        // Proves
        for i in 0..num_rows {
            trace[i].c1 = F::from_canonical_u8(200);
            trace[i].d1 = F::from_canonical_u8(201);

            trace[i].c2 = F::from_canonical_u8(100);
            trace[i].d2 = F::from_canonical_u8(101);

            trace[i].sel = F::from_bool(true);
        }

        AirInstance::from_trace(pctx.clone(), ectx.clone(), sctx.clone(), None, &mut trace, None, None);
    }
}

impl<F: PrimeField + Copy> WitnessComponent<F> for Permutation2<F> {
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
