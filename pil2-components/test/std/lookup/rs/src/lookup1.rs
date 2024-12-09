use std::sync::Arc;

use proofman::{WitnessComponent, WitnessManager};
use proofman_common::{AirInstance, ExecutionCtx, ProofCtx, SetupCtx};

use p3_field::PrimeField;
use rand::{distributions::Standard, prelude::Distribution, Rng};

use crate::Lookup1Trace;

pub struct Lookup1<F> {
    _phantom: std::marker::PhantomData<F>,
}

impl<F: PrimeField + Copy> Lookup1<F>
where
    Standard: Distribution<F>,
{
    const MY_NAME: &'static str = "Lookup_1";

    pub fn new(wcm: Arc<WitnessManager<F>>) -> Arc<Self> {
        let lookup1 = Arc::new(Self { _phantom: std::marker::PhantomData });

        let airgroup_id = Lookup1Trace::<F>::get_airgroup_id();
        let air_id = Lookup1Trace::<F>::get_air_id();

        wcm.register_component(lookup1.clone(), airgroup_id, air_id);

        lookup1
    }

    pub fn execute(&self, pctx: Arc<ProofCtx<F>>, ectx: Arc<ExecutionCtx>, sctx: Arc<SetupCtx>) {
        let mut rng = rand::thread_rng();

        let mut trace = Lookup1Trace::new();
        let num_rows = trace.num_rows();

        log::debug!("{} ··· Starting witness computation stage {}", Self::MY_NAME, 1);

        let num_lookups = trace[0].sel.len();

        for i in 0..num_rows {
            let val = rng.gen();
            let mut n_sel = 0;
            for j in 0..num_lookups {
                trace[i].f[j] = val;
                let selected = rng.gen_bool(0.5);
                trace[i].sel[j] = F::from_bool(selected);
                if selected {
                    n_sel += 1;
                }
            }
            trace[i].t = val;
            trace[i].mul = F::from_canonical_usize(n_sel);
        }

        AirInstance::from_trace(pctx.clone(), ectx.clone(), sctx.clone(), None, &mut trace, None, None);
    }
}

impl<F: PrimeField + Copy> WitnessComponent<F> for Lookup1<F>
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
