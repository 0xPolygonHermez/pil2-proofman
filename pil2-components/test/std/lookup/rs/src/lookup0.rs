use std::sync::Arc;

use proofman::{WitnessComponent, WitnessManager};
use proofman_common::{add_air_instance, FromTrace, AirInstance, ExecutionCtx, ProofCtx, SetupCtx};

use p3_field::PrimeField;
use rand::{distributions::Standard, prelude::Distribution, Rng};

use crate::Lookup0Trace;

pub struct Lookup0<F> {
    _phantom: std::marker::PhantomData<F>,
}

impl<F: PrimeField + Copy> Lookup0<F>
where
    Standard: Distribution<F>,
{
    const MY_NAME: &'static str = "Lookup0";

    pub fn new(wcm: Arc<WitnessManager<F>>) -> Arc<Self> {
        let lookup0 = Arc::new(Self { _phantom: std::marker::PhantomData });

        let airgroup_id = Lookup0Trace::<F>::AIRGROUP_ID;
        let air_id = Lookup0Trace::<F>::AIR_ID;

        wcm.register_component(lookup0.clone(), airgroup_id, air_id);

        lookup0
    }

    pub fn execute(&self, pctx: Arc<ProofCtx<F>>, ectx: Arc<ExecutionCtx>, _sctx: Arc<SetupCtx>) {
        let mut rng = rand::thread_rng();

        let mut trace = Lookup0Trace::new();
        let num_rows = trace.num_rows();

        log::debug!("{} ··· Starting witness computation stage {}", Self::MY_NAME, 1);

        let num_lookups = trace[0].sel.len();

        for j in 0..num_lookups {
            for i in 0..num_rows {
                // Assumes
                trace[i].f[2 * j] = rng.gen();
                trace[i].f[2 * j + 1] = rng.gen();
                let selected = rng.gen_bool(0.5);
                trace[i].sel[j] = F::from_bool(selected);

                // Proves
                trace[i].t[2 * j] = trace[i].f[2 * j];
                trace[i].t[2 * j + 1] = trace[i].f[2 * j + 1];
                if selected {
                    trace[i].mul[j] = F::one();
                }
            }
        }

        let air_instance = AirInstance::new_from_trace( FromTrace::new(&mut trace));
        add_air_instance::<F>(air_instance, ectx, pctx.clone());
    }
}

impl<F: PrimeField + Copy> WitnessComponent<F> for Lookup0<F>
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
