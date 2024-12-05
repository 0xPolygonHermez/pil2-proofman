use std::sync::Arc;

use proofman::{WitnessComponent, WitnessManager};
use proofman_common::{AirInstance, ExecutionCtx, ProofCtx, SetupCtx};

use p3_field::PrimeField;
use rand::{distributions::Standard, prelude::Distribution, Rng};

use crate::{Lookup2_13Trace, LOOKUP_2_13_AIR_IDS, LOOKUP_AIRGROUP_ID};

pub struct Lookup2_13<F> {
    _phantom: std::marker::PhantomData<F>,
}

impl<F: PrimeField + Copy> Lookup2_13<F>
where
    Standard: Distribution<F>,
{
    const MY_NAME: &'static str = "Lkup2_13";

    pub fn new(wcm: Arc<WitnessManager<F>>) -> Arc<Self> {
        let lookup2_13 = Arc::new(Self { _phantom: std::marker::PhantomData });

        wcm.register_component(lookup2_13.clone(), Some(LOOKUP_AIRGROUP_ID), Some(LOOKUP_2_13_AIR_IDS));

        lookup2_13
    }

    pub fn execute(&self, pctx: Arc<ProofCtx<F>>, ectx: Arc<ExecutionCtx>, sctx: Arc<SetupCtx>) {
        let mut rng = rand::thread_rng();

        let mut trace = Lookup2_13Trace::new();
        let num_rows = trace.num_rows();

        // TODO: Add the ability to send inputs to lookup3
        //       and consequently add random selectors

        log::debug!("{} ··· Starting witness computation stage {}", Self::MY_NAME, 1);

        for i in 0..num_rows {
            // Inner lookups
            trace[i].a1 = rng.gen();
            trace[i].b1 = rng.gen();
            trace[i].c1 = trace[i].a1;
            trace[i].d1 = trace[i].b1;

            trace[i].a3 = rng.gen();
            trace[i].b3 = rng.gen();
            trace[i].c2 = trace[i].a3;
            trace[i].d2 = trace[i].b3;
            let selected = rng.gen_bool(0.5);
            trace[i].sel1 = F::from_bool(selected);
            if selected {
                trace[i].mul = trace[i].sel1;
            }

            // Outer lookups
            trace[i].a2 = F::from_canonical_usize(i);
            trace[i].b2 = F::from_canonical_usize(i);

            trace[i].a4 = F::from_canonical_usize(i);
            trace[i].b4 = F::from_canonical_usize(i);
            trace[i].sel2 = F::from_bool(true);
        }

        AirInstance::from_trace(pctx.clone(), ectx.clone(), sctx.clone(), None, &mut trace, None, None);
    }
}

impl<F: PrimeField + Copy> WitnessComponent<F> for Lookup2_13<F>
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
