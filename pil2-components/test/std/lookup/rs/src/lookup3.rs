use std::sync::Arc;

use proofman::{WitnessComponent, WitnessManager};
use proofman_common::{add_air_instance, FromTrace, AirInstance,  ProofCtx, SetupCtx};

use p3_field::PrimeField;

use crate::Lookup3Trace;

pub struct Lookup3<F> {
    _phantom: std::marker::PhantomData<F>,
}

impl<F: PrimeField + Copy> Lookup3<F> {
    const MY_NAME: &'static str = "Lookup_3";

    pub fn new(wcm: Arc<WitnessManager<F>>) -> Arc<Self> {
        let lookup3 = Arc::new(Self { _phantom: std::marker::PhantomData });

        let airgroup_id = Lookup3Trace::<F>::AIRGROUP_ID;
        let air_id = Lookup3Trace::<F>::AIR_ID;

        wcm.register_component(lookup3.clone(), airgroup_id, air_id);

        lookup3
    }

    pub fn execute(&self, pctx: Arc<ProofCtx<F>>) {
        // For simplicity, add a single instance of each air
        let mut trace = Lookup3Trace::new();
        let num_rows = trace.num_rows();

        log::debug!("{} ··· Starting witness computation stage {}", Self::MY_NAME, 1);

        for i in 0..num_rows {
            trace[i].c1 = F::from_canonical_usize(i);
            trace[i].d1 = F::from_canonical_usize(i);
            if i < (1 << 12) {
                trace[i].mul1 = F::from_canonical_usize(4);
            } else if i < (1 << 13) {
                trace[i].mul1 = F::from_canonical_usize(3);
            } else {
                trace[i].mul1 = F::from_canonical_usize(2);
            }

            trace[i].c2 = F::from_canonical_usize(i);
            trace[i].d2 = F::from_canonical_usize(i);
            if i < (1 << 12) {
                trace[i].mul2 = F::from_canonical_usize(4);
            } else if i < (1 << 13) {
                trace[i].mul2 = F::from_canonical_usize(3);
            } else {
                trace[i].mul2 = F::from_canonical_usize(2);
            }
        }

        let air_instance = AirInstance::new_from_trace( FromTrace::new(&mut trace));
        add_air_instance::<F>(air_instance, pctx.clone());
    }
}

impl<F: PrimeField + Copy> WitnessComponent<F> for Lookup3<F> {
    fn calculate_witness(
        &self,
        _stage: u32,
        _air_instance_id: Option<usize>,
        _pctx: Arc<ProofCtx<F>>,
        _
        _sctx: Arc<SetupCtx>,
    ) {
    }
}
