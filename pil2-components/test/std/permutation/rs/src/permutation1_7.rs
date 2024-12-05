use std::sync::Arc;

use proofman::{WitnessComponent, WitnessManager};
use proofman_common::{AirInstance, ExecutionCtx, ProofCtx, SetupCtx};

use p3_field::PrimeField;
use rand::{distributions::Standard, prelude::Distribution, Rng};

use crate::{Permutation1_7Trace, PERMUTATION_1_7_AIR_IDS, PERMUTATION_AIRGROUP_ID};

pub struct Permutation1_7<F> {
    _phantom: std::marker::PhantomData<F>,
}

impl<F: PrimeField + Copy> Permutation1_7<F>
where
    Standard: Distribution<F>,
{
    const MY_NAME: &'static str = "Perm1_7 ";

    pub fn new(wcm: Arc<WitnessManager<F>>) -> Arc<Self> {
        let permutation1_7 = Arc::new(Self { _phantom: std::marker::PhantomData });

        wcm.register_component(permutation1_7.clone(), Some(PERMUTATION_AIRGROUP_ID), Some(PERMUTATION_1_7_AIR_IDS));

        permutation1_7
    }

    pub fn execute(&self, pctx: Arc<ProofCtx<F>>, ectx: Arc<ExecutionCtx>, sctx: Arc<SetupCtx>) {
        let mut rng = rand::thread_rng();
        let mut trace = Permutation1_7Trace::new();
        let num_rows = trace.num_rows();

        log::debug!("{} ··· Starting witness computation stage {}", Self::MY_NAME, 1);

        // TODO: Add the ability to send inputs to permutation2
        //       and consequently add random selectors

        // Assumes
        for i in 0..num_rows {
            trace[i].a1 = rng.gen();
            trace[i].b1 = rng.gen();

            trace[i].a2 = F::from_canonical_u8(200);
            trace[i].b2 = F::from_canonical_u8(201);

            trace[i].a3 = rng.gen();
            trace[i].b3 = rng.gen();

            trace[i].a4 = F::from_canonical_u8(100);
            trace[i].b4 = F::from_canonical_u8(101);

            trace[i].sel1 = F::one();
            trace[i].sel3 = F::one(); // F::from_canonical_u8(rng.gen_range(0..=1));
        }

        // TODO: Add the permutation of indexes

        // Proves
        for i in 0..num_rows {
            let index = num_rows - i - 1;
            // let mut index = rng.gen_range(0..num_rows);
            trace[i].c1 = trace[index].a1;
            trace[i].d1 = trace[index].b1;

            // index = rng.gen_range(0..num_rows);
            trace[i].c2 = trace[index].a3;
            trace[i].d2 = trace[index].b3;

            trace[i].sel2 = trace[i].sel1;
        }

        AirInstance::from_trace(pctx.clone(), ectx.clone(), sctx.clone(), None, &mut trace);
    }
}

impl<F: PrimeField + Copy> WitnessComponent<F> for Permutation1_7<F>
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
