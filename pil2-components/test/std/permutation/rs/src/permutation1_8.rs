use std::sync::Arc;

use proofman::{WitnessComponent, WitnessManager};
use proofman_common::{FromTrace, AirInstance, ExecutionCtx, ProofCtx, SetupCtx};

use p3_field::PrimeField;
use rand::{distributions::Standard, prelude::Distribution, Rng};

use crate::Permutation1_8Trace;

pub struct Permutation1_8<F> {
    _phantom: std::marker::PhantomData<F>,
}

impl<F: PrimeField + Copy> Permutation1_8<F>
where
    Standard: Distribution<F>,
{
    const MY_NAME: &'static str = "Perm1_8 ";

    pub fn new(wcm: Arc<WitnessManager<F>>) -> Arc<Self> {
        let permutation1_8 = Arc::new(Self { _phantom: std::marker::PhantomData });

        let airgroup_id = Permutation1_8Trace::<F>::AIRGROUP_ID;
        let air_id = Permutation1_8Trace::<F>::AIR_ID;

        wcm.register_component(permutation1_8.clone(), airgroup_id, air_id);

        permutation1_8
    }

    pub fn execute(&self, pctx: Arc<ProofCtx<F>>, _ectx: Arc<ExecutionCtx>, sctx: Arc<SetupCtx>) {
        let mut rng = rand::thread_rng();
        let mut trace = Permutation1_8Trace::new();
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

        let air_instance = AirInstance::new_from_trace(sctx.clone(), FromTrace::new(&mut trace));
        pctx.air_instance_repo.add_air_instance(air_instance, Some(0));
    }
}

impl<F: PrimeField + Copy> WitnessComponent<F> for Permutation1_8<F>
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
