use std::sync::Arc;

use proofman::{WitnessComponent, WitnessManager};
use proofman_common::{AirInstance, ExecutionCtx, ProofCtx, SetupCtx};

use p3_field::PrimeField;

use crate::{Permutation2_6Trace, PERMUTATION_2_6_AIR_IDS, PERMUTATION_AIRGROUP_ID};

pub struct Permutation2<F> {
    _phantom: std::marker::PhantomData<F>,
}

impl<F: PrimeField + Copy> Permutation2<F> {
    const MY_NAME: &'static str = "Perm2   ";

    pub fn new(wcm: Arc<WitnessManager<F>>) -> Arc<Self> {
        let permutation2 = Arc::new(Self { _phantom: std::marker::PhantomData });

        wcm.register_component(permutation2.clone(), Some(PERMUTATION_AIRGROUP_ID), Some(PERMUTATION_2_6_AIR_IDS));

        permutation2
    }

    pub fn execute(&self, pctx: Arc<ProofCtx<F>>, ectx: Arc<ExecutionCtx>, sctx: Arc<SetupCtx>) {
        let num_rows = pctx.global_info.airs[PERMUTATION_AIRGROUP_ID][PERMUTATION_2_6_AIR_IDS[0]].num_rows;
        let air = pctx.pilout.get_air(PERMUTATION_AIRGROUP_ID, PERMUTATION_2_6_AIR_IDS[0]);

        log::debug!(
            "{}: ··· Witness computation for AIR '{}' at stage 1",
            Self::MY_NAME,
            air.name().unwrap_or("unknown"),
        );

        let mut trace = Permutation2_6Trace::new(num_rows);

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

        let air_instance = AirInstance::new(
            sctx.clone(),
            PERMUTATION_AIRGROUP_ID,
            PERMUTATION_2_6_AIR_IDS[0],
            None,
            trace.buffer.unwrap(),
        );
        let (is_myne, gid) =
            ectx.dctx.write().unwrap().add_instance(PERMUTATION_AIRGROUP_ID, PERMUTATION_2_6_AIR_IDS[0], 1);
        if is_myne {
            pctx.air_instance_repo.add_air_instance(air_instance, Some(gid));
        }
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
