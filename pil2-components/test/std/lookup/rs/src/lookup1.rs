use std::sync::Arc;

use proofman::{WitnessComponent, WitnessManager};
use proofman_common::{AirInstance, ExecutionCtx, ProofCtx, SetupCtx};

use p3_field::PrimeField;
use rand::{distributions::Standard, prelude::Distribution, Rng};

use crate::{Lookup1Trace, LOOKUP_1_AIR_IDS, LOOKUP_AIRGROUP_ID};

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

        wcm.register_component(lookup1.clone(), Some(LOOKUP_AIRGROUP_ID), Some(LOOKUP_1_AIR_IDS));

        lookup1
    }

    pub fn execute(&self, pctx: Arc<ProofCtx<F>>, ectx: Arc<ExecutionCtx>, sctx: Arc<SetupCtx>) {
        let mut rng = rand::thread_rng();

        let num_rows = pctx.global_info.airs[LOOKUP_AIRGROUP_ID][LOOKUP_1_AIR_IDS[0]].num_rows;
        let air = pctx.pilout.get_air(LOOKUP_AIRGROUP_ID, LOOKUP_1_AIR_IDS[0]);

        log::debug!(
            "{}: ··· Witness computation for AIR '{}' at stage 1",
            Self::MY_NAME,
            air.name().unwrap_or("unknown"),
        );

        let mut trace = Lookup1Trace::new(num_rows);

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

        let air_instance =
            AirInstance::new(sctx.clone(), LOOKUP_AIRGROUP_ID, LOOKUP_1_AIR_IDS[0], None, trace.buffer.unwrap());

        let (is_myne, gid) = ectx.dctx.write().unwrap().add_instance(LOOKUP_AIRGROUP_ID, LOOKUP_1_AIR_IDS[0], 1);
        if is_myne {
            pctx.air_instance_repo.add_air_instance(air_instance, Some(gid));
        }
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
