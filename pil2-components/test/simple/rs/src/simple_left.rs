use std::sync::Arc;

use proofman::{WitnessComponent, WitnessManager};
use proofman_common::{AirInstance, ExecutionCtx, ProofCtx, SetupCtx};

use p3_field::PrimeField;
use rand::{distributions::Standard, prelude::Distribution, seq::SliceRandom};

use crate::{SimpleLeftTrace, SIMPLE_AIRGROUP_ID, SIMPLE_LEFT_AIR_IDS};

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

        wcm.register_component(simple_left.clone(), Some(SIMPLE_AIRGROUP_ID), Some(SIMPLE_LEFT_AIR_IDS));

        simple_left
    }

    pub fn execute(&self, pctx: Arc<ProofCtx<F>>, ectx: Arc<ExecutionCtx>, sctx: Arc<SetupCtx>) {
        let num_rows = pctx.global_info.airs[SIMPLE_AIRGROUP_ID][SIMPLE_LEFT_AIR_IDS[0]].num_rows;
        let trace = SimpleLeftTrace::new(num_rows);

        let air_instance =
            AirInstance::new(sctx.clone(), SIMPLE_AIRGROUP_ID, SIMPLE_LEFT_AIR_IDS[0], None, trace.buffer.unwrap());

        let (is_myne, gid) = ectx.dctx.write().unwrap().add_instance(SIMPLE_AIRGROUP_ID, SIMPLE_LEFT_AIR_IDS[0], 1);
        if is_myne {
            pctx.air_instance_repo.add_air_instance(air_instance, Some(gid));
        }
    }
}

impl<F: PrimeField + Copy> WitnessComponent<F> for SimpleLeft<F>
where
    Standard: Distribution<F>,
{
    fn calculate_witness(
        &self,
        stage: u32,
        air_instance_id: Option<usize>,
        pctx: Arc<ProofCtx<F>>,
        ectx: Arc<ExecutionCtx>,
        sctx: Arc<SetupCtx>,
    ) {
        let mut rng = rand::thread_rng();

        let air_instances_vec = &mut pctx.air_instance_repo.air_instances.write().unwrap();
        let air_instance = &mut air_instances_vec[air_instance_id.unwrap()];

        let airgroup_id = air_instance.airgroup_id;
        let air_id = air_instance.air_id;
        let air = pctx.pilout.get_air(airgroup_id, air_id);

        log::debug!(
            "{}: ··· Computing witness computation for AIR '{}' at stage {}",
            Self::MY_NAME,
            air.name().unwrap_or("unknown"),
            stage
        );

        if stage == 1 {
            let buffer = &mut air_instance.trace;
            let num_rows = pctx.pilout.get_air(airgroup_id, air_id).num_rows();

            // I cannot, programatically, link the permutation trace with its air_id
            let mut trace = SimpleLeftTrace::map_buffer(buffer.as_mut_slice(), num_rows, 0).unwrap();

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
        }
    }
}
