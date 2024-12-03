use std::sync::Arc;

use proofman::{WitnessComponent, WitnessManager};
use proofman_common::{AirInstance, ExecutionCtx, ProofCtx, SetupCtx};

use p3_field::PrimeField;
use rand::{distributions::Standard, prelude::Distribution, seq::SliceRandom, Rng};

use crate::{Permutation1_6Trace, PERMUTATION_1_6_AIR_IDS, PERMUTATION_AIRGROUP_ID};

pub struct Permutation1_6<F> {
    _phantom: std::marker::PhantomData<F>,
}

impl<F: PrimeField + Copy> Permutation1_6<F>
where
    Standard: Distribution<F>,
{
    const MY_NAME: &'static str = "Perm1_6 ";

    pub fn new(wcm: Arc<WitnessManager<F>>) -> Arc<Self> {
        let permutation1_6 = Arc::new(Self { _phantom: std::marker::PhantomData });

        wcm.register_component(permutation1_6.clone(), Some(PERMUTATION_AIRGROUP_ID), Some(PERMUTATION_1_6_AIR_IDS));

        permutation1_6
    }

    pub fn execute(&self, pctx: Arc<ProofCtx<F>>, ectx: Arc<ExecutionCtx>, sctx: Arc<SetupCtx>) {
        let num_rows = pctx.global_info.airs[PERMUTATION_AIRGROUP_ID][PERMUTATION_1_6_AIR_IDS[0]].num_rows;
        let trace = Permutation1_6Trace::new(num_rows);

        let air_instance =
            AirInstance::new(sctx.clone(), PERMUTATION_AIRGROUP_ID, PERMUTATION_1_6_AIR_IDS[0], None, trace.buffer.unwrap());
        let (is_myne, gid) =
            ectx.dctx.write().unwrap().add_instance(PERMUTATION_AIRGROUP_ID, PERMUTATION_1_6_AIR_IDS[0], 1);
        if is_myne {
            pctx.air_instance_repo.add_air_instance(air_instance, Some(gid));
        }
    }
}

impl<F: PrimeField + Copy> WitnessComponent<F> for Permutation1_6<F>
where
    Standard: Distribution<F>,
{
    fn calculate_witness(
        &self,
        stage: u32,
        air_instance_id: Option<usize>,
        pctx: Arc<ProofCtx<F>>,
        _ectx: Arc<ExecutionCtx>,
        _sctx: Arc<SetupCtx>,
    ) {
        let mut rng = rand::thread_rng();

        let air_instances_vec = &mut pctx.air_instance_repo.air_instances.write().unwrap();
        let air_instance = &mut air_instances_vec[air_instance_id.unwrap()];

        log::debug!(
            Self::MY_NAME,
            air.name().unwrap_or("unknown"),
            stage
        );

        if stage == 1 {
            let buffer = &mut air_instance.trace;
            let num_rows = pctx.pilout.get_air(airgroup_id, air_id).num_rows();

            // I cannot, programatically, link the permutation trace with its air_id
            let mut trace =
                Permutation1_6Trace::map_buffer(buffer.as_mut_slice(), num_rows, 0).unwrap();

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

                trace[i].sel1 = F::from_bool(rng.gen_bool(0.5));
                trace[i].sel3 = F::one();
            }

            let mut indices: Vec<usize> = (0..num_rows).collect();
            indices.shuffle(&mut rng);

            // Proves
            for i in 0..num_rows {
                // We take a random permutation of the indices to show that the permutation check is passing
                trace[i].c1 = trace[indices[i]].a1;
                trace[i].d1 = trace[indices[i]].b1;

                trace[i].c2 = trace[indices[i]].a3;
                trace[i].d2 = trace[indices[i]].b3;

                trace[i].sel2 = trace[indices[i]].sel1;
            }
        }
    }
}
