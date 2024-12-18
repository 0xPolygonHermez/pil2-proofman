use std::sync::Arc;

use proofman::{WitnessComponent, WitnessManager};
use proofman_common::{AirInstance, ExecutionCtx, ProofCtx, SetupCtx};

use p3_field::PrimeField;
use num_traits::ToPrimitive;
use rand::{distributions::Standard, prelude::Distribution, Rng};

use crate::{DirectUpdateProdLocalTrace, DIRECT_UPDATE_PROD_AIRGROUP_ID, DIRECT_UPDATE_PROD_LOCAL_AIR_IDS};

pub struct DirectUpdateProdLocal<F: PrimeField> {
    _phantom: std::marker::PhantomData<F>,
}

impl<F: PrimeField> DirectUpdateProdLocal<F>
where
    Standard: Distribution<F>,
{
    const MY_NAME: &'static str = "DUPL";

    pub fn new(wcm: Arc<WitnessManager<F>>) -> Arc<Self> {
        let direct_update_prod_local = Arc::new(Self { _phantom: std::marker::PhantomData });

        wcm.register_component(
            direct_update_prod_local.clone(),
            Some(DIRECT_UPDATE_PROD_AIRGROUP_ID),
            Some(DIRECT_UPDATE_PROD_LOCAL_AIR_IDS),
        );

        direct_update_prod_local
    }

    pub fn execute(&self, pctx: Arc<ProofCtx<F>>, ectx: Arc<ExecutionCtx>, sctx: Arc<SetupCtx>) {
        let num_rows =
            pctx.global_info.airs[DIRECT_UPDATE_PROD_AIRGROUP_ID][DIRECT_UPDATE_PROD_LOCAL_AIR_IDS[0]].num_rows;
        let trace = DirectUpdateProdLocalTrace::new(num_rows);

        let air_instance = AirInstance::new(
            sctx.clone(),
            DIRECT_UPDATE_PROD_AIRGROUP_ID,
            DIRECT_UPDATE_PROD_LOCAL_AIR_IDS[0],
            None,
            trace.buffer.unwrap(),
        );
        let (is_myne, gid) = ectx.dctx.write().unwrap().add_instance(
            DIRECT_UPDATE_PROD_AIRGROUP_ID,
            DIRECT_UPDATE_PROD_LOCAL_AIR_IDS[0],
            1,
        );
        if is_myne {
            pctx.air_instance_repo.add_air_instance(air_instance, Some(gid));
        }
    }
}

impl<F: PrimeField> WitnessComponent<F> for DirectUpdateProdLocal<F>
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

        log::debug!(
            "{}: ··· Witness computation for AIR '{}' at stage {}",
            Self::MY_NAME,
            "DirectUpdateProdLocal",
            stage
        );

        if stage == 1 {
            let air_instances_vec = &mut pctx.air_instance_repo.air_instances.write().unwrap();
            let air_instance = &mut air_instances_vec[air_instance_id.unwrap()];
            let buffer = &mut air_instance.trace;
            let num_rows =
                pctx.pilout.get_air(DIRECT_UPDATE_PROD_AIRGROUP_ID, DIRECT_UPDATE_PROD_LOCAL_AIR_IDS[0]).num_rows();

            let mut trace = DirectUpdateProdLocalTrace::map_buffer(buffer.as_mut_slice(), num_rows, 0).unwrap();

            let chosen_index = rng.gen_range(0..=num_rows - 1);
            let mut values: [F; 6] = [F::zero(); 6];
            for i in 0..num_rows {
                for j in 0..2 {
                    trace[i].a[j] = F::from_canonical_u64(rng.gen_range(0..=(1 << 63) - 1));
                    trace[i].b[j] = F::from_canonical_u64(rng.gen_range(0..=(1 << 63) - 1));
                    trace[i].c[j] = F::from_canonical_u64(rng.gen_range(0..=(1 << 63) - 1));
                }

                if i == chosen_index {
                    trace[i].perform_operation = F::from_bool(true);
                    values[0] = trace[i].a[0];
                    values[1] = trace[i].a[1];
                    values[2] = trace[i].b[0];
                    values[3] = trace[i].b[1];
                    values[4] = trace[i].c[0];
                    values[5] = trace[i].c[1];
                }
            }

            // Set public values
            pctx.set_public_value_by_name(
                values[0].as_canonical_biguint().to_u64().expect("Cannot convert to usize"),
                "a_public",
                Some(vec![0]),
            );
            pctx.set_public_value_by_name(
                values[1].as_canonical_biguint().to_u64().expect("Cannot convert to usize"),
                "a_public",
                Some(vec![1]),
            );

            // Set proof values
            pctx.set_proof_value("b_proofval_0", values[2]);
            pctx.set_proof_value("b_proofval_1", values[3]);

            // Set air values
            air_instance.set_airvalue("DirectUpdateProdLocal.c_airval", Some(vec![0]), values[4]);
            air_instance.set_airvalue("DirectUpdateProdLocal.c_airval", Some(vec![1]), values[5]);

            // Choose one direct update
            let h = rng.gen_bool(0.5);
            air_instance.set_airvalue("DirectUpdateProdLocal.perform_direct_update", Some(vec![0]), F::from_bool(h));
            air_instance.set_airvalue("DirectUpdateProdLocal.perform_direct_update", Some(vec![1]), F::from_bool(!h));
        }
    }
}
