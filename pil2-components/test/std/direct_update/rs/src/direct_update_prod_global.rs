use std::sync::Arc;

use proofman::{WitnessComponent, WitnessManager};
use proofman_common::{AirInstance, ExecutionCtx, ProofCtx, SetupCtx};

use p3_field::PrimeField;
use num_traits::ToPrimitive;
use rand::{distributions::Standard, prelude::Distribution, Rng};

use crate::{DirectUpdateProdGlobalTrace, DIRECT_UPDATE_PROD_AIRGROUP_ID, DIRECT_UPDATE_PROD_GLOBAL_AIR_IDS};

pub struct DirectUpdateProdGlobal<F: PrimeField> {
    _phantom: std::marker::PhantomData<F>,
}

impl<F: PrimeField> DirectUpdateProdGlobal<F>
where
    Standard: Distribution<F>,
{
    const MY_NAME: &'static str = "DUPG";

    pub fn new(wcm: Arc<WitnessManager<F>>) -> Arc<Self> {
        let direct_update_prod_global = Arc::new(Self { _phantom: std::marker::PhantomData });

        wcm.register_component(
            direct_update_prod_global.clone(),
            Some(DIRECT_UPDATE_PROD_AIRGROUP_ID),
            Some(DIRECT_UPDATE_PROD_GLOBAL_AIR_IDS),
        );

        direct_update_prod_global
    }

    pub fn execute(&self, pctx: Arc<ProofCtx<F>>, ectx: Arc<ExecutionCtx>, sctx: Arc<SetupCtx>) {
        let num_rows =
            pctx.global_info.airs[DIRECT_UPDATE_PROD_AIRGROUP_ID][DIRECT_UPDATE_PROD_GLOBAL_AIR_IDS[0]].num_rows;
        let trace = DirectUpdateProdGlobalTrace::new(num_rows);

        let air_instance = AirInstance::new(
            sctx.clone(),
            DIRECT_UPDATE_PROD_AIRGROUP_ID,
            DIRECT_UPDATE_PROD_GLOBAL_AIR_IDS[0],
            None,
            trace.buffer.unwrap(),
        );
        let (is_myne, gid) = ectx.dctx.write().unwrap().add_instance(
            DIRECT_UPDATE_PROD_AIRGROUP_ID,
            DIRECT_UPDATE_PROD_GLOBAL_AIR_IDS[0],
            1,
        );
        if is_myne {
            pctx.air_instance_repo.add_air_instance(air_instance, Some(gid));
        }
    }
}

impl<F: PrimeField> WitnessComponent<F> for DirectUpdateProdGlobal<F>
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
            "DirectUpdateProdGlobal",
            stage
        );

        if stage == 1 {
            let air_instances_vec = &mut pctx.air_instance_repo.air_instances.write().unwrap();
            let air_instance = &mut air_instances_vec[air_instance_id.unwrap()];
            let buffer = &mut air_instance.trace;
            let num_rows =
                pctx.pilout.get_air(DIRECT_UPDATE_PROD_AIRGROUP_ID, DIRECT_UPDATE_PROD_GLOBAL_AIR_IDS[0]).num_rows();

            let mut trace = DirectUpdateProdGlobalTrace::map_buffer(buffer.as_mut_slice(), num_rows, 0).unwrap();

            let chosen_index = rng.gen_range(0..=num_rows - 1);
            let mut values: [F; 4] = [F::zero(); 4];
            for i in 0..num_rows {
                for j in 0..2 {
                    trace[i].c[j] = F::from_canonical_u64(rng.gen_range(0..=(1 << 63) - 1));
                    trace[i].d[j] = F::from_canonical_u64(rng.gen_range(0..=(1 << 63) - 1));
                }

                if i == chosen_index {
                    trace[i].perform_operation = F::from_bool(true);
                    values[0] = trace[i].c[0];
                    values[1] = trace[i].c[1];
                    values[2] = trace[i].d[0];
                    values[3] = trace[i].d[1];
                }
            }

            // Set public values
            pctx.set_public_value_by_name(
                values[0].as_canonical_biguint().to_u64().expect("Cannot convert to usize"),
                "c_public",
                Some(vec![0]),
            );
            pctx.set_public_value_by_name(
                values[1].as_canonical_biguint().to_u64().expect("Cannot convert to usize"),
                "c_public",
                Some(vec![1]),
            );

            // Set proof values
            pctx.set_proof_value("d_proofval_0", values[2]);
            pctx.set_proof_value("d_proofval_1", values[3]);

            // Choose one direct update
            let h = rng.gen_bool(0.5);
            pctx.set_proof_value("perform_global_update_0", F::from_bool(h));
            pctx.set_proof_value("perform_global_update_1", F::from_bool(!h));
        }
    }
}
