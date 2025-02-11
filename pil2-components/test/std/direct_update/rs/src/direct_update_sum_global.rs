use std::sync::{Arc, RwLock};

use witness::WitnessComponent;
use proofman_common::{FromTrace, AirInstance, ProofCtx, SetupCtx};

use p3_field::PrimeField;
use rand::{distributions::Standard, prelude::Distribution, Rng, SeedableRng, rngs::StdRng};

use crate::{DirectUpdateSumGlobalTrace, DirectUpdatePublicValues, DirectUpdateProofValues};

pub struct DirectUpdateSumGlobal {
    instance_ids: RwLock<Vec<usize>>,
}

impl DirectUpdateSumGlobal {
    const MY_NAME: &'static str = "DUSG    ";

    pub fn new() -> Arc<Self> {
        Arc::new(Self { instance_ids: RwLock::new(Vec::new()) })
    }
}

impl<F: PrimeField + Copy> WitnessComponent<F> for DirectUpdateSumGlobal
where
    Standard: Distribution<F>,
{
    fn execute(&self, pctx: Arc<ProofCtx<F>>) -> Vec<usize> {
        let global_ids = vec![pctx.add_instance(
            DirectUpdateSumGlobalTrace::<usize>::AIRGROUP_ID,
            DirectUpdateSumGlobalTrace::<usize>::AIR_ID,
        )];
        self.instance_ids.write().unwrap().push(global_ids[0]);
        global_ids
    }

    fn calculate_witness(&self, stage: u32, pctx: Arc<ProofCtx<F>>, _sctx: Arc<SetupCtx<F>>, instance_ids: &[usize]) {
        if stage == 1 {
            let seed = if cfg!(feature = "debug") { 0 } else { rand::thread_rng().gen::<u64>() };
            let mut rng = StdRng::seed_from_u64(seed);

            let mut trace = DirectUpdateSumGlobalTrace::new();
            let num_rows = trace.num_rows();

            log::debug!("{} ··· Starting witness computation stage {}", Self::MY_NAME, 1);

            let chosen_index = rng.gen_range(0..=num_rows - 1);
            let mut values: [F; 4] = [F::zero(); 4];
            for i in 0..num_rows {
                for j in 0..2 {
                    trace[i].c[j] = F::from_canonical_u64(rng.gen_range(0..=(1 << 63) - 1));
                    trace[i].d[j] = F::from_canonical_u64(rng.gen_range(0..=(1 << 63) - 1));
                }

                trace[i].perform_operation = F::from_bool(i == chosen_index);
                if i == chosen_index {
                    values[0] = trace[i].c[0];
                    values[1] = trace[i].c[1];
                    values[2] = trace[i].d[0];
                    values[3] = trace[i].d[1];
                }
            }

            let mut public_values = DirectUpdatePublicValues::from_vec_guard(pctx.get_publics());
            public_values.c_public_s[0] = values[0];
            public_values.c_public_s[1] = values[1];

            let mut proof_values = DirectUpdateProofValues::from_vec_guard(pctx.get_proof_values());
            proof_values.d_proofval_0_s = values[2];
            proof_values.d_proofval_1_s = values[3];

            // Choose one direct update
            let h = rng.gen::<bool>();
            proof_values.perform_global_update_0_s = F::from_bool(h);
            proof_values.perform_global_update_1_s = F::from_bool(!h);

            let air_instance = AirInstance::new_from_trace(FromTrace::new(&mut trace));
            pctx.add_air_instance(air_instance, instance_ids[0]);
        }
    }
}
