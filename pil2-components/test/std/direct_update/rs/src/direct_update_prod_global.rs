use std::sync::Arc;

use witness::{WitnessComponent, execute, define_wc};
use proofman_common::{FromTrace, AirInstance, ProofCtx, SetupCtx};

use fields::PrimeField64;
use rand::{
    distr::{StandardUniform, Distribution},
    Rng, SeedableRng,
    rngs::StdRng,
};

use crate::{DirectUpdateProdGlobalTrace, DirectUpdatePublicValues, DirectUpdateProofValues};

define_wc!(DirectUpdateProdGlobal, "DUPG    ");

impl<F: PrimeField64> WitnessComponent<F> for DirectUpdateProdGlobal
where
    StandardUniform: Distribution<F>,
{
    execute!(DirectUpdateProdGlobalTrace, 1);

    fn calculate_witness(&self, stage: u32, pctx: Arc<ProofCtx<F>>, _sctx: Arc<SetupCtx<F>>, instance_ids: &[usize]) {
        if stage == 1 {
            let seed = if cfg!(feature = "debug") { 0 } else { rand::rng().random::<u64>() };
            let mut rng = StdRng::seed_from_u64(seed);

            let mut trace = DirectUpdateProdGlobalTrace::new();
            let num_rows = trace.num_rows();

            tracing::debug!("··· Starting witness computation stage {}", 1);

            let chosen_index = rng.random_range(0..=num_rows - 1);
            let mut values: [F; 4] = [F::ZERO; 4];
            for i in 0..num_rows {
                for j in 0..2 {
                    trace[i].c[j] = F::from_u64(rng.random_range(0..=(1 << 63) - 1));
                    trace[i].d[j] = F::from_u64(rng.random_range(0..=(1 << 63) - 1));
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
            public_values.c_public[0] = values[0];
            public_values.c_public[1] = values[1];

            let mut proof_values = DirectUpdateProofValues::from_vec_guard(pctx.get_proof_values());
            proof_values.d_proofval_0 = values[2];
            proof_values.d_proofval_1 = values[3];

            // Choose one direct update
            let h = rng.random::<bool>();
            proof_values.perform_global_update_0 = F::from_bool(h);
            proof_values.perform_global_update_1 = F::from_bool(!h);

            let air_instance = AirInstance::new_from_trace(FromTrace::new(&mut trace));
            pctx.add_air_instance(air_instance, instance_ids[0]);
        }
    }
}
