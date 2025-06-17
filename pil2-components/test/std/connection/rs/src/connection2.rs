use std::sync::Arc;

use witness::{WitnessComponent, execute, define_wc};
use proofman_common::{FromTrace, AirInstance, ProofCtx, SetupCtx};

use fields::PrimeField64;
use rand::{
    distr::{StandardUniform, Distribution},
    Rng, SeedableRng,
    rngs::StdRng,
};

use crate::Connection2Trace;

define_wc!(Connection2, "Connct_2");

impl<F: PrimeField64> WitnessComponent<F> for Connection2
where
    StandardUniform: Distribution<F>,
{
    execute!(Connection2Trace, 1);

    fn calculate_witness(
        &self,
        stage: u32,
        pctx: Arc<ProofCtx<F>>,
        _sctx: Arc<SetupCtx<F>>,
        instance_ids: &[usize],
        _n_cores: usize,
        witness_buffer: &mut Vec<Vec<F>>,
    ) {
        if stage == 1 {
            let seed = if cfg!(feature = "debug") { 0 } else { rand::rng().random::<u64>() };
            let mut rng = StdRng::seed_from_u64(seed);

            let mut trace = Connection2Trace::new_from_vec(witness_buffer.remove(0));
            let num_rows = trace.num_rows();

            tracing::debug!("··· Starting witness computation stage {}", 1);

            for i in 0..num_rows {
                trace[i].a = rng.random();
                trace[i].b = rng.random();
                trace[i].c = rng.random();
            }

            trace[0].a = trace[1].a;

            let air_instance = AirInstance::new_from_trace(FromTrace::new(&mut trace));
            pctx.add_air_instance(air_instance, instance_ids[0]);
        }
    }
}
