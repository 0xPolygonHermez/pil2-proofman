use std::sync::Arc;

use witness::{WitnessComponent, execute, define_wc};
use proofman_common::{FromTrace, AirInstance, ProofCtx, SetupCtx};

use p3_field::PrimeField64;
use rand::{
    distr::{StandardUniform, Distribution},
    Rng, SeedableRng,
    rngs::StdRng,
};

use crate::Lookup1Trace;

define_wc!(Lookup1, "Lookup_1");

impl<F: PrimeField64> WitnessComponent<F> for Lookup1
where
    StandardUniform: Distribution<F>,
{
    execute!(Lookup1Trace, 1);

    fn calculate_witness(
        &self,
        stage: u32,
        pctx: Arc<ProofCtx<F>>,
        _sctx: Arc<SetupCtx<F>>,
        instance_ids: &[usize],
        _n_cores: usize,
    ) {
        if stage == 1 {
            let seed = if cfg!(feature = "debug") { 0 } else { rand::rng().random::<u64>() };
            let mut rng = StdRng::seed_from_u64(seed);

            let mut trace = Lookup1Trace::new();
            let num_rows = trace.num_rows();

            tracing::debug!("··· Starting witness computation stage {}", 1);

            let num_lookups = trace[0].sel.len();

            for i in 0..num_rows {
                let val = rng.random();
                let mut n_sel = 0;
                for j in 0..num_lookups {
                    trace[i].f[j] = val;
                    let selected = rng.random::<bool>();
                    trace[i].sel[j] = F::from_bool(selected);
                    if selected {
                        n_sel += 1;
                    }
                }
                trace[i].t = val;
                trace[i].mul = F::from_usize(n_sel);
            }

            let air_instance = AirInstance::new_from_trace(FromTrace::new(&mut trace));
            pctx.add_air_instance(air_instance, instance_ids[0]);
        }
    }
}
