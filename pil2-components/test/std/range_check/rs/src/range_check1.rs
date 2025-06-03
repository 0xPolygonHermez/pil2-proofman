use std::sync::Arc;

use witness::{WitnessComponent, execute, define_wc_with_std};

use proofman_common::{FromTrace, AirInstance, ProofCtx, SetupCtx};

use fields::PrimeField64;
use rand::{
    distr::{StandardUniform, Distribution},
    Rng, SeedableRng,
    rngs::StdRng,
};

use crate::RangeCheck1Trace;

define_wc_with_std!(RangeCheck1, "RngChck1");

impl<F: PrimeField64> WitnessComponent<F> for RangeCheck1<F>
where
    StandardUniform: Distribution<F>,
{
    execute!(RangeCheck1Trace, 1);

    fn calculate_witness(
        &self,
        stage: u32,
        pctx: Arc<ProofCtx<F>>,
        _sctx: Arc<SetupCtx<F>>,
        instance_ids: &[usize],
        _n_cores: usize,
    ) {
        if stage == 1 {
            let mut rng = StdRng::seed_from_u64(self.seed.load(Ordering::Relaxed));

            let mut trace = RangeCheck1Trace::new();
            let num_rows = trace.num_rows();

            tracing::debug!("··· Starting witness computation stage {}", 1);

            let range1 = self.std_lib.get_range(0, (1 << 8) - 1, Some(false));
            let range2 = self.std_lib.get_range(0, (1 << 4) - 1, Some(false));
            let range3 = self.std_lib.get_range(60, (1 << 16) - 1, Some(false));
            let range4 = self.std_lib.get_range(8228, 17400, Some(false));

            for i in 0..num_rows {
                trace[i].a1 = F::ZERO;
                trace[i].a2 = F::ZERO;
                trace[i].a3 = F::ZERO;
                trace[i].a4 = F::ZERO;
                trace[i].a5 = F::ZERO;

                let selected1 = rng.random::<bool>();
                trace[i].sel1 = F::from_bool(selected1);

                let selected2 = rng.random::<bool>();
                trace[i].sel2 = F::from_bool(selected2);

                let selected3 = rng.random::<bool>();
                trace[i].sel3 = F::from_bool(selected3);

                if selected1 {
                    let val1 = rng.random_range(0..=(1 << 8) - 1);
                    let val2 = rng.random_range(60..=(1 << 16) - 1);
                    trace[i].a1 = F::from_u16(val1);
                    trace[i].a3 = F::from_u32(val2);

                    self.std_lib.range_check(val1 as i64, 1, range1);
                    self.std_lib.range_check(val2 as i64, 1, range3);
                }

                if selected2 {
                    let val1 = rng.random_range(0..=(1 << 4) - 1);
                    let val2 = rng.random_range(8228..=17400);
                    trace[i].a2 = F::from_u8(val1);
                    trace[i].a4 = F::from_u16(val2);

                    self.std_lib.range_check(val1 as i64, 1, range2);
                    self.std_lib.range_check(val2 as i64, 1, range4);
                }

                if selected3 {
                    let val = rng.random_range(0..=(1 << 8) - 1);
                    trace[i].a5 = F::from_u16(val);

                    self.std_lib.range_check(val as i64, 1, range1);
                }
            }

            let air_instance = AirInstance::new_from_trace(FromTrace::new(&mut trace));
            pctx.add_air_instance(air_instance, instance_ids[0]);
        }
    }
}
