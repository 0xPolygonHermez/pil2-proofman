use std::sync::Arc;

use witness::{WitnessComponent, execute, define_wc_with_std};

use proofman_common::{FromTrace, AirInstance, ProofCtx, SetupCtx};

use num_bigint::BigInt;
use num_traits::ToPrimitive;
use p3_field::PrimeField;
use rand::{
    distr::{StandardUniform, Distribution},
    Rng, SeedableRng,
    rngs::StdRng,
};

use crate::RangeCheckDynamic2Trace;

define_wc_with_std!(RangeCheckDynamic2, "RngChDy2");

impl<F: PrimeField> WitnessComponent<F> for RangeCheckDynamic2<F>
where
    StandardUniform: Distribution<F>,
{
    execute!(RangeCheckDynamic2Trace, 1);

    fn calculate_witness(&self, stage: u32, pctx: Arc<ProofCtx<F>>, _sctx: Arc<SetupCtx<F>>, instance_ids: &[usize]) {
        if stage == 1 {
            let mut rng = StdRng::seed_from_u64(self.seed.load(Ordering::Relaxed));

            let mut trace = RangeCheckDynamic2Trace::new();
            let num_rows = trace.num_rows();

            log::debug!("{} ··· Starting witness computation stage {}", Self::MY_NAME, 1);

            let range1 = self.std_lib.get_range(BigInt::from(5225), BigInt::from(29023), Some(false));
            let range2 = self.std_lib.get_range(BigInt::from(-8719), BigInt::from(-7269), Some(false));
            let range3 = self.std_lib.get_range(BigInt::from(-10), BigInt::from(10), Some(false));
            let range4 = self.std_lib.get_range(BigInt::from(0), BigInt::from((1 << 8) - 1), Some(false));
            let range5 = self.std_lib.get_range(BigInt::from(0), BigInt::from((1 << 7) - 1), Some(false));

            for i in 0..num_rows {
                let range = rng.random_range(0..=4);

                match range {
                    0 => {
                        trace[i].sel_1 = F::ONE;
                        trace[i].sel_2 = F::ZERO;
                        trace[i].sel_3 = F::ZERO;
                        trace[i].sel_4 = F::ZERO;
                        trace[i].sel_5 = F::ZERO;
                        trace[i].colu = F::from_u16(rng.random_range(5225..=29023));

                        self.std_lib.range_check(trace[i].colu, F::ONE, range1);
                    }
                    1 => {
                        trace[i].sel_1 = F::ZERO;
                        trace[i].sel_2 = F::ONE;
                        trace[i].sel_3 = F::ZERO;
                        trace[i].sel_4 = F::ZERO;
                        trace[i].sel_5 = F::ZERO;
                        let colu_val = rng.random_range(-8719..=-7269) + F::order().to_i128().unwrap();
                        trace[i].colu = F::from_u64(colu_val as u64);

                        self.std_lib.range_check(trace[i].colu, F::ONE, range2);
                    }
                    2 => {
                        trace[i].sel_1 = F::ZERO;
                        trace[i].sel_2 = F::ZERO;
                        trace[i].sel_3 = F::ONE;
                        trace[i].sel_4 = F::ZERO;
                        trace[i].sel_5 = F::ZERO;
                        let mut colu_val: i128 = rng.random_range(-10..=10);
                        if colu_val < 0 {
                            colu_val += F::order().to_i128().unwrap();
                        }
                        trace[i].colu = F::from_u64(colu_val as u64);

                        self.std_lib.range_check(trace[i].colu, F::ONE, range3);
                    }
                    3 => {
                        trace[i].sel_1 = F::ZERO;
                        trace[i].sel_2 = F::ZERO;
                        trace[i].sel_3 = F::ZERO;
                        trace[i].sel_4 = F::ONE;
                        trace[i].sel_5 = F::ZERO;
                        trace[i].colu = F::from_u32(rng.random_range(0..=(1 << 8) - 1));

                        self.std_lib.range_check(trace[i].colu, F::ONE, range4);
                    }
                    4 => {
                        trace[i].sel_1 = F::ZERO;
                        trace[i].sel_2 = F::ZERO;
                        trace[i].sel_3 = F::ZERO;
                        trace[i].sel_4 = F::ZERO;
                        trace[i].sel_5 = F::ONE;
                        trace[i].colu = F::from_u32(rng.random_range(0..=(1 << 7) - 1));

                        self.std_lib.range_check(trace[i].colu, F::ONE, range5);
                    }
                    _ => panic!("Invalid range"),
                }
            }

            let air_instance = AirInstance::new_from_trace(FromTrace::new(&mut trace));
            pctx.add_air_instance(air_instance, instance_ids[0]);
        }
    }
}
