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

use crate::RangeCheck4Trace;

define_wc_with_std!(RangeCheck4, "RngChck4");

impl<F: PrimeField> WitnessComponent<F> for RangeCheck4<F>
where
    StandardUniform: Distribution<F>,
{
    execute!(RangeCheck4Trace, 1);

    fn calculate_witness(&self, stage: u32, pctx: Arc<ProofCtx<F>>, _sctx: Arc<SetupCtx<F>>, instance_ids: &[usize]) {
        if stage == 1 {
            let mut rng = StdRng::seed_from_u64(self.seed.load(Ordering::Relaxed));
            let mut trace = RangeCheck4Trace::new();
            let num_rows = trace.num_rows();

            log::debug!("{} ··· Starting witness computation stage {}", Self::MY_NAME, 1);

            let range1 = self.std_lib.get_range(BigInt::from(0), BigInt::from((1 << 16) - 1), Some(true));
            let range2 = self.std_lib.get_range(BigInt::from(0), BigInt::from((1 << 8) - 1), Some(true));
            let range3 = self.std_lib.get_range(BigInt::from(50), BigInt::from((1 << 7) - 1), Some(true));
            let range4 = self.std_lib.get_range(BigInt::from(127), BigInt::from(1 << 8), Some(true));
            let range5 = self.std_lib.get_range(BigInt::from(1), BigInt::from((1 << 16) + 1), Some(true));
            let range6 = self.std_lib.get_range(BigInt::from(127), BigInt::from(1 << 16), Some(true));
            let range7 = self.std_lib.get_range(BigInt::from(-1), BigInt::from(1 << 3), Some(true));
            let range8 = self.std_lib.get_range(BigInt::from(-(1 << 7) + 1), BigInt::from(-50), Some(true));
            let range9 = self.std_lib.get_range(BigInt::from(-(1 << 8) + 1), BigInt::from(-127), Some(true));

            for i in 0..num_rows {
                let selected1 = rng.random::<bool>();
                trace[i].sel1 = F::from_bool(selected1);

                // selected1 and selected2 have to be disjoint for the range check to pass
                let selected2 = if selected1 { false } else { rng.random_bool(0.5) };
                trace[i].sel2 = F::from_bool(selected2);

                if selected1 {
                    trace[i].a1 = F::from_u32(rng.random_range(0..=(1 << 16) - 1));
                    trace[i].a2 = F::ZERO;
                    trace[i].a3 = F::ZERO;
                    trace[i].a4 = F::ZERO;
                    trace[i].a5 = F::from_u32(rng.random_range(127..=(1 << 16)));
                    let mut a6_val: i128 = rng.random_range(-1..=2i128.pow(3));
                    if a6_val < 0 {
                        a6_val += F::order().to_i128().unwrap();
                    }
                    trace[i].a6 = F::from_u64(a6_val as u64);

                    self.std_lib.range_check(trace[i].a1, F::ONE, range1);
                    self.std_lib.range_check(trace[i].a5, F::ONE, range6);
                    self.std_lib.range_check(trace[i].a6, F::ONE, range7);
                }
                if selected2 {
                    trace[i].a1 = F::from_u16(rng.random_range(0..=(1 << 8) - 1));
                    trace[i].a2 = F::from_u8(rng.random_range(50..=(1 << 7) - 1));
                    trace[i].a3 = F::from_u16(rng.random_range(127..=(1 << 8)));
                    trace[i].a4 = F::from_u32(rng.random_range(1..=(1 << 16) + 1));
                    trace[i].a5 = F::ZERO;
                    trace[i].a6 = F::ZERO;

                    self.std_lib.range_check(trace[i].a1, F::ONE, range2);
                    self.std_lib.range_check(trace[i].a2, F::ONE, range3);
                    self.std_lib.range_check(trace[i].a3, F::ONE, range4);
                    self.std_lib.range_check(trace[i].a4, F::ONE, range5);
                }

                if !selected1 && !selected2 {
                    trace[i].a1 = F::ZERO;
                    trace[i].a2 = F::ZERO;
                    trace[i].a3 = F::ZERO;
                    trace[i].a4 = F::ZERO;
                    trace[i].a5 = F::ZERO;
                    trace[i].a6 = F::ZERO;
                }

                let mut a7_val: i128 = rng.random_range(-(2i128.pow(7)) + 1..=-50);
                if a7_val < 0 {
                    a7_val += F::order().to_i128().unwrap();
                }
                trace[i].a7 = F::from_u64(a7_val as u64);
                self.std_lib.range_check(trace[i].a7, F::ONE, range8);

                let mut a8_val: i128 = rng.random_range(-(2i128.pow(8)) + 1..=-127);
                if a8_val < 0 {
                    a8_val += F::order().to_i128().unwrap();
                }
                trace[i].a8 = F::from_u64(a8_val as u64);
                self.std_lib.range_check(trace[i].a8, F::ONE, range9);
            }

            let air_instance = AirInstance::new_from_trace(FromTrace::new(&mut trace));
            pctx.add_air_instance(air_instance, instance_ids[0]);
        }
    }
}
