use std::sync::Arc;

use pil_std_lib::Std;
use witness::WitnessComponent;

use proofman_common::{add_air_instance, FromTrace, AirInstance, ProofCtx};

use p3_field::PrimeField64;
use rand::{distributions::Standard, prelude::Distribution, Rng};

use crate::RangeCheckDynamic2Trace;

pub struct RangeCheckDynamic2<F: PrimeField64> {
    std_lib: Arc<Std<F>>,
}

impl<F: PrimeField64> RangeCheckDynamic2<F>
where
    Standard: Distribution<F>,
{
    const MY_NAME: &'static str = "RngChDy2";

    pub fn new(std_lib: Arc<Std<F>>) -> Arc<Self> {
        Arc::new(Self { std_lib })
    }
}

impl<F: PrimeField64> WitnessComponent<F> for RangeCheckDynamic2<F>
where
    Standard: Distribution<F>,
{
    fn execute(&self, pctx: Arc<ProofCtx<F>>) {
        let mut rng = rand::thread_rng();

        let mut trace = RangeCheckDynamic2Trace::new_zeroes();
        let num_rows = trace.num_rows();

        log::debug!("{} ··· Starting witness computation stage {}", Self::MY_NAME, 1);

        let range1 = self.std_lib.get_range(5225, 29023, Some(false));
        let range2 = self.std_lib.get_range(-8719, -7269, Some(false));
        let range3 = self.std_lib.get_range(-10, 10, Some(false));
        let range4 = self.std_lib.get_range(0, (1 << 8) - 1, Some(false));
        let range5 = self.std_lib.get_range(0, (1 << 7) - 1, Some(false));

        for i in 0..num_rows {
            let range = rng.gen_range(0..=4);

            match range {
                0 => {
                    trace[i].sel_1 = F::one();
                    let val = rng.gen_range(5225..=29023);
                    trace[i].colu = F::from_canonical_u16(val);

                    self.std_lib.range_check(val as i64, 1, range1);
                }
                1 => {
                    trace[i].sel_2 = F::one();
                    let colu_val: i16 = rng.gen_range(-8719..=-7269);
                    trace[i].colu = F::from_canonical_u64((colu_val as i128 + F::ORDER_U64 as i128) as u64);

                    self.std_lib.range_check(colu_val as i64, 1, range2);
                }
                2 => {
                    trace[i].sel_3 = F::one();
                    let colu_val: i8 = rng.gen_range(-10..=10);
                    trace[i].colu = if colu_val < 0 {
                        F::from_canonical_u64((colu_val as i128 + F::ORDER_U64 as i128) as u64)
                    } else {
                        F::from_canonical_u8(colu_val as u8)
                    };

                    self.std_lib.range_check(colu_val as i64, 1, range3);
                }
                3 => {
                    trace[i].sel_4 = F::one();
                    let val = rng.gen_range(0..=(1 << 8) - 1);
                    trace[i].colu = F::from_canonical_u32(val);

                    self.std_lib.range_check(val as i64, 1, range4);
                }
                4 => {
                    trace[i].sel_5 = F::one();
                    let val = rng.gen_range(0..=(1 << 7) - 1);
                    trace[i].colu = F::from_canonical_u32(val);

                    self.std_lib.range_check(val as i64, 1, range5);
                }
                _ => panic!("Invalid range"),
            }
        }

        let air_instance = AirInstance::new_from_trace(FromTrace::new(&mut trace));
        add_air_instance::<F>(air_instance, pctx.clone());
    }
}
