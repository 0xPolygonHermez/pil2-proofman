use std::sync::Arc;

use pil_std_lib::Std;
use witness::WitnessComponent;

use proofman_common::{add_air_instance, FromTrace, AirInstance, ProofCtx};

use p3_field::PrimeField64;
use rand::{distributions::Standard, prelude::Distribution, Rng};

use crate::RangeCheckMixTrace;

pub struct RangeCheckMix<F: PrimeField64> {
    std_lib: Arc<Std<F>>,
}

impl<F: PrimeField64> RangeCheckMix<F>
where
    Standard: Distribution<F>,
{
    const MY_NAME: &'static str = "RngChMix";

    pub fn new(std_lib: Arc<Std<F>>) -> Arc<Self> {
        Arc::new(Self { std_lib })
    }
}

impl<F: PrimeField64> WitnessComponent<F> for RangeCheckMix<F>
where
    Standard: Distribution<F>,
{
    fn execute(&self, pctx: Arc<ProofCtx<F>>) {
        let mut rng = rand::thread_rng();

        let mut trace = RangeCheckMixTrace::new_zeroes();
        let num_rows = trace.num_rows();

        log::debug!("{} ··· Starting witness computation stage {}", Self::MY_NAME, 1);

        let range1 = self.std_lib.get_range(0, (1 << 8) - 1, Some(true));
        let range2 = self.std_lib.get_range(50, (1 << 7) - 1, Some(true));
        let range3 = self.std_lib.get_range(-1, 1 << 3, Some(true));
        let range4 = self.std_lib.get_range(-(1 << 7) + 1, -50, Some(true));

        let range5 = self.std_lib.get_range(0, (1 << 7) - 1, Some(false));
        let range6 = self.std_lib.get_range(0, (1 << 4) - 1, Some(false));
        let range7 = self.std_lib.get_range(1 << 5, (1 << 8) - 1, Some(false));
        let range8 = self.std_lib.get_range(1 << 8, (1 << 9) - 1, Some(false));

        let range9 = self.std_lib.get_range(5225, 29023, Some(false));
        // let range10 = self.std_lib.get_range(-8719, -7269, Some(false));
        let range11 = self.std_lib.get_range(-10, 10, Some(false));

        for i in 0..num_rows {
            // First interface
            let val0 = rng.gen_range(0..=(1 << 8) - 1);
            trace[i].a[0] = F::from_canonical_u16(val0);
            self.std_lib.range_check(val0 as i64, 1, range1);

            let val1 = rng.gen_range(50..=(1 << 7) - 1);
            trace[i].a[1] = F::from_canonical_u8(val1);
            self.std_lib.range_check(val1 as i64, 1, range2);

            let val2: i8 = rng.gen_range(-1..=(1 << 3));
            trace[i].a[2] = if val2 < 0 {
                F::from_canonical_u64((val2 as i128 + F::ORDER_U64 as i128) as u64)
            } else {
                F::from_canonical_u8(val2 as u8)
            };
            self.std_lib.range_check(val2 as i64, 1, range3);

            let val3: i16 = rng.gen_range(-(1 << 7) + 1..=-50);
            trace[i].a[3] = F::from_canonical_u64((val3 as i128 + F::ORDER_U64 as i128) as u64);
            self.std_lib.range_check(val3 as i64, 1, range4);

            // Second interface
            let range_selector1 = rng.gen_bool(0.5);
            trace[i].range_sel[0] = F::from_bool(range_selector1);

            let range_selector2 = rng.gen_bool(0.5);
            trace[i].range_sel[1] = F::from_bool(range_selector2);

            if range_selector1 {
                let val = rng.gen_range(0..=(1 << 7) - 1);
                trace[i].b[0] = F::from_canonical_u16(val);

                self.std_lib.range_check(val as i64, 1, range5);
            } else {
                let val = rng.gen_range(0..=(1 << 4) - 1);
                trace[i].b[0] = F::from_canonical_u16(val);

                self.std_lib.range_check(val as i64, 1, range6);
            }

            if range_selector2 {
                let val = rng.gen_range((1 << 5)..=(1 << 8) - 1);
                trace[i].b[1] = F::from_canonical_u16(val);

                self.std_lib.range_check(val as i64, 1, range7);
            } else {
                let val = rng.gen_range((1 << 8)..=(1 << 9) - 1);
                trace[i].b[1] = F::from_canonical_u16(val);

                self.std_lib.range_check(val as i64, 1, range8);
            }

            // Third interface
            let range = rng.gen_range(0..=2);

            match range {
                0 => {
                    trace[i].range_sel[2] = F::one();
                    let val = rng.gen_range(5225..=29023);
                    trace[i].c[0] = F::from_canonical_u32(val);

                    self.std_lib.range_check(val as i64, 1, range9);
                }
                1 => {
                    trace[i].range_sel[3] = F::one();
                    let colu_val: i8 = rng.gen_range(-10..=10);
                    trace[i].c[0] = if colu_val < 0 {
                        F::from_canonical_u64((colu_val as i128 + F::ORDER_U64 as i128) as u64)
                    } else {
                        F::from_canonical_u8(colu_val as u8)
                    };

                    self.std_lib.range_check(colu_val as i64, 1, range11);
                }
                2 => {
                    trace[i].range_sel[4] = F::one();
                    let val = rng.gen_range(0..=(1 << 7) - 1);
                    trace[i].c[0] = F::from_canonical_u32(val);

                    self.std_lib.range_check(val as i64, 1, range5);
                }
                _ => panic!("Invalid range"),
            }
        }

        let air_instance = AirInstance::new_from_trace(FromTrace::new(&mut trace));
        add_air_instance::<F>(air_instance, pctx.clone());
    }
}
