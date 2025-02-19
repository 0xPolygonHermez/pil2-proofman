use std::sync::Arc;

use pil_std_lib::Std;
use witness::WitnessComponent;

use proofman_common::{add_air_instance, FromTrace, AirInstance, ProofCtx};

use p3_field::PrimeField64;
use rand::{distributions::Standard, prelude::Distribution, Rng};

use crate::RangeCheck4Trace;

pub struct RangeCheck4<F: PrimeField64> {
    std_lib: Arc<Std<F>>,
}

impl<F: PrimeField64> RangeCheck4<F>
where
    Standard: Distribution<F>,
{
    const MY_NAME: &'static str = "RngChck4";

    pub fn new(std_lib: Arc<Std<F>>) -> Arc<Self> {
        Arc::new(Self { std_lib })
    }
}

impl<F: PrimeField64> WitnessComponent<F> for RangeCheck4<F>
where
    Standard: Distribution<F>,
{
    fn execute(&self, pctx: Arc<ProofCtx<F>>) {
        let mut rng = rand::thread_rng();

        let mut trace = RangeCheck4Trace::new_zeroes();
        let num_rows = trace.num_rows();

        log::debug!("{} ··· Starting witness computation stage {}", Self::MY_NAME, 1);

        let range1 = self.std_lib.get_range(0, (1 << 16) - 1, Some(true));
        let range2 = self.std_lib.get_range(0, (1 << 8) - 1, Some(true));
        let range3 = self.std_lib.get_range(50, (1 << 7) - 1, Some(true));
        let range4 = self.std_lib.get_range(127, 1 << 8, Some(true));
        let range5 = self.std_lib.get_range(1, (1 << 16) + 1, Some(true));
        let range6 = self.std_lib.get_range(127, 1 << 16, Some(true));
        let range7 = self.std_lib.get_range(-1, 1 << 3, Some(true));
        let range8 = self.std_lib.get_range(-(1 << 7) + 1, -50, Some(true));
        let range9 = self.std_lib.get_range(-(1 << 8) + 1, -127, Some(true));

        for i in 0..num_rows {
            let selected1 = rng.gen_bool(0.5);
            trace[i].sel1 = F::from_bool(selected1);

            // selected1 and selected2 have to be disjoint for the range check to pass
            let selected2 = if selected1 { false } else { rng.gen_bool(0.5) };
            trace[i].sel2 = F::from_bool(selected2);

            if selected1 {
                let val1 = rng.gen_range(0..=(1 << 16) - 1);
                let val2 = rng.gen_range(127..=(1 << 16));
                let val3: i8 = rng.gen_range(-1..=(1 << 3));
                trace[i].a1 = F::from_canonical_u32(val1);
                trace[i].a5 = F::from_canonical_u32(val2);
                trace[i].a6 = if val3 < 0 {
                    F::from_canonical_u64((val3 as i128 + F::ORDER_U64 as i128) as u64)
                } else {
                    F::from_canonical_u8(val3 as u8)
                };

                self.std_lib.range_check(val1 as i64, 1, range1);
                self.std_lib.range_check(val2 as i64, 1, range6);
                self.std_lib.range_check(val3 as i64, 1, range7);
            }
            if selected2 {
                let val1 = rng.gen_range(0..=(1 << 8) - 1);
                let val2 = rng.gen_range(50..=(1 << 7) - 1);
                let val3 = rng.gen_range(127..=(1 << 8));
                let val4 = rng.gen_range(1..=(1 << 16) + 1);
                trace[i].a1 = F::from_canonical_u16(val1);
                trace[i].a2 = F::from_canonical_u8(val2);
                trace[i].a3 = F::from_canonical_u16(val3);
                trace[i].a4 = F::from_canonical_u32(val4);

                self.std_lib.range_check(val1 as i64, 1, range2);
                self.std_lib.range_check(val2 as i64, 1, range3);
                self.std_lib.range_check(val3 as i64, 1, range4);
                self.std_lib.range_check(val4 as i64, 1, range5);
            }

            let val7: i16 = rng.gen_range(-(1 << 7) + 1..=-50);
            trace[i].a7 = F::from_canonical_u64((val7 as i128 + F::ORDER_U64 as i128) as u64);
            self.std_lib.range_check(val7 as i64, 1, range8);

            let val8: i16 = rng.gen_range(-(1 << 8) + 1..=-127);
            trace[i].a8 = F::from_canonical_u64((val8 as i128 + F::ORDER_U64 as i128) as u64);
            self.std_lib.range_check(val8 as i64, 1, range9);
        }

        let air_instance = AirInstance::new_from_trace(FromTrace::new(&mut trace));
        add_air_instance::<F>(air_instance, pctx.clone());
    }
}
