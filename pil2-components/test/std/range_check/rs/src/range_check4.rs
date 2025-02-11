use std::sync::{Arc, RwLock};

use pil_std_lib::Std;
use witness::WitnessComponent;

use proofman_common::{FromTrace, AirInstance, ProofCtx, SetupCtx};

use num_bigint::BigInt;
use num_traits::ToPrimitive;
use p3_field::PrimeField;
use rand::{distributions::Standard, prelude::Distribution, Rng, SeedableRng, rngs::StdRng};

use crate::RangeCheck4Trace;

pub struct RangeCheck4<F: PrimeField> {
    instance_ids: RwLock<Vec<usize>>,
    std_lib: Arc<Std<F>>,
}

impl<F: PrimeField> RangeCheck4<F>
where
    Standard: Distribution<F>,
{
    const MY_NAME: &'static str = "RngChck4";

    pub fn new(std_lib: Arc<Std<F>>) -> Arc<Self> {
        Arc::new(Self { std_lib, instance_ids: RwLock::new(Vec::new()) })
    }
}

impl<F: PrimeField> WitnessComponent<F> for RangeCheck4<F>
where
    Standard: Distribution<F>,
{
    fn execute(&self, pctx: Arc<ProofCtx<F>>) -> Vec<usize> {
        let global_ids =
            vec![pctx.add_instance(RangeCheck4Trace::<usize>::AIRGROUP_ID, RangeCheck4Trace::<usize>::AIR_ID)];
        self.instance_ids.write().unwrap().push(global_ids[0]);
        global_ids
    }

    fn calculate_witness(&self, stage: u32, pctx: Arc<ProofCtx<F>>, _sctx: Arc<SetupCtx<F>>, instance_ids: &[usize]) {
        if stage == 1 {
            let seed = if cfg!(feature = "debug") { 0 } else { rand::thread_rng().gen::<u64>() };
            let mut rng = StdRng::seed_from_u64(seed);
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
                let selected1 = rng.gen::<bool>();
                trace[i].sel1 = F::from_bool(selected1);

                // selected1 and selected2 have to be disjoint for the range check to pass
                let selected2 = if selected1 { false } else { rng.gen_bool(0.5) };
                trace[i].sel2 = F::from_bool(selected2);

                if selected1 {
                    trace[i].a1 = F::from_canonical_u32(rng.gen_range(0..=(1 << 16) - 1));
                    trace[i].a2 = F::zero();
                    trace[i].a3 = F::zero();
                    trace[i].a4 = F::zero();
                    trace[i].a5 = F::from_canonical_u32(rng.gen_range(127..=(1 << 16)));
                    let mut a6_val: i128 = rng.gen_range(-1..=2i128.pow(3));
                    if a6_val < 0 {
                        a6_val += F::order().to_i128().unwrap();
                    }
                    trace[i].a6 = F::from_canonical_u64(a6_val as u64);

                    self.std_lib.range_check(trace[i].a1, F::one(), range1);
                    self.std_lib.range_check(trace[i].a5, F::one(), range6);
                    self.std_lib.range_check(trace[i].a6, F::one(), range7);
                }
                if selected2 {
                    trace[i].a1 = F::from_canonical_u16(rng.gen_range(0..=(1 << 8) - 1));
                    trace[i].a2 = F::from_canonical_u8(rng.gen_range(50..=(1 << 7) - 1));
                    trace[i].a3 = F::from_canonical_u16(rng.gen_range(127..=(1 << 8)));
                    trace[i].a4 = F::from_canonical_u32(rng.gen_range(1..=(1 << 16) + 1));
                    trace[i].a5 = F::zero();
                    trace[i].a6 = F::zero();

                    self.std_lib.range_check(trace[i].a1, F::one(), range2);
                    self.std_lib.range_check(trace[i].a2, F::one(), range3);
                    self.std_lib.range_check(trace[i].a3, F::one(), range4);
                    self.std_lib.range_check(trace[i].a4, F::one(), range5);
                }

                if !selected1 && !selected2 {
                    trace[i].a1 = F::zero();
                    trace[i].a2 = F::zero();
                    trace[i].a3 = F::zero();
                    trace[i].a4 = F::zero();
                    trace[i].a5 = F::zero();
                    trace[i].a6 = F::zero();
                }

                let mut a7_val: i128 = rng.gen_range(-(2i128.pow(7)) + 1..=-50);
                if a7_val < 0 {
                    a7_val += F::order().to_i128().unwrap();
                }
                trace[i].a7 = F::from_canonical_u64(a7_val as u64);
                self.std_lib.range_check(trace[i].a7, F::one(), range8);

                let mut a8_val: i128 = rng.gen_range(-(2i128.pow(8)) + 1..=-127);
                if a8_val < 0 {
                    a8_val += F::order().to_i128().unwrap();
                }
                trace[i].a8 = F::from_canonical_u64(a8_val as u64);
                self.std_lib.range_check(trace[i].a8, F::one(), range9);
            }

            let air_instance = AirInstance::new_from_trace(FromTrace::new(&mut trace));
            pctx.add_air_instance(air_instance, instance_ids[0]);
        }
    }
}
