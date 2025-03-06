use std::sync::Arc;

use pil_std_lib::Std;
use witness::WitnessComponent;
use proofman_common::{add_air_instance, FromTrace, AirInstance, ProofCtx};

use p3_field::PrimeField64;
use rand::{distributions::Standard, prelude::Distribution, seq::SliceRandom, Rng};

use crate::SimpleLeftTrace;

pub struct SimpleLeft<F: PrimeField64> {
    std_lib: Arc<Std<F>>,
}

impl<F: PrimeField64> SimpleLeft<F> {
    const MY_NAME: &'static str = "SimLeft ";

    pub fn new(std_lib: Arc<Std<F>>) -> Arc<Self> {
        Arc::new(Self { std_lib })
    }
}

impl<F: PrimeField64 + Copy> WitnessComponent<F> for SimpleLeft<F>
where
    Standard: Distribution<F>,
{
    fn execute(&self, pctx: Arc<ProofCtx<F>>) {
        let mut rng = rand::thread_rng();

        let mut trace = SimpleLeftTrace::new();
        let num_rows = trace.num_rows();

        log::debug!("{} ··· Starting witness computation stage {}", Self::MY_NAME, 1);

        let range = [
            self.std_lib.get_range(0, (1 << 8) - 1, Some(true)),
            self.std_lib.get_range(0, (1 << 16) - 1, Some(true)),
            self.std_lib.get_range(1, (1 << 8) - 1, Some(true)),
            self.std_lib.get_range(0, 1 << 8, Some(true)),
            self.std_lib.get_range(0, (1 << 8) - 1, Some(false)),
            self.std_lib.get_range(-(1 << 7), -1, Some(false)),
            self.std_lib.get_range(-(1 << 7) - 1, (1 << 7) - 1, Some(false)),
        ];

        // Assumes
        for i in 0..num_rows {
            trace[i].a = F::from_canonical_u64(rng.gen_range(0..=(1 << 63) - 1));
            trace[i].b = F::from_canonical_u64(rng.gen_range(0..=(1 << 63) - 1));

            trace[i].e = F::from_canonical_u8(200);
            trace[i].f = F::from_canonical_u8(201);

            trace[i].g = F::from_canonical_usize(i);
            trace[i].h = F::from_canonical_usize(num_rows - i - 1);

            let val = [
                rng.gen_range(0..=(1 << 8) - 1),
                rng.gen_range(0..=(1 << 16) - 1),
                rng.gen_range(1..=(1 << 8) - 1),
                rng.gen_range(0..=(1 << 8)),
                rng.gen_range(0..=(1 << 8) - 1),
                rng.gen_range(-(1 << 7)..-1),
                rng.gen_range(-(1 << 7) - 1..(1 << 7) - 1),
            ];

            for j in 0..7 {
                // Specific values for specific ranges
                if j == 4 {
                    if i == 0 {
                        let val = 0;
                        trace[i].k[j] = F::from_canonical_u32(val);
                        self.std_lib.range_check(val as i64, 1, range[j]);
                        continue;
                    } else if i == 1 {
                        let val = 1 << 4;
                        trace[i].k[j] = F::from_canonical_u32(val);
                        self.std_lib.range_check(val as i64, 1, range[j]);
                        continue;
                    } else if i == 2 {
                        let val = (1 << 8) - 1;
                        trace[i].k[j] = F::from_canonical_u32(val);
                        self.std_lib.range_check(val as i64, 1, range[j]);
                        continue;
                    }
                } else if j == 5 {
                    if i == 0 {
                        let val = -(1 << 7);
                        trace[i].k[j] = F::from_canonical_u64((val as i128 + F::ORDER_U64 as i128) as u64);
                        self.std_lib.range_check(val as i64, 1, range[j]);
                        continue;
                    } else if i == 1 {
                        let val = -(1 << 2);
                        trace[i].k[j] = F::from_canonical_u64((val as i128 + F::ORDER_U64 as i128) as u64);
                        self.std_lib.range_check(val as i64, 1, range[j]);
                        continue;
                    } else if i == 2 {
                        let val = -1;
                        trace[i].k[j] = F::from_canonical_u64((val as i128 + F::ORDER_U64 as i128) as u64);
                        self.std_lib.range_check(val as i64, 1, range[j]);
                        continue;
                    }
                } else if j == 6 {
                    if i == 0 {
                        let val = -(1 << 7) - 1;
                        trace[i].k[j] = F::from_canonical_u64((val as i128 + F::ORDER_U64 as i128) as u64);
                        self.std_lib.range_check(val as i64, 1, range[j]);
                        continue;
                    } else if i == 1 {
                        let val = -(1 << 2);
                        trace[i].k[j] = F::from_canonical_u64((val as i128 + F::ORDER_U64 as i128) as u64);
                        self.std_lib.range_check(val as i64, 1, range[j]);
                        continue;
                    } else if i == 2 {
                        let val = -1;
                        trace[i].k[j] = F::from_canonical_u64((val as i128 + F::ORDER_U64 as i128) as u64);
                        self.std_lib.range_check(val as i64, 1, range[j]);
                        continue;
                    } else if i == 3 {
                        let val = 0;
                        trace[i].k[j] = F::from_canonical_u32(val);
                        self.std_lib.range_check(val as i64, 1, range[j]);
                        continue;
                    } else if i == 4 {
                        let val = (1 << 7) - 1;
                        trace[i].k[j] = F::from_canonical_u32(val);
                        self.std_lib.range_check(val as i64, 1, range[j]);
                        continue;
                    } else if i == 5 {
                        let val = 10;
                        trace[i].k[j] = F::from_canonical_u32(val);
                        self.std_lib.range_check(val as i64, 1, range[j]);
                        continue;
                    }
                }

                trace[i].k[j] = if val[j] < 0 {
                    F::from_canonical_u64((val[j] as i128 + F::ORDER_U64 as i128) as u64)
                } else {
                    F::from_canonical_u32(val[j] as u32)
                };
                self.std_lib.range_check(val[j] as i64, 1, range[j]);
            }
        }

        let mut indices: Vec<usize> = (0..num_rows).collect();
        indices.shuffle(&mut rng);

        // Proves
        for i in 0..num_rows {
            // We take a random permutation of the indices to show that the permutation check is passing
            trace[i].c = trace[indices[i]].a;
            trace[i].d = trace[indices[i]].b;
        }

        let air_instance = AirInstance::new_from_trace(FromTrace::new(&mut trace));
        add_air_instance::<F>(air_instance, pctx.clone());
    }
}
