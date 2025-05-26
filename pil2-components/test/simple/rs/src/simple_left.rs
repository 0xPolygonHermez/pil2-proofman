use std::sync::Arc;

use witness::{WitnessComponent, execute, define_wc_with_std};
use proofman_common::{FromTrace, AirInstance, ProofCtx, SetupCtx};

use p3_field::PrimeField64;
use rand::{
    distr::{Distribution, StandardUniform},
    rngs::StdRng,
    seq::SliceRandom,
    Rng, SeedableRng,
};

use crate::SimpleLeftTrace;

define_wc_with_std!(SimpleLeft, "SimLeft ");

impl<F: PrimeField64> WitnessComponent<F> for SimpleLeft<F>
where
    StandardUniform: Distribution<F>,
{
    execute!(SimpleLeftTrace, 1);

    fn calculate_witness(&self, stage: u32, pctx: Arc<ProofCtx<F>>, _sctx: Arc<SetupCtx<F>>, instance_ids: &[usize]) {
        if stage == 1 {
            let mut rng = StdRng::seed_from_u64(self.seed.load(Ordering::Relaxed));

            let mut trace = SimpleLeftTrace::new();
            let num_rows = trace.num_rows();

            tracing::debug!("··· Starting witness computation stage {}", 1);

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
                trace[i].a = F::from_u64(rng.random_range(0..=(1 << 63) - 1));
                trace[i].b = F::from_u64(rng.random_range(0..=(1 << 63) - 1));

                trace[i].e = F::from_u8(200);
                trace[i].f = F::from_u8(201);

                trace[i].g = F::from_usize(i);
                trace[i].h = F::from_usize(num_rows - i - 1);

                let val = [
                    rng.random_range(0..=(1 << 8) - 1),
                    rng.random_range(0..=(1 << 16) - 1),
                    rng.random_range(1..=(1 << 8) - 1),
                    rng.random_range(0..=(1 << 8)),
                    rng.random_range(0..=(1 << 8) - 1),
                    rng.random_range(-(1 << 7)..-1),
                    rng.random_range(-(1 << 7) - 1..(1 << 7) - 1),
                ];

                for j in 0..7 {
                    // Specific values for specific ranges
                    if j == 4 {
                        if i == 0 {
                            let val = 0;
                            trace[i].k[j] = F::from_u32(val);
                            self.std_lib.range_check(val as i64, 1, range[j]);
                            continue;
                        } else if i == 1 {
                            let val = 1 << 4;
                            trace[i].k[j] = F::from_u32(val);
                            self.std_lib.range_check(val as i64, 1, range[j]);
                            continue;
                        } else if i == 2 {
                            let val = (1 << 8) - 1;
                            trace[i].k[j] = F::from_u32(val);
                            self.std_lib.range_check(val as i64, 1, range[j]);
                            continue;
                        }
                    } else if j == 5 {
                        if i == 0 {
                            let val = -(1 << 7);
                            trace[i].k[j] = F::from_u64((val as i128 + F::ORDER_U64 as i128) as u64);
                            self.std_lib.range_check(val as i64, 1, range[j]);
                            continue;
                        } else if i == 1 {
                            let val = -(1 << 2);
                            trace[i].k[j] = F::from_u64((val as i128 + F::ORDER_U64 as i128) as u64);
                            self.std_lib.range_check(val as i64, 1, range[j]);
                            continue;
                        } else if i == 2 {
                            let val = -1;
                            trace[i].k[j] = F::from_u64((val as i128 + F::ORDER_U64 as i128) as u64);
                            self.std_lib.range_check(val as i64, 1, range[j]);
                            continue;
                        }
                    } else if j == 6 {
                        if i == 0 {
                            let val = -(1 << 7) - 1;
                            trace[i].k[j] = F::from_u64((val as i128 + F::ORDER_U64 as i128) as u64);
                            self.std_lib.range_check(val as i64, 1, range[j]);
                            continue;
                        } else if i == 1 {
                            let val = -(1 << 2);
                            trace[i].k[j] = F::from_u64((val as i128 + F::ORDER_U64 as i128) as u64);
                            self.std_lib.range_check(val as i64, 1, range[j]);
                            continue;
                        } else if i == 2 {
                            let val = -1;
                            trace[i].k[j] = F::from_u64((val as i128 + F::ORDER_U64 as i128) as u64);
                            self.std_lib.range_check(val as i64, 1, range[j]);
                            continue;
                        } else if i == 3 {
                            let val = 0;
                            trace[i].k[j] = F::from_u32(val);
                            self.std_lib.range_check(val as i64, 1, range[j]);
                            continue;
                        } else if i == 4 {
                            let val = (1 << 7) - 1;
                            trace[i].k[j] = F::from_u32(val);
                            self.std_lib.range_check(val as i64, 1, range[j]);
                            continue;
                        } else if i == 5 {
                            let val = 10;
                            trace[i].k[j] = F::from_u32(val);
                            self.std_lib.range_check(val as i64, 1, range[j]);
                            continue;
                        }
                    }

                    trace[i].k[j] = if val[j] < 0 {
                        F::from_u64((val[j] as i128 + F::ORDER_U64 as i128) as u64)
                    } else {
                        F::from_u32(val[j] as u32)
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
            pctx.add_air_instance(air_instance, instance_ids[0]);
        }
    }
}
