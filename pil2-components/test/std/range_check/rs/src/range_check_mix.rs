use std::sync::Arc;

use witness::{WitnessComponent, execute, define_wc_with_std};

use proofman_common::{BufferPool, FromTrace, AirInstance, ProofCtx, SetupCtx};

use fields::PrimeField64;
use rand::{Rng, SeedableRng, rngs::StdRng};
use crate::RangeCheckMixTrace;

define_wc_with_std!(RangeCheckMix, "RngChMix");

impl<F: PrimeField64> WitnessComponent<F> for RangeCheckMix<F> {
    execute!(RangeCheckMixTrace, 1);

    fn calculate_witness(
        &self,
        stage: u32,
        pctx: Arc<ProofCtx<F>>,
        _sctx: Arc<SetupCtx<F>>,
        instance_ids: &[usize],
        _n_cores: usize,
        buffer_pool: &dyn BufferPool<F>,
    ) {
        if stage == 1 {
            let mut rng = StdRng::seed_from_u64(self.seed.load(Ordering::Relaxed));

            let mut trace = RangeCheckMixTrace::new_from_vec(buffer_pool.take_buffer());
            let num_rows = trace.num_rows();

            tracing::debug!("··· Starting witness computation stage {}", 1);

            let range1 = self.std_lib.get_range_id(0, (1 << 8) - 1, Some(true));
            let range2 = self.std_lib.get_range_id(50, (1 << 7) - 1, Some(true));
            let range3 = self.std_lib.get_range_id(-1, 1 << 3, Some(true));
            let range4 = self.std_lib.get_range_id(-(1 << 7) + 1, -50, Some(true));

            let range5 = self.std_lib.get_range_id(0, (1 << 7) - 1, Some(false));
            let range6 = self.std_lib.get_range_id(0, (1 << 4) - 1, Some(false));
            let range7 = self.std_lib.get_range_id(1 << 5, (1 << 8) - 1, Some(false));
            let range8 = self.std_lib.get_range_id(1 << 8, (1 << 9) - 1, Some(false));

            let range9 = self.std_lib.get_range_id(5225, 29023, Some(false));
            // let range10 = self.std_lib.get_range_id(-8719, -7269, Some(false));
            let range11 = self.std_lib.get_range_id(-10, 10, Some(false));

            for i in 0..num_rows {
                // First interface
                let val0 = rng.random_range(0..=(1 << 8) - 1);
                trace[i].a[0] = F::from_u16(val0);
                self.std_lib.range_check(range1, val0 as i64, 1);

                let val1 = rng.random_range(50..=(1 << 7) - 1);
                trace[i].a[1] = F::from_u8(val1);
                self.std_lib.range_check(range2, val1 as i64, 1);

                let val2: i8 = rng.random_range(-1..=(1 << 3));
                trace[i].a[2] = if val2 < 0 {
                    F::from_u64((val2 as i128 + F::ORDER_U64 as i128) as u64)
                } else {
                    F::from_u8(val2 as u8)
                };
                self.std_lib.range_check(range3, val2 as i64, 1);

                let val3: i16 = rng.random_range(-(1 << 7) + 1..=-50);
                trace[i].a[3] = F::from_u64((val3 as i128 + F::ORDER_U64 as i128) as u64);
                self.std_lib.range_check(range4, val3 as i64, 1);

                // Second interface
                let range_selector1 = rng.random_bool(0.5);
                trace[i].range_sel[0] = F::from_bool(range_selector1);

                let range_selector2 = rng.random_bool(0.5);
                trace[i].range_sel[1] = F::from_bool(range_selector2);

                if range_selector1 {
                    let val = rng.random_range(0..=(1 << 7) - 1);
                    trace[i].b[0] = F::from_u16(val);

                    self.std_lib.range_check(range5, val as i64, 1);
                } else {
                    let val = rng.random_range(0..=(1 << 4) - 1);
                    trace[i].b[0] = F::from_u16(val);

                    self.std_lib.range_check(range6, val as i64, 1);
                }

                if range_selector2 {
                    let val = rng.random_range((1 << 5)..=(1 << 8) - 1);
                    trace[i].b[1] = F::from_u16(val);

                    self.std_lib.range_check(range7, val as i64, 1);
                } else {
                    let val = rng.random_range((1 << 8)..=(1 << 9) - 1);
                    trace[i].b[1] = F::from_u16(val);

                    self.std_lib.range_check(range8, val as i64, 1);
                }

                // Third interface
                let range = rng.random_range(0..=2);

                match range {
                    0 => {
                        trace[i].range_sel[2] = F::ONE;
                        trace[i].range_sel[3] = F::ZERO;
                        trace[i].range_sel[4] = F::ZERO;
                        let val = rng.random_range(5225..=29023);
                        trace[i].c[0] = F::from_u32(val);

                        self.std_lib.range_check(range9, val as i64, 1);
                    }
                    1 => {
                        trace[i].range_sel[2] = F::ZERO;
                        trace[i].range_sel[3] = F::ONE;
                        trace[i].range_sel[4] = F::ZERO;
                        let colu_val: i8 = rng.random_range(-10..=10);
                        trace[i].c[0] = if colu_val < 0 {
                            F::from_u64((colu_val as i128 + F::ORDER_U64 as i128) as u64)
                        } else {
                            F::from_u8(colu_val as u8)
                        };

                        self.std_lib.range_check(range11, colu_val as i64, 1);
                    }
                    2 => {
                        trace[i].range_sel[2] = F::ZERO;
                        trace[i].range_sel[3] = F::ZERO;
                        trace[i].range_sel[4] = F::ONE;
                        let val = rng.random_range(0..=(1 << 7) - 1);
                        trace[i].c[0] = F::from_u32(val);

                        self.std_lib.range_check(range5, val as i64, 1);
                    }
                    _ => panic!("Invalid range"),
                }
            }

            let air_instance = AirInstance::new_from_trace(FromTrace::new(&mut trace));
            pctx.add_air_instance(air_instance, instance_ids[0]);
        }
    }
}
