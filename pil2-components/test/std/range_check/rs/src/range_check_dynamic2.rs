use std::sync::Arc;

use witness::{WitnessComponent, execute, define_wc_with_std};

use proofman_common::{BufferPool, FromTrace, AirInstance, ProofCtx, SetupCtx};

use fields::PrimeField64;
use rand::{Rng, SeedableRng, rngs::StdRng};

use crate::RangeCheckDynamic2Trace;

define_wc_with_std!(RangeCheckDynamic2, "RngChDy2");

impl<F: PrimeField64> WitnessComponent<F> for RangeCheckDynamic2<F> {
    execute!(RangeCheckDynamic2Trace, 1);

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

            let mut trace = RangeCheckDynamic2Trace::new_from_vec(buffer_pool.take_buffer());
            let num_rows = trace.num_rows();

            tracing::debug!("··· Starting witness computation stage {}", 1);

            let range1 = self.std_lib.get_range_id(5225, 29023, Some(false));
            let range2 = self.std_lib.get_range_id(-8719, -7269, Some(false));
            let range3 = self.std_lib.get_range_id(-10, 10, Some(false));
            let range4 = self.std_lib.get_range_id(0, (1 << 8) - 1, Some(false));
            let range5 = self.std_lib.get_range_id(0, (1 << 7) - 1, Some(false));

            for i in 0..num_rows {
                let range = rng.random_range(0..=4);

                match range {
                    0 => {
                        trace[i].sel_1 = F::ONE;
                        trace[i].sel_2 = F::ZERO;
                        trace[i].sel_3 = F::ZERO;
                        trace[i].sel_4 = F::ZERO;
                        trace[i].sel_5 = F::ZERO;
                        let val = rng.random_range(5225..=29023);
                        trace[i].colu = F::from_u16(val);

                        self.std_lib.range_check(range1, val as i64, 1);
                    }
                    1 => {
                        trace[i].sel_1 = F::ZERO;
                        trace[i].sel_2 = F::ONE;
                        trace[i].sel_3 = F::ZERO;
                        trace[i].sel_4 = F::ZERO;
                        trace[i].sel_5 = F::ZERO;
                        let colu_val = rng.random_range(-8719..=-7269);
                        trace[i].colu = F::from_u64((colu_val as i128 + F::ORDER_U64 as i128) as u64);

                        self.std_lib.range_check(range2, colu_val as i64, 1);
                    }
                    2 => {
                        trace[i].sel_1 = F::ZERO;
                        trace[i].sel_2 = F::ZERO;
                        trace[i].sel_3 = F::ONE;
                        trace[i].sel_4 = F::ZERO;
                        trace[i].sel_5 = F::ZERO;
                        let colu_val: i8 = rng.random_range(-10..=10);
                        trace[i].colu = if colu_val < 0 {
                            F::from_u64((colu_val as i128 + F::ORDER_U64 as i128) as u64)
                        } else {
                            F::from_u8(colu_val as u8)
                        };

                        self.std_lib.range_check(range3, colu_val as i64, 1);
                    }
                    3 => {
                        trace[i].sel_1 = F::ZERO;
                        trace[i].sel_2 = F::ZERO;
                        trace[i].sel_3 = F::ZERO;
                        trace[i].sel_4 = F::ONE;
                        trace[i].sel_5 = F::ZERO;
                        let val = rng.random_range(0..=(1 << 8) - 1);
                        trace[i].colu = F::from_u32(val);

                        self.std_lib.range_check(range4, val as i64, 1);
                    }
                    4 => {
                        trace[i].sel_1 = F::ZERO;
                        trace[i].sel_2 = F::ZERO;
                        trace[i].sel_3 = F::ZERO;
                        trace[i].sel_4 = F::ZERO;
                        trace[i].sel_5 = F::ONE;
                        let val = rng.random_range(0..=(1 << 7) - 1);
                        trace[i].colu = F::from_u32(val);

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
