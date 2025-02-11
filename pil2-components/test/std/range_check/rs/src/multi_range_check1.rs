use std::sync::{Arc, RwLock};

use pil_std_lib::Std;
use witness::WitnessComponent;

use proofman_common::{FromTrace, AirInstance, ProofCtx, SetupCtx};

use num_bigint::BigInt;
use p3_field::PrimeField;
use rand::{distributions::Standard, prelude::Distribution, Rng, SeedableRng, rngs::StdRng};

use crate::MultiRangeCheck1Trace;

pub struct MultiRangeCheck1<F: PrimeField> {
    std_lib: Arc<Std<F>>,
    instance_ids: RwLock<Vec<usize>>,
}

impl<F: PrimeField> MultiRangeCheck1<F>
where
    Standard: Distribution<F>,
{
    const MY_NAME: &'static str = "MtRngCh1";

    pub fn new(std_lib: Arc<Std<F>>) -> Arc<Self> {
        Arc::new(Self { std_lib, instance_ids: RwLock::new(Vec::new()) })
    }
}

impl<F: PrimeField> WitnessComponent<F> for MultiRangeCheck1<F>
where
    Standard: Distribution<F>,
{
    fn execute(&self, pctx: Arc<ProofCtx<F>>) -> Vec<usize> {
        let global_ids = vec![
            pctx.add_instance(MultiRangeCheck1Trace::<usize>::AIRGROUP_ID, MultiRangeCheck1Trace::<usize>::AIR_ID)
        ];
        self.instance_ids.write().unwrap().push(global_ids[0]);
        global_ids
    }

    fn calculate_witness(&self, stage: u32, pctx: Arc<ProofCtx<F>>, _sctx: Arc<SetupCtx<F>>, instance_ids: &[usize]) {
        if stage == 1 {
            let seed = if cfg!(feature = "debug") { 0 } else { rand::thread_rng().gen::<u64>() };
            let mut rng = StdRng::seed_from_u64(seed);

            let mut trace = MultiRangeCheck1Trace::new();
            let num_rows = trace.num_rows();

            log::debug!("{} ··· Starting witness computation stage {}", Self::MY_NAME, 1);

            let range1 = self.std_lib.get_range(BigInt::from(0), BigInt::from((1 << 7) - 1), Some(false));
            let range2 = self.std_lib.get_range(BigInt::from(0), BigInt::from((1 << 8) - 1), Some(false));
            let range3 = self.std_lib.get_range(BigInt::from(0), BigInt::from((1 << 6) - 1), Some(false));
            let range4 = self.std_lib.get_range(BigInt::from(1 << 5), BigInt::from((1 << 8) - 1), Some(false));
            let range5 = self.std_lib.get_range(BigInt::from(1 << 8), BigInt::from((1 << 9) - 1), Some(false));

            for i in 0..num_rows {
                let selected1 = rng.gen::<bool>();
                let range_selector1 = rng.gen::<bool>();
                trace[i].sel[0] = F::from_bool(selected1);
                trace[i].range_sel[0] = F::from_bool(range_selector1);

                let selected2 = rng.gen::<bool>();
                let range_selector2 = rng.gen::<bool>();
                trace[i].sel[1] = F::from_bool(selected2);
                trace[i].range_sel[1] = F::from_bool(range_selector2);

                let selected3 = rng.gen::<bool>();
                let range_selector3 = rng.gen::<bool>();
                trace[i].sel[2] = F::from_bool(selected3);
                trace[i].range_sel[2] = F::from_bool(range_selector3);

                trace[i].a[0] = F::zero();
                trace[i].a[1] = F::zero();
                trace[i].a[2] = F::zero();

                if selected1 {
                    if range_selector1 {
                        trace[i].a[0] = F::from_canonical_u16(rng.gen_range(0..=(1 << 7) - 1));

                        self.std_lib.range_check(trace[i].a[0], F::one(), range1);
                    } else {
                        trace[i].a[0] = F::from_canonical_u16(rng.gen_range(0..=(1 << 8) - 1));

                        self.std_lib.range_check(trace[i].a[0], F::one(), range2);
                    }
                }

                if selected2 {
                    if range_selector2 {
                        trace[i].a[1] = F::from_canonical_u16(rng.gen_range(0..=(1 << 7) - 1));

                        self.std_lib.range_check(trace[i].a[1], F::one(), range1);
                    } else {
                        trace[i].a[1] = F::from_canonical_u16(rng.gen_range(0..=(1 << 6) - 1));

                        self.std_lib.range_check(trace[i].a[1], F::one(), range3);
                    }
                }

                if selected3 {
                    if range_selector3 {
                        trace[i].a[2] = F::from_canonical_u16(rng.gen_range((1 << 5)..=(1 << 8) - 1));

                        self.std_lib.range_check(trace[i].a[2], F::one(), range4);
                    } else {
                        trace[i].a[2] = F::from_canonical_u16(rng.gen_range((1 << 8)..=(1 << 9) - 1));

                        self.std_lib.range_check(trace[i].a[2], F::one(), range5);
                    }
                }
            }

            let air_instance = AirInstance::new_from_trace(FromTrace::new(&mut trace));
            pctx.add_air_instance(air_instance, instance_ids[0]);
        }
    }
}
