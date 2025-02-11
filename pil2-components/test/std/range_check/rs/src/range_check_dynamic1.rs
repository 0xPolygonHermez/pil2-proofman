use std::sync::{Arc, RwLock};

use pil_std_lib::Std;
use witness::WitnessComponent;

use proofman_common::{FromTrace, AirInstance, ProofCtx, SetupCtx};

use num_bigint::BigInt;
use p3_field::PrimeField;
use rand::{distributions::Standard, prelude::Distribution, Rng, SeedableRng, rngs::StdRng};

use crate::RangeCheckDynamic1Trace;

pub struct RangeCheckDynamic1<F: PrimeField> {
    instance_ids: RwLock<Vec<usize>>,
    std_lib: Arc<Std<F>>,
}

impl<F: PrimeField> RangeCheckDynamic1<F>
where
    Standard: Distribution<F>,
{
    const MY_NAME: &'static str = "RngChDy1";

    pub fn new(std_lib: Arc<Std<F>>) -> Arc<Self> {
        Arc::new(Self { std_lib, instance_ids: RwLock::new(Vec::new()) })
    }
}

impl<F: PrimeField> WitnessComponent<F> for RangeCheckDynamic1<F>
where
    Standard: Distribution<F>,
{
    fn execute(&self, pctx: Arc<ProofCtx<F>>) -> Vec<usize> {
        let global_ids =
            vec![pctx
                .add_instance(RangeCheckDynamic1Trace::<usize>::AIRGROUP_ID, RangeCheckDynamic1Trace::<usize>::AIR_ID)];
        self.instance_ids.write().unwrap().push(global_ids[0]);
        global_ids
    }

    fn calculate_witness(&self, stage: u32, pctx: Arc<ProofCtx<F>>, _sctx: Arc<SetupCtx<F>>, instance_ids: &[usize]) {
        if stage == 1 {
            let seed = if cfg!(feature = "debug") { 0 } else { rand::thread_rng().gen::<u64>() };
            let mut rng = StdRng::seed_from_u64(seed);

            let mut trace = RangeCheckDynamic1Trace::new();
            let num_rows = trace.num_rows();

            log::debug!("{} ··· Starting witness computation stage {}", Self::MY_NAME, 1);

            let range7 = self.std_lib.get_range(BigInt::from(0), BigInt::from((1 << 7) - 1), Some(false));
            let range8 = self.std_lib.get_range(BigInt::from(0), BigInt::from((1 << 8) - 1), Some(false));
            let range16 = self.std_lib.get_range(BigInt::from(0), BigInt::from((1 << 16) - 1), Some(false));
            let range17 = self.std_lib.get_range(BigInt::from(0), BigInt::from((1 << 17) - 1), Some(false));

            for i in 0..num_rows {
                let range = rng.gen_range(0..=3);

                match range {
                    0 => {
                        trace[i].sel_7 = F::one();
                        trace[i].sel_8 = F::zero();
                        trace[i].sel_16 = F::zero();
                        trace[i].sel_17 = F::zero();
                        trace[i].colu = F::from_canonical_u16(rng.gen_range(0..=(1 << 7) - 1));

                        self.std_lib.range_check(trace[i].colu, F::one(), range7);
                    }
                    1 => {
                        trace[i].sel_7 = F::zero();
                        trace[i].sel_8 = F::one();
                        trace[i].sel_16 = F::zero();
                        trace[i].sel_17 = F::zero();
                        trace[i].colu = F::from_canonical_u16(rng.gen_range(0..=(1 << 8) - 1));

                        self.std_lib.range_check(trace[i].colu, F::one(), range8);
                    }
                    2 => {
                        trace[i].sel_7 = F::zero();
                        trace[i].sel_8 = F::zero();
                        trace[i].sel_16 = F::one();
                        trace[i].sel_17 = F::zero();
                        trace[i].colu = F::from_canonical_u32(rng.gen_range(0..=(1 << 16) - 1));

                        self.std_lib.range_check(trace[i].colu, F::one(), range16);
                    }
                    3 => {
                        trace[i].sel_7 = F::zero();
                        trace[i].sel_8 = F::zero();
                        trace[i].sel_16 = F::zero();
                        trace[i].sel_17 = F::one();
                        trace[i].colu = F::from_canonical_u32(rng.gen_range(0..=(1 << 17) - 1));

                        self.std_lib.range_check(trace[i].colu, F::one(), range17);
                    }
                    _ => panic!("Invalid range"),
                }
            }

            let air_instance = AirInstance::new_from_trace(FromTrace::new(&mut trace));
            pctx.add_air_instance(air_instance, instance_ids[0]);
        }
    }
}
