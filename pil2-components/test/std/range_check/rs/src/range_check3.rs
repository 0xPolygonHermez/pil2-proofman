use std::sync::{Arc, RwLock};

use pil_std_lib::Std;
use witness::WitnessComponent;

use proofman_common::{FromTrace, AirInstance, ProofCtx, SetupCtx};

use num_bigint::BigInt;
use p3_field::PrimeField;
use rand::{distributions::Standard, prelude::Distribution, Rng, SeedableRng, rngs::StdRng};

use crate::RangeCheck3Trace;

pub struct RangeCheck3<F: PrimeField> {
    instance_ids: RwLock<Vec<usize>>,
    std_lib: Arc<Std<F>>,
}

impl<F: PrimeField> RangeCheck3<F>
where
    Standard: Distribution<F>,
{
    const MY_NAME: &'static str = "RngChck3";

    pub fn new(std_lib: Arc<Std<F>>) -> Arc<Self> {
        Arc::new(Self { std_lib, instance_ids: RwLock::new(Vec::new()) })
    }
}

impl<F: PrimeField> WitnessComponent<F> for RangeCheck3<F>
where
    Standard: Distribution<F>,
{
    fn execute(&self, pctx: Arc<ProofCtx<F>>) -> Vec<usize> {
        let global_ids =
            vec![pctx.add_instance(RangeCheck3Trace::<usize>::AIRGROUP_ID, RangeCheck3Trace::<usize>::AIR_ID)];
        self.instance_ids.write().unwrap().push(global_ids[0]);
        global_ids
    }

    fn calculate_witness(&self, stage: u32, pctx: Arc<ProofCtx<F>>, _sctx: Arc<SetupCtx<F>>, instance_ids: &[usize]) {
        if stage == 1 {
            let seed = if cfg!(feature = "debug") { 0 } else { rand::thread_rng().gen::<u64>() };
            let mut rng = StdRng::seed_from_u64(seed);
            let mut trace = RangeCheck3Trace::new();
            let num_rows = trace.num_rows();

            log::debug!("{} ··· Starting witness computation stage {}", Self::MY_NAME, 1);

            let range1 = self.std_lib.get_range(BigInt::from(0), BigInt::from((1 << 4) - 1), Some(false));
            let range2 = self.std_lib.get_range(BigInt::from(0), BigInt::from((1 << 8) - 1), Some(false));

            for i in 0..num_rows {
                trace[i].c1 = F::from_canonical_u16(rng.gen_range(0..=(1 << 4) - 1));
                trace[i].c2 = F::from_canonical_u16(rng.gen_range(0..=(1 << 8) - 1));

                self.std_lib.range_check(trace[i].c1, F::one(), range1);
                self.std_lib.range_check(trace[i].c2, F::one(), range2);
            }

            let air_instance = AirInstance::new_from_trace(FromTrace::new(&mut trace));
            pctx.add_air_instance(air_instance, instance_ids[0]);
        }
    }
}
