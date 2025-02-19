use std::sync::Arc;

use pil_std_lib::Std;
use witness::WitnessComponent;

use proofman_common::{add_air_instance, FromTrace, AirInstance, ProofCtx};

use p3_field::PrimeField64;
use rand::{distributions::Standard, prelude::Distribution, Rng};

use crate::RangeCheck3Trace;

pub struct RangeCheck3<F: PrimeField64> {
    std_lib: Arc<Std<F>>,
}

impl<F: PrimeField64> RangeCheck3<F>
where
    Standard: Distribution<F>,
{
    const MY_NAME: &'static str = "RngChck3";

    pub fn new(std_lib: Arc<Std<F>>) -> Arc<Self> {
        Arc::new(Self { std_lib })
    }
}

impl<F: PrimeField64> WitnessComponent<F> for RangeCheck3<F>
where
    Standard: Distribution<F>,
{
    fn execute(&self, pctx: Arc<ProofCtx<F>>) {
        let mut rng = rand::thread_rng();

        let mut trace = RangeCheck3Trace::new_zeroes();
        let num_rows = trace.num_rows();

        log::debug!("{} ··· Starting witness computation stage {}", Self::MY_NAME, 1);

        let range1 = self.std_lib.get_range(0, (1 << 4) - 1, Some(false));
        let range2 = self.std_lib.get_range(0, (1 << 8) - 1, Some(false));

        for i in 0..num_rows {
            let val1 = rng.gen_range(0..=(1 << 4) - 1);
            let val2 = rng.gen_range(0..=(1 << 8) - 1);
            trace[i].c1 = F::from_canonical_u16(val1);
            trace[i].c2 = F::from_canonical_u16(val2);

            self.std_lib.range_check(val1 as i64, 1, range1);
            self.std_lib.range_check(val2 as i64, 1, range2);
        }

        let air_instance = AirInstance::new_from_trace(FromTrace::new(&mut trace));
        add_air_instance::<F>(air_instance, pctx.clone());
    }
}
