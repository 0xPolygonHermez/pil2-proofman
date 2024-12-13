use std::sync::Arc;

use pil_std_lib::Std;
use proofman::{WitnessComponent, WitnessManager};
use proofman_common::{add_air_instance, FromTrace, AirInstance, ProofCtx};

use num_bigint::BigInt;
use p3_field::PrimeField;
use rand::{distributions::Standard, prelude::Distribution, Rng};

use crate::RangeCheckDynamic1Trace;

pub struct RangeCheckDynamic1<F: PrimeField> {
    std_lib: Arc<Std<F>>,
}

impl<F: PrimeField> RangeCheckDynamic1<F>
where
    Standard: Distribution<F>,
{
    const MY_NAME: &'static str = "RngChDy1";

    pub fn new(wcm: Arc<WitnessManager<F>>, std_lib: Arc<Std<F>>) -> Arc<Self> {
        let range_check_dynamic1 = Arc::new(Self { std_lib });

        wcm.register_component(range_check_dynamic1.clone());

        // Register dependency relations
        range_check_dynamic1.std_lib.register_predecessor();

        range_check_dynamic1
    }

    pub fn execute(&self, pctx: Arc<ProofCtx<F>>) {
        let mut rng = rand::thread_rng();

        let mut trace = RangeCheckDynamic1Trace::new_zeroes();
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
                    trace[i].colu = F::from_canonical_u16(rng.gen_range(0..=(1 << 7) - 1));

                    self.std_lib.range_check(trace[i].colu, F::one(), range7);
                }
                1 => {
                    trace[i].sel_8 = F::one();
                    trace[i].colu = F::from_canonical_u16(rng.gen_range(0..=(1 << 8) - 1));

                    self.std_lib.range_check(trace[i].colu, F::one(), range8);
                }
                2 => {
                    trace[i].sel_16 = F::one();
                    trace[i].colu = F::from_canonical_u32(rng.gen_range(0..=(1 << 16) - 1));

                    self.std_lib.range_check(trace[i].colu, F::one(), range16);
                }
                3 => {
                    trace[i].sel_17 = F::one();
                    trace[i].colu = F::from_canonical_u32(rng.gen_range(0..=(1 << 17) - 1));

                    self.std_lib.range_check(trace[i].colu, F::one(), range17);
                }
                _ => panic!("Invalid range"),
            }
        }

        let air_instance = AirInstance::new_from_trace(FromTrace::new(&mut trace));
        add_air_instance::<F>(air_instance, pctx.clone());
        self.std_lib.unregister_predecessor(pctx, None);
    }
}

impl<F: PrimeField> WitnessComponent<F> for RangeCheckDynamic1<F> where Standard: Distribution<F> {}
