use std::sync::Arc;

use pil_std_lib::Std;
use proofman::{WitnessComponent, WitnessManager};
use proofman_common::{AirInstance, ExecutionCtx, ProofCtx, SetupCtx};

use num_bigint::BigInt;
use num_traits::ToPrimitive;
use p3_field::PrimeField;
use rand::{distributions::Standard, prelude::Distribution, Rng};

use crate::{RangeCheckDynamic2Trace, RANGE_CHECK_DYNAMIC_2_AIRGROUP_ID, RANGE_CHECK_DYNAMIC_2_AIR_IDS};

pub struct RangeCheckDynamic2<F: PrimeField> {
    std_lib: Arc<Std<F>>,
}

impl<F: PrimeField> RangeCheckDynamic2<F>
where
    Standard: Distribution<F>,
{
    const MY_NAME: &'static str = "RngChDy2";

    pub fn new(wcm: Arc<WitnessManager<F>>, std_lib: Arc<Std<F>>) -> Arc<Self> {
        let range_check_dynamic2 = Arc::new(Self { std_lib });

        wcm.register_component(
            range_check_dynamic2.clone(),
            Some(RANGE_CHECK_DYNAMIC_2_AIRGROUP_ID),
            Some(RANGE_CHECK_DYNAMIC_2_AIR_IDS),
        );

        // Register dependency relations
        range_check_dynamic2.std_lib.register_predecessor();

        range_check_dynamic2
    }

    pub fn execute(&self, pctx: Arc<ProofCtx<F>>, ectx: Arc<ExecutionCtx>, sctx: Arc<SetupCtx>) {
        let mut rng = rand::thread_rng();
        let num_rows =
            pctx.global_info.airs[RANGE_CHECK_DYNAMIC_2_AIRGROUP_ID][RANGE_CHECK_DYNAMIC_2_AIR_IDS[0]].num_rows;
        let air = pctx.pilout.get_air(RANGE_CHECK_DYNAMIC_2_AIRGROUP_ID, RANGE_CHECK_DYNAMIC_2_AIR_IDS[0]);

        log::debug!(
            "{}: ··· Witness computation for AIR '{}' at stage 1",
            Self::MY_NAME,
            air.name().unwrap_or("unknown"),
        );

        let mut trace = RangeCheckDynamic2Trace::new_zeroes(num_rows);

        let range1 = self.std_lib.get_range(BigInt::from(5225), BigInt::from(29023), Some(false));
        let range2 = self.std_lib.get_range(BigInt::from(-8719), BigInt::from(-7269), Some(false));
        let range3 = self.std_lib.get_range(BigInt::from(-10), BigInt::from(10), Some(false));
        let range4 = self.std_lib.get_range(BigInt::from(0), BigInt::from((1 << 8) - 1), Some(false));
        let range5 = self.std_lib.get_range(BigInt::from(0), BigInt::from((1 << 7) - 1), Some(false));

        for i in 0..num_rows {
            let range = rng.gen_range(0..=4);

            match range {
                0 => {
                    trace[i].sel_1 = F::one();
                    trace[i].colu = F::from_canonical_u16(rng.gen_range(5225..=29023));

                    self.std_lib.range_check(trace[i].colu, F::one(), range1);
                }
                1 => {
                    trace[i].sel_2 = F::one();
                    let colu_val = rng.gen_range(-8719..=-7269) + F::order().to_i128().unwrap();
                    trace[i].colu = F::from_canonical_u64(colu_val as u64);

                    self.std_lib.range_check(trace[i].colu, F::one(), range2);
                }
                2 => {
                    trace[i].sel_3 = F::one();
                    let mut colu_val: i128 = rng.gen_range(-10..=10);
                    if colu_val < 0 {
                        colu_val += F::order().to_i128().unwrap();
                    }
                    trace[i].colu = F::from_canonical_u64(colu_val as u64);

                    self.std_lib.range_check(trace[i].colu, F::one(), range3);
                }
                3 => {
                    trace[i].sel_4 = F::one();
                    trace[i].colu = F::from_canonical_u32(rng.gen_range(0..=(1 << 8) - 1));

                    self.std_lib.range_check(trace[i].colu, F::one(), range4);
                }
                4 => {
                    trace[i].sel_5 = F::one();
                    trace[i].colu = F::from_canonical_u32(rng.gen_range(0..=(1 << 7) - 1));

                    self.std_lib.range_check(trace[i].colu, F::one(), range5);
                }
                _ => panic!("Invalid range"),
            }
        }

        let air_instance = AirInstance::new(
            sctx.clone(),
            RANGE_CHECK_DYNAMIC_2_AIRGROUP_ID,
            RANGE_CHECK_DYNAMIC_2_AIR_IDS[0],
            None,
            trace.buffer.unwrap(),
        );
        let (is_myne, gid) = ectx.dctx.write().unwrap().add_instance(
            RANGE_CHECK_DYNAMIC_2_AIRGROUP_ID,
            RANGE_CHECK_DYNAMIC_2_AIR_IDS[0],
            1,
        );
        if is_myne {
            pctx.air_instance_repo.add_air_instance(air_instance, Some(gid));
        }
    }
}

impl<F: PrimeField> WitnessComponent<F> for RangeCheckDynamic2<F>
where
    Standard: Distribution<F>,
{
    fn calculate_witness(
        &self,
        _stage: u32,
        _air_instance_id: Option<usize>,
        pctx: Arc<ProofCtx<F>>,
        _ectx: Arc<ExecutionCtx>,
        _sctx: Arc<SetupCtx>,
    ) {
        self.std_lib.unregister_predecessor(pctx, None);
    }
}
