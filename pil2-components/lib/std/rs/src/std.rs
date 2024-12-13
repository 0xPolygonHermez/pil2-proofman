use std::sync::{
    atomic::{AtomicU32, Ordering},
    Arc,
};

use num_bigint::BigInt;
use p3_field::PrimeField;
use rayon::Scope;

use proofman::WitnessManager;
use proofman_common::ProofCtx;

use crate::{StdProd, StdRangeCheck, RangeCheckAir, StdSum};

pub struct Std<F: PrimeField> {
    range_check: Arc<StdRangeCheck<F>>,
    range_check_predecessors: AtomicU32,
}

impl<F: PrimeField> Std<F> {
    const MY_NAME: &'static str = "STD     ";

    pub fn new(wcm: Arc<WitnessManager<F>>) -> Arc<Self> {
        let mode = wcm.get_pctx().options.std_mode.clone();

        log::info!("{}: ··· The PIL2 STD library has been initialized on mode {}", Self::MY_NAME, mode.name);

        // Instantiate the STD components
        let _ = StdProd::new(mode.clone(), wcm.clone());
        let _ = StdSum::new(mode.clone(), wcm.clone());
        let range_check = StdRangeCheck::new(mode, wcm);

        Arc::new(Self { range_check, range_check_predecessors: AtomicU32::new(0) })
    }

    pub fn register_predecessor(&self) {
        self.range_check_predecessors.fetch_add(1, Ordering::SeqCst);
    }

    pub fn unregister_predecessor(&self, pctx: Arc<ProofCtx<F>>, scope: Option<&Scope>) {
        if self.range_check_predecessors.fetch_sub(1, Ordering::SeqCst) == 1 {
            self.range_check.drain_inputs(pctx, scope);
        }
    }

    /// Gets the range for the range check.
    pub fn get_range(&self, min: BigInt, max: BigInt, predefined: Option<bool>) -> usize {
        self.range_check.get_range(min, max, predefined)
    }

    /// Processes the inputs for the range check.
    pub fn range_check(&self, val: F, multiplicity: F, id: usize) {
        self.range_check.assign_values(val, multiplicity, id);
    }

    pub fn get_ranges(&self) -> Vec<(usize, usize, RangeCheckAir)> {
        self.range_check.get_ranges()
    }

    pub fn drain_inputs(&self, rc_type: &RangeCheckAir) {
        match rc_type {
            RangeCheckAir::U8Air => {
                self.range_check.u8air.as_ref().unwrap().drain_inputs();
            }
            RangeCheckAir::U16Air => {
                self.range_check.u16air.as_ref().unwrap().drain_inputs();
            }
            RangeCheckAir::SpecifiedRanges => {
                self.range_check.specified_ranges.as_ref().unwrap().drain_inputs();
            }
        };
    }
}
