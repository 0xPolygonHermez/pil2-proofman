use std::sync::{Arc, RwLock};

use fields::PrimeField64;

use proofman_common::{ProofCtx, SetupCtx, StdMode};

use crate::{StdProd, StdRangeCheck, StdSum};

pub struct Std<F: PrimeField64> {
    pub range_check: Arc<StdRangeCheck<F>>,
    pub std_prod: Arc<StdProd<F>>,
    pub std_sum: Arc<StdSum<F>>,
    pub std_mode: RwLock<StdMode>,
}

impl<F: PrimeField64> Std<F> {
    pub fn new(pctx: Arc<ProofCtx<F>>, sctx: Arc<SetupCtx<F>>) -> Arc<Self> {
        // Instantiate the STD components
        let std_prod = StdProd::new();
        let std_sum = StdSum::new();
        let range_check = StdRangeCheck::new(pctx.clone(), &sctx);

        Arc::new(Self { range_check, std_prod, std_sum, std_mode: RwLock::new(StdMode::default()) })
    }

    // Gets the range for the range check.
    pub fn get_range(&self, min: i64, max: i64, predefined: Option<bool>) -> usize {
        self.range_check.get_range(min, max, predefined)
    }

    // Processes the inputs for the range check.
    pub fn range_check(&self, val: i64, multiplicity: u64, id: usize) {
        self.range_check.assign_values(val, multiplicity, id);
    }
}
