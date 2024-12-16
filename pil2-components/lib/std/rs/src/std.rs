use std::sync::{
    atomic::{AtomicU32, Ordering},
    Arc,
};

use num_bigint::BigInt;
use p3_field::PrimeField;

use proofman::WitnessManager;

use crate::{AirComponent, StdProd, StdRangeCheck, StdSum};

pub struct Std<F: PrimeField> {
    // Data of components with public API
    range_check: Arc<StdRangeCheck<F>>,
    range_check_predecessors: AtomicU32,
}

impl<F: PrimeField> Std<F> {
    const MY_NAME: &'static str = "STD     ";

    pub fn new(wcm: Arc<WitnessManager<F>>) -> Arc<Self> {
        let mode = wcm.get_ectx().std_mode.clone();

        log::info!("{}: ··· The PIL2 STD library has been initialized on mode {}", Self::MY_NAME, mode.name);

        // Instantiate the STD components
        StdProd::new(wcm.clone(), Some(mode.clone()), None, None);
        StdSum::new(wcm.clone(), Some(mode.clone()), None, None);
        let range_check = StdRangeCheck::new(mode, wcm);

        Arc::new(Self { range_check, range_check_predecessors: AtomicU32::new(0) })
    }

    pub fn register_predecessor(&self) {
        self.range_check_predecessors.fetch_add(1, Ordering::SeqCst);
    }

    pub fn unregister_predecessor(&self) {
        if self.range_check_predecessors.fetch_sub(1, Ordering::SeqCst) == 1 {
            self.range_check.drain_inputs();
        }
    }

    // Gets the range for the range check.
    pub fn get_range(&self, min: BigInt, max: BigInt, predefined: Option<bool>) -> usize {
        self.range_check.get_range(min, max, predefined)
    }

    // Processes the inputs for the range check.
    pub fn range_check(&self, val: F, multiplicity: F, id: usize) {
        self.range_check.assign_values(val, multiplicity, id);
    }
}
