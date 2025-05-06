use std::sync::Arc;

use p3_field::PrimeField64;

use witness::WitnessManager;

use crate::{StdProd, StdRangeCheck, StdSum};

pub struct Std<F: PrimeField64> {
    pub range_check: Arc<StdRangeCheck<F>>,
    pub std_prod: Arc<StdProd<F>>,
    pub std_sum: Arc<StdSum<F>>,
}

impl<F: PrimeField64> Std<F> {
    const MY_NAME: &'static str = "STD     ";

    pub fn new(wcm: Arc<WitnessManager<F>>) -> Arc<Self> {
        let pctx = wcm.get_pctx();
        let sctx = wcm.get_sctx();

        let std_mode = pctx.options.debug_info.std_mode.clone();
        log::info!("{}: ··· The PIL2 STD library has been initialized on mode {}", Self::MY_NAME, std_mode.name);

        // Instantiate the STD components
        let std_prod = StdProd::new();
        let std_sum = StdSum::new();
        let range_check = StdRangeCheck::new(pctx.clone(), &sctx);

        wcm.register_component_std(std_prod.clone());
        wcm.register_component_std(std_sum.clone());
        wcm.register_component_std(range_check.clone());

        if range_check.u8air.is_some() {
            wcm.register_component_std(range_check.u8air.clone().unwrap());
        }

        if range_check.u16air.is_some() {
            wcm.register_component_std(range_check.u16air.clone().unwrap());
        }

        if range_check.specified_ranges.is_some() {
            wcm.register_component_std(range_check.specified_ranges.clone().unwrap());
        }

        Arc::new(Self { range_check, std_prod, std_sum })
    }

    pub fn new_dev(wcm: Arc<WitnessManager<F>>, register_u8: bool, register_u16: bool, register_specified_ranges: bool) -> Arc<Self> {
        let pctx = wcm.get_pctx();
        let sctx = wcm.get_sctx();

        let std_mode = pctx.options.debug_info.std_mode.clone();
        log::info!("{}: ··· The PIL2 STD library has been initialized on mode {}", Self::MY_NAME, std_mode.name);

        // Instantiate the STD components
        let std_prod = StdProd::new();
        let std_sum = StdSum::new();
        let range_check = StdRangeCheck::new(pctx.clone(), &sctx);

        wcm.register_component_std(std_prod.clone());
        wcm.register_component_std(std_sum.clone());
        wcm.register_component_std(range_check.clone());

        if register_u8 && range_check.u8air.is_some() {
            wcm.register_component_std(range_check.u8air.clone().unwrap());
        }

        if register_u16 && range_check.u16air.is_some() {
            wcm.register_component_std(range_check.u16air.clone().unwrap());
        }

        if register_specified_ranges && range_check.specified_ranges.is_some() {
            wcm.register_component_std(range_check.specified_ranges.clone().unwrap());
        }

        Arc::new(Self { range_check, std_prod, std_sum })
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
