use std::sync::{Arc, RwLock};

use fields::PrimeField64;

use proofman_common::{ProofCtx, SetupCtx, StdMode};

use crate::{StdProd, StdRangeCheck, StdSum, StdVirtualTable};

pub struct Std<F: PrimeField64> {
    // STD mode
    pub mode: RwLock<StdMode>,

    // STD components
    pub prod_bus: Arc<StdProd<F>>,
    pub sum_bus: Arc<StdSum<F>>,
    pub range_check: Arc<StdRangeCheck<F>>,
    pub virtual_table: Arc<StdVirtualTable<F>>,
}

#[derive(Debug)]
pub enum VirtualTableIdInput {
    Id(usize),
    FromParams { min: i64, max: i64, predefined: Option<bool> },
}

impl<F: PrimeField64> Std<F> {
    pub fn new(pctx: Arc<ProofCtx<F>>, sctx: Arc<SetupCtx<F>>) -> Arc<Self> {
        // Get the mode
        let mode = RwLock::new(StdMode::default());

        // Instantiate the components
        let prod_bus = StdProd::new();
        let sum_bus = StdSum::new();
        let virtual_table = StdVirtualTable::new(pctx.clone(), &sctx);
        let range_check = StdRangeCheck::new(pctx.clone(), &sctx, virtual_table.clone());

        Arc::new(Self { mode, prod_bus, sum_bus, range_check, virtual_table })
    }

    /// Gets the range id for a given range subject to the range check
    pub fn get_range_id(&self, min: i64, max: i64, predefined: Option<bool>) -> usize {
        self.range_check.get_range_id(min, max, predefined)
    }

    /// Gets the virtual table ID for a given ID
    pub fn get_virtual_table_id(&self, id: usize) -> usize {
        self.virtual_table.get_id(id)
    }

    pub fn range_check(&self, id: usize, val: i64, multiplicity: u64) {
        self.range_check.assign_value(id, val, multiplicity);
    }

    pub fn range_checks(&self, id: usize, values: Vec<u32>) {
        self.range_check.assign_values(id, values);
    }

    pub fn inc_virtual_row(&self, id: usize, row: u64, multiplicity: u64) {
        self.virtual_table.inc_virtual_row(id, row, multiplicity);
    }

    pub fn inc_virtual_rows(&self, id: usize, rows: Vec<u64>, multiplicities: Vec<u32>) {
        #[cfg(all(debug_assertions, feature = "verify-rc-values"))]
        assert_eq!(rows.len(), multiplicities.len(), "Rows and multiplicities must have the same length");

        self.virtual_table.inc_virtual_rows(id, rows, multiplicities);
    }

    pub fn inc_virtual_rows_same_mul(&self, id: usize, rows: Vec<u64>, multiplicity: u64) {
        self.virtual_table.inc_virtual_rows_same_mul(id, rows, multiplicity);
    }
}
