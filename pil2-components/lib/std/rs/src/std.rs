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
        let range_check = StdRangeCheck::new(pctx.clone(), &sctx);
        let virtual_table = StdVirtualTable::new(pctx.clone(), &sctx);

        Arc::new(Self { mode, prod_bus, sum_bus, range_check, virtual_table })
    }

    // Gets the range id for the range check.
    pub fn get_range_id(&self, min: i64, max: i64, predefined: Option<bool>) -> usize {
        self.range_check.get_range_id(min, max, predefined)
    }

    // Processes the inputs for the range check.
    pub fn range_check(&self, id: usize, val: i64, multiplicity: u64) {
        self.range_check.assign_value(id, val, multiplicity);
    }

    pub fn range_checks(&self, id: usize, values: Vec<u32>) {
        self.range_check.assign_values(id, values);
    }

    pub fn range_check_virtual(&self, id: usize, val: i64, multiplicity: u64) {
        // Get the uid
        let uid = self.virtual_table.virtual_table_air.as_ref().unwrap().get_uid(id);

        // Get the range check id
        let rc_id = self.range_check.get_range_id_by_opid(uid as u64);

        // Get the row
        let rows = self.range_check.get_rows(rc_id, val);

        // Update the virtual table
        self.virtual_table.virtual_table_air.as_ref().unwrap().update_multiplicities(id, rows, multiplicity);
    }

    pub fn get_virtual_table_id(&self, input: VirtualTableIdInput) -> usize {
        let id = match input {
            VirtualTableIdInput::Id(id) => id,
            VirtualTableIdInput::FromParams { min, max, predefined } => {
                self.range_check.get_range_opid(min, max, predefined) as usize
            }
        };

        self.virtual_table.virtual_table_air.as_ref().unwrap().get_id(id)
    }

    pub fn inc_virtual_row(&self, id: usize, row: u64, multiplicity: u64) {
        self.virtual_table.virtual_table_air.as_ref().unwrap().update_multiplicity(id, row, multiplicity);
    }

    // pub fn inc_virtual_rows(&self, id: usize, rows: Vec<u64>) {
    //     self.virtual_table.update_multiplicities(id, rows);
    // }
}
