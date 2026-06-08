use std::sync::Arc;

use fields::PrimeField64;

use proofman_common::{ProofCtx, ProofmanResult, SetupCtx};

use crate::{StdProd, StdRangeCheck, StdSum, StdVirtualTable};

/// Trait for types that can be used as range check values
pub trait RCValue: Copy {
    fn to_i64(self) -> i64;
}

macro_rules! impl_range_value {
    ($($t:ty),*) => {
        $(impl RCValue for $t {
            #[inline(always)]
            fn to_i64(self) -> i64 { self as i64 }
        })*
    };
}

impl_range_value!(i8, i16, i32, i64, isize, u8, u16, u32, u64, usize);

/// Trait for types that can be used as multiplicities
pub trait RCMultiplicity: Copy {
    fn to_u64(self) -> u64;
}

macro_rules! impl_multiplicity {
    ($($t:ty),*) => {
        $(impl RCMultiplicity for $t {
            #[inline(always)]
            fn to_u64(self) -> u64 { self as u64 }
        })*
    };
}

impl_multiplicity!(u16, u32, u64, usize);

/// Marker bit to distinguish range check IDs from virtual table IDs
/// This assumes that we will never have more than 2^62 range checks or virtual tables
#[cfg(debug_assertions)]
const RANGE_CHECK_MARKER: usize = 1 << 63;
#[cfg(debug_assertions)]
const VIRTUAL_TABLE_MARKER: usize = 1 << 62;
#[cfg(debug_assertions)]
const ID_MASK: usize = !(RANGE_CHECK_MARKER | VIRTUAL_TABLE_MARKER);

pub struct Std<F: PrimeField64> {
    // STD components
    pub prod_bus: Arc<StdProd<F>>,
    pub sum_bus: Arc<StdSum<F>>,
    pub range_check: Arc<StdRangeCheck<F>>,
    pub virtual_table: Arc<StdVirtualTable<F>>,
}

impl<F: PrimeField64> Std<F> {
    pub fn new(pctx: Arc<ProofCtx<F>>, sctx: Arc<SetupCtx<F>>, shared_tables: bool) -> ProofmanResult<Arc<Self>> {
        // Instantiate the components
        let prod_bus = StdProd::new(&sctx)?;
        let sum_bus = StdSum::new(&sctx)?;
        let virtual_table = StdVirtualTable::new(&pctx, &sctx, shared_tables)?;
        let range_check = StdRangeCheck::new(pctx.clone(), &sctx, virtual_table.clone(), shared_tables)?;

        Ok(Arc::new(Self { prod_bus, sum_bus, range_check, virtual_table }))
    }

    // ==================== Range Check API ====================

    /// Gets the range id for a given range subject to the range check
    pub fn get_range_id<V: RCValue>(&self, min: V, max: V, predefined: Option<bool>) -> ProofmanResult<usize> {
        let id = self.range_check.get_range_id(min.to_i64(), max.to_i64(), predefined)?;

        #[cfg(debug_assertions)]
        let id = id | RANGE_CHECK_MARKER;

        Ok(id)
    }

    /// Increments the multiplicity `mul` of a given value `val` in the range check with id `id`
    pub fn range_check<V: RCValue, M: RCMultiplicity>(&self, id: usize, val: V, mul: M) {
        let id = self.unwrap_range_check_id(id);
        self.range_check.assign_value(id, val.to_i64(), mul.to_u64());
    }

    /// Increments the multiplicity of a given value `val` by 1
    pub fn range_check_one<V: RCValue>(&self, id: usize, val: V) {
        let id = self.unwrap_range_check_id(id);
        self.range_check.assign_value(id, val.to_i64(), 1);
    }

    /// Increments the multiplicities for multiple value/multiplicity pairs in the range check with id `id`
    pub fn range_check_batch<V: RCValue, M: RCMultiplicity>(&self, id: usize, vals: &[V], muls: &[M]) {
        let id = self.unwrap_range_check_id(id);
        let vals: Vec<i64> = vals.iter().map(|&v| v.to_i64()).collect();
        let muls: Vec<u64> = muls.iter().map(|&m| m.to_u64()).collect();
        self.range_check.assign_values(id, &vals, &muls);
    }

    /// Increments the multiplicity by 1 for each value in `vals`
    pub fn range_check_batch_one<V: RCValue>(&self, id: usize, vals: &[V]) {
        let id = self.unwrap_range_check_id(id);
        let vals: Vec<i64> = vals.iter().map(|&v| v.to_i64()).collect();
        self.range_check.assign_values_same_mul(id, &vals, 1);
    }

    /// Increments the multiplicities for multiple values with the same multiplicity in the range check with id `id`
    pub fn range_checks_same_mul<V: RCValue, M: RCMultiplicity>(&self, id: usize, vals: &[V], mul: M) {
        let id = self.unwrap_range_check_id(id);
        let vals: Vec<i64> = vals.iter().map(|&v| v.to_i64()).collect();
        self.range_check.assign_values_same_mul(id, &vals, mul.to_u64());
    }

    /// Increments the multiplicities of a list of values `[start, start + N]` in the range check with id `id`.
    /// If `start` is `None`, then it is set to be the minimum of the range
    pub fn range_check_ranged<M: RCMultiplicity>(&self, id: usize, start: Option<i64>, muls: &[M]) {
        let id = self.unwrap_range_check_id(id);
        let start = start.map(|s| s.to_i64());
        let muls: Vec<u64> = muls.iter().map(|&m| m.to_u64()).collect();
        self.range_check.assign_values_ranged(id, start, &muls)
    }

    #[inline(always)]
    fn unwrap_range_check_id(&self, id: usize) -> usize {
        #[cfg(debug_assertions)]
        {
            assert!(
                (id & RANGE_CHECK_MARKER) != 0,
                "Invalid range check ID: {}. Expected an ID from get_range_id().",
                id & ID_MASK
            );
            id & ID_MASK
        }

        #[cfg(not(debug_assertions))]
        {
            id
        }
    }

    // ==================== Virtual Table API ====================

    /// Gets the virtual table ID for a given ID
    pub fn get_virtual_table_id(&self, id: usize) -> ProofmanResult<usize> {
        let id = self.virtual_table.get_global_id(id)?;

        #[cfg(debug_assertions)]
        let id = id | VIRTUAL_TABLE_MARKER;

        Ok(id)
    }

    /// Increments the multiplicity `mul` of a given row `row` in the virtual table with id `id`
    pub fn inc_virtual_row<M: RCMultiplicity>(&self, id: usize, row: M, mul: M) {
        let id = self.unwrap_virtual_table_id(id);
        self.virtual_table.inc_virtual_row(id, row.to_u64(), mul.to_u64());
    }

    /// Increments the multiplicity of a given row `row` by 1
    pub fn inc_virtual_row_one<M: RCMultiplicity>(&self, id: usize, row: M) {
        let id = self.unwrap_virtual_table_id(id);
        self.virtual_table.inc_virtual_row(id, row.to_u64(), 1);
    }

    /// Increments the multiplicities for multiple row/multiplicity pairs in the virtual table with id `id`
    pub fn inc_virtual_row_batch<M: RCMultiplicity>(&self, id: usize, rows: &[M], muls: &[M]) {
        let id = self.unwrap_virtual_table_id(id);
        let rows: Vec<u64> = rows.iter().map(|&r| r.to_u64()).collect();
        let muls: Vec<u64> = muls.iter().map(|&m| m.to_u64()).collect();
        self.virtual_table.inc_virtual_rows(id, &rows, &muls);
    }

    /// Increments the multiplicity by 1 for each row in `rows`
    pub fn inc_virtual_row_batch_one<M: RCMultiplicity>(&self, id: usize, rows: &[M]) {
        let id = self.unwrap_virtual_table_id(id);
        let rows: Vec<u64> = rows.iter().map(|&r| r.to_u64()).collect();
        self.virtual_table.inc_virtual_rows_same_mul(id, &rows, 1);
    }

    /// Increments the multiplicities for multiple rows with the same multiplicity in the virtual table with id `id`
    pub fn inc_virtual_rows_same_mul<M: RCMultiplicity>(&self, id: usize, rows: &[M], mul: M) {
        let id = self.unwrap_virtual_table_id(id);
        let rows: Vec<u64> = rows.iter().map(|&r| r.to_u64()).collect();
        self.virtual_table.inc_virtual_rows_same_mul(id, &rows, mul.to_u64());
    }

    /// Increments the multiplicities of a list of rows `[start, start + N]` in the virtual table with id `id`.
    /// If `start` is `None`, then it is set to be 0
    pub fn inc_virtual_rows_ranged<M: RCMultiplicity>(&self, id: usize, start: Option<u64>, muls: &[M]) {
        let id = self.unwrap_virtual_table_id(id);
        let start = start.map(|s| s.to_u64());
        let muls: Vec<u64> = muls.iter().map(|&m| m.to_u64()).collect();
        self.virtual_table.inc_virtual_rows_ranged(id, start, &muls);
    }

    #[inline(always)]
    fn unwrap_virtual_table_id(&self, id: usize) -> usize {
        #[cfg(debug_assertions)]
        {
            assert!(
                (id & VIRTUAL_TABLE_MARKER) != 0,
                "Invalid virtual table ID: {}. Expected an ID from get_virtual_table_id()",
                id & ID_MASK
            );
            id & ID_MASK
        }

        #[cfg(not(debug_assertions))]
        {
            id
        }
    }
}
