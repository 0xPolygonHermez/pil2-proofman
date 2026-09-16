use std::sync::{Arc, RwLock};

use proofman_fields::PrimeField64;

use proofman_common::{ProofCtx, ProofmanResult, SetupCtx, StdMode};

use crate::{StdProd, StdSum, StdVirtualTable};

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

/// Marker bit tagging a virtual table ID, so a raw index cannot be passed by mistake.
/// Assumes we will never have more than 2^62 virtual tables.
#[cfg(debug_assertions)]
const VIRTUAL_TABLE_MARKER: usize = 1 << 62;
#[cfg(debug_assertions)]
const ID_MASK: usize = !VIRTUAL_TABLE_MARKER;

pub struct Std<F: PrimeField64> {
    // STD mode
    pub mode: RwLock<StdMode>,

    // STD components
    pub prod_bus: Arc<StdProd<F>>,
    pub sum_bus: Arc<StdSum<F>>,
    pub virtual_table: Arc<StdVirtualTable<F>>,
}

impl<F: PrimeField64> Std<F> {
    pub fn new(pctx: Arc<ProofCtx<F>>, sctx: Arc<SetupCtx<F>>, shared_tables: bool) -> ProofmanResult<Arc<Self>> {
        // Get the mode
        let mode = RwLock::new(StdMode::default());

        // Instantiate the components
        let prod_bus = StdProd::new(&sctx)?;
        let sum_bus = StdSum::new(&sctx)?;
        let virtual_table = StdVirtualTable::new(&pctx, &sctx, shared_tables)?;

        Ok(Arc::new(Self { mode, prod_bus, sum_bus, virtual_table }))
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
        self.virtual_table.inc_virtual_rows(id, rows, muls);
    }

    /// Increments the multiplicity by 1 for each row in `rows`
    pub fn inc_virtual_row_batch_one<M: RCMultiplicity>(&self, id: usize, rows: &[M]) {
        let id = self.unwrap_virtual_table_id(id);
        self.virtual_table.inc_virtual_rows_same_mul(id, rows, 1);
    }

    /// Increments the multiplicities for multiple rows with the same multiplicity in the virtual table with id `id`
    pub fn inc_virtual_rows_same_mul<M: RCMultiplicity>(&self, id: usize, rows: &[M], mul: M) {
        let id = self.unwrap_virtual_table_id(id);
        self.virtual_table.inc_virtual_rows_same_mul(id, rows, mul.to_u64());
    }

    /// Increments the multiplicities of a list of rows `[start, start + N]` in the virtual table with id `id`.
    /// If `start` is `None`, then it is set to be 0
    pub fn inc_virtual_rows_ranged<M: RCMultiplicity>(&self, id: usize, start: Option<u64>, muls: &[M]) {
        let id = self.unwrap_virtual_table_id(id);
        self.virtual_table.inc_virtual_rows_ranged(id, start, muls);
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

/// Compile-only guard for the call shapes ZisK uses against `Std`. Nothing here runs; the
/// assertion is that it type-checks, so a change to these signatures (or to the widths the
/// generics accept) fails the build here instead of downstream.
#[cfg(test)]
#[allow(dead_code)]
fn zisk_call_shapes<F: PrimeField64>(std: &Arc<Std<F>>, id: usize) {
    // `&vec[a..b]` of u32 counts — precompiles cache
    let counts: Vec<u32> = vec![0; 64];

    // `&Vec<u64>` by reference, and an explicit u64 start — state-machines/binary binary_tally
    let spill: Vec<u64> = vec![0; 8];
    std.inc_virtual_row_batch_one(id, &spill);
    std.inc_virtual_rows_ranged(id, Some(3u64), &counts[..8]);

    // A borrowed `&Vec<u32>` straight from an iterator — precompiles/dma dma_pre_post
    let nested: Vec<Vec<u32>> = vec![vec![0; 4]];
    for muls in nested.iter() {
        std.inc_virtual_rows_ranged(id, None, muls);
    }

    // Non-ranged batches, mixed widths
    let muls32: Vec<u32> = vec![1; 8];
    std.inc_virtual_row_batch(id, &muls32, &muls32);
    std.inc_virtual_rows_same_mul(id, &muls32, 2u32);
}
