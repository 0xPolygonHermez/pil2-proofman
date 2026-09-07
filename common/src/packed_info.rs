use std::os::raw::c_void;
use serde::Serialize;

/// FFI mirror of the C++ `PackedInfo` (starks_api.hpp). Field order MUST match.
#[derive(Debug)]
#[repr(C)]
pub struct PackedInfoFFI {
    pub is_packed: bool,
    pub num_packed_words: u64,
    pub unpack_info: *mut u64, // raw pointer for C++
    pub col_source: *const u8, // per column: 0 = row, 1 = table; null if not indexed
    pub col_lane: *const u8,   // per column: lane whose index selects its entry; null if not indexed
    pub index_bits: u64,
    pub words_per_entry: u64,
    pub lanes: u64,
}

impl PackedInfoFFI {
    pub fn get_ptr(&self) -> *mut c_void {
        self as *const PackedInfoFFI as *mut c_void
    }
}

/// Safe Rust version
#[derive(Default, Debug, Clone, Serialize)]
pub struct PackedInfo {
    pub is_packed: bool,
    pub num_packed_words: u64,
    pub unpack_info: Vec<u64>,
    /// Per column source (0 = compact row, 1 = instruction table); empty if not indexed.
    pub col_source: Vec<u8>,
    /// Per column, the lane whose index selects the entry it is read from; 0 for
    /// row-sourced columns. Empty if not indexed.
    pub col_lane: Vec<u8>,
    /// Width of ONE instruction index in the compact row's header (bits).
    pub index_bits: u64,
    /// u64 words per instruction-table entry.
    pub words_per_entry: u64,
    /// Execution steps a row packs, i.e. indices in its header; 0 or 1 is single-lane.
    pub lanes: u64,
}

impl PackedInfo {
    pub fn new(is_packed: bool, num_packed_words: u64, unpack_info: Vec<u64>) -> Self {
        Self { is_packed, num_packed_words, unpack_info, ..Default::default() }
    }

    /// Attach the indexed-variant descriptor; `num_packed_words` must be the compact row size.
    /// `col_lane` must have one entry per column, like `col_source`.
    pub fn with_indexed(
        mut self,
        col_source: Vec<u8>,
        col_lane: Vec<u8>,
        index_bits: u64,
        words_per_entry: u64,
        lanes: u64,
    ) -> Self {
        assert_eq!(
            col_source.len(),
            col_lane.len(),
            "indexed descriptor: col_source and col_lane must both cover every column"
        );
        // The unpackers write a table column in the pass its lane names, so a column
        // tagged for a lane the row does not carry would be written by no pass at all.
        let n_lanes = lanes.max(1);
        assert!(
            n_lanes <= u8::MAX as u64,
            "indexed descriptor: at most {} lanes (col_lane is a u8)",
            u8::MAX
        );
        assert!(
            col_lane.iter().all(|&l| (l as u64) < n_lanes),
            "indexed descriptor: every col_lane must be below lanes ({n_lanes})"
        );
        // The unpackers read lane l's index at bit l * index_bits of the row, so the whole
        // header has to fit the compact row -- the CUDA ones read it unguarded.
        assert!(
            n_lanes * index_bits <= self.num_packed_words * 64,
            "indexed descriptor: a {n_lanes}-lane header of {index_bits}-bit indices does not \
             fit a {}-word compact row",
            self.num_packed_words
        );
        self.col_source = col_source;
        self.col_lane = col_lane;
        self.index_bits = index_bits;
        self.words_per_entry = words_per_entry;
        self.lanes = lanes;
        self
    }

    pub fn is_indexed(&self) -> bool {
        !self.col_source.is_empty()
    }

    pub fn as_ffi(&self) -> PackedInfoFFI {
        PackedInfoFFI {
            is_packed: self.is_packed,
            num_packed_words: self.num_packed_words,
            unpack_info: self.unpack_info.as_ptr() as *mut u64,
            // Empty Vec::as_ptr() is dangling-but-non-null; C++ null-checks this, so pass real null.
            col_source: if self.col_source.is_empty() { std::ptr::null() } else { self.col_source.as_ptr() },
            col_lane: if self.col_lane.is_empty() { std::ptr::null() } else { self.col_lane.as_ptr() },
            index_bits: self.index_bits,
            words_per_entry: self.words_per_entry,
            lanes: self.lanes,
        }
    }
}

/// Safe Rust version
#[derive(Default, Debug, Clone, Serialize)]
pub struct PackedInfoConst {
    pub is_packed: bool,
    pub num_packed_words: u64,
    pub unpack_info: &'static [u64],
}

impl PackedInfoConst {
    pub fn new(is_packed: bool, num_packed_words: u64, unpack_info: &'static [u64]) -> Self {
        Self { is_packed, num_packed_words, unpack_info }
    }
}
