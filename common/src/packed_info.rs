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
    pub index_bits: u64,
    pub words_per_entry: u64,
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
    /// Width of the compact row's leading instruction-index header (bits).
    pub index_bits: u64,
    /// u64 words per instruction-table entry.
    pub words_per_entry: u64,
}

impl PackedInfo {
    pub fn new(is_packed: bool, num_packed_words: u64, unpack_info: Vec<u64>) -> Self {
        Self { is_packed, num_packed_words, unpack_info, ..Default::default() }
    }

    /// Attach the indexed-variant descriptor; `num_packed_words` must be the compact row size.
    pub fn with_indexed(mut self, col_source: Vec<u8>, index_bits: u64, words_per_entry: u64) -> Self {
        self.col_source = col_source;
        self.index_bits = index_bits;
        self.words_per_entry = words_per_entry;
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
            index_bits: self.index_bits,
            words_per_entry: self.words_per_entry,
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
