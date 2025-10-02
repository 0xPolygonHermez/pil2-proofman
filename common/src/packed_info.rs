use std::os::raw::c_void;

#[derive(Debug)]
#[repr(C)]
pub struct PackedInfoFFI {
    pub is_packed: bool,
    pub num_packed_words: u64,
    pub unpack_info: *mut u64, // raw pointer for C++
}

impl PackedInfoFFI {
    pub fn get_ptr(&self) -> *mut c_void {
        self as *const PackedInfoFFI as *mut c_void
    }
}

/// Safe Rust version
#[derive(Default, Debug, Clone)]
pub struct PackedInfo {
    pub is_packed: bool,
    pub num_packed_words: u64,
    pub unpack_info: Vec<u64>,
}

impl PackedInfo {
    pub fn new(is_packed: bool, num_packed_words: u64, unpack_info: Vec<u64>) -> Self {
        Self { is_packed, num_packed_words, unpack_info }
    }

    pub fn as_ffi(&self) -> PackedInfoFFI {
        PackedInfoFFI {
            is_packed: self.is_packed,
            num_packed_words: self.num_packed_words,
            unpack_info: self.unpack_info.as_ptr() as *mut u64,
        }
    }
}
