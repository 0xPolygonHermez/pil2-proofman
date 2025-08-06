// Goldilocks
extern "C" {
    pub fn goldilocks_add_ffi(in1: *const u64, in2: *const u64) -> u64;
    pub fn goldilocks_add_assign_ffi(result: *mut u64, in1: *const u64, in2: *const u64);
    pub fn goldilocks_sub_ffi(in1: *const u64, in2: *const u64) -> u64;
    pub fn goldilocks_sub_assign_ffi(result: *mut u64, in1: *const u64, in2: *const u64);
    pub fn goldilocks_mul_ffi(in1: *const u64, in2: *const u64) -> u64;
    pub fn goldilocks_mul_assign_ffi(result: *mut u64, in1: *const u64, in2: *const u64);
    pub fn goldilocks_div_ffi(in1: *const u64, in2: *const u64) -> u64;
    pub fn goldilocks_div_assign_ffi(result: *mut u64, in1: *const u64, in2: *const u64);
    pub fn goldilocks_neg_ffi(in1: *const u64) -> u64;
    pub fn goldilocks_inv_ffi(in1: *const u64) -> u64;
}
