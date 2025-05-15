// Goldilocks
extern "C" {
    #[link_name = "\u{1}_Z18goldilocks_add_ffiPKmS0_"]
    pub fn goldilocks_add_ffi(in1: *const u64, in2: *const u64) -> u64;
}
extern "C" {
    #[link_name = "\u{1}_Z25goldilocks_add_assign_ffiPmPKmS1_"]
    pub fn goldilocks_add_assign_ffi(result: *mut u64, in1: *const u64, in2: *const u64);
}
extern "C" {
    #[link_name = "\u{1}_Z18goldilocks_sub_ffiPKmS0_"]
    pub fn goldilocks_sub_ffi(in1: *const u64, in2: *const u64) -> u64;
}
extern "C" {
    #[link_name = "\u{1}_Z25goldilocks_sub_assign_ffiPmPKmS1_"]
    pub fn goldilocks_sub_assign_ffi(result: *mut u64, in1: *const u64, in2: *const u64);
}
extern "C" {
    #[link_name = "\u{1}_Z18goldilocks_mul_ffiPKmS0_"]
    pub fn goldilocks_mul_ffi(in1: *const u64, in2: *const u64) -> u64;
}
extern "C" {
    #[link_name = "\u{1}_Z25goldilocks_mul_assign_ffiPmPKmS1_"]
    pub fn goldilocks_mul_assign_ffi(result: *mut u64, in1: *const u64, in2: *const u64);
}
extern "C" {
    #[link_name = "\u{1}_Z18goldilocks_div_ffiPKmS0_"]
    pub fn goldilocks_div_ffi(in1: *const u64, in2: *const u64) -> u64;
}
extern "C" {
    #[link_name = "\u{1}_Z25goldilocks_div_assign_ffiPmPKmS1_"]
    pub fn goldilocks_div_assign_ffi(result: *mut u64, in1: *const u64, in2: *const u64);
}
extern "C" {
    #[link_name = "\u{1}_Z18goldilocks_neg_ffiPKm"]
    pub fn goldilocks_neg_ffi(in1: *const u64) -> u64;
}
extern "C" {
    #[link_name = "\u{1}_Z18goldilocks_inv_ffiPKm"]
    pub fn goldilocks_inv_ffi(in1: *const u64) -> u64;
}
