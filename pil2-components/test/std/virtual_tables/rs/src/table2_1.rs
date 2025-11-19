pub struct Table2_1;

impl Table2_1 {
    pub const N: u64 = 16; // 2**4

    pub fn calculate_table_row(val: u64) -> u64 {
        debug_assert!(val < Self::N, "Value must be less than N");
        (Self::N - 1) - val
    }
}
