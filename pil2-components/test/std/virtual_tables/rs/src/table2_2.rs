pub struct Table2_2;

impl Table2_2 {
    pub const N: u64 = 32_768; // 2**15

    pub fn calculate_table_row(val: u64) -> u64 {
        debug_assert!(val >= Self::N, "Value must be greater than or equal to N");
        debug_assert!(val < 2 * Self::N, "Value must be less than 2 * N");
        val - Self::N
    }
}
