pub struct Table2_3;

impl Table2_3 {
    pub const N: u64 = 32; // 2**5
    pub const OFFSET: u64 = 32_768; // 2**15

    pub fn calculate_table_row(val: u64) -> u64 {
        debug_assert!(val >= Self::OFFSET + Self::N, "Value must be greater than or equal to N");
        debug_assert!(val < Self::OFFSET + 2 * Self::N, "Value must be less than 2 * N");
        val - (Self::OFFSET + Self::N)
    }
}
