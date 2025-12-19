pub struct Table4;

impl Table4 {
    const N: u64 = 128; // 2**7

    pub fn calculate_table_row(val: u64) -> u64 {
        (Self::N - 1) - val
    }
}
