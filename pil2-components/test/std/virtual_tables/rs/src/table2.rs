pub struct Table2;

impl Table2 {
    const N: u64 = 16; // 2**4

    pub fn calculate_table_row(val: u64) -> u64 {
        (Self::N - 1) - val
    }
}
