pub struct Table1;

impl Table1 {
    const N: u64 = 32; // 2**5

    pub fn calculate_table_row(val: u64) -> u64 {
        (Self::N - 1) - val
    }
}
