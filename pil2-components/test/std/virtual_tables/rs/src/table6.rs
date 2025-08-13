pub struct Table6;

impl Table6 {
    const N: u64 = 1024; // 2**10

    pub fn calculate_table_row(val: u64) -> u64 {
        (Self::N - 1) - val
    }
}
