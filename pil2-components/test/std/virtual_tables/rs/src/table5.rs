pub struct Table5;

impl Table5 {
    const N: u64 = 64; // 2**6

    pub fn calculate_table_row(val: u64) -> u64 {
        (Self::N - 1) - val
    }
}
