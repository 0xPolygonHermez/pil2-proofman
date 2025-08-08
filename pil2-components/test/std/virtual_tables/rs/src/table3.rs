pub struct Table3;

impl Table3 {
    const N: u64 = 4; // 2**2

    pub fn calculate_table_row(val: u64) -> u64 {
        (Self::N - 1) - val
    }
}
