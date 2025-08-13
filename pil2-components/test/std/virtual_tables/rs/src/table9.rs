pub struct Table9;

impl Table9 {
    const N: u64 = 1_048_576; // 2**20

    pub fn calculate_table_row(val: u64) -> u64 {
        (Self::N - 1) - val
    }
}
