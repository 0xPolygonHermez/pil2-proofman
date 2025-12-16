pub struct Table8;

const P2_2: u8 = 1 << 2;
const P2_4: u8 = 1 << 4;

impl Table8 {
    const N: u64 = 48;

    pub fn calculate_table_row(a: u8, b: u8, c: u8) -> u64 {
        let offset_a = a;
        let offset_b = b * P2_2;
        let offset_c = c * P2_4;

        let row = (offset_a + offset_b + offset_c) as u64;
        debug_assert!(row < Self::N, "Row index out of bounds: {}", row);

        row
    }
}
