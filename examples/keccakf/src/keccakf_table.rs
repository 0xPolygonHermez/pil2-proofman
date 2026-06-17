use super::TABLE_SIZE;

pub struct KeccakfTable;

impl KeccakfTable {
    pub const TABLE_ID: usize = 126;

    /// Calculates the table row offset based on the provided parameters.
    ///
    /// # Arguments
    /// * `a` - The input value used to calculate the table row.
    ///
    /// # Returns
    /// The calculated table row offset.
    pub const fn calculate_table_row(a: u32) -> u32 {
        debug_assert!(a < TABLE_SIZE, "Operand 'a' exceeds maximum value");
        a
    }
}
