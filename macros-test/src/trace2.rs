#[cfg(test)]
mod tests {
    use proofman_common::GenericTrace;
    use proofman_macros::packed_row;

    packed_row!(
        PackedRow<F> {
            field0: F,
            field1: u8,
            field2: [[ubit(40);2]; 3],
            field3: bit,
            field4: [F; 2],
            field5: [i32; 5],
        }
    );

    pub type PackedTrace<F, const NUM_ROWS: usize, const AIRGROUP_ID: usize, const AIR_ID: usize> =
        GenericTrace<F, PackedRow<F>, 128, 0, 0>;

    #[test]
    fn test_packed_trace() {
        let _trace: PackedTrace<u64, 1024, 0, 0> = PackedTrace::new();
    }
}
