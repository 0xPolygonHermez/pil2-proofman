#[cfg(test)]
mod tests {
    use proofman_common::GenericTrace;
    use proofman_macros::trace_row;
    use fields::{Goldilocks, PrimeField64};

    trace_row!(
        MainRow<F> {
            field0: F,
            field1: u8,
            field2: [[bit; 2]; 3],
        }
    );

    // This will generate MainRowPacked and MainRowUnpacked structs
    pub type MainTrace<F> = GenericTrace<F, MainRow<F>, 128, 0, 0>;
    pub type MainTracePacked<F> = GenericTrace<F, MainRowPacked<F>, 128, 0, 0>;

    #[test]
    fn test_packed_trace() {
        let mut trace: MainTrace<Goldilocks> = MainTrace::new();
        let mut trace_packed: MainTracePacked<Goldilocks> = MainTracePacked::new();

        // Test packed version
        trace_packed[0].field0 = Goldilocks::from_u8(42);
        trace_packed[0].set_field1(125u8);
        trace_packed[0].set_field2(0, 0, true);
        trace_packed[0].set_field2(0, 1, false);
        trace_packed[0].set_field2(1, 0, false);
        trace_packed[0].set_field2(1, 1, true);
        trace_packed[0].set_field2(2, 0, true);
        trace_packed[0].set_field2(2, 1, false);

        // Test unpacked version
        trace[0].field0 = Goldilocks::from_u8(42);
        trace[0].set_field1(125u8);
        trace[0].set_field2(0, 0, true);
        trace[0].set_field2(0, 1, false);
        trace[0].set_field2(1, 0, false);
        trace[0].set_field2(1, 1, true);
        trace[0].set_field2(2, 0, true);
        trace[0].set_field2(2, 1, false);

        assert_eq!(trace[0].field0, trace_packed[0].field0);
        assert_eq!(trace[0].get_field1(), trace_packed[0].get_field1());
        assert_eq!(trace[0].get_field2(0, 0), trace_packed[0].get_field2(0, 0));
        assert_eq!(trace[0].get_field2(0, 1), trace_packed[0].get_field2(0, 1));
        assert_eq!(trace[0].get_field2(1, 0), trace_packed[0].get_field2(1, 0));
        assert_eq!(trace[0].get_field2(1, 1), trace_packed[0].get_field2(1, 1));
        assert_eq!(trace[0].get_field2(2, 0), trace_packed[0].get_field2(2, 0));
        assert_eq!(trace[0].get_field2(2, 1), trace_packed[0].get_field2(2, 1));
    }
}
