#[cfg(test)]
mod tests {
    use proofman_macros::trace;
    use proofman_common as common;

    trace!(SampleTrace<F> { field1: F, field2: [F; 2] },  0, 0, 32 );

    use proofman_common::GenericTrace;
    use proofman_macros::trace_row;
    use fields::{Goldilocks, PrimeField64};

    trace_row!(
        MainRow<F> {
            field0: F,
            field1: u8,
            field2: [[u32; 2]; 3],
            field3: [[[u16; 4]; 2]; 3],
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
        trace_packed[0].set_field2(0, 0, 1);
        trace_packed[0].set_field2(0, 1, 24);
        trace_packed[0].set_field2(1, 0, 55);
        trace_packed[0].set_field2(1, 1, 333);
        trace_packed[0].set_field2(2, 0, 97);
        trace_packed[0].set_field2(2, 1, 4);
        
        // Test field3: [[[u16; 4]; 2], 3] - 3D array
        trace_packed[0].set_field3(0, 0, 0, 100u16);
        trace_packed[0].set_field3(0, 0, 1, 101u16);
        trace_packed[0].set_field3(0, 0, 2, 102u16);
        trace_packed[0].set_field3(0, 0, 3, 103u16);
        trace_packed[0].set_field3(0, 1, 0, 200u16);
        trace_packed[0].set_field3(0, 1, 1, 201u16);
        trace_packed[0].set_field3(0, 1, 2, 202u16);
        trace_packed[0].set_field3(0, 1, 3, 203u16);
        trace_packed[0].set_field3(1, 0, 0, 300u16);
        trace_packed[0].set_field3(1, 0, 1, 301u16);
        trace_packed[0].set_field3(1, 0, 2, 302u16);
        trace_packed[0].set_field3(1, 0, 3, 303u16);
        trace_packed[0].set_field3(1, 1, 0, 400u16);
        trace_packed[0].set_field3(1, 1, 1, 401u16);
        trace_packed[0].set_field3(1, 1, 2, 402u16);
        trace_packed[0].set_field3(1, 1, 3, 403u16);
        trace_packed[0].set_field3(2, 0, 0, 500u16);
        trace_packed[0].set_field3(2, 0, 1, 501u16);
        trace_packed[0].set_field3(2, 0, 2, 502u16);
        trace_packed[0].set_field3(2, 0, 3, 503u16);
        trace_packed[0].set_field3(2, 1, 0, 600u16);
        trace_packed[0].set_field3(2, 1, 1, 601u16);
        trace_packed[0].set_field3(2, 1, 2, 602u16);
        trace_packed[0].set_field3(2, 1, 3, 603u16);

        // Test unpacked version
        trace[0].field0 = Goldilocks::from_u8(42);
        trace[0].set_field1(125u8);
        trace[0].set_field2(0, 0, 1);
        trace[0].set_field2(0, 1, 24);
        trace[0].set_field2(1, 0, 55);
        trace[0].set_field2(1, 1, 333);
        trace[0].set_field2(2, 0, 97);
        trace[0].set_field2(2, 1, 4);
        
        // Test field3: [[[u16; 4]; 2], 3] - 3D array
        trace[0].set_field3(0, 0, 0, 100u16);
        trace[0].set_field3(0, 0, 1, 101u16);
        trace[0].set_field3(0, 0, 2, 102u16);
        trace[0].set_field3(0, 0, 3, 103u16);
        trace[0].set_field3(0, 1, 0, 200u16);
        trace[0].set_field3(0, 1, 1, 201u16);
        trace[0].set_field3(0, 1, 2, 202u16);
        trace[0].set_field3(0, 1, 3, 203u16);
        trace[0].set_field3(1, 0, 0, 300u16);
        trace[0].set_field3(1, 0, 1, 301u16);
        trace[0].set_field3(1, 0, 2, 302u16);
        trace[0].set_field3(1, 0, 3, 303u16);
        trace[0].set_field3(1, 1, 0, 400u16);
        trace[0].set_field3(1, 1, 1, 401u16);
        trace[0].set_field3(1, 1, 2, 402u16);
        trace[0].set_field3(1, 1, 3, 403u16);
        trace[0].set_field3(2, 0, 0, 500u16);
        trace[0].set_field3(2, 0, 1, 501u16);
        trace[0].set_field3(2, 0, 2, 502u16);
        trace[0].set_field3(2, 0, 3, 503u16);
        trace[0].set_field3(2, 1, 0, 600u16);
        trace[0].set_field3(2, 1, 1, 601u16);
        trace[0].set_field3(2, 1, 2, 602u16);
        trace[0].set_field3(2, 1, 3, 603u16);

        assert_eq!(trace[0].field0, trace_packed[0].field0);
        assert_eq!(trace[0].get_field1(), trace_packed[0].get_field1());
        assert_eq!(trace[0].get_field2(0, 0), trace_packed[0].get_field2(0, 0));
        assert_eq!(trace[0].get_field2(0, 1), trace_packed[0].get_field2(0, 1));
        assert_eq!(trace[0].get_field2(1, 0), trace_packed[0].get_field2(1, 0));
        assert_eq!(trace[0].get_field2(1, 1), trace_packed[0].get_field2(1, 1));
        assert_eq!(trace[0].get_field2(2, 0), trace_packed[0].get_field2(2, 0));
        assert_eq!(trace[0].get_field2(2, 1), trace_packed[0].get_field2(2, 1));
        
        // Test field3 assertions
        assert_eq!(trace[0].get_field3(0, 0, 0), trace_packed[0].get_field3(0, 0, 0));
        assert_eq!(trace[0].get_field3(0, 0, 1), trace_packed[0].get_field3(0, 0, 1));
        assert_eq!(trace[0].get_field3(0, 0, 2), trace_packed[0].get_field3(0, 0, 2));
        assert_eq!(trace[0].get_field3(0, 0, 3), trace_packed[0].get_field3(0, 0, 3));
        assert_eq!(trace[0].get_field3(0, 1, 0), trace_packed[0].get_field3(0, 1, 0));
        assert_eq!(trace[0].get_field3(0, 1, 1), trace_packed[0].get_field3(0, 1, 1));
        assert_eq!(trace[0].get_field3(0, 1, 2), trace_packed[0].get_field3(0, 1, 2));
        assert_eq!(trace[0].get_field3(0, 1, 3), trace_packed[0].get_field3(0, 1, 3));
        assert_eq!(trace[0].get_field3(1, 0, 0), trace_packed[0].get_field3(1, 0, 0));
        assert_eq!(trace[0].get_field3(1, 0, 1), trace_packed[0].get_field3(1, 0, 1));
        assert_eq!(trace[0].get_field3(1, 0, 2), trace_packed[0].get_field3(1, 0, 2));
        assert_eq!(trace[0].get_field3(1, 0, 3), trace_packed[0].get_field3(1, 0, 3));
        assert_eq!(trace[0].get_field3(1, 1, 0), trace_packed[0].get_field3(1, 1, 0));
        assert_eq!(trace[0].get_field3(1, 1, 1), trace_packed[0].get_field3(1, 1, 1));
        assert_eq!(trace[0].get_field3(1, 1, 2), trace_packed[0].get_field3(1, 1, 2));
        assert_eq!(trace[0].get_field3(1, 1, 3), trace_packed[0].get_field3(1, 1, 3));
        assert_eq!(trace[0].get_field3(2, 0, 0), trace_packed[0].get_field3(2, 0, 0));
        assert_eq!(trace[0].get_field3(2, 0, 1), trace_packed[0].get_field3(2, 0, 1));
        assert_eq!(trace[0].get_field3(2, 0, 2), trace_packed[0].get_field3(2, 0, 2));
        assert_eq!(trace[0].get_field3(2, 0, 3), trace_packed[0].get_field3(2, 0, 3));
        assert_eq!(trace[0].get_field3(2, 1, 0), trace_packed[0].get_field3(2, 1, 0));
        assert_eq!(trace[0].get_field3(2, 1, 1), trace_packed[0].get_field3(2, 1, 1));
        assert_eq!(trace[0].get_field3(2, 1, 2), trace_packed[0].get_field3(2, 1, 2));
        assert_eq!(trace[0].get_field3(2, 1, 3), trace_packed[0].get_field3(2, 1, 3));
        
    }
}
