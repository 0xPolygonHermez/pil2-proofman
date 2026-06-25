#[cfg(test)]
mod tests {
    // `int_plus_one` fires on `trace_row!`-generated code, not on this file.
    // `needless_range_loop`: indices are passed to per-element getters/setters
    // (e.g. `get_flags(i)`, `get_nibbles(i, j)`), so the range index is intrinsic.
    #![allow(clippy::int_plus_one, clippy::needless_range_loop)]

    use proofman_common::GenericTrace;
    use proofman_macros::trace_row;
    use fields::{Goldilocks, PrimeField64};

    trace_row!(
        MainRow<F> {
            field0: F,
            field1: u8,
            field3: [[[u16; 4]; 2]; 3],
            field2: [[u32; 2]; 3],

        }
    );

    // Row with purely bit-packed fields for whole-array tests.
    trace_row!(
        BitRow<F> {
            flags:   [bit; 64],
            nibbles: [[ubit(4); 8]; 4],
            matrix:  [[[u8; 4]; 2]; 3],
        }
    );

    // --- Compile-time constant assertions -----------------------------------------
    // These are evaluated by the compiler at build time; a mismatch is a compile error.

    // flags: 64×1 = 64 bits, nibbles: 4×8×4 = 128 bits, matrix: 3×2×4×8 = 192 bits
    const _: () = assert!(BitRowPacked::<Goldilocks>::PACKED_BITS == 384);
    const _: () = assert!(BitRowPacked::<Goldilocks>::PACKED_WORDS == 6);

    // ROW_SIZE == PACKED_WORDS because BitRow has no generic F fields
    const _: () = assert!(<BitRowPacked<Goldilocks> as proofman_common::trace::TraceRow>::ROW_SIZE == 6);
    const _: () = assert!(<BitRowPacked<Goldilocks> as proofman_common::trace::TraceRow>::IS_PACKED);

    // Unpacked counterpart must report IS_PACKED = false
    const _: () = assert!(!<BitRow<Goldilocks> as proofman_common::trace::TraceRow>::IS_PACKED);

    // MainRowPacked: field1(8) + field3(3×2×4×16=384) + field2(3×2×32=192) = 584 bits
    // PACKED_WORDS = ceil(584/64) = 10, + 1 generic field (field0:F) → ROW_SIZE = 11
    const _: () = assert!(MainRowPacked::<Goldilocks>::PACKED_BITS == 584);
    const _: () = assert!(MainRowPacked::<Goldilocks>::PACKED_WORDS == 10);
    const _: () = assert!(<MainRowPacked<Goldilocks> as proofman_common::trace::TraceRow>::ROW_SIZE == 11);
    const _: () = assert!(<MainRowPacked<Goldilocks> as proofman_common::trace::TraceRow>::IS_PACKED);

    // This will generate MainRowPacked and MainRowUnpacked structs
    pub type MainTrace<F> = GenericTrace<MainRow<F>, 128, 0, 0>;
    pub type MainTracePacked<F> = GenericTrace<MainRowPacked<F>, 128, 0, 0>;

    pub type BitTracePacked = GenericTrace<BitRowPacked<Goldilocks>, 16, 1, 0>;

    #[test]
    fn test_packed_trace() {
        let mut trace: MainTrace<Goldilocks> = MainTrace::new();
        let mut trace_packed: MainTracePacked<Goldilocks> = MainTracePacked::new();

        // Test packed version — runtime accessors
        trace_packed[0].field0 = Goldilocks::from_u8(42);
        trace_packed[0].set_field1(125u8);
        trace_packed[0].set_field2(0, 0, 1);
        trace_packed[0].set_field2(0, 1, 24);
        trace_packed[0].set_field2(1, 0, 55);
        trace_packed[0].set_field2(1, 1, 333);
        trace_packed[0].set_field2(2, 0, 97);
        trace_packed[0].set_field2(2, 1, 4);

        // Test field3: [[[u16; 4]; 2], 3] - 3D array, runtime
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

        // Test unpacked version — runtime accessors
        trace[0].field0 = Goldilocks::from_u8(42);
        trace[0].set_field1(125u8);
        trace[0].set_field2(0, 0, 1);
        trace[0].set_field2(0, 1, 24);
        trace[0].set_field2(1, 0, 55);
        trace[0].set_field2(1, 1, 333);
        trace[0].set_field2(2, 0, 97);
        trace[0].set_field2(2, 1, 4);

        // Test field3: [[[u16; 4]; 2], 3] - 3D array, runtime
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

    // --- Whole-array set_all / get_all tests ---

    /// 1D bit array: set_all_flags roundtrip, and agreement with per-element setters.
    #[test]
    fn test_set_all_1d_bit_array_roundtrip() {
        let mut row = BitRowPacked::<Goldilocks>::default();

        // Build a known pattern: alternate true/false
        let values: [bool; 64] = std::array::from_fn(|i| i % 2 == 0);

        row.set_all_flags(&values);

        // get_all_flags must return the same array
        let got = row.get_all_flags();
        assert_eq!(got, values, "get_all_flags did not return what set_all_flags wrote");

        // Also agree with the per-element getter
        for (i, &value) in values.iter().enumerate() {
            assert_eq!(row.get_flags(i), value, "per-element get_flags({i}) disagrees with set_all_flags");
        }
    }

    /// 1D bit array: set_all agrees with per-element setters on every bit position.
    #[test]
    fn test_set_all_1d_bit_array_agrees_with_per_element() {
        // Set via per-element setters
        let mut row_elem = BitRowPacked::<Goldilocks>::default();
        for i in 0usize..64 {
            row_elem.set_flags(i, i % 3 == 0);
        }

        // Set via set_all
        let values: [bool; 64] = std::array::from_fn(|i| i % 3 == 0);
        let mut row_all = BitRowPacked::<Goldilocks>::default();
        row_all.set_all_flags(&values);

        assert_eq!(row_elem.packed, row_all.packed, "packed words differ: per-element vs set_all_flags");
    }

    /// 2D nibble array: set_all_nibbles roundtrip and per-element agreement.
    #[test]
    fn test_set_all_2d_array_roundtrip() {
        let mut row = BitRowPacked::<Goldilocks>::default();

        // Shape: [[u8; 8]; 4] — values are i*8 + j, staying within 4-bit range (0..15 via % 16)
        let values: [[u8; 8]; 4] = std::array::from_fn(|i| std::array::from_fn(|j| ((i * 8 + j) % 16) as u8));

        row.set_all_nibbles(&values);

        // Roundtrip via get_all
        let got = row.get_all_nibbles();
        assert_eq!(got, values, "get_all_nibbles did not return what set_all_nibbles wrote");

        // Agree with per-element getter
        for (i, row_vals) in values.iter().enumerate() {
            for (j, &value) in row_vals.iter().enumerate() {
                assert_eq!(row.get_nibbles(i, j), value, "per-element get_nibbles({i},{j}) disagrees");
            }
        }
    }

    /// 2D nibble array: set_all agrees with per-element setters on the packed words.
    #[test]
    fn test_set_all_2d_array_agrees_with_per_element() {
        let mut row_elem = BitRowPacked::<Goldilocks>::default();
        for i in 0usize..4 {
            for j in 0usize..8 {
                row_elem.set_nibbles(i, j, ((i * 8 + j) % 16) as u8);
            }
        }

        let values: [[u8; 8]; 4] = std::array::from_fn(|i| std::array::from_fn(|j| ((i * 8 + j) % 16) as u8));
        let mut row_all = BitRowPacked::<Goldilocks>::default();
        row_all.set_all_nibbles(&values);

        assert_eq!(row_elem.packed, row_all.packed, "packed words differ: per-element vs set_all_nibbles");
    }

    /// 3D byte array: set_all_matrix roundtrip and per-element agreement.
    #[test]
    fn test_set_all_3d_array_roundtrip() {
        let mut row = BitRowPacked::<Goldilocks>::default();

        let values: [[[u8; 4]; 2]; 3] =
            std::array::from_fn(|i| std::array::from_fn(|j| std::array::from_fn(|k| (i * 8 + j * 4 + k) as u8)));

        row.set_all_matrix(&values);

        let got = row.get_all_matrix();
        assert_eq!(got, values, "get_all_matrix did not return what set_all_matrix wrote");

        for (i, plane) in values.iter().enumerate() {
            for (j, row_vals) in plane.iter().enumerate() {
                for (k, &value) in row_vals.iter().enumerate() {
                    assert_eq!(row.get_matrix(i, j, k), value, "per-element get_matrix({i},{j},{k}) disagrees");
                }
            }
        }
    }

    /// 3D byte array: set_all agrees with per-element setters on the packed words.
    #[test]
    fn test_set_all_3d_array_agrees_with_per_element() {
        let mut row_elem = BitRowPacked::<Goldilocks>::default();
        for i in 0usize..3 {
            for j in 0usize..2 {
                for k in 0usize..4 {
                    row_elem.set_matrix(i, j, k, (i * 8 + j * 4 + k) as u8);
                }
            }
        }

        let values: [[[u8; 4]; 2]; 3] =
            std::array::from_fn(|i| std::array::from_fn(|j| std::array::from_fn(|k| (i * 8 + j * 4 + k) as u8)));
        let mut row_all = BitRowPacked::<Goldilocks>::default();
        row_all.set_all_matrix(&values);

        assert_eq!(row_elem.packed, row_all.packed, "packed words differ: per-element vs set_all_matrix");
    }

    // --- Unpacked BitRow set_all / get_all tests ---

    /// Unpacked BitRow: set_all_flags roundtrip and per-element agreement.
    #[test]
    fn test_unpacked_set_all_1d_roundtrip() {
        let mut row = BitRow::<Goldilocks>::default();

        let values: [bool; 64] = std::array::from_fn(|i| i % 2 == 0);
        row.set_all_flags(&values);

        let got = row.get_all_flags();
        assert_eq!(got, values, "unpacked get_all_flags did not return what set_all_flags wrote");

        for (i, &value) in values.iter().enumerate() {
            assert_eq!(row.get_flags(i), value, "unpacked per-element get_flags({i}) disagrees with set_all_flags");
        }
    }

    /// Unpacked BitRow: set_all_nibbles roundtrip and per-element agreement.
    #[test]
    fn test_unpacked_set_all_2d_roundtrip() {
        let mut row = BitRow::<Goldilocks>::default();

        let values: [[u8; 8]; 4] = std::array::from_fn(|i| std::array::from_fn(|j| ((i * 8 + j) % 16) as u8));
        row.set_all_nibbles(&values);

        let got = row.get_all_nibbles();
        assert_eq!(got, values, "unpacked get_all_nibbles did not return what set_all_nibbles wrote");

        for (i, row_vals) in values.iter().enumerate() {
            for (j, &value) in row_vals.iter().enumerate() {
                assert_eq!(row.get_nibbles(i, j), value, "unpacked per-element get_nibbles({i},{j}) disagrees");
            }
        }
    }

    /// Unpacked BitRow: set_all_matrix roundtrip and per-element agreement.
    #[test]
    fn test_unpacked_set_all_3d_roundtrip() {
        let mut row = BitRow::<Goldilocks>::default();

        let values: [[[u8; 4]; 2]; 3] =
            std::array::from_fn(|i| std::array::from_fn(|j| std::array::from_fn(|k| (i * 8 + j * 4 + k) as u8)));
        row.set_all_matrix(&values);

        let got = row.get_all_matrix();
        assert_eq!(got, values, "unpacked get_all_matrix did not return what set_all_matrix wrote");

        for (i, plane) in values.iter().enumerate() {
            for (j, row_vals) in plane.iter().enumerate() {
                for (k, &value) in row_vals.iter().enumerate() {
                    assert_eq!(
                        row.get_matrix(i, j, k),
                        value,
                        "unpacked per-element get_matrix({i},{j},{k}) disagrees"
                    );
                }
            }
        }
    }

    /// Unpacked and packed set_all_* must produce identical logical values for all fields.
    #[test]
    fn test_unpacked_vs_packed_set_all_agreement() {
        let mut unpacked = BitRow::<Goldilocks>::default();
        let mut packed = BitRowPacked::<Goldilocks>::default();

        let flags: [bool; 64] = std::array::from_fn(|i| i % 3 == 0);
        let nibbles: [[u8; 8]; 4] = std::array::from_fn(|i| std::array::from_fn(|j| ((i * 8 + j) % 16) as u8));
        let matrix: [[[u8; 4]; 2]; 3] =
            std::array::from_fn(|i| std::array::from_fn(|j| std::array::from_fn(|k| (i * 8 + j * 4 + k) as u8)));

        unpacked.set_all_flags(&flags);
        packed.set_all_flags(&flags);
        unpacked.set_all_nibbles(&nibbles);
        packed.set_all_nibbles(&nibbles);
        unpacked.set_all_matrix(&matrix);
        packed.set_all_matrix(&matrix);

        assert_eq!(unpacked.get_all_flags(), packed.get_all_flags(), "flags: unpacked vs packed disagree");
        assert_eq!(unpacked.get_all_nibbles(), packed.get_all_nibbles(), "nibbles: unpacked vs packed disagree");
        assert_eq!(unpacked.get_all_matrix(), packed.get_all_matrix(), "matrix: unpacked vs packed disagree");
    }

    /// Writing multiple rows via set_all doesn't bleed into adjacent rows.
    #[test]
    fn test_set_all_row_isolation() {
        let mut trace: BitTracePacked = BitTracePacked::new();

        let flags_0: [bool; 64] = std::array::from_fn(|i| i % 2 == 0);
        let flags_1: [bool; 64] = std::array::from_fn(|i| i % 2 != 0);

        trace[0].set_all_flags(&flags_0);
        trace[1].set_all_flags(&flags_1);

        assert_eq!(trace[0].get_all_flags(), flags_0, "row 0 corrupted after writing row 1");
        assert_eq!(trace[1].get_all_flags(), flags_1, "row 1 has wrong values");
    }

    // --- Compile-time / layout tests ----------------------------------------------

    /// The packed struct occupies exactly PACKED_WORDS * 8 bytes (no padding surprises).
    #[test]
    fn test_packed_struct_size_matches_packed_words() {
        assert_eq!(
            std::mem::size_of::<BitRowPacked<Goldilocks>>(),
            BitRowPacked::<Goldilocks>::PACKED_WORDS * std::mem::size_of::<u64>(),
            "BitRowPacked size_of does not match PACKED_WORDS * 8"
        );
        assert_eq!(
            std::mem::size_of::<MainRowPacked<Goldilocks>>(),
            // PACKED_WORDS * 8 bytes + 1 Goldilocks field (8 bytes)
            MainRowPacked::<Goldilocks>::PACKED_WORDS * std::mem::size_of::<u64>() + std::mem::size_of::<Goldilocks>(),
            "MainRowPacked size_of does not match expected layout"
        );
    }

    // --- Bulk-clear edge-case tests -----------------------------------------------
    // These specifically target correctness of the one-AND-per-word optimisation in
    // set_all_*: a wrong combined mask would either leave stale bits (overwrite test),
    // fail to clear all bits (all-zeros test), or clobber a neighbouring field that
    // shares the same packed word (neighbour-preservation test).

    /// Calling set_all twice must fully overwrite the first write.
    /// If the combined clear mask missed any bit position, the old value would bleed
    /// through when ORing in the new one.
    #[test]
    fn test_set_all_overwrite_clears_previous() {
        let mut row = BitRowPacked::<Goldilocks>::default();

        // First write: all flags true (all 64 bits set in packed[0])
        let all_true: [bool; 64] = [true; 64];
        row.set_all_flags(&all_true);
        assert_eq!(row.get_all_flags(), all_true);

        // Second write: alternating pattern — if the AND didn't clear everything, OR
        // would accidentally keep the previously-set bits.
        let alternating: [bool; 64] = std::array::from_fn(|i| i % 2 == 0);
        row.set_all_flags(&alternating);
        assert_eq!(row.get_all_flags(), alternating, "set_all_flags second write left stale bits from the first write");
    }

    /// Writing an all-zeros array must zero out bits that were previously set.
    /// A bulk-clear mask of 0 (empty) would leave old bits intact; this test catches that.
    #[test]
    fn test_set_all_zeros_clears_all_bits() {
        let mut row = BitRowPacked::<Goldilocks>::default();

        // Set a known non-zero pattern across all three fields.
        let all_true: [bool; 64] = [true; 64];
        let all_max_nibbles: [[u8; 8]; 4] = [[0x0F; 8]; 4];
        let all_max_matrix: [[[u8; 4]; 2]; 3] = [[[0xFF; 4]; 2]; 3];
        row.set_all_flags(&all_true);
        row.set_all_nibbles(&all_max_nibbles);
        row.set_all_matrix(&all_max_matrix);

        // Now zero everything out via set_all.
        row.set_all_flags(&[false; 64]);
        row.set_all_nibbles(&[[0u8; 8]; 4]);
        row.set_all_matrix(&[[[0u8; 4]; 2]; 3]);

        assert_eq!(row.get_all_flags(), [false; 64], "flags not zeroed");
        assert_eq!(row.get_all_nibbles(), [[0u8; 8]; 4], "nibbles not zeroed");
        assert_eq!(row.get_all_matrix(), [[[0u8; 4]; 2]; 3], "matrix not zeroed");
        // The packed array must be entirely zero.
        assert_eq!(row.packed, [0u64; 6], "packed words not fully zeroed");
    }

    /// set_all on an array field must not disturb a scalar field that shares the same
    /// packed word. In MainRowPacked, field1 (u8, bits 0-7 of packed[0]) and field3
    /// ([[[u16;4];2];3], starting at bit 8 of packed[0]) share packed[0].
    /// The bulk-clear mask for field3 must only cover bits ≥8, leaving bits 0-7 intact.
    #[test]
    fn test_set_all_does_not_clobber_neighbour_in_same_word() {
        let mut row = MainRowPacked::<Goldilocks>::default();

        // Set the scalar field that lives in the low byte of packed[0].
        row.set_field1(0xAB);
        assert_eq!(row.get_field1(), 0xAB, "precondition: field1 is 0xAB");

        // Now set_all on field3, which starts at bit 8 of the same packed[0].
        let vals: [[[u16; 4]; 2]; 3] =
            std::array::from_fn(|i| std::array::from_fn(|j| std::array::from_fn(|k| (i * 8 + j * 4 + k + 1) as u16)));
        row.set_all_field3(&vals);

        // field1 must still have the original value.
        assert_eq!(row.get_field1(), 0xAB, "set_all_field3 clobbered field1 in the same packed word");
        // field3 must have the new values.
        for i in 0..3usize {
            for j in 0..2usize {
                for k in 0..4usize {
                    assert_eq!(
                        row.get_field3(i, j, k),
                        (i * 8 + j * 4 + k + 1) as u16,
                        "field3[{i}][{j}][{k}] has wrong value after set_all"
                    );
                }
            }
        }
    }
}
