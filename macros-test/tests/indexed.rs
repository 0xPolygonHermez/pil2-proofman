// The generated packed accessors trip int_plus_one on narrow array columns -- a lint in
// generated code, which is why the generated pil-helpers allow clippy wholesale.
#![allow(clippy::int_plus_one)]
// Coverage for indexed_trace_row!: the compact-row/instruction-table split, COL_SOURCE
// (which the C++ unpack consumes verbatim), and the @instr setters being no-ops.
use proofman_fields::{Goldilocks, PrimeField64};
use proofman_common::trace::IndexedFill;
use proofman_macros::{indexed_trace_row, trace_row};

trace_row!(
    IxRow<F> {
        a: u16,
        op: u8,
        b: ubit(12),
        flag: bit,
    }
);

indexed_trace_row!(
    IxRow<F> {
        a: u16,
        op: u8 @instr,
        b: ubit(12),
        flag: bit @instr,
    }
);

// The indexed discriminator is what a generic filler branches on to compile out the
// instruction-derived columns, so assert it where a regression is a build failure
// rather than a test failure (same style as the constant checks in src/lib.rs).
const _: () = assert!(<IxRowPackedIndexed<Goldilocks> as IndexedFill>::IS_INDEXED);
const _: () = assert!(!<IxRow<Goldilocks> as IndexedFill>::IS_INDEXED);
const _: () = assert!(!<IxRowPacked<Goldilocks> as IndexedFill>::IS_INDEXED);

#[test]
fn compiles_and_routes() {
    // COL_SOURCE must mark op and flag as table-sourced.
    assert_eq!(IxRowPackedIndexed::<Goldilocks>::COL_SOURCE, [0u8, 1, 0, 1]);
    assert_eq!(IxRowPackedIndexed::<Goldilocks>::INDEX_BITS, 32);

    // Compact row: index(32) + a(16) + b(12) = 60 bits -> 1 word.
    assert_eq!(IxRowPackedIndexed::<Goldilocks>::PACKED_BITS, 60);
    assert_eq!(IxRowPackedIndexed::<Goldilocks>::PACKED_WORDS, 1);
    // Table entry: op(8) + flag(1) = 9 bits -> 1 word.
    assert_eq!(IxRowInstrTable::<Goldilocks>::PACKED_BITS, 9);

    // Trait routing: runtime setters land, @instr setters are no-ops.
    let mut r = IxRowPackedIndexed::<Goldilocks>::default();
    IxRowOps::set_a(&mut r, 0xBEEF);
    IxRowOps::set_b(&mut r, 0xABC);
    IxRowOps::set_op(&mut r, 7);
    IxRowOps::set_flag(&mut r, true);
    r.set_row_index(0, 5);

    assert_eq!(IxRowOps::get_a(&r), 0xBEEF);
    assert_eq!(IxRowOps::get_b(&r), 0xABC);
    assert_eq!(IxRowOps::get_op(&r), 0, "@instr setter must be a no-op");
    assert!(!IxRowOps::get_flag(&r), "@instr setter must be a no-op");
    assert_eq!(r.get_index(0), 5);
}

// ---------------------------------------------------------------------------
// Lane-packed rows: one row carries several execution steps, so the compact row
// needs one instruction index per lane, and every output column has to name the
// lane whose table entry it comes from -- COL_LANE, which the C++/CUDA unpack
// consumes the same way it consumes COL_SOURCE. The lane is the OUTER array
// dimension, matching how the pil-helpers generate a lane-packed row.
// ---------------------------------------------------------------------------

trace_row!(
    LxRow<F> {
        a: [[u16; 2]; 2],
        op: [u8; 2],
        imm: [[u32; 2]; 2],
        flag: [bit; 2],
    }
);

indexed_trace_row!(
    LxRow<F> {
        a: [[u16; 2]; 2],
        op: [u8; 2] @instr,
        imm: [[u32; 2]; 2] @instr,
        flag: [bit; 2],
    }
);

const _: () = assert!(<LxRowPackedIndexed<Goldilocks> as IndexedFill>::IS_INDEXED);

#[test]
fn lane_packed_descriptor_names_a_lane_per_column() {
    assert_eq!(LxRowPackedIndexed::<Goldilocks>::LANES, 2);
    // a(4) + op(2) + imm(4) + flag(2) = 12 output columns.
    assert_eq!(
        LxRowPackedIndexed::<Goldilocks>::COL_SOURCE,
        [0u8, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0]
    );
    // Each field's block is lane-major: `op` contributes one column per lane, `imm` two.
    // Runtime columns never select an entry, so their lane is 0.
    assert_eq!(
        LxRowPackedIndexed::<Goldilocks>::COL_LANE,
        [0u8, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0]
    );
}

#[test]
fn lane_packed_row_carries_one_index_per_lane() {
    // index[2](64) + a(4 x 16) + flag(2) = 130 bits -> 3 words.
    assert_eq!(LxRowPackedIndexed::<Goldilocks>::PACKED_BITS, 130);
    assert_eq!(LxRowPackedIndexed::<Goldilocks>::PACKED_WORDS, 3);

    let mut r = LxRowPackedIndexed::<Goldilocks>::default();
    r.set_row_index(0, 7);
    r.set_row_index(1, 9);
    assert_eq!(r.get_index(0), 7);
    assert_eq!(r.get_index(1), 9);
}

#[test]
fn lane_packed_table_entry_is_one_lane_wide() {
    // One entry is one instruction, lane dimension stripped: op(8) + imm(2 x 32) = 72 bits.
    assert_eq!(LxRowInstrTable::<Goldilocks>::PACKED_BITS, 72);

    // The writer that fills a trace row also fills a table entry: `@instr` setters land
    // (the lane argument names the entry's own single lane), runtime setters are no-ops.
    let mut e = LxRowInstrTable::<Goldilocks>::default();
    LxRowOps::set_op(&mut e, 1, 0xAB);
    LxRowOps::set_imm(&mut e, 1, 0, 0xDEAD_BEEF);
    LxRowOps::set_flag(&mut e, 0, true);

    assert_eq!(LxRowOps::get_op(&e, 0), 0xAB);
    assert_eq!(LxRowOps::get_op(&e, 1), 0xAB, "an entry broadcasts its single instruction");
    assert_eq!(LxRowOps::get_imm(&e, 1, 0), 0xDEAD_BEEF);
    assert!(!LxRowOps::get_flag(&e, 0), "runtime setter must be a no-op on a table entry");
}

#[test]
fn lane_packed_compact_row_routes_runtime_columns_per_lane() {
    let mut r = LxRowPackedIndexed::<Goldilocks>::default();
    LxRowOps::set_a(&mut r, 1, 0, 0xBEEF);
    LxRowOps::set_flag(&mut r, 1, true);
    LxRowOps::set_op(&mut r, 1, 0xAB);

    assert_eq!(LxRowOps::get_a(&r, 1, 0), 0xBEEF);
    assert!(LxRowOps::get_flag(&r, 1));
    assert_eq!(LxRowOps::get_op(&r, 1), 0, "`@instr` setter must be a no-op on the compact row");
}
