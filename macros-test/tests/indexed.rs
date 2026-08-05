// Coverage for indexed_trace_row!: the compact-row/instruction-table split, COL_SOURCE
// (which the C++ unpack consumes verbatim), and the @instr setters being no-ops.
use fields::{Goldilocks, PrimeField64};
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
    r.set_row_index(5);

    assert_eq!(IxRowOps::get_a(&r), 0xBEEF);
    assert_eq!(IxRowOps::get_b(&r), 0xABC);
    assert_eq!(IxRowOps::get_op(&r), 0, "@instr setter must be a no-op");
    assert!(!IxRowOps::get_flag(&r), "@instr setter must be a no-op");
    assert_eq!(r.get_index(), 5);
}
