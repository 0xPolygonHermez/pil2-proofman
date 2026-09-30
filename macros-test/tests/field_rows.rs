// The generated packed accessors trip int_plus_one on narrow array columns -- a lint in
// generated code, which is why the generated pil-helpers allow clippy wholesale.
#![allow(clippy::int_plus_one)]
// Rows over a field that is not 64-bit: BN254's `Fr`, which a pilfflonk witness is computed in
// (D4). An unpacked row and its `*Ops` trait take any field; only the typed columns' accessors,
// which convert through 64 bits, and the packed row need `PrimeField64`.
use proofman_common::trace::TraceRow;
use proofman_common::GenericTrace;
use proofman_fields::{Bn254, Field, Goldilocks, PrimeField64, QuotientMap};
use proofman_macros::trace_row;

// A row as pil-helpers writes it for a BN254 pilout: every column an `F`.
trace_row!(
    FrRow<F> {
        a: F,
        b: [F; 3],
        c: [[F; 2]; 2],
    }
);

// Typed and generic columns together: over Goldilocks the typed accessors are there, over BN254
// the row is still a row of its `F` columns.
trace_row!(
    MixedRow<F> {
        x: F,
        byte: u8,
        flags: [bit; 4],
    }
);

const _: () = assert!(<FrRow<Bn254> as TraceRow>::ROW_SIZE == 8);
const _: () = assert!(!<FrRow<Bn254> as TraceRow>::IS_PACKED);

type FrTrace<F> = GenericTrace<FrRow<F>, 4, 0, 0>;

fn fr(n: i128) -> Bn254 {
    Bn254::from_int(n)
}

/// A filler that only knows the row's trait, over any field.
fn fill<F: Copy + Default + Send + 'static, R: FrRowOps<F>>(row: &mut R, value: F) {
    row.set_a(value);
}

#[test]
fn a_row_over_bn254_is_its_columns_in_order() {
    assert_eq!(std::mem::size_of::<FrRow<Bn254>>(), 8 * std::mem::size_of::<Bn254>());
    let mut trace = FrTrace::<Bn254>::new_zeroes();
    for i in 0..trace.num_rows() {
        let base = 10 * i as i128;
        fill(&mut trace[i], fr(base));
        trace[i].b = [fr(base + 1), fr(base + 2), fr(-(base + 3))];
        trace[i].c = [[fr(base + 4), fr(base + 5)], [fr(base + 6), fr(base + 7)]];
    }
    assert_eq!(trace[2].get_a(), fr(20));

    // Row after row, each its columns in declaration order, arrays flattened: the layout
    // `Stage1Witness::from_rows` reads.
    let flat: Vec<Bn254> = trace.get_buffer();
    assert_eq!(flat.len(), 4 * 8);
    for (k, value) in flat.iter().enumerate() {
        let (row, col) = (k / 8, k % 8);
        let expected = 10 * row as i128 + col as i128;
        assert_eq!(*value, if col == 3 { fr(-expected) } else { fr(expected) }, "row {row}, column {col}");
    }
    assert_eq!(flat[3], -fr(3), "a negative value is r − |x|");
}

#[test]
fn typed_columns_keep_their_accessors_over_goldilocks() {
    fn set_typed<F: PrimeField64, R: MixedRowOps<F>>(row: &mut R) {
        row.set_byte(0xAB);
        row.set_flags(2, true);
    }

    let mut row = MixedRow::<Goldilocks>::default();
    set_typed(&mut row);
    assert_eq!((row.get_byte(), row.get_flags(2), row.get_flags(1)), (0xAB, true, false));
    assert_eq!(row.byte, Goldilocks::from_u8(0xAB));

    let mut packed = MixedRowPacked::<Goldilocks>::default();
    set_typed(&mut packed);
    assert_eq!((packed.get_byte(), packed.get_flags(2)), (0xAB, true));
}

#[test]
fn a_row_with_typed_columns_still_holds_bn254_values() {
    let mut row = MixedRow::<Bn254>::default();
    row.set_x(fr(7));
    row.byte = Bn254::ONE;
    row.flags[3] = Bn254::TWO;
    assert_eq!(
        (row.get_x(), row.byte, row.flags),
        (fr(7), Bn254::ONE, [Bn254::ZERO, Bn254::ZERO, Bn254::ZERO, Bn254::TWO])
    );
}
