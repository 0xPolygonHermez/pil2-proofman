//! The fixed columns of an AIR and `<air>.const` (spec A.6): decoding the pilout's values, the
//! file's layout, its round trip and what the reader refuses.

use std::fs;

use num_bigint::BigUint;
use pilfflonk_setup::fixed::FixedColumns;
use pilfflonk_setup::SetupError;
use proofman_pilfflonk::FrBytes;

use crate::common::*;

fn to_fr(value: &BigUint) -> FrBytes {
    fr(&value.to_str_radix(10))
}

fn columns() -> FixedColumns {
    FixedColumns::from_air(&pilout().air_groups[0].airs[0]).unwrap()
}

#[test]
fn the_pilouts_values_are_decoded_as_they_are() {
    let fixed = columns();
    assert_eq!((fixed.n_bits(), fixed.n_rows(), fixed.n_columns()), (N_BITS, N, 4));
    for (c, values) in fixed_values().iter().enumerate() {
        let decoded: Vec<FrBytes> = values.iter().map(to_fr).collect();
        assert_eq!(fixed.column(c).unwrap(), decoded.as_slice(), "column {c}");
    }
    assert!(fixed.column(4).is_none());
}

/// A value's bytes may be empty (0), or have leading zeros: only the number counts.
#[test]
fn any_big_endian_spelling_of_a_value_is_read() {
    let mut pilout = pilout();
    let air = the_air(&mut pilout);
    let spellings: Vec<Vec<u8>> = vec![
        vec![],
        vec![0],
        vec![0; 40],
        vec![0, 0, 1, 2],
        [vec![0; 8], big(R_MINUS_ONE).to_bytes_be()].concat(),
        big(WIDE).to_bytes_be(),
        vec![7],
        vec![0, 0, 0, 7],
    ];
    air.fixed_cols[0].values = spellings;
    let fixed = FixedColumns::from_air(air).unwrap();
    let expected: Vec<FrBytes> = ["0", "0", "0", "258", R_MINUS_ONE, WIDE, "7", "7"].iter().map(|v| fr(v)).collect();
    assert_eq!(fixed.column(0).unwrap(), expected.as_slice());
}

#[test]
fn a_column_without_a_value_per_row_is_refused() {
    for n_values in [0, N - 1, N + 1] {
        let mut pilout = pilout();
        let air = the_air(&mut pilout);
        air.fixed_cols[1].values.resize(n_values, vec![1]);
        let err = FixedColumns::from_air(air).unwrap_err();
        assert!(matches!(err, SetupError::FixedValues { column: 1, n_values: v, n_rows: N } if v == n_values), "{err}");
    }
}

/// Written row by row: column `c` of row `i` at byte `(i·C + c)·32`, little-endian; read back,
/// the same values.
#[test]
fn the_const_file_round_trips_row_by_row() {
    let dir = TestDir::new("const_round_trip");
    let path = dir.file("Sample.const");
    let fixed = columns();
    fixed.write_const(&path).unwrap();

    let bytes = fs::read(&path).unwrap();
    assert_eq!(bytes.len(), N * 4 * 32);
    let values = fixed_values();
    for (i, c) in [(0, 0), (0, 3), (1, 2), (2, 1), (7, 3)] {
        let at = (i * 4 + c) * 32;
        assert_eq!(BigUint::from_bytes_le(&bytes[at..at + 32]), values[c][i], "row {i}, column {c}");
    }

    assert_eq!(FixedColumns::read_const(&path, N_BITS, 4).unwrap(), fixed);
    // Writing again replaces the file with the same bytes.
    fixed.write_const(&path).unwrap();
    assert_eq!(fs::read(&path).unwrap(), bytes);
}

#[test]
fn an_air_without_fixed_columns_has_an_empty_const_file() {
    let dir = TestDir::new("const_empty");
    let path = dir.file("Empty.const");
    let fixed = FixedColumns::new(N_BITS, vec![]).unwrap();
    fixed.write_const(&path).unwrap();
    assert_eq!(fs::read(&path).unwrap(), Vec::<u8>::new());
    assert_eq!(FixedColumns::read_const(&path, N_BITS, 0).unwrap(), fixed);
}

#[test]
fn the_reader_refuses_a_file_of_another_size_or_a_value_not_below_r() {
    let dir = TestDir::new("const_refused");
    let path = dir.file("Sample.const");
    columns().write_const(&path).unwrap();
    let good = fs::read(&path).unwrap();

    for (n_bits, n_columns) in [(N_BITS, 3), (N_BITS, 5), (N_BITS + 1, 4), (N_BITS - 1, 4)] {
        let err = FixedColumns::read_const(&path, n_bits, n_columns).unwrap_err();
        assert!(matches!(&err, SetupError::ConstFile { reason, .. } if reason.contains("bytes")), "{err}");
    }
    fs::write(&path, &good[..good.len() - 1]).unwrap();
    assert!(matches!(FixedColumns::read_const(&path, N_BITS, 4), Err(SetupError::ConstFile { .. })));

    // r at row 6, column 1.
    let mut bad = good.clone();
    let at = (6 * 4 + 1) * 32;
    let mut r_le = r().to_bytes_le();
    r_le.resize(32, 0);
    bad[at..at + 32].copy_from_slice(&r_le);
    fs::write(&path, &bad).unwrap();
    let err = FixedColumns::read_const(&path, N_BITS, 4).unwrap_err();
    assert!(
        matches!(&err, SetupError::ConstFile { reason, .. } if reason.contains("row 6 of column 1 is not below r")),
        "{err}"
    );

    assert!(matches!(FixedColumns::read_const(&dir.file("missing"), N_BITS, 4), Err(SetupError::Io { .. })));
    assert!(matches!(FixedColumns::read_const(&path, 29, 4), Err(SetupError::NBits { n_bits: 29 })));
}

#[test]
fn columns_must_have_a_value_per_row() {
    let column = vec![FrBytes::ZERO; N];
    FixedColumns::new(N_BITS, vec![column.clone(), column.clone()]).unwrap();
    let err = FixedColumns::new(N_BITS, vec![column, vec![FrBytes::ZERO; N - 1]]).unwrap_err();
    assert!(matches!(err, SetupError::FixedValues { column: 1, n_values: 7, n_rows: 8 }), "{err}");
    assert!(matches!(FixedColumns::new(29, vec![]), Err(SetupError::NBits { n_bits: 29 })));
}
