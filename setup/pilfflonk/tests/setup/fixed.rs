//! The fixed columns of an AIR and `<air>.const` (pilfflonk/docs/formats.md#fixed-columns):
//! decoding the pilout's values, the external columns of those it has none of, the file's layout,
//! its round trip and what the reader refuses.

use std::fs;

use num_bigint::BigUint;
use pil2_pilout::pilout::{self as pb, SymbolType};
use pilfflonk_setup::fixed::FixedColumns;
use pilfflonk_setup::validate::validate;
use pilfflonk_setup::{ExternalFixedColumn, SetupError};
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

fn with_external(pilout: &pb::PilOut, external: Vec<ExternalFixedColumn>) -> Result<FixedColumns, SetupError> {
    FixedColumns::from_air_with_external(pilout, validate(pilout).unwrap(), external)
}

/// The columns the pilout has no values of take those of the external columns that name them, in
/// any order; the others keep the pilout's.
#[test]
fn external_columns_fill_the_columns_the_pilout_has_no_values_of() {
    let (pilout, mut external) = pilout_with_external_fixed();
    assert_eq!(with_external(&pilout, external.clone()).unwrap(), columns());
    external.reverse();
    assert_eq!(with_external(&pilout, external).unwrap(), columns());
}

/// Without external columns, a column without values is refused as it was before there were any,
/// and a pilout with all its values gives the same columns.
#[test]
fn without_external_columns_a_column_without_values_is_refused_as_before() {
    let (without_values, _) = pilout_with_external_fixed();
    let err = with_external(&without_values, vec![]).unwrap_err();
    assert!(matches!(err, SetupError::FixedValues { column: 1, n_values: 0, n_rows: N }), "{err}");
    let before = FixedColumns::from_air(&without_values.air_groups[0].airs[0]).unwrap_err();
    assert_eq!(err.to_string(), before.to_string());
    assert_eq!(with_external(&pilout(), vec![]).unwrap(), columns());
}

/// What the merge refuses: a column without values that no external column fills, an external
/// column of no fixed column of the AIR or of one with values in the pilout, the same column twice,
/// and a column without a value per row. A value that is not below `r` cannot be given: an
/// `FrBytes` is below `r`, and its constructors refuse any other value.
#[test]
fn the_external_columns_that_do_not_fill_the_columns_without_values_are_refused() {
    let (pilout, external) = pilout_with_external_fixed();
    let without = |i: usize| -> Vec<ExternalFixedColumn> {
        external.iter().enumerate().filter(|&(j, _)| j != i).map(|(_, c)| c.clone()).collect()
    };
    let named = |name: &str, index: usize| ExternalFixedColumn {
        name: name.to_string(),
        index,
        values: vec![FrBytes::ZERO; N],
    };

    let err = with_external(&pilout, without(1)).unwrap_err();
    assert!(
        matches!(&err, SetupError::ExternalFixedMissing { column: 2, name, index: 1 } if name == "Sample.C"),
        "{err}"
    );
    assert_eq!(
        err.to_string(),
        "fixed column 2 of the AIR, Sample.C (index 1), has no values in the pilout and no external column gives \
         them: a column declared `#pragma fixed_external` takes its values from the caller of the setup \
         (pilfflonk/docs/formats.md#fixed-columns)"
    );
    let err = with_external(&pilout, without(2)).unwrap_err();
    assert!(matches!(&err, SetupError::ExternalFixedMissing { column: 3, index: 0, .. }), "{err}");

    // Names and indices of no fixed column of the AIR: the index is past the array's, the name is
    // a witness column's, or is not the symbol's whole name.
    for (name, index) in [("Sample.C", 2), ("Sample.U", 1), ("Sample.a", 0), ("C", 0), ("Sample.V", 0)] {
        let mut given = external.clone();
        given.push(named(name, index));
        let err = with_external(&pilout, given).unwrap_err();
        assert!(
            matches!(&err, SetupError::ExternalFixedUnknown { air, name: n, index: i }
                if air == "Sample" && n == name && *i == index),
            "{err}"
        );
        assert!(err.to_string().contains("is not a fixed column of air Sample"), "{err}");
    }

    let mut given = external.clone();
    given.push(named("Sample.L1", 0));
    let err = with_external(&pilout, given).unwrap_err();
    assert!(matches!(err, SetupError::ExternalFixedNotEmpty { column: 0, index: 0, .. }), "{err}");
    assert!(err.to_string().contains("which has its values in the pilout"), "{err}");

    let mut given = external.clone();
    given.push(external[0].clone());
    let err = with_external(&pilout, given).unwrap_err();
    assert!(matches!(&err, SetupError::ExternalFixedTwice { name, index: 0 } if name == "Sample.C"), "{err}");

    for n_values in [0, N - 1, N + 1] {
        let mut given = external.clone();
        given[2].values.resize(n_values, FrBytes::ZERO);
        let err = with_external(&pilout, given).unwrap_err();
        assert!(
            matches!(&err, SetupError::ExternalFixedValues { name, index: 0, n_values: v, n_rows: N }
                if name == "Sample.U" && *v == n_values),
            "{err}"
        );
    }

    let mut r_le = r().to_bytes_le();
    r_le.resize(32, 0);
    assert!(FrBytes::from_le_bytes(r_le.try_into().unwrap()).is_err());
    assert!(FrBytes::from_decimal(&r().to_str_radix(10)).is_err());
}

/// With external columns, the pilout's own columns are decoded and refused as `from_air` does: a
/// column with some values but not one per row, and a value that is not below `r`.
#[test]
fn with_external_columns_the_pilouts_own_columns_are_refused_as_before() {
    let (mut pilout, external) = pilout_with_external_fixed();
    the_air(&mut pilout).fixed_cols[0].values.truncate(N - 1);
    let err = with_external(&pilout, external.clone()).unwrap_err();
    assert!(matches!(err, SetupError::FixedValues { column: 0, n_values: 7, n_rows: N }), "{err}");

    let (mut pilout, external) = pilout_with_external_fixed();
    the_air(&mut pilout).fixed_cols[0].values[5] = be(&r());
    let err = with_external(&pilout, external).unwrap_err();
    assert!(
        matches!(&err, SetupError::ConstantNotBelowR { location, .. } if location == "row 5 of fixed column 0"),
        "{err}"
    );
}

/// The index of an external column is its position among the fixed columns of the AIR of its name,
/// in the pilout's order (`ExternalFixedColumn::index`): row-major in an array of two dimensions,
/// and `k` for the `k`-th of the columns that share a name and are not arrays.
#[test]
fn the_index_is_the_position_among_the_columns_of_that_name() {
    let mut pilout = pilout();
    for col in &mut the_air(&mut pilout).fixed_cols {
        col.values.clear();
    }
    // The four columns as one array, M[2][2]; then as four columns named T, whose symbols are in
    // the reverse order of their columns.
    let others: Vec<pb::Symbol> =
        pilout.symbols.iter().filter(|s| s.r#type != SymbolType::FixedCol as i32).cloned().collect();
    let m = pb::Symbol { lengths: vec![2, 2], dim: 2, ..symbol("Sample.M", SymbolType::FixedCol, 0, Some(0), true) };
    pilout.symbols = [others.clone(), vec![m]].concat();

    let values = fixed_values();
    let column = |name: &str, index: usize, column: usize| ExternalFixedColumn {
        name: name.to_string(),
        index,
        values: values[column].iter().map(to_fr).collect(),
    };
    let all: Vec<ExternalFixedColumn> = (0..4).map(|i| column("Sample.M", i, i)).collect();
    assert_eq!(with_external(&pilout, all).unwrap(), columns());

    let t = (0..4).rev().map(|id| symbol("Sample.T", SymbolType::FixedCol, id, Some(0), true));
    pilout.symbols = others.into_iter().chain(t).collect();
    let all: Vec<ExternalFixedColumn> = (0..4).map(|i| column("Sample.T", i, i)).collect();
    assert_eq!(with_external(&pilout, all).unwrap(), columns());
}

/// A pilout whose fixed symbols do not fit its columns is refused, not merged: a symbol of a
/// column the AIR does not have, and two symbols of one column.
#[test]
fn fixed_symbols_that_do_not_fit_the_columns_are_refused() {
    let (mut pilout, external) = pilout_with_external_fixed();
    pilout.symbols.push(symbol("Sample.X", SymbolType::FixedCol, 4, Some(0), true));
    let err = with_external(&pilout, external.clone()).unwrap_err();
    assert!(matches!(&err, SetupError::InvalidPilout(m) if m.contains("Sample.X")), "{err}");

    let (mut pilout, external) = pilout_with_external_fixed();
    pilout.symbols.push(symbol("Sample.X", SymbolType::FixedCol, 3, Some(0), true));
    let err = with_external(&pilout, external).unwrap_err();
    assert!(
        matches!(&err, SetupError::InvalidPilout(m) if m == "fixed column 3 has two symbols, Sample.U and Sample.X"),
        "{err}"
    );
}
