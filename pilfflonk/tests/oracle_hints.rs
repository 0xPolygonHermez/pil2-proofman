//! The Rust oracle's reference for the std's prover hints (plans M30, M31) on pilouts built in code:
//! the columns of stage 2 that `gsum_col`, `gprod_col` and `im_col` give, against a computation here
//! with `num-bigint` alone (each inverse by Fermat, `x^(r−2)`), and what it refuses. The prover's
//! columns are checked against the oracle's end to end (`cli/tests/pilfflonk_prove.rs`, on the
//! fixtures of the std's buses).
//!
//! The AIR, of `N = 8` rows, has the witness column `a` of stage 1, the columns `s` and `p` of stage
//! 2, and a challenge `g` of stage 2. Its hints are the std's, as pil2com writes them in
//! `STD_MODE_ONE_INSTANCE` (the reference an expression `column + 0`, `result` a number):
//!
//! - `gsum_col`: `s = Σ a/(a + g)`, `numerator_air` the column `a` and `denominator_air` the
//!   expression `a + g`;
//! - `gprod_col`: `p = Π (a' + g)/2`, `numerator_air` the expression `a' + g`, which reads the next
//!   row, across the wrap too, and `denominator_air` the number 2.
//!
//! The AIR with im_col (plan M31) has two more columns of stage 2, `m` and `n`, which two `im_col`
//! hints give, after the others in the pilout, and `gsum_col` reads `m`:
//!
//! - `im_col` of `m`: `m = a/(a + g)`, `numerator` the column `a` and `denominator` the expression
//!   `a + g`;
//! - `im_col` of `n`: `n = m'·(a' + g)/3`, which reads `m` at the next row, and is `a'/3`;
//! - `gsum_col`: `s = Σ m/1`, the same `s`.

use num_bigint::BigUint;
use pil2_pilout::pilout::{self as pb, expression, hint_field, operand, SymbolType};
use proofman_pilfflonk::oracle::{AirOracle, Fr, HintKind, Values};
use proofman_pilfflonk::{AirInstanceRef, FrBytes, InstanceWitness, Stage1Witness, Witness, BN254_R};

const N: usize = 8;
const G: u64 = 1000;

fn r() -> BigUint {
    BigUint::parse_bytes(BN254_R.as_bytes(), 10).unwrap()
}

fn op(o: operand::Operand) -> Option<pb::Operand> {
    Some(pb::Operand { operand: Some(o) })
}

fn column(stage: u32, col_idx: u32, row_offset: i32) -> Option<pb::Operand> {
    op(operand::Operand::WitnessCol(operand::WitnessCol { stage, col_idx, row_offset }))
}

fn constant(value: u64) -> Option<pb::Operand> {
    let bytes = if value == 0 { vec![] } else { BigUint::from(value).to_bytes_be() };
    op(operand::Operand::Constant(operand::Constant { value: bytes }))
}

fn exp(idx: u32) -> Option<pb::Operand> {
    op(operand::Operand::Expression(operand::Expression { idx }))
}

fn add(lhs: Option<pb::Operand>, rhs: Option<pb::Operand>) -> pb::Expression {
    pb::Expression { operation: Some(expression::Operation::Add(expression::Add { lhs, rhs })) }
}

fn mul(lhs: Option<pb::Operand>, rhs: Option<pb::Operand>) -> pb::Expression {
    pb::Expression { operation: Some(expression::Operation::Mul(expression::Mul { lhs, rhs })) }
}

fn field(name: &str, operand: Option<pb::Operand>) -> pb::HintField {
    pb::HintField { name: Some(name.into()), value: operand.map(hint_field::Value::Operand) }
}

/// A hint of the AIR as pil2com writes one: its fields in the array of its first field.
fn hint(name: &str, fields: Vec<pb::HintField>) -> pb::Hint {
    let array = pb::HintField {
        name: None,
        value: Some(hint_field::Value::HintFieldArray(pb::HintFieldArray { hint_fields: fields })),
    };
    pb::Hint { name: name.into(), hint_fields: vec![array], air_group_id: Some(0), air_id: Some(0) }
}

/// The std's fields of a hint of reference `reference`, `numerator_air` and `denominator_air`.
fn bus_hint(name: &str, reference: u32, numerator: Option<pb::Operand>, denominator: Option<pb::Operand>) -> pb::Hint {
    hint(
        name,
        vec![
            field("reference", exp(reference)),
            field("numerator_air", numerator),
            field("denominator_air", denominator),
            field("numerator_direct", constant(0)),
            field("denominator_direct", constant(1)),
            field("result", constant(0)),
        ],
    )
}

/// The fields of an `im_col` of reference `reference`: `numerator` and `denominator`.
fn im_col(reference: u32, numerator: Option<pb::Operand>, denominator: Option<pb::Operand>) -> pb::Hint {
    hint(
        "im_col",
        vec![field("reference", exp(reference)), field("numerator", numerator), field("denominator", denominator)],
    )
}

fn pilout() -> pb::PilOut {
    let g = op(operand::Operand::Challenge(operand::Challenge { stage: 2, idx: 0 }));
    let expressions = vec![
        add(column(2, 0, 0), constant(0)), // 0: s + 0, gsum_col's reference
        add(column(2, 1, 0), constant(0)), // 1: p + 0, gprod_col's reference
        add(column(1, 0, 0), g.clone()),   // 2: a + g
        add(column(1, 0, 1), g),           // 3: a' + g
    ];
    let air = pb::Air {
        name: Some("Buses".into()),
        num_rows: Some(N as u32),
        stage_widths: vec![1, 2],
        expressions,
        ..Default::default()
    };
    pb::PilOut {
        name: Some("buses".into()),
        base_field: r().to_bytes_be(),
        air_groups: vec![pb::AirGroup { name: Some("Buses".into()), airs: vec![air], ..Default::default() }],
        num_challenges: vec![0, 1],
        symbols: vec![pb::Symbol {
            name: "a".into(),
            r#type: SymbolType::WitnessCol as i32,
            stage: Some(1),
            ..Default::default()
        }],
        hints: vec![
            hint("gsum_debug_data", vec![]),
            bus_hint("gsum_col", 0, column(1, 0, 0), exp(2)),
            bus_hint("gprod_col", 1, exp(3), constant(2)),
        ],
        ..Default::default()
    }
}

/// The AIR with im_col (see the module): `m` and `n`, columns 2 and 3 of stage 2, and `gsum_col`
/// reading `m`. The `im_col` hints are the last ones of the pilout.
fn im_pilout() -> pb::PilOut {
    let mut pilout = pilout();
    let air = &mut pilout.air_groups[0].airs[0];
    air.stage_widths = vec![1, 4];
    air.expressions.extend([
        add(column(2, 2, 0), constant(0)), // 4: m + 0, the first im_col's reference
        add(column(2, 3, 0), constant(0)), // 5: n + 0, the second's
        mul(column(2, 2, 1), exp(3)),      // 6: m'·(a' + g)
    ]);
    pilout.hints[1] = bus_hint("gsum_col", 0, column(2, 2, 0), constant(1));
    pilout.hints.push(im_col(4, column(1, 0, 0), exp(2)));
    pilout.hints.push(im_col(5, exp(6), constant(3)));
    pilout
}

/// `a[i] = 3·i + 1`.
fn column_a() -> Vec<u64> {
    (0..N as u64).map(|i| 3 * i + 1).collect()
}

fn witness(a: &[u64]) -> Witness {
    let col: Vec<FrBytes> = a.iter().map(|&v| FrBytes::from_u64(v)).collect();
    Witness {
        instances: vec![InstanceWitness {
            air: AirInstanceRef { airgroup_id: 0, air_id: 0 },
            stage1: Stage1Witness::from_columns(N, &[col], vec![]).unwrap(),
        }],
        publics: vec![],
        proof_values: vec![],
    }
}

fn values(oracle: &AirOracle, a: &[u64], g: u64) -> Values {
    let mut values = oracle.values(&witness(a), 0).unwrap();
    values.challenges[1] = vec![Fr::from_u64(g)];
    values
}

/// `1/x mod r`, by Fermat.
fn inverse(x: &BigUint) -> BigUint {
    x.modpow(&(r() - 2u32), &r())
}

/// The two columns, with `num-bigint` alone: `s[i] = Σ_{j ≤ i} a_j/(a_j + g)` and
/// `p[i] = Π_{j ≤ i} (a_{j+1} + g)/2`, rows cyclic.
fn expected(a: &[u64], g: u64) -> (Vec<BigUint>, Vec<BigUint>) {
    let r = r();
    let (mut s, mut p) = (Vec::new(), Vec::new());
    let (mut sum, mut product) = (BigUint::ZERO, BigUint::from(1u32));
    for i in 0..N {
        sum = (sum + BigUint::from(a[i]) * inverse(&BigUint::from(a[i] + g))) % &r;
        product = (product * BigUint::from(a[(i + 1) % N] + g) * inverse(&BigUint::from(2u32))) % &r;
        s.push(sum.clone());
        p.push(product.clone());
    }
    (s, p)
}

/// The two im_col columns, with `num-bigint` alone: `m[i] = a_i/(a_i + g)` and
/// `n[i] = m_{i+1}·(a_{i+1} + g)/3 = a_{i+1}/3`, rows cyclic.
fn expected_im(a: &[u64], g: u64) -> (Vec<BigUint>, Vec<BigUint>) {
    let r = r();
    let m = (0..N).map(|i| BigUint::from(a[i]) * inverse(&BigUint::from(a[i] + g)) % &r).collect();
    let n = (0..N).map(|i| BigUint::from(a[(i + 1) % N]) * inverse(&BigUint::from(3u32)) % &r).collect();
    (m, n)
}

fn big(column: &[Fr]) -> Vec<BigUint> {
    column.iter().map(|v| v.as_biguint().clone()).collect()
}

#[test]
fn the_hints_are_the_pilouts() {
    let kinds = |pilout: &pb::PilOut| -> Vec<(HintKind, usize, usize)> {
        AirOracle::new(pilout, 0, 0).unwrap().bus_hints().iter().map(|h| (h.kind, h.stage, h.idx)).collect()
    };
    let (sum, prod, im) = (HintKind::GsumCol, HintKind::GprodCol, HintKind::ImCol);
    assert_eq!(kinds(&pilout()), [(sum, 2, 0), (prod, 2, 1)], "the debug hint is not one of them");
    assert_eq!(kinds(&im_pilout()), [(sum, 2, 0), (prod, 2, 1), (im, 2, 2), (im, 2, 3)], "in the pilout's order");
    assert_eq!([im.name(), prod.name(), sum.name()], ["im_col", "gprod_col", "gsum_col"]);
    assert!(im < prod && prod < sum, "the STARK's order");
}

#[test]
fn the_columns_are_the_running_sum_and_product_row_after_row() {
    let oracle = AirOracle::new(&pilout(), 0, 0).unwrap();
    let a = column_a();
    let columns = oracle.hint_columns(&values(&oracle, &a, G), 2).unwrap();
    let (s, p) = expected(&a, G);
    assert_eq!(columns.keys().copied().collect::<Vec<_>>(), [0, 1]);
    assert_eq!(big(&columns[&0]), s);
    assert_eq!(big(&columns[&1]), p);
    // Other challenges, other columns.
    let other = oracle.hint_columns(&values(&oracle, &a, G + 1), 2).unwrap();
    assert_ne!(other[&0], columns[&0]);
    // None of stage 1 or 3.
    assert!(oracle.hint_columns(&values(&oracle, &a, G), 1).unwrap().is_empty());
}

#[test]
fn filling_the_stage_sets_its_columns_in_the_values() {
    let oracle = AirOracle::new(&pilout(), 0, 0).unwrap();
    let a = column_a();
    let mut v = values(&oracle, &a, G);
    assert!(v.witness[1].is_empty());
    oracle.fill_hint_columns(&mut v, 2).unwrap();
    let (s, p) = expected(&a, G);
    assert_eq!((big(&v.witness[1][0]), big(&v.witness[1][1])), (s, p));
    // Their expressions read them now: s + 0 is s.
    assert_eq!(big(&oracle.expression_rows(&v, 0).unwrap()), big(&v.witness[1][0]));
}

#[test]
fn what_the_oracle_cannot_compute_is_an_error() {
    let oracle = AirOracle::new(&pilout(), 0, 0).unwrap();
    // A denominator 0 on a row: a[5] = r − g, so a[5] + g = 0.
    let mut v = values(&oracle, &column_a(), G);
    v.witness[0][0][5] = &Fr::zero() - &Fr::from_u64(G);
    let err = oracle.hint_columns(&v, 2).unwrap_err().to_string();
    assert!(err.contains("the denominator of the gsum_col hint of column 0 of stage 2 is 0 at row 5"), "{err}");
    // No challenge.
    let no_challenge = oracle.values(&witness(&column_a()), 0).unwrap();
    assert!(oracle.hint_columns(&no_challenge, 2).unwrap_err().to_string().contains("no value for challenge"));

    // A column of the stage no hint gives, and one two do.
    let mut missing = pilout();
    missing.hints.pop();
    let missing = AirOracle::new(&missing, 0, 0).unwrap();
    let err = missing.fill_hint_columns(&mut values(&missing, &column_a(), G), 2).unwrap_err().to_string();
    assert!(err.contains("no im_col, gsum_col or gprod_col hint gives column 1 of stage 2"), "{err}");
    let mut twice = pilout();
    twice.hints[2] = bus_hint("gsum_col", 0, column(1, 0, 0), exp(2));
    let twice = AirOracle::new(&twice, 0, 0).unwrap();
    assert!(twice.hint_columns(&values(&twice, &column_a(), G), 2).unwrap_err().to_string().contains("two hints"));

    // A reference that is not a column of stage 2 at its own row.
    for reference in [2, 3] {
        let mut bad = pilout();
        bad.hints[1] = bus_hint("gsum_col", reference, column(1, 0, 0), exp(2));
        let err = AirOracle::new(&bad, 0, 0).unwrap_err().to_string();
        assert!(err.contains("the reference of hint gsum_col is not a column of stage 2 or above"), "{err}");
    }
    let mut no_field = pilout();
    no_field.hints[1] = hint("gsum_col", vec![field("reference", exp(0))]);
    assert!(AirOracle::new(&no_field, 0, 0).unwrap_err().to_string().contains("has no operand numerator_air"));
}

/// The im_col columns (plan M31) are computed first, in the pilout's order, although the pilout has
/// them last, as the STARK's `calculateImHints` does before `calculateWitnessSTD`: `m` is the quotient
/// on each row, `n` reads `m` at the next row, across the wrap too, and `gsum_col` reads `m`, which
/// gives the same `s`. `p` does not change.
#[test]
fn the_im_col_columns_are_the_quotients_computed_first() {
    let oracle = AirOracle::new(&im_pilout(), 0, 0).unwrap();
    let a = column_a();
    let mut v = values(&oracle, &a, G);
    oracle.fill_hint_columns(&mut v, 2).unwrap();
    let (s, p) = expected(&a, G);
    let (m, n) = expected_im(&a, G);
    let columns: Vec<Vec<BigUint>> = v.witness[1].iter().map(|c| big(c)).collect();
    assert_eq!(columns, [s, p, m, n]);
    // The same from hint_columns, with the values' stage 2 filled already: it does not read it.
    let again = oracle.hint_columns(&v, 2).unwrap();
    assert_eq!(again.values().cloned().collect::<Vec<_>>(), v.witness[1]);
}

/// What the oracle cannot compute of an im_col: a denominator 0 on a row, which it finds before the
/// gsum_col that reads the column; an im_col that reads one after it in the pilout, whose column is
/// not computed yet (the STARK's would read what the buffer held); and a field missing.
#[test]
fn what_the_oracle_cannot_compute_of_an_im_col_is_an_error() {
    let oracle = AirOracle::new(&im_pilout(), 0, 0).unwrap();
    let mut v = values(&oracle, &column_a(), G);
    v.witness[0][0][5] = &Fr::zero() - &Fr::from_u64(G);
    let err = oracle.hint_columns(&v, 2).unwrap_err().to_string();
    assert!(err.contains("the denominator of the im_col hint of column 2 of stage 2 is 0 at row 5"), "{err}");

    let mut swapped = im_pilout();
    let last = swapped.hints.len() - 1;
    swapped.hints.swap(last - 1, last);
    let swapped = AirOracle::new(&swapped, 0, 0).unwrap();
    let err = swapped.hint_columns(&values(&swapped, &column_a(), G), 2).unwrap_err().to_string();
    assert!(err.contains("Witness { stage: 2, idx: 2 } has 0 values"), "{err}");

    let mut no_field = im_pilout();
    no_field.hints[last] = hint("im_col", vec![field("reference", exp(5)), field("numerator", exp(6))]);
    let err = AirOracle::new(&no_field, 0, 0).unwrap_err().to_string();
    assert!(err.contains("hint im_col has no operand denominator"), "{err}");
}
