//! The Rust oracle (pilfflonk/docs/README.md#tests) on a pilout built in code: the four constraint
//! domains (pilfflonk/docs/protocol.md#constraint-polynomial), every kind of operand, signed
//! offsets that wrap around, im pols, and what it refuses. The Fibonacci fixture compiled by
//! pil2com is in `tests/fibonacci.rs`.
//!
//! The AIR, of `N = 8` rows, has the witness columns `a` and `b`, a fixed column `K`, a periodic
//! column `P` of cycle `[1, 2]`, a public `p0`, a stage-1 air value `av`, a stage-1 proof value
//! `pv` and a stage-2 challenge `ch`:
//!
//! | # | Domain | Constraint |
//! |---|---|---|
//! | 0 | `firstRow` | `a − p0 − pv` |
//! | 1 | `lastRow` | `a − ch·av` |
//! | 2 | `everyFrame {1, 2}` (rows 1 … 5) | `a' − a − K` |
//! | 3 | `everyFrame {1, 0}` (rows 1 … 7) | `(−a(−1) + a − K(−1))·1` |
//! | 4 | `everyRow` | `b + (r − 1)·(a·P)` |
//!
//! so a satisfying witness has `a[i] = a[i−1] + K[i−1]` from `a[0] = p0 + pv` to `a[7] = ch·av`,
//! and `b = a·P`.

use std::collections::BTreeSet;

use num_bigint::BigUint;
use pil2_pilout::pilout::{self as pb, constraint, expression, operand, SymbolType};
use proofman_pilfflonk::oracle::{omega, AirOracle, ColumnRef, Domain, Failure, Fr, Values};
use proofman_pilfflonk::{
    oracle, AirInstanceRef, AirShape, FrBytes, InstanceWitness, Stage1Witness, Witness, WitnessShape, BN128_R,
};

const N: usize = 8;
const GOLDILOCKS: u64 = 0xffff_ffff_0000_0001;

fn r() -> BigUint {
    BigUint::parse_bytes(BN128_R.as_bytes(), 10).unwrap()
}

fn fr(v: u64) -> Fr {
    Fr::from_u64(v)
}

// ---------------------------------------------------------------------------------------------
// Building the pilout
// ---------------------------------------------------------------------------------------------

fn op(o: operand::Operand) -> Option<pb::Operand> {
    Some(pb::Operand { operand: Some(o) })
}

fn witness_col(col_idx: u32, row_offset: i32) -> Option<pb::Operand> {
    op(operand::Operand::WitnessCol(operand::WitnessCol { stage: 1, col_idx, row_offset }))
}

fn a(row_offset: i32) -> Option<pb::Operand> {
    witness_col(0, row_offset)
}

fn b() -> Option<pb::Operand> {
    witness_col(1, 0)
}

fn k(row_offset: i32) -> Option<pb::Operand> {
    op(operand::Operand::FixedCol(operand::FixedCol { idx: 0, row_offset }))
}

fn p() -> Option<pb::Operand> {
    op(operand::Operand::PeriodicCol(operand::PeriodicCol { idx: 0, row_offset: 0 }))
}

fn constant(value: &BigUint) -> Option<pb::Operand> {
    op(operand::Operand::Constant(operand::Constant { value: value.to_bytes_be() }))
}

/// The expressions of the AIR, each operation pushed and referred to by the operand it returns.
#[derive(Default)]
struct Exprs(Vec<pb::Expression>);

impl Exprs {
    fn push(&mut self, operation: expression::Operation) -> Option<pb::Operand> {
        self.0.push(pb::Expression { operation: Some(operation) });
        op(operand::Operand::Expression(operand::Expression { idx: self.0.len() as u32 - 1 }))
    }

    fn add(&mut self, lhs: Option<pb::Operand>, rhs: Option<pb::Operand>) -> Option<pb::Operand> {
        self.push(expression::Operation::Add(expression::Add { lhs, rhs }))
    }

    fn sub(&mut self, lhs: Option<pb::Operand>, rhs: Option<pb::Operand>) -> Option<pb::Operand> {
        self.push(expression::Operation::Sub(expression::Sub { lhs, rhs }))
    }

    fn mul(&mut self, lhs: Option<pb::Operand>, rhs: Option<pb::Operand>) -> Option<pb::Operand> {
        self.push(expression::Operation::Mul(expression::Mul { lhs, rhs }))
    }

    fn neg(&mut self, value: Option<pb::Operand>) -> Option<pb::Operand> {
        self.push(expression::Operation::Neg(expression::Neg { value }))
    }

    fn last(&self) -> u32 {
        self.0.len() as u32 - 1
    }
}

fn expression_idx(idx: u32) -> Option<operand::Expression> {
    Some(operand::Expression { idx })
}

fn symbol(name: &str, kind: SymbolType, id: u32, stage: u32) -> pb::Symbol {
    pb::Symbol { name: name.into(), r#type: kind as i32, id, stage: Some(stage), ..Default::default() }
}

/// `K`, with a value of every size: small ones and `r − 1`.
fn k_values() -> Vec<Fr> {
    [3, 1, 4, 1, 5, 0, 2, 6].iter().enumerate().map(|(i, v)| if i == 5 { -&fr(1) } else { fr(*v) }).collect()
}

/// The expression of `a·P`, an im pol candidate.
const A_TIMES_P: usize = 10;

fn pilout() -> pb::PilOut {
    let mut e = Exprs::default();
    let p0 = op(operand::Operand::PublicValue(operand::PublicValue { idx: 0 }));
    let pv = op(operand::Operand::ProofValue(operand::ProofValue { stage: 1, idx: 0 }));
    let ch = op(operand::Operand::Challenge(operand::Challenge { stage: 2, idx: 0 }));
    let av = op(operand::Operand::AirValue(operand::AirValue { idx: 0 }));

    let e0 = e.sub(a(0), p0);
    e.sub(e0, pv);
    let c0 = e.last();
    let ch_av = e.mul(ch, av);
    e.sub(a(0), ch_av);
    let c1 = e.last();
    let next = e.sub(a(1), a(0));
    e.sub(next, k(0));
    let c2 = e.last();
    let prev = e.neg(a(-1));
    let diff = e.add(prev, a(0));
    let diff = e.sub(diff, k(-1));
    e.mul(diff, constant(&BigUint::from(1u32)));
    let c3 = e.last();
    let a_p = e.mul(a(0), p());
    assert_eq!(e.last() as usize, A_TIMES_P);
    let scaled = e.mul(constant(&(r() - 1u32)), a_p);
    e.add(b(), scaled);
    let c4 = e.last();

    use constraint::Constraint as C;
    let constraints = vec![
        C::FirstRow(constraint::FirstRow { expression_idx: expression_idx(c0), debug_line: Some("c0".into()) }),
        C::LastRow(constraint::LastRow { expression_idx: expression_idx(c1), debug_line: Some("c1".into()) }),
        C::EveryFrame(constraint::EveryFrame {
            expression_idx: expression_idx(c2),
            offset_min: 1,
            offset_max: 2,
            debug_line: Some("c2".into()),
        }),
        C::EveryFrame(constraint::EveryFrame {
            expression_idx: expression_idx(c3),
            offset_min: 1,
            offset_max: 0,
            debug_line: Some("c3".into()),
        }),
        C::EveryRow(constraint::EveryRow { expression_idx: expression_idx(c4), debug_line: Some("c4".into()) }),
    ];
    let bytes = |v: &Fr| v.as_biguint().to_bytes_be();
    let air = pb::Air {
        name: Some("Synthetic".into()),
        num_rows: Some(N as u32),
        periodic_cols: vec![pb::PeriodicCol { values: vec![vec![1], vec![2]] }],
        fixed_cols: vec![pb::FixedCol { values: k_values().iter().map(bytes).collect() }],
        stage_widths: vec![2],
        expressions: e.0,
        constraints: constraints.into_iter().map(|c| pb::Constraint { constraint: Some(c) }).collect(),
        air_values: vec![pb::AirValue { stage: 1 }],
        ..Default::default()
    };
    pb::PilOut {
        name: Some("synthetic".into()),
        base_field: r().to_bytes_be(),
        air_groups: vec![pb::AirGroup { name: Some("Synthetic".into()), airs: vec![air], ..Default::default() }],
        num_challenges: vec![0, 1],
        num_proof_values: vec![1],
        num_public_values: 1,
        symbols: vec![
            symbol("Synthetic.a", SymbolType::WitnessCol, 0, 1),
            symbol("Synthetic.b", SymbolType::WitnessCol, 1, 1),
            symbol("p0", SymbolType::PublicValue, 0, 1),
            symbol("pv", SymbolType::ProofValue, 0, 1),
        ],
        ..Default::default()
    }
}

// ---------------------------------------------------------------------------------------------
// A witness
// ---------------------------------------------------------------------------------------------

const PV: u64 = 7;
const AV: u64 = 5;
const CH: u64 = 4;

/// The satisfying witness of the module's table, as the prover would get it.
fn good_witness() -> Witness {
    let k = k_values();
    let p = [fr(1), fr(2)];
    // a[7] = ch·av and a[i] = a[i−1] + K[i−1]: a[0] = ch·av − Σ_{j<7} K[j], and p0 = a[0] − pv.
    let sum_k = k[..N - 1].iter().fold(Fr::zero(), |acc, v| &acc + v);
    let mut col_a = vec![&(&fr(CH) * &fr(AV)) - &sum_k];
    for i in 1..N {
        col_a.push(&col_a[i - 1] + &k[i - 1]);
    }
    let col_b: Vec<Fr> = col_a.iter().enumerate().map(|(i, v)| v * &p[i % 2]).collect();
    let p0 = &col_a[0] - &fr(PV);
    let to_bytes = |col: &[Fr]| col.iter().map(Fr::to_bytes).collect::<Vec<FrBytes>>();
    let stage1 = Stage1Witness::from_columns(N, &[to_bytes(&col_a), to_bytes(&col_b)], vec![FrBytes::from_u64(AV)]);
    Witness {
        instances: vec![InstanceWitness { air: AirInstanceRef { airgroup_id: 0, air_id: 0 }, stage1: stage1.unwrap() }],
        publics: vec![p0.to_bytes()],
        proof_values: vec![FrBytes::from_u64(PV)],
    }
}

fn good_values(oracle: &AirOracle) -> Values {
    let mut values = oracle.values(&good_witness(), 0).unwrap();
    values.challenges[1] = vec![fr(CH)];
    values
}

/// A cell to mutate, `(column, row)`, and the `(constraint, row)` that must then fail.
type Mutation = (usize, usize, &'static [(usize, usize)]);

fn failures(oracle: &AirOracle, values: &Values) -> BTreeSet<(usize, usize)> {
    oracle.check(values).unwrap().into_iter().map(|f| (f.constraint, f.row)).collect()
}

fn set(expected: &[(usize, usize)]) -> BTreeSet<(usize, usize)> {
    expected.iter().copied().collect()
}

/// A point off `H`, fixed so that the test is deterministic: as good as a random one.
fn z() -> Fr {
    fr(7).pow_u64(1000)
}

// ---------------------------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------------------------

#[test]
fn the_shape_and_the_values_come_from_the_pilout() {
    let pilout = pilout();
    let shape = oracle::witness_shape(&pilout).unwrap();
    let air = AirShape { airgroup_id: 0, air_id: 0, n_bits: 3, n_cols: 2, n_air_values: 1 };
    assert_eq!(shape, WitnessShape::new(vec![air], 1, 1).unwrap());
    shape.check(&good_witness()).unwrap();

    let oracle = AirOracle::new(&pilout, 0, 0).unwrap();
    assert_eq!((oracle.n_bits(), oracle.n_rows()), (3, N));
    assert_eq!(oracle.omega(), &omega(3).unwrap());
    assert_eq!(oracle.fixed(0).unwrap(), k_values());
    let domains: Vec<Domain> = oracle.constraints().iter().map(|c| c.domain).collect();
    assert_eq!(
        domains,
        [
            Domain::FirstRow,
            Domain::LastRow,
            Domain::EveryFrame { offset_min: 1, offset_max: 2 },
            Domain::EveryFrame { offset_min: 1, offset_max: 0 },
            Domain::EveryRow
        ]
    );
    assert_eq!(oracle.constraints()[3].debug_line, "c3");

    let values = oracle.values(&good_witness(), 0).unwrap();
    assert_eq!(values.witness.len(), 1);
    assert_eq!(values.witness[0].len(), 2);
    assert_eq!(values.air_values, [Some(fr(AV))]);
    assert_eq!(values.proof_values, [Some(fr(PV))]);
    assert_eq!(values.challenges, [vec![], vec![]], "the challenges are the caller's");
    assert!(values.airgroup_values.is_empty());
}

#[test]
fn a_good_witness_satisfies_every_constraint_on_its_domain_only() {
    let oracle = AirOracle::new(&pilout(), 0, 0).unwrap();
    let values = good_values(&oracle);
    assert_eq!(oracle.check(&values).unwrap(), []);

    // Out of their domains, the boundary constraints are not 0: the domains are what makes them
    // hold.
    let numerators = oracle.numerators(&values).unwrap();
    let zero_rows = |c: usize| (0..N).filter(|&row| numerators[c][row].is_zero()).collect::<Vec<_>>();
    assert_eq!(zero_rows(0), [0]);
    assert_eq!(zero_rows(1), [7]);
    assert_eq!(zero_rows(2), [0, 1, 2, 3, 4, 5, 6], "a' − a − K holds on row 0 too, but not across the wrap");
    assert_eq!(zero_rows(3), [1, 2, 3, 4, 5, 6, 7]);
    assert_eq!(zero_rows(4), (0..N).collect::<Vec<_>>());
}

#[test]
fn a_mutated_cell_fails_the_constraints_that_read_it_on_the_rows_of_their_domains() {
    let oracle = AirOracle::new(&pilout(), 0, 0).unwrap();
    let cases: [Mutation; 5] = [
        // a[3]: c2 on rows 2 and 3, c3 on rows 3 and 4, c4 on row 3.
        (0, 3, &[(2, 2), (2, 3), (3, 3), (3, 4), (4, 3)]),
        // a[0]: c0 and c4 on row 0, c3 on row 1; c2 on row 0 and c3 on row 0 are out of their domains.
        (0, 0, &[(0, 0), (3, 1), (4, 0)]),
        // a[7]: c1, c3 and c4 on row 7; c3 on row 0 reads a[7] at offset −1 but is out of its domain.
        (0, 7, &[(1, 7), (3, 7), (4, 7)]),
        // a[6]: c2 on row 5 (row 6 is out of its domain), c3 on rows 6 and 7, c4 on row 6.
        (0, 6, &[(2, 5), (3, 6), (3, 7), (4, 6)]),
        // b[2]: c4 on row 2 only.
        (1, 2, &[(4, 2)]),
    ];
    for (col, row, expected) in cases {
        let mut values = good_values(&oracle);
        values.witness[0][col][row] = &values.witness[0][col][row] + &fr(1);
        assert_eq!(failures(&oracle, &values), set(expected), "column {col}, row {row}");
    }

    // A failure carries the value of the numerator.
    let mut values = good_values(&oracle);
    values.witness[0][1][2] = &values.witness[0][1][2] + &fr(5);
    assert_eq!(oracle.check(&values).unwrap(), [Failure { constraint: 4, row: 2, value: fr(5) }]);

    // The scalars: the public, the proof value, the air value and the challenge.
    let mut values = good_values(&oracle);
    values.publics[0] = &values.publics[0] + &fr(1);
    assert_eq!(failures(&oracle, &values), set(&[(0, 0)]));
    let mut values = good_values(&oracle);
    values.proof_values[0] = Some(fr(PV + 1));
    assert_eq!(failures(&oracle, &values), set(&[(0, 0)]));
    let mut values = good_values(&oracle);
    values.air_values[0] = Some(fr(AV + 1));
    assert_eq!(failures(&oracle, &values), set(&[(1, 7)]));
    let mut values = good_values(&oracle);
    values.challenges[1][0] = fr(CH + 1);
    assert_eq!(failures(&oracle, &values), set(&[(1, 7)]));
}

#[test]
fn columns_at_points_are_their_interpolants_with_the_offset_on_the_point() {
    let oracle = AirOracle::new(&pilout(), 0, 0).unwrap();
    let values = good_values(&oracle);
    let w = omega(3).unwrap();
    let a_col = ColumnRef::Witness { stage: 1, idx: 0 };
    // On H: rows, cyclic.
    assert_eq!(oracle.column_at(&values, a_col, 0, &w.pow_u64(3)).unwrap(), values.witness[0][0][3]);
    assert_eq!(oracle.column_at(&values, a_col, 1, &w.pow_u64(3)).unwrap(), values.witness[0][0][4]);
    assert_eq!(oracle.column_at(&values, a_col, -1, &Fr::one()).unwrap(), values.witness[0][0][7]);
    assert_eq!(oracle.column_at(&values, ColumnRef::Periodic(0), 0, &w.pow_u64(5)).unwrap(), fr(2));
    let im = ColumnRef::Im(A_TIMES_P);
    assert_eq!(oracle.column_at(&values, im, 0, &w.pow_u64(5)).unwrap(), values.witness[0][1][5], "a·P = b");
    // Off H, an offset s moves the point to z·ω^s.
    let z = z();
    for s in [-3, -1, 1, 2, 9] {
        let shifted = &z * &w.pow_u64((s as i64).rem_euclid(N as i64) as u64);
        assert_eq!(
            oracle.column_at(&values, a_col, s, &z).unwrap(),
            oracle.column_at(&values, a_col, 0, &shifted).unwrap(),
            "offset {s}"
        );
    }
}

#[test]
fn q_at_a_point_is_the_polynomial_of_the_exact_divisions() {
    let oracle = AirOracle::new(&pilout(), 0, 0).unwrap();
    let values = good_values(&oracle);
    let std_vc = fr(11);
    let q = oracle.q_polynomial(&values, &[], &std_vc).unwrap();
    assert!(q.is_exact(), "{:?}", q.remainders);
    for z in [z(), fr(2), -&fr(3)] {
        assert_eq!(oracle.q_at(&values, &[], &std_vc, &z).unwrap(), q.evaluate(&z), "at {z}");
    }
    // Each quotient has degree 6 at most: a column has degree 7, over X − 1 or X − ω^7 for c0
    // and c1, and a·P has degree 7 + 4 (P = 3/2 − X^4/2) over X^8 − 1 for c4.
    assert!(!q.coefficients.is_empty() && q.coefficients.len() <= 7, "{} coefficients", q.coefficients.len());

    // The last term of the fold gets std_vc^0: with std_vc = 0, Q(z) = c4(z)/(z^N − 1).
    let z = z();
    let at = |col, offset| oracle.column_at(&values, col, offset, &z).unwrap();
    let (a_z, b_z, p_z) = (
        at(ColumnRef::Witness { stage: 1, idx: 0 }, 0),
        at(ColumnRef::Witness { stage: 1, idx: 1 }, 0),
        at(ColumnRef::Periodic(0), 0),
    );
    let c4 = &b_z - &(&a_z * &p_z);
    let expected = &c4 * &(&z.pow_u64(N as u64) - &Fr::one()).inv().unwrap();
    assert_eq!(oracle.q_at(&values, &[], &Fr::zero(), &z).unwrap(), expected);
    assert_eq!(oracle.q_polynomial(&values, &[], &Fr::zero()).unwrap().evaluate(&z), expected);
}

#[test]
fn a_mutated_witness_does_not_make_q_a_polynomial() {
    let oracle = AirOracle::new(&pilout(), 0, 0).unwrap();
    let mut values = good_values(&oracle);
    values.witness[0][0][3] = &values.witness[0][0][3] + &fr(1);
    let std_vc = fr(11);
    let q = oracle.q_polynomial(&values, &[], &std_vc).unwrap();
    let terms: Vec<usize> = q.remainders.iter().map(|(term, _)| *term).collect();
    assert_eq!(terms, [2, 3, 4], "the constraints that fail on some row of their domain");
    let z = z();
    assert_ne!(oracle.q_at(&values, &[], &std_vc, &z).unwrap(), q.evaluate(&z));
}

#[test]
fn im_pols_change_the_fold_but_not_what_it_proves() {
    let oracle = AirOracle::new(&pilout(), 0, 0).unwrap();
    let values = good_values(&oracle);
    let std_vc = fr(11);
    let z = z();
    let with_im = oracle.q_polynomial(&values, &[A_TIMES_P], &std_vc).unwrap();
    assert!(with_im.is_exact());
    assert_eq!(oracle.q_at(&values, &[A_TIMES_P], &std_vc, &z).unwrap(), with_im.evaluate(&z));
    let without = oracle.q_polynomial(&values, &[], &std_vc).unwrap();
    assert_ne!(with_im.coefficients, without.coefficients, "another fold, with one more term");

    // Its own term, im − a·P, is exact whatever b is; with b mutated c4 is not.
    let mut values = good_values(&oracle);
    values.witness[0][1][4] = &values.witness[0][1][4] + &fr(1);
    let terms: Vec<usize> =
        oracle.q_polynomial(&values, &[A_TIMES_P], &std_vc).unwrap().remainders.iter().map(|(t, _)| *t).collect();
    assert_eq!(terms, [4]);
}

#[test]
fn the_oracle_refuses_what_it_cannot_evaluate() {
    let err = |result: Result<AirOracle, proofman_pilfflonk::PilfflonkError>| result.unwrap_err().to_string();

    let mut goldilocks = pilout();
    goldilocks.base_field = BigUint::from(GOLDILOCKS).to_bytes_be();
    assert!(err(AirOracle::new(&goldilocks, 0, 0)).contains("not over BN128"));
    assert!(err(AirOracle::new(&pilout(), 0, 1)).contains("no air 1"));
    assert!(err(AirOracle::new(&pilout(), 1, 0)).contains("no airgroup 1"));

    let mut to_file = pilout();
    to_file.air_groups[0].airs[0].fixed_cols[0].values.clear();
    assert!(err(AirOracle::new(&to_file, 0, 0)).contains("fixed columns to a file"));

    let mut wide = pilout();
    wide.air_groups[0].airs[0].fixed_cols[0].values[2] = r().to_bytes_be();
    assert!(err(AirOracle::new(&wide, 0, 0)).contains("not below r"));

    let mut rows = pilout();
    rows.air_groups[0].airs[0].num_rows = Some(6);
    assert!(err(AirOracle::new(&rows, 0, 0)).contains("not a power of two"));

    let mut frame = pilout();
    if let Some(constraint::Constraint::EveryFrame(c)) = &mut frame.air_groups[0].airs[0].constraints[2].constraint {
        c.offset_max = 7;
    }
    assert!(err(AirOracle::new(&frame, 0, 0)).contains("no row of 8 is left"));

    let mut dangling = pilout();
    if let Some(constraint::Constraint::EveryRow(c)) = &mut dangling.air_groups[0].airs[0].constraints[4].constraint {
        c.expression_idx = expression_idx(99);
    }
    assert!(err(AirOracle::new(&dangling, 0, 0)).contains("constraint 4 has no expression"));

    // On evaluation: a missing challenge, a custom column, a cycle, z in H, an im pol twice.
    let oracle = AirOracle::new(&pilout(), 0, 0).unwrap();
    let no_challenge = oracle.values(&good_witness(), 0).unwrap();
    assert!(oracle.check(&no_challenge).unwrap_err().to_string().contains("no value for challenge 0 of stage 2"));

    let mut custom = pilout();
    let exprs = &mut custom.air_groups[0].airs[0].expressions;
    exprs[0] = pb::Expression {
        operation: Some(expression::Operation::Neg(expression::Neg {
            value: op(operand::Operand::CustomCol(operand::CustomCol {
                commit_id: 0,
                stage: 1,
                col_idx: 0,
                row_offset: 0,
            })),
        })),
    };
    let custom = AirOracle::new(&custom, 0, 0).unwrap();
    assert!(custom.check(&good_values(&custom)).unwrap_err().to_string().contains("custom commit"));

    let mut cycle = pilout();
    let exprs = &mut cycle.air_groups[0].airs[0].expressions;
    exprs[0] = pb::Expression {
        operation: Some(expression::Operation::Neg(expression::Neg {
            value: op(operand::Operand::Expression(operand::Expression { idx: 1 })),
        })),
    };
    let cycle = AirOracle::new(&cycle, 0, 0).unwrap();
    assert!(cycle.check(&good_values(&cycle)).unwrap_err().to_string().contains("refers to itself"));

    let values = good_values(&oracle);
    let w = omega(3).unwrap();
    assert!(oracle.q_at(&values, &[], &fr(1), &w.pow_u64(2)).unwrap_err().to_string().contains("is in H"));
    assert!(oracle.q_at(&values, &[A_TIMES_P, A_TIMES_P], &fr(1), &z()).unwrap_err().to_string().contains("twice"));
    assert!(oracle.q_polynomial(&values, &[99], &fr(1)).unwrap_err().to_string().contains("not an expression"));

    // A witness of another AIR, or of another size.
    let mut other = good_witness();
    other.instances[0].air.air_id = 1;
    assert!(oracle.values(&other, 0).unwrap_err().to_string().contains("is of air 0/1"));
    let mut short = good_witness();
    short.instances[0].stage1 =
        Stage1Witness::from_columns(4, &vec![vec![FrBytes::ZERO; 4]; 2], vec![FrBytes::ZERO]).unwrap();
    assert!(oracle.values(&short, 0).unwrap_err().to_string().contains("4 rows"));
    let mut no_proof_value = good_witness();
    no_proof_value.proof_values.clear();
    assert!(oracle.values(&no_proof_value, 0).unwrap_err().to_string().contains("0 proof values"));
}
