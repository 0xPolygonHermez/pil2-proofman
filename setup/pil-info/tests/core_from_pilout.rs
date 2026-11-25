//! `from_pilout`: strict ingestion into the typed IR.

mod common;

use common::*;
use num_bigint::BigUint;
use pil2_pilout::pilout::{self as pb, operand, SymbolType};
use pil_info::core::from_pilout::{from_pilout, PiloutError};
use pil_info::core::ir::{degree, Boundary, ColType, Expr};
use pil_info::{FieldCfg, Opening};

fn col(id: u32, offset: i64) -> Expr {
    Expr::Col { col: ColType::Witness, commit_id: None, id, offset }
}

#[test]
fn rejects_missing_expression_idx() {
    let p = pilout(&["a"], &[], vec![sub(witness(0, 0), constant("1"))], vec![every_row(0), every_row_no_idx()]);
    let err = from_pilout(&p, 0, 0, &gl(), Opening::Deep).unwrap_err();
    assert_eq!(err, PiloutError::MissingExpressionIdx { constraint: 1 });
    assert!(err.to_string().contains("constraint 1"));
}

#[test]
fn rejects_empty_constraint() {
    let p = pilout(&["a"], &[], vec![sub(witness(0, 0), constant("1"))], vec![empty_constraint()]);
    assert_eq!(
        from_pilout(&p, 0, 0, &gl(), Opening::Deep).unwrap_err(),
        PiloutError::EmptyConstraint { constraint: 0 }
    );
}

#[test]
fn deep_rejects_first_row_last_row_every_frame() {
    for (c, name) in [(first_row(0), "firstRow"), (last_row(0), "lastRow"), (every_frame(0, 0, 1), "everyFrame")] {
        let p = pilout(&["a"], &[], vec![sub(witness(0, 0), constant("1"))], vec![c]);
        match from_pilout(&p, 0, 0, &gl(), Opening::Deep).unwrap_err() {
            PiloutError::BoundaryNotEveryRow { constraint: 0, line, boundary } => {
                assert_eq!(boundary, name);
                assert!(!line.is_empty());
            }
            e => panic!("unexpected {e:?}"),
        }
    }
}

#[test]
fn shplonk_keeps_first_row_and_last_row() {
    let p = pilout(
        &["a"],
        &[],
        vec![sub(witness(0, 0), constant("1"))],
        vec![first_row(0), last_row(0), every_frame(0, 1, 2), every_row(0)],
    );
    let cs = from_pilout(&p, 0, 0, &FieldCfg::bn128(), Opening::Shplonk).unwrap();
    let b: Vec<_> = cs.constraints.iter().map(|c| c.boundary.clone()).collect();
    assert_eq!(
        b,
        vec![Boundary::FirstRow, Boundary::LastRow, Boundary::EveryFrame { min: 1, max: 2 }, Boundary::EveryRow]
    );
}

#[test]
fn rejects_dangling_and_cyclic_ref() {
    let p = pilout(&["a"], &[], vec![add(exp(5), witness(0, 0))], vec![every_row(0)]);
    assert_eq!(from_pilout(&p, 0, 0, &gl(), Opening::Deep).unwrap_err(), PiloutError::DanglingRef { expr: 0, to: 5 });
    let p = pilout(&["a"], &[], vec![add(exp(1), witness(0, 0)), add(exp(0), witness(0, 0))], vec![every_row(0)]);
    assert!(matches!(from_pilout(&p, 0, 0, &gl(), Opening::Deep).unwrap_err(), PiloutError::CyclicRef { .. }));
    let p = pilout(&["a"], &[], vec![add(exp(0), witness(0, 0))], vec![every_row(0)]);
    assert_eq!(from_pilout(&p, 0, 0, &gl(), Opening::Deep).unwrap_err(), PiloutError::CyclicRef { expr: 0 });
}

#[test]
fn rejects_unknown_air_and_bad_operand() {
    let p = pilout(&["a"], &[], vec![add(witness(0, 0), constant("1"))], vec![every_row(0)]);
    assert_eq!(
        from_pilout(&p, 3, 0, &gl(), Opening::Deep).unwrap_err(),
        PiloutError::UnknownAir { airgroup: 3, air: 0 }
    );
    let p = pilout(&["a"], &[], vec![add(witness(0, 0), None)], vec![every_row(0)]);
    assert!(matches!(
        from_pilout(&p, 0, 0, &gl(), Opening::Deep).unwrap_err(),
        PiloutError::BadOperand { expr: 0, .. }
    ));
    let p = pilout(&["a"], &[], vec![add(witness(0, 0), constant("1"))], vec![every_row(9)]);
    assert!(matches!(from_pilout(&p, 0, 0, &gl(), Opening::Deep).unwrap_err(), PiloutError::BadOperand { .. }));
}

#[test]
fn lowers_column_kinds_and_offsets() {
    let mut p = pilout(
        &["a", "b"],
        &["L1"],
        vec![
            add(witness(0, -1), fixed(0)),    // 0
            mul(challenge(1, 0), public(2)),  // 1
            sub(air_value(0), constant("7")), // 2
        ],
        vec![every_row(0)],
    );
    p.num_challenges = vec![1];
    p.num_public_values = 3;
    p.symbols.push(symbol("c", SymbolType::Challenge, 1, 0));
    p.symbols.extend((0..3).map(|i| symbol("p", SymbolType::PublicValue, 1, i)));
    p.air_groups[0].airs[0].air_values = vec![pb::AirValue { stage: 1 }];
    let cs = from_pilout(&p, 0, 0, &gl(), Opening::Deep).unwrap();
    let e0 =
        Expr::Add(Box::new(col(0, -1)), Box::new(Expr::Col { col: ColType::Fixed, commit_id: None, id: 0, offset: 0 }));
    let e1 = Expr::Mul(Box::new(Expr::Challenge { stage: 1, stage_id: 0 }), Box::new(Expr::Public(2)));
    let e2 = Expr::Sub(Box::new(Expr::AirValue(0)), Box::new(Expr::Const(BigUint::from(7u32))));
    let mut got = cs.exprs.clone();
    got.sort();
    assert_eq!(got, [e0.clone(), e2, e1]);
    assert_eq!(cs.constraints[0].e, Expr::Ref(find(&cs, &e0)));
    assert_eq!(cs.constraints[0].label, "constraint on expression 0");
    assert_eq!(cs.n_rows_bits, 4);
    assert_eq!(cs.n_stages, 1);
}

#[test]
fn keeps_pilout_order() {
    let p = pilout(
        &["z", "a", "m"],
        &["F1", "F0"],
        vec![add(witness(0, 0), witness(1, 0)), sub(witness(2, 0), constant("1"))],
        vec![every_row(1), every_row(0)],
    );
    let cs = from_pilout(&p, 0, 0, &gl(), Opening::Deep).unwrap();
    let names = |v: &[pil_info::core::ir::Column]| v.iter().map(|c| c.name.clone()).collect::<Vec<_>>();
    assert_eq!(names(&cs.witness), ["z", "a", "m"]);
    assert_eq!(names(&cs.fixed), ["F1", "F0"]);
    assert_eq!(cs.constraints.iter().map(|c| c.e.clone()).collect::<Vec<_>>(), [Expr::Ref(1), Expr::Ref(0)]);
    assert_eq!(cs.exprs.len(), 2);
}

#[test]
fn degree_follows_rules() {
    let p = pilout(
        &["a", "b", "c"],
        &[],
        vec![
            mul(witness(0, 0), witness(1, 0)),   // 0: a*b
            mul(exp(0), witness(2, 0)),          // 1: Ref->a*b times c = 3
            mul(constant("5"), witness(0, 0)),   // 2: const*a
            mul(challenge(1, 0), witness(0, 0)), // 3: challenge*a
            add(exp(0), witness(0, 0)),          // 4: max
        ],
        vec![every_row(1)],
    );
    let mut p = p;
    p.num_challenges = vec![1];
    p.symbols.push(symbol("alpha", SymbolType::Challenge, 1, 0));
    let cs = from_pilout(&p, 0, 0, &gl(), Opening::Deep).unwrap();
    let mul_e = |a: Expr, b: Expr| Expr::Mul(Box::new(a), Box::new(b));
    let ab = mul_e(col(0, 0), col(1, 0));
    let bodies = [
        ab.clone(),
        mul_e(ab.clone(), col(2, 0)),
        mul_e(Expr::Const(BigUint::from(5u32)), col(0, 0)),
        mul_e(Expr::Challenge { stage: 1, stage_id: 0 }, col(0, 0)),
        Expr::Add(Box::new(ab), Box::new(col(0, 0))),
    ];
    let d = |e: &Expr| degree(&cs, &Expr::Ref(find(&cs, e))).unwrap();
    assert_eq!(bodies.each_ref().map(d), [2, 3, 1, 1, 2]);
    assert_eq!(degree(&cs, &Expr::Const(BigUint::from(3u32))), Ok(0));
    assert_eq!(degree(&cs, &Expr::Neg(Box::new(col(0, 0)))), Ok(1));
}

/// Every air of every example pilout lowers, the global constraints too.
#[test]
fn real_pilouts_lower() {
    for (f, p) in real_pilouts() {
        pil_info::core::global::global_from_pilout(&p, &gl()).unwrap_or_else(|e| panic!("{f:?} global: {e}"));
        for (g, ag) in p.air_groups.iter().enumerate() {
            for a in 0..ag.airs.len() {
                from_pilout(&p, g, a, &gl(), Opening::Deep).unwrap_or_else(|e| panic!("{f:?} {g}/{a}: {e}"));
            }
        }
    }
}

fn rejects(p: &pb::PilOut) {
    assert!(matches!(from_pilout(p, 0, 0, &gl(), Opening::Deep).unwrap_err(), PiloutError::BadOperand { expr: 0, .. }));
}

#[test]
fn rejects_out_of_range_operands() {
    let mk = |o| pilout(&["a"], &["F"], vec![add(o, constant("1"))], vec![every_row(0)]);
    // witness: column past the stage width, and a stage past the widths or 0
    rejects(&mk(witness(1, 0)));
    rejects(&mk(witness_at(2, 0, 0)));
    rejects(&mk(witness_at(0, 0, 0)));
    // fixed and periodic
    rejects(&mk(fixed(1)));
    rejects(&mk(op(operand::Operand::PeriodicCol(operand::PeriodicCol { idx: 0, row_offset: 0 }))));
    // challenge (numChallenges = [0]), public (0), air/airgroup/proof values (none)
    rejects(&mk(challenge(1, 0)));
    rejects(&mk(public(0)));
    rejects(&mk(air_value(0)));
    rejects(&mk(op(operand::Operand::AirGroupValue(operand::AirGroupValue { idx: 0 }))));
    rejects(&mk(op(operand::Operand::ProofValue(operand::ProofValue { stage: 1, idx: 0 }))));
    rejects(&mk(op(operand::Operand::CustomCol(operand::CustomCol {
        commit_id: 0,
        stage: 0,
        col_idx: 0,
        row_offset: 0,
    }))));
    // and the in-range boundary is accepted
    let ok = pilout(&["a"], &["F"], vec![add(witness(0, 0), fixed(0))], vec![every_row(0)]);
    assert!(from_pilout(&ok, 0, 0, &gl(), Opening::Deep).is_ok());
}

#[test]
fn custom_columns_are_stage_zero_and_bounded() {
    let mut p = pilout(
        &["a"],
        &[],
        vec![add(
            op(operand::Operand::CustomCol(operand::CustomCol { commit_id: 0, stage: 0, col_idx: 1, row_offset: 0 })),
            witness(0, 0),
        )],
        vec![every_row(0)],
    );
    p.air_groups[0].airs[0].custom_commits =
        vec![pb::CustomCommit { name: Some("c".into()), stage_widths: vec![2], ..Default::default() }];
    let cs = from_pilout(&p, 0, 0, &gl(), Opening::Deep).unwrap();
    assert!(
        matches!(&cs.exprs[0], Expr::Add(l, _) if **l == Expr::Col { col: ColType::Custom, commit_id: Some(0), id: 1, offset: 0 })
    );
    p.air_groups[0].airs[0].custom_commits[0].stage_widths = vec![1];
    rejects(&p);
    p.air_groups[0].airs[0].custom_commits[0].stage_widths = vec![2];
    p.air_groups[0].airs[0].expressions = vec![add(
        op(operand::Operand::CustomCol(operand::CustomCol { commit_id: 0, stage: 1, col_idx: 0, row_offset: 0 })),
        witness(0, 0),
    )];
    rejects(&p);
}

#[test]
fn rejects_surplus_value_symbols() {
    let mut p = pilout(&["a"], &[], vec![add(witness(0, 0), constant("1"))], vec![every_row(0)]);
    p.symbols.push(symbol("av", SymbolType::AirValue, 1, 0));
    assert!(from_pilout(&p, 0, 0, &gl(), Opening::Deep).is_err());
}

#[test]
fn rejects_a_periodic_operand() {
    let periodic = op(operand::Operand::PeriodicCol(operand::PeriodicCol { idx: 0, row_offset: 0 }));
    let mut p = pilout(&["a"], &[], vec![sub(witness(0, 0), periodic)], vec![every_row(0)]);
    p.air_groups[0].airs[0].periodic_cols = vec![Default::default()];
    let err = from_pilout(&p, 0, 0, &gl(), Opening::Deep).unwrap_err();
    assert!(matches!(err, PiloutError::BadOperand { expr: 0, .. }), "{err:?}");
}

#[test]
fn publics_are_tied_to_num_public_values() {
    let mut p = pilout(&["a"], &[], vec![sub(witness(0, 0), constant("1"))], vec![every_row(0)]);
    p.num_public_values = 2;
    p.symbols.push(symbol("p", SymbolType::PublicValue, 1, 0));
    let err = from_pilout(&p, 0, 0, &gl(), Opening::Deep).unwrap_err();
    assert_eq!(err, PiloutError::PublicCount { declared: 2, named: 1 });
    p.symbols.push(symbol("q", SymbolType::PublicValue, 1, 1));
    p.air_groups[0].airs[0].custom_commits = vec![pb::CustomCommit {
        name: Some("rom".into()),
        public_values: vec![pb::global_operand::PublicValue { idx: 2 }],
        ..Default::default()
    }];
    let err = from_pilout(&p, 0, 0, &gl(), Opening::Deep).unwrap_err();
    assert_eq!(err, PiloutError::CommitPublic { commit: "rom".into(), idx: 2, declared: 2 });
}

/// Witness, fixed and challenge symbols must name each declared slot once: ids are counted per stage downstream.
#[test]
fn rejects_symbol_count_mismatch() {
    let base = || pilout(&["a", "b"], &["L"], vec![mul(witness(0, 0), witness(1, 0))], vec![every_row(0)]);
    assert!(from_pilout(&base(), 0, 0, &gl(), Opening::Deep).is_ok());
    let mut p = base();
    p.air_groups[0].airs[0].stage_widths = vec![3];
    let mut q = base();
    q.air_groups[0].airs[0].fixed_cols.push(pb::FixedCol { values: Vec::new() });
    let mut r = base();
    r.num_challenges = vec![1];
    let mut d = base();
    d.symbols.push(symbol("dup", SymbolType::WitnessCol, 1, 1));
    d.air_groups[0].airs[0].stage_widths = vec![3];
    for p in [p, q, r, d] {
        let err = from_pilout(&p, 0, 0, &gl(), Opening::Deep).unwrap_err();
        assert!(matches!(err, PiloutError::SymbolCount { .. }), "{err:?}");
    }
}
