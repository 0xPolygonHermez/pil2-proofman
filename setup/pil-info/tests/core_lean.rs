//! Task 8b: the core has no silent defaults, typed hints, canonical input, starkinfo JSON, stage rule.

mod common;

use common::*;
use pil2_pilout::pilout::{self as pb, hint_field, SymbolType};
use pil_info::core::from_pilout::{from_pilout, from_pilout_core, PiloutError};
use pil_info::core::ir::{ColType, ConstraintSystem, Expr, HintField, HintValue};
use pil_info::core::layout::{layout, LayoutParams};
use pil_info::core::openings::openings;
use pil_info::core::pilout_core::PiloutCore;
use pil_info::core::project::{stark_info, starkinfo_output, StarkParams};
use pil_info::Opening;

fn base() -> pb::PilOut {
    pilout(&["a"], &[], vec![sub(witness(0, 0), constant("1"))], vec![every_row(0)])
}
fn lower(p: &pb::PilOut) -> Result<ConstraintSystem, PiloutError> {
    from_pilout(p, 0, 0, &gl(), Opening::Deep)
}

// ---- (a) errors, not defaults ----

#[test]
fn missing_num_rows_is_an_error() {
    let mut p = base();
    p.air_groups[0].airs[0].num_rows = None;
    assert!(matches!(lower(&p), Err(PiloutError::MissingNumRows { .. })), "{:?}", lower(&p).map(|c| c.n_rows_bits));
}

#[test]
fn non_power_of_two_rows_is_an_error() {
    for rows in [0u32, 3, 12, 17] {
        let mut p = base();
        p.air_groups[0].airs[0].num_rows = Some(rows);
        assert!(matches!(lower(&p), Err(PiloutError::NumRowsNotPowerOfTwo { .. })), "rows {rows}");
    }
}

#[test]
fn missing_stage_on_a_witness_symbol_is_an_error() {
    let mut p = base();
    p.symbols.iter_mut().find(|s| s.r#type == SymbolType::WitnessCol as i32).unwrap().stage = None;
    assert!(matches!(lower(&p), Err(PiloutError::MissingStage { .. })));
}

#[test]
fn missing_stage_on_a_challenge_symbol_is_an_error() {
    let mut p = pilout2(&[], vec![sub(witness(0, 0), constant("1"))], vec![every_row(0)]);
    p.symbols.iter_mut().find(|s| s.r#type == SymbolType::Challenge as i32).unwrap().stage = None;
    assert!(matches!(lower(&p), Err(PiloutError::MissingStage { .. })));
}

/// The proto leaves a proof value's stage optional; absent is stage 1, the only stage a proof value has.
#[test]
fn a_proof_value_symbol_without_stage_is_stage_1() {
    let mut p = base();
    p.num_proof_values = vec![1];
    let mut s = symbol("pv", SymbolType::ProofValue, 1, 0);
    s.stage = None;
    p.symbols.push(s);
    assert_eq!(lower(&p).unwrap().proof_values[0].stage, 1);
}

#[test]
fn custom_column_without_commit_id_is_an_error() {
    let mut p = base();
    p.air_groups[0].airs[0].custom_commits =
        vec![pb::CustomCommit { name: Some("c".into()), public_values: vec![], stage_widths: vec![1] }];
    let mut s = symbol("cc", SymbolType::CustomCol, 0, 0);
    s.commit_id = None;
    p.symbols.push(s);
    assert!(matches!(lower(&p), Err(PiloutError::MissingCommitId { .. })));
}

fn params(_cs: &ConstraintSystem) -> LayoutParams {
    LayoutParams { im_pols_stages: true, deep_challenges: true, q_deg: 1, air_name: "T".into() }
}

/// A system whose im expression 0 is `e` (no pilout can produce this: the lookups must still fail).
fn with_im_expr(e: Expr) -> (ConstraintSystem, pil_info::core::im::ImChoice) {
    let mut cs = system(chain(), &[3]);
    cs.exprs[0] = e;
    (cs, choice(&[0]))
}

#[test]
fn layout_errors_on_an_air_value_the_air_does_not_have() {
    let (cs, im) = with_im_expr(Expr::AirValue(5));
    assert!(layout(&cs, &im, &params(&cs)).is_err());
}

#[test]
fn layout_errors_on_a_witness_column_the_air_does_not_have() {
    let (cs, im) = with_im_expr(Expr::Col { col: ColType::Witness, commit_id: None, id: 99, offset: 0 });
    assert!(layout(&cs, &im, &params(&cs)).is_err());
}

#[test]
fn layout_errors_on_a_dangling_ref() {
    let (cs, im) = with_im_expr(Expr::Ref(999));
    assert!(layout(&cs, &im, &params(&cs)).is_err());
}

/// Guard (passes today through `reach`): a custom column with no commit is an error, never commit 0.
#[test]
fn openings_error_on_a_custom_column_without_commit_id() {
    let r = run(&pilout(&["a", "b", "c", "d"], &[], chain(), vec![every_row(3)]), 0, 0, 3).expect("feasible");
    let mut cs = r.cs.clone();
    cs.exprs[0] = Expr::Col { col: ColType::Custom, commit_id: None, id: 0, offset: 0 };
    let lay = layout(&r.cs, &r.im, &params(&r.cs)).unwrap();
    assert!(openings(&cs, &r.im, &lay).is_err());
}

#[test]
fn a_proof_value_without_a_symbol_below_the_last_is_an_error() {
    let mut p = base();
    p.num_proof_values = vec![2];
    let mut s = symbol("pv", SymbolType::ProofValue, 1, 1);
    s.air_group_id = None;
    p.symbols.push(s);
    assert!(matches!(lower(&p), Err(PiloutError::BadOperand { .. })), "{:?}", lower(&p).map(|c| c.proof_values));
}

/// A symbol of unknown type anywhere fails both entries alike.
#[test]
fn a_bad_symbol_of_another_air_fails_both_entries() {
    let mut p = base();
    p.symbols.push(pb::Symbol { r#type: 99, air_id: Some(1), ..symbol("z", SymbolType::WitnessCol, 1, 0) });
    assert!(lower(&p).is_err());
    assert!(PiloutCore::from_pb(&p).is_err());
}

// ---- (c) typed hints ----

fn hint_pilout() -> pb::PilOut {
    let mut p = pilout(
        &["a", "b"],
        &[],
        vec![mul(witness(0, 0), witness(1, -7)), sub(exp(0), constant("1"))],
        vec![every_row(1)],
    );
    let field = |name: &str, v: hint_field::Value| pb::HintField { name: Some(name.into()), value: Some(v) };
    p.hints = vec![pb::Hint {
        name: "h".into(),
        air_group_id: Some(0),
        air_id: Some(0),
        hint_fields: vec![
            field("s", hint_field::Value::StringValue("text".into())),
            field("e", hint_field::Value::Operand(exp(1).unwrap())),
            field(
                "arr",
                hint_field::Value::HintFieldArray(pb::HintFieldArray {
                    hint_fields: vec![field("c", hint_field::Value::Operand(witness(0, 0).unwrap()))],
                }),
            ),
        ],
    }];
    p
}

#[test]
fn a_hint_lowers_to_typed_ir() {
    let cs = lower(&hint_pilout()).unwrap();
    let f = &cs.hints[0].fields;
    let col0 = Expr::Col { col: ColType::Witness, commit_id: None, id: 0, offset: 0 };
    assert_eq!(f[0], HintField { name: Some("s".into()), value: HintValue::Str("text".into()) });
    assert_eq!(f[1], HintField { name: Some("e".into()), value: HintValue::Expr(Expr::Ref(1)) });
    assert_eq!(f[2].value, HintValue::Array(vec![HintField { name: Some("c".into()), value: HintValue::Expr(col0) }]));
}

/// After a leading array, the other fields are not read, so not lowered.
#[test]
fn fields_after_a_leading_array_are_not_lowered() {
    let mut p = hint_pilout();
    p.hints[0].hint_fields.rotate_left(2);
    p.hints[0].hint_fields[1].value = Some(hint_field::Value::Operand(public(7).unwrap()));
    let cs = lower(&p).unwrap();
    assert_eq!(cs.hints[0].fields.len(), 1);
}

/// Behaviour guard across the Hint rewrite: a hint expression alone contributes the -7 opening point.
#[test]
fn openings_read_the_typed_hint() {
    let r = run_any(&hint_pilout(), 0, 0).expect("feasible");
    let out = core_out(&r);
    assert!(out.ops.opening_points.contains(&-7), "{:?}", out.ops.opening_points);
}

// ---- (d) canonical input ----

#[test]
fn adapter_round_trips_on_every_real_pilout() {
    for (path, p) in real_pilouts() {
        let c = PiloutCore::from_pb(&p).unwrap_or_else(|e| panic!("{path:?}: {e}"));
        assert_eq!(PiloutCore::from_json(&c.to_json().unwrap()).unwrap(), c, "{path:?}");
    }
}

#[test]
fn core_input_lowers_like_the_protobuf_on_every_real_air() {
    for (path, p) in real_pilouts() {
        let c = PiloutCore::from_pb(&p).unwrap();
        for (g, ag) in p.air_groups.iter().enumerate() {
            for a in 0..ag.airs.len() {
                let old = from_pilout(&p, g, a, &gl(), Opening::Deep);
                let new = from_pilout_core(&c, g, a, &gl(), Opening::Deep);
                assert_eq!(new, old, "{path:?} air {g}/{a}");
            }
        }
    }
}

/// The hand-written form of `a - 1 = 0` over 16 rows, as the spec writes it.
const HAND_WRITTEN: &str = r#"{
  "numChallenges": [0], "numProofValues": [], "numPublicValues": 0,
  "airGroups": [{"airGroupValues": [], "airs": [{
    "numRows": 16, "stageWidths": [1], "nFixedCols": 0, "nPeriodicCols": 0, "airValues": [], "customCommits": [],
    "expressions": [{"sub": [{"witnessCol": {"stage": 1, "colIdx": 0, "rowOffset": 0}}, {"constant": "1"}]}],
    "constraints": [{"kind": "everyRow", "expressionIdx": 0}]
  }]}],
  "symbols": [{"name": "a", "type": "witnessCol", "airGroupId": 0, "airId": 0, "id": 0, "stage": 1, "dim": 0, "lengths": []}],
  "hints": []
}"#;

#[test]
fn a_hand_written_core_input_parses_and_lowers() {
    let c = PiloutCore::from_json(HAND_WRITTEN).unwrap();
    let cs = from_pilout_core(&c, 0, 0, &gl(), Opening::Deep).unwrap();
    assert_eq!((cs.n_rows_bits, cs.n_stages, cs.witness.len(), cs.constraints.len()), (4, 1, 1, 1));
}

// ---- (e) starkinfo JSON from the core; casts ----

#[test]
fn starkinfo_output_has_the_core_owned_keys() {
    let r = run_any(&pilout(&["a", "b", "c", "d"], &[], chain(), vec![every_row(3)]), 0, 0).expect("feasible");
    let out = core_out(&r);
    let sp = StarkParams {
        n_bits: 4,
        n_bits_ext: 5,
        c_exp_id: out.exprs.roles.c_exp as u32,
        circuit_type: None,
        multiproof: false,
    };
    let core = stark_info(&r.cs, &r.im, &r.chk, &out.lay, &out.ops, &sp).unwrap();
    let j = serde_json::to_value(starkinfo_output(&core, &r.air_name, 0, 0, (), None::<()>).unwrap()).unwrap();
    for k in ["cmPolsMap", "constPolsMap", "evMap", "openingPoints", "qDeg", "qDim", "nStages", "boundaries"] {
        assert!(j.get(k).is_some(), "missing {k}");
    }
}

/// A stage-width sum past u32 must be an error, not a wrap or a panic.
#[test]
fn overflowing_stage_widths_are_an_error() {
    let mut p = pilout2(&[], vec![sub(witness_at(2, 0, 0), constant("1"))], vec![every_row(0)]);
    p.air_groups[0].airs[0].stage_widths = vec![u32::MAX, u32::MAX];
    assert!(lower(&p).is_err());
}

// ---- (f) stage rule: n_stages = max(highest declared stage, numChallenges.len(), 1) ----

#[test]
fn no_challenge_stages_gives_one_stage() {
    let mut p = base();
    p.num_challenges.clear();
    assert_eq!(lower(&p).unwrap().n_stages, 1);
}

#[test]
fn highest_declared_witness_stage_wins_over_empty_challenges() {
    let mut p = pilout2(&[], vec![sub(witness(0, 0), constant("1"))], vec![every_row(0)]);
    p.num_challenges.clear();
    p.symbols.retain(|s| s.r#type != SymbolType::Challenge as i32);
    assert_eq!(lower(&p).unwrap().n_stages, 2);
}

#[test]
fn challenge_stages_beyond_the_witness_stages_count() {
    let mut p = base();
    p.num_challenges = vec![0, 0, 0];
    assert_eq!(lower(&p).unwrap().n_stages, 3);
}
