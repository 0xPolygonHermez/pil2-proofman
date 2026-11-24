//! `layout`, `openings`, `project`: the maps, the evaluation map and the starkinfo projection.

mod common;

use std::collections::BTreeMap;

use common::*;
use pil2_pilout::pilout::{self as pb, operand, SymbolType};
use pil_info::core::from_pilout::from_pilout;
use pil_info::core::im::{check_im, ImChoice};
use pil_info::core::ir::{Boundary, ColRef, ColType, ConstraintSystem, Expr, Hint, HintField, HintValue};
use pil_info::core::layout::{layout, Layout, LayoutParams};
use pil_info::core::openings::{fri_batch_size, openings, EvEntry, Openings, OpeningsError};
use pil_info::core::project::{stark_info, StarkInfoCore, StarkParams, VerifierView};
use pil_info::Opening;

fn lp(q_deg: u32) -> LayoutParams {
    LayoutParams { im_pols_stages: false, deep_challenges: true, q_deg, air_name: "Synthetic".into() }
}

fn lay_of(cs: &ConstraintSystem, im: &[u32], q_deg: u32) -> Layout {
    layout(cs, &choice(im), &lp(q_deg)).unwrap()
}

fn fixed_at(idx: u32, row_offset: i32) -> Option<pb::Operand> {
    op(operand::Operand::FixedCol(operand::FixedCol { idx, row_offset }))
}

fn ev(col: ColType, commit_id: Option<u32>, id: u32, offset: i64, opening_pos: u32) -> EvEntry {
    EvEntry { pol: ColRef { col, commit_id, id, offset }, opening_pos }
}

fn ops_of(cs: &ConstraintSystem, im: &[u32], q_deg: u32) -> Openings {
    openings(cs, &choice(im), &lay_of(cs, im, q_deg)).unwrap()
}

/// `cs` with one hint whose fields are `exprs`.
fn hinted(cs: &ConstraintSystem, exprs: &[Expr]) -> ConstraintSystem {
    let fields = exprs.iter().map(|e| HintField { name: None, value: HintValue::Expr(e.clone()) }).collect();
    let mut cs = cs.clone();
    cs.hints.push(Hint { name: "h".into(), fields });
    cs
}

fn sections(l: &Layout) -> BTreeMap<String, u32> {
    l.map_sections_n.iter().cloned().collect()
}

// ---- layout ----------------------------------------------------------------------------------

#[test]
fn witness_columns_keep_pilout_order() {
    let cs =
        cs_of(&pilout(&["z", "a", "m"], &["F1", "F0"], vec![sub(witness(0, 0), constant("1"))], vec![every_row(0)]));
    let l = lay_of(&cs, &[], 0);
    let cols: Vec<_> =
        l.cm_pols_map.iter().map(|p| (p.name.as_str(), p.stage, p.dim, p.stage_id, p.stage_pos)).collect();
    assert_eq!(cols, [("z", 1, 1, 0, 0), ("a", 1, 1, 1, 1), ("m", 1, 1, 2, 2)]);
    assert!(l.cm_pols_map.iter().all(|p| !p.im_pol && p.commit_id.is_none()));
    let fixed: Vec<_> = l.const_pols_map.iter().map(|p| p.name.as_str()).collect();
    assert_eq!(fixed, ["F1", "F0"]);
}

#[test]
fn im_columns_follow_witness_in_im_order() {
    let cs = system(chain(), &[3]);
    let l = lay_of(&cs, &[1, 2], 0);
    assert_eq!(l.cm_pols_map.len(), 6);
    let names: Vec<_> = l.cm_pols_map.iter().map(|p| p.name.as_str()).collect();
    assert_eq!(names, ["a", "b", "c", "d", "Synthetic.ImPol", "Synthetic.ImPol"]);
    for (k, p) in l.cm_pols_map[4..].iter().enumerate() {
        assert!(p.im_pol, "im column {k}");
        assert_eq!((p.stage, p.dim, p.stage_id, p.stage_pos), (1, 1, 4 + k as u32, 4 + k as u32));
    }
    assert!(l.cm_pols_map[..4].iter().all(|p| !p.im_pol));
}

/// 0: a*b (stage 1), 1: e0*x (stage 2); root 1; e0 promoted.
fn two_stage_system() -> ConstraintSystem {
    let e = vec![mul(witness(0, 0), witness(1, 0)), mul(exp(0), witness_at(2, 0, 0))];
    cs_of(&pilout2(&[], e, vec![every_row(1)]))
}

#[test]
fn im_pols_stages_false() {
    let cs = two_stage_system();
    let l = layout(&cs, &choice(&[0]), &LayoutParams { im_pols_stages: false, ..lp(0) }).unwrap();
    let im = &l.cm_pols_map[4];
    assert!(im.im_pol);
    // the last stage, after x and y (3 + 3 base columns), of dimension 1 (a*b is a base value)
    assert_eq!((im.stage, im.dim, im.stage_id, im.stage_pos), (2, 1, 2, 6));
}

#[test]
fn im_pols_stages_true() {
    let cs = two_stage_system();
    let l = layout(&cs, &choice(&[0]), &LayoutParams { im_pols_stages: true, ..lp(0) }).unwrap();
    let im = &l.cm_pols_map[4];
    assert!(im.im_pol);
    // the stage of a*b, after a and b
    assert_eq!((im.stage, im.dim, im.stage_id, im.stage_pos), (1, 1, 2, 2));
    // a promoted expression that reads a stage-2 column lands in stage 2 either way
    let l = layout(&cs, &choice(&[1]), &LayoutParams { im_pols_stages: true, ..lp(0) }).unwrap();
    assert_eq!((l.cm_pols_map[4].stage, l.cm_pols_map[4].dim), (2, 3));
}

/// A fixed-only im is still a witness column: stage 1, never 0.
#[test]
fn im_pols_stages_true_fixed_only_im_is_stage_1() {
    let e = vec![mul(fixed(0), fixed(1)), mul(exp(0), witness(0, 0))];
    let cs = cs_of(&pilout(&["a"], &["F0", "F1"], e, vec![every_row(1)]));
    let l = layout(&cs, &choice(&[0]), &LayoutParams { im_pols_stages: true, ..lp(0) }).unwrap();
    assert_eq!(l.cm_pols_map[1].stage, 1);
    assert!(l.map_sections_n.iter().all(|(k, _)| k != "cm0"));
}

#[test]
fn stage_pos_is_cumulative_dim() {
    let cs = two_stage_system();
    let l = lay_of(&cs, &[], 2);
    let cols: Vec<_> =
        l.cm_pols_map.iter().map(|p| (p.name.as_str(), p.stage, p.dim, p.stage_id, p.stage_pos)).collect();
    assert_eq!(
        cols,
        [
            ("a", 1, 1, 0, 0),
            ("b", 1, 1, 1, 1),
            ("x", 2, 3, 0, 0),
            ("y", 2, 3, 1, 3),
            // the quotient pieces, stage n_stages + 1
            ("Q0", 3, 3, 0, 0),
            ("Q1", 3, 3, 1, 3),
        ]
    );
}

#[test]
fn challenges_end_with_vc_xi_vf1_vf2() {
    let names = |l: &Layout| -> Vec<(String, u32, u32, u32)> {
        l.challenges_map.iter().map(|c| (c.name.clone(), c.stage, c.stage_id, c.dim)).collect()
    };
    let s = |n: &str, st, id, dim| (n.to_string(), st, id, dim);
    // nStages 2
    let l = lay_of(&two_stage_system(), &[], 1);
    assert_eq!(
        names(&l),
        [
            s("alpha", 2, 0, 3),
            s("beta", 2, 1, 3),
            s("std_vc", 3, 0, 3),
            s("std_xi", 4, 0, 3),
            s("std_vf1", 5, 0, 3),
            s("std_vf2", 5, 1, 3)
        ]
    );
    // nStages 1
    let l = lay_of(&system(chain(), &[1]), &[], 1);
    assert_eq!(names(&l), [s("std_vc", 2, 0, 3), s("std_xi", 3, 0, 3), s("std_vf1", 4, 0, 3), s("std_vf2", 4, 1, 3)]);
    // without the DEEP challenges the list ends at std_xi
    let l = layout(&system(chain(), &[1]), &choice(&[]), &LayoutParams { deep_challenges: false, ..lp(1) }).unwrap();
    assert_eq!(names(&l), [s("std_vc", 2, 0, 3), s("std_xi", 3, 0, 3)]);
}

#[test]
fn map_sections_n_counts_by_stage() {
    let e = vec![mul(witness(0, 0), witness(1, 0)), mul(exp(0), witness_at(2, 0, 0))];
    let cs = cs_of(&pilout2(&["L1", "L2"], e, vec![every_row(1)]));
    let l = lay_of(&cs, &[0], 2);
    let want: BTreeMap<String, u32> =
        [("const", 2), ("cm1", 2), ("cm2", 3 + 3 + 1), ("cm3", 2 * 3)].map(|(k, v)| (k.to_string(), v)).into();
    assert_eq!(sections(&l), want);
    assert_eq!(l.map_sections_n.len(), 4, "no section twice");
}

// ---- openings --------------------------------------------------------------------------------

#[test]
fn points_are_sorted_and_include_zero() {
    // offsets -10, 3, 10, -2: numeric order, not the string order -10, -2, 0, 10, 3
    let e = vec![add(witness(0, -10), witness(1, 3)), mul(exp(0), witness(2, 10)), sub(witness(3, -2), constant("1"))];
    let cs = system(e, &[1, 2]);
    assert_eq!(ops_of(&cs, &[], 0).opening_points, [-10, -2, 0, 3, 10]);
    // no offset anywhere still opens at 0
    let cs = system(chain(), &[0]);
    let o = ops_of(&cs, &[], 0);
    assert_eq!(o.opening_points, [0]);
    assert_eq!(o.ev_map, [ev(ColType::Witness, None, 0, 0, 0), ev(ColType::Witness, None, 1, 0, 0)]);
}

#[test]
fn ev_map_sort_is_type_then_id_then_offset() {
    // within an opening point: custom, then const, then cm (today's order), each by id then offset
    let e = vec![
        add(witness(0, 0), witness(0, 1)),
        add(exp(0), witness(1, 0)),
        add(exp(1), fixed_at(1, 0)),
        add(exp(2), fixed_at(0, 1)),
        add(
            exp(3),
            op(operand::Operand::CustomCol(operand::CustomCol { commit_id: 0, stage: 0, col_idx: 0, row_offset: 0 })),
        ),
    ];
    let mut p = pilout(&["a", "b"], &["F0", "F1"], e, vec![every_row(4)]);
    p.air_groups[0].airs[0].custom_commits =
        vec![pb::CustomCommit { name: Some("rom".into()), stage_widths: vec![1], ..Default::default() }];
    p.symbols.push(pb::Symbol { commit_id: Some(0), ..symbol("rom", SymbolType::CustomCol, 0, 0) });
    let cs = cs_of(&p);
    let o = ops_of(&cs, &[], 0);
    assert_eq!(o.opening_points, [0, 1]);
    assert_eq!(
        o.ev_map,
        [
            ev(ColType::Custom, Some(0), 0, 0, 0),
            ev(ColType::Fixed, None, 1, 0, 0),
            ev(ColType::Witness, None, 0, 0, 0),
            ev(ColType::Witness, None, 1, 0, 0),
            ev(ColType::Fixed, None, 0, 1, 1),
            ev(ColType::Witness, None, 0, 1, 1),
        ]
    );
}

#[test]
fn im_columns_appear_at_offset_zero() {
    // e0 = a'*b promoted; the root reads the im column, the im root reads a' and b
    let e = vec![mul(witness(0, 1), witness(1, 0)), mul(exp(0), witness(2, 0))];
    let cs = system(e, &[1]);
    let o = ops_of(&cs, &[0], 0);
    assert_eq!(o.opening_points, [0, 1]);
    assert_eq!(
        o.ev_map,
        [
            ev(ColType::Witness, None, 1, 0, 0),
            ev(ColType::Witness, None, 2, 0, 0),
            ev(ColType::Witness, None, 4, 0, 0), // the im column
            ev(ColType::Witness, None, 0, 1, 1),
        ]
    );
}

#[test]
fn quotient_pieces_are_opened_at_zero_under_deep() {
    let cs = system(chain(), &[1]);
    // a, b, c at 0, then Q0 and Q1 (ids 4 and 5) at 0
    let o = ops_of(&cs, &[], 2);
    let ids: Vec<u32> = o.ev_map.iter().map(|e| e.pol.id).collect();
    assert_eq!(ids, [0, 1, 2, 4, 5]);
}

#[test]
fn hint_expressions_add_points() {
    let mut e = chain();
    e.push(mul(witness(1, -7), witness(2, 0))); // 4: only a hint reads it
    let cs = system(e, &[1]);
    let l = lay_of(&cs, &[], 0);
    let plain = openings(&cs, &choice(&[]), &l).unwrap();
    // only the expressions a hint names add points, as the prover keeps only those
    let col = Expr::Col { col: ColType::Witness, commit_id: None, id: 0, offset: 5 };
    let hint = Expr::Mul(
        Box::new(Expr::Col { col: ColType::Witness, commit_id: None, id: 1, offset: -7 }),
        Box::new(Expr::Col { col: ColType::Witness, commit_id: None, id: 2, offset: 0 }),
    );
    let o = openings(&hinted(&cs, &[col, Expr::Ref(find(&cs, &hint))]), &choice(&[]), &l).unwrap();
    assert_eq!(o.opening_points, [-7, 0]);
    // a point from a hint has no evaluation: the evMap only changes its opening positions
    assert!(o.ev_map.iter().all(|e| e.pol.offset == 0));
    let offsets = |x: &Openings| x.ev_map.iter().map(|e| e.pol).collect::<Vec<_>>();
    assert_eq!(offsets(&o), offsets(&plain));
    assert!(o.ev_map.iter().all(|e| e.opening_pos == 1), "0 is now the second point");
}

#[test]
fn witness_read_past_the_witness_columns_is_error() {
    // a root reading column 4 (the id of Q0) would alias the quotient piece
    let mut cs = system(chain(), &[2]);
    let q = Expr::Col { col: ColType::Witness, commit_id: None, id: 4, offset: 0 };
    cs.exprs[2] = Expr::Mul(Box::new(Expr::Ref(1)), Box::new(q));
    let l = lay_of(&system(chain(), &[2]), &[], 1);
    let err = openings(&cs, &choice(&[]), &l).unwrap_err();
    assert!(matches!(err, OpeningsError::Unresolved { .. }), "{err:?}");
}

#[test]
fn unknown_reference_is_error() {
    let cs = system(chain(), &[1]);
    let l = lay_of(&cs, &[], 0);
    let err = openings(&hinted(&cs, &[Expr::Ref(99)]), &choice(&[]), &l).unwrap_err();
    assert!(matches!(err, OpeningsError::Unresolved { .. }), "{err:?}");
}

#[test]
fn fri_batch_size_formula() {
    // 3 evaluations, 3 openings with evaluations, 3 polynomials: 3 + 3 - 1 beats 3
    let o = Openings {
        opening_points: vec![-1, 0, 1, 5], // 5 has no evaluation
        ev_map: vec![
            ev(ColType::Witness, None, 0, -1, 0),
            ev(ColType::Witness, None, 1, 0, 1),
            ev(ColType::Fixed, None, 0, 1, 2),
        ],
    };
    assert_eq!(fri_batch_size(&o), 5);
    // 6 evaluations beat 3 + 2 - 1
    let at = |id, pos| ev(ColType::Witness, None, id, pos as i64, pos);
    let o = Openings {
        opening_points: vec![0, 1, 2],
        ev_map: vec![at(0, 0), at(0, 1), at(0, 2), at(1, 0), at(1, 1), at(1, 2)],
    };
    assert_eq!(fri_batch_size(&o), 6);
    // the same id in different custom commits is two polynomials
    let o = Openings {
        opening_points: vec![0],
        ev_map: vec![ev(ColType::Custom, Some(0), 0, 0, 0), ev(ColType::Custom, Some(1), 0, 0, 0)],
    };
    assert_eq!(fri_batch_size(&o), 2);
}

/// 64 levels of `x_{k+1} = x_k + x_k`: inlining would be 2^64 nodes, the memoised walk is linear.
#[test]
fn doubling_chain_is_not_inlined() {
    let mut e = vec![mul(witness(0, 1), witness(1, 0))];
    for k in 0..64 {
        e.push(add(exp(k), exp(k)));
    }
    e.push(mul(exp(64), exp(0)));
    let cs = system(e, &[65]);
    for im in [vec![], vec![32], vec![64]] {
        let o = ops_of(&cs, &im, 1);
        assert_eq!(o.opening_points, [0, 1]);
    }
    let hint = Expr::Ref(64);
    let l = lay_of(&cs, &[], 0);
    assert_eq!(openings(&hinted(&cs, &[hint]), &choice(&[]), &l).unwrap().opening_points, [0, 1]);
}

// ---- projection ------------------------------------------------------------------------------

fn params() -> StarkParams {
    StarkParams { n_bits: 4, n_bits_ext: 5, c_exp_id: 7, circuit_type: None, multiproof: false }
}

/// Everything `stark_info` reads, for the two-stage system with e0 promoted.
fn projected(p: &StarkParams) -> (ConstraintSystem, ImChoice, Layout, Openings, StarkInfoCore, u32) {
    let cs = two_stage_system();
    let im = choice(&[0]);
    let chk = check_im(&cs, &im, 3, true).unwrap();
    let l = layout(&cs, &im, &lp(chk.q_deg)).unwrap();
    let o = openings(&cs, &im, &l).unwrap();
    let core = stark_info(&cs, &im, &chk, &l, &o, p).unwrap();
    (cs, im, l, o, core, chk.q_deg)
}

#[test]
fn projection_has_no_decisions() {
    let (cs, im, l, o, core, q_deg) = projected(&params());
    assert_eq!(core.verifier.cm_pols_map, l.cm_pols_map);
    assert_eq!(core.verifier.const_pols_map, l.const_pols_map);
    assert_eq!(core.verifier.custom_pols_map, l.custom_pols_map);
    assert_eq!(core.verifier.map_sections_n, l.map_sections_n);
    assert_eq!(core.verifier.challenges_map, l.challenges_map);
    assert_eq!(core.verifier.publics_map, l.publics_map);
    assert_eq!(core.verifier.proof_values_map, l.proof_values_map);
    assert_eq!(core.verifier.air_values_map, l.air_values_map);
    assert_eq!(core.verifier.airgroup_values_map, l.airgroup_values_map);
    assert_eq!(
        (core.verifier.ev_map.clone(), core.verifier.opening_points.clone()),
        (o.ev_map.clone(), o.opening_points.clone())
    );
    assert_eq!((core.verifier.n_stages, core.verifier.q_deg, core.verifier.q_dim), (2, q_deg, 3));
    assert_eq!(core.verifier.n_constraints as usize, cs.constraints.len() + im.exprs.len());
    assert_eq!(
        (core.verifier.n_constants as usize, core.verifier.n_publics as usize),
        (cs.fixed.len(), cs.publics.len())
    );
    assert_eq!(core.prover.im_exp_ids, im.exprs);
    assert_eq!(core.verifier.fri_batch_size, fri_batch_size(&o));
    // the parameters are copied and nothing else moves with them
    let (_, _, _, _, core2, _) = projected(&StarkParams { c_exp_id: 3, n_bits: 11, n_bits_ext: 13, ..params() });
    let want = StarkInfoCore {
        verifier: VerifierView { n_bits: 11, n_bits_ext: 13, c_exp_id: 3, ..core.verifier.clone() },
        prover: core.prover.clone(),
    };
    assert_eq!(core2, want);
    assert_eq!((core.verifier.n_bits, core.verifier.n_bits_ext, core.verifier.c_exp_id), (4, 5, 7));
}

#[test]
fn boundaries_are_every_row_only_for_deep() {
    let project = |cs: &ConstraintSystem, deep: bool| {
        let im = choice(&[]);
        let chk = check_im(cs, &im, 9, false).unwrap();
        let q_deg = if deep { chk.q_deg } else { 0 };
        let l = layout(cs, &im, &LayoutParams { deep_challenges: deep, ..lp(q_deg) }).unwrap();
        let o = openings(cs, &im, &l).unwrap();
        stark_info(cs, &im, &chk, &l, &o, &params()).unwrap().verifier.boundaries
    };
    // Deep: always everyRow
    assert_eq!(project(&system(chain(), &[1]), true), [Boundary::EveryRow]);
    // Shplonk keeps pilfflonk's, distinct in first-appearance order
    let e = vec![sub(witness(0, 0), constant("1"))];
    let cons = vec![last_row(0), every_row(0), first_row(0), last_row(0), every_frame(0, 0, 1)];
    let cs = from_pilout(&pilout(&["a", "b", "c", "d"], &[], e, cons), 0, 0, &gl(), Opening::Shplonk).unwrap();
    assert_eq!(
        project(&cs, false),
        [Boundary::LastRow, Boundary::EveryRow, Boundary::FirstRow, Boundary::EveryFrame { min: 0, max: 1 }]
    );
}
