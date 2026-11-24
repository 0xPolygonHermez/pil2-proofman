//! `im`: check_im, substitute, order, and the untrusted search.

mod common;

use common::*;
use num_bigint::BigUint;
use pil_info::core::from_pilout::{from_pilout, PiloutError};
use pil_info::core::im::{check_im, im_order, substitute, Checked, ImError};
use pil_info::core::layout::quotient_dim;
use pil_info::core::ir::{degree, eval_expr, visit, Boundary, ColRef, ColType, Env, Expr};
use pil_info::search::im::search_im;
use pil_info::{DegreePolicy, Opening};

#[test]
fn accepts_empty_choice_when_degree_fits() {
    let cs = system(chain(), &[1]);
    let got = check_im(&cs, &choice(&[]), 3, true).unwrap();
    assert_eq!(got, Checked { q_deg: 2, root_degrees: vec![3], im_degrees: vec![] });
}

#[test]
fn rejects_root_too_high() {
    let cs = system(chain(), &[2]);
    assert_eq!(
        check_im(&cs, &choice(&[]), 3, true).unwrap_err(),
        ImError::RootTooHigh { constraint: 0, degree: 4, max: 3 }
    );
}

#[test]
fn accepts_promoting_ab() {
    let cs = system(chain(), &[2]);
    let got = check_im(&cs, &choice(&[0]), 3, true).unwrap();
    assert_eq!(got, Checked { q_deg: 2, root_degrees: vec![3], im_degrees: vec![2] });
}

#[test]
fn rejects_im_too_high() {
    // e2 is degree 4 even after substituting nothing else; the root e3 = e2*a fits once e2 is a column
    let cs = system(chain(), &[3]);
    assert_eq!(check_im(&cs, &choice(&[2]), 3, true).unwrap_err(), ImError::ImTooHigh { expr: 2, degree: 4, max: 3 });
}

#[test]
fn im_degree_is_taken_after_substituting_the_others() {
    // e2 = e1*d, e1 = e0*c: with e1 promoted, e2 is degree 2, not 4 (e2 appears first)
    let cs = system(chain(), &[3]);
    let got = check_im(&cs, &choice(&[2, 1]), 3, true).unwrap();
    assert_eq!(got.im_degrees, vec![2, 3]);
    assert_eq!(got.root_degrees, vec![2]);
}

#[test]
fn rejects_non_promotable() {
    let mut e = chain();
    e.push(sub(constant("7"), constant("2"))); // 4: a degree-0 expression
    let cs = system(e, &[1]);
    assert_eq!(check_im(&cs, &choice(&[99]), 3, true).unwrap_err(), ImError::NotPromotable { expr: 99 });
    let k = find(&cs, &Expr::Sub(Box::new(Expr::Const(7u32.into())), Box::new(Expr::Const(2u32.into()))));
    assert_eq!(check_im(&cs, &choice(&[k]), 3, true).unwrap_err(), ImError::NotPromotable { expr: k });
}

#[test]
fn rejects_unsorted_and_duplicate() {
    let cs = system(chain(), &[1]);
    // the root e1 appears before e0
    assert_eq!(check_im(&cs, &choice(&[0, 1]), 4, true).unwrap_err(), ImError::Unsorted);
    assert_eq!(check_im(&cs, &choice(&[0, 0]), 4, true).unwrap_err(), ImError::Duplicate { expr: 0 });
}

#[test]
fn q_deg_is_derived() {
    // (1) degree 2 root: q_deg 1
    let cs = system(chain(), &[0]);
    assert_eq!(check_im(&cs, &choice(&[]), 3, true).unwrap().q_deg, 1);
    // (2) max over all quotient roots: e1 (3), e3 (5 -> 2 with e2 promoted) and the im root of e2 (4)
    let cs = system(chain(), &[1, 3]);
    let got = check_im(&cs, &choice(&[2]), 4, true).unwrap();
    assert_eq!((got.root_degrees, got.im_degrees, got.q_deg), (vec![3, 2], vec![4], 3));
    // (3) a degree-1 root: 0 unless the STARK floor applies
    let cs = system(vec![sub(witness(0, 0), constant("1"))], &[0]);
    assert_eq!(check_im(&cs, &choice(&[]), 3, false).unwrap().q_deg, 0);
    assert_eq!(check_im(&cs, &choice(&[]), 3, true).unwrap().q_deg, 1);
}

#[test]
fn substitute_resolves_refs_and_promoted_become_columns() {
    let cs = system(chain(), &[3]);
    let im = choice(&[1, 2]);
    let root = Expr::Ref(3);
    let sub_root = substitute(&cs, &im, &root).unwrap();
    let mut refs = 0;
    visit(&sub_root, &mut |n| refs += matches!(n, Expr::Ref(_)) as u32);
    assert_eq!(refs, 0, "no Ref survives");
    assert_eq!(degree(&cs, &sub_root), Ok(2));
    // each promoted Ref is its own distinct, offset-0 column past the witness columns
    let cols: Vec<Expr> = im.exprs.iter().map(|i| substitute(&cs, &im, &Expr::Ref(*i)).unwrap()).collect();
    let mut ids = Vec::new();
    for c in &cols {
        match c {
            Expr::Col { col: ColType::Witness, commit_id: None, id, offset: 0 } => ids.push(*id),
            other => panic!("a promoted Ref must be a witness column, got {other:?}"),
        }
    }
    assert!(ids.windows(2).all(|w| w[0] < w[1]) && ids[0] as usize >= cs.witness.len(), "{ids:?}");
    // semantics: with each im column set to the value of its expression, the root evaluates the same
    let mut env = Env::default();
    for (k, v) in [(0u32, 3u32), (1, 5), (2, 7), (3, 11)] {
        env.cols.insert(ColRef { col: ColType::Witness, commit_id: None, id: k, offset: 0 }, BigUint::from(v));
    }
    let mut full = env.clone();
    for (c, i) in cols.iter().zip(&im.exprs) {
        let Expr::Col { id, .. } = c else { unreachable!() };
        let k = ColRef { col: ColType::Witness, commit_id: None, id: *id, offset: 0 };
        full.cols.insert(k, eval_expr(&cs, &Expr::Ref(*i), &env).unwrap());
    }
    assert_eq!(eval_expr(&cs, &sub_root, &full), eval_expr(&cs, &root, &env));
    // an empty choice only inlines
    let inlined = substitute(&cs, &choice(&[]), &root).unwrap();
    assert_eq!(eval_expr(&cs, &inlined, &env), eval_expr(&cs, &root, &env));
}

// ---- the search, and pilfflonk's old one -------------------------------------------------------------

fn policies() -> Vec<DegreePolicy> {
    vec![
        DegreePolicy::FromBlowup { blowup_bits: 1 },
        DegreePolicy::FromBlowup { blowup_bits: 2 },
        DegreePolicy::FromBlowup { blowup_bits: 3 },
        DegreePolicy::Search { max: 3 },
        DegreePolicy::Search { max: 5 },
    ]
}

fn stark_floor(p: &DegreePolicy) -> bool {
    matches!(p, DegreePolicy::FromBlowup { .. })
}

/// pilfflonk protocol.md#degree-search: qDeg = max(deg c_i + δ_i) - 1, δ_i = 1 off everyRow; seed 4 is a
/// degree-2 everyFrame root (qDeg 2, which the old search under-sized to 1).
#[test]
fn shplonk_boundaries_follow_the_spec() {
    let mut checked = 0;
    for seed in 0..200 {
        let (_, cs) = random_pilout_in(seed, Opening::Shplonk);
        for policy in policies() {
            let q_dim = quotient_dim(&cs, &choice(&[])).unwrap() as usize;
            let Ok(set) = search_im(&cs, &policy, q_dim) else { continue };
            let im = im_order(&cs, &set).unwrap();
            let got = check_im(&cs, &im, policy.max_constraint_degree() as u32, stark_floor(&policy)).unwrap();
            let top = got.root_degrees.iter().chain(&got.im_degrees).copied().max().unwrap_or(0);
            let want = if stark_floor(&policy) { top.saturating_sub(1).max(1) } else { top.saturating_sub(1) };
            assert_eq!(got.q_deg, want, "seed {seed} {policy:?}");
            checked += 1;
        }
    }
    assert!(checked > 500, "{checked} checked");
    let (_, cs) = random_pilout_in(4, Opening::Shplonk);
    assert_eq!(
        cs.constraints.iter().map(|c| c.boundary.clone()).collect::<Vec<_>>(),
        vec![Boundary::EveryFrame { min: 0, max: 1 }]
    );
    assert_eq!(
        check_im(&cs, &choice(&[]), 3, false).unwrap(),
        Checked { q_deg: 2, root_degrees: vec![3], im_degrees: vec![] }
    );
}

#[test]
fn search_output_is_checked() {
    let mut checked = 0;
    for seed in 0..200 {
        let (_, cs) = random_pilout(seed);
        for policy in policies() {
            let Ok(set) = search_im(&cs, &policy, quotient_dim(&cs, &choice(&[])).unwrap() as usize) else { continue };
            let max = policy.max_constraint_degree() as u32;
            let im = im_order(&cs, &set).unwrap();
            check_im(&cs, &im, max, stark_floor(&policy)).unwrap_or_else(|e| panic!("seed {seed} {policy:?}: {e:?}"));
            checked += 1;
        }
    }
    assert!(checked > 500);
}

/// Only this seed's im root (not a constraint root) sets q_deg, which a roots-only derivation misses.
const HAZARD_SEED: u64 = 2781;

#[test]
fn memo_hazard_regression() {
    let (_, cs) = random_pilout(HAZARD_SEED);
    for policy in policies() {
        let set = search_im(&cs, &policy, quotient_dim(&cs, &choice(&[])).unwrap() as usize).expect("feasible");
        let im = im_order(&cs, &set).unwrap();
        let got = check_im(&cs, &im, policy.max_constraint_degree() as u32, stark_floor(&policy))
            .unwrap_or_else(|e| panic!("{policy:?}: set rejected: {e:?}"));
        let roots =
            cs.constraints.iter().map(|c| degree(&cs, &substitute(&cs, &im, &c.e).unwrap()).unwrap()).max().unwrap();
        assert_eq!((roots, got.q_deg), (2, 2), "{policy:?}: the im root drives q_deg");
    }
}

#[test]
fn an_air_without_constraints_is_rejected() {
    let p = pilout(&["a", "b", "c", "d"], &[], chain(), vec![]);
    let err = from_pilout(&p, 0, 0, &gl(), Opening::Deep).unwrap_err();
    assert_eq!(err, PiloutError::NoConstraints { airgroup: 0, air: 0 });
}

/// 64 levels of `x_{k+1} = x_k + x_k`: inlining would be 2^64 nodes, the memoised degree is linear.
#[test]
fn doubling_chain_is_not_inlined() {
    let mut e = vec![mul(witness(0, 0), witness(1, 0))];
    for k in 0..64 {
        e.push(add(exp(k), exp(k)));
    }
    e.push(mul(exp(64), exp(0)));
    let cs = system(e, &[65]);
    assert_eq!(check_im(&cs, &choice(&[]), 4, true).unwrap().root_degrees, vec![4]);
    assert_eq!(check_im(&cs, &choice(&[64]), 3, true).unwrap().im_degrees, vec![2]);
    let set = search_im(&cs, &DegreePolicy::FromBlowup { blowup_bits: 1 }, 1).unwrap();
    check_im(&cs, &im_order(&cs, &set).unwrap(), 3, true).unwrap();
}

#[test]
fn rejects_leftover_refs() {
    let mut cs = system(chain(), &[3]);
    cs.exprs[1] = Expr::Ref(99);
    assert!(matches!(check_im(&cs, &choice(&[]), 9, true).unwrap_err(), ImError::BadRef(_)));
    let mut cs = system(chain(), &[3]);
    cs.exprs[0] = Expr::Mul(Box::new(Expr::Ref(2)), Box::new(Expr::Ref(1)));
    assert!(matches!(check_im(&cs, &choice(&[]), 9, true).unwrap_err(), ImError::BadRef(_)));
    let mut cs = system(chain(), &[3]);
    cs.constraints[0].e = Expr::Ref(99);
    assert!(matches!(check_im(&cs, &choice(&[]), 9, true).unwrap_err(), ImError::BadRef(_)));
}
