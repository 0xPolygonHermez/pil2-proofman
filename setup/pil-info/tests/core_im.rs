//! `im`: check_im, substitute, order, and the untrusted search.

mod common;

use common::*;
use num_bigint::BigUint;
use pil_info::core::im::{check_im, im_order, substitute, Checked, ImError};
use pil_info::core::layout::quotient_dim;
use pil_info::core::ir::{degree, eval_expr, visit, ColRef, ColType, Env, Expr};
use pil_info::search::im::search_im;
use pil_info::DegreePolicy;

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
fn rejects_leftover_refs() {
    let mut cs = system(chain(), &[3]);
    cs.exprs[1] = Expr::Ref(99);
    assert!(matches!(check_im(&cs, &choice(&[]), 9, true).unwrap_err(), ImError::Unresolved(_)));
    let mut cs = system(chain(), &[3]);
    cs.exprs[0] = Expr::Mul(Box::new(Expr::Ref(2)), Box::new(Expr::Ref(1)));
    assert!(matches!(check_im(&cs, &choice(&[]), 9, true).unwrap_err(), ImError::Unresolved(_)));
    let mut cs = system(chain(), &[3]);
    cs.constraints[0].e = Expr::Ref(99);
    assert!(matches!(check_im(&cs, &choice(&[]), 9, true).unwrap_err(), ImError::Unresolved(_)));
}

/// The search walks a 100k-deep chain of references without recursing over it (it used to overflow the stack).
#[test]
fn search_is_iterative_on_a_deep_chain() {
    let n = 100_000;
    let mut e = vec![mul(witness(0, 0), witness(1, 0))];
    e.extend((1..n).map(|k| add(exp(k - 1), witness(0, 0))));
    let cs = cs_of(&pilout(&["a", "b"], &[], e, vec![every_row(n - 1)]));
    let set = std::thread::spawn(move || search_im(&cs, &policies()[0], 3)).join().expect("no stack overflow");
    assert!(set.is_ok());
}
