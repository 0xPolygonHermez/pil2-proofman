//! `im`: check_im, substitute, order, and the untrusted search against the old one.

mod common;

use std::collections::BTreeSet;

use common::*;
use num_bigint::BigUint;
use pil2_pilout::pilout as pb;
use pil_info::core::from_pilout::from_pilout;
use pil_info::core::im::{check_im, im_order, substitute, Checked, ImChoice, ImError};
use pil_info::core::ir::{degree, eval_expr, visit, Boundary, ColType, ConstraintSystem, Env, Expr};
use pil_info::pil::im_polynomials::calculate_intermediate_polynomials;
use pil_info::pil::prepare::prepare_pil;
use pil_info::search::im::search_im;
use pil_info::{DegreePolicy, Opening};

fn cs_of(p: &pb::PilOut) -> ConstraintSystem {
    from_pilout(p, 0, 0, &gl(), Opening::Deep).expect("a lowerable pilout")
}

/// 4 witness columns a,b,c,d; one every-row constraint per entry of `roots`.
fn system(exprs: Vec<pb::Expression>, roots: &[u32]) -> ConstraintSystem {
    cs_of(&pilout(&["a", "b", "c", "d"], &[], exprs, roots.iter().map(|r| every_row(*r)).collect()))
}

/// 0: a*b, 1: e0*c, 2: e1*d (degree 4), 3: e2*a (degree 5).
fn chain() -> Vec<pb::Expression> {
    vec![
        mul(witness(0, 0), witness(1, 0)),
        mul(exp(0), witness(2, 0)),
        mul(exp(1), witness(3, 0)),
        mul(exp(2), witness(0, 0)),
    ]
}

fn choice(v: &[u32]) -> ImChoice {
    ImChoice { exprs: v.to_vec() }
}

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
    // e2 = e1*d, e1 = e0*c: with e1 promoted, e2 is degree 2, not 4
    let cs = system(chain(), &[3]);
    let got = check_im(&cs, &choice(&[1, 2]), 3, true).unwrap();
    assert_eq!(got.im_degrees, vec![3, 2]);
    assert_eq!(got.root_degrees, vec![2]);
}

#[test]
fn rejects_non_promotable() {
    let mut e = chain();
    e.push(sub(constant("7"), constant("2"))); // 4: a degree-0 expression
    let cs = system(e, &[1]);
    assert_eq!(check_im(&cs, &choice(&[99]), 3, true).unwrap_err(), ImError::NotPromotable { expr: 99 });
    assert_eq!(check_im(&cs, &choice(&[4]), 3, true).unwrap_err(), ImError::NotPromotable { expr: 4 });
}

#[test]
fn rejects_unsorted_and_duplicate() {
    let cs = system(chain(), &[1]);
    assert_eq!(check_im(&cs, &choice(&[1, 0]), 4, true).unwrap_err(), ImError::Unsorted);
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
fn im_order_is_ascending_expression_id() {
    assert_eq!(im_order(&BTreeSet::from([7, 3])).unwrap(), choice(&[3, 7]));
    assert_eq!(im_order(&BTreeSet::new()).unwrap(), choice(&[]));
}

#[test]
fn substitute_resolves_refs_and_promoted_become_columns() {
    let cs = system(chain(), &[3]);
    let im = choice(&[1, 2]);
    let root = Expr::Ref(3);
    let sub_root = substitute(&cs, &im, &root);
    let mut refs = 0;
    visit(&sub_root, &mut |n| refs += matches!(n, Expr::Ref(_)) as u32);
    assert_eq!(refs, 0, "no Ref survives");
    assert_eq!(degree(&cs, &sub_root), 2);
    // each promoted Ref is its own distinct, offset-0 column past the witness columns
    let cols: Vec<Expr> = im.exprs.iter().map(|i| substitute(&cs, &im, &Expr::Ref(*i))).collect();
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
        env.cols.insert((ColType::Witness, None, k, 0), BigUint::from(v));
    }
    let mut full = env.clone();
    for (c, i) in cols.iter().zip(&im.exprs) {
        let Expr::Col { id, .. } = c else { unreachable!() };
        full.cols.insert((ColType::Witness, None, *id, 0), eval_expr(&cs, &Expr::Ref(*i), &env));
    }
    assert_eq!(eval_expr(&cs, &sub_root, &full), eval_expr(&cs, &root, &env));
    // an empty choice only inlines
    let inlined = substitute(&cs, &choice(&[]), &root);
    assert_eq!(eval_expr(&cs, &inlined, &env), eval_expr(&cs, &root, &env));
}

// ---- the old search as oracle -------------------------------------------------------------

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    fn below(&mut self, n: u64) -> u64 {
        self.next() % n
    }
}

/// A seeded random DAG of <= 9 expressions over 4 columns, depth <= 6, every constraint of degree 1..=24.
fn random_pilout(seed: u64) -> (pb::PilOut, ConstraintSystem) {
    random_pilout_in(seed, Opening::Deep)
}

/// As `random_pilout`; under Shplonk the roots also take firstRow/lastRow/everyFrame boundaries.
fn random_pilout_in(seed: u64, opening: Opening) -> (pb::PilOut, ConstraintSystem) {
    let mut rng = Rng(seed);
    loop {
        let n = 3 + rng.below(7) as u32;
        let mut depth: Vec<u32> = Vec::new();
        let mut exprs = Vec::new();
        for i in 0..n {
            let operand = |rng: &mut Rng, depth: &Vec<u32>| -> (Option<pb::Operand>, u32) {
                match rng.below(20) {
                    0..=7 => (witness(rng.below(4) as u32, rng.below(2) as i32), 0),
                    8..=10 => (constant(&(1 + rng.below(9)).to_string()), 0),
                    _ if i > 0 => {
                        let j = rng.below(i as u64) as usize;
                        if depth[j] < 6 {
                            (exp(j as u32), depth[j])
                        } else {
                            (witness(rng.below(4) as u32, 0), 0)
                        }
                    }
                    _ => (witness(rng.below(4) as u32, 0), 0),
                }
            };
            let (l, dl) = operand(&mut rng, &depth);
            let (r, dr) = operand(&mut rng, &depth);
            let (e, d) = match rng.below(9) {
                0..=2 => (add(l, r), dl.max(dr) + 1),
                3..=4 => (sub(l, r), dl.max(dr) + 1),
                5..=7 => (mul(l, r), dl.max(dr) + 1),
                _ => (neg(l), dl + 1),
            };
            exprs.push(e);
            depth.push(d);
        }
        let mut roots = vec![n - 1];
        for _ in 0..rng.below(3) {
            roots.push(rng.below(n as u64) as u32);
        }
        let cons = roots
            .iter()
            .map(|r| match (opening, rng.below(4)) {
                (Opening::Deep, _) | (_, 0) => every_row(*r),
                (_, 1) => first_row(*r),
                (_, 2) => last_row(*r),
                _ => every_frame(*r, 0, 1),
            })
            .collect();
        let p = pilout(&["a", "b", "c", "d"], &[], exprs, cons);
        let Ok(cs) = from_pilout(&p, 0, 0, &gl(), opening) else { continue };
        let degs: Vec<u32> = cs.constraints.iter().map(|c| degree(&cs, &c.e)).collect();
        if degs.iter().all(|d| (1..=24).contains(d)) {
            return (p, cs);
        }
    }
}

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

/// The old pipeline on air (`g`, `air`): the promoted set, its q_deg and the constraint dimension, or None if infeasible.
fn old_search(p: &pb::PilOut, g: usize, air: usize, policy: &DegreePolicy) -> Option<(BTreeSet<u32>, i64, usize)> {
    let prep = prepare_pil(p, g, air, &gl()).expect("prepare_pil");
    let q_dim = prep.constraint_poly.q_dim;
    let r = calculate_intermediate_polynomials(
        &prep.expressions,
        prep.constraint_poly.c_exp_id,
        policy,
        q_dim,
        &prep.symbols,
    )
    .ok()?;
    Some((r.im_exps.iter().map(|&e| e as u32).collect(), r.q_deg, q_dim))
}

/// `search_im` returns the old set, and `check_im` the old q_deg, on air (`g`, `a`) under every policy.
fn assert_matches_old(p: &pb::PilOut, cs: &ConstraintSystem, g: usize, a: usize, what: &str) -> usize {
    let mut compared = 0;
    for policy in policies() {
        let old = old_search(p, g, a, &policy);
        let new = search_im(cs, &policy, old.as_ref().map_or(1, |o| o.2));
        match (old, new) {
            (Some((set, old_q, _)), Ok(got)) => {
                assert_eq!(got, set, "{what} {policy:?}");
                let im = im_order(&got).unwrap();
                let max = policy.max_constraint_degree() as u32;
                let checked = check_im(cs, &im, max, stark_floor(&policy)).unwrap_or_else(|e| panic!("{what}: {e:?}"));
                assert_eq!(checked.q_deg as i64, old_q, "{what} {policy:?}");
                compared += 1;
            }
            (None, Err(_)) => {}
            (o, n) => panic!("{what} {policy:?}: old {o:?} vs new {n:?}"),
        }
    }
    compared
}

#[test]
fn search_equals_old_set() {
    let compared: usize = (0..200)
        .map(|seed| {
            let (p, cs) = random_pilout(seed);
            assert_matches_old(&p, &cs, 0, 0, &format!("seed {seed}"))
        })
        .sum();
    assert!(compared > 500, "only {compared} feasible comparisons");
}

/// pilfflonk protocol.md#degree-search: qDeg = max(deg c_i + δ_i) - 1, δ_i = 1 off everyRow. The old search
/// never counts that δ (its boundary `mul(e, Zi)` is built after add_info, so Zi has expDeg 0): the new q_deg
/// is never below the old one, and seed 4 (a degree-2 everyFrame root) is where the old under-sizes Q.
#[test]
fn shplonk_boundaries_follow_the_spec() {
    let (mut checked, mut above_old) = (0, 0);
    for seed in 0..200 {
        let (p, cs) = random_pilout_in(seed, Opening::Shplonk);
        for policy in policies() {
            let Some((_, old_q, q_dim)) = old_search(&p, 0, 0, &policy) else { continue };
            let Ok(set) = search_im(&cs, &policy, q_dim) else { continue };
            let im = im_order(&set).unwrap();
            let got = check_im(&cs, &im, policy.max_constraint_degree() as u32, stark_floor(&policy)).unwrap();
            assert!(got.q_deg as i64 >= old_q, "seed {seed} {policy:?}");
            above_old += (got.q_deg as i64 > old_q) as u32;
            checked += 1;
        }
    }
    assert!(checked > 500 && above_old > 0, "{checked} checked, {above_old} above the old q_deg");
    let (p, cs) = random_pilout_in(4, Opening::Shplonk);
    assert_eq!(
        cs.constraints.iter().map(|c| c.boundary.clone()).collect::<Vec<_>>(),
        vec![Boundary::EveryFrame { min: 0, max: 1 }]
    );
    let policy = DegreePolicy::Search { max: 3 };
    assert_eq!(old_search(&p, 0, 0, &policy).map(|o| (o.0, o.1)), Some((BTreeSet::new(), 1)));
    assert_eq!(
        check_im(&cs, &choice(&[]), 3, false).unwrap(),
        Checked { q_deg: 2, root_degrees: vec![3], im_degrees: vec![] }
    );
}

#[test]
fn search_output_is_checked() {
    let mut checked = 0;
    for seed in 0..200 {
        let (p, cs) = random_pilout(seed);
        for policy in policies() {
            let q_dim = old_search(&p, 0, 0, &policy).map_or(1, |o| o.2);
            let Ok(set) = search_im(&cs, &policy, q_dim) else { continue };
            let max = policy.max_constraint_degree() as u32;
            let im = im_order(&set).unwrap();
            check_im(&cs, &im, max, stark_floor(&policy)).unwrap_or_else(|e| panic!("seed {seed} {policy:?}: {e:?}"));
            checked += 1;
        }
    }
    assert!(checked > 500);
}

/// No memo hazard in 20000 seeds x 5 policies: the old q_deg always equals the derived one. This seed is kept
/// because only its im root (not a constraint root) sets q_deg, which a roots-only derivation misses.
const HAZARD_SEED: u64 = 2781;

#[test]
fn memo_hazard_regression() {
    let (p, cs) = random_pilout(HAZARD_SEED);
    for policy in policies() {
        let (set, old_q, _) = old_search(&p, 0, 0, &policy).expect("feasible");
        let im = im_order(&set).unwrap();
        let got = check_im(&cs, &im, policy.max_constraint_degree() as u32, stark_floor(&policy))
            .unwrap_or_else(|e| panic!("{policy:?}: old set rejected: {e:?}"));
        let roots = cs.constraints.iter().map(|c| degree(&cs, &substitute(&cs, &im, &c.e))).max().unwrap();
        assert_eq!(got.q_deg as i64, old_q, "{policy:?}");
        assert_eq!((roots, got.q_deg), (2, 2), "{policy:?}: the im root drives q_deg");
    }
}

#[test]
fn empty_system_has_q_deg_zero_or_the_floor() {
    let cs = system(chain(), &[]);
    let want = Checked { q_deg: 0, root_degrees: vec![], im_degrees: vec![] };
    assert_eq!(check_im(&cs, &choice(&[]), 3, false).unwrap(), want);
    assert_eq!(check_im(&cs, &choice(&[]), 3, true).unwrap().q_deg, 1);
    assert_eq!(search_im(&cs, &DegreePolicy::FromBlowup { blowup_bits: 1 }, 1).unwrap(), BTreeSet::new());
}

#[test]
fn real_pilouts_match_old_search() {
    let mut airs = 0;
    for (f, p) in real_pilouts() {
        for (g, ag) in p.air_groups.iter().enumerate() {
            for a in 0..ag.airs.len() {
                let cs = from_pilout(&p, g, a, &gl(), Opening::Deep).unwrap_or_else(|e| panic!("{f:?} {g}/{a}: {e}"));
                airs += assert_matches_old(&p, &cs, g, a, &format!("{f:?} {g}/{a}"));
            }
        }
    }
    assert!(airs > 0, "no air compared");
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
    check_im(&cs, &im_order(&set).unwrap(), 3, true).unwrap();
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
