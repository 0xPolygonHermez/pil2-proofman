//! Task 9: topological numbering, table passes, canonical im order, fixed roles.

mod common;

use std::collections::BTreeSet;

use common::*;
use num_bigint::BigUint;
use pil2_pilout::pilout::{self as pb, constraint, expression, operand};
use pil_info::core::dag::build::build_dag;
use pil_info::core::from_pilout::{from_pilout, PiloutError};
use pil_info::core::im::{im_order, ImChoice};
use pil_info::core::ir::{
    degree, degree_table, dim, dim_table, eval_expr, eval_table, visit, ColRef, ColType, ConstraintSystem, Env, Expr,
};
use pil_info::core::layout::{layout, LayoutParams};
use pil_info::emit::prover::to_arena;
use pil_info::expr::expression::{ExprChild, Expression};
use pil_info::Opening;

// ---- helpers ----

fn remap_operand(o: &mut Option<pb::Operand>, p: &[u32]) {
    if let Some(pb::Operand { operand: Some(operand::Operand::Expression(e)) }) = o {
        e.idx = p[e.idx as usize];
    }
}

fn remap_constraint(c: &mut pb::Constraint, p: &[u32]) {
    use constraint::Constraint as C;
    let slot = match c.constraint.as_mut() {
        Some(C::EveryRow(x)) => &mut x.expression_idx,
        Some(C::FirstRow(x)) => &mut x.expression_idx,
        Some(C::LastRow(x)) => &mut x.expression_idx,
        Some(C::EveryFrame(x)) => &mut x.expression_idx,
        None => return,
    };
    if let Some(e) = slot {
        e.idx = p[e.idx as usize];
    }
}

/// The pilout with expression `i` moved to index `p[i]`, every reference rewritten.
fn permute(pil: &pb::PilOut, p: &[u32]) -> pb::PilOut {
    let mut out = pil.clone();
    let air = &mut out.air_groups[0].airs[0];
    let mut exprs = air.expressions.clone();
    for (i, e) in air.expressions.iter().enumerate() {
        let mut e = e.clone();
        match e.operation.as_mut() {
            Some(expression::Operation::Add(x)) => {
                remap_operand(&mut x.lhs, p);
                remap_operand(&mut x.rhs, p);
            }
            Some(expression::Operation::Sub(x)) => {
                remap_operand(&mut x.lhs, p);
                remap_operand(&mut x.rhs, p);
            }
            Some(expression::Operation::Mul(x)) => {
                remap_operand(&mut x.lhs, p);
                remap_operand(&mut x.rhs, p);
            }
            Some(expression::Operation::Neg(x)) => remap_operand(&mut x.value, p),
            None => {}
        }
        exprs[p[i] as usize] = e;
    }
    air.expressions = exprs;
    air.constraints.iter_mut().for_each(|c| remap_constraint(c, p));
    out
}

fn random_perm(n: usize, rng: &mut Rng) -> Vec<u32> {
    let mut v: Vec<u32> = (0..n as u32).collect();
    for i in (1..n).rev() {
        v.swap(i, rng.below(i as u64 + 1) as usize);
    }
    v
}

fn lower(p: &pb::PilOut) -> Result<ConstraintSystem, PiloutError> {
    from_pilout(p, 0, 0, &gl(), Opening::Deep)
}

fn refs(e: &Expr) -> Vec<u32> {
    let mut r = Vec::new();
    visit(e, &mut |n| {
        if let Expr::Ref(t) = n {
            r.push(*t);
        }
    });
    r
}

fn col(id: u32) -> Expr {
    Expr::Col { col: ColType::Witness, commit_id: None, id, offset: 0 }
}

fn mul_e(a: Expr, b: Expr) -> Expr {
    Expr::Mul(Box::new(a), Box::new(b))
}

/// The promoted bodies in `im_order` order.
fn ordered_bodies(cs: &ConstraintSystem, wanted: &[Expr]) -> Vec<Expr> {
    let set: BTreeSet<u32> = wanted.iter().map(|w| find(cs, w)).collect();
    let im = im_order(cs, &set).unwrap();
    im.exprs.iter().map(|&i| inlined(cs, &Expr::Ref(i))).collect()
}

// ---- topological renumbering ----

fn assert_topological(cs: &ConstraintSystem) {
    for (i, e) in cs.exprs.iter().enumerate() {
        assert!(refs(e).iter().all(|&t| (t as usize) < i), "expr {i} refs {:?}", refs(e));
    }
}

#[test]
fn refs_point_to_smaller_indices_after_lowering() {
    // forward references (expr 0 uses expr 2) are legal input
    let e = vec![add(exp(2), witness(0, 0)), mul(exp(2), exp(2)), mul(witness(1, 0), witness(2, 0))];
    let cs = system(e, &[0, 1]);
    assert_topological(&cs);
    for seed in 0..40 {
        let (_, cs) = random_pilout(seed);
        assert_topological(&cs);
    }
    for (_, p) in real_pilouts() {
        for (g, ag) in p.air_groups.iter().enumerate() {
            for a in 0..ag.airs.len() {
                if let Ok(cs) = from_pilout(&p, g, a, &gl(), Opening::Deep) {
                    assert_topological(&cs);
                }
            }
        }
    }
}

#[test]
fn permuted_input_gives_the_same_system_and_dag() {
    let mut rng = Rng(7);
    for seed in 0..40 {
        let (p, cs) = random_pilout(seed);
        let n = p.air_groups[0].airs[0].expressions.len();
        let q = permute(&p, &random_perm(n, &mut rng));
        let cs2 = lower(&q).unwrap();
        assert_topological(&cs2);
        assert_eq!(cs.constraints.len(), cs2.constraints.len());
        for (a, b) in cs.constraints.iter().zip(&cs2.constraints) {
            assert_eq!(inlined(&cs, &a.e), inlined(&cs2, &b.e), "seed {seed}");
            assert_eq!((&a.boundary, &a.label), (&b.boundary, &b.label), "seed {seed}");
        }
        assert_eq!((&cs.witness, &cs.fixed, cs.n_rows_bits), (&cs2.witness, &cs2.fixed, cs2.n_rows_bits));
        let none = ImChoice { exprs: Vec::new() };
        assert_eq!(build_dag(&cs, &none).unwrap(), build_dag(&cs2, &none).unwrap(), "seed {seed}");
    }
}

#[test]
fn input_reference_cycle_is_an_ingestion_error() {
    let cyc3 = vec![add(exp(1), witness(0, 0)), add(exp(2), witness(0, 0)), add(exp(0), witness(0, 0))];
    let p = pilout(&["a"], &[], cyc3, vec![every_row(0)]);
    assert!(matches!(lower(&p), Err(PiloutError::CyclicRef { .. })));
    // a cycle hanging off a valid prefix, reached only through a later expression
    let e = vec![mul(witness(0, 0), witness(0, 0)), add(exp(2), exp(0)), add(exp(1), witness(0, 0))];
    let p = pilout(&["a"], &[], e, vec![every_row(0)]);
    assert!(matches!(lower(&p), Err(PiloutError::CyclicRef { .. })));
}

// ---- degree / dim / eval as table passes ----

fn small_env(rng: &mut Rng) -> Env {
    let mut env = Env::default();
    for id in 0..4 {
        for offset in 0..2 {
            let k = ColRef { col: ColType::Witness, commit_id: None, id, offset };
            env.cols.insert(k, BigUint::from(1 + rng.below(1000)));
        }
    }
    env
}

#[test]
fn table_passes_match_inlining_on_random_pilouts() {
    let mut rng = Rng(11);
    for seed in 0..60 {
        let (_, cs) = random_pilout(seed);
        let env = small_env(&mut rng);
        let (dg, dm, ev) = (degree_table(&cs).unwrap(), dim_table(&cs).unwrap(), eval_table(&cs, &env).unwrap());
        for i in 0..cs.exprs.len() {
            let flat = inlined(&cs, &Expr::Ref(i as u32));
            assert_eq!(Ok(dg[i]), degree(&cs, &flat), "degree seed {seed} expr {i}");
            assert_eq!(dm[i], dim(&cs, &flat).unwrap(), "dim seed {seed} expr {i}");
            assert_eq!(Ok(ev[i].clone()), eval_expr(&cs, &flat, &env), "eval seed {seed} expr {i}");
        }
    }
}

/// Node count of each expression inlined, saturating (a test-local table pass).
fn inlined_sizes(cs: &ConstraintSystem) -> Vec<u64> {
    let mut t: Vec<u64> = Vec::new();
    for e in &cs.exprs {
        let mut n = 0u64;
        visit(e, &mut |x| n = n.saturating_add(if let Expr::Ref(r) = x { t[*r as usize] } else { 1 }));
        t.push(n);
    }
    t
}

#[test]
fn table_passes_match_inlining_on_real_pilouts() {
    let mut rng = Rng(3);
    let mut checked = 0;
    for (path, p) in real_pilouts() {
        for (g, ag) in p.air_groups.iter().enumerate() {
            for a in 0..ag.airs.len() {
                let Ok(cs) = from_pilout(&p, g, a, &gl(), Opening::Deep) else { continue };
                let env = small_env(&mut rng);
                let (dg, dm, ev) =
                    (degree_table(&cs).unwrap(), dim_table(&cs).unwrap(), eval_table(&cs, &env).unwrap());
                for (i, n) in inlined_sizes(&cs).into_iter().enumerate() {
                    if n > 1000 {
                        continue;
                    }
                    let flat = inlined(&cs, &Expr::Ref(i as u32));
                    let at = format!("{path:?} {g}/{a} expr {i}");
                    assert_eq!(Ok(dg[i]), degree(&cs, &flat), "{at}");
                    assert_eq!(dm[i], dim(&cs, &flat).unwrap(), "{at}");
                    assert_eq!(Ok(ev[i].clone()), eval_expr(&cs, &flat, &env), "{at}");
                    checked += 1;
                }
            }
        }
    }
    assert!(checked > 1000, "{checked}");
}

/// 64 levels of x_{k+1} = x_k + x_k: inlining is 2^64 nodes; the tables must stay linear.
#[test]
fn table_passes_finish_fast_on_a_doubling_chain() {
    let mut e = vec![mul(witness(0, 1), witness(1, 0))];
    for k in 0..64 {
        e.push(add(exp(k), exp(k)));
    }
    let cs = system(e, &[64]);
    let t = std::time::Instant::now();
    let dg = degree_table(&cs).unwrap();
    let dm = dim_table(&cs).unwrap();
    let mut env = Env::default();
    env.cols.insert(ColRef { col: ColType::Witness, commit_id: None, id: 0, offset: 1 }, BigUint::from(3u32));
    env.cols.insert(ColRef { col: ColType::Witness, commit_id: None, id: 1, offset: 0 }, BigUint::from(5u32));
    let ev = eval_table(&cs, &env).unwrap();
    assert!(t.elapsed().as_secs() < 2, "{:?}", t.elapsed());
    assert_eq!((dg[64], dm[64]), (2, 1));
    assert_eq!(ev[64], (BigUint::from(15u32) << 64usize) % cs.field.modulus());
}

// ---- canonical im order ----

/// 7: c*d and 3: a*b referenced in the order (e7, e3); fillers keep the indices.
fn two_im() -> pb::PilOut {
    let e = vec![
        add(witness(0, 0), witness(1, 0)),
        sub(witness(0, 0), witness(1, 0)),
        add(witness(2, 0), witness(3, 0)),
        mul(witness(0, 0), witness(1, 0)), // 3
        sub(witness(2, 0), witness(3, 0)),
        add(witness(0, 0), witness(2, 0)),
        add(witness(1, 0), witness(3, 0)),
        mul(witness(2, 0), witness(3, 0)), // 7
        mul(exp(7), witness(0, 0)),
        mul(exp(3), witness(1, 0)),
    ];
    pilout(&["a", "b", "c", "d"], &[], e, vec![every_row(8), every_row(9)])
}

#[test]
fn im_order_is_first_appearance() {
    let (ab, cd) = (mul_e(col(0), col(1)), mul_e(col(2), col(3)));
    let cs = lower(&two_im()).unwrap();
    assert_eq!(ordered_bodies(&cs, &[ab.clone(), cd.clone()]), vec![cd.clone(), ab.clone()]);
    // a renumbered pilout promotes the same expressions in the same order
    let mut rng = Rng(5);
    for _ in 0..20 {
        let q = permute(&two_im(), &random_perm(10, &mut rng));
        let cs = lower(&q).unwrap();
        assert_eq!(ordered_bodies(&cs, &[ab.clone(), cd.clone()]), vec![cd.clone(), ab.clone()]);
    }
}

/// Five independent products in one stage, referenced out of index order through combiners.
fn five_im() -> (pb::PilOut, Vec<Expr>) {
    let pairs = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3)];
    let mut e: Vec<pb::Expression> = pairs.iter().map(|&(i, j)| mul(witness(i, 0), witness(j, 0))).collect();
    // reference order: 3, 0, 4, 2, 1
    e.push(add(exp(3), exp(0))); // 5
    e.push(add(exp(4), exp(2))); // 6
    e.push(mul(exp(5), witness(3, 0))); // 7
    e.push(mul(exp(6), witness(2, 0))); // 8
    e.push(mul(exp(1), witness(0, 0))); // 9
    let p = pilout(&["a", "b", "c", "d"], &[], e, vec![every_row(7), every_row(8), every_row(9)]);
    let body = |k: usize| mul_e(col(pairs[k].0), col(pairs[k].1));
    (p, [3, 0, 4, 2, 1].iter().map(|&k| body(k)).collect())
}

#[test]
fn several_im_pols_in_one_stage_follow_appearance() {
    let (p, want) = five_im();
    let all: Vec<Expr> = (0..5).map(|k| want[[1, 4, 3, 0, 2][k]].clone()).collect();
    let cs = lower(&p).unwrap();
    assert_eq!(ordered_bodies(&cs, &all), want);
    let mut rng = Rng(9);
    for _ in 0..30 {
        let cs = lower(&permute(&p, &random_perm(10, &mut rng))).unwrap();
        assert_eq!(ordered_bodies(&cs, &all), want);
    }
}

#[test]
fn im_order_sorts_by_stage_first() {
    // stage 1: a*b (P), a*a (S); stage 2: x*y (Q), alpha*a (R); referenced Q, P, R, S
    let e = vec![
        mul(witness(0, 0), witness(1, 0)),             // 0 P
        mul(witness(0, 0), witness(0, 0)),             // 1 S
        mul(witness_at(2, 0, 0), witness_at(2, 1, 0)), // 2 Q
        mul(challenge(2, 0), witness(0, 0)),           // 3 R
        mul(exp(2), witness(0, 0)),                    // 4
        mul(exp(0), witness(1, 0)),                    // 5
        mul(exp(3), witness(1, 0)),                    // 6
        mul(exp(1), witness(1, 0)),                    // 7
    ];
    let p = pilout2(&[], e, (4..8).map(every_row).collect());
    let (a, b) = (col(0), col(1));
    let (x, y) = (col(2), col(3));
    let alpha = Expr::Challenge { stage: 2, stage_id: 0 };
    let (pp, s, q, r) = (mul_e(a.clone(), b), mul_e(a.clone(), a.clone()), mul_e(x, y), mul_e(alpha, a));
    let want = vec![pp.clone(), s.clone(), q.clone(), r.clone()];
    let mut rng = Rng(13);
    for k in 0..10 {
        let q_in = if k == 0 { p.clone() } else { permute(&p, &random_perm(8, &mut rng)) };
        let cs = lower(&q_in).unwrap();
        assert_eq!(ordered_bodies(&cs, &[q.clone(), r.clone(), s.clone(), pp.clone()]), want);
    }
}

// ---- arena order and roles ----

/// The arena entry with arena references followed; numbering-free.
fn fp(arena: &[Expression], e: &Expression) -> String {
    if e.op == "exp" {
        return fp(arena, &arena[e.id.unwrap()]);
    }
    let kids: Vec<String> = e
        .values
        .iter()
        .map(|c| match c {
            ExprChild::Id(i) => fp(arena, &arena[*i]),
            ExprChild::Inline(x) => fp(arena, x),
        })
        .collect();
    format!("{}:{:?}:{:?}:{:?}:{:?}:({})", e.op, e.id, e.value, e.row_offset, e.stage_id, kids.join(","))
}

fn arena_of(p: &pb::PilOut, bodies: &[Expr]) -> Vec<String> {
    let cs = lower(p).unwrap();
    let set: BTreeSet<u32> = bodies.iter().map(|w| find(&cs, w)).collect();
    let im = im_order(&cs, &set).unwrap();
    let params = LayoutParams { im_pols_stages: false, deep_challenges: true, q_deg: 2, air_name: "S".into() };
    let lay = layout(&cs, &im, &params).unwrap();
    let t = to_arena(&cs, &im, &lay).unwrap();
    t.arena.iter().map(|e| fp(&t.arena, e)).collect()
}

#[test]
fn arena_order_independent_of_pilout_numbering() {
    let bodies = [mul_e(col(0), col(1)), mul_e(col(2), col(3))];
    let want = arena_of(&two_im(), &bodies);
    let mut rng = Rng(21);
    for _ in 0..20 {
        assert_eq!(arena_of(&permute(&two_im(), &random_perm(10, &mut rng)), &bodies), want);
    }
}

#[test]
fn roles_are_fixed() {
    let cs = lower(&two_im()).unwrap();
    let (ab, cd) = (find(&cs, &mul_e(col(0), col(1))), find(&cs, &mul_e(col(2), col(3))));
    let n = cs.exprs.len();
    let mut prefix: Option<Vec<String>> = None;
    for set in [vec![], vec![cd], vec![cd, ab]] {
        let set: BTreeSet<u32> = set.into_iter().collect();
        let im = im_order(&cs, &set).unwrap();
        let params = LayoutParams { im_pols_stages: false, deep_challenges: true, q_deg: 2, air_name: "S".into() };
        let lay = layout(&cs, &im, &params).unwrap();
        let t = to_arena(&cs, &im, &lay).unwrap();
        // pilout expressions in index order first (whatever the im choice), C last
        let head: Vec<String> = t.arena[..n].iter().map(|e| fp(&t.arena, e)).collect();
        assert_eq!(*prefix.get_or_insert(head.clone()), head);
        assert_eq!(t.roles.c_exp, t.arena.len() - 1);
        assert!(t.arena.len() > n + im.exprs.len());
    }
}
