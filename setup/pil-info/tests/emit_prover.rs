//! `to_arena`, `emit_prover`: today's arena, roles and prover code, rebuilt from the typed IR.

mod common;

use common::*;
use pil2_pilout::pilout as pb;
use pil_info::core::im::ImChoice;
use pil_info::core::layout::{layout, LayoutParams};
use pil_info::emit::prover::{emit_prover, to_arena, EmitParams, ProverCode};
use pil_info::expr::expression::{ExprChild, Expression};

/// Every feasible air, random then real.
fn for_each_run(min: usize, mut f: impl FnMut(&Run, &str)) {
    let any = |p: &pb::PilOut, g, a| (1..=3).find_map(|b| run(p, g, a, b));
    let mut n = 0;
    for seed in 0..40u64 {
        let (p, _) = random_pilout(seed);
        let Some(run) = any(&p, 0, 0) else { continue };
        f(&run, &format!("random {seed}"));
        n += 1;
    }
    assert!(n >= 20, "only {n} feasible random pilouts");
    for (file, p) in real_pilouts() {
        for (g, ag) in p.air_groups.iter().enumerate() {
            for a in 0..ag.airs.len() {
                let Some(run) = any(&p, g, a) else { continue };
                f(&run, &format!("{file:?} {g}/{a}"));
                n += 1;
            }
        }
    }
    assert!(n > min);
}

fn emitted(run: &Run) -> ProverCode {
    let c = core_out(run);
    emit_prover(&run.cs, &run.im, &c.lay, &c.ops, &run.chk, c.exprs, &EmitParams { air_id: 0, airgroup_id: 0 }).unwrap()
}

#[test]
fn emit_is_deterministic() {
    for_each_run(20, |run, what| {
        let (x, y) = (emitted(run), emitted(run));
        let json = |r: &ProverCode| serde_json::to_string(&r.code.expressions_info).unwrap();
        assert_eq!(json(&x), json(&y), "{what}");
    });
}

/// One printed line per im column, and one constraint per im root after the pilout's.
#[test]
fn im_columns_are_printed_and_rooted() {
    let mut with_im = 0;
    for_each_run(20, |run, what| {
        let e = emitted(run);
        let n_im = run.im.exprs.len();
        assert_eq!(e.im_pols_info.0.len() + e.im_pols_info.1.len(), n_im, "{what}");
        assert_eq!(e.constraints.len(), run.cs.constraints.len() + n_im, "{what}");
        assert!(
            e.code.expressions_info.constraints.iter().all(|c| c.line.as_ref().is_some_and(|l| l.ends_with(" == 0"))),
            "{what}"
        );
        with_im += (n_im > 0) as usize;
    });
    assert!(with_im > 0, "no air with im columns");
}

/// 64 levels of `x_{k+1} = x_k + x_k`: inlining would be 2^64 nodes, the memoised arena is linear.
#[test]
fn to_arena_doubling_chain_is_linear() {
    let mut e = vec![mul(witness(0, 1), witness(1, 0))];
    for k in 0..64 {
        e.push(add(exp(k), exp(k)));
    }
    e.push(mul(exp(64), exp(0)));
    let cs = system(e, &[65]);
    for im in [vec![], vec![32], vec![64]] {
        let im = ImChoice { exprs: im };
        let params = LayoutParams { im_pols_stages: false, deep_challenges: true, q_deg: 1, air_name: "S".into() };
        let lay = layout(&cs, &im, &params).unwrap();
        let t = to_arena(&cs, &im, &lay).unwrap();
        assert!(t.arena.len() < cs.exprs.len() + 8 * (im.exprs.len() + 2), "arena {} nodes", t.arena.len());
        fn nodes(e: &Expression) -> usize {
            1 + e.values.iter().map(|c| if let ExprChild::Inline(x) = c { nodes(x) } else { 1 }).sum::<usize>()
        }
        let total: usize = t.arena.iter().map(nodes).sum();
        assert!(total < 100_000, "{total} nodes");
    }
}
