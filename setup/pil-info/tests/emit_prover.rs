//! `to_arena`, `emit_prover`: today's arena, roles and prover code, rebuilt from the typed IR.

mod common;

use std::collections::BTreeMap;

use common::*;
use pil2_pilout::pilout as pb;
use pil_info::core::im::ImChoice;
use pil_info::core::layout::{layout, LayoutParams};
use pil_info::emit::prover::{emit_prover, to_arena, EmitParams};
use pil_info::expr::expression::{ExprChild, Expression};
use pil_info::pil::gen_code::ExpressionCodeEntry;
use pil_info::pil::im_polynomials::add_im_polynomials;
use pil_info::pil::prepare::prepare_pil;
use pil_info::types::output::CodeRef;

/// The old arena as `pil_info` leaves it before code generation, plus the old roles.
struct OldArena {
    arena: Vec<Expression>,
    c_exp: usize,
    kept: Vec<usize>,
}

fn old_arena(p: &pb::PilOut, g: usize, a: usize, run: &OldRun) -> OldArena {
    let cfg = pil_info::PilInfoCfg::goldilocks(run.blowup_bits);
    let prep = prepare_pil(p, g, a, &cfg.field).unwrap();
    let (mut exprs, mut cons, mut syms) = (prep.expressions, prep.constraints, prep.symbols);
    let kept = (0..exprs.len()).filter(|i| exprs[*i].keep.unwrap_or(false)).collect();
    let bounds: Vec<_> = prep
        .boundaries
        .iter()
        .map(|b| (b.name.clone(), b.offset_min.map(|v| v as i64), b.offset_max.map(|v| v as i64)))
        .collect();
    let mut c_exp = prep.constraint_poly.c_exp_id;
    let im_exps: Vec<usize> = run.im.exprs.iter().map(|e| *e as usize).collect();
    let mut n_commitments = prep.setup.n_commitments;
    add_im_polynomials(
        &mut exprs,
        &mut cons,
        &mut syms,
        &prep.setup.name,
        a,
        g,
        prep.setup.n_stages,
        &mut n_commitments,
        &mut c_exp,
        &im_exps,
        run.old.q_deg,
        run.im_pols_stages,
        &bounds,
        &cfg.field,
    )
    .unwrap();
    OldArena { arena: exprs, c_exp, kept }
}

/// Structure and the derived fields the old passes set (line is the generator's).
fn same(a: &Expression, b: &Expression, path: &str) {
    assert_eq!(a.op, b.op, "{path}: op");
    assert_eq!(a.id, b.id, "{path}: id");
    assert_eq!(a.value, b.value, "{path}: value");
    assert_eq!(a.row_offset, b.row_offset, "{path}: row_offset");
    assert_eq!(a.commit_id, b.commit_id, "{path}: commit_id");
    assert_eq!(a.stage_id, b.stage_id, "{path}: stage_id");
    assert_eq!(a.airgroup_id, b.airgroup_id, "{path}: airgroup_id");
    assert_eq!(a.boundary_id, b.boundary_id, "{path}: boundary_id");
    assert_eq!(a.boundary, b.boundary, "{path}: boundary");
    assert_eq!((a.im_pol, a.pol_id), (b.im_pol, b.pol_id), "{path}: im_pol/pol_id");
    assert_eq!(a.keep.unwrap_or(false), b.keep.unwrap_or(false), "{path}: keep");
    assert_eq!((a.exp_deg, a.dim, a.stage), (b.exp_deg, b.dim, b.stage), "{path}: exp_deg/dim/stage");
    assert_eq!(a.rows_offsets, b.rows_offsets, "{path}: rows_offsets");
    assert_eq!(a.values.len(), b.values.len(), "{path}: arity");
    for (i, (x, y)) in a.values.iter().zip(&b.values).enumerate() {
        match (x, y) {
            (ExprChild::Id(x), ExprChild::Id(y)) => assert_eq!(x, y, "{path}.{i}: child id"),
            (ExprChild::Inline(x), ExprChild::Inline(y)) => same(x, y, &format!("{path}.{i}")),
            _ => panic!("{path}.{i}: Id vs Inline"),
        }
    }
}

/// Every air, random then real, that the old pipeline accepts.
fn for_each_run(min: usize, f: impl FnMut(&pb::PilOut, usize, usize, &OldRun, &str)) {
    for_each_run_in(false, min, f)
}

/// `for_each_run`, both pipelines given `im_pols_stages`.
fn for_each_run_in(ips: bool, min: usize, mut f: impl FnMut(&pb::PilOut, usize, usize, &OldRun, &str)) {
    let any = |p: &pb::PilOut, g, a| (1..=3).find_map(|b| old_run_with(p, g, a, b, ips));
    let mut n = 0;
    for seed in 0..40u64 {
        let (p, _) = random_pilout(seed);
        let Some(run) = any(&p, 0, 0) else { continue };
        f(&p, 0, 0, &run, &format!("random {seed}"));
        n += 1;
    }
    assert!(n >= 20, "only {n} feasible random pilouts");
    for (file, p) in real_pilouts() {
        for (g, ag) in p.air_groups.iter().enumerate() {
            for a in 0..ag.airs.len() {
                let Some(run) = any(&p, g, a) else { continue };
                f(&p, g, a, &run, &format!("{file:?} {g}/{a}"));
                n += 1;
            }
        }
    }
    assert!(n > min);
}

#[test]
fn arena_equals_old() {
    for_each_run(20, |p, g, a, run, what| {
        let old = old_arena(p, g, a, run);
        let new = core_out(run).exprs;
        assert_eq!(new.arena.len(), old.arena.len(), "{what}: arena length");
        for (i, (n, o)) in new.arena.iter().zip(&old.arena).enumerate() {
            same(n, o, &format!("{what} e{i}"));
        }
    });
}

#[test]
fn roles_point_at_old_ids() {
    for_each_run(20, |p, g, a, run, what| {
        let old = old_arena(p, g, a, run);
        let r = core_out(run).exprs.roles;
        assert_eq!(r.c_exp, old.c_exp, "{what}: cExpId");
        assert_eq!(r.c_exp, run.old.c_exp_id, "{what}: cExpId of the old result");
        assert_eq!(r.hints, old.kept, "{what}: kept hint expressions, ascending");
    });
}

fn emitted(run: &OldRun) -> pil_info::pil::gen_code::PilCodeResult {
    let c = core_out(run);
    let s = &run.old.setup;
    let params = EmitParams { air_id: s.air_id, airgroup_id: s.airgroup_id, air_name: s.name.clone(), debug: false };
    emit_prover(&run.cs, &run.im, &c.lay, &c.ops, &run.chk, c.exprs, &params).unwrap().0
}

#[test]
fn emit_is_deterministic() {
    for_each_run(20, |_, _, _, run, what| {
        let (x, y) = (emitted(run), emitted(run));
        let json = |r: &pil_info::pil::gen_code::PilCodeResult| serde_json::to_string(&r.expressions_info).unwrap();
        assert_eq!(json(&x), json(&y), "{what}");
    });
}

/// (nOps, tmp1, tmp3, op multiset) of one block; tmp1/tmp3 count the distinct tmps per dimension.
fn shape(e: &ExpressionCodeEntry, ext: usize) -> (usize, usize, usize, BTreeMap<String, usize>) {
    let mut t1 = std::collections::BTreeSet::new();
    let mut t3 = std::collections::BTreeSet::new();
    let mut ops = BTreeMap::new();
    let mut tmp = |r: &CodeRef| {
        if r.ref_type == "tmp" {
            if r.dim == ext && ext != 1 {
                t3.insert(r.id)
            } else {
                t1.insert(r.id)
            };
        }
    };
    for c in &e.code {
        tmp(&c.dest);
        c.src.iter().for_each(&mut tmp);
        let r = |r: &CodeRef| format!("{}:{}:{:?}:{:?}:{}", r.ref_type, r.id, r.prime, r.value, r.dim);
        let key = format!("{} {}", c.op, c.src.iter().map(r).collect::<Vec<_>>().join(" "));
        *ops.entry(key).or_insert(0) += 1;
    }
    (e.code.len(), t1.len(), t3.len(), ops)
}

#[test]
fn prover_ops_equal_old() {
    for_each_run(20, |_, _, _, run, what| {
        let new = emitted(run);
        let (n, o) = (&new.expressions_info.expressions_code, &run.old.pil_code.expressions_info.expressions_code);
        assert_eq!(n.len(), o.len(), "{what}: number of blocks");
        let ext = run.cs.field.ext_dim();
        for (x, y) in n.iter().zip(o) {
            assert_eq!((x.exp_id, x.stage, x.tmp_used), (y.exp_id, y.stage, y.tmp_used), "{what}: block ids");
            assert_eq!(shape(x, ext), shape(y, ext), "{what}: expression {}", x.exp_id);
        }
    });
}

#[test]
fn hints_gsum_gprod_expressions_equal_old() {
    let mut with_hints = 0;
    for_each_run(20, |_, _, _, run, what| {
        let new = emitted(run);
        let (n, o) = (&new.expressions_info, &run.old.pil_code.expressions_info);
        let (nh, oh) = (serde_json::to_value(&n.hints_info).unwrap(), serde_json::to_value(&o.hints_info).unwrap());
        assert_eq!(nh, oh, "{what}: hintsInfo");
        // the kept expressions the hints read are the same blocks
        let ids = |i: &pil_info::pil::gen_code::ExpressionsInfo| {
            i.expressions_code.iter().map(|e| (e.exp_id, e.tmp_used, e.code.len())).collect::<Vec<_>>()
        };
        assert_eq!(ids(n), ids(o), "{what}: hint expression blocks");
        if !o.hints_info.is_empty() {
            with_hints += 1;
        }
    });
    assert!(with_hints > 0, "no real air with hints compared");
}

#[test]
fn expressions_info_and_q_verifier_equal_old() {
    for ips in [false, true] {
        for_each_run_in(ips, 20, |_, _, _, run, what| {
            let new = emitted(run);
            let old = &run.old.pil_code;
            let json = |v: &dyn erased::Ser| v.json();
            assert_eq!(json(&new.expressions_info), json(&old.expressions_info), "{what} {ips}: expressionsinfo");
            assert_eq!(json(&new.verifier_info), json(&old.verifier_info), "{what} {ips}: qVerifier");
            assert_eq!(new.fri_batch_size, old.fri_batch_size, "{what} {ips}: friBatchSize");
        });
    }
}

mod erased {
    pub trait Ser {
        fn json(&self) -> String;
    }
    impl<T: serde::Serialize> Ser for T {
        fn json(&self) -> String {
            serde_json::to_string(self).unwrap()
        }
    }
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
