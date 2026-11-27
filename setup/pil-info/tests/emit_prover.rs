//! `to_arena`, `emit_prover`: today's arena, roles and prover code, rebuilt from the typed IR.

mod common;

use common::*;
use pil2_pilout::pilout as pb;
use pil_info::emit::prover::{emit_prover, EmitParams, ProverCode};

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
    emit_prover(&run.cs, &run.im, &c.lay, &c.ops, c.exprs, &EmitParams { air_id: 0, airgroup_id: 0 }).unwrap()
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
