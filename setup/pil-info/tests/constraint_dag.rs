//! pilfflonk's `qVerifier` (Opening::Shplonk, BN128) is the lowered constraint DAG: structurally equal to
//! `expression_code(lower_to_code(dag, ctx, Fold::Shplonk{..}))`, and equal in value to the pre-DAG generator's.
//!
//! The pre-DAG reference is frozen in `fixtures/constraint_dag/shplonk_q_verifier_before.json` (write it
//! from the old generator with `PIL_INFO_UPDATE_FIXTURES=1`; TODO(piece 3 G2: after circom E): delete it with the old generator).

use std::path::PathBuf;

use num_bigint::BigUint;
use pil2_pilout::pilout as pb;
use pil_info::core::dag::format::DagBoundary;
use pil_info::core::dag::lower_code::{lower_to_code, Fold, LowerCtx};
use pil_info::emit::verifier::expression_code;
use pil_info::pil::gen_code::lower_ctx_from_pil;
use pil_info::pil::info::{constraint_dag, dag_boundaries};
use pil_info::{PilInfoCfg, PilInfoResult};

mod common;
use common::*;

/// a*b + a' (everyRow), L1*(a - 1) (firstRow), b - a (lastRow), a' - b (everyFrame 0..1): four boundaries.
fn mixed_boundaries_pilout() -> pb::PilOut {
    pilout(
        &["a", "b"],
        &["L1"],
        vec![
            sub(witness(0, 0), constant("1")), // 0
            mul(fixed(0), exp(0)),             // 1
            mul(witness(0, 0), witness(1, 0)), // 2
            add(exp(2), witness(0, 1)),        // 3
            sub(witness(1, 0), witness(0, 0)), // 4
            sub(witness(0, 1), witness(1, 0)), // 5
        ],
        vec![first_row(1), every_row(3), last_row(4), every_frame(5, 0, 1)],
    )
}

fn cfg() -> PilInfoCfg {
    PilInfoCfg::bn128()
}

fn run(p: &pb::PilOut) -> PilInfoResult {
    pil_info::run(p, 0, 0, &cfg(), &Default::default()).unwrap()
}

fn boundaries(r: &PilInfoResult) -> Vec<DagBoundary> {
    dag_boundaries(&r.boundaries).unwrap()
}

/// The DAG of the same air from the core, with the passes' im choice (`r`'s symbols).
fn dag_of(p: &pb::PilOut, r: &PilInfoResult) -> pil_info::core::dag::format::ConstraintDag {
    constraint_dag(p, 0, 0, &cfg(), &r.setup.symbols).unwrap()
}

fn ctx_of(r: &PilInfoResult) -> LowerCtx {
    lower_ctx_from_pil(&r.setup.symbols, &r.pil_code.ev_map, &cfg().field).unwrap()
}

#[test]
fn shplonk_q_verifier_is_lowered_dag() {
    let p = mixed_boundaries_pilout();
    let r = run(&p);
    let b = boundaries(&r);
    assert_eq!(b.len(), 4);
    let want = expression_code(&lower_to_code(&dag_of(&p, &r), &ctx_of(&r), Fold::Shplonk { boundaries: &b }).unwrap());
    let got = &r.pil_code.verifier_info.q_verifier;
    assert_eq!(got.tmp_used, want.tmp_used);
    assert_eq!(serde_json::to_value(&got.code).unwrap(), serde_json::to_value(&want.code).unwrap());
}

// ---- value test: against the frozen pre-DAG qVerifier ------------------------------------------

fn fixture() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/constraint_dag/shplonk_q_verifier_before.json")
}

struct Point {
    evals: Vec<BigUint>,
    challenges: Vec<BigUint>,
    publics: Vec<BigUint>,
    zi: Vec<BigUint>,
}

fn point(seed: u64, p: &BigUint) -> Point {
    let mut rng = Rng(seed);
    let mut fr = |n: usize| -> Vec<BigUint> {
        (0..n)
            .map(|_| {
                let w = [rng.next(), rng.next(), rng.next(), rng.next()];
                w.iter().fold(BigUint::from(0u8), |a, x| (a << 64) + BigUint::from(*x)) % p
            })
            .collect()
    };
    Point { evals: fr(64), challenges: fr(8), publics: fr(8), zi: fr(8) }
}

/// Value of the last instruction over Fr, from the JSON code (the fixture's refs omit some fields).
fn eval(code: &serde_json::Value, pt: &Point, p: &BigUint) -> BigUint {
    let idx = |r: &serde_json::Value, k: &str| r[k].as_u64().unwrap_or(0) as usize;
    let mut tmp: Vec<BigUint> = Vec::new();
    let mut last = BigUint::from(0u8);
    for c in code.as_array().unwrap() {
        let v: Vec<BigUint> = c["src"]
            .as_array()
            .unwrap()
            .iter()
            .map(|r| match r["type"].as_str().unwrap() {
                "tmp" => tmp[idx(r, "id")].clone(),
                "eval" => pt.evals[idx(r, "id")].clone(),
                "challenge" => pt.challenges[idx(r, "id")].clone(),
                "public" => pt.publics[idx(r, "id")].clone(),
                "Zi" => pt.zi[idx(r, "boundaryId")].clone(),
                "number" => r["value"].as_str().unwrap().parse::<BigUint>().unwrap() % p,
                t => panic!("operand {t}"),
            })
            .collect();
        last = match c["op"].as_str().unwrap() {
            "add" => (&v[0] + &v[1]) % p,
            "sub" => (&v[0] + p - &v[1]) % p,
            "mul" => (&v[0] * &v[1]) % p,
            "copy" => v[0].clone(),
            o => panic!("op {o}"),
        };
        assert_eq!(c["dest"]["type"], "tmp");
        let d = idx(&c["dest"], "id");
        if tmp.len() <= d {
            tmp.resize(d + 1, BigUint::from(0u8));
        }
        tmp[d] = last.clone();
    }
    last
}

#[test]
fn shplonk_q_verifier_value_matches_the_pre_dag_generator() {
    let r = run(&mixed_boundaries_pilout());
    if std::env::var("PIL_INFO_UPDATE_FIXTURES").is_ok() {
        std::fs::write(fixture(), serde_json::to_string_pretty(&r.pil_code.verifier_info.q_verifier.code).unwrap())
            .unwrap();
    }
    let old: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(fixture()).expect("fixture")).unwrap();
    let new = serde_json::to_value(&r.pil_code.verifier_info.q_verifier.code).unwrap();
    let p = cfg().field.modulus().clone();
    for seed in 0..8 {
        let pt = point(seed, &p);
        assert_eq!(eval(&new, &pt, &p), eval(&old, &pt, &p), "seed {seed}");
    }
}

// Dimension corruption: a context whose extension degree disagrees with the (BN128, ext 1) DAG.
#[test]
fn shplonk_refuses_dimension_mismatch() {
    let p = mixed_boundaries_pilout();
    let r = run(&p);
    let b = boundaries(&r);
    let mut t = ctx_of(&r).tables().clone();
    t.ext_dim = 3;
    let ctx = LowerCtx::new(t).unwrap();
    assert!(lower_to_code(&dag_of(&p, &r), &ctx, Fold::Shplonk { boundaries: &b }).is_err());
}

// A root whose boundary is not in the list has no Zi.
#[test]
fn shplonk_refuses_an_unlisted_boundary() {
    let p = mixed_boundaries_pilout();
    let r = run(&p);
    let only_every_row = [DagBoundary::EveryRow];
    assert!(lower_to_code(&dag_of(&p, &r), &ctx_of(&r), Fold::Shplonk { boundaries: &only_every_row }).is_err());
}
