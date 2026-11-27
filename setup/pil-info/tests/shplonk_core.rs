//! pilfflonk's setup on the core (`shplonk_info`): qDeg with the boundary degree, refusals, and every
//! compiled fixture (`#[ignore]`, needs PIL2C_EXEC):
//!
//! ```text
//! PIL2C_EXEC=<pil2-compiler>/src/pil.js cargo test -p pil-info --test shplonk_core -- --ignored
//! ```

mod common;

use std::path::PathBuf;

use common::*;
use pil2_pilout::pilout as pb;
use pil_info::{shplonk_info, DegreePolicy, PilInfoCfg, ShplonkResult};

fn cfg(max: usize) -> PilInfoCfg {
    PilInfoCfg { degree_policy: DegreePolicy::Search { max }, ..PilInfoCfg::bn128() }
}

fn new(p: &pb::PilOut, cfg: &PilInfoCfg) -> pil_info::error::Result<ShplonkResult> {
    shplonk_info(p, 0, 0, cfg, &Default::default())
}

fn n_im(r: &ShplonkResult) -> usize {
    r.setup.cm_pols_map.iter().filter(|p| p.im_pol).count()
}

/// `w0·w1·w2` (degree 3) under `boundary`, `w0·w1` a named expression.
fn cubic_pilout(boundary: pb::Constraint) -> pb::PilOut {
    pilout(
        &["w0", "w1", "w2"],
        &[],
        vec![mul(witness(0, 0), witness(1, 0)), mul(exp(0), witness(2, 0))],
        vec![boundary],
    )
}

type Kind = (&'static str, fn(u32) -> pb::Constraint);

fn off_every_row() -> [Kind; 3] {
    [("firstRow", first_row), ("lastRow", last_row), ("everyFrame", |i| every_frame(i, 0, 1))]
}

/// A linear root: qDeg 0 everyRow, 1 off it (pilfflonk/docs/protocol.md#degree-search, δ = 1).
#[test]
fn qdeg_adds_the_boundary_degree_to_a_linear_root() {
    let p = |b| pilout(&["a", "b"], &[], vec![sub(witness(0, 0), witness(1, 0))], vec![b]);
    assert_eq!(new(&p(every_row(0)), &cfg(4)).unwrap().q_deg, 0);
    for (name, b) in off_every_row() {
        assert_eq!(new(&p(b(0)), &cfg(4)).unwrap().q_deg, 1, "{name}");
    }
}

/// nImPols + qDeg ties go to the lowest degree (2): everyRow promotes `w0·w1` (root 2, qDeg 1); off everyRow the
/// root is `im·w2·Z` (3), so the search promotes both expressions to keep qDeg 1.
#[test]
fn qdeg_of_a_cubic_follows_the_search() {
    for max in [3, 4, 9] {
        let r = new(&cubic_pilout(every_row(1)), &cfg(max)).unwrap();
        assert_eq!((r.q_deg, n_im(&r)), (1, 1), "everyRow max {max}");
        for (name, b) in off_every_row() {
            let r = new(&cubic_pilout(b(1)), &cfg(max)).unwrap();
            assert_eq!((r.q_deg, n_im(&r)), (1, 2), "{name} max {max}");
        }
    }
}

// ---- degenerate inputs ----

#[test]
fn no_constraints_is_a_clear_error() {
    let e = new(&pilout(&["a"], &[], vec![], vec![]), &cfg(4)).err().expect("refused");
    assert!(e.to_string().contains("no constraint"), "{e}");
}

#[test]
fn first_row_only_sets_up() {
    let p = pilout(&["a", "b"], &[], vec![sub(witness(0, 0), witness(1, 0))], vec![first_row(0)]);
    let r = new(&p, &cfg(4)).unwrap();
    assert_eq!(r.boundaries.len(), 2);
    let p = pilout(&["a", "b"], &[], vec![mul(witness(0, 0), witness(1, 0))], vec![first_row(0)]);
    assert_eq!(new(&p, &cfg(2)).unwrap().q_deg, 1);
}

// ---- compiled fixtures (PIL2C_EXEC) ----

fn fixtures() -> Vec<PathBuf> {
    let mut out = Vec::new();
    let mut dirs = vec![repo_root().join("pilfflonk/tests/fixtures")];
    while let Some(d) = dirs.pop() {
        for e in std::fs::read_dir(d).unwrap().flatten() {
            let p = e.path();
            if p.is_dir() {
                dirs.push(p);
            } else if p.extension().is_some_and(|x| x == "pil") {
                out.push(p);
            }
        }
    }
    out.push(repo_root().join("setup/pil-info/tests/fixtures/wide_constants.pil"));
    out.sort();
    out
}

#[test]
#[ignore = "needs PIL2C_EXEC"]
fn every_pilfflonk_fixture_sets_up_deterministically() {
    let all = fixtures();
    assert!(all.len() >= 15, "{all:?}");
    let mut compiled = 0;
    for pil in all {
        let Some(p) = compile_bn128(&pil) else { continue };
        if p.air_groups.len() != 1 || p.air_groups[0].airs.len() != 1 {
            continue;
        }
        compiled += 1;
        for max in [3, 4, 5] {
            let (a, b) = (new(&p, &cfg(max)).unwrap(), new(&p, &cfg(max)).unwrap());
            let json = |r: &ShplonkResult| serde_json::to_string(&r.pil_code.expressions_info).unwrap();
            assert_eq!(json(&a), json(&b), "{} max {max}", pil.display());
        }
    }
    assert!(compiled >= 10, "only {compiled} fixtures compiled");
}
