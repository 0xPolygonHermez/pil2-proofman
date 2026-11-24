//! pilfflonk's setup on the core (`shplonk_info`): qDeg with the boundary degree, refusals, and every
//! compiled fixture (`#[ignore]`, needs PIL2C_EXEC):
//!
//! ```text
//! PIL2C_EXEC=<pil2-compiler>/src/pil.js cargo test -p pil-info --test shplonk_core -- --ignored
//! ```

mod common;

use std::path::{Path, PathBuf};
use std::process::Command;

use common::*;
use pil2_pilout::pilout as pb;
use pil2_pilout::pilout_proxy::PilOutProxy;
use pil_info::{shplonk_info, DegreePolicy, PilInfoCfg, PilInfoResult};

fn cfg(max: usize) -> PilInfoCfg {
    PilInfoCfg { degree_policy: DegreePolicy::Search { max }, ..PilInfoCfg::bn128() }
}

fn new(p: &pb::PilOut, cfg: &PilInfoCfg) -> pil_info::error::Result<PilInfoResult> {
    shplonk_info(p, 0, 0, cfg, &Default::default())
}

fn n_im(r: &PilInfoResult) -> usize {
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

// ---- the old passes are gone ----

fn code(src: &str) -> String {
    src.lines().filter(|l| !l.trim_start().starts_with("//")).collect::<Vec<_>>().join("\n")
}

#[test]
fn the_core_path_does_not_reach_the_old_passes() {
    let src = code(include_str!("../src/pil/shplonk.rs"));
    for old in [
        "constraint_poly",
        "pil::map",
        "map::map",
        "prepare_pil",
        "calculate_intermediate_polynomials",
        "add_im_polynomials",
        "im_polynomials",
        "info::run",
        "pil_info(",
    ] {
        assert!(!src.contains(old), "shplonk.rs reaches the old pass `{old}`");
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

#[test]
fn deep_opening_is_refused() {
    let deep = PilInfoCfg { opening: pil_info::Opening::Deep, ..PilInfoCfg::bn128() };
    let p = cubic_pilout(every_row(1));
    assert!(new(&p, &deep).is_err());
}

// ---- compiled fixtures (PIL2C_EXEC) ----

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..").canonicalize().expect("the repository root")
}

fn compile_bn128(pil: &Path) -> Option<pb::PilOut> {
    let compiler = std::env::var("PIL2C_EXEC").expect("PIL2C_EXEC must name a pil2com that has `--field`");
    let stem = pil.file_stem()?.to_string_lossy().into_owned();
    let out = Path::new(env!("CARGO_TARGET_TMPDIR")).join(format!("shplonk_core_{stem}.pilout"));
    let status = Command::new(compiler)
        .current_dir(repo_root())
        .arg(pil)
        .args(["-I", "pil2-components/lib/std/pil", "--field", "bn128", "-o"])
        .arg(&out)
        .status()
        .expect("PIL2C_EXEC runs");
    if !status.success() {
        eprintln!("skipped {}: does not compile", pil.display());
        return None;
    }
    Some(PilOutProxy::new(out.to_str().unwrap()).expect("a pilout").pilout)
}

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
            let json = |r: &PilInfoResult| serde_json::to_string(&r.pil_code.expressions_info).unwrap();
            assert_eq!(json(&a), json(&b), "{} max {max}", pil.display());
        }
    }
    assert!(compiled >= 10, "only {compiled} fixtures compiled");
}
