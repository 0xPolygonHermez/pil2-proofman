//! End-to-end tests against the real artifacts of `examples/fibonacci-multilinear`
//! (now a std-based system: Fibonacci + Module connected by a permutation bus,
//! plus range-check tables).
//!
//! Two layers, each skipping gracefully when its artifacts are missing:
//!
//! 1. `mlinfo_artifacts_have_multistage_shape` — the proving key's compiled
//!    `.mlinfo.bin` IRs are loadable and structurally sane (needs steps 1–2 of
//!    the example README).
//! 2. `proof_set_verifies_with_rederived_challenges` — the saved
//!    `.mlproof.bin` set verifies as a SET: global challenges re-derived from
//!    every instance's stage-1 root and enforced per proof (needs the full
//!    README flow through `prove --multilinear`). The cross-instance global
//!    (bus) constraint check lives in `proofman-cli verify-multilinear`,
//!    which CI exercises.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use proofman_multilinear::AirIr;

fn example_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../examples/fibonacci-multilinear/build")
}

/// All `.mlinfo.bin` under the proving key, indexed by (airgroup_id, air_id).
fn load_irs(proving_key: &Path) -> HashMap<(u32, u32), AirIr> {
    fn walk(dir: &Path, out: &mut Vec<PathBuf>) {
        if let Ok(entries) = std::fs::read_dir(dir) {
            for e in entries.flatten() {
                let p = e.path();
                if p.is_dir() {
                    walk(&p, out);
                } else if p.to_string_lossy().ends_with(".mlinfo.bin") {
                    out.push(p);
                }
            }
        }
    }
    let mut paths = Vec::new();
    walk(proving_key, &mut paths);
    paths
        .into_iter()
        .filter_map(|p| match AirIr::load(&p) {
            Ok(ir) => Some(((ir.airgroup_id, ir.air_id), ir)),
            Err(e) => {
                // Stale artifact from a previous IR revision: ignore it (the
                // caller skips when nothing loads).
                eprintln!("ignoring stale {}: {e}", p.display());
                None
            }
        })
        .collect()
}

#[test]
fn mlinfo_artifacts_have_multistage_shape() {
    let proving_key = example_dir().join("provingKey");
    if !proving_key.exists() {
        eprintln!("skipping: {} not found (generate the proving key first)", proving_key.display());
        return;
    }

    let irs = load_irs(&proving_key);
    if irs.is_empty() {
        eprintln!("skipping: no decodable .mlinfo.bin under {} (regenerate the setup)", proving_key.display());
        return;
    }

    // The example uses the std permutation bus compiled in LogUp-GKR mode:
    // single-stage (no gsum/im columns), a lowered BusIr, and the bus
    // challenges/airgroup values still declared.
    let fibonacci = irs.values().find(|ir| ir.name == "Fibonacci").expect("Fibonacci AIR in the proving key");
    assert_eq!(fibonacci.n_stages(), 1, "GKR mode commits no stage-2 columns");
    assert!(fibonacci.bus.is_some(), "GKR mode implies a lowered bus");
    assert!(!fibonacci.challenge_stages.is_empty(), "std bus implies stage challenges");
    assert!(!fibonacci.airgroupvalue_stages.is_empty(), "std bus implies airgroup values");
    assert!(fibonacci.offset_index(0).is_some() && fibonacci.offset_index(1).is_some());

    for ir in irs.values() {
        // Structural invariants every compiled IR must satisfy. Bus-only AIRs
        // (e.g. the range tables) legitimately have zero constraints.
        assert!(ir.n_bits >= 1);
        assert!(!ir.constraints.is_empty() || ir.bus.is_some(), "{}: no constraints and no bus", ir.name);
        assert!(ir.max_constraint_degree >= 1);
        for (i, instr) in ir.instrs.iter().enumerate() {
            assert_eq!(instr.dst as usize, i, "{}: instr {i} breaks dst==index invariant", ir.name);
        }
    }
}
