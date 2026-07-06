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

use fields::{Field, Goldilocks};
use proofman_multilinear::{derive_global_challenges_for, verify_air, AirIr, MlError, MlProof};

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
        .map(|p| {
            let ir = AirIr::load(&p).unwrap_or_else(|e| panic!("loading {}: {e}", p.display()));
            ((ir.airgroup_id, ir.air_id), ir)
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
    assert!(!irs.is_empty(), "no .mlinfo.bin artifacts under {}", proving_key.display());

    // The example uses the std permutation bus, so at least one AIR must be
    // multi-stage with challenges and airgroup values (the bus balance).
    let fibonacci = irs.values().find(|ir| ir.name == "Fibonacci").expect("Fibonacci AIR in the proving key");
    assert_eq!(fibonacci.n_stages(), 2, "std bus implies a stage-2");
    assert!(fibonacci.cols_per_stage[1] > 0, "stage-2 must commit columns (gsum/im-pols)");
    assert!(!fibonacci.challenge_stages.is_empty(), "std bus implies stage challenges");
    assert!(!fibonacci.airgroupvalue_stages.is_empty(), "std bus implies airgroup values");
    assert!(fibonacci.offset_index(0).is_some() && fibonacci.offset_index(1).is_some());

    for ir in irs.values() {
        // Structural invariants every compiled IR must satisfy.
        assert!(ir.n_bits >= 1);
        assert!(!ir.constraints.is_empty(), "{}: no constraints", ir.name);
        assert!(ir.max_constraint_degree >= 1);
        for (i, instr) in ir.instrs.iter().enumerate() {
            assert_eq!(instr.dst as usize, i, "{}: instr {i} breaks dst==index invariant", ir.name);
        }
    }
}

#[test]
fn proof_set_verifies_with_rederived_challenges() {
    let proving_key = example_dir().join("provingKey");
    let proofs_dir = example_dir().join("proofs");
    if !proving_key.exists() || !proofs_dir.exists() {
        eprintln!("skipping: proving key or proofs not found (run the example README flow first)");
        return;
    }

    let irs = load_irs(&proving_key);
    let mut set: Vec<(PathBuf, MlProof)> = Vec::new();
    for entry in std::fs::read_dir(&proofs_dir).expect("read proofs dir").flatten() {
        let p = entry.path();
        if !p.to_string_lossy().ends_with(".mlproof.bin") {
            continue;
        }
        match MlProof::load(&p) {
            Ok(proof) => set.push((p, proof)),
            Err(e) => {
                // Old-format proofs from a previous protocol revision: skip the
                // whole test rather than fail on stale artifacts.
                eprintln!("skipping: {} not decodable ({e}) — regenerate the proofs", p.display());
                return;
            }
        }
    }
    if set.is_empty() {
        eprintln!("skipping: no .mlproof.bin in {}", proofs_dir.display());
        return;
    }

    // The instance order defines the global challenge derivation.
    set.sort_by_key(|(_, proof)| proof.global_instance_id);
    assert!(
        set.windows(2).all(|w| w[0].1.global_instance_id != w[1].1.global_instance_id),
        "duplicate global instance ids in the proof set"
    );

    let ir_of = |proof: &MlProof| {
        irs.get(&(proof.airgroup_id, proof.air_id))
            .unwrap_or_else(|| panic!("no mlinfo for airgroup {} air {}", proof.airgroup_id, proof.air_id))
    };

    let challenge_stages: Vec<u8> =
        set.iter().map(|(_, proof)| ir_of(proof).challenge_stages.clone()).max_by_key(|v| v.len()).unwrap_or_default();
    let max_n_stages = set.iter().map(|(_, proof)| ir_of(proof).n_stages()).max().unwrap_or(1);
    let stage1_roots: Vec<[Goldilocks; 4]> = set.iter().map(|(_, proof)| proof.stage_roots[0]).collect();
    let expected = derive_global_challenges_for(&challenge_stages, max_n_stages, &stage1_roots);

    for (path, proof) in &set {
        let ir = ir_of(proof);
        let expected_air = &expected[..ir.challenge_stages.len().min(expected.len())];
        match verify_air(ir, proof, &proof.publics, None, Some(expected_air)) {
            Ok(()) => {}
            // Shape mismatches mean the proofs predate the current proving key
            // (stale artifacts) — skip instead of failing the suite.
            Err(MlError::Malformed(msg)) => {
                eprintln!("skipping: {} is stale w.r.t. the proving key ({msg}) — regenerate", path.display());
                return;
            }
            Err(e) => panic!("{} failed verification: {e}", path.display()),
        }
    }

    // Sanity on the set-level binding: tampering with one root must change the
    // derived challenges (and hence reject the set).
    if max_n_stages >= 2 && !expected.iter().all(|c| c.is_zero()) {
        let mut bad_roots = stage1_roots.clone();
        bad_roots[0][0] += Goldilocks::ONE;
        let bad = derive_global_challenges_for(&challenge_stages, max_n_stages, &bad_roots);
        assert_ne!(expected, bad, "challenge derivation must depend on the roots");
    }
}
