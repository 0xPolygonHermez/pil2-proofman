//! `setup-pilfflonk --solidity` and `proofman-setup pilfflonk-solidity`
//! (pilfflonk/docs/verifier.md#solidity-verifier) on `common`'s pilout: the verifier goes next to
//! the vkey and no other file changes, what the generator refuses, and the calldata it reads.
//! Foundry runs verifiers on real proofs in `tests/solidity.rs`, with the pinned tools.

use std::fs;
use std::path::{Path, PathBuf};

use pilfflonk_setup::command::{DEFAULT_EXTRA_MULS, DEFAULT_MAX_CONSTRAINT_DEGREE, DEFAULT_MAX_Q_DEGREE, PROVING_KEY_DIR};
use pilfflonk_setup::digest::seal_vkey;
use pilfflonk_setup::solidity::{export_verifier_sol, verifier_sol, VERIFIER_SOL_FILE};
use pilfflonk_setup::test_ptau::{fixed_tau_ptau, test_tau, write_tau_one_ptau};
use pilfflonk_setup::{run_setup_pilfflonk, SetupError, SetupPilfflonkOptions};
use prost::Message;
use proofman_pilfflonk::{Boundary, CalldataLayout, FqBytes, G1Affine, G2Affine, JsonFile, PilfflonkGlobalInfo, Vkey};

use crate::common::*;

/// Every file under `dir`, relative to it, sorted, with its bytes.
fn contents(dir: &Path) -> Vec<(String, Vec<u8>)> {
    fn walk(root: &Path, dir: &Path, out: &mut Vec<(String, Vec<u8>)>) {
        for entry in fs::read_dir(dir).unwrap() {
            let path = entry.unwrap().path();
            if path.is_dir() {
                walk(root, &path, out);
            } else {
                let name = path.strip_prefix(root).unwrap().to_string_lossy().into_owned();
                out.push((name, fs::read(&path).unwrap()));
            }
        }
    }
    let mut out = Vec::new();
    walk(dir, dir, &mut out);
    out.sort();
    out
}

/// Sets up `common`'s pilout in `dir/<build>` with `--no-packing` and a ptau with `τ = 1`, with
/// `--solidity` or not, and returns its `provingKey/`.
fn setup(dir: &TestDir, build: &str, solidity: bool) -> PathBuf {
    let opts = SetupPilfflonkOptions {
        airout_path: dir.file("synthetic.pilout"),
        build_dir: dir.file(build),
        powers_of_tau: dir.file("tau_one.ptau"),
        max_constraint_degree: DEFAULT_MAX_CONSTRAINT_DEGREE,
        extra_muls: DEFAULT_EXTRA_MULS,
        max_q_degree: DEFAULT_MAX_Q_DEGREE,
        no_packing: true,
        solidity,
    };
    if !opts.airout_path.exists() {
        fs::write(&opts.airout_path, pilout().encode_to_vec()).unwrap();
        write_tau_one_ptau(&opts.powers_of_tau, 64).unwrap();
    }
    run_setup_pilfflonk(&opts).unwrap();
    opts.build_dir.join(PROVING_KEY_DIR)
}

fn vkey_of(proving_key: &Path) -> (PathBuf, Vkey) {
    let path = PilfflonkGlobalInfo::from_proving_key(proving_key).unwrap().vkey_path(proving_key);
    let vkey = Vkey::read(&path).unwrap();
    (path, vkey)
}

#[test]
fn solidity_writes_the_verifier_next_to_the_vkey_and_changes_nothing_else() {
    let _cpp = cpp_core();
    let dir = TestDir::new("solidity");
    let without = contents(&setup(&dir, "without", false));
    let proving_key = setup(&dir, "with", true);
    let with = contents(&proving_key);

    // The same files, byte for byte, and the verifier in the backend directory, next to the vkey.
    let sol_name = format!("Synthetic/pilfflonk/{VERIFIER_SOL_FILE}");
    let (sol, rest): (Vec<_>, Vec<_>) = with.into_iter().partition(|(name, _)| *name == sol_name);
    assert_eq!(rest, without);
    assert_eq!(sol.len(), 1, "no {sol_name}");
    let sol = String::from_utf8(sol[0].1.clone()).unwrap();

    // It is the vkey's, the same from the vkey alone, and the same every time.
    let (vkey_path, vkey) = vkey_of(&proving_key);
    assert_eq!(verifier_sol(&vkey).unwrap(), sol);
    let exported = dir.file("exported.sol");
    export_verifier_sol(&vkey_path, &exported).unwrap();
    assert_eq!(fs::read_to_string(&exported).unwrap(), sol);

    // The contract of snarkjs's interface, of the vkey's sizes: 3 commitments (a, b and Q), W and W',
    // the evaluations of the evMap, inv and invZh; no auxiliary inverse (everyRow only); 2 publics.
    let layout = CalldataLayout::of(&vkey);
    assert_eq!(
        layout,
        CalldataLayout { n_commitments: 3, n_evaluations: vkey.ev_map.len() as u64, n_q_pieces: 0, aux_rows: vec![] }
    );
    assert_eq!(layout.words(), 2 * (3 + 2) + vkey.ev_map.len() as u64 + 2);
    assert!(
        sol.starts_with("// SPDX-License-Identifier: MIT OR Apache-2.0\npragma solidity >=0.7.0 <0.9.0;\n"),
        "{sol}"
    );
    assert!(sol.contains("contract PilfflonkVerifier {"), "{sol}");
    let signature = format!(
        "function verifyProof(bytes32[{}] calldata proof, uint256[2] calldata pubSignals) public view returns (bool)",
        layout.words()
    );
    assert!(sol.contains(&signature), "{sol}");
    assert!(sol.contains(&format!("digest {}", vkey.digest.to_hex())), "{sol}");
    assert!(sol.contains(&format!("uint256 constant DIGEST = {};", vkey.digest.to_fr().to_decimal())), "{sol}");
    // The fixed commitments, the vkey's: with τ = 1, G, 2G and 3G.
    for (i, g) in [multiple_of_g(1), multiple_of_g(2), multiple_of_g(3)].iter().enumerate() {
        assert!(sol.contains(&format!("uint256 constant f{i}x = {};", g.x.to_decimal())), "f{i}: {sol}");
        assert!(sol.contains(&format!("uint256 constant f{i}y = {};", g.y.to_decimal())), "f{i}: {sol}");
    }
    // [τ]₂ = [1]₂ with τ = 1.
    assert!(sol.contains(&format!("uint256 constant X2x1 = {};", G2_GENERATOR[0])), "{sol}");
    assert!(sol.contains(&format!("uint256 constant X2y2 = {};", G2_GENERATOR[3])), "{sol}");
}

#[test]
fn the_verifier_of_a_vkey_the_js_verifier_accepts_nothing_of_is_refused() {
    let _cpp = cpp_core();
    let dir = TestDir::new("solidity_refused");
    let (vkey_path, vkey) = vkey_of(&setup(&dir, "build", false));
    let refused = |vkey: &Vkey| verifier_sol(vkey).unwrap_err().to_string();

    // A digest that is not the digest of the vkey (pilfflonk/docs/formats.md#digest): the JS
    // verifier rejects every proof.
    let mut wrong_digest = vkey.clone();
    wrong_digest.digest.0[0] ^= 1;
    let err = refused(&wrong_digest);
    assert!(err.contains("digest is not the digest of its contents"), "{err}");
    // So does the command, which writes nothing.
    fs::write(&vkey_path, wrong_digest.to_json_string().unwrap()).unwrap();
    let out = dir.file("refused.sol");
    let err = export_verifier_sol(&vkey_path, &out).unwrap_err();
    assert!(matches!(err, SetupError::Solidity(_)), "{err}");
    assert!(!out.exists());

    // A fixed commitment off the curve, with its digest right (vkey.js, g1FromObject).
    let mut off_curve = vkey.clone();
    off_curve.fixed_commitments.0[1] = G1Affine { x: FqBytes::from_u64(1), y: FqBytes::from_u64(3) };
    let err = refused(&seal_vkey(off_curve).unwrap());
    assert!(err.contains("fixed commitment f1 is not a point of G1"), "{err}");
    // The point at infinity is one: a fixed column that vanishes at τ commits to it (vkey.js).
    let mut infinity = vkey.clone();
    infinity.fixed_commitments.0[1] = G1Affine::INFINITY;
    verifier_sol(&seal_vkey(infinity).unwrap()).unwrap();

    // A vkey the verifier would not read (Vkey::validate).
    let mut invalid = vkey.clone();
    invalid.power_w = 2;
    let err = refused(&seal_vkey(invalid).unwrap());
    assert!(err.contains("powerW"), "{err}");
}

/// The calldata's auxiliary inverses (pilfflonk/docs/formats.md#calldata): `1/(ξ − ω^j)`, one per
/// `firstRow` (`j = 0`) or `lastRow` (`j = N − 1`) boundary, in the order of the boundaries, after
/// the proof's words; none for `everyRow` and `everyFrame`.
#[test]
fn the_calldata_has_an_inverse_per_first_row_and_last_row_boundary() {
    let _cpp = cpp_core();
    let dir = TestDir::new("solidity_calldata");
    let (_, vkey) = vkey_of(&setup(&dir, "build", false));
    let mut domains = vkey.clone();
    domains.boundaries = vec![
        Boundary::EveryRow,
        Boundary::LastRow,
        Boundary::EveryFrame { offset_min: 1, offset_max: 2 },
        Boundary::FirstRow,
    ];
    let layout = CalldataLayout::of(&domains);
    assert_eq!(layout.aux_rows, [N as u64 - 1, 0]);
    assert_eq!(layout.words(), CalldataLayout::of(&vkey).words() + 2);
    assert_eq!(layout.proof_words(), CalldataLayout::of(&vkey).proof_words());
}

/// X_2 as the JS verifier reads it (elements.js, g2FromObject;
/// pilfflonk/docs/verifier.md#refused-vkeys): the point at infinity of G2, for which a forged proof
/// passes the pairing precompile, is refused by `Vkey::validate`, and so by the generator and by
/// the reader of the vkey, digest right or not.
#[test]
fn a_vkey_whose_x_2_is_not_a_point_of_g2_is_refused() {
    let _cpp = cpp_core();
    let dir = TestDir::new("solidity_x2");
    let (vkey_path, vkey) = vkey_of(&setup(&dir, "build", false));
    let mut infinity = vkey.clone();
    infinity.x_2 = G2Affine::default();
    let infinity = seal_vkey(infinity).unwrap();
    let err = verifier_sol(&infinity).unwrap_err().to_string();
    assert!(err.contains("X_2 is not a point of G2 other than the point at infinity"), "{err}");
    assert!(err.contains("the point at infinity of G2"), "{err}");
    // Nor is it written, nor read.
    assert!(infinity.to_json_string().is_err());
    let text = fs::read_to_string(&vkey_path).unwrap();
    let digest = format!("\"digest\": \"{}\"", vkey.digest.to_hex());
    assert!(text.contains(&digest));
    let mut value: serde_json::Value = serde_json::from_str(&text).unwrap();
    value["X_2"] = serde_json::json!([["0", "0"], ["0", "0"]]);
    value["digest"] = serde_json::json!(infinity.digest.to_hex());
    let err = Vkey::from_json_str(&value.to_string()).unwrap_err().to_string();
    assert!(err.contains("X_2 is not a point of G2"), "{err}");
}

/// The setup refuses a ptau whose [τ]₂ is the point at infinity (τ = 0 in G2), before it writes a
/// vkey with it (pilfflonk/docs/verifier.md#refused-vkeys).
#[test]
fn a_ptau_whose_tau_g2_is_the_point_at_infinity_is_refused() {
    let _cpp = cpp_core();
    let dir = TestDir::new("solidity_tau_g2");
    let opts = SetupPilfflonkOptions {
        airout_path: dir.file("synthetic.pilout"),
        build_dir: dir.file("build"),
        powers_of_tau: dir.file("zero_tau_g2.ptau"),
        max_constraint_degree: DEFAULT_MAX_CONSTRAINT_DEGREE,
        extra_muls: DEFAULT_EXTRA_MULS,
        max_q_degree: DEFAULT_MAX_Q_DEGREE,
        no_packing: true,
        solidity: true,
    };
    fs::write(&opts.airout_path, pilout().encode_to_vec()).unwrap();
    // [τ]₂ is section 3's second point, the file's last 128 bytes.
    let mut ptau = fixed_tau_ptau(64, &test_tau());
    let at = ptau.len() - 128;
    ptau[at..].fill(0);
    fs::write(&opts.powers_of_tau, ptau).unwrap();
    let err = format!("{:#}", run_setup_pilfflonk(&opts).unwrap_err());
    assert!(err.contains("[τ^1]₂ is the point at infinity"), "{err}");
    let proving_key = opts.build_dir.join(PROVING_KEY_DIR);
    assert!(!proving_key.join("Synthetic/pilfflonk/pilfflonk.vkey.json").exists());
}

/// Offsets as the JS verifier's checkLayout reads them (shplonk.js;
/// pilfflonk/docs/verifier.md#refused-vkeys): an offset of `N + 1` in place of 1, with its evMap
/// entries and the digest to match, is refused.
#[test]
fn a_vkey_with_an_offset_of_n_or_more_is_refused() {
    let _cpp = cpp_core();
    let dir = TestDir::new("solidity_offsets");
    let (_, vkey) = vkey_of(&setup(&dir, "build", false));
    let n = 1i64 << vkey.power;
    let mut far = vkey.clone();
    for f in &mut far.layout.0 {
        for s in &mut f.offsets {
            if *s == 1 {
                *s = n + 1;
            }
        }
    }
    for e in &mut far.ev_map {
        if e.prime == 1 {
            e.prime = n + 1;
        }
    }
    assert_ne!(far.layout, vkey.layout, "the sample opens a column at offset 1");
    let err = verifier_sol(&seal_vkey(far).unwrap()).unwrap_err().to_string();
    assert!(err.contains(&format!("offset {}, and an offset must be below N = {n}", n + 1)), "{err}");
}
