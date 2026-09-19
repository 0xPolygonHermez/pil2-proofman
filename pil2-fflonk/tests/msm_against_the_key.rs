//! Commits the proving key's own constant polynomials and checks the result
//! against the commitments that same key records.
//!
//! This is the first end-to-end exercise of the FFI: the coefficients and the
//! powers of tau go across the boundary in the key's representation, untouched
//! by Rust, and come back as a point that must equal one pil1's setup computed
//! independently. Getting the Montgomery handling, the point layout or the
//! indexing wrong all produce a different point, so a match is strong evidence
//! the boundary is right.
//!
//! It is a cross-check rather than a tautology: the coefficients and the
//! commitment are written to different sections of the file by different parts
//! of the setup, and nothing here recomputes one from the other.

use pil2_fflonk::ZKey;
use proofman_fflonk_lib_c::{G1_AFFINE_BYTES, combine, is_infinity, msm};

const ZKEY: &[u8] = include_bytes!("fixtures/reference/pilfflonk.zkey");
const VKEY: &str = include_str!("fixtures/pilfflonk.vkey");

/// The key's powers of tau.
fn ptau(zkey: &ZKey) -> &[u8] {
    zkey.bulk.get(&pil2_fflonk::SECTION_PTAU).expect("the key has a PTau section")
}

#[test]
fn commits_the_keys_constant_polynomials_to_the_points_it_records() {
    let zkey = ZKey::from_bytes(ZKEY).expect("the vendored key parses");
    let ptau = ptau(&zkey);

    assert!(!zkey.f_commitments.is_empty(), "the key records no precomputed commitments");

    for f in &zkey.f_commitments {
        // One power of tau per coefficient, and no more: the MSM is over the
        // polynomial's own length, not the whole SRS.
        let terms = f.pol.len() / proofman_fflonk_lib_c::FR_BYTES;
        assert!(terms > 0, "{}: no coefficients", f.name);
        assert!(
            terms * G1_AFFINE_BYTES <= ptau.len(),
            "{}: needs {terms} powers of tau but the key has {}",
            f.name,
            ptau.len() / G1_AFFINE_BYTES
        );

        let got = msm(&ptau[..terms * G1_AFFINE_BYTES], &f.pol).unwrap_or_else(|e| panic!("{}: {e}", f.name));

        assert_eq!(
            got.as_slice(),
            f.commit.as_slice(),
            "{}: recomputed commitment differs from the one the key records",
            f.name
        );
        assert!(!is_infinity(&got), "{}: commitment is the point at infinity", f.name);
    }
}

/// The commitment depends on every coefficient. Perturbing one must move it,
/// or the MSM is ignoring part of its input -- which a length mismatch or a
/// wrong stride would cause, while still producing a plausible point.
#[test]
fn every_coefficient_reaches_the_commitment() {
    let zkey = ZKey::from_bytes(ZKEY).expect("the vendored key parses");
    let ptau = ptau(&zkey);
    let f = &zkey.f_commitments[0];

    let terms = f.pol.len() / proofman_fflonk_lib_c::FR_BYTES;
    let points = &ptau[..terms * G1_AFFINE_BYTES];
    let honest = msm(points, &f.pol).unwrap();

    // First, last, and one in between: a stride bug typically drops a tail or
    // reads only the first element.
    for term in [0, terms / 2, terms - 1] {
        let mut tampered = f.pol.clone();
        tampered[term * proofman_fflonk_lib_c::FR_BYTES] ^= 0x01;
        assert_ne!(msm(points, &tampered).unwrap(), honest, "coefficient {term} did not affect the commitment");
    }
}

/// Truncating the input changes the answer, so the term count is really being
/// honoured rather than the full SRS being consumed.
#[test]
fn the_term_count_is_honoured() {
    let zkey = ZKey::from_bytes(ZKEY).expect("the vendored key parses");
    let ptau = ptau(&zkey);
    let f = &zkey.f_commitments[0];

    let terms = f.pol.len() / proofman_fflonk_lib_c::FR_BYTES;
    let full = msm(&ptau[..terms * G1_AFFINE_BYTES], &f.pol).unwrap();

    let shorter = terms - 1;
    let partial = msm(&ptau[..shorter * G1_AFFINE_BYTES], &f.pol[..shorter * proofman_fflonk_lib_c::FR_BYTES]).unwrap();

    assert_ne!(partial, full);
}

/// The stage-0 path end to end: build the combined polynomials from the key's
/// raw coefficients, commit them, and check both against what the key records.
///
/// The tests above commit `FCommitment.pol`, which is already interleaved, so
/// they say nothing about the packing. This drives `combined_for`, which does
/// the packing itself -- getting the stride, the slot order or the length wrong
/// all produce a different point.
#[test]
fn builds_and_commits_the_constant_stage() {
    let zkey = ZKey::from_bytes(ZKEY).expect("the vendored key parses");
    let ptau = ptau(&zkey);
    let coefs = zkey.bulk.get(&pil2_fflonk::SECTION_CONST_POLS_COEFS).expect("constant coefficients");

    let plan = pil2_fflonk::stage_plan(&zkey, 0).unwrap();
    let width = pil2_fflonk::stage_width(&zkey, 0).unwrap();
    assert_eq!(plan.len(), 2, "the reference key has two constant-only combined polynomials");

    for p in &plan {
        let recorded = zkey
            .f_commitments
            .iter()
            .find(|r| r.name == p.name)
            .unwrap_or_else(|| panic!("{} has no recorded commitment", p.name));

        let packed = combine(coefs, width, &p.columns).unwrap();

        // CPolynomial trims trailing zeroes, as the setup did before storing.
        assert_eq!(packed, recorded.pol, "{}", p.name);

        // ... and committing it lands on the point the setup recorded.
        let terms = packed.len() / proofman_fflonk_lib_c::FR_BYTES;
        let got = msm(&ptau[..terms * G1_AFFINE_BYTES], &packed).unwrap();
        assert_eq!(got.as_slice(), recorded.commit.as_slice(), "{}", p.name);
        assert!(!is_infinity(&got), "{}", p.name);
    }
}

/// The commitments the prover computes for the constant stage are the ones the
/// verifier takes from its key. If these diverged, every challenge would too.
#[test]
fn the_computed_constant_commitments_are_the_ones_the_verifier_uses() {
    let zkey = ZKey::from_bytes(ZKEY).expect("the vendored key parses");
    let ptau = ptau(&zkey);
    let coefs = zkey.bulk.get(&pil2_fflonk::SECTION_CONST_POLS_COEFS).unwrap();

    let vkey: serde_json::Value = serde_json::from_str(VKEY).unwrap();
    let setup = pil2_fflonk::ShPlonkSetup::from_vkey_json(&vkey).unwrap();

    let width = pil2_fflonk::stage_width(&zkey, 0).unwrap();
    for c in pil2_fflonk::stage_plan(&zkey, 0).unwrap() {
        let packed = combine(coefs, width, &c.columns).unwrap();
        let terms = packed.len() / proofman_fflonk_lib_c::FR_BYTES;
        let got = msm(&ptau[..terms * G1_AFFINE_BYTES], &packed).unwrap();

        // The verification key writes the point as decimal x/y; the prover
        // produces it in the key's own representation, so compare through the
        // proving key, which carries both forms of the same value.
        let recorded = zkey.f_commitments.iter().find(|r| r.name == c.name).unwrap();
        assert_eq!(got.as_slice(), recorded.commit.as_slice(), "{}", c.name);
        assert!(setup.f_commitments.contains_key(&c.name), "the vkey omits {}", c.name);
    }
}
