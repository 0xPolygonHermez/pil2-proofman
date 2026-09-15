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
use proofman_fflonk_lib_c::{G1_AFFINE_BYTES, is_infinity, msm};

const ZKEY: &[u8] = include_bytes!("fixtures/reference/pilfflonk.zkey");

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
