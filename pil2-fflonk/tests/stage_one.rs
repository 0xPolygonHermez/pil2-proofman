//! The stage-1 path end to end: trace, interpolate, pack, commit.
//!
//! This is the first stage that needs a witness rather than the key alone, and
//! it exercises everything a committed stage does:
//!
//! ```text
//! trace evaluations ──iNTT──► coefficients ──pack──► f_i ──MSM──► commitment
//! ```
//!
//! The commitments are checked against a real proof. That is only possible
//! because the proof is the deterministic one: blinding adds a fresh multiple
//! of the vanishing polynomial to every committed polynomial each run, so a
//! blinded proof's stage commitments cannot be recomputed from the trace at
//! all. See `fixtures/reference/deterministic.patch`.

use pil2_fflonk::ZKey;
use proofman_fflonk_lib_c::{G1_AFFINE_BYTES, intt, msm};

const ZKEY: &[u8] = include_bytes!("fixtures/reference/pilfflonk.zkey");
const TRACE: &[u8] = include_bytes!("fixtures/reference/pilfflonk.commit");
const PROOF: &str = include_str!("fixtures/pilfflonk.deterministic.proof.json");

/// The reference AIR: 2^8 rows, and stage 1 commits 15 columns.
const N_BITS: u32 = 8;
const STAGE: u32 = 1;

fn commitments_from_the_proof() -> serde_json::Value {
    serde_json::from_str::<serde_json::Value>(PROOF).unwrap()["polynomials"].clone()
}

#[test]
fn the_trace_has_the_shape_the_key_describes() {
    let zkey = ZKey::from_bytes(ZKEY).unwrap();
    let names = zkey.pols_names_stage.get(&STAGE).expect("stage 1 is named in the key");

    let rows = 1usize << N_BITS;
    assert_eq!(
        TRACE.len(),
        rows * names.len() * proofman_fflonk_lib_c::FR_BYTES,
        "the trace is {} rows of {} columns",
        rows,
        names.len()
    );
}

/// Interpolating the trace and committing each stage-1 combined polynomial must
/// reproduce the proof's commitments.
#[test]
fn commits_stage_one_to_the_points_the_proof_carries() {
    let zkey = ZKey::from_bytes(ZKEY).unwrap();
    let ptau = zkey.bulk.get(&pil2_fflonk::SECTION_PTAU).unwrap();
    let names = zkey.pols_names_stage.get(&STAGE).unwrap();

    let rows = 1usize << N_BITS;
    let interpolated = intt(TRACE, rows, names.len()).expect("the trace interpolates");

    // The stage reserves more coefficients than the trace has rows -- room for
    // blinding, which writes just above the domain. Those rows are zero here
    // because this proof was produced with blinding disabled.
    let reserved = zkey
        .f
        .iter()
        .flat_map(|f| f.stages.iter().filter(|s| s.stage == STAGE))
        .flat_map(|s| s.pols.iter().map(|p| p.degree as usize))
        .max()
        .expect("stage 1 declares degrees");
    assert!(reserved > rows, "blinding needs rows above the domain");

    let coefficients = pil2_fflonk::pad_rows(&interpolated, names.len(), reserved).unwrap();

    let built = pil2_fflonk::combined_for(&zkey, STAGE, &coefficients).expect("stage 1 packs");
    assert!(!built.is_empty(), "stage 1 commits nothing");

    let proof = commitments_from_the_proof();
    for c in &built {
        let terms = c.coefficients.len() / proofman_fflonk_lib_c::FR_BYTES;
        assert!(terms * G1_AFFINE_BYTES <= ptau.len(), "{}: not enough powers of tau", c.name);

        let got = msm(&ptau[..terms * G1_AFFINE_BYTES], &c.coefficients).unwrap();
        let expected = &proof[&c.name];
        assert!(!expected.is_null(), "the proof carries no {}", c.name);

        // Compare in canonical form: the proof records decimal coordinates,
        // the library works in the key's representation, and to_bytes_be is
        // the bridge between them.
        let canonical = proofman_fflonk_lib_c::to_bytes_be(&got).unwrap();
        assert_eq!(decimals(&canonical), expected_decimals(expected), "{}: commitment differs from the proof", c.name);
    }
}

/// The canonical coordinates as the decimal strings a proof records.
fn decimals(canonical: &[u8]) -> (String, String) {
    let half = canonical.len() / 2;
    let x = num_bigint::BigUint::from_bytes_be(&canonical[..half]);
    let y = num_bigint::BigUint::from_bytes_be(&canonical[half..]);
    (x.to_str_radix(10), y.to_str_radix(10))
}

fn expected_decimals(point: &serde_json::Value) -> (String, String) {
    let c = point.as_array().expect("a proof point is [x, y, z]");
    assert_eq!(c[2].as_str(), Some("1"), "the proof's point is not affine");
    (c[0].as_str().unwrap().to_string(), c[1].as_str().unwrap().to_string())
}
