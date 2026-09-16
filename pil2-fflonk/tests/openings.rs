//! Evaluating committed polynomials at the opening points.
//!
//! A proof carries, for each polynomial, its value at `xi` and -- where the AIR
//! opens the next row -- at `xi·w`. This checks those claims against the
//! polynomials themselves, for the two stages whose coefficients can be
//! reconstructed: the constants, which the key stores outright, and stage 1,
//! which comes from the trace.
//!
//! Only the deterministic proof can be checked this way. Blinding adds a fresh
//! multiple of the vanishing polynomial each run, which leaves evaluations *on*
//! the trace domain unchanged but moves them at `xi`, which is off it.

use num_bigint::BigUint;
use pil2_fflonk::ZKey;
use proofman_fflonk_lib_c::{FR_BYTES, eval, intt};

const ZKEY: &[u8] = include_bytes!("fixtures/reference/pilfflonk.zkey");
const TRACE: &[u8] = include_bytes!("fixtures/reference/pilfflonk.commit");
const PROOF: &str = include_str!("fixtures/pilfflonk.deterministic.proof.json");
const VECTORS: &str = include_str!("fixtures/pilfflonk.deterministic.json");
const VKEY: &str = include_str!("fixtures/pilfflonk.vkey");

const N_BITS: u32 = 8;

fn scalar(s: &str) -> [u8; FR_BYTES] {
    let v = BigUint::parse_bytes(s.as_bytes(), 10).expect("a decimal scalar");
    let bytes = v.to_bytes_be();
    let mut out = [0u8; FR_BYTES];
    out[FR_BYTES - bytes.len()..].copy_from_slice(&bytes);
    out
}

fn decimal(bytes: &[u8]) -> String {
    BigUint::from_bytes_be(bytes).to_str_radix(10)
}

/// The opening points: `xi`, and `xi·w` for a next-row opening.
fn opening_point(xi: &BigUint, w: &BigUint, point: u32, modulus: &BigUint) -> [u8; FR_BYTES] {
    let v = (xi * w.modpow(&BigUint::from(point), modulus)) % modulus;
    scalar(&v.to_str_radix(10))
}

/// Every polynomial of a stage, by name, with its coefficients.
fn columns(zkey: &ZKey, stage: u32, buffer: &[u8]) -> Vec<(String, Vec<u8>)> {
    let names = zkey.pols_names_stage.get(&stage).expect("the stage is named");
    let entry =
        zkey.f.iter().flat_map(|f| f.stages.iter()).find(|s| s.stage == stage).expect("the stage declares degrees");

    names
        .iter()
        .enumerate()
        .map(|(id, name)| {
            let degree = entry.pols.iter().find(|p| &p.name == name).map(|p| p.degree).unwrap_or(1 << N_BITS);
            let c = pil2_fflonk::read_column(buffer, id, names.len(), degree as usize).unwrap();
            (name.clone(), c)
        })
        .collect()
}

/// The claimed openings must be what the polynomials actually evaluate to.
#[test]
fn the_proofs_openings_match_the_polynomials() {
    let zkey = ZKey::from_bytes(ZKEY).unwrap();
    let proof: serde_json::Value = serde_json::from_str(PROOF).unwrap();
    let vectors: serde_json::Value = serde_json::from_str(VECTORS).unwrap();
    let vkey: serde_json::Value = serde_json::from_str(VKEY).unwrap();

    let modulus = BigUint::parse_bytes(pil2_fflonk::FR_MODULUS.as_bytes(), 10).unwrap();
    let xi = BigUint::parse_bytes(vectors["xi"].as_str().unwrap().as_bytes(), 10).unwrap();
    let w = BigUint::parse_bytes(vkey["w"].as_str().unwrap().as_bytes(), 10).unwrap();

    // Which opening points each polynomial is claimed at, from the key.
    let setup = pil2_fflonk::ShPlonkSetup::from_vkey_json(&vkey).unwrap();

    // Stage 0 from the key, stage 1 from the trace.
    let rows = 1usize << N_BITS;
    let names1 = zkey.pols_names_stage.get(&1).unwrap();
    let interpolated = intt(TRACE, rows, names1.len()).unwrap();
    let reserved = zkey
        .f
        .iter()
        .flat_map(|f| f.stages.iter().filter(|s| s.stage == 1))
        .flat_map(|s| s.pols.iter().map(|p| p.degree as usize))
        .max()
        .unwrap();
    let stage1 = pil2_fflonk::pad_rows(&interpolated, names1.len(), reserved).unwrap();

    let mut all = columns(&zkey, 0, zkey.bulk.get(&pil2_fflonk::SECTION_CONST_POLS_COEFS).unwrap());
    all.extend(columns(&zkey, 1, &stage1));

    let mut checked = 0;
    for (name, coeffs) in &all {
        // The points this polynomial is opened at, from whichever f_i packs it.
        let points: Vec<u32> =
            setup.f.iter().find(|f| f.pols.contains(name)).map(|f| f.opening_points.clone()).unwrap_or_default();

        for point in points {
            let key = pil2_fflonk::evaluation_key(name, point);
            let Some(claimed) = proof["evaluations"][&key].as_str() else { continue };

            let at = opening_point(&xi, &w, point, &modulus);
            let got = eval(coeffs, &at).unwrap();

            assert_eq!(decimal(&got), claimed, "{key}: the proof's opening is not the polynomial's value");
            checked += 1;
        }
    }

    assert!(checked >= 20, "expected to check many openings, checked {checked}");
    eprintln!("checked {checked} openings across stages 0 and 1");
}
