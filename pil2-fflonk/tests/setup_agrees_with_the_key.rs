//! The shkey and the zkey must describe the same opening plan.
//!
//! The shkey is what pil-fflonk's setup reads; the zkey is what it writes. The
//! `f_i` plan -- which polynomials each combined polynomial packs, in which
//! slot order, at which opening points, to what degree -- appears in both, and
//! nothing derives one from the other at read time. So they agreeing is a
//! cross-check on the setup's output rather than a tautology.
//!
//! This is as far as the setup can be exercised in-tree: `generateZkey` needs a
//! powers-of-tau file, which is a ceremony artifact deliberately not vendored.
//! The moved `pilfflonk_setup.cpp` is byte-identical and links, but producing a
//! key from scratch needs that input supplied.

use pil2_fflonk::ZKey;

const ZKEY: &[u8] = include_bytes!("fixtures/reference/pilfflonk.zkey");
const SHKEY: &str = include_str!("fixtures/pilfflonk.shkey.json");

#[test]
fn the_shkey_and_the_zkey_describe_the_same_plan() {
    let zkey = ZKey::from_bytes(ZKEY).expect("the vendored key parses");
    let shkey: serde_json::Value = serde_json::from_str(SHKEY).unwrap();

    let planned = shkey["f"].as_array().expect("the shkey lists its combined polynomials");
    assert_eq!(planned.len(), zkey.f.len(), "the two disagree on how many f_i there are");

    for entry in planned {
        let index = entry["index"].as_u64().unwrap() as u32;
        let f = zkey.f.iter().find(|f| f.index == index).unwrap_or_else(|| panic!("the zkey has no f{index}"));

        // Slot order is load-bearing: it decides which coefficient of the
        // combined polynomial each one contributes to.
        let pols: Vec<&str> = entry["pols"].as_array().unwrap().iter().map(|p| p.as_str().unwrap()).collect();
        assert_eq!(pols, f.pols, "f{index}: slot order differs");

        let points: Vec<u32> =
            entry["openingPoints"].as_array().unwrap().iter().map(|p| p.as_u64().unwrap() as u32).collect();
        assert_eq!(points, f.opening_points, "f{index}: opening points differ");

        assert_eq!(entry["degree"].as_u64().unwrap(), f.degree, "f{index}: degree differs");

        // And the per-stage degrees each slot was planned with.
        for stage in entry["stages"].as_array().unwrap() {
            let s = stage["stage"].as_u64().unwrap() as u32;
            let recorded = f.stages.iter().find(|x| x.stage == s).unwrap_or_else(|| panic!("f{index}: no stage {s}"));

            for pol in stage["pols"].as_array().unwrap() {
                let name = pol["name"].as_str().unwrap();
                let degree = pol["degree"].as_u64().unwrap();
                let got = recorded
                    .pols
                    .iter()
                    .find(|p| p.name == name)
                    .unwrap_or_else(|| panic!("f{index} stage {s}: {name} missing"));
                assert_eq!(degree, got.degree, "f{index} stage {s}: {name} degree differs");
            }
        }
    }
}

/// The omegas the plan needs are the ones the key carries.
///
/// Each `f_i` opens at a coset of the `nPols`-th roots of unity, so the key
/// must provide `w{nPols}` and, for a non-zero opening point, the coset's
/// starting root. A missing one is not recoverable by the verifier, which
/// cannot take roots itself.
#[test]
fn the_key_carries_every_omega_the_plan_needs() {
    let zkey = ZKey::from_bytes(ZKEY).expect("the vendored key parses");
    let present: Vec<&str> = zkey.omegas.iter().map(|(name, _)| name.as_str()).collect();

    for f in &zkey.f {
        let n = f.pols.len();
        assert!(present.contains(&format!("w{n}").as_str()), "f{}: the key has no w{n}", f.index);

        for &point in &f.opening_points {
            if point == 0 {
                continue; // that coset starts at 1, so it needs no omega
            }
            let name = format!("w{n}_{point}d{n}");
            assert!(present.contains(&name.as_str()), "f{}: the key has no {name}", f.index);
        }
    }
}
