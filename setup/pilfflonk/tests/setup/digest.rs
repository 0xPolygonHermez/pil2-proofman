//! The digest of the vkey (spec A.6, N10): `keccak256("pilfflonk-v1" ‖ canonical(vkey without
//! digest))`, with Keccak-256 from the C++ core.

use std::fs;

use pilfflonk_setup::digest::{keccak256, seal_vkey, vkey_digest};
use proofman_pilfflonk::tag::{Curve, Protocol};
use proofman_pilfflonk::{
    Boundary, Digest, EvMapEntry, FixedCommitments, FqBytes, G2Affine, JsonFile, Layout, LayoutEntry, LayoutPol,
    PolType, Vkey, FORMAT_VERSION,
};
use serde_json::json;

use crate::common::*;

/// The digest of [`sample_vkey`], computed by @noble/hashes' `keccak_256` over the canonical JSON
/// that a JavaScript canonicaliser (keys sorted by `sort()`, `JSON.stringify` of each value) writes
/// for `pilfflonk.vkey.json` as this test writes it: an implementation of A.6 independent of this
/// one, and of the Keccak of the C++ core.
const SAMPLE_DIGEST: &str = "0x0d0a871547df85cf3f811d77135800ef101a2538ac40e315441369ee3ae0eb42";

fn f(stage: u64, pols: &[(u64, &str)], offsets: &[i64], degree: u64) -> LayoutEntry {
    LayoutEntry {
        stage,
        pols: pols.iter().map(|&(id, name)| LayoutPol { id, name: name.into() }).collect(),
        k: pols.len() as u64,
        offsets: offsets.to_vec(),
        degree,
    }
}

fn ev(pol_type: PolType, id: u64, prime: i64, opening_pos: u64) -> EvMapEntry {
    EvMapEntry { pol_type, id, prime, opening_pos }
}

/// A vkey of an AIR of 8 rows: a fixed f, two of stage 1 (one of them packed, k = 2) and Q's; its
/// digest not set yet.
fn sample_vkey() -> Vkey {
    let [x_c0, x_c1, y_c0, y_c1] = G2_GENERATOR.map(|d| FqBytes::from_decimal(d).unwrap());
    let vkey = Vkey {
        protocol: Protocol,
        curve: Curve,
        format_version: FORMAT_VERSION,
        n_public: 2,
        power: 3,
        power_w: 2,
        x_2: G2Affine { x: [x_c0, x_c1], y: [y_c0, y_c1] },
        num_challenges: vec![0],
        ev_map: vec![
            ev(PolType::Const, 0, 0, 1),
            ev(PolType::Cm, 0, -1, 0),
            ev(PolType::Cm, 0, 0, 1),
            ev(PolType::Cm, 1, 0, 1),
            ev(PolType::Cm, 2, 0, 1),
        ],
        layout: Layout(vec![
            f(0, &[(0, "L1")], &[0], 8),
            f(1, &[(0, "a")], &[-1, 0], 11),
            f(1, &[(1, "b"), (2, "c")], &[0], 21),
            f(2, &[(3, "Q")], &[0], 21),
        ]),
        boundaries: vec![Boundary::EveryRow],
        fixed_commitments: FixedCommitments(vec![multiple_of_g(1)]),
        q_deg: 2,
        max_q_degree: 0,
        q_verifier: json!({"tmpUsed": 1, "code": [{"op": "copy", "dest": {"type": "tmp", "id": 0, "dim": 1},
            "src": [{"type": "eval", "id": 1, "dim": 1}]}]}),
        digest: Digest::default(),
    };
    vkey.validate().unwrap();
    vkey
}

/// Keccak-256, not SHA3-256 (whose hash of "" is a7ffc6f8…): the published vectors.
#[test]
fn keccak256_is_keccak_not_sha3() {
    let hash = |hex: &str| -> [u8; 32] {
        let mut bytes = [0u8; 32];
        for (i, byte) in bytes.iter_mut().enumerate() {
            *byte = u8::from_str_radix(&hex[2 * i..2 * i + 2], 16).unwrap();
        }
        bytes
    };
    assert_eq!(keccak256(b"").unwrap(), hash("c5d2460186f7233c927e7db2dcc703c0e500b653ca82273b7bfad8045d85a470"));
    assert_eq!(keccak256(b"abc").unwrap(), hash("4e03657aea45a94fc7d47ba826c8d667c0d1e6e33a64a036ec44f58fa12d6c45"));
}

#[test]
fn the_digest_is_keccak256_of_the_preimage() {
    let vkey = sample_vkey();
    let preimage = vkey.digest_preimage().unwrap();
    assert!(preimage.starts_with(b"pilfflonk-v1{\"X_2\":"));
    let digest = vkey_digest(&vkey).unwrap();
    assert_eq!(digest, Digest(keccak256(&preimage).unwrap()));
    assert_eq!(digest.to_hex(), SAMPLE_DIGEST);
}

/// `seal_vkey` is M12's `Vkey::seal` with this Keccak-256, and the digest does not depend on the
/// one the vkey had.
#[test]
fn sealing_sets_the_digest_and_nothing_else() {
    let vkey = sample_vkey();
    let sealed = seal_vkey(vkey.clone()).unwrap();
    assert_eq!(sealed.digest.to_hex(), SAMPLE_DIGEST);
    assert_eq!(Vkey { digest: Digest::default(), ..sealed.clone() }, vkey);
    assert_eq!(vkey.clone().seal(|p| keccak256(p).unwrap()).unwrap(), sealed);
    assert!(sealed.digest_matches(|p| keccak256(p).unwrap()).unwrap());

    let mut resealed = sealed.clone();
    resealed.digest.0[31] ^= 1;
    assert!(!resealed.digest_matches(|p| keccak256(p).unwrap()).unwrap());
    assert_eq!(seal_vkey(resealed).unwrap(), sealed);

    // Written and read back, the file has the same digest.
    let dir = TestDir::new("digest_file");
    let path = dir.file("pilfflonk.vkey.json");
    sealed.write(&path).unwrap();
    let back = Vkey::read(&path).unwrap();
    assert_eq!(vkey_digest(&back).unwrap(), sealed.digest);
    assert!(fs::read_to_string(&path).unwrap().contains(&format!("\"digest\": \"{SAMPLE_DIGEST}\"")));
}

/// Every field of the vkey but `digest` is in the preimage: changing any of them changes the
/// digest.
#[test]
fn the_digest_changes_with_every_field() {
    let base = vkey_digest(&sample_vkey()).unwrap();
    let changes: [fn(&mut Vkey); 13] = [
        |v| v.n_public = 3,
        |v| v.power = 4,
        |v| v.power_w = 4,
        |v| v.x_2.y[1] = FqBytes::from_u64(1),
        |v| v.num_challenges = vec![1],
        |v| v.ev_map[1].prime = -2,
        |v| v.ev_map[2].opening_pos = 0,
        |v| v.layout.0[2].degree = 22,
        |v| v.boundaries.push(Boundary::FirstRow),
        |v| v.fixed_commitments.0[0] = multiple_of_g(2),
        |v| v.q_deg = 3,
        |v| v.max_q_degree = 1,
        |v| v.q_verifier["tmpUsed"] = json!(2),
    ];
    let mut digests = vec![base];
    for (i, change) in changes.iter().enumerate() {
        let mut vkey = sample_vkey();
        change(&mut vkey);
        let digest = vkey_digest(&vkey).unwrap();
        assert!(!digests.contains(&digest), "change {i} gives a digest seen before");
        digests.push(digest);
    }
}
