//! `pilfflonk.srs.bin`, `[τ]₂` and `<air>.verkey.json` (pilfflonk/docs/formats.md#srs,
//! pilfflonk/docs/formats.md#verkey), on the fixed columns of `common`'s pilout and the official
//! ptau of the tests (pilfflonk/docs/README.md#tests), whose `τ` is nobody's.
//!
//! The verkey must hold the commitment of each fixed `f` of the layout, its columns packed in the
//! layout's order, `commit_fixed_f` of those columns. That the commitments and `X_2` are those the
//! prover's proofs verify against, the end-to-end tests check (`cli/tests/pilfflonk_prove.rs`).

use std::fs;

use pilfflonk_setup::fixed::FixedColumns;
use pilfflonk_setup::keys::{air_verkey, commit_fixed_f, load_srs, write_srs, x_2};
use pilfflonk_setup::SetupError;
use proofman_pilfflonk::{AirVerkey, G2Affine, JsonFile, Layout, LayoutEntry, LayoutPol};
use proofman_starks_lib_c::{pilfflonk_g2_check_c, PilFflonkErrorKind, PilFflonkSrs};

use crate::common::*;
use crate::ptau;

fn columns() -> FixedColumns {
    FixedColumns::from_air(&pilout().air_groups[0].airs[0]).unwrap()
}

/// An SRS of the first `n_g1` powers of the official ptau.
fn srs(dir: &TestDir, n_g1: u64) -> PilFflonkSrs {
    let path = dir.file("pilfflonk.srs.bin");
    write_srs(&ptau::official(n_g1 as usize), n_g1, &path).unwrap();
    load_srs(&path).unwrap()
}

fn f(stage: u64, pols: &[u64], offsets: &[i64], degree: u64) -> LayoutEntry {
    LayoutEntry {
        stage,
        pols: pols.iter().map(|&id| LayoutPol { id, name: format!("p{id}") }).collect(),
        k: pols.len() as u64,
        offsets: offsets.to_vec(),
        degree,
    }
}

#[test]
fn the_srs_holds_the_powers_the_layout_needs() {
    let _cpp = cpp_core();
    let dir = TestDir::new("srs");
    let path = dir.file("pilfflonk.srs.bin");
    write_srs(&ptau::official(N), N as u64, &path).unwrap();
    // The binfile (pilfflonk/docs/formats.md#srs): "pfsr", version 1, 3 sections; a 88-byte
    // header, N points of 64 bytes and two of 128 (the section headers are 12 bytes each).
    let bytes = fs::read(&path).unwrap();
    assert_eq!(&bytes[..4], b"pfsr");
    assert_eq!(bytes.len(), 12 + 3 * 12 + 88 + N * 64 + 2 * 128);
    // Exactly N powers: a fixed f of k = 1 has N coefficients, and needs no more.
    let srs = load_srs(&path).unwrap();
    commit_fixed_f(&srs, &columns(), &[0]).unwrap();
    let err = commit_fixed_f(&srs, &columns(), &[0, 1]).unwrap_err();
    assert!(matches!(&err, SetupError::Native { source, .. } if source.kind == PilFflonkErrorKind::InvalidArgument));
}

/// A ptau with fewer powers than the largest degree of the layout is refused
/// (pilfflonk/docs/README.md#what-the-setup-refuses), and the error says how many it has: the
/// official one cut down to 16 powers.
#[test]
fn a_ptau_with_too_few_powers_is_refused() {
    let _cpp = cpp_core();
    let dir = TestDir::new("ptau_too_small");
    let ptau = dir.file("small.ptau");
    let path = dir.file("pilfflonk.srs.bin");
    ptau::truncated_copy(&ptau::official(16), &ptau, 16);
    write_srs(&ptau, 16, &path).unwrap();
    fs::remove_file(&path).unwrap();

    let err = write_srs(&ptau, 17, &path).unwrap_err();
    match &err {
        SetupError::Native { context, source } => {
            assert_eq!(source.kind, PilFflonkErrorKind::InvalidArgument);
            assert!(context.contains("17 powers [τ^i]₁ of the largest f of the layout"), "{context}");
            assert!(source.message.contains("holds 16 powers [τ^i]₁, fewer than the 17 requested"), "{source}");
        }
        other => panic!("{other}"),
    }
    assert!(!path.exists(), "nothing is written");

    let err = write_srs(&dir.file("missing.ptau"), 4, &path).unwrap_err();
    assert!(matches!(&err, SetupError::Native { source, .. } if source.kind == PilFflonkErrorKind::Io), "{err}");
}

/// A ptau whose `[τ]₂` is on the twist but outside G2, its r-torsion group
/// (pilfflonk/docs/README.md#what-the-setup-refuses): the JS verifier refuses such an `X_2`, so
/// the setup writes no SRS, and so no vkey, from it. A copy of the official ptau with that point
/// written over its `[τ]₂`: `(1, y)`, the twist's point of smallest `x = 1 + 0·u` (the one
/// `pilfflonk/js/test/elements.test.js` and the C++ SRS tests use), in the Montgomery form a ptau
/// stores (`c·2^256 mod q`, little-endian), computed apart with Python's integers.
#[test]
fn a_ptau_whose_tau_in_g2_is_outside_the_r_torsion_is_refused() {
    let _cpp = cpp_core();
    const OUTSIDE_G2_MONTGOMERY_LE: [&str; 4] = [
        "9d0d8fc58d435dd33d0bc7f528eb780a2c4679786fa36e662fdf079ac1770a0e",
        "0000000000000000000000000000000000000000000000000000000000000000",
        "36ee36d23eb8b9e7c27fecf7e7636d8d9f4141d6add1be6a9001fd267b474015",
        "617bf4465741b77c0941eaddb3a4117393ad101eb6bdec3ddb950d039f643c07",
    ];
    let dir = TestDir::new("tau_2_outside_g2");
    let ptau = dir.file("outside.ptau");
    let path = dir.file("pilfflonk.srs.bin");
    // [τ]₂, section 3's second point.
    let point: Vec<u8> = OUTSIDE_G2_MONTGOMERY_LE
        .iter()
        .flat_map(|c| (0..c.len()).step_by(2).map(move |i| u8::from_str_radix(&c[i..i + 2], 16).unwrap()))
        .collect();
    ptau::patched_copy(&ptau::official(8), &ptau, 3, ptau::G2_BYTES as u64, &point);

    let err = write_srs(&ptau, 8, &path).unwrap_err();
    match &err {
        SetupError::Native { source, .. } => {
            assert_eq!(source.kind, PilFflonkErrorKind::Format);
            assert!(
                source.message.contains("[τ^1]₂ is a point of the G2 twist not in the r-torsion group"),
                "{source}"
            );
        }
        other => panic!("{other}"),
    }
    assert!(!path.exists(), "nothing is written");
}

/// `X_2` is the SRS's `[τ]₂`, a point of G2 that is not `[1]₂`, in the vkey's JSON.
#[test]
fn x_2_is_tau_in_g2_canonical() {
    let _cpp = cpp_core();
    let dir = TestDir::new("x_2");
    let x_2 = x_2(&srs(&dir, 8)).unwrap();
    pilfflonk_g2_check_c(&x_2.to_le_bytes()).unwrap();
    assert_ne!(x_2, G2Affine::generator().unwrap());
    // The vkey's X_2: [["x.c0", "x.c1"], ["y.c0", "y.c1"]].
    let json = serde_json::to_string(&x_2).unwrap();
    let [x_c0, x_c1] = x_2.x.map(|c| c.to_decimal());
    let [y_c0, y_c1] = x_2.y.map(|c| c.to_decimal());
    assert_eq!(json, format!(r#"[["{x_c0}","{x_c1}"],["{y_c0}","{y_c1}"]]"#));
}

/// The unpacked layout (pilfflonk/docs/protocol.md#unpacked-layout): one fixed f per column,
/// k = 1, first in the layout. The verkey has their commitments in that order, and nothing of the
/// other stages.
#[test]
fn the_verkey_of_the_unpacked_layout_commits_to_each_column() {
    let _cpp = cpp_core();
    let dir = TestDir::new("verkey_unpacked");
    let srs = srs(&dir, N as u64);
    let n = N as u64;
    let layout = Layout(vec![
        f(0, &[0], &[0], n),
        f(0, &[1], &[0, 1], n),
        f(0, &[2], &[0], n),
        f(0, &[3], &[1], n),
        f(1, &[0], &[0, 1], n + 3),
        f(1, &[1], &[0], n + 2),
        f(2, &[2], &[0], 3 * n),
    ]);
    let verkey = air_verkey(&srs, &columns(), &layout).unwrap();
    let commitment = |pols: &[u64]| commit_fixed_f(&srs, &columns(), pols).unwrap();
    assert_eq!(verkey, AirVerkey(vec![commitment(&[0]), commitment(&[1]), commitment(&[2]), commitment(&[3])]));
    // Four different points.
    assert!((1..4).all(|i| (0..i).all(|j| verkey.0[i] != verkey.0[j])), "{verkey:?}");

    // Its file: the points as decimal strings, in the same order.
    let path = dir.file("Sample.verkey.json");
    verkey.write(&path).unwrap();
    assert_eq!(AirVerkey::read(&path).unwrap(), verkey);
    let text = fs::read_to_string(&path).unwrap();
    let first = &verkey.0[0];
    let start = format!("[\n [\n  \"{}\",\n  \"{}\"\n ],", first.x.to_decimal(), first.y.to_decimal());
    assert!(text.starts_with(&start), "{text}");

    // The layout's order, not the columns': the same f in another order.
    let reordered = Layout(vec![f(0, &[2], &[0], n), f(0, &[0], &[0], n)]);
    let verkey = air_verkey(&srs, &columns(), &reordered).unwrap();
    assert_eq!(verkey, AirVerkey(vec![commitment(&[2]), commitment(&[0])]));
}

/// A packed fixed f commits to its columns interleaved in the layout's order: `[0, 1]` is not
/// `[1, 0]`.
#[test]
fn the_verkey_of_a_packed_layout_commits_to_each_f() {
    let _cpp = cpp_core();
    let dir = TestDir::new("verkey_packed");
    let srs = srs(&dir, 2 * N as u64);
    let n = N as u64;
    let layout = Layout(vec![f(0, &[0, 1], &[0], 2 * n), f(0, &[1, 2, 3], &[0], 3 * n)]);
    // The second one takes 3N powers, more than the SRS has.
    let err = air_verkey(&srs, &columns(), &layout).unwrap_err();
    assert!(matches!(&err, SetupError::Native { source, .. } if source.message.contains("exceed")), "{err}");
    let layout = Layout(vec![f(0, &[0, 1], &[0], 2 * n), f(0, &[3, 2], &[0], 2 * n)]);
    let verkey = air_verkey(&srs, &columns(), &layout).unwrap();
    let commitment = |pols: &[u64]| commit_fixed_f(&srs, &columns(), pols).unwrap();
    assert_eq!(verkey, AirVerkey(vec![commitment(&[0, 1]), commitment(&[3, 2])]));
    assert_ne!(commitment(&[0, 1]), commitment(&[1, 0]));
    let layout = Layout(vec![f(0, &[1, 2], &[0], 2 * n)]);
    assert_eq!(air_verkey(&srs, &columns(), &layout).unwrap(), AirVerkey(vec![commitment(&[1, 2])]));
}

#[test]
fn a_layout_that_does_not_fit_the_columns_is_refused() {
    let _cpp = cpp_core();
    let dir = TestDir::new("verkey_refused");
    let srs = srs(&dir, N as u64);
    let n = N as u64;
    let err = air_verkey(&srs, &columns(), &Layout(vec![f(0, &[4], &[0], n)])).unwrap_err();
    assert!(matches!(&err, SetupError::Layout(m) if m.contains("column 4, and the AIR has 4")), "{err}");
    let mut wrong_k = f(0, &[0], &[0], n);
    wrong_k.k = 2;
    let err = air_verkey(&srs, &columns(), &Layout(vec![wrong_k])).unwrap_err();
    assert!(matches!(&err, SetupError::Layout(m) if m.contains("k = 2")), "{err}");
    // A layout with no fixed f: an empty verkey.
    let layout = Layout(vec![f(1, &[0], &[0], n + 2), f(2, &[1], &[0], 2 * n)]);
    assert_eq!(air_verkey(&srs, &columns(), &layout).unwrap(), AirVerkey(vec![]));
}
