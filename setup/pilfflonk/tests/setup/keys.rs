//! `pilfflonk.srs.bin`, `[τ]₂` and `<air>.verkey.json` (spec §4.2.5, A.6), on the fixed columns of
//! `common`'s pilout and a ptau with `τ = 1` (`pilfflonk_setup::test_ptau`).
//!
//! With `τ = 1` the commitment of a fixed `f = Σ_j p_j(X^k)·X^j` is `(Σ_j p_j(1))·G`, the sum of
//! the first rows of its columns times the generator: the points are known in advance
//! (`common::multiple_of_g`), independently of the C++ core. What `τ = 1` cannot see, the order
//! of the columns within a packed `f` and the interpolation of the other rows, the C++ tests of
//! `pilfflonk_commit_fixed` check with a full-width `τ` (`make -C pil2-stark pilfflonk_test`).

use std::fs;

use pilfflonk_setup::fixed::FixedColumns;
use pilfflonk_setup::keys::{air_verkey, commit_fixed_f, load_srs, write_srs, x_2};
use pilfflonk_setup::test_ptau::write_tau_one_ptau;
use pilfflonk_setup::SetupError;
use proofman_pilfflonk::{AirVerkey, FqBytes, G2Affine, JsonFile, Layout, LayoutEntry, LayoutPol};
use proofman_starks_lib_c::{PilFflonkErrorKind, PilFflonkSrs};

use crate::common::*;

fn columns() -> FixedColumns {
    FixedColumns::from_air(&pilout().air_groups[0].airs[0]).unwrap()
}

/// An SRS of `n_g1` powers from a ptau with `τ = 1` of `ptau_g1` powers.
fn srs(dir: &TestDir, ptau_g1: usize, n_g1: u64) -> PilFflonkSrs {
    let ptau = dir.file("tau_one.ptau");
    let path = dir.file("pilfflonk.srs.bin");
    write_tau_one_ptau(&ptau, ptau_g1).unwrap();
    write_srs(&ptau, n_g1, &path).unwrap();
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
    let ptau = dir.file("tau_one.ptau");
    let path = dir.file("pilfflonk.srs.bin");
    write_tau_one_ptau(&ptau, 32).unwrap();
    write_srs(&ptau, N as u64, &path).unwrap();
    // The binfile of A.6: "pfsr", version 1, 3 sections; a 88-byte header, N points of 64 bytes
    // and two of 128 (the section headers are 12 bytes each).
    let bytes = fs::read(&path).unwrap();
    assert_eq!(&bytes[..4], b"pfsr");
    assert_eq!(bytes.len(), 12 + 3 * 12 + 88 + N * 64 + 2 * 128);
    // Exactly N powers: a fixed f of k = 1 has N coefficients, and needs no more.
    let srs = load_srs(&path).unwrap();
    commit_fixed_f(&srs, &columns(), &[0]).unwrap();
    let err = commit_fixed_f(&srs, &columns(), &[0, 1]).unwrap_err();
    assert!(matches!(&err, SetupError::Native { source, .. } if source.kind == PilFflonkErrorKind::InvalidArgument));
}

/// Spec §4.2.1: a ptau with fewer powers than the largest degree of the layout is refused, and
/// the error says how many it has.
#[test]
fn a_ptau_with_too_few_powers_is_refused() {
    let _cpp = cpp_core();
    let dir = TestDir::new("ptau_too_small");
    let ptau = dir.file("tau_one.ptau");
    let path = dir.file("pilfflonk.srs.bin");
    write_tau_one_ptau(&ptau, 16).unwrap();
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

/// A ptau whose `[τ]₂` is on the twist but outside G2, its r-torsion group (plan M26): the JS
/// verifier refuses such an `X_2`, so the setup writes no SRS, and so no vkey, from it. The point
/// is `(1, y)`, the twist's point of smallest `x = 1 + 0·u` (the one `pilfflonk/js/test/
/// elements.test.js` and the C++ SRS tests use), in the Montgomery form a ptau stores (`c·2^256 mod
/// q`, little-endian), computed apart with Python's integers.
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
    // Section 3, [1]₂ then [τ]₂, is the last of the file.
    let mut bytes = pilfflonk_setup::test_ptau::tau_one_ptau(8);
    let point: Vec<u8> = OUTSIDE_G2_MONTGOMERY_LE
        .iter()
        .flat_map(|c| (0..c.len()).step_by(2).map(move |i| u8::from_str_radix(&c[i..i + 2], 16).unwrap()))
        .collect();
    let at = bytes.len() - point.len();
    bytes[at..].copy_from_slice(&point);
    fs::write(&ptau, &bytes).unwrap();

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

#[test]
fn x_2_is_tau_in_g2_canonical() {
    let _cpp = cpp_core();
    let dir = TestDir::new("x_2");
    let x_2 = x_2(&srs(&dir, 8, 8)).unwrap();
    let [x_c0, x_c1, y_c0, y_c1] = G2_GENERATOR.map(|d| FqBytes::from_decimal(d).unwrap());
    assert_eq!(x_2, G2Affine { x: [x_c0, x_c1], y: [y_c0, y_c1] });
    // The vkey's X_2: [["x.c0", "x.c1"], ["y.c0", "y.c1"]].
    let json = serde_json::to_string(&x_2).unwrap();
    assert_eq!(
        json,
        format!(r#"[["{}","{}"],["{}","{}"]]"#, G2_GENERATOR[0], G2_GENERATOR[1], G2_GENERATOR[2], G2_GENERATOR[3])
    );
}

/// The unpacked layout of shortcut R1: one fixed f per column, k = 1, first in the layout. The
/// verkey has their commitments in that order, and nothing of the other stages.
#[test]
fn the_verkey_of_the_unpacked_layout_commits_to_each_column() {
    let _cpp = cpp_core();
    let dir = TestDir::new("verkey_unpacked");
    let srs = srs(&dir, N, N as u64);
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
    // First rows 1, 2, 3 and 0.
    assert_eq!(verkey, AirVerkey(vec![multiple_of_g(1), multiple_of_g(2), multiple_of_g(3), multiple_of_g(0)]));

    // Its file: the points as decimal strings, in the same order.
    let path = dir.file("Sample.verkey.json");
    verkey.write(&path).unwrap();
    assert_eq!(AirVerkey::read(&path).unwrap(), verkey);
    let text = fs::read_to_string(&path).unwrap();
    assert!(text.starts_with("[\n [\n  \"1\",\n  \"2\"\n ],"), "{text}");

    // The layout's order, not the columns': the same f in another order.
    let reordered = Layout(vec![f(0, &[2], &[0], n), f(0, &[0], &[0], n)]);
    let verkey = air_verkey(&srs, &columns(), &reordered).unwrap();
    assert_eq!(verkey, AirVerkey(vec![multiple_of_g(3), multiple_of_g(1)]));
}

/// A packed fixed f commits to the sum of its columns' first rows with τ = 1.
#[test]
fn the_verkey_of_a_packed_layout_commits_to_each_f() {
    let _cpp = cpp_core();
    let dir = TestDir::new("verkey_packed");
    let srs = srs(&dir, 2 * N, 2 * N as u64);
    let n = N as u64;
    let layout = Layout(vec![f(0, &[0, 1], &[0], 2 * n), f(0, &[1, 2, 3], &[0], 3 * n)]);
    // 1 + 2 and 2 + 3 + 0; the second one takes 3N powers, more than the SRS has.
    let err = air_verkey(&srs, &columns(), &layout).unwrap_err();
    assert!(matches!(&err, SetupError::Native { source, .. } if source.message.contains("exceed")), "{err}");
    let layout = Layout(vec![f(0, &[0, 1], &[0], 2 * n), f(0, &[3, 2], &[0], 2 * n)]);
    let verkey = air_verkey(&srs, &columns(), &layout).unwrap();
    assert_eq!(verkey, AirVerkey(vec![multiple_of_g(3), multiple_of_g(3)]));
    let layout = Layout(vec![f(0, &[1, 2], &[0], 2 * n)]);
    assert_eq!(air_verkey(&srs, &columns(), &layout).unwrap(), AirVerkey(vec![multiple_of_g(5)]));
}

#[test]
fn a_layout_that_does_not_fit_the_columns_is_refused() {
    let _cpp = cpp_core();
    let dir = TestDir::new("verkey_refused");
    let srs = srs(&dir, N, N as u64);
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
