//! The SRS and fixed-commitment wrappers end to end, on the official ptau of the tests
//! (pilfflonk/docs/README.md#tests) and on copies of it, changed.
//!
//! A fixed `f = Σ_j p_j(X^k)·X^j` whose columns are constant, the first `c` and the others zero, is
//! the constant `c` and commits to `c·G` whatever `τ`: that checks the path (files, interpolation,
//! packing, MSM, encodings) against points known in advance. The commitments of other polynomials
//! the C++ tests check (`make -C pil2-stark pilfflonk_test`), and the end-to-end tests verify the
//! proofs made with them.

#[path = "../../../pilfflonk/tests/data/ptau.rs"]
mod ptau;

use std::fs;
use std::path::{Path, PathBuf};

use proofman_starks_lib_c::{
    pilfflonk_g2_check_c, pilfflonk_srs_from_ptau_c, PilFflonkErrorKind, PilFflonkSrs, PILFFLONK_FR_BYTES,
    PILFFLONK_G1_BYTES, PILFFLONK_G2_BYTES,
};

/// 32 bytes from 64 hex digits in byte order: little-endian as they are written.
fn le_bytes(hex: &str) -> [u8; 32] {
    assert_eq!(hex.len(), 64, "{hex}");
    let mut bytes = [0u8; 32];
    for (i, byte) in bytes.iter_mut().enumerate() {
        *byte = u8::from_str_radix(&hex[2 * i..2 * i + 2], 16).unwrap();
    }
    bytes
}

/// 32 little-endian bytes from 64 hex digits of a big-endian number.
fn from_hex(hex: &str) -> [u8; 32] {
    let mut bytes = le_bytes(hex);
    bytes.reverse();
    bytes
}

/// BN128's base field modulus q, little-endian.
const Q: &str = "47fd7cd8168c203c8dca7168916a81975d588181b64550b829a031e1724e6430";

/// BN128's scalar field modulus r, big-endian.
const R: &str = "30644e72e131a029b85045b68181585d2833e84879b9709143e1f593f0000001";

/// k·G for the generator G = (1, 2), canonical big-endian coordinates: the constants of the
/// transcript tests in src/ffi_pilfflonk.rs.
const P1: [&str; 2] = [
    "0000000000000000000000000000000000000000000000000000000000000001",
    "0000000000000000000000000000000000000000000000000000000000000002",
];
const P2: [&str; 2] = [
    "030644e72e131a029b85045b68181585d97816a916871ca8d3c208c16d87cfd3",
    "15ed738c0e0a7c92e7845f96b2ae9c0a68a6a449e3538fc7ff3ebf7a5a18a2c4",
];
const P3: [&str; 2] = [
    "0769bf9ac56bea3ff40232bcb1b6bd159315d84715b8e679f2d355961915abf0",
    "2ab799bee0489429554fdb7c8d086475319e63b40b9c5b57cdf1ff3dd9fe2261",
];
const P5: [&str; 2] = [
    "17c139df0efee0f766bc0204762b774362e4ded88953a39ce849a8a7fa163fa9",
    "01e0559bacb160664764a357af8a9fe70baa9258e0b959273ffc5718c6d4cc7c",
];

fn point(coordinates: [&str; 2]) -> [u8; PILFFLONK_G1_BYTES] {
    let mut bytes = [0u8; PILFFLONK_G1_BYTES];
    bytes[..32].copy_from_slice(&from_hex(coordinates[0]));
    bytes[32..].copy_from_slice(&from_hex(coordinates[1]));
    bytes
}

fn scalar(value: u64) -> [u8; PILFFLONK_FR_BYTES] {
    let mut bytes = [0u8; PILFFLONK_FR_BYTES];
    bytes[..8].copy_from_slice(&value.to_le_bytes());
    bytes
}

/// A fresh directory under target/tmp for one test, removed when it ends.
struct TestDir(PathBuf);

impl TestDir {
    fn new(name: &str) -> Self {
        let dir = Path::new(env!("CARGO_TARGET_TMPDIR")).join(format!("pilfflonk_srs_{name}_{}", std::process::id()));
        fs::create_dir_all(&dir).unwrap();
        Self(dir)
    }

    fn file(&self, name: &str) -> PathBuf {
        self.0.join(name)
    }
}

impl Drop for TestDir {
    fn drop(&mut self) {
        // Best effort: a failed test may leave it behind, under target/.
        let _ = fs::remove_dir_all(&self.0);
    }
}

/// `columns` one after another, as `commit_fixed` takes them.
fn evals(columns: &[&[u64]]) -> Vec<[u8; PILFFLONK_FR_BYTES]> {
    columns.iter().flat_map(|column| column.iter().map(|&v| scalar(v))).collect()
}

#[test]
fn commits_constant_fixed_columns_to_multiples_of_g() {
    let dir = TestDir::new("commits");
    let srs_path = dir.file("pilfflonk.srs.bin");
    pilfflonk_srs_from_ptau_c(&ptau::official(64), 64, &srs_path).unwrap();
    let srs = PilFflonkSrs::load(&srs_path).unwrap();

    /// Columns on a domain of 2^n_bits points, and their commitment.
    struct Case {
        n_bits: u64,
        columns: &'static [&'static [u64]],
        expected: [u8; PILFFLONK_G1_BYTES],
    }
    let cases = [
        Case { n_bits: 2, columns: &[&[1, 1, 1, 1]], expected: point(P1) },
        Case { n_bits: 2, columns: &[&[2, 2, 2, 2], &[0, 0, 0, 0], &[0, 0, 0, 0]], expected: point(P2) },
        Case { n_bits: 3, columns: &[&[3; 8], &[0; 8]], expected: point(P3) },
        Case { n_bits: 0, columns: &[&[5]], expected: point(P5) },
        // The point at infinity.
        Case { n_bits: 1, columns: &[&[0, 0], &[0, 0]], expected: [0u8; PILFFLONK_G1_BYTES] },
    ];
    for Case { n_bits, columns, expected } in cases {
        let k = columns.len() as u64;
        assert_eq!(srs.commit_fixed(n_bits, k, &evals(columns)).unwrap(), expected, "n_bits {n_bits}, k {k}");
    }
    // A column that is not constant commits to another point. Using an SRS only reads it: again,
    // the same.
    let x = srs.commit_fixed(2, 1, &evals(&[&[1, 7, 8, 9]])).unwrap();
    assert!(x != point(P1) && x != [0u8; PILFFLONK_G1_BYTES]);
    assert_eq!(srs.commit_fixed(2, 1, &evals(&[&[1, 7, 8, 9]])).unwrap(), x);
}

/// The generator of G2, canonical big-endian coordinates `x.c0, x.c1, y.c0, y.c1`, as every
/// BN128 library writes them.
const G2_CANONICAL: [&str; 4] = [
    "1800deef121f1e76426a00665e5c4479674322d4f75edadd46debd5cd992f6ed",
    "198e9393920d483a7260bfb731fb5d25f1aa493335a9e71297e485b7aef312c2",
    "12c85ea5db8c6deb4aab71808dcb408fe3d1e7690c43d37b4ce6cc0166fa7daa",
    "090689d0585ff075ec9e99ad690c3395bc4b313370b38ef355acdadcd122975b",
];

#[test]
fn gives_the_g2_powers_canonical() {
    let dir = TestDir::new("g2");
    let ptau = ptau::official(4);
    let srs_path = dir.file("pilfflonk.srs.bin");
    pilfflonk_srs_from_ptau_c(&ptau, 4, &srs_path).unwrap();
    let srs = PilFflonkSrs::load(&srs_path).unwrap();

    // [1]₂ is the generator, and [τ]₂ another point of G2.
    assert_eq!(srs.g2(0).unwrap(), g2(G2_CANONICAL));
    pilfflonk_g2_check_c(&srs.g2(1).unwrap()).unwrap();
    assert_ne!(srs.g2(1).unwrap(), g2(G2_CANONICAL));

    let err = srs.g2(2).unwrap_err();
    assert_eq!(err.kind, PilFflonkErrorKind::InvalidArgument, "{err}");
    assert!(err.message.contains("pilfflonk_srs_g2: i = 2"), "{err}");
}

/// A G2 point from canonical big-endian coordinates `x.c0, x.c1, y.c0, y.c1`, as the C API takes it.
fn g2(coordinates: [&str; 4]) -> [u8; PILFFLONK_G2_BYTES] {
    let mut point = [0u8; PILFFLONK_G2_BYTES];
    for (chunk, coordinate) in point.chunks_exact_mut(32).zip(coordinates) {
        chunk.copy_from_slice(&from_hex(coordinate));
    }
    point
}

/// A point of the twist outside G2, its r-torsion group: `x = 2 + u` and a square root of
/// `x³ + 3/(9+u)`, found with ffjavascript (`G2.isValid`, and `r·P ≠ 0`).
const TWIST_NOT_G2: [&str; 4] = [
    "0000000000000000000000000000000000000000000000000000000000000002",
    "0000000000000000000000000000000000000000000000000000000000000001",
    "101f7278419308b95099eca02dcee0c5381f4d26d1d62313f057167f064101ce",
    "2b76c179599bb92a963dac85546a005a777f7c13f6a7b75d5918b6b5808f5fde",
];

/// `pilfflonk_g2_check`: what the JS verifier requires of the vkey's X_2 (elements.js,
/// g2FromObject), a point of G2 other than the point at infinity
/// (pilfflonk/docs/verifier.md#refused-vkeys).
#[test]
fn checks_g2_points_as_the_js_verifier_does() {
    pilfflonk_g2_check_c(&g2(G2_CANONICAL)).unwrap();
    let mut off_twist = g2(G2_CANONICAL);
    off_twist[64] ^= 1;
    let mut non_canonical = g2(G2_CANONICAL);
    non_canonical[32..64].copy_from_slice(&le_bytes(Q));
    for (point, kind, text) in [
        ([0u8; PILFFLONK_G2_BYTES], PilFflonkErrorKind::InvalidPoint, "the point at infinity"),
        (off_twist, PilFflonkErrorKind::InvalidPoint, "not on the twist"),
        (g2(TWIST_NOT_G2), PilFflonkErrorKind::InvalidPoint, "not in G2"),
        (non_canonical, PilFflonkErrorKind::NonCanonical, "coordinate 1 is not below"),
    ] {
        let err = pilfflonk_g2_check_c(&point).unwrap_err();
        assert_eq!(err.kind, kind, "{err}");
        assert!(err.message.contains("pilfflonk_g2_check") && err.message.contains(text), "{err}");
    }
}

/// A ptau whose [τ]₂ is the point at infinity (τ = 0) is refused, and says so
/// (pilfflonk/docs/verifier.md#refused-vkeys): a copy of the official one with that point zeroed.
#[test]
fn refuses_a_ptau_whose_tau_g2_is_the_point_at_infinity() {
    let dir = TestDir::new("tau_g2_infinity");
    let ptau = dir.file("tau_g2_infinity.ptau");
    let srs_path = dir.file("pilfflonk.srs.bin");
    // Section 3's second point, [τ]₂.
    ptau::patched_copy(&ptau::official(4), &ptau, 3, ptau::G2_BYTES as u64, &[0u8; PILFFLONK_G2_BYTES]);
    let err = pilfflonk_srs_from_ptau_c(&ptau, 4, &srs_path).unwrap_err();
    assert_eq!(err.kind, PilFflonkErrorKind::Format, "{err}");
    assert!(err.message.contains("[τ^1]₂ is the point at infinity"), "{err}");
    assert!(!srs_path.exists());
}

#[test]
fn refuses_bad_files_and_arguments() {
    let dir = TestDir::new("refuses");
    let ptau = ptau::official(16);
    let n_g1 = ptau::section(&ptau, 2).1 as usize / PILFFLONK_G1_BYTES;
    let srs_path = dir.file("pilfflonk.srs.bin");
    let missing = dir.file("missing");

    let fewer = format!("fewer than the {}", n_g1 + 1);
    let refusals = [
        (pilfflonk_srs_from_ptau_c(&missing, 16, &srs_path), PilFflonkErrorKind::Io, "open"),
        (
            pilfflonk_srs_from_ptau_c(&ptau, n_g1 as u64 + 1, &srs_path),
            PilFflonkErrorKind::InvalidArgument,
            fewer.as_str(),
        ),
        (pilfflonk_srs_from_ptau_c(&ptau, 0, &srs_path), PilFflonkErrorKind::InvalidArgument, "nG1 = 0"),
        (pilfflonk_srs_from_ptau_c(Path::new("a\0b"), 16, &srs_path), PilFflonkErrorKind::InvalidArgument, "NUL"),
    ];
    for (result, kind, text) in refusals {
        let err = result.unwrap_err();
        assert_eq!(err.kind, kind, "{err}");
        assert!(err.message.contains("pilfflonk_srs_from_ptau") && err.message.contains(text), "{err}");
    }
    assert!(!srs_path.exists());

    for (path, kind, text) in [
        (missing.as_path(), PilFflonkErrorKind::Io, "open"),
        (ptau.as_path(), PilFflonkErrorKind::Format, "Invalid file type"),
        (Path::new("a\0b"), PilFflonkErrorKind::InvalidArgument, "NUL"),
    ] {
        let err = PilFflonkSrs::load(path).unwrap_err();
        assert_eq!(err.kind, kind, "{err}");
        assert!(err.message.contains("pilfflonk_srs_load") && err.message.contains(text), "{err}");
    }

    pilfflonk_srs_from_ptau_c(&ptau, 16, &srs_path).unwrap();
    let srs = PilFflonkSrs::load(&srs_path).unwrap();
    let mut non_canonical = evals(&[&[1, 2, 3, 4], &[5, 6, 7, 8]]);
    non_canonical[6] = from_hex(R);
    let refusals = [
        (srs.commit_fixed(2, 2, &evals(&[&[1, 2, 3, 4]])), PilFflonkErrorKind::InvalidArgument, "evals holds 4"),
        (srs.commit_fixed(64, 1, &[]), PilFflonkErrorKind::InvalidArgument, "evals holds 0"),
        (srs.commit_fixed(2, 0, &[]), PilFflonkErrorKind::InvalidArgument, "k = 0"),
        (srs.commit_fixed(29, 0, &[]), PilFflonkErrorKind::InvalidArgument, "k = 0"),
        (srs.commit_fixed(3, 3, &vec![scalar(1); 24]), PilFflonkErrorKind::InvalidArgument, "exceed the 16 powers"),
        (srs.commit_fixed(2, 2, &non_canonical), PilFflonkErrorKind::NonCanonical, "column 1, row 2"),
    ];
    for (result, kind, text) in refusals {
        let err = result.unwrap_err();
        assert_eq!(err.kind, kind, "{err}");
        assert!(err.message.contains("pilfflonk_commit_fixed") && err.message.contains(text), "{err}");
    }
}
