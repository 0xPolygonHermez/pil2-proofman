//! The SRS and fixed-commitment wrappers end to end, on a ptau with τ = 1 written here.
//!
//! With τ = 1 every power `[τ^i]₁` is the generator G, so the commitment of a fixed
//! `f = Σ_j p_j(X^k)·X^j` is `f(1)·G = (Σ_j p_j(1))·G`, and `p_j(1)` is the first evaluation of
//! column j (the domain starts at ω^0 = 1). That checks the whole path (files, interpolation,
//! packing, MSM, encodings) against points known in advance, with no curve arithmetic here. What
//! τ = 1 cannot see, the order of the interleaving, the C++ tests check with a full-width τ
//! (`make -C pil2-stark pilfflonk_test`).

use std::fs;
use std::path::{Path, PathBuf};

use proofman_starks_lib_c::{
    pilfflonk_srs_from_ptau_c, PilFflonkErrorKind, PilFflonkSrs, PILFFLONK_FR_BYTES, PILFFLONK_G1_BYTES,
    PILFFLONK_G2_BYTES,
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

/// BN254's base field modulus q, little-endian.
const Q: &str = "47fd7cd8168c203c8dca7168916a81975d588181b64550b829a031e1724e6430";

/// The generators of G1, (1, 2), and G2 in Montgomery form (c·2^256 mod q), little-endian: as a
/// snarkjs ptau stores them. Computed offline; the reader checks that `[1]₁` and `[1]₂` are the
/// generators, so a wrong digit fails the tests.
const G1_MONTGOMERY: [&str; 2] = [
    "9d0d8fc58d435dd33d0bc7f528eb780a2c4679786fa36e662fdf079ac1770a0e",
    "3a1b1e8b1b87baa67b168eeb51d6f114588cf2f0de46ddcc5ebe0f3483ef141c",
];
const G2_MONTGOMERY: [&str; 4] = [
    "2620bc02d1b5838e72017b493519ebdcdf1a81974726b8fb3b5096af41385719",
    "40614ca87d73b4afc4d802585add4360862fa052fc50e9096b7bea3a83f0fe14",
    "f6e96b889dfa9d61789b9ef597d27ffefe7d1b23621a9eff06429eaeeb7efd28",
    "ee5618c7565b0964bb3c7d3222f957dc76103533be35f9558264fd93e6a0a40d",
];

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

/// A binfile as snarkjs and rapidsnark write them: type, version, sections as (id, size, bytes).
fn binfile(file_type: &[u8; 4], sections: &[(u32, Vec<u8>)]) -> Vec<u8> {
    let mut bytes = file_type.to_vec();
    bytes.extend(1u32.to_le_bytes());
    bytes.extend((sections.len() as u32).to_le_bytes());
    for (id, contents) in sections {
        bytes.extend(id.to_le_bytes());
        bytes.extend((contents.len() as u64).to_le_bytes());
        bytes.extend(contents);
    }
    bytes
}

/// A ptau with τ = 1: the header of power 7, `n_g1` copies of G1's generator and two of G2's.
fn ptau_of_tau_one(n_g1: usize) -> Vec<u8> {
    let mut header = 32u32.to_le_bytes().to_vec();
    header.extend(le_bytes(Q));
    header.extend(7u32.to_le_bytes());
    header.extend(7u32.to_le_bytes());
    let g1: Vec<u8> = G1_MONTGOMERY.iter().flat_map(|c| le_bytes(c)).collect();
    let g2: Vec<u8> = G2_MONTGOMERY.iter().flat_map(|c| le_bytes(c)).collect();
    binfile(b"ptau", &[(1, header), (2, g1.repeat(n_g1)), (3, g2.repeat(2))])
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
fn commits_fixed_columns_with_a_tau_one_srs() {
    let dir = TestDir::new("commits");
    let ptau = dir.file("tau_one.ptau");
    let srs_path = dir.file("pilfflonk.srs.bin");
    fs::write(&ptau, ptau_of_tau_one(64)).unwrap();
    pilfflonk_srs_from_ptau_c(&ptau, 64, &srs_path).unwrap();
    let srs = PilFflonkSrs::load(&srs_path).unwrap();

    /// Columns on a domain of 2^n_bits points, and f(1)·G: the sum of their first rows, times G.
    struct Case {
        n_bits: u64,
        columns: &'static [&'static [u64]],
        expected: [u8; PILFFLONK_G1_BYTES],
    }
    let cases = [
        Case { n_bits: 2, columns: &[&[1, 7, 8, 9]], expected: point(P1) },
        Case { n_bits: 2, columns: &[&[1, 4, 4, 4], &[1, 0, 0, 0], &[0, 5, 6, 7]], expected: point(P2) },
        Case { n_bits: 3, columns: &[&[2, 1, 1, 1, 1, 1, 1, 1], &[3, 9, 9, 9, 9, 9, 9, 9]], expected: point(P5) },
        Case { n_bits: 0, columns: &[&[1], &[1], &[1]], expected: point(P3) },
        // The point at infinity.
        Case { n_bits: 1, columns: &[&[0, 0], &[0, 0]], expected: [0u8; PILFFLONK_G1_BYTES] },
    ];
    for Case { n_bits, columns, expected } in cases {
        let k = columns.len() as u64;
        assert_eq!(srs.commit_fixed(n_bits, k, &evals(columns)).unwrap(), expected, "n_bits {n_bits}, k {k}");
    }
    // Using an SRS only reads it: again, the same.
    assert_eq!(srs.commit_fixed(2, 1, &evals(&[&[1, 7, 8, 9]])).unwrap(), point(P1));
}

/// The generator of G2, canonical big-endian coordinates `x.c0, x.c1, y.c0, y.c1`, as every
/// BN254 library writes them.
const G2_CANONICAL: [&str; 4] = [
    "1800deef121f1e76426a00665e5c4479674322d4f75edadd46debd5cd992f6ed",
    "198e9393920d483a7260bfb731fb5d25f1aa493335a9e71297e485b7aef312c2",
    "12c85ea5db8c6deb4aab71808dcb408fe3d1e7690c43d37b4ce6cc0166fa7daa",
    "090689d0585ff075ec9e99ad690c3395bc4b313370b38ef355acdadcd122975b",
];

#[test]
fn gives_the_g2_powers_canonical() {
    let dir = TestDir::new("g2");
    let ptau = dir.file("tau_one.ptau");
    let srs_path = dir.file("pilfflonk.srs.bin");
    fs::write(&ptau, ptau_of_tau_one(4)).unwrap();
    pilfflonk_srs_from_ptau_c(&ptau, 4, &srs_path).unwrap();
    let srs = PilFflonkSrs::load(&srs_path).unwrap();

    // With τ = 1, [1]₂ and [τ]₂ are both the generator.
    let mut generator = [0u8; PILFFLONK_G2_BYTES];
    for (chunk, coordinate) in generator.chunks_exact_mut(32).zip(G2_CANONICAL) {
        chunk.copy_from_slice(&from_hex(coordinate));
    }
    assert_eq!(srs.g2(0).unwrap(), generator);
    assert_eq!(srs.g2(1).unwrap(), generator);

    let err = srs.g2(2).unwrap_err();
    assert_eq!(err.kind, PilFflonkErrorKind::InvalidArgument, "{err}");
    assert!(err.message.contains("pilfflonk_srs_g2: i = 2"), "{err}");
}

#[test]
fn refuses_bad_files_and_arguments() {
    let dir = TestDir::new("refuses");
    let ptau = dir.file("tau_one.ptau");
    let srs_path = dir.file("pilfflonk.srs.bin");
    let missing = dir.file("missing");
    fs::write(&ptau, ptau_of_tau_one(16)).unwrap();

    let refusals = [
        (pilfflonk_srs_from_ptau_c(&missing, 16, &srs_path), PilFflonkErrorKind::Io, "open"),
        (pilfflonk_srs_from_ptau_c(&ptau, 17, &srs_path), PilFflonkErrorKind::InvalidArgument, "fewer than the 17"),
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
    let r = from_hex("30644e72e131a029b85045b68181585d2833e84879b9709143e1f593f0000001");
    let mut non_canonical = evals(&[&[1, 2, 3, 4], &[5, 6, 7, 8]]);
    non_canonical[6] = r;
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
