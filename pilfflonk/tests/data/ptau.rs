//! The powers-of-tau files of the tests (pilfflonk/docs/README.md#tests): the official ones of the
//! Hermez ceremony, of the table in snarkjs's README, which `pilfflonk/tests/download-ptau.sh`
//! downloads into `PILFFLONK_PTAU_DIR` (by default `$XDG_CACHE_HOME/pilfflonk/ptau`, or
//! `~/.cache/pilfflonk/ptau`) and checks against the README's blake2b. The tests never download
//! one, and never make one: their `τ` is nobody's, so a test checks a commitment against another
//! implementation, or a proof by verifying it, and a test of a malformed file, or of another SRS,
//! changes a copy of an official one.
//!
//! The file of power `k` holds `2^(k+1) − 1` powers `[τ^i]₁` (section 2) and `2^k` powers `[τ^i]₂`
//! (section 3), every coordinate in Montgomery form (`c·2^256 mod q`), 32 bytes little-endian; it
//! is a binfile: `"ptau"`, its version and its number of sections, then each section as its id
//! (`u32`), its size (`u64`) and its bytes.
//!
//! Include it with `#[path = ".../pilfflonk/tests/data/ptau.rs"] mod ptau;`.

// Each test crate that includes it uses some of it.
#![allow(dead_code)]

use std::fs;
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};

/// The script that downloads them.
pub const DOWNLOAD: &str = "pilfflonk/tests/download-ptau.sh";

/// Bytes of a point `[τ^i]₁` and of a point `[τ^i]₂`.
pub const G1_BYTES: usize = 64;
pub const G2_BYTES: usize = 128;

/// The powers of the files of the README's table.
const POWERS: std::ops::RangeInclusive<u32> = 8..=28;

/// Where they are: `PILFFLONK_PTAU_DIR`, or the default of [`DOWNLOAD`]. A variable set to nothing
/// is not set, as for the script.
pub fn dir() -> PathBuf {
    let var = |name: &str| std::env::var_os(name).filter(|value| !value.is_empty()).map(PathBuf::from);
    if let Some(dir) = var("PILFFLONK_PTAU_DIR") {
        return dir;
    }
    let cache = var("XDG_CACHE_HOME")
        .or_else(|| var("HOME").map(|home| home.join(".cache")))
        .expect("neither PILFFLONK_PTAU_DIR, XDG_CACHE_HOME nor HOME is set");
    cache.join("pilfflonk/ptau")
}

/// The name of the file of `power`.
pub fn file_name(power: u32) -> String {
    if power == 28 {
        "powersOfTau28_hez_final.ptau".to_string()
    } else {
        format!("powersOfTau28_hez_final_{power:02}.ptau")
    }
}

/// The powers `[τ^i]₁` of the file of `power`: `2^(power+1) − 1`.
pub fn g1_powers(power: u32) -> usize {
    (2usize << power) - 1
}

/// The official file of the fewest powers `[τ^i]₁` that holds at least `n_g1`, of those in
/// [`dir`]. Panics, naming the file to download, if there is none.
pub fn official(n_g1: usize) -> PathBuf {
    let dir = dir();
    let fits = POWERS.clone().filter(|&power| g1_powers(power) >= n_g1);
    for power in fits.clone() {
        let path = dir.join(file_name(power));
        if path.is_file() {
            return path;
        }
    }
    match fits.min() {
        Some(power) => panic!(
            "no official ptau of at least {n_g1} powers [τ^i]₁ in {} (PILFFLONK_PTAU_DIR): run {DOWNLOAD} {power}",
            dir.display()
        ),
        None => panic!("no official ptau holds {n_g1} powers [τ^i]₁"),
    }
}

/// Where the contents of section `id` of the binfile at `path` start, and their size.
pub fn section(path: &Path, id: u32) -> (u64, u64) {
    let mut file = fs::File::open(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    let mut header = [0u8; 12];
    file.read_exact(&mut header).unwrap();
    let sections = u32::from_le_bytes(header[8..12].try_into().unwrap());
    let mut at = 12u64;
    for _ in 0..sections {
        let mut entry = [0u8; 12];
        file.seek(SeekFrom::Start(at)).unwrap();
        file.read_exact(&mut entry).unwrap();
        let size = u64::from_le_bytes(entry[4..12].try_into().unwrap());
        if u32::from_le_bytes(entry[..4].try_into().unwrap()) == id {
            return (at + 12, size);
        }
        at += 12 + size;
    }
    panic!("{} has no section {id}", path.display())
}

/// `n` bytes of section `id` of the binfile at `path`, from `offset` on.
pub fn read_section(path: &Path, id: u32, offset: u64, n: usize) -> Vec<u8> {
    let (start, size) = section(path, id);
    assert!(offset + n as u64 <= size, "{}: section {id} has {size} bytes", path.display());
    let mut file = fs::File::open(path).unwrap();
    file.seek(SeekFrom::Start(start + offset)).unwrap();
    let mut bytes = vec![0u8; n];
    file.read_exact(&mut bytes).unwrap();
    bytes
}

/// `[τ^i]₁` of the ptau at `path`, `x‖y` as it stores them.
pub fn g1_power(path: &Path, i: usize) -> Vec<u8> {
    read_section(path, 2, (i * G1_BYTES) as u64, G1_BYTES)
}

/// `[τ^i]₂` of the ptau at `path`, `x.c0‖x.c1‖y.c0‖y.c1` as it stores them.
pub fn g2_power(path: &Path, i: usize) -> Vec<u8> {
    read_section(path, 3, (i * G2_BYTES) as u64, G2_BYTES)
}

/// A copy of the ptau at `from` at `to`, with `bytes` written over section `id` from `offset` on.
pub fn patched_copy(from: &Path, to: &Path, id: u32, offset: u64, bytes: &[u8]) {
    let (start, size) = section(from, id);
    assert!(offset + bytes.len() as u64 <= size, "{}: section {id} has {size} bytes", from.display());
    let mut contents = fs::read(from).unwrap();
    let at = (start + offset) as usize;
    contents[at..at + bytes.len()].copy_from_slice(bytes);
    fs::write(to, contents).unwrap();
}

/// A copy of the ptau at `from` at `to` with only its first `n_g1` powers `[τ^i]₁`: section 2 cut
/// short, the others as they are. A ptau with fewer powers than any official one, for the tests of
/// a ptau too small.
pub fn truncated_copy(from: &Path, to: &Path, n_g1: usize) {
    let contents = fs::read(from).unwrap();
    let n_sections = u32::from_le_bytes(contents[8..12].try_into().unwrap());
    let mut out = contents[..12].to_vec();
    let mut at = 12;
    for _ in 0..n_sections {
        let id = u32::from_le_bytes(contents[at..at + 4].try_into().unwrap());
        let size = u64::from_le_bytes(contents[at + 4..at + 12].try_into().unwrap()) as usize;
        let kept = if id == 2 { n_g1 * G1_BYTES } else { size };
        assert!(kept <= size, "{} holds {} powers [τ^i]₁", from.display(), size / G1_BYTES);
        out.extend(id.to_le_bytes());
        out.extend((kept as u64).to_le_bytes());
        out.extend(&contents[at + 12..at + 12 + kept]);
        at += 12 + size;
    }
    fs::write(to, out).unwrap();
}

/// A copy of the ptau at `from` at `to` whose `[τ]₂` is its `[τ²]₂`: an SRS whose `[τ^i]₁` are
/// those of `from`, and whose `[τ]₂`, a point of G2 all the same, is not their `τ`'s. A key set up
/// with it has the commitments of `from`'s, and another `X_2`: its pairing refuses their proofs.
pub fn with_another_tau_in_g2(from: &Path, to: &Path) {
    patched_copy(from, to, 3, G2_BYTES as u64, &g2_power(from, 2));
}
