//! What the BN254 tests share: the committed circom, the snarkjs that computes a circuit's witness,
//! a scratch directory of their own and the reader of a `.wtns` file.
//!
//! A circuit is compiled with the committed circom (`setup/circom`), so its r1cs cannot go stale,
//! and its witness is computed by the circom-generated wasm and the snarkjs of
//! `setup/pil2-stark/node_modules` (`npm install` there). Without Node.js or that snarkjs a test
//! says why and passes, as the circom tests of `stark2circom` do.

// Each test crate that includes it uses some of it.
#![allow(dead_code)]

use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{SystemTime, UNIX_EPOCH};

use pil2_stark_recurser::plonk2pil::field::PlonkField;

pub fn manifest() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

pub fn repo_root() -> PathBuf {
    manifest().join("../..")
}

/// The committed circom, as the plonk2pil golden picks it.
pub fn circom() -> PathBuf {
    repo_root().join("setup/circom").join(if cfg!(target_os = "macos") { "circom_mac" } else { "circom" })
}

/// The circom library of the BN254 verifier, whose `custom/` has the `PoseidonT` gate.
pub fn circuits_bn128() -> PathBuf {
    manifest().join("stark2circom/circom_verifier/circuits.bn128")
}

pub fn snarkjs() -> PathBuf {
    repo_root().join("setup/pil2-stark/node_modules/snarkjs/build/cli.cjs")
}

/// What the witness needs and is missing, if anything.
pub fn missing_prerequisite() -> Option<String> {
    let node = Command::new("node").arg("--version").output().map(|o| o.status.success()).unwrap_or(false);
    if !node {
        return Some("node not on PATH".into());
    }
    if !snarkjs().is_file() {
        return Some(format!("{} not present (npm install in setup/pil2-stark)", snarkjs().display()));
    }
    None
}

/// A directory of the test's own under the temporary directory, removed when dropped.
pub struct Scratch(pub PathBuf);

impl Scratch {
    pub fn new(name: &str) -> Self {
        let nanos = SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_nanos()).unwrap_or_default();
        let dir = std::env::temp_dir().join(format!("{name}_{}_{nanos}", std::process::id()));
        fs::create_dir_all(&dir).unwrap_or_else(|e| panic!("{}: {e}", dir.display()));
        Self(dir)
    }

    pub fn file(&self, name: &str) -> PathBuf {
        self.0.join(name)
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        // Best effort: a leftover directory in the temporary directory harms nothing.
        let _ = fs::remove_dir_all(&self.0);
    }
}

pub fn run(cmd: &mut Command, what: &str) {
    let out = cmd.output().unwrap_or_else(|e| panic!("{what}: {e}"));
    assert!(
        out.status.success(),
        "{what} failed:\n{}\n{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
}

/// Compiles `src` for BN254 into `dir`, with the BN254 verifier's circom library on the include
/// path, and computes its witness of `input`: the r1cs and the `.wtns`.
pub fn compile_and_witness(dir: &Path, src: &Path, input: &Path) -> (Vec<u8>, Vec<u8>) {
    let name = src.file_stem().and_then(|s| s.to_str()).expect("a circuit file name");
    // --O1 keeps the linear constraints, which are what reach plonk2pil's sum gates.
    run(
        Command::new(circom())
            .args(["--O1", "--r1cs", "--wasm", "--prime", "bn128", "-l"])
            .arg(circuits_bn128())
            .arg(src)
            .arg("-o")
            .arg(dir),
        "circom",
    );
    let wtns = dir.join(format!("{name}.wtns"));
    run(
        Command::new("node")
            .arg(snarkjs())
            .args(["wtns", "calculate"])
            .arg(dir.join(format!("{name}_js/{name}.wasm")))
            .arg(input)
            .arg(&wtns),
        "snarkjs wtns calculate",
    );
    let read = |p: PathBuf| fs::read(&p).unwrap_or_else(|e| panic!("{}: {e}", p.display()));
    (read(dir.join(format!("{name}.r1cs"))), read(wtns))
}

/// The witness in a `.wtns` file, as snarkjs's `wtns_utils.js` writes one: the magic `wtns`, a
/// version and a section count, then sections of `(type: u32, size: u64, data)`. Section 1 is
/// `n8: u32`, the prime in `n8` bytes and the witness length `u32`; section 2 the values, `n8` bytes
/// each, canonical and little-endian.
pub fn read_wtns<F: PlonkField>(data: &[u8]) -> Vec<F> {
    let u32_at = |at: usize| u32::from_le_bytes(data[at..at + 4].try_into().unwrap()) as usize;
    let u64_at = |at: usize| u64::from_le_bytes(data[at..at + 8].try_into().unwrap()) as usize;
    assert_eq!(&data[..4], b"wtns", "not a .wtns file");
    let mut sections = std::collections::HashMap::new();
    let mut at = 12;
    for _ in 0..u32_at(8) {
        let (kind, size) = (u32_at(at), u64_at(at + 4));
        sections.insert(kind, &data[at + 12..at + 12 + size]);
        at += 12 + size;
    }

    let header = sections[&1];
    let n8 = u32::from_le_bytes(header[..4].try_into().unwrap()) as usize;
    assert_eq!(header[4..4 + n8], F::PRIME.modulus_le(), "the witness is not over {}", F::PRIME);
    let n = u32::from_le_bytes(header[4 + n8..8 + n8].try_into().unwrap()) as usize;
    let values = sections[&2];
    assert_eq!(values.len(), n * n8);
    values.chunks_exact(n8).map(|v| F::from_canonical_le(v).expect("a canonical witness value")).collect()
}
