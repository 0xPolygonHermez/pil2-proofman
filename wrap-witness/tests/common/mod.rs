//! What the tests of the wrap's witness share: a final circuit's files laid out as setup-snark lays
//! out `provingKeySnark/final/`, `final/final.{so,dat}`, and the reference, snarkjs's witness.
//!
//! A circuit is compiled with the committed circom (`setup/circom`), for BN254, as setup-snark
//! compiles the final circuit (snark_setup.rs), with the wasm for snarkjs; its witness calculator is
//! built as setup-snark builds `final.so`, with `WitnessTracker` and the Makefile of
//! `setup/final_snark_circom/`. The reference is the witness of the same input that snarkjs of
//! `setup/pil2-stark/node_modules` (`npm install` there) computes from the wasm. Without Node.js or
//! that snarkjs a test says why and passes, as plonk2pil's BN254 tests do.

// Each test crate that includes it uses some of it.
#![allow(dead_code)]

use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

use pil2_stark_setup::output::witness_gen::WitnessTracker;
use proofman_fields::Bn254;

pub fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("..").canonicalize().expect("the repository root")
}

/// A file of plonk2pil's BN254 fixtures.
pub fn fixture(name: &str) -> PathBuf {
    repo_root().join("setup/stark-recurser/tests/fixtures/bn254").join(name)
}

/// The circom library of the BN254 verifier, whose `custom/` has the wrap's custom gates.
pub fn circuits_bn128() -> PathBuf {
    repo_root().join("setup/stark-recurser/stark2circom/circom_verifier/circuits.bn128")
}

/// The committed circom, as plonk2pil's BN254 test picks it.
pub fn circom() -> PathBuf {
    repo_root().join("setup/circom").join(if cfg!(target_os = "macos") { "circom_mac" } else { "circom" })
}

pub fn snarkjs() -> PathBuf {
    repo_root().join("setup/pil2-stark/node_modules/snarkjs/build/cli.cjs")
}

/// What the reference witness needs and is missing, if anything.
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

pub fn run(cmd: &mut Command, what: &str) {
    let out = cmd.output().unwrap_or_else(|e| panic!("{what}: {e}"));
    assert!(
        out.status.success(),
        "{what} failed:\n{}\n{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
}

/// Compiles `circuit` into `dir/build`, with `libraries` on circom's include path, and lays its
/// files out in `dir/final/` as setup-snark does: `final.so` (`.dylib` on macOS) and `final.dat`.
/// The circuit's r1cs.
pub fn build_final(dir: &Path, circuit: &Path, libraries: &[PathBuf]) -> Vec<u8> {
    let name = circuit.file_stem().and_then(|s| s.to_str()).expect("a circuit file name");
    let (build, files) = (dir.join("build"), dir.join("final"));
    fs::create_dir_all(&build).unwrap();
    // --O1 keeps the linear constraints, which plonk2pil's sum gates take.
    let mut compile = Command::new(circom());
    compile.args(["--O1", "--r1cs", "--c", "--wasm", "--prime", "bn128"]);
    for library in libraries {
        compile.arg("-l").arg(library);
    }
    run(compile.arg(circuit).arg("-o").arg(&build), "circom");

    let helpers = repo_root().join("setup/final_snark_circom");
    let tracker = WitnessTracker::new();
    tracker.run_witness_library_generation(
        dir.to_str().unwrap(),
        files.to_str().unwrap(),
        name,
        "final",
        helpers.to_str().unwrap(),
    );
    tracker.await_all().expect("final.so builds");
    fs::copy(build.join(format!("{name}_cpp/{name}.dat")), files.join("final.dat")).unwrap();
    fs::read(build.join(format!("{name}.r1cs"))).unwrap()
}

/// snarkjs's witness of `input` for the circuit `name` that [`build_final`] built in `dir`, from
/// the circom wasm: a value per witness index, wire 0 the constant one.
pub fn snarkjs_witness(dir: &Path, name: &str, input: &Path) -> Vec<Bn254> {
    let (wtns, json) = (dir.join(format!("{name}.wtns")), dir.join(format!("{name}.wtns.json")));
    let node = |args: &[&str], paths: &[&Path], what: &str| {
        run(Command::new("node").arg(snarkjs()).args(args).args(paths), what);
    };
    let wasm = dir.join(format!("build/{name}_js/{name}.wasm"));
    node(&["wtns", "calculate"], &[&wasm, input, &wtns], "snarkjs wtns calculate");
    node(&["wtns", "export", "json"], &[&wtns, &json], "snarkjs wtns export json");
    let values: Vec<String> = serde_json::from_slice(&fs::read(&json).unwrap()).unwrap();
    values.iter().map(|v| Bn254::from_decimal(v).expect("a canonical value")).collect()
}
