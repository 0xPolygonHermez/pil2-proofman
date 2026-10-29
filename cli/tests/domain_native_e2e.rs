//! The native Rust verifier on a domain proof (`examples/domain-native`), CPU only: setup, emit the AIR's
//! native verifier, prove it with `prove-air` (self-contained), then run the emitted verifier, compiled in
//! a scratch crate, on honest and tampered proofs.
//!
//! ```text
//! cargo test --release -p proofman-cli --test domain_native_e2e
//! ```

#[path = "../../pilfflonk/tests/data/witness_libraries.rs"]
mod witness_libraries;

use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};

use pil2_stark_setup::commands::setup::{run_setup, SetupOptions};
use pil2_stark_setup::output::verifier::write_verifier_rust_file;
use pil2_stark_setup::types::stark_info::{StarkInfo, VerifierInfo};
use proofman_common::hash_family::{compressed_final_by_default, default_aggregation_arity, DEFAULT_HASH_ID};
use serde_json::Value;

const AIR: &str = "DomNative";

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("..").canonicalize().unwrap()
}

fn tmp(name: &str) -> PathBuf {
    let dir = Path::new(env!("CARGO_TARGET_TMPDIR")).join(format!("domain_native_e2e_{name}"));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();
    dir
}

fn output(out: &Output) -> String {
    format!("{}{}", String::from_utf8_lossy(&out.stdout), String::from_utf8_lossy(&out.stderr))
}

fn setup(build: &Path) -> PathBuf {
    run_setup(&SetupOptions {
        airout_path: repo_root().join("examples/domain-native/pil/build.pilout").display().to_string(),
        build_dir: build.display().to_string(),
        fixed_dir: None,
        stark_structs_path: None,
        recursive: false,
        recursive_jobs: 1,
        setup_jobs: 1,
        stats_output_path: None,
        hash: DEFAULT_HASH_ID.to_string(),
        agg_arity: default_aggregation_arity(DEFAULT_HASH_ID),
        compressed_final: compressed_final_by_default(DEFAULT_HASH_ID),
        setup_version: None,
        recursive_n_bits: None,
        gen_exps: false,
        exps_arch: "auto".to_string(),
        exps_cap: 60000,
        exps_chunk: None,
        exps_stark_src: None,
    })
    .unwrap_or_else(|e| panic!("setup failed: {e:#}"));
    build.join("provingKey")
}

fn find_file(dir: &Path, name: &str) -> Option<PathBuf> {
    for entry in fs::read_dir(dir).ok()? {
        let path = entry.ok()?.path();
        if path.is_dir() {
            if let Some(found) = find_file(&path, name) {
                return Some(found);
            }
        } else if path.file_name().is_some_and(|f| f == name) {
            return Some(path);
        }
    }
    None
}

fn json(path: &Path) -> Value {
    serde_json::from_str(&fs::read_to_string(path).unwrap()).unwrap()
}

/// A binary crate around the emitted verifier: `harness <verkey.bin> <proof.bin>` exits 0 iff it verifies.
fn build_native_verifier(key: &Path, dir: &Path) -> PathBuf {
    let starkinfo = find_file(key, &format!("{AIR}.starkinfo.json")).unwrap();
    let files = starkinfo.parent().unwrap();
    let stark_info = StarkInfo::from_json(&json(&starkinfo)).unwrap();
    let verifier_info = VerifierInfo::from_json(&json(&files.join(format!("{AIR}.verifierinfo.json")))).unwrap();
    fs::create_dir_all(dir.join("src")).unwrap();
    let rs = dir.join("src/verifier.rs");
    write_verifier_rust_file(rs.to_str().unwrap(), &stark_info, &verifier_info, false, DEFAULT_HASH_ID).unwrap();

    let root = repo_root();
    fs::write(
        dir.join("Cargo.toml"),
        format!(
            "[package]\nname = \"native_harness\"\nversion = \"0.1.0\"\nedition = \"2021\"\n\n[workspace]\n\n\
             [dependencies]\nproofman-verifier = {{ path = \"{}\" }}\nproofman-fields = {{ path = \"{}\" }}\n\
             tracing-subscriber = \"0.3.19\"\n",
            root.join("verifier").display(),
            root.join("fields").display()
        ),
    )
    .unwrap();
    // The workspace's lock, so the offline build resolves to what is already downloaded.
    fs::copy(root.join("Cargo.lock"), dir.join("Cargo.lock")).unwrap();
    fs::write(
        dir.join("src/main.rs"),
        r#"extern crate alloc;
pub use proofman_verifier::*;
#[allow(dead_code)]
mod verifier;

fn words(path: &str) -> Vec<u64> {
    std::fs::read(path).unwrap().chunks_exact(8).map(|c| u64::from_le_bytes(c.try_into().unwrap())).collect()
}

fn main() {
    tracing_subscriber::fmt().with_ansi(false).init();
    let args: Vec<String> = std::env::args().collect();
    let ok = verifier::verify(&words(&args[2]), &words(&args[1]));
    println!("native verify: {ok}");
    std::process::exit(if ok { 0 } else { 1 });
}
"#,
    )
    .unwrap();

    let target = Path::new(env!("CARGO_TARGET_TMPDIR")).join("domain_native_harness_target");
    let cargo = std::env::var("CARGO").unwrap_or_else(|_| "cargo".into());
    let out = Command::new(cargo)
        .args(["build", "--release", "--offline", "--manifest-path"])
        .arg(dir.join("Cargo.toml"))
        .env("CARGO_TARGET_DIR", &target)
        .output()
        .unwrap();
    assert!(out.status.success(), "the emitted verifier does not build:\n{}", output(&out));
    target.join("release/native_harness")
}

/// `prove-air` of DomNative on the CPU, saved to `proof`; `tamper` is `<col>:<row>:<value>`.
fn prove(key: &Path, proof: &Path, tamper: Option<&str>) -> Output {
    let lib = witness_libraries::built_library("domain_native");
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_proofman-cli"));
    cmd.args(["prove-air", "--witness-lib", lib.to_str().unwrap(), "--air", AIR, "--proving-key"])
        .arg(key)
        .arg("--save-proof")
        .arg(proof)
        .env("NO_COLOR", "1")
        .env_remove("DOMAIN_NATIVE_TAMPER");
    if let Some(tamper) = tamper {
        // the C++ verifier would refuse to save it
        cmd.arg("--no-verify").env("DOMAIN_NATIVE_TAMPER", tamper);
    }
    cmd.output().unwrap()
}

fn native_verify(harness: &Path, key: &Path, proof: &Path) -> Output {
    let verkey = find_file(key, &format!("{AIR}.verkey.bin")).unwrap();
    Command::new(harness).arg(verkey).arg(proof).output().unwrap()
}

#[test]
fn native_verifier_checks_domain_constraints() {
    let dir = tmp("run");
    let key = setup(&dir.join("build"));
    let harness = build_native_verifier(&key, &dir.join("harness"));

    let honest = dir.join("honest.bin");
    let out = prove(&key, &honest, None);
    assert!(out.status.success(), "prove-air failed:\n{}", output(&out));
    let out = native_verify(&harness, &key, &honest);
    assert!(out.status.success(), "the native verifier rejects an honest proof:\n{}", output(&out));

    // a is PHASE on !Q1 (row 1) and b is 0 on ODD (row 3): each breaks one domain constraint
    for tamper in ["a:1:9", "b:3:5"] {
        let proof = dir.join(format!("inside_{}.bin", tamper.replace(':', "_")));
        let out = prove(&key, &proof, Some(tamper));
        assert!(out.status.success(), "prove-air with {tamper} failed:\n{}", output(&out));
        let out = native_verify(&harness, &key, &proof);
        let text = output(&out);
        assert_eq!(out.status.code(), Some(1), "{tamper}: not rejected:\n{text}");
        assert!(
            text.contains("Quotient polynomial verification failed"),
            "{tamper}: rejected for another reason:\n{text}"
        );
    }

    // row 4 is on Q1 and even: outside both domains
    let outside = dir.join("outside.bin");
    let out = prove(&key, &outside, Some("a:4:9"));
    assert!(out.status.success(), "prove-air with a:4:9 failed:\n{}", output(&out));
    let out = native_verify(&harness, &key, &outside);
    assert!(out.status.success(), "a cell outside every domain is rejected:\n{}", output(&out));
}
