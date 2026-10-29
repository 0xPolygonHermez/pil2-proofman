//! The emitted native Rust verifier on domain proofs (`examples/domain-native`, CPU, `prove-air`).

#[path = "../../pilfflonk/tests/data/witness_libraries.rs"]
mod witness_libraries;
#[path = "common/domains.rs"]
mod common;

use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};

use common::{find_file, output, repo_root, TestDir};
use pil2_stark_setup::output::verifier::write_verifier_rust_file;
use pil2_stark_setup::types::stark_info::{StarkInfo, VerifierInfo};
use proofman_common::hash_family::DEFAULT_HASH_ID;
use serde_json::Value;

const AIR: &str = "DomNative";

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
    common::build_harness(dir, "native_harness", HARNESS_MAIN, "tracing-subscriber = \"0.3.19\"\n")
}

const HARNESS_MAIN: &str = r#"extern crate alloc;
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
"#;

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
    let dir = TestDir::new("domain_native_e2e");
    let pilout = repo_root().join("examples/domain-native/pil/build.pilout");
    let key = common::setup(&pilout, &dir.path("build"), false);
    let harness = build_native_verifier(&key, &dir.path("harness"));

    let honest = dir.path("honest.bin");
    let out = prove(&key, &honest, None);
    assert!(out.status.success(), "prove-air failed:\n{}", output(&out));
    let out = native_verify(&harness, &key, &honest);
    assert!(out.status.success(), "the native verifier rejects an honest proof:\n{}", output(&out));

    // a is PHASE on !Q1 and b is 0 on ODD: each breaks one domain constraint, in the first or a later cycle
    for tamper in ["a:1:9", "b:3:5", "a:1022:9", "b:1021:5"] {
        let proof = dir.path(&format!("inside_{}.bin", tamper.replace(':', "_")));
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
    let outside = dir.path("outside.bin");
    let out = prove(&key, &outside, Some("a:4:9"));
    assert!(out.status.success(), "prove-air with a:4:9 failed:\n{}", output(&out));
    let out = native_verify(&harness, &key, &outside);
    assert!(out.status.success(), "a cell outside every domain is rejected:\n{}", output(&out));
}
