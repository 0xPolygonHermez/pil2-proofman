// Render a native Rust verifier from a circuit's starkinfo.json + verifierinfo.json — the same
// codegen (`output::verifier::write_verifier_rust_file`) that produces the committed files in
// verifier/src/. Ignored by default and driven by env vars, so it can be pointed at any proving
// key to regenerate or cross-check a verifier without re-running the whole recursive setup:
//
//   STARKINFO=<circuit.starkinfo.json> VERIFIERINFO=<circuit.verifierinfo.json> \
//   OUT=<out.rs> [HASH=Poseidon2] \
//     cargo test --test render_rust_verifier -- --ignored
use pil2_stark_setup::output::verifier::write_verifier_rust_file;
use pil2_stark_setup::types::stark_info::{StarkInfo, VerifierInfo};

#[test]
#[ignore = "env-driven: set STARKINFO, VERIFIERINFO and OUT"]
fn render_rust_verifier_from_files() {
    let starkinfo_path = std::env::var("STARKINFO").expect("set STARKINFO to a starkinfo.json");
    let verifierinfo_path = std::env::var("VERIFIERINFO").expect("set VERIFIERINFO to a verifierinfo.json");
    let out_path = std::env::var("OUT").expect("set OUT to the .rs file to write");
    let hash = std::env::var("HASH").unwrap_or_else(|_| "Poseidon2".to_string());

    let starkinfo_json: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&starkinfo_path).expect("read STARKINFO"))
            .expect("parse STARKINFO");
    let stark_info = StarkInfo::from_json(&starkinfo_json).expect("StarkInfo::from_json");

    let verifierinfo_json: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&verifierinfo_path).expect("read VERIFIERINFO"))
            .expect("parse VERIFIERINFO");
    let verifier_info = VerifierInfo::from_json(&verifierinfo_json).expect("VerifierInfo::from_json");

    write_verifier_rust_file(&out_path, &stark_info, &verifier_info, true, &hash).expect("render");
    println!("wrote {out_path}");
}
