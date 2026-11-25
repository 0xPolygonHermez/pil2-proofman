// Render a native Rust verifier from a circuit's starkinfo.json + verifierinfo.json with the setup's codegen
// (`output::verifier::write_verifier_rust_file`). Ignored by default and env-driven, so it can be pointed at any
// proving key without re-running the recursive setup. MUTATE_CONST=1 changes one non-zero DAG constant first: the
// rendered verifier must then reject the real proof (verifier/tests/generated_verifier.rs, PIL2_EXPECT_REJECT=1).
//
//   STARKINFO=<circuit.starkinfo.json> VERIFIERINFO=<circuit.verifierinfo.json> \
//   OUT=<out.rs> [HASH=blake3] [MUTATE_CONST=1] \
//     cargo test --test render_rust_verifier -- --ignored
use pil2_stark_setup::output::verifier::write_verifier_rust_file;
use pil2_stark_setup::types::stark_info::{StarkInfo, VerifierInfo};
use pil_info::core::dag::format::DagNode;

#[test]
#[ignore = "env-driven: set STARKINFO, VERIFIERINFO and OUT"]
fn render_rust_verifier_from_files() {
    let env = |k: &str| std::env::var(k).unwrap_or_else(|_| panic!("set {k}"));
    let json = |k: &str| -> serde_json::Value {
        serde_json::from_str(&std::fs::read_to_string(env(k)).expect("read")).expect("parse")
    };
    let stark_info = StarkInfo::from_json(&json("STARKINFO")).expect("StarkInfo::from_json");
    let mut verifier_info = VerifierInfo::from_json(&json("VERIFIERINFO")).expect("VerifierInfo::from_json");
    if std::env::var("MUTATE_CONST").is_ok_and(|v| v == "1") {
        let c = verifier_info.constraint_dag.nodes.iter_mut().find_map(|n| match n {
            DagNode::Const { value } if value != "0" => Some(value),
            _ => None,
        });
        *c.expect("the DAG has a non-zero constant") = "1234567".into();
    }
    let hash = std::env::var("HASH").unwrap_or_else(|_| "blake3".to_string());
    let out = env("OUT");
    write_verifier_rust_file(&out, &stark_info, &verifier_info, &hash).expect("render");
    println!("wrote {out}");
}
