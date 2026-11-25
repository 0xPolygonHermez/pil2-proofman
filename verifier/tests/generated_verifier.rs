// A generated verifier (the setup's `vadcop_final.verifier.rs`, or one rendered from the DAG) against a real proof.
// Run: PIL2_GENERATED_VERIFIER=<abs .rs> PIL2_PROOF=<vadcop_final_proof.bin> PIL2_VERKEY=<vadcop_final.verkey.bin> \
//   cargo test --release -p proofman-verifier --features generated-verifier-test --test generated_verifier
#![cfg(feature = "generated-verifier-test")]

extern crate alloc;
use proofman_verifier::*;

mod gen {
    include!(env!("PIL2_GENERATED_VERIFIER"));
}

fn load() -> (VadcopFinalProof, Vec<u64>) {
    let proof = VadcopFinalProof::load(std::env::var("PIL2_PROOF").expect("set PIL2_PROOF")).unwrap();
    let bytes = std::fs::read(std::env::var("PIL2_VERKEY").expect("set PIL2_VERKEY")).unwrap();
    let vk = bytes.chunks_exact(8).map(|c| u64::from_le_bytes(c.try_into().unwrap())).collect();
    (proof, vk)
}

/// The real proof verifies and a tampered one does not; with PIL2_EXPECT_REJECT=1 the verifier was rendered from a
/// changed DAG (pil2-stark-setup's `render_rust_verifier` with MUTATE_CONST=1) and must reject the real proof.
#[test]
fn real_tampered_and_changed_dag() {
    let (mut proof, vk) = load();
    if std::env::var("PIL2_EXPECT_REJECT").is_ok_and(|v| v == "1") {
        assert!(!gen::verify(&proof, &vk), "a verifier from a changed DAG must reject the real proof");
        return;
    }
    assert!(gen::verify(&proof, &vk), "the real proof must verify");
    // the first evals word follows the n_stages + 1 roots; the generator emits nStages = 2 only
    proof.proof[4 * 3] = proof.proof[4 * 3].wrapping_add(1);
    assert!(!gen::verify(&proof, &vk), "a tampered evals word must be rejected");
}

#[test]
#[ignore = "timing: -- --ignored --nocapture"]
fn time_verify_100() {
    let (proof, vk) = load();
    let t = std::time::Instant::now();
    for _ in 0..100 {
        assert!(gen::verify(&proof, &vk));
    }
    println!("mean verify: {:?}", t.elapsed() / 100);
}
