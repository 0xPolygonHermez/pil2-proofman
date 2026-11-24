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

// PIL2_EXPECT_REJECT=1: the verifier was rendered from a changed DAG (pil2-stark-setup's
// `writes_a_verifier_from_a_changed_dag`) and must reject the real proof.
fn expect_reject() -> bool {
    std::env::var("PIL2_EXPECT_REJECT").is_ok_and(|v| v == "1")
}

#[test]
fn accepts_real_and_rejects_tampered() {
    if expect_reject() {
        return;
    }
    let (mut proof, vk) = load();
    assert!(gen::verify(&proof, &vk), "the real proof must verify");
    // proof.proof = [4*(n_stages+1) roots][evals ...], n_stages = 2: word 12 is the first evals word
    proof.proof[4 * 3] = proof.proof[4 * 3].wrapping_add(1);
    assert!(!gen::verify(&proof, &vk), "a tampered evals word must be rejected");
}

#[test]
fn a_changed_dag_rejects_the_real_proof() {
    if !expect_reject() {
        return;
    }
    let (proof, vk) = load();
    assert!(!gen::verify(&proof, &vk), "a verifier from a changed DAG must reject the real proof");
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
