//! A ProofMan that proves, wraps its proof in the final SNARK and proves again, as ZisK's prover does
//! (`prove`, `prove -s`, `prove` in one process): the wrap writes proofman's buffers (the
//! recursivef and the SNARK on the GPU, the fixed pols it reloads), and the proof after it must
//! verify (`verify_proofs`) with the publics of the one before it, and wrap in a SNARK that verifies.
//! Not byte for byte: two vadcop_final proofs of one input differ, in two processes too.
//!
//! It takes the fibonacci-square build of `setup`, `gen-custom-commits-fixed`, `setup-recursivef`
//! (blake3) and `setup-snark` from `SNARK_REUSE_BUILD`: `provingKey/`, `provingKeySnark/` and
//! `rom.bin`, and its witness library from a release build (`cargo build --release -p
//! fibonacci-square`). Without a GPU it says so and passes; with `PILFFLONK_GPU=1` a missing GPU
//! fails.
//!
//! ```text
//! SNARK_REUSE_BUILD=<build> cargo test --release -p proofman-cli --test snark_reuse -- --ignored
//! ```

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use proofman::{verify_snark_proof, ProofMan, ProvePhase, ProvePhaseInputs, ProvePhaseResult, SnarkWrapper};
use proofman_common::{ProofOptions, ProofmanOptions, VerboseMode};
use proofman_fields::Goldilocks;
use proofman_pilfflonk::gpu_available;
use proofman_verifier::VadcopFinalProof;
use proofman_witness::WitnessLibInitFn;

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("..").canonicalize().unwrap()
}

#[test]
#[ignore = "needs SNARK_REUSE_BUILD and a GPU (see the module)"]
fn a_proof_after_the_wrap_verifies_and_wraps_again() {
    if !gpu_available() {
        assert_ne!(std::env::var("PILFFLONK_GPU").as_deref(), Ok("1"), "PILFFLONK_GPU=1, and there is no GPU");
        eprintln!("skipping the reuse test: no GPU");
        return;
    }
    let build = PathBuf::from(std::env::var_os("SNARK_REUSE_BUILD").expect("SNARK_REUSE_BUILD (see the module)"));
    let (proving_key, proving_key_snark) = (build.join("provingKey"), build.join("provingKeySnark"));
    let witness_lib = repo_root().join("target/release/libfibonacci_square.so");
    let inputs = repo_root().join("examples/fibonacci-square/src/inputs.json");
    let custom_commits = HashMap::from([("rom".to_string(), build.join("rom.bin"))]);
    let vkey = proving_key_snark.join("final/provingKey/final/pilfflonk/pilfflonk.vkey.json");

    let mut options = ProofmanOptions::new();
    options.gpu();
    options.final_snark();
    options.custom_commits_fixed(custom_commits.clone());
    let proofman = ProofMan::<Goldilocks>::new(proving_key, options).unwrap();
    proofman.register_custom_commits(custom_commits).unwrap();
    // On proofman's buffers and preloaded, as `prove -s` and ZisK make it.
    let (aux_trace, d_buffers, reload_fixed_pols_gpu) = proofman.get_preallocated_buffers();
    let wrapper = SnarkWrapper::<Goldilocks>::new_with_preallocated_buffers(
        &proving_key_snark,
        VerboseMode::Info,
        Some(aux_trace),
        Some(d_buffers),
        Some(reload_fixed_pols_gpu),
        true,
        true,
    )
    .unwrap();

    // The witness library registered once, as ZisK does: `generate_proof` registers it at each call.
    let library = unsafe { libloading::Library::new(&witness_lib) }.unwrap();
    let init: libloading::Symbol<WitnessLibInitFn<Goldilocks>> = unsafe { library.get(b"init_library") }.unwrap();
    let mut lib = init(VerboseMode::Info, Some(proofman.get_rank_info())).unwrap();
    proofman.get_wcm().set_public_inputs_path(Some(inputs));
    proofman.register_witness(&mut *lib, library).unwrap();

    let prove = || -> VadcopFinalProof {
        proofman.set_barrier();
        proofman.set_partition(1, vec![0], 0).unwrap();
        // Aggregated to the vadcop_final, and each proof verified as it is made.
        let options = ProofOptions::new(false, true, true, false, true, false);
        match proofman.generate_proof_from_lib(ProvePhaseInputs::Full(), options, ProvePhase::Full) {
            Ok(ProvePhaseResult::Full(_, Some(proof))) => proof,
            other => panic!("no vadcop_final proof: {:?}", other.err()),
        }
    };
    let wrap = |proof: &VadcopFinalProof| {
        let snark = proofman.generate_final_snark_proof(&wrapper, proof, None).unwrap();
        verify_snark_proof(&snark, &vkey).unwrap();
    };

    let before = prove();
    wrap(&before);
    let after = prove();
    assert_eq!(before.public_values, after.public_values);
    wrap(&after);
}
