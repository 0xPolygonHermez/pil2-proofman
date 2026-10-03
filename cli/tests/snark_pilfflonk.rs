//! `prove-snark` and `verify-snark` with pilfflonk as the final SNARK: a `SnarkProof` of protocol
//! pilfflonk (`proofman::PILFFLONK_PROTOCOL_ID`), its bytes pilfflonk's (pilfflonk/docs/formats.md#proof)
//! and its public the final circuit's publics hash, verified by pilfflonk's JS verifier.
//!
//! - `verify_snark_accepts_a_pilfflonk_proof_and_refuses_every_change`: a pilfflonk proof of the
//!   Fibonacci fixture, set up with the full-width `τ` of the tests (pilfflonk/docs/README.md#tests),
//!   in a `snark_proof.bin`: it is read back as it was written, its JSON views are pilfflonk's, and
//!   `verify-snark` accepts it, and refuses it with a byte of the proof changed, with a public
//!   changed, cut short, and against the vkey of a setup with another `τ`. It compiles the fixture
//!   with the compiler `PIL2C_EXEC` names, which must honour `prime`, and needs Node.js.
//! - `prove_snark_wraps_a_vadcop_final_proof_in_pilfflonk`: the wrap of a real vadcop_final proof,
//!   `prove-snark` on the `provingKeySnark/` of `setup-snark --final-snark pilfflonk`, and
//!   `verify-snark` on its proof, with the same refusals; the publics the Solidity verifier hashes
//!   are the vadcop_final proof's, as for PLONK and FFLONK. It takes its inputs from the
//!   environment: `PROVE_SNARK_PROVING_KEY`, the `provingKeySnark/` (the recursivef's `.consttree`
//!   is written there if it is not), `PROVE_SNARK_PROOF`, the `vadcop_final_proof.bin`, and
//!   `PROVE_SNARK_OTHER_VKEY`, a pilfflonk vkey of another key.
//!
//! Both are `#[ignore]`d:
//!
//! ```text
//! PIL2C_EXEC=<pil2-compiler>/src/pil.js cargo test --release -p proofman-cli \
//!     --features proofman-starks-lib-c/cpu-only --test snark_pilfflonk \
//!     -- --ignored verify_snark_accepts_a_pilfflonk_proof_and_refuses_every_change
//! PROVE_SNARK_PROVING_KEY=<provingKeySnark> PROVE_SNARK_PROOF=<vadcop_final_proof.bin> \
//!     PROVE_SNARK_OTHER_VKEY=<pilfflonk.vkey.json> cargo test --release -p proofman-cli \
//!     --features proofman-starks-lib-c/cpu-only --test snark_pilfflonk \
//!     -- --ignored prove_snark_wraps_a_vadcop_final_proof_in_pilfflonk
//! ```

#[path = "../../pilfflonk/tests/data/fibonacci.rs"]
mod fibonacci;

use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};

use num_bigint::BigUint;
use pilfflonk_setup::command::{DEFAULT_EXTRA_MULS, DEFAULT_MAX_CONSTRAINT_DEGREE, DEFAULT_MAX_Q_DEGREE, PROVING_KEY_DIR};
use pilfflonk_setup::test_ptau::{test_tau, write_fixed_tau_ptau};
use pilfflonk_setup::{run_setup_pilfflonk, SetupPilfflonkOptions};
use proofman::{get_public_bytes_solidity, verify_snark_proof, SnarkProof, SnarkProtocol, PILFFLONK_PROTOCOL_ID};
use proofman_common::{ProofmanError, PublicsInfo};
use proofman_pilfflonk::{
    prove, FrBytes, JsonFile, PilfflonkGlobalInfo, Proof, ProofJson, ProofNames, ProveOptions, ProvingKey, Vkey,
};
use proofman_verifier::VadcopFinalProof;

/// A fresh directory for the test under the target's temporary directory, removed when dropped.
struct TestDir(PathBuf);

impl TestDir {
    fn new(name: &str) -> Self {
        let dir = Path::new(env!("CARGO_TARGET_TMPDIR")).join(format!("snark_pilfflonk_{name}_{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        TestDir(dir)
    }

    fn file(&self, name: &str) -> PathBuf {
        self.0.join(name)
    }
}

impl Drop for TestDir {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("..").canonicalize().unwrap()
}

/// What a command wrote: the CLI logs to stdout, the verifiers to stderr.
fn output(out: &Output) -> String {
    format!("{}{}", String::from_utf8_lossy(&out.stdout), String::from_utf8_lossy(&out.stderr))
}

fn cli(args: &[&str], paths: &[&Path]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_proofman-cli")).args(args).args(paths).output().expect("proofman-cli runs")
}

/// `proofman-cli verify-snark -p <proof> -k <vkey>`.
fn verify_snark(proof: &Path, vkey: &Path) -> Output {
    cli(&["verify-snark", "-p", proof.to_str().unwrap(), "-k", vkey.to_str().unwrap()], &[])
}

/// Compiles the Fibonacci fixture over BN254 to `pilout` with `PIL2C_EXEC`.
fn compile_fibonacci(pilout: &Path) {
    let compiler = std::env::var("PIL2C_EXEC").expect("PIL2C_EXEC must name a pil2com that honours `prime`");
    let out = Command::new(compiler)
        .current_dir(repo_root())
        .arg("pilfflonk/tests/fixtures/fibonacci/fibonacci.pil")
        .args(["-I", "pil2-components/lib/std/pil", "-P", "pilfflonk/tests/fixtures/fibonacci/bn254.json", "-o"])
        .arg(pilout)
        .output()
        .expect("PIL2C_EXEC runs");
    assert!(out.status.success(), "pil2com: {}", output(&out));
}

/// The pilfflonk key of `pilout` in `build`, set up with a ptau of the fixed `tau`, and its vkey.
fn set_up(pilout: &Path, build: &Path, tau: &BigUint) -> (PathBuf, PathBuf) {
    let ptau = build.with_extension("ptau");
    // More powers than the Fibonacci's largest degree, grouped by default (pilfflonk_verify.rs).
    write_fixed_tau_ptau(&ptau, 1024, tau).unwrap();
    let opts = SetupPilfflonkOptions {
        airout_path: pilout.to_path_buf(),
        build_dir: build.to_path_buf(),
        powers_of_tau: ptau,
        max_constraint_degree: DEFAULT_MAX_CONSTRAINT_DEGREE,
        extra_muls: DEFAULT_EXTRA_MULS,
        max_q_degree: DEFAULT_MAX_Q_DEGREE,
        no_packing: false,
        solidity: false,
    };
    run_setup_pilfflonk(&opts).unwrap();
    let proving_key = build.join(PROVING_KEY_DIR);
    let vkey = vkey_of(&proving_key);
    (proving_key, vkey)
}

/// The vkey of a pilfflonk `provingKey/`.
fn vkey_of(proving_key: &Path) -> PathBuf {
    PilfflonkGlobalInfo::from_proving_key(proving_key).unwrap().vkey_path(proving_key)
}

/// `proof` with the last byte of its first evaluation flipped: a scalar still below `r` (unless it
/// was `r − 1`, which gives `r`, a proof the reader refuses), and no longer the proof's.
fn with_an_evaluation_changed(proof: &SnarkProof, vkey: &Vkey) -> SnarkProof {
    let shape = ProofNames::of_vkey(vkey).unwrap().shape();
    let first_evaluation = (shape.n_commitments + 2) * 64;
    let mut changed = proof.clone();
    changed.proof_bytes[first_evaluation + 31] ^= 1;
    changed
}

/// `proof` with the last byte of its first public flipped.
fn with_a_public_changed(proof: &SnarkProof) -> SnarkProof {
    let mut changed = proof.clone();
    changed.public_snark_bytes[31] ^= 1;
    changed
}

/// `proof` without its last byte.
fn cut_short(proof: &SnarkProof) -> SnarkProof {
    let mut changed = proof.clone();
    changed.proof_bytes.pop();
    changed
}

/// `verify-snark` refuses `proof` against `vkey`, saved in `dir` as `name`: another exit status
/// than 0, and the reason it gives.
fn refused(dir: &TestDir, name: &str, proof: &SnarkProof, vkey: &Path, reason: &str) {
    let path = dir.file(name);
    proof.save(&path).unwrap();
    let out = verify_snark(&path, vkey);
    assert!(!out.status.success(), "{name}: {}", output(&out));
    assert!(output(&out).contains(reason), "{name}: {}", output(&out));
}

#[test]
#[ignore = "needs PIL2C_EXEC and Node.js"]
fn verify_snark_accepts_a_pilfflonk_proof_and_refuses_every_change() {
    let dir = TestDir::new("fibonacci");
    let pilout = dir.file("fibonacci.pilout");
    compile_fibonacci(&pilout);
    let (proving_key, vkey_path) = set_up(&pilout, &dir.file("build"), &test_tau());
    let (_, other_vkey) = set_up(&pilout, &dir.file("other"), &(test_tau() + 1u32));

    let proved = {
        let key = ProvingKey::load(&proving_key).unwrap();
        prove(&key, &fibonacci::witness(8, [1, 2]), &ProveOptions::default()).unwrap()
    };
    let public_snark_bytes = proved.publics.0.iter().flat_map(FrBytes::to_be_bytes).collect();
    let proof = SnarkProof::new(proved.proof.to_bytes(), vec![], public_snark_bytes, PILFFLONK_PROTOCOL_ID);

    // snark_proof.bin is read back as it was written.
    let path = dir.file("snark_proof.bin");
    proof.save(&path).unwrap();
    let back = SnarkProof::load(&path).unwrap();
    assert_eq!(
        (&back.proof_bytes, &back.public_bytes, &back.public_snark_bytes, back.protocol_id),
        (&proof.proof_bytes, &proof.public_bytes, &proof.public_snark_bytes, proof.protocol_id)
    );
    assert!(matches!(SnarkProtocol::from_protocol_id(back.protocol_id), Ok(SnarkProtocol::Pilfflonk)));

    // Its JSON views are the prover's proof.json and publics.json, and give the proof back.
    let vkey = Vkey::read(&vkey_path).unwrap();
    let (proof_json, publics_json) = back.convert_to_json_with_vkey(&vkey_path).unwrap();
    assert_eq!(proof_json, serde_json::to_value(proved.proof_json().unwrap()).unwrap());
    assert_eq!(publics_json, serde_json::to_value(&proved.publics).unwrap());
    let names = ProofNames::of_vkey(&vkey).unwrap();
    let read_json = ProofJson::from_json_str(&proof_json.to_string()).unwrap();
    assert_eq!(Proof::from_json(&read_json, &names).unwrap(), proved.proof);

    let out = verify_snark(&path, &vkey_path);
    assert!(out.status.success(), "{}", output(&out));
    assert!(output(&out).contains("SNARK proof was verified"), "{}", output(&out));
    assert!(verify_snark_proof(&back, &vkey_path).is_ok());

    let not_verified = "SNARK proof was not verified";
    refused(&dir, "evaluation.bin", &with_an_evaluation_changed(&proof, &vkey), &vkey_path, not_verified);
    refused(&dir, "public.bin", &with_a_public_changed(&proof), &vkey_path, not_verified);
    refused(&dir, "other_vkey.bin", &proof, &other_vkey, not_verified);
    refused(&dir, "short.bin", &cut_short(&proof), &vkey_path, "verification failed");
    match verify_snark_proof(&cut_short(&proof), &vkey_path) {
        Err(ProofmanError::InvalidProof(message)) => {
            assert!(message.contains("not those of a proof of the vkey"), "{message}")
        }
        other => panic!("{other:?}"),
    }
}

/// An environment variable the wrap's test takes its inputs from.
fn env_path(name: &str) -> PathBuf {
    PathBuf::from(std::env::var_os(name).unwrap_or_else(|| panic!("{name} must be set (see the module)")))
}

#[test]
#[ignore = "needs a provingKeySnark/ of pilfflonk, a vadcop_final proof and Node.js (see the module)"]
fn prove_snark_wraps_a_vadcop_final_proof_in_pilfflonk() {
    let proving_key_snark = env_path("PROVE_SNARK_PROVING_KEY");
    let vadcop_final_proof = env_path("PROVE_SNARK_PROOF");
    let other_vkey = env_path("PROVE_SNARK_OTHER_VKEY");
    let dir = TestDir::new("wrap");

    let out = cli(
        &["prove-snark", "-p", vadcop_final_proof.to_str().unwrap(), "-k"],
        &[&proving_key_snark, Path::new("-o"), &dir.0],
    );
    assert!(out.status.success(), "{}", output(&out));
    let path = dir.file("snark_proof.bin");
    let proof = SnarkProof::load(&path).unwrap();
    assert_eq!(proof.protocol_id, PILFFLONK_PROTOCOL_ID);

    // One public, the final circuit's publics hash, and as many bytes as a proof of the key has; the
    // publics the Solidity verifier hashes, as for PLONK and FFLONK.
    let vkey_path = vkey_of(&proving_key_snark.join("final").join(PROVING_KEY_DIR));
    let vkey = Vkey::read(&vkey_path).unwrap();
    assert_eq!(vkey.n_public, 1);
    assert_eq!(proof.public_snark_bytes.len(), 32);
    assert_eq!(proof.proof_bytes.len(), ProofNames::of_vkey(&vkey).unwrap().shape().byte_len());
    let vadcop_publics = VadcopFinalProof::load(&vadcop_final_proof).unwrap().proof_with_publics();
    let publics_info = PublicsInfo::from_folder(&proving_key_snark).unwrap();
    let solidity_publics = &vadcop_publics[1..1 + vadcop_publics[0] as usize];
    assert_eq!(proof.public_bytes, get_public_bytes_solidity(&publics_info, solidity_publics).unwrap());

    let out = verify_snark(&path, &vkey_path);
    assert!(out.status.success(), "{}", output(&out));

    let not_verified = "SNARK proof was not verified";
    refused(&dir, "evaluation.bin", &with_an_evaluation_changed(&proof, &vkey), &vkey_path, not_verified);
    refused(&dir, "public.bin", &with_a_public_changed(&proof), &vkey_path, not_verified);
    // Another key's vkey: a proof of its shape that does not verify, or bytes of another shape.
    let other = dir.file("other_vkey.bin");
    proof.save(&other).unwrap();
    let out = verify_snark(&other, &other_vkey);
    assert!(!out.status.success(), "{}", output(&out));
}
