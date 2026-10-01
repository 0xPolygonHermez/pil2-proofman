//! `proofman-cli pilfflonk calldata` (pilfflonk/docs/verifier.md#calldata-encoder): the arguments
//! of the Solidity verifier's `verifyProof` for a proof, as snarkjs's
//! `zkey export soliditycalldata` prints them, on the synthetic pilouts of
//! `pilfflonk/tests/data/domains.rs`, built in code: `Domains`, whose `firstRow` and `lastRow` give
//! its calldata two auxiliary inverses, and `Frames`, whose `everyFrame` give none and which has no
//! publics.
//!
//! The keys are set up with the ptau of the full-width `τ` of the C++ test helper
//! (pilfflonk/docs/README.md#tests), and the proofs are the prover's, with a fixed blinding seed;
//! the calldata is checked against the prover's `ξ` and against the selectors solc 0.8.37 gives the
//! signatures (`solc --hashes`). The command must write the same calldata from the proof's JSON
//! view and from its bytes, print it without `-o`, and refuse, with exit code 1, a clear message
//! and no file written, inputs that do not go together. That Foundry accepts the calldata of the
//! proofs of every fixture is `pilfflonk_prove.rs`'s test.
//!
//! It needs neither `PIL2C_EXEC` nor Node.js nor Foundry, and runs in CI.

#[allow(dead_code)]
#[path = "../../pilfflonk/tests/data/domains.rs"]
mod domains;

use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};
use std::sync::{Mutex, MutexGuard, PoisonError};

use num_bigint::BigUint;
use pilfflonk_setup::command::{DEFAULT_EXTRA_MULS, DEFAULT_MAX_CONSTRAINT_DEGREE, DEFAULT_MAX_Q_DEGREE, PROVING_KEY_DIR};
use pilfflonk_setup::test_ptau::{test_tau, write_fixed_tau_ptau};
use pilfflonk_setup::{run_setup_pilfflonk, SetupPilfflonkOptions};
use prost::Message;
use proofman_pilfflonk::{
    prove, Calldata, CalldataLayout, FrBytes, JsonFile, PilfflonkGlobalInfo, ProofOutput, ProveOptions, ProvingKey,
    Vkey, BN254_R,
};
use serde_json::{json, Value};

/// Held by each test while it calls the C++ core in this process (the setup, the prover), which must
/// not run its OpenMP code from several test threads at once (pilfflonk/docs/README.md#tests;
/// `setup/pilfflonk/tests/setup/common.rs`, `cpp_core`).
fn cpp_core() -> MutexGuard<'static, ()> {
    static CPP_CORE: Mutex<()> = Mutex::new(());
    CPP_CORE.lock().unwrap_or_else(PoisonError::into_inner)
}

/// A fresh directory for the test under the target's temporary directory, removed when dropped.
struct TestDir(PathBuf);

impl TestDir {
    fn new(name: &str) -> Self {
        let dir =
            Path::new(env!("CARGO_TARGET_TMPDIR")).join(format!("pilfflonk_calldata_{name}_{}", std::process::id()));
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

/// What a command wrote: the CLI logs to stdout.
fn output(out: &Output) -> String {
    format!("{}{}", String::from_utf8_lossy(&out.stdout), String::from_utf8_lossy(&out.stderr))
}

/// A key of a domains AIR and the prover's proof of its witness, written as `pilfflonk prove`
/// writes it (`proof.json`, `publics.json`), and as its bytes (`proof.bin`).
struct Fixture {
    dir: TestDir,
    vkey_path: PathBuf,
    vkey: Vkey,
    out: ProofOutput,
    proof_json: PathBuf,
    proof_bin: PathBuf,
    publics: PathBuf,
}

fn fixture(name: &str, air: domains::Air) -> Fixture {
    let _cpp = cpp_core();
    let dir = TestDir::new(name);
    let opts = SetupPilfflonkOptions {
        airout_path: dir.file("program.pilout"),
        build_dir: dir.file("build"),
        powers_of_tau: dir.file("fixed_tau.ptau"),
        max_constraint_degree: DEFAULT_MAX_CONSTRAINT_DEGREE,
        extra_muls: DEFAULT_EXTRA_MULS,
        max_q_degree: DEFAULT_MAX_Q_DEGREE,
        no_packing: false,
        solidity: false,
    };
    fs::write(&opts.airout_path, domains::pilout(air).encode_to_vec()).unwrap();
    write_fixed_tau_ptau(&opts.powers_of_tau, 1024, &test_tau()).unwrap();
    run_setup_pilfflonk(&opts).unwrap();
    let proving_key = opts.build_dir.join(PROVING_KEY_DIR);
    let vkey_path = PilfflonkGlobalInfo::from_proving_key(&proving_key).unwrap().vkey_path(&proving_key);
    let pk = ProvingKey::load(&proving_key).unwrap();
    let options = ProveOptions { insecure_blinding_seed: Some([0x41; 32]), ..ProveOptions::default() };
    let out = prove(&pk, &domains::witness(air), &options).unwrap();
    let proof_dir = dir.file("proof");
    out.write(&proof_dir).unwrap();
    let proof_bin = proof_dir.join("proof.bin");
    fs::write(&proof_bin, out.proof.to_bytes()).unwrap();
    let vkey = Vkey::read(&vkey_path).unwrap();
    Fixture {
        vkey_path,
        vkey,
        out,
        proof_json: proof_dir.join("proof.json"),
        proof_bin,
        publics: proof_dir.join("publics.json"),
        dir,
    }
}

/// `proofman-cli pilfflonk calldata -k <vkey> -p <proof> --publics <publics> <extra>`.
fn calldata_cli(vkey: &Path, proof: &Path, publics: &Path, extra: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_proofman-cli"))
        .args(["pilfflonk", "calldata", "-k"])
        .arg(vkey)
        .arg("-p")
        .arg(proof)
        .arg("--publics")
        .arg(publics)
        .args(extra)
        .output()
        .expect("proofman-cli runs")
}

/// The calldata the command writes to a file in `format`, which it must write and end with a
/// newline.
fn written(f: &Fixture, proof: &Path, format: &str) -> String {
    let path = f.dir.file(&format!("calldata.{format}"));
    let _ = fs::remove_file(&path);
    let run = calldata_cli(&f.vkey_path, proof, &f.publics, &["--format", format, "-o", path.to_str().unwrap()]);
    assert!(run.status.success(), "calldata --format {format}: {}", output(&run));
    let text = fs::read_to_string(&path).unwrap();
    let line = text.strip_suffix('\n').unwrap_or_else(|| panic!("{} does not end with a newline", path.display()));
    assert!(!line.contains('\n'), "{} is not one line", path.display());
    line.to_string()
}

fn r() -> BigUint {
    BigUint::parse_bytes(BN254_R.as_bytes(), 10).unwrap()
}

fn big(value: &FrBytes) -> BigUint {
    BigUint::from_bytes_le(&value.to_le_bytes())
}

/// The words of a calldata in snarkjs's form, `[0x…,…],[0x…,…]`, as 64 hexadecimal digits each:
/// those of `proof` and those of `pubSignals`, which is absent if there are no publics.
fn solidity_words(text: &str) -> Vec<Vec<String>> {
    let body = text.strip_prefix('[').and_then(|t| t.strip_suffix(']')).unwrap_or_else(|| panic!("{text}"));
    body.split("],[")
        .map(|array| {
            array
                .split(',')
                .map(|word| {
                    let digits = word.strip_prefix("0x").unwrap_or_else(|| panic!("{word} in {text}"));
                    assert!(digits.len() == 64 && digits.bytes().all(|b| b.is_ascii_hexdigit()), "{word}");
                    digits.to_string()
                })
                .collect()
        })
        .collect()
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

/// That the command's calldata of `f`'s proof is `selector`, the proof's bytes, `aux` and the
/// publics, in both forms, from `proof.json` and from `proof.bin`, and printed without `-o`; and that
/// it is the library's (`Calldata::read`).
fn writes_the_calldata(f: &Fixture, selector: &str, aux: &[FrBytes]) {
    let words = |bytes: &[u8]| bytes.chunks(32).map(hex).collect::<Vec<_>>();
    let proof_words: Vec<String> =
        words(&f.out.proof.to_bytes()).into_iter().chain(aux.iter().map(|a| hex(&a.to_be_bytes()))).collect();
    let public_words: Vec<String> = f.out.publics.0.iter().map(|p| hex(&p.to_be_bytes())).collect();
    assert_eq!(proof_words.len() as u64, CalldataLayout::of(&f.vkey).words());

    let abi = written(f, &f.proof_json, "hex");
    assert_eq!(abi, format!("0x{selector}{}{}", proof_words.concat(), public_words.concat()));
    let solidity = written(f, &f.proof_json, "solidity");
    let mut expected = vec![proof_words];
    if !public_words.is_empty() {
        expected.push(public_words);
    }
    assert_eq!(solidity_words(&solidity), expected);

    // The proof's bytes give the same calldata; the Solidity form is the default, and without -o it
    // is printed, on a line of its own.
    assert_eq!(written(f, &f.proof_bin, "hex"), abi);
    assert_eq!(written(f, &f.proof_bin, "solidity"), solidity);
    let printed = calldata_cli(&f.vkey_path, &f.proof_bin, &f.publics, &[]);
    assert!(printed.status.success(), "{}", output(&printed));
    assert!(String::from_utf8_lossy(&printed.stdout).lines().any(|line| line == solidity), "{}", output(&printed));

    let library = Calldata::read(&f.vkey_path, &f.proof_json, &f.publics).unwrap();
    assert_eq!((library.to_hex().unwrap(), library.to_solidity()), (abi, solidity));
}

/// `Domains`: `boundaries` everyRow, firstRow, lastRow and everyFrame {1, 2}, `N = 16`, two
/// publics. Its calldata is the proof's 26 words, `1/(ξ − 1)` and `1/(ξ − ω^15)` for the prover's
/// `ξ`, and the publics: `verifyProof(bytes32[28],uint256[2])`, whose selector solc gives as
/// `0x556e1ba3`.
#[test]
fn the_calldata_is_the_proof_and_an_inverse_per_first_row_and_last_row() {
    let f = fixture("domains", domains::Air::All);
    let layout = CalldataLayout::of(&f.vkey);
    let n = 1u64 << f.vkey.power;
    assert_eq!((layout.aux_rows.as_slice(), layout.proof_words(), f.out.publics.0.len()), (&[0, n - 1][..], 26, 2));

    // 1/(ξ − ω^j), with ξ = xiSeed^powerW of the prover and ω = 5^((r − 1)/N): each times ξ − ω^j is 1.
    let r = r();
    let xi = big(&f.out.challenges.xi_seed).modpow(&BigUint::from(f.vkey.power_w), &r);
    let omega = BigUint::from(5u32).modpow(&((&r - 1u32) / n), &r);
    let aux: Vec<FrBytes> = proofman_pilfflonk::auxiliary_inverses(&f.vkey, &f.out.challenges.xi_seed).unwrap();
    for (&j, a) in layout.aux_rows.iter().zip(&aux) {
        let row = omega.modpow(&BigUint::from(j), &r);
        assert_eq!((&xi + &r - row) % &r * big(a) % &r, BigUint::from(1u32), "the inverse of row {j}");
    }
    writes_the_calldata(&f, "556e1ba3", &aux);
}

/// `Frames`: six `everyFrame`, no `firstRow` nor `lastRow`, and no publics. Its calldata is the
/// proof's bytes and nothing else, and `verifyProof` has no `pubSignals`:
/// `verifyProof(bytes32[34])`, whose selector solc gives as `0xf85fc817`.
#[test]
fn the_calldata_of_a_key_without_first_or_last_row_is_the_proof() {
    let f = fixture("frames", domains::Air::Frames);
    assert!(CalldataLayout::of(&f.vkey).aux_rows.is_empty());
    assert!(f.out.publics.0.is_empty());
    assert_eq!(CalldataLayout::of(&f.vkey).words(), 34);
    writes_the_calldata(&f, "f85fc817", &[]);
    assert!(!written(&f, &f.proof_json, "solidity").contains("],["));
}

/// Inputs that do not go together, which `pilfflonk verify` rejects too, each refused with exit
/// code 1, a message that says why and no file written: a vkey whose digest is not its contents',
/// the proof of another key, a proof's bytes of another length, publics of another number or not
/// below `r`, a proof with a commitment the transcript (pilfflonk/docs/protocol.md#transcript) does
/// not absorb (off the curve, or with a coordinate below 2^192), and a file that is not there.
#[test]
fn the_calldata_of_inputs_that_do_not_go_together_is_refused() {
    let f = fixture("refused", domains::Air::All);
    let other = fixture("refused_other", domains::Air::Frames);
    let file = |name: &str, text: String| {
        let path = f.dir.file(name);
        fs::write(&path, text).unwrap();
        path
    };
    let read = |path: &Path| -> Value { serde_json::from_str(&fs::read_to_string(path).unwrap()).unwrap() };

    let mut digest = read(&f.vkey_path);
    let flipped = if digest["digest"].as_str().unwrap().ends_with('0') { "1" } else { "0" };
    let text = digest["digest"].as_str().unwrap();
    digest["digest"] = json!(format!("{}{flipped}", &text[..text.len() - 1]));
    let wrong_digest = file("wrong_digest.vkey.json", digest.to_string());

    let bytes = fs::read(&f.proof_bin).unwrap();
    let short_bin = f.dir.file("short.bin");
    fs::write(&short_bin, &bytes[..bytes.len() - 32]).unwrap();

    let publics = read(&f.publics);
    let one_public = file("one_public.json", json!([publics[0]]).to_string());
    let r_public = file("r_public.json", json!([publics[0], BN254_R]).to_string());

    let mut proof = read(&f.proof_json);
    let first = proof["polynomials"].as_object().unwrap().keys().find(|k| k.starts_with('f')).unwrap().clone();
    proof["polynomials"][&first] = json!(["1", "2", "1"]);
    let generator = file("generator.json", proof.to_string());
    proof["polynomials"][&first] = json!(["1", "3", "1"]);
    let off_curve = file("off_curve.json", proof.to_string());
    let missing = f.dir.file("missing.json");

    let absorbs = format!("{first} of the proof is not a point the transcript absorbs");
    for (label, vkey, proof, publics, expected) in [
        (
            "digest",
            &wrong_digest,
            &f.proof_json,
            &f.publics,
            "the digest of the vkey is not the digest of its contents",
        ),
        (
            "another key",
            &other.vkey_path,
            &f.proof_json,
            &f.publics,
            "the proof's evaluations do not have the names of its AIRs",
        ),
        ("bytes", &f.vkey_path, &short_bin, &f.publics, "a proof of this shape has"),
        ("one public", &f.vkey_path, &f.proof_json, &one_public, "1 publics, and the vkey has nPublic = 2"),
        ("public r", &f.vkey_path, &f.proof_json, &r_public, "is not a FrBytes: a decimal number below r"),
        ("(1, 2)", &f.vkey_path, &generator, &f.publics, absorbs.as_str()),
        ("off the curve", &f.vkey_path, &off_curve, &f.publics, "is not on the curve"),
        ("missing", &f.vkey_path, &missing, &f.publics, "IO error on"),
    ] {
        let out_file = f.dir.file("refused.txt");
        let run = calldata_cli(vkey, proof, publics, &["-o", out_file.to_str().unwrap()]);
        let text = output(&run);
        assert_eq!(run.status.code(), Some(1), "{label}: {text}");
        assert!(text.contains(expected), "{label}: {expected:?} not in:\n{text}");
        assert!(!out_file.exists(), "{label}: a calldata was written");
    }
    // The point off the curve is named as the one (1, 2) is.
    let run = calldata_cli(&f.vkey_path, &off_curve, &f.publics, &[]);
    assert!(output(&run).contains(&absorbs), "{}", output(&run));
}
