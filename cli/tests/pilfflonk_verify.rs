//! `proofman-cli pilfflonk verify` (spec §4.5, plan M19) on the keys of the Fibonacci fixture,
//! set up with the ptau of `τ = 1` (plan N13). Until the prover exists (M18), the proof is forged
//! by `forgeProof` of `pilfflonk/js/test/proofs.js`, which opens anything since it knows `τ`. The
//! command must exit with 0 on it, and with another status on a tampered copy, on malformed files
//! and on every vkey `Vkey::validate` refuses, which the JS verifier must refuse too.
//!
//! Pilouts are not versioned: the test compiles the fixture with the compiler `PIL2C_EXEC` names,
//! which must honour `prime`, and is `#[ignore]` without it. It needs Node.js:
//!
//! ```text
//! PIL2C_EXEC=<pil2-compiler>/src/pil.js cargo test -p proofman-cli --features proofman-starks-lib-c/cpu-only \
//!     --test pilfflonk_verify -- --ignored
//! ```

use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};

use pilfflonk_setup::command::{DEFAULT_EXTRA_MULS, DEFAULT_MAX_CONSTRAINT_DEGREE, DEFAULT_MAX_Q_DEGREE, PROVING_KEY_DIR};
use pilfflonk_setup::digest::seal_vkey;
use pilfflonk_setup::test_ptau::write_tau_one_ptau;
use pilfflonk_setup::{run_setup_pilfflonk, SetupPilfflonkOptions};
use proofman_pilfflonk::{JsonFile, PilfflonkGlobalInfo, Vkey, BN254_R};
use serde_json::{json, Value};

/// The Fibonacci's publics for the inputs 1 and 2 (plan M13, `pil-fflonk/runtime/public.json`).
const PUBLICS: [&str; 3] = ["1", "2", "590308608561184158373097535019708483037277117989374906445627411437315467687"];

/// Writes the proof `forgeProof` forges for the vkey and the publics at the paths it is given.
const FORGE: &str = r#"
import { readFileSync, writeFileSync } from "node:fs";
import { pathToFileURL } from "node:url";

const [js, vkeyPath, publicsPath, proofPath] = process.argv.slice(1);
const { forgeProof } = await import(pathToFileURL(`${js}/test/proofs.js`).href);
const { newCurve } = await import(pathToFileURL(`${js}/test/support.js`).href);
const read = (path) => JSON.parse(readFileSync(path, "utf8"));
const proof = forgeProof(await newCurve(), read(vkeyPath), read(publicsPath));
writeFileSync(proofPath, JSON.stringify(proof, null, 1));
"#;

/// A fresh directory for the test under the target's temporary directory, removed when dropped.
struct TestDir(PathBuf);

impl TestDir {
    fn new(name: &str) -> Self {
        let dir =
            Path::new(env!("CARGO_TARGET_TMPDIR")).join(format!("pilfflonk_verify_{name}_{}", std::process::id()));
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

fn stderr(out: &Output) -> String {
    String::from_utf8_lossy(&out.stderr).into_owned()
}

/// What a command wrote: the CLI logs to stdout, the verifier to stderr.
fn output(out: &Output) -> String {
    format!("{}{}", String::from_utf8_lossy(&out.stdout), stderr(out))
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
    assert!(out.status.success(), "pil2com: {}", stderr(&out));
}

/// `proofman-cli pilfflonk verify vkey publics proof`.
fn verify(vkey: &Path, publics: &Path, proof: &Path) -> Output {
    Command::new(env!("CARGO_BIN_EXE_proofman-cli"))
        .args(["pilfflonk", "verify"])
        .args([vkey, publics, proof])
        .output()
        .expect("proofman-cli runs")
}

fn forge_proof(vkey: &Path, publics: &Path, proof: &Path) {
    let out = Command::new("node")
        .args(["--input-type=module", "-e", FORGE])
        .arg(repo_root().join("pilfflonk/js"))
        .args([vkey, publics, proof])
        .output()
        .expect("node runs");
    assert!(out.status.success(), "forgeProof: {}", stderr(&out));
}

/// `value + 1 mod r`, in decimal.
fn plus_one(value: &Value) -> Value {
    let r = num_bigint::BigUint::parse_bytes(BN254_R.as_bytes(), 10).unwrap();
    let v = num_bigint::BigUint::parse_bytes(value.as_str().unwrap().as_bytes(), 10).unwrap();
    json!(((v + 1u32) % r).to_string())
}

fn write_json(path: &Path, value: &Value) {
    fs::write(path, serde_json::to_string_pretty(value).unwrap()).unwrap();
}

/// A change to the vkey that `Vkey::validate` refuses, and why, as Rust and the JS verifier say it.
struct Refusal {
    name: &'static str,
    change: fn(&mut Value),
    rust: &'static str,
    js: &'static str,
}

/// The first operand of the qVerifier's code of type `kind`.
fn first_operand<'a>(vkey: &'a mut Value, kind: &str) -> &'a mut Value {
    let code = vkey["qVerifier"]["code"].as_array_mut().unwrap();
    code.iter_mut()
        .flat_map(|entry| entry["src"].as_array_mut().unwrap().iter_mut())
        .find(|operand| operand["type"] == kind)
        .unwrap_or_else(|| panic!("the qVerifier reads no {kind}"))
}

#[test]
#[ignore = "needs PIL2C_EXEC and Node.js"]
fn it_verifies_a_proof_of_the_fibonacci_and_rejects_every_change() {
    let dir = TestDir::new("fibonacci");
    let opts = SetupPilfflonkOptions {
        airout_path: dir.file("fibonacci.pilout"),
        build_dir: dir.file("build"),
        powers_of_tau: dir.file("tau_one.ptau"),
        max_constraint_degree: DEFAULT_MAX_CONSTRAINT_DEGREE,
        extra_muls: DEFAULT_EXTRA_MULS,
        max_q_degree: DEFAULT_MAX_Q_DEGREE,
        no_packing: true,
    };
    compile_fibonacci(&opts.airout_path);
    // More powers than the Fibonacci's largest degree, 261 (plan M16).
    write_tau_one_ptau(&opts.powers_of_tau, 1024).unwrap();
    run_setup_pilfflonk(&opts).unwrap();
    let proving_key = opts.build_dir.join(PROVING_KEY_DIR);
    let vkey_path = PilfflonkGlobalInfo::from_proving_key(&proving_key).unwrap().vkey_path(&proving_key);
    let publics_path = dir.file("publics.json");
    write_json(&publics_path, &json!(PUBLICS));
    let proof_path = dir.file("proof.json");
    // Another exit status than 0, and the reason.
    let rejected = |vkey: &Path, publics: &Path, proof: &Path, reason: &str| {
        let out = verify(vkey, publics, proof);
        assert!(!out.status.success(), "{reason}: {}", output(&out));
        assert!(output(&out).contains(reason), "{reason}: {}", output(&out));
    };

    // A proof that is not JSON, which also installs the verifier's dependencies if they are
    // missing: the forging needs them.
    fs::write(&proof_path, "not JSON").unwrap();
    rejected(&vkey_path, &publics_path, &proof_path, "cannot read the vkey, the publics or the proof");

    forge_proof(&vkey_path, &publics_path, &proof_path);
    let out = verify(&vkey_path, &publics_path, &proof_path);
    assert!(out.status.success(), "{}", output(&out));
    assert!(stderr(&out).contains("OK: the proof verifies"), "{}", output(&out));
    let proof: Value = serde_json::from_str(&fs::read_to_string(&proof_path).unwrap()).unwrap();

    // A tampered proof, tampered publics and malformed files.
    let other = dir.file("other.json");
    let mut tampered = proof.clone();
    tampered["evaluations"]["l1w"] = plus_one(&tampered["evaluations"]["l1w"]);
    write_json(&other, &tampered);
    rejected(&vkey_path, &publics_path, &other, "INVALID: the proof does not verify");
    let mut tampered = proof.clone();
    tampered["polynomials"]["f2"] = proof["polynomials"]["W"].clone();
    write_json(&other, &tampered);
    rejected(&vkey_path, &publics_path, &other, "INVALID: the proof does not verify");
    write_json(&other, &json!([PUBLICS[0], PUBLICS[1], plus_one(&json!(PUBLICS[2]))]));
    rejected(&vkey_path, &other, &proof_path, "INVALID: the proof does not verify");

    write_json(&other, &json!({}));
    rejected(&vkey_path, &publics_path, &other, "INVALID: the proof does not verify");
    write_json(&other, &json!(["1", "2"]));
    rejected(&vkey_path, &other, &proof_path, "INVALID: the proof does not verify");
    fs::write(&other, "{").unwrap();
    rejected(&other, &publics_path, &proof_path, "cannot read the vkey, the publics or the proof");
    rejected(&dir.file("missing.json"), &publics_path, &proof_path, "no such file or directory");

    // Every vkey Vkey::validate refuses, sealed with its digest, is refused by the verifier too,
    // for the same reason: what the setup writes, the verifier runs.
    let vkey: Value = serde_json::from_str(&fs::read_to_string(&vkey_path).unwrap()).unwrap();
    let refusals = [
        Refusal {
            name: "a challenge of stage 1",
            change: |v| v["numChallenges"] = json!([1]),
            rust: "A.4 squeezes no challenge of stage 1",
            js: "stage 1 has challenges, which A.4 never squeezes",
        },
        Refusal {
            name: "challenges of a stage the layout does not have",
            change: |v| v["numChallenges"] = json!([0, 0]),
            rust: "numChallenges has 2 stages, and the layout 1",
            js: "numChallenges has 2 stages, and the layout 1",
        },
        Refusal {
            name: "another boundary first",
            change: |v| v["boundaries"] = json!([{"name": "firstRow"}, {"name": "everyRow"}]),
            rust: "boundaries[0] must be everyRow",
            js: "boundaries[0] is not everyRow",
        },
        Refusal {
            name: "an everyFrame that leaves no row",
            change: |v| {
                let every_frame = json!({"name": "everyFrame", "offsetMin": 128, "offsetMax": 128});
                v["boundaries"].as_array_mut().unwrap().push(every_frame);
            },
            rust: "no row of 256 is left",
            js: "no row of 256 is left",
        },
        Refusal {
            name: "an op the verifier does not run",
            change: |v| v["qVerifier"]["code"][0]["op"] = json!("div"),
            rust: "op \"div\" is not add, sub, mul or copy",
            js: "op \"div\" is not add, sub, mul or copy",
        },
        Refusal {
            name: "a challenge whose id is not its position",
            change: |v| first_operand(v, "challenge")["id"] = json!(1),
            rust: "stageId 0 is 0, not 1",
            js: "stageId 0 is 0, not 1",
        },
        Refusal {
            name: "an evaluation out of the evMap",
            change: |v| first_operand(v, "eval")["id"] = json!(7),
            rust: "eval 7 is not one of the 7",
            js: "eval 7 is not one of the 7",
        },
    ];
    for Refusal { name, change, rust, js } in refusals {
        let mut changed = vkey.clone();
        change(&mut changed);
        let sealed = seal_vkey(serde_json::from_value::<Vkey>(changed).unwrap()).unwrap();
        let text = serde_json::to_string_pretty(&sealed).unwrap();
        let err = Vkey::from_json_str(&text).expect_err(name).to_string();
        assert!(err.contains(rust), "{name}: {err}");
        fs::write(&other, text).unwrap();
        rejected(&other, &publics_path, &proof_path, js);
    }
    // Resealed unchanged, the vkey is the setup's, byte for byte.
    let sealed = seal_vkey(serde_json::from_value::<Vkey>(vkey).unwrap()).unwrap();
    assert_eq!(sealed.to_json_string().unwrap(), fs::read_to_string(&vkey_path).unwrap());
}
