//! The std's cyclic gsum end to end, on `pil2-components/test/cyclic_sum`: an AIR whose one bus
//! constraint is degree 1 (Q degree 0, floored to one piece by setup) and has a direct term, so the
//! prover's correction is not 0. `cargo test --release -p proofman-cli --test cyclic_sum_e2e`.

#[path = "../../pilfflonk/tests/data/witness_libraries.rs"]
mod witness_libraries;
#[path = "common/domains.rs"]
mod common;

use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};
use std::sync::OnceLock;

use common::{find_file, gpu_lock, gpu_ready, output, repo_root, scratch, TestDir};
use pil2_stark_setup::commands::compile_pil::{run_compile_pil, CompilePilOptions};
use serde_json::Value;

/// The Q-degree-0 AIR (row and direct terms) and the one with direct terms only.
const AIRS: [&str; 2] = ["CyclicSumQ0", "CyclicSumDirect"];
/// The witness library's switch for an unbalanced bus (one extra multiplicity on row 3).
const TAMPER: &str = "CYCLIC_SUM_TAMPER";

/// The CPU key, compiled and set up once per test process.
fn key() -> &'static Path {
    static KEY: OnceLock<PathBuf> = OnceLock::new();
    KEY.get_or_init(|| {
        let dir = scratch("cyclic_sum_e2e_key");
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        let pilout = dir.join("build.pilout");
        let root = repo_root();
        run_compile_pil(&CompilePilOptions {
            pil_path: root.join("pil2-components/test/cyclic_sum/cyclic_sum.pil").display().to_string(),
            output_path: pilout.display().to_string(),
            include_paths: vec![root.join("pil2-components/lib/std/pil").display().to_string()],
            ..Default::default()
        })
        .expect("compile-pil");
        common::setup(&pilout, &dir.join("build"), false)
    })
}

/// `key` with its `.exps.so`.
fn gpu_key() -> &'static Path {
    static KEY: OnceLock<&Path> = OnceLock::new();
    KEY.get_or_init(|| {
        let arch = common::cuda_arch().expect("gpu_ready checked it");
        let key = key();
        assert_ok(&cli(&["gen-exps", "--proving-key", key.to_str().unwrap(), "--arch", &arch], &[]), "gen-exps");
        key
    })
}

fn json_of(file: &str) -> Value {
    let path = find_file(key(), file).unwrap_or_else(|| panic!("no {file}"));
    serde_json::from_str(&fs::read_to_string(path).unwrap()).unwrap()
}

fn cli(args: &[&str], envs: &[(&str, &str)]) -> Output {
    let _gpu = (args.contains(&"--gpu") || args.first() == Some(&"gen-exps")).then(gpu_lock);
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_proofman-cli"));
    cmd.args(args).env("NO_COLOR", "1").env_remove(TAMPER).envs(envs.iter().copied());
    cmd.output().expect("proofman-cli runs")
}

fn assert_ok(out: &Output, what: &str) {
    assert!(out.status.success(), "{what} failed:\n{}", output(out));
}

fn assert_rejected(out: &Output, what: &str, rejection: &str) {
    let text = output(out);
    assert!(!out.status.success(), "{what} succeeded, and should not:\n{text}");
    assert!(!text.contains("panicked"), "{what} panicked instead of being rejected:\n{text}");
    assert!(text.contains(rejection), "{what} failed without `{rejection}`:\n{text}");
}

fn lib() -> PathBuf {
    witness_libraries::built_library("cyclic_sum")
}

fn verify_constraints(key: &Path, gpu: bool, envs: &[(&str, &str)]) -> Output {
    let (lib, key) = (lib(), key.display().to_string());
    let mut args = vec!["verify-constraints", "--witness-lib", lib.to_str().unwrap(), "--proving-key", &key];
    if gpu {
        args.push("--gpu");
    }
    cli(&args, envs)
}

fn prove(key: &Path, out_dir: &Path, gpu: bool, envs: &[(&str, &str)]) -> Output {
    let (lib, key, out) = (lib(), key.display().to_string(), out_dir.display().to_string());
    let mut args = vec!["prove", "--witness-lib", lib.to_str().unwrap(), "--proving-key", &key];
    args.extend(["--output-dir", &out, "--verify-proofs"]);
    if gpu {
        args.push("--gpu");
    }
    cli(&args, envs)
}

/// Proves into `dump` with `PIL2_DUMP_ZKIN=basic` and returns the basic proofs, by file name.
fn dumped_proofs(key: &Path, dump: &Path, gpu: bool) -> Vec<(String, Vec<u8>)> {
    fs::create_dir_all(dump).unwrap();
    let envs = [("PIL2_DUMP_ZKIN", "basic"), ("TMPDIR", dump.to_str().unwrap())];
    assert_ok(&prove(key, dump, gpu, &envs), &format!("prove (gpu {gpu})"));
    let mut proofs: Vec<(String, Vec<u8>)> = fs::read_dir(dump)
        .unwrap()
        .map(|e| e.unwrap().path())
        .filter(|p| p.file_name().unwrap().to_string_lossy().ends_with("_tbasic.bin"))
        .map(|p| (p.file_name().unwrap().to_string_lossy().into_owned(), fs::read(&p).unwrap()))
        .collect();
    proofs.sort();
    assert_eq!(proofs.len(), AIRS.len(), "one basic proof per AIR in {}", dump.display());
    proofs
}

#[test]
fn setup_floors_a_degree_one_air_to_one_q_piece_and_marks_the_sum_cyclic() {
    let info = json_of(&format!("{}.starkinfo.json", AIRS[0]));
    assert_eq!(info["qDeg"].as_u64(), Some(1), "qDeg");
    assert_eq!(info["mapSectionsN"]["cm3"].as_u64(), Some(3), "one cubic Q piece");
    // The Q piece is a committed polynomial of stage nStages + 1, and the verifier opens it.
    let q_stage = info["nStages"].as_u64().unwrap() + 1;
    let pols = info["cmPolsMap"].as_array().unwrap();
    let q: Vec<usize> = (0..pols.len()).filter(|&i| pols[i]["stage"].as_u64() == Some(q_stage)).collect();
    assert_eq!(q.len(), 1, "one Q piece in cmPolsMap");
    let opened =
        info["evMap"].as_array().unwrap().iter().any(|e| e["type"] == "cm" && e["id"].as_u64() == Some(q[0] as u64));
    assert!(opened, "Q's piece is in evMap");

    for air in AIRS {
        let exprs = json_of(&format!("{air}.expressionsinfo.json"));
        let hints = exprs["hintsInfo"].as_array().unwrap();
        let gsum = hints.iter().find(|h| h["name"] == "gsum_col").expect("gsum_col");
        let cyclic = gsum["fields"].as_array().unwrap().iter().find(|f| f["name"] == "cyclic").expect("cyclic field");
        assert_eq!(cyclic["values"][0]["value"], "1", "{air}");
    }
}

#[test]
fn cpu_proves_and_gpu_proof_is_byte_identical() {
    let dir = TestDir::new("cyclic_sum_e2e_bytes");
    assert_ok(&verify_constraints(key(), false, &[]), "verify-constraints");
    let cpu = dumped_proofs(key(), &dir.path("cpu"), false);
    if !gpu_ready("cpu_proves_and_gpu_proof_is_byte_identical") {
        return;
    }
    assert_ok(&verify_constraints(gpu_key(), true, &[]), "verify-constraints --gpu");
    let gpu = dumped_proofs(gpu_key(), &dir.path("gpu"), true);
    for ((name, c), (_, g)) in cpu.iter().zip(&gpu) {
        assert!(c == g, "the GPU basic proof {name} differs from the CPU one");
    }
}

/// With the cyclic form an unbalanced bus breaks the gsum constraint on every row, not only the last.
#[test]
fn an_unbalanced_bus_is_rejected() {
    let dir = TestDir::new("cyclic_sum_e2e_tamper");
    let tamper = [(TAMPER, "1")];
    let out = verify_constraints(key(), false, &tamper);
    assert_rejected(&out, "verify-constraints", "Constraints were not verified");
    let text = output(&out);
    for air in AIRS {
        let at = text.find(&format!("Instance #0 of {air} [")).unwrap_or_else(|| panic!("{air} not checked:\n{text}"));
        let report = &text[at..];
        let report = &report[..report.find("constraints for Instance").unwrap_or(report.len())];
        for row in [0, 1, 255] {
            assert!(report.contains(&format!("Failed at row {row} ")), "{air}: row {row} not named:\n{text}");
        }
    }
    assert_rejected(&prove(key(), &dir.path("cpu"), false, &tamper), "prove", "Basic proofs were not verified");
    if !gpu_ready("an_unbalanced_bus_is_rejected") {
        return;
    }
    assert_rejected(
        &verify_constraints(gpu_key(), true, &tamper),
        "verify-constraints --gpu",
        "Constraints were not verified",
    );
    assert_rejected(
        &prove(gpu_key(), &dir.path("gpu"), true, &tamper),
        "prove --gpu",
        "Basic proofs were not verified",
    );
}
