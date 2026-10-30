//! A GPU proof of AIRs whose extended domain is under one 256-thread block: the std `simple` test
//! (SimpleLeft and SimpleRight, N = 8). `cargo test -p proofman-cli --test gpu_small_domain`;
//! without a GPU it skips, and `PROOFMAN_GPU_TESTS=1` makes that a failure.

#[path = "../../pilfflonk/tests/data/witness_libraries.rs"]
mod witness_libraries;

use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};

use pil2_stark_setup::commands::compile_pil::{run_compile_pil, CompilePilOptions};
use pil2_stark_setup::commands::setup::{run_setup, SetupOptions};
use proofman_common::hash_family::{compressed_final_by_default, default_aggregation_arity, DEFAULT_HASH_ID};
use serde_json::Value;

const SMALL_AIRS: [&str; 2] = ["SimpleLeft", "SimpleRight"];

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("..").canonicalize().unwrap()
}

fn output(out: &Output) -> String {
    format!("{}{}", String::from_utf8_lossy(&out.stdout), String::from_utf8_lossy(&out.stderr))
}

fn find_file(dir: &Path, name: &str) -> Option<PathBuf> {
    for entry in fs::read_dir(dir).ok()?.flatten() {
        let path = entry.path();
        if path.is_dir() {
            if let Some(found) = find_file(&path, name) {
                return Some(found);
            }
        } else if path.file_name().is_some_and(|f| f == name) {
            return Some(path);
        }
    }
    None
}

/// The first GPU's compute capability as a `gen-exps` arch (12.0 -> sm_120).
fn cuda_arch() -> Option<String> {
    let out = Command::new("nvidia-smi").args(["--query-gpu=compute_cap", "--format=csv,noheader"]).output().ok()?;
    let cap = String::from_utf8_lossy(&out.stdout).lines().next()?.trim().replace('.', "");
    (out.status.success() && !cap.is_empty() && cap.chars().all(|c| c.is_ascii_digit())).then(|| format!("sm_{cap}"))
}

fn gpu_ready() -> Option<String> {
    let nvcc = Command::new("nvcc").arg("--version").output().is_ok_and(|o| o.status.success());
    let arch = cuda_arch();
    let why = if !proofman_pilfflonk::gpu_available() {
        "no GPU"
    } else if arch.is_none() {
        "no CUDA arch"
    } else if !nvcc {
        "no nvcc on PATH"
    } else {
        return arch;
    };
    assert_ne!(std::env::var("PROOFMAN_GPU_TESTS").as_deref(), Ok("1"), "PROOFMAN_GPU_TESTS=1, and {why}");
    eprintln!("skipping gpu_small_domain: {why}");
    None
}

fn cli(args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_proofman-cli")).args(args).env("NO_COLOR", "1").output().unwrap()
}

fn assert_ok(out: &Output, what: &str) {
    assert!(out.status.success(), "{what} failed:\n{}", output(out));
}

#[test]
fn gpu_proves_airs_with_an_extended_domain_under_256_rows() {
    let Some(arch) = gpu_ready() else { return };
    let dir = Path::new(env!("CARGO_TARGET_TMPDIR")).join(format!("gpu_small_domain_{}", std::process::id()));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();
    let root = repo_root();

    let pilout = dir.join("simple.pilout");
    run_compile_pil(&CompilePilOptions {
        pil_path: root.join("pil2-components/test/simple/simple.pil").display().to_string(),
        output_path: pilout.display().to_string(),
        include_paths: vec![root.join("pil2-components/lib/std/pil").display().to_string()],
        ..Default::default()
    })
    .expect("compile-pil");
    let build = dir.join("build");
    run_setup(&SetupOptions {
        airout_path: pilout.display().to_string(),
        build_dir: build.display().to_string(),
        fixed_dir: None,
        stark_structs_path: None,
        recursive: false,
        recursive_jobs: 1,
        setup_jobs: 1,
        stats_output_path: None,
        hash: DEFAULT_HASH_ID.to_string(),
        agg_arity: default_aggregation_arity(DEFAULT_HASH_ID),
        compressed_final: compressed_final_by_default(DEFAULT_HASH_ID),
        setup_version: None,
        recursive_n_bits: None,
        gen_exps: false,
        exps_arch: "auto".to_string(),
        exps_cap: 60000,
        exps_chunk: None,
        exps_stark_src: None,
    })
    .expect("setup");
    let key = build.join("provingKey");

    for air in SMALL_AIRS {
        let path = find_file(&key, &format!("{air}.starkinfo.json")).unwrap_or_else(|| panic!("no {air} starkinfo"));
        let info: Value = serde_json::from_str(&fs::read_to_string(path).unwrap()).unwrap();
        let n_bits_ext = info["starkStruct"]["nBitsExt"].as_u64().unwrap();
        assert!(n_bits_ext < 8, "{air}: nBitsExt {n_bits_ext} no longer exercises a domain under 256 rows");
    }

    let key = key.display().to_string();
    assert_ok(&cli(&["gen-exps", "--proving-key", &key, "--arch", &arch]), "gen-exps");
    let lib = witness_libraries::built_library("simple");
    let out = dir.join("proofs").display().to_string();
    let args = ["prove", "--witness-lib", lib.to_str().unwrap(), "--proving-key", &key, "--output-dir", &out];
    assert_ok(&cli(&[&args[..], &["--verify-proofs", "--gpu"]].concat()), "prove --gpu --verify-proofs");
    let _ = fs::remove_dir_all(&dir);
}
