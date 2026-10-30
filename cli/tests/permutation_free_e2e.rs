//! A free-selector permutation on the product bus (std_prod.pil, `PermutationFree` in
//! `pil2-components/test/permutation`): balanced it proves, unbalanced it is rejected.
//! `cargo test -p proofman-cli --test permutation_free_e2e`.

#[path = "../../pilfflonk/tests/data/witness_libraries.rs"]
mod witness_libraries;

use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};
use std::sync::OnceLock;

use pil2_stark_setup::commands::compile_pil::{run_compile_pil, CompilePilOptions};
use pil2_stark_setup::commands::setup::{run_setup, SetupOptions};
use proofman_common::hash_family::{compressed_final_by_default, default_aggregation_arity, DEFAULT_HASH_ID};

/// The witness library's switch for an unbalanced bus (permutation_free.rs).
const TAMPER: &str = "PERMUTATION_FREE_TAMPER";

fn scratch() -> PathBuf {
    Path::new(env!("CARGO_TARGET_TMPDIR")).join(format!("permutation_free_e2e_{}", std::process::id()))
}

/// The key, compiled and set up once per test process.
fn key() -> &'static Path {
    static KEY: OnceLock<PathBuf> = OnceLock::new();
    KEY.get_or_init(|| {
        let dir = scratch();
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("..").canonicalize().unwrap();
        let pilout = dir.join("permutation.pilout");
        run_compile_pil(&CompilePilOptions {
            pil_path: root.join("pil2-components/test/permutation/permutation.pil").display().to_string(),
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
        build.join("provingKey")
    })
}

fn output(out: &Output) -> String {
    format!("{}{}", String::from_utf8_lossy(&out.stdout), String::from_utf8_lossy(&out.stderr))
}

fn cli(command: &str, out_dir: Option<&Path>, tamper: bool) -> Output {
    let lib = witness_libraries::built_library("permutation");
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_proofman-cli"));
    cmd.args([command, "--witness-lib", lib.to_str().unwrap(), "--proving-key", key().to_str().unwrap()]);
    if let Some(dir) = out_dir {
        cmd.args(["--output-dir", dir.to_str().unwrap(), "--verify-proofs"]);
    }
    cmd.env("NO_COLOR", "1").env_remove(TAMPER);
    if tamper {
        cmd.env(TAMPER, "1");
    }
    cmd.output().expect("proofman-cli runs")
}

fn assert_rejected(out: &Output, what: &str, rejection: &str) {
    let text = output(out);
    assert!(!out.status.success(), "{what} succeeded, and should not:\n{text}");
    assert!(text.contains(rejection), "{what} failed without `{rejection}`:\n{text}");
}

#[test]
fn a_balanced_free_selector_product_bus_proves() {
    let out = cli("verify-constraints", None, false);
    assert!(out.status.success(), "verify-constraints failed:\n{}", output(&out));
    let dir = scratch().join("proofs");
    let out = cli("prove", Some(&dir), false);
    assert!(out.status.success(), "prove --verify-proofs failed:\n{}", output(&out));
}

/// With ONE_HALF = 1/2 = 0 every free-selector factor was 1, so this passed.
#[test]
fn an_unbalanced_free_selector_product_bus_is_rejected() {
    assert_rejected(&cli("verify-constraints", None, true), "verify-constraints", "Constraints were not verified");
    let dir = scratch().join("tampered");
    assert_rejected(&cli("prove", Some(&dir), true), "prove", "were not verified");
}
