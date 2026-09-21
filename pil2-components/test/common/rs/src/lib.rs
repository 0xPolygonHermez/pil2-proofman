use std::collections::HashMap;
use std::path::{Path, PathBuf};

use proofman_fields::Goldilocks;
use pil2_stark_setup::commands::compile_pil::{run_compile_pil, CompilePilOptions};
use pil2_stark_setup::commands::setup::{run_setup, SetupOptions};
use proofman::ProofMan;
use proofman_common::{DebugInfo, ProofmanOptions};

const TESTS_DIR: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../.."); // .../pil2-components/test
const STD: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../../../lib/std/pil");

fn target_dir() -> PathBuf {
    if let Ok(dir) = std::env::var("CARGO_TARGET_DIR") {
        return PathBuf::from(dir);
    }
    PathBuf::from(concat!(env!("CARGO_MANIFEST_DIR"), "/../../../../target"))
}

/// Locate the prebuilt witness library `lib<name>.so` under `target/{debug,release}`.
fn witness_lib(name: &str) -> Option<PathBuf> {
    let file = format!("lib{name}.so");
    ["debug", "release"].iter().map(|p| target_dir().join(p).join(&file)).find(|p| p.exists())
}

/// Compile `pil_path` into `<build>/<stem>.pilout` (fixed columns under `<build>/fixed`),
/// using the shared std library as the include path. Returns the pilout path.
pub fn compile(pil_path: &str, build: &Path) -> Result<PathBuf, String> {
    let fixed = build.join("fixed");
    std::fs::create_dir_all(&fixed).map_err(|e| format!("create build dir: {e}"))?;

    let stem = Path::new(pil_path).file_stem().and_then(|s| s.to_str()).unwrap_or("pilout");
    let pilout = build.join(format!("{stem}.pilout"));

    run_compile_pil(&CompilePilOptions {
        pil_path: pil_path.to_string(),
        output_path: pilout.to_string_lossy().into_owned(),
        include_paths: vec![STD.to_string()],
        fixed_dir: Some(fixed.to_string_lossy().into_owned()),
        fixed_to_file: true,
        no_proto_fixed_data: false,
    })
    .map_err(|e| format!("compile-pil: {e}"))?;

    Ok(pilout)
}

/// Generate the setup (proving key under `<build>/provingKey`) from `pilout`.
pub fn setup(pilout: &Path, build: &Path) -> Result<(), String> {
    run_setup(&SetupOptions {
        airout_path: pilout.to_string_lossy().into_owned(),
        build_dir: build.to_string_lossy().into_owned(),
        fixed_dir: Some(build.join("fixed").to_string_lossy().into_owned()),
        stark_structs_path: None,
        recursive: false,
        recursive_jobs: 1,
        setup_jobs: 1,
        stats_output_path: None,
        hash: proofman_common::hash_family::DEFAULT_HASH_ID.to_string(),
        // Inert for a non-recursive setup -- there are no recursive airs to size or a final to
        // compress -- but derived from the family rather than hardcoded, so this cannot drift if
        // the default hash ever changes.
        compressed_final: proofman_common::hash_family::compressed_final_by_default(
            proofman_common::hash_family::DEFAULT_HASH_ID,
        ),
        recursive_n_bits: None,
        gen_exps: false,
        exps_arch: "auto".to_string(),
        exps_cap: 60000,
        exps_chunk: None,
        exps_stark_src: None,
        agg_arity: proofman_common::hash_family::default_aggregation_arity(
            proofman_common::hash_family::DEFAULT_HASH_ID,
        ),
    })
    .map_err(|e| format!("setup: {e}"))
}

/// `compile` -> `setup` -> `verify_proof_constraints` for one witness-computation test.
/// Call from that test crate's own `tests/*.rs` so it runs in its own process (see the
/// module docs on the MPI single-init constraint).
pub fn run_pipeline(dir: &str, pil_file: &str, lib: &str) -> Result<(), String> {
    // Generate the build folder alongside the test's `rs` crate (i.e. `test/<dir>/build`)
    let build = PathBuf::from(format!("{TESTS_DIR}/{dir}/build"));

    let pilout = compile(&format!("{TESTS_DIR}/{dir}/{pil_file}"), &build)?;
    setup(&pilout, &build)?;

    let witness = witness_lib(lib).ok_or_else(|| {
        format!("witness lib lib{lib}.so not found under target/; run `cargo build --workspace` first")
    })?;

    let mut options = ProofmanOptions::default();
    options.verify_constraints();
    options.verbose_mode(0u8.into());

    let proofman =
        ProofMan::<Goldilocks>::new(build.join("provingKey"), options).map_err(|e| format!("ProofMan::new: {e}"))?;
    proofman
        .register_custom_commits(HashMap::<String, PathBuf>::new())
        .map_err(|e| format!("register_custom_commits: {e}"))?;
    proofman
        .verify_proof_constraints(witness, None, None, &DebugInfo::default(), 0u8.into())
        .map_err(|e| format!("verify_proof_constraints: {e}"))?;

    Ok(())
}
