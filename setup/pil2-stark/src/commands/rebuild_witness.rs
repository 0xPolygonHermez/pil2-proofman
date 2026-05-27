//! Rebuild every witness library (`.so` / `.dylib`) in a `provingKey/` tree
//! without re-running the full setup pipeline.
//!
//! Skips `pil_info`, `bctree`, `plonk2pil`, and the const-tree computation —
//! only re-emits the circom verifier / wrapper sources, runs `circom --c`, and
//! recompiles the generated C++ to a shared library.
//!
//! The proving key directory is the only required input: pass its path with
//! `--proving-key`.  Intermediate artifacts (`.circom`, `.cpp`, `.r1cs`) land
//! in a tempdir that is removed once the rebuild completes — unless the user
//! supplies `--build-dir`, in which case they are kept there for inspection.

use std::fs;
use std::path::PathBuf;

use anyhow::{Context, Result};

use crate::commands::recursive_setup::{resolve_circom_exec, resolve_path_env};
use crate::output::witness_gen::WitnessTracker;
use crate::proving_key::witness_rebuild::{rebuild_all_witness_libs, RebuildPaths};

/// Options for the `rebuild-witness-libs` subcommand.
pub struct RebuildWitnessOptions {
    /// Path to the `provingKey/` directory.
    pub proving_key: String,
    /// Optional build directory for the generated `.circom`/`.cpp` files.
    /// If `None`, a tempdir is used and removed when the command finishes.
    pub build_dir: Option<String>,
    /// Number of circom compiles to run in parallel (1 = serial).
    pub jobs: usize,
}

/// Holds the build directory either as a managed tempdir or as a
/// user-supplied path.  In the tempdir case the contents are removed when
/// `Self` is dropped — keep this alive until `witness_tracker.await_all()`
/// has returned, otherwise spawned witness builds will fail to read their
/// `.cpp` source.
enum BuildDir {
    Temp(tempfile::TempDir),
    Persistent(PathBuf),
}

impl BuildDir {
    fn path(&self) -> &std::path::Path {
        match self {
            BuildDir::Temp(t) => t.path(),
            BuildDir::Persistent(p) => p,
        }
    }
}

pub fn run_rebuild_witness(opts: &RebuildWitnessOptions) -> Result<()> {
    // Defaults point at the Rust port's circom library locations
    // (`setup/stark-recurser/stark2circom/circom_verifier/...`).  The legacy
    // JS layout (`node_modules/stark-recurser/src/...`) is still selectable
    // via the env vars below.
    let circuits_gl_path =
        resolve_path_env("CIRCUITS_GL_PATH", "setup/stark-recurser/stark2circom/circom_verifier/circuits.gl");
    let recurser_circuits_path =
        resolve_path_env("RECURSER_CIRCUITS_PATH", "setup/stark-recurser/stark2circom/circom_verifier/helper_circuits");
    let recurser_circuits_compressed_final_path = resolve_path_env(
        "RECURSER_CIRCUITS_COMPRESSED_FINAL_PATH",
        "setup/stark-recurser/stark2circom/circom_verifier/helper_circuits",
    );
    let circom_helpers_dir = resolve_path_env("CIRCOM_HELPERS_DIR", "setup/circom");
    let goldilocks_src_dir = resolve_path_env("GOLDILOCKS_SRC_DIR", "pil2-stark/src/goldilocks/src");

    let circom_exec = resolve_circom_exec(&circom_helpers_dir);
    let witness_tracker = WitnessTracker::with_goldilocks_src(&goldilocks_src_dir);

    let paths = RebuildPaths {
        circom_exec: &circom_exec,
        circuits_gl_path: &circuits_gl_path,
        recurser_circuits_path: &recurser_circuits_path,
        recurser_circuits_compressed_final_path: &recurser_circuits_compressed_final_path,
        circom_helpers_dir: &circom_helpers_dir,
    };

    let build_dir = match opts.build_dir.as_deref() {
        Some(p) => {
            let pb = PathBuf::from(p);
            fs::create_dir_all(&pb).with_context(|| format!("Cannot create build dir {}", pb.display()))?;
            BuildDir::Persistent(pb)
        }
        None => BuildDir::Temp(tempfile::tempdir().context("Failed to create build tempdir")?),
    };
    let build_dir_str = build_dir.path().to_string_lossy().into_owned();

    tracing::info!("Rebuilding witness libraries from {}", opts.proving_key);
    tracing::info!("Build dir: {}", build_dir_str);
    let count = rebuild_all_witness_libs(&opts.proving_key, &build_dir_str, &paths, &witness_tracker, opts.jobs)
        .context("rebuild_all_witness_libs failed")?;
    tracing::info!("Spawned {} witness library build(s); waiting for completion", count);

    // `await_all` must return before `build_dir` is dropped — the witness
    // builds read `.cpp` files from inside it.
    witness_tracker.await_all().context("Witness library build failed")?;
    drop(build_dir);
    tracing::info!("Rebuilt {} witness library(ies)", count);
    Ok(())
}
