//! Rebuild every witness library (`.so` / `.dylib`) in a `provingKey/` tree
//! without re-running circom, by recompiling the `.cpp` sources persisted in
//! the proving key during setup.
//!
//! The proving key directory is the only required input: pass its path with
//! `--proving-key`. `--jobs` bounds how many `make` builds run concurrently.

use anyhow::{Context, Result};

use crate::commands::recursive_setup::resolve_path_env;
use crate::output::witness_gen::WitnessTracker;
use crate::proving_key::witness_rebuild::rebuild_all_witness_libs;

/// Options for the `rebuild-witness-libs` subcommand.
pub struct RebuildWitnessOptions {
    /// Path to the `provingKey/` directory.
    pub proving_key: String,
    /// Max number of concurrent `make` builds.
    pub jobs: usize,
}

pub fn run_rebuild_witness(opts: &RebuildWitnessOptions) -> Result<()> {
    let circom_helpers_dir = resolve_path_env("CIRCOM_HELPERS_DIR", "setup/circom");
    let goldilocks_src_dir = resolve_path_env("GOLDILOCKS_SRC_DIR", "pil2-stark/src/goldilocks/src");
    let witness_tracker = WitnessTracker::with_goldilocks_src(&goldilocks_src_dir);

    tracing::info!("Rebuilding witness libraries from {}", opts.proving_key);
    let count = rebuild_all_witness_libs(&opts.proving_key, &circom_helpers_dir, &witness_tracker, opts.jobs)
        .context("rebuild_all_witness_libs failed")?;
    tracing::info!("Rebuilt {} witness library(ies)", count);
    Ok(())
}
