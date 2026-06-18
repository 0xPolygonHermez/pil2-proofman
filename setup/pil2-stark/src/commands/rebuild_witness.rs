//! Rebuild every witness library (`.so` / `.dylib`) in a `provingKey/` tree
//! without re-running circom, by recompiling the `.cpp` sources persisted in
//! the proving key during setup.
//!
//! The proving key directory is the only required input: pass its path with
//! `--proving-key`. `--jobs` bounds how many `make` builds run concurrently.
//!
//! Optionally, pass `--proving-key-snark <provingKeySnark/>` to ALSO rebuild the
//! snark witness libraries (`recursivef`, `final`). These live outside the
//! regular `provingKey/` tree and each needs a different circom helpers dir, so
//! they are handled by a dedicated walk — see [`rebuild_snark_witness_libs`].

use anyhow::{Context, Result};

use crate::commands::recursive_setup::resolve_path_env;
use crate::output::witness_gen::WitnessTracker;
use crate::proving_key::witness_rebuild::{rebuild_all_witness_libs, rebuild_snark_witness_libs};

/// Options for the `rebuild-witness-libs` subcommand.
pub struct RebuildWitnessOptions {
    /// Path to the `provingKey/` directory.
    pub proving_key: String,
    /// Optional path to the `provingKeySnark/` directory. When set, the snark
    /// witness libraries (`recursivef`, `final`) are rebuilt too.
    pub proving_key_snark: Option<String>,
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

    if let Some(proving_key_snark) = opts.proving_key_snark.as_deref() {
        // `final` is BN128 (fr.cpp/fr.asm) — use the dedicated helpers dir, not
        // the goldilocks one `recursivef` uses.
        let final_snark_circom_helpers_dir =
            resolve_path_env("FINAL_SNARK_CIRCOM_HELPERS_DIR", "setup/final_snark_circom");
        tracing::info!("Rebuilding snark witness libraries from {}", proving_key_snark);
        let snark_count = rebuild_snark_witness_libs(
            proving_key_snark,
            &circom_helpers_dir,
            &final_snark_circom_helpers_dir,
            &witness_tracker,
            opts.jobs,
        )
        .context("rebuild_snark_witness_libs failed")?;
        tracing::info!("Rebuilt {} snark witness library(ies)", snark_count);
    }

    Ok(())
}
