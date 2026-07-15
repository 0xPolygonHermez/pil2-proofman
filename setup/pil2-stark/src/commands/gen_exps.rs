//! Generate per-AIR Q-expression CUDA kernels (`.exps.so`) for an existing
//! `provingKey/` tree, without re-running setup.
//!
//! The proving key directory is the only required input: pass its path with
//! `--proving-key`. Each AIR's `.exps.so` is placed next to its `.bin`; the
//! prover loads it by convention and falls back to the bytecode interpreter for
//! any AIR without one. No-op (logged) if `nvcc` is not on PATH.

use std::path::PathBuf;

use anyhow::Result;

use crate::commands::setup::nvcc_present;

/// Options for the `gen-exps` subcommand and the `--gen-exps` step of `setup`.
pub struct GenExpsOptions {
    /// Path to the `provingKey/` directory (globbed for each AIR's
    /// `*.starkinfo.json` + `*.expressionsinfo.json`).
    pub proving_key: PathBuf,
    /// CUDA arch spec: `auto` | `major` | e.g. `89,120` / `sm_120`.
    pub arch: String,
    /// Skip an AIR whose Q has more than this many ops (stays on the interpreter).
    pub cap: usize,
    /// Fixed ops/chunk for every AIR; `None` => the no-spill autotuner.
    pub chunk: Option<usize>,
    /// pil2-stark source root for the nvcc includes; `None` resolves relative to
    /// the exps-codegen crate.
    pub stark_src: Option<PathBuf>,
}

/// Generate + compile the expression kernels for every AIR under `proving_key`.
///
/// Returns `Ok(())` when `nvcc` is missing (the prover uses the interpreter) so
/// the caller can treat a toolchain-less host as success rather than an error.
pub fn run_gen_exps(opts: &GenExpsOptions) -> Result<()> {
    if !nvcc_present() {
        tracing::warn!(
            "gen-exps requested but nvcc not found on PATH; skipping expression kernel codegen (AIRs use the interpreter)"
        );
        return Ok(());
    }

    tracing::info!("Generating expression kernels (.exps.so) under {}", opts.proving_key.display());
    let cfg = exps_codegen::GenConfig {
        cap: opts.cap,
        chunk: opts.chunk,
        archspec: opts.arch.clone(),
        stark_src: opts.stark_src.clone(),
        keep_dir: None,
        dry_run: false,
    };
    let summary = exps_codegen::generate_all(&opts.proving_key, &cfg)?;
    tracing::info!(
        "Expression kernels: {} generated -> {} .exps.so ({} skipped)",
        summary.generated.len(),
        summary.placed,
        summary.skipped.len()
    );
    Ok(())
}
