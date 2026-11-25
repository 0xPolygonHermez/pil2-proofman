//! Setup-recursivef command: the recursivef of a blake3 proving key, a Goldilocks STARK in the
//! family's own geometry that verifies one vadcop_final proof, in `provingKeySnark/recursivef/`.
//! proofman proves it after the vadcop_final, before the final SNARK.
//!
//! The poseidon families have their recursivef in `setup-snark` instead: a BN128 STARK, the bridge
//! to the final SNARK's circuit, in the same place. blake3 has no hash to switch to
//! (the final circuit hashes blake3 as the recursivef does), so its recursivef is one more layer of
//! the Goldilocks chain, which shortens the proof the final circuit verifies.

use std::fs;
use std::path::PathBuf;

use anyhow::{bail, Context, Result};

use crate::commands::common::{witness_tracker_from_env, RecursiveKey};
use crate::proving_key::pipeline::Tools;
use crate::proving_key::snark_setup::{gen_recursivef_setup, RecursivefSetupConfig};
use crate::types::stark_struct::StarkSettings;

/// The blake3 recursivef's geometry by default: what its BN128 wrap verifies, so set for the wrap's
/// rows, not for a proof's size (41 queries, a 2^19 wrap; blowup 4: 51 queries, 2^20).
pub const RECURSIVEF_BLOWUP: usize = 5;
pub const RECURSIVEF_LANES: usize = 2;
pub const RECURSIVEF_FOLDING: usize = 4;
pub const RECURSIVEF_GRINDING: usize = 27;
pub const RECURSIVEF_LAST_LEVELS: usize = 6;
/// The FRI terminal, as log2 of its domain.
pub const RECURSIVEF_TERMINAL: usize = 8;

/// The recursivef's FRI and grinding knobs.
#[derive(Clone, Copy, Debug)]
pub struct RecursivefGeometry {
    pub blowup: usize,
    pub folding: usize,
    pub grinding: usize,
    pub last_levels: usize,
    pub terminal: usize,
}

/// The geometry of the defaults above, `setup-recursivef`'s without flags.
impl Default for RecursivefGeometry {
    fn default() -> Self {
        RecursivefGeometry {
            blowup: RECURSIVEF_BLOWUP,
            folding: RECURSIVEF_FOLDING,
            grinding: RECURSIVEF_GRINDING,
            last_levels: RECURSIVEF_LAST_LEVELS,
            terminal: RECURSIVEF_TERMINAL,
        }
    }
}

impl RecursivefGeometry {
    /// Its stark struct settings, whose `finalDegree` is a log degree: the terminal's domain less
    /// the blowup, as the final setup's.
    fn settings(&self) -> Result<StarkSettings> {
        let Some(final_degree) = self.terminal.checked_sub(self.blowup) else {
            bail!(
                "the FRI terminal's domain, 2^{}, is below the blowup, 2^{}: it holds no degree bound",
                self.terminal,
                self.blowup
            );
        };
        Ok(StarkSettings {
            initial_blowup_factor: Some(self.blowup),
            initial_folding_factor: Some(self.folding),
            final_degree: Some(final_degree),
            last_level_verification: Some(self.last_levels),
            grinding_bits: Some(self.grinding),
            ..Default::default()
        })
    }
}

pub struct SetupRecursivefOptions {
    /// Build directory containing `provingKey/<name>/vadcop_final/`.
    pub build_dir: String,
    pub geometry: RecursivefGeometry,
    /// Its blake3 lanes ([`RECURSIVEF_LANES`] by default).
    pub lanes: usize,
    /// CUDA arch spec of its Q kernel, `.exps.so` ([`crate::commands::gen_exps`]).
    pub exps_arch: String,
}

pub fn run_setup_recursivef(opts: &SetupRecursivefOptions) -> Result<()> {
    let build_dir = &opts.build_dir;
    let key = RecursiveKey::load(build_dir)?;
    let (hash, name) = (key.hash, &key.name);
    if proofman_common::hash_family::has_bn128_recursivef(hash.as_str()) {
        bail!(
            "the {hash} proving key at {:?} has its recursivef in `setup-snark` (a BN128 STARK, in \
             provingKeySnark/recursivef/); setup-recursivef is for the families without one (blake3)",
            key.global_info_path
        );
    }
    let vf = key.vadcop_final()?;
    let tools = Tools::from_env();
    let witness_tracker = witness_tracker_from_env();
    let config = RecursivefSetupConfig { build_dir, hash, tools: &tools };
    let recursivef_dir = PathBuf::from(build_dir).join("provingKeySnark").join("recursivef");
    fs::create_dir_all(&recursivef_dir)?;
    tracing::info!(
        "Running the {hash} recursivef setup for '{name}' in {}: {:?}, {} lane(s)",
        recursivef_dir.display(),
        opts.geometry,
        opts.lanes
    );
    gen_recursivef_setup(
        &config,
        &witness_tracker,
        &vf.const_root,
        &vf.stark_info,
        &vf.verifier_info,
        &recursivef_dir,
        &opts.geometry.settings()?,
        Some(opts.lanes),
    )
    .context("recursivef setup failed")?;
    // Its Q kernel, as `setup --gen-exps` makes the recursion's: without it the GPU interprets Q
    // (measured 0.32 s against 0.14 s a proof at blowup 3).
    if crate::commands::setup::nvcc_present() {
        match proofman_exps_codegen::generate_air(
            &recursivef_dir,
            &crate::commands::setup::exps_config(&opts.exps_arch),
        ) {
            Ok(so) => tracing::info!("recursivef Q kernel: {}", so.display()),
            Err(e) if e.downcast_ref::<proofman_exps_codegen::CompileFailed>().is_some() => {
                return Err(e.context("recursivef Q kernel"))
            }
            Err(e) => tracing::warn!("recursivef Q kernel skipped (it stays on the interpreter): {e:#}"),
        }
    } else {
        tracing::warn!("nvcc not found: the recursivef's Q stays on the GPU's interpreter (gen-exps makes its kernel)");
    }
    witness_tracker.await_all()?;
    tracing::info!("recursivef setup complete");
    Ok(())
}
