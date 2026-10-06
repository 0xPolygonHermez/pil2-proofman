//! Setup-recursivef command: the recursivef of a blake3 proving key, a Goldilocks STARK in the
//! family's own geometry that verifies one vadcop_final proof, in `provingKey/<name>/recursivef/`.
//! proofman proves it after the vadcop_final, before the final SNARK.
//!
//! The poseidon families have their recursivef in `setup-snark` instead: a BN128 STARK, the bridge
//! to the final SNARK's circuit, in `provingKeySnark/recursivef/`. blake3 has no hash to switch to
//! (the final circuit hashes blake3 as the recursivef does), so its recursivef is one more layer of
//! the Goldilocks chain, which shortens the proof the final circuit verifies.

use std::fs;
use std::path::PathBuf;

use anyhow::{bail, Context, Result};
use serde_json::Value;

use crate::commands::recursive_setup::{resolve_circom_exec, resolve_path_env};
use crate::commands::setup_snark::parse_const_root;
use crate::output::witness_gen::WitnessTracker;
use crate::proving_key::recursive::{self, RecursiveTemplate::Recursive2};
use crate::proving_key::snark_setup::{gen_recursivef_setup, RecursivefSetupConfig};
use crate::types::stark_struct::StarkSettings;

/// The blake3 recursivef's blowup by default: 3, measured against 4 on the blake3 wrap: 70 queries
/// and an extension of 2^23 (14 GB of prover) against 53 and 2^24 (27.7 GB), a GPU proof of 136 ms
/// against 234 ms, and the same wrap (2^20 rows, the same gas).
pub const RECURSIVEF_BLOWUP: usize = 3;

/// Its lanes by default: it verifies one proof, so what it pays for is its width, which is what its
/// verifier opens (2 lanes: 3% more hashes, for 31% less prover memory).
pub const RECURSIVEF_LANES: usize = 1;

/// The blake3 recursivef's stark struct at `blowup`: the recursion's grinding and kept levels, with
/// finalDegree (a terminal DOMAIN size) shifted by the blowup so that the terminal polynomial's
/// degree is the recursion's.
pub fn recursivef_settings(hash: &str, blowup: usize) -> StarkSettings {
    let terminal =
        proofman_common::hash_family::fri_terminal_degree(hash) - recursive::recursive_blowup(Recursive2, hash);
    StarkSettings {
        blowup_factor: Some(blowup),
        final_degree: Some(terminal + blowup),
        last_level_verification: recursive::recursive_last_level_verification(Recursive2, hash),
        pow_bits: Some(recursive::recursive_grinding_bits(Recursive2, hash)),
        ..Default::default()
    }
}

pub struct SetupRecursivefOptions {
    /// Build directory containing `provingKey/<name>/vadcop_final/`.
    pub build_dir: String,
    /// Its blowup ([`RECURSIVEF_BLOWUP`] by default) and its blake3 lanes ([`RECURSIVEF_LANES`]).
    pub blowup: usize,
    pub lanes: usize,
    /// CUDA arch spec of its Q kernel, `.exps.so` ([`crate::commands::gen_exps`]).
    pub exps_arch: String,
}

pub fn run_setup_recursivef(opts: &SetupRecursivefOptions) -> Result<()> {
    let build_dir = &opts.build_dir;

    let global_info_path = PathBuf::from(build_dir).join("provingKey").join("pilout.globalInfo.json");
    if !global_info_path.exists() {
        bail!("Global info file not found: {:?}. Run `setup --recursive` first.", global_info_path);
    }
    let global_info: Value = serde_json::from_str(&fs::read_to_string(&global_info_path)?)?;
    let name = global_info.get("name").and_then(|v| v.as_str()).unwrap_or("pilout").to_string();
    let hash = global_info
        .get("hash")
        .and_then(|v| v.as_str())
        .with_context(|| format!("'hash' missing from {:?}; re-run `setup --recursive`", global_info_path))?
        .to_string();
    if !proofman_common::hash_family::is_known_family(&hash) {
        bail!(
            "unknown hash family {:?} in {:?}; known: {:?}",
            hash,
            global_info_path,
            proofman_common::hash_family::FAMILIES
        );
    }
    if proofman_common::hash_family::supports_snark(&hash) {
        bail!(
            "the {hash} proving key at {:?} has its recursivef in `setup-snark` (a BN128 STARK, in \
             provingKeySnark/recursivef/); setup-recursivef is for the families without one (blake3)",
            global_info_path
        );
    }
    // The const tree is built with the family's hash, which the linked starks library takes from
    // this process-global (as setup-compressed-final sets it).
    proofman_starks_lib_c::set_hash_family_c(&hash);

    let proving_key_dir = PathBuf::from(build_dir).join("provingKey").join(&name);
    let vadcop_dir = proving_key_dir.join("vadcop_final");
    let const_root_path = vadcop_dir.join("vadcop_final.verkey.json");
    let starkinfo_path = vadcop_dir.join("vadcop_final.starkinfo.json");
    let verifier_info_path = vadcop_dir.join("vadcop_final.verifierinfo.json");
    for p in [&const_root_path, &starkinfo_path, &verifier_info_path] {
        if !p.exists() {
            bail!("Required file not found: {:?}. Run `setup --recursive` first.", p);
        }
    }
    let const_root_json: Value = serde_json::from_str(&fs::read_to_string(&const_root_path)?)?;
    let const_root = parse_const_root(&const_root_json).context("Failed to parse vadcop_final.verkey.json")?;
    let stark_info: Value = serde_json::from_str(&fs::read_to_string(&starkinfo_path)?)?;
    let verifier_info: Value = serde_json::from_str(&fs::read_to_string(&verifier_info_path)?)?;

    let circuits_gl_path =
        resolve_path_env("CIRCUITS_GL_PATH", "setup/stark-recurser/stark2circom/circom_verifier/circuits.gl");
    let recurser_circuits_path =
        resolve_path_env("RECURSER_CIRCUITS_PATH", "setup/stark-recurser/stark2circom/circom_verifier/helper_circuits");
    let std_pil_path = resolve_path_env("STD_PIL_PATH", "pil2-components/lib/std/pil");
    let recurser_pil_path = resolve_path_env("RECURSER_PIL_PATH", "setup/stark-recurser/plonk2pil/pil");
    let circom_helpers_dir = resolve_path_env("CIRCOM_HELPERS_DIR", "setup/circom");
    let goldilocks_src_dir = resolve_path_env("GOLDILOCKS_SRC_DIR", "pil2-stark/src/goldilocks/src");
    let circom_exec = resolve_circom_exec(&circom_helpers_dir);
    let witness_tracker = WitnessTracker::with_goldilocks_src(&goldilocks_src_dir);

    let config = RecursivefSetupConfig {
        build_dir,
        hash: &hash,
        circom_exec: &circom_exec,
        circuits_gl_path: &circuits_gl_path,
        recurser_circuits_path: &recurser_circuits_path,
        std_pil_path: &std_pil_path,
        recurser_pil_path: &recurser_pil_path,
        circom_helpers_dir: &circom_helpers_dir,
    };
    let recursivef_dir = proving_key_dir.join("recursivef");
    tracing::info!(
        "Running the {hash} recursivef setup for '{name}' in {}: blowup {}, {} lane(s)",
        recursivef_dir.display(),
        opts.blowup,
        opts.lanes
    );
    gen_recursivef_setup(
        &config,
        &witness_tracker,
        &const_root,
        &stark_info,
        &verifier_info,
        &recursivef_dir,
        &recursivef_settings(&hash, opts.blowup),
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
            Err(e) => tracing::error!("recursivef Q kernel codegen failed (it stays on the interpreter): {e:#}"),
        }
    } else {
        tracing::warn!("nvcc not found: the recursivef's Q stays on the GPU's interpreter (gen-exps makes its kernel)");
    }
    witness_tracker.await_all()?;
    tracing::info!("recursivef setup complete");
    Ok(())
}
