//! Setup-compressed-final command: run only `gen_compressed_final_setup` on top
//! of an existing `provingKey/<name>/vadcop_final/`. Useful for iterating on the
//! compressed_final stage without re-running the full recursive pipeline.

use std::fs;
use std::path::PathBuf;

use anyhow::{bail, Context, Result};
use serde_json::Value;

use crate::commands::recursive_setup::{resolve_circom_exec, resolve_path_env};
use crate::output::witness_gen::WitnessTracker;
use crate::proving_key::compressed_final::{gen_compressed_final_setup, CompressedFinalConfig};

pub struct SetupCompressedFinalOptions {
    /// Build directory containing `provingKey/<name>/vadcop_final/`.
    pub build_dir: String,
}

pub fn run_setup_compressed_final(opts: &SetupCompressedFinalOptions) -> Result<()> {
    let build_dir = &opts.build_dir;

    let global_info_path = PathBuf::from(build_dir).join("provingKey").join("pilout.globalInfo.json");
    if !global_info_path.exists() {
        bail!("Global info file not found: {:?}. Run `setup --recursive` first.", global_info_path);
    }
    let global_info: Value = serde_json::from_str(&fs::read_to_string(&global_info_path)?)?;
    let name = global_info.get("name").and_then(|v| v.as_str()).unwrap_or("pilout").to_string();
    // The hash family must come from the setup that produced this provingKey; a
    // silent default would let the compressed_final stage disagree with the rest
    // of the proof chain and surface only at verification time. Fail loud instead.
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
    // Register the hash family with the linked starks library before any
    // const-tree build. The full `setup` path does this via `set_hash_family_c`
    // (see commands/setup.rs); this standalone command must do it too, otherwise
    // `build_const_tree_c` aborts with "hash family not set in this linked image".
    proofman_starks_lib_c::set_hash_family_c(&hash);

    let vadcop_dir = PathBuf::from(build_dir).join("provingKey").join(&name).join("vadcop_final");
    let const_root_path = vadcop_dir.join("vadcop_final.verkey.json");
    let starkinfo_path = vadcop_dir.join("vadcop_final.starkinfo.json");
    let verifier_info_path = vadcop_dir.join("vadcop_final.verifierinfo.json");
    for p in [&const_root_path, &starkinfo_path, &verifier_info_path] {
        if !p.exists() {
            bail!("Required file not found: {:?}. Run `setup --recursive` first.", p);
        }
    }

    let const_root_json: Value = serde_json::from_str(&fs::read_to_string(&const_root_path)?)?;
    let const_root_u64 = parse_const_root(&const_root_json).context("Failed to parse vadcop_final.verkey.json")?;
    let const_root: [String; 4] = [
        const_root_u64[0].to_string(),
        const_root_u64[1].to_string(),
        const_root_u64[2].to_string(),
        const_root_u64[3].to_string(),
    ];

    let stark_info: Value = serde_json::from_str(&fs::read_to_string(&starkinfo_path)?)?;
    let verifier_info: Value = serde_json::from_str(&fs::read_to_string(&verifier_info_path)?)?;

    let circuits_gl_path =
        resolve_path_env("CIRCUITS_GL_PATH", "setup/stark-recurser/stark2circom/circom_verifier/circuits.gl");
    let recurser_circuits_path = resolve_path_env(
        "RECURSER_CIRCUITS_COMPRESSED_FINAL_PATH",
        "setup/stark-recurser/stark2circom/circom_verifier/helper_circuits",
    );
    let std_pil_path = resolve_path_env("STD_PIL_PATH", "pil2-components/lib/std/pil");
    let recurser_pil_path = resolve_path_env("RECURSER_PIL_PATH", "setup/stark-recurser/plonk2pil/pil");
    let circom_helpers_dir = resolve_path_env("CIRCOM_HELPERS_DIR", "setup/circom");
    let goldilocks_src_dir = resolve_path_env("GOLDILOCKS_SRC_DIR", "pil2-stark/src/goldilocks/src");
    let circom_exec = resolve_circom_exec(&circom_helpers_dir);
    let witness_tracker = WitnessTracker::with_goldilocks_src(&goldilocks_src_dir);

    let config = CompressedFinalConfig {
        build_dir,
        hash: &hash,
        name: &name,
        const_root: &const_root,
        verification_keys: &[],
        stark_info: &stark_info,
        verifier_info: &verifier_info,
        circom_exec: &circom_exec,
        circuits_gl_path: &circuits_gl_path,
        recurser_circuits_path: &recurser_circuits_path,
        std_pil_path: &std_pil_path,
        recurser_pil_path: &recurser_pil_path,
        circom_helpers_dir: &circom_helpers_dir,
    };

    tracing::info!("Running compressed final setup (standalone) for '{}'", name);
    gen_compressed_final_setup(&config, &witness_tracker).context("Compressed final setup failed")?;
    witness_tracker.await_all()?;
    tracing::info!("Compressed final setup complete");
    Ok(())
}

fn parse_const_root(json: &Value) -> Result<[u64; 4]> {
    let arr = json.as_array().ok_or_else(|| anyhow::anyhow!("verkey.json is not an array"))?;
    if arr.len() < 4 {
        bail!("verkey.json has {} elements, expected 4", arr.len());
    }
    let parse_one = |v: &Value, idx: usize| -> Result<u64> {
        v.as_u64()
            .or_else(|| v.as_str()?.parse::<u64>().ok())
            .ok_or_else(|| anyhow::anyhow!("verkey.json element {} is not a valid u64: {}", idx, v))
    };
    Ok([parse_one(&arr[0], 0)?, parse_one(&arr[1], 1)?, parse_one(&arr[2], 2)?, parse_one(&arr[3], 3)?])
}
