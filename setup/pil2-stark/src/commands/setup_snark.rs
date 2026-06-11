//! Setup-snark command: orchestrate the final SNARK setup from vadcop_final artifacts.

use std::fs;
use std::path::PathBuf;

use anyhow::{bail, Context, Result};
use serde_json::Value;

use crate::output::witness_gen::WitnessTracker;
use crate::proving_key::snark_setup::{gen_snark_setup, SnarkSetupConfig};
use crate::commands::recursive_setup::{resolve_circom_exec, resolve_path_env};

/// Options for the setup-snark subcommand.
pub struct SetupSnarkOptions {
    /// Build directory (must contain provingKey/).
    pub build_dir: String,
    /// Powers-of-tau (.ptau) file path.
    pub powers_of_tau: Option<String>,
    /// SNARK type: "fflonk" (default) or "plonk".
    pub final_snark: String,
    /// Optional path to publics hash info JSON.
    pub publics_info: Option<String>,
    /// Only generate the recursivef step; skip the final SNARK.
    pub only_recursive_final: bool,
}

/// Run the setup-snark pipeline.
pub fn run_setup_snark(opts: &SetupSnarkOptions) -> Result<()> {
    let build_dir = &opts.build_dir;

    // Read globalInfo to get the name.
    let global_info_path = PathBuf::from(build_dir).join("provingKey").join("pilout.globalInfo.json");

    if !global_info_path.exists() {
        bail!("Global info file not found: {:?}. Run the regular setup first.", global_info_path);
    }

    let global_info: Value = serde_json::from_str(&fs::read_to_string(&global_info_path)?)?;
    let name = global_info.get("name").and_then(|v| v.as_str()).unwrap_or("pilout").to_string();

    tracing::info!("setup-snark: name='{}', build_dir='{}'", name, build_dir);

    // Read vadcop_final artifacts.
    let vadcop_dir = PathBuf::from(build_dir).join("provingKey").join(&name).join("vadcop_final");

    let const_root_path = vadcop_dir.join("vadcop_final.verkey.json");
    let starkinfo_path = vadcop_dir.join("vadcop_final.starkinfo.json");
    let verifier_info_path = vadcop_dir.join("vadcop_final.verifierinfo.json");

    for p in [&const_root_path, &starkinfo_path, &verifier_info_path] {
        if !p.exists() {
            bail!("Required file not found: {:?}. Make sure you have run the regular setup first.", p);
        }
    }

    let const_root_json: Value = serde_json::from_str(&fs::read_to_string(&const_root_path)?)?;
    let const_root: [u64; 4] =
        parse_const_root(&const_root_json).context("Failed to parse vadcop_final.verkey.json")?;

    let stark_info: Value = serde_json::from_str(&fs::read_to_string(&starkinfo_path)?)?;
    let verifier_info: Value = serde_json::from_str(&fs::read_to_string(&verifier_info_path)?)?;

    // Read optional publics info.
    let publics_info: Option<Value> = if let Some(ref pi_path) = opts.publics_info {
        let content =
            fs::read_to_string(pi_path).with_context(|| format!("Failed to read publics info: {}", pi_path))?;
        Some(serde_json::from_str(&content)?)
    } else {
        None
    };

    // Resolve tool paths (same logic as recursive_setup).
    let circuits_gl_path =
        resolve_path_env("CIRCUITS_GL_PATH", "setup/stark-recurser/stark2circom/circom_verifier/circuits.gl");
    let recurser_circuits_path =
        resolve_path_env("RECURSER_CIRCUITS_PATH", "setup/stark-recurser/stark2circom/circom_verifier/helper_circuits");
    let std_pil_path = resolve_path_env("STD_PIL_PATH", "pil2-components/lib/std/pil");
    let recurser_pil_path = resolve_path_env("RECURSER_PIL_PATH", "setup/stark-recurser/plonk2pil/pil");
    let circom_helpers_dir = resolve_path_env("CIRCOM_HELPERS_DIR", "setup/circom");
    let final_snark_circom_helpers_dir = resolve_path_env("FINAL_SNARK_CIRCOM_HELPERS_DIR", "setup/final_snark_circom");
    let goldilocks_src_dir = resolve_path_env("GOLDILOCKS_SRC_DIR", "pil2-stark/src/goldilocks/src");
    let circom_exec = resolve_circom_exec(&circom_helpers_dir);

    // BN128 and circomlib paths.
    let circuits_bn128_path =
        resolve_path_env("CIRCUITS_BN128_PATH", "setup/stark-recurser/stark2circom/circom_verifier/circuits.bn128");
    let circomlib_path =
        crate::proving_key::recursive::ensure_node_module_subpath("CIRCOMLIB_PATH", "circomlib", "circuits");

    // Create provingKeySnark directory.
    let snark_dir = PathBuf::from(build_dir).join("provingKeySnark");
    fs::create_dir_all(&snark_dir)?;

    let witness_tracker = WitnessTracker::with_goldilocks_src(&goldilocks_src_dir);

    let snark_config = SnarkSetupConfig {
        build_dir,
        name: &name,
        circom_exec: &circom_exec,
        circuits_gl_path: &circuits_gl_path,
        circuits_bn128_path: &circuits_bn128_path,
        circomlib_path: &circomlib_path,
        recurser_circuits_path: &recurser_circuits_path,
        std_pil_path: &std_pil_path,
        recurser_pil_path: &recurser_pil_path,
        circom_helpers_dir: &circom_helpers_dir,
        final_snark_circom_helpers_dir: &final_snark_circom_helpers_dir,
        powers_of_tau: opts.powers_of_tau.as_deref(),
        final_snark: &opts.final_snark,
        publics_info,
        only_recursive_final: opts.only_recursive_final,
    };

    gen_snark_setup(&snark_config, &witness_tracker, &const_root, &stark_info, &verifier_info)
        .context("Final SNARK setup failed")?;

    tracing::info!("setup-snark completed successfully");
    Ok(())
}

/// Parse a vadcop_final.verkey.json array ([[u64;4]]) into [u64;4].
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
