//! High-level wrappers for the JS-based circom generation tools.
//!
//! `pil2circom` and `gen_circom` mirror the APIs that `recursive_setup` expects, but are implemented by writing temp files and invoking
//! the Node.js scripts from `stark-recurser` — exactly the same subprocess
//! pattern used throughout `recursive_setup`.

use std::fs;
use std::path::Path;

use anyhow::{Context, Result};
use serde_json::Value;

use crate::proving_key::recursive::{run_gencircom_js, run_pil2circom_js};

// ── pil2circom ───────────────────────────────────────────────────────────────

/// Options controlling verifier circom generation (mirrors the JS API).
#[derive(Debug, Default)]
pub struct Pil2CircomOptions {
    /// Omit the `component main` line so the file can be `include`d.
    pub skip_main: bool,
    /// Pass the constant root as a public input instead of embedding it.
    pub verkey_input: bool,
    /// Emit an `enable` input signal.
    pub enable_input: bool,
    /// Pass challenges as public inputs.
    pub input_challenges: bool,
}

/// Generate a verifier circom by calling `node src/pil2circom/main_pil2circom.js`.
///
/// Writes temporary JSON files for the inputs, invokes the script, and returns
/// the output circom source as a `String`.
pub fn pil2circom(
    const_root: &[String; 4],
    stark_info: &Value,
    verifier_info: &Value,
    opts: &Pil2CircomOptions,
) -> Result<String> {
    let tmp_dir = tempfile::tempdir().context("Failed to create temp dir for pil2circom")?;
    let tmp = tmp_dir.path();

    let si_path = tmp.join("starkinfo.json");
    let vi_path = tmp.join("verifierinfo.json");
    let out_path = tmp.join("verifier.circom");

    fs::write(&si_path, serde_json::to_string(stark_info)?)?;
    fs::write(&vi_path, serde_json::to_string(verifier_info)?)?;

    // When not using verkey as a witness input, embed the constant root in the
    // circom by passing the file to pil2circom.  When verkey_input is true, no
    // verkey file is needed (it comes in as a public signal).
    let vk_path_buf;
    let verkey_path: Option<&Path> = if !opts.verkey_input {
        let vk_json = serde_json::json!({
            "constRoot": const_root
                .iter()
                .map(|s| s.parse::<u64>().unwrap_or(0))
                .collect::<Vec<_>>()
        });
        vk_path_buf = tmp.join("verkey.json");
        fs::write(&vk_path_buf, serde_json::to_string(&vk_json)?)?;
        Some(vk_path_buf.as_path())
    } else {
        None
    };

    run_pil2circom_js(
        &si_path,
        &vi_path,
        verkey_path,
        &out_path,
        opts.skip_main,
        opts.verkey_input,
        opts.enable_input,
        opts.input_challenges,
    )
    .context("pil2circom JS failed")?;

    fs::read_to_string(&out_path).context("Failed to read pil2circom output")
}

// ── gen_circom ───────────────────────────────────────────────────────────────

/// Options controlling recursive/final circom generation.
#[derive(Debug, Default)]
pub struct GenCircomOptions {
    pub airgroup_id: Option<u64>,
    pub has_compressor: bool,
    pub has_recursion: bool,
    /// Selects the final-circuit template; the template path conveys this too,
    /// so this flag is informational and is not forwarded as a CLI flag.
    pub is_final: bool,
}

/// All inputs for a single `gen_circom` call.
pub struct GenCircomInput<'a> {
    /// EJS/Tera template path relative to the stark-recurser package root
    /// (e.g. `"src/vadcop/templates/final.circom.ejs"` or
    /// `"vadcop/final.circom.tera"`).
    pub template_name: &'a str,
    /// StarkInfo JSON objects, one per AIR.
    pub stark_infos: &'a [Value],
    /// The vadcop / global info JSON.
    pub vadcop_info: &'a Value,
    /// Verifier circom filenames (base names, not full paths).
    pub verifier_filenames: &'a [String],
    /// Per-airgroup, per-air constant roots for basic (recursive1) verification.
    pub basic_verification_keys: &'a [Vec<Vec<String>>],
    /// Per-airgroup constant roots for aggregation (recursive2) verification.
    pub agg_verification_keys: &'a [Vec<String>],
    /// Public inputs (currently unused by the script; reserved for future use).
    pub publics: &'a [Value],
    pub options: &'a GenCircomOptions,
}

/// Generate a circom circuit by calling `node src/main_gencircom.js`.
///
/// Writes all inputs to temporary JSON files, invokes the script, and returns
/// the output circom source as a `String`.
pub fn gen_circom(input: &GenCircomInput<'_>) -> Result<String> {
    let tmp_dir = tempfile::tempdir().context("Failed to create temp dir for gen_circom")?;
    let tmp = tmp_dir.path();

    // StarkInfo files (one per AIR)
    let mut si_paths: Vec<std::path::PathBuf> = Vec::new();
    for (i, si) in input.stark_infos.iter().enumerate() {
        let p = tmp.join(format!("starkinfo_{i}.json"));
        fs::write(&p, serde_json::to_string(si)?)?;
        si_paths.push(p);
    }
    let si_refs: Vec<&Path> = si_paths.iter().map(|p| p.as_path()).collect();

    // VadcopInfo / global info
    let vadcop_path = tmp.join("vadcop.json");
    fs::write(&vadcop_path, serde_json::to_string(input.vadcop_info)?)?;

    // Basic (recursive1) verification keys.
    // - final setup (is_final=true): one file PER AIRGROUP where constRoot is an array-of-arrays
    //   [[air0_roots], [air1_roots], ...] so that main_gencircom.js produces
    //   basicVK[airgroup][air][values] as expected by final.circom.ejs (`basicVK[i][j].join(",")`).
    // - other templates (e.g. recursive2.circom.ejs): one file PER AIR, flat, so that
    //   basicVK[air][values] is produced (`basicVK[i].join(",")`).
    let mut basic_vk_paths: Vec<std::path::PathBuf> = Vec::new();
    if input.options.is_final {
        for (ag_idx, ag_vkeys) in input.basic_verification_keys.iter().enumerate() {
            let p = tmp.join(format!("basic_vk_{ag_idx}.json"));
            let vk_json = serde_json::json!({
                "constRoot": ag_vkeys.iter()
                    .map(|vk| vk.iter()
                        .map(|s| s.parse::<u64>().unwrap_or(0))
                        .collect::<Vec<_>>())
                    .collect::<Vec<_>>()
            });
            fs::write(&p, serde_json::to_string(&vk_json)?)?;
            basic_vk_paths.push(p);
        }
    } else {
        for (ag_idx, ag_vkeys) in input.basic_verification_keys.iter().enumerate() {
            for (air_idx, vk) in ag_vkeys.iter().enumerate() {
                let p = tmp.join(format!("basic_vk_{ag_idx}_{air_idx}.json"));
                let vk_json = serde_json::json!({
                    "constRoot": vk.iter()
                        .map(|s| s.parse::<u64>().unwrap_or(0))
                        .collect::<Vec<_>>()
                });
                fs::write(&p, serde_json::to_string(&vk_json)?)?;
                basic_vk_paths.push(p);
            }
        }
    }
    let basic_vk_refs: Vec<&Path> = basic_vk_paths.iter().map(|p| p.as_path()).collect();

    // Aggregation (recursive2) verification keys — one file per airgroup
    let mut agg_vk_paths: Vec<std::path::PathBuf> = Vec::new();
    for (i, vk) in input.agg_verification_keys.iter().enumerate() {
        let p = tmp.join(format!("agg_vk_{i}.json"));
        let vk_json = serde_json::json!({
            "constRoot": vk.iter()
                .map(|s| s.parse::<u64>().unwrap_or(0))
                .collect::<Vec<_>>()
        });
        fs::write(&p, serde_json::to_string(&vk_json)?)?;
        agg_vk_paths.push(p);
    }
    let agg_vk_refs: Vec<&Path> = agg_vk_paths.iter().map(|p| p.as_path()).collect();

    // Publics — one file per public input value
    let mut publics_paths: Vec<std::path::PathBuf> = Vec::new();
    for (i, pub_val) in input.publics.iter().enumerate() {
        let p = tmp.join(format!("publics_{i}.json"));
        fs::write(&p, serde_json::to_string(pub_val)?)?;
        publics_paths.push(p);
    }
    let publics_refs: Vec<&Path> = publics_paths.iter().map(|p| p.as_path()).collect();

    let out_path = tmp.join("output.circom");
    let vf_refs: Vec<&str> = input.verifier_filenames.iter().map(|s| s.as_str()).collect();

    run_gencircom_js(
        input.template_name,
        &si_refs,
        Some(vadcop_path.as_path()),
        &vf_refs,
        &basic_vk_refs,
        &agg_vk_refs,
        &publics_refs,
        &out_path,
        input.options.airgroup_id,
        input.options.has_compressor,
        input.options.has_recursion,
    )
    .context("gencircom JS failed")?;

    fs::read_to_string(&out_path).context("Failed to read gen_circom output")
}
