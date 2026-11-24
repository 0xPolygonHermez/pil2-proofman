//! Port of `generateFinalSetup.js`: generate the vadcop_final circuit,
//! compile it, and produce the proving key artifacts.
//!
//! The final setup:
//! 1. Reads all recursive2 starkInfo/verifierInfo/vks for each airgroup
//! 2. Generates the final circom via gencircom with the "final" template
//! 3. Compiles circom to R1CS + C++
//! 4. Converts R1CS to PIL
//! 5. Compiles PIL
//! 6. Runs starkSetup with specific final settings
//! 7. Computes constant tree and writes all artifacts
//! 8. Writes verifier.rs for the final circuit

use std::fs;
use std::path::PathBuf;

use anyhow::{bail, Context, Result};
use pil2_stark_recurser::stark2circom::CircomGenOptions;
use proofman_common::hash_family as hf;
use serde_json::Value;

use crate::output::witness_gen::WitnessTracker;
use crate::proving_key::pipeline::{self, kinds, CircuitInput, PipelineCtx, Tools, VerifierInput};
use crate::types::stark_struct::StarkSettings;

/// Configuration for the final setup.
pub struct FinalSetupConfig<'a> {
    pub build_dir: &'a str,
    pub hash: &'a str,
    pub global_info: &'a Value,
    pub global_constraints: &'a Value,

    // Tool paths
    pub circom_exec: &'a str,
    pub circuits_gl_path: &'a str,
    pub recurser_circuits_path: &'a str,
    pub std_pil_path: &'a str,
    pub recurser_pil_path: &'a str,
    pub circom_helpers_dir: &'a str,
}

/// Result of the final setup.
pub struct FinalSetupResult {
    pub stark_info: Value,
    pub verifier_info: Value,
    pub const_root: [u64; 4],
}

/// Parse a JSON value that may be either a string or a u64 number into a String.
/// Used when reading vks.json fields that are stored as u64 integers.
fn json_val_to_str(v: &Value) -> String {
    v.as_str().map(|s| s.to_string()).or_else(|| v.as_u64().map(|n| n.to_string())).unwrap_or_else(|| "0".to_string())
}

/// Run the final setup (`genFinalSetup()`).
pub fn gen_final_setup(config: &FinalSetupConfig<'_>, witness_tracker: &WitnessTracker) -> Result<FinalSetupResult> {
    let build_dir = PathBuf::from(config.build_dir);
    let global_name = config.global_info.get("name").and_then(|v| v.as_str()).unwrap_or("pilout");

    let air_groups: Vec<String> = config
        .global_info
        .get("air_groups")
        .and_then(|v| v.as_array())
        .map(|a| a.iter().map(|v| v.as_str().unwrap_or("unnamed").to_string()).collect())
        .unwrap_or_default();

    let mut stark_infos = Vec::new();
    let mut verifier_infos = Vec::new();
    let mut agg_keys_recursive2 = Vec::new();
    let mut basic_keys_recursive1 = Vec::new();
    let mut verifier_names = Vec::new();

    // Read recursive2 artifacts for each airgroup
    for ag_name in air_groups.iter() {
        let r2_dir = build_dir.join("provingKey").join(global_name).join(ag_name).join("recursive2");

        let si_path = r2_dir.join("recursive2.starkinfo.json");
        let vi_path = r2_dir.join("recursive2.verifierinfo.json");
        let vks_path = r2_dir.join("recursive2.vks.json");

        if si_path.exists() && vi_path.exists() && vks_path.exists() {
            let si: Value = serde_json::from_str(&fs::read_to_string(&si_path)?)?;
            let vi: Value = serde_json::from_str(&fs::read_to_string(&vi_path)?)?;
            let vks: Value = serde_json::from_str(&fs::read_to_string(&vks_path)?)?;

            stark_infos.push(si);
            verifier_infos.push(vi);

            if let Some(root) = vks.get("rootCRecursive2") {
                agg_keys_recursive2.push(
                    root.as_array().map(|a| a.iter().map(json_val_to_str).collect::<Vec<_>>()).unwrap_or_default(),
                );
            } else {
                agg_keys_recursive2.push(vec![]);
            }

            if let Some(keys) = vks.get("rootCRecursives1") {
                // rootCRecursives1 = [[v0,v1,v2,v3], [v0,...], ...] — flat 2-D
                basic_keys_recursive1.push(
                    keys.as_array()
                        .map(|airs| {
                            airs.iter()
                                .map(|air_vk| {
                                    air_vk
                                        .as_array()
                                        .map(|vals| vals.iter().map(json_val_to_str).collect::<Vec<_>>())
                                        .unwrap_or_default()
                                })
                                .collect::<Vec<_>>()
                        })
                        .unwrap_or_default(),
                );
            } else {
                basic_keys_recursive1.push(vec![]);
            }
        } else {
            bail!(
                "Recursive2 artifacts not found for airgroup '{}'. \
                 Run recursive setup first.",
                ag_name
            );
        }

        verifier_names.push(format!("{}_recursive2.verifier.circom", ag_name));
    }

    let files_dir = build_dir.join("provingKey").join(global_name).join("vadcop_final");
    let family = config.hash.parse().map_err(anyhow::Error::msg)?;
    let mut ctx = PipelineCtx::new(build_dir, files_dir, "vadcop_final", family);
    ctx.tools = Tools {
        circom_exec: config.circom_exec.into(),
        libs: vec![config.recurser_circuits_path.into(), config.circuits_gl_path.into()],
        std_pil_path: config.std_pil_path.into(),
        recurser_pil_path: config.recurser_pil_path.into(),
        circom_helpers_dir: config.circom_helpers_dir.into(),
    };
    ctx.witness = Some(witness_tracker.clone());
    // One verifier per airgroup's recursive2, from the recursive2 starkinfo (not recursive1's).
    ctx.verifiers = stark_infos
        .iter()
        .zip(&verifier_infos)
        .zip(&verifier_names)
        .enumerate()
        .map(|(i, ((si, vi), file))| VerifierInput {
            file: file.clone(),
            const_root: agg_keys_recursive2
                .get(i)
                .filter(|k| k.len() == 4)
                .map(|k| [k[0].clone(), k[1].clone(), k[2].clone(), k[3].clone()])
                .unwrap_or_else(|| ["0".into(), "0".into(), "0".into(), "0".into()]),
            stark_info: si.clone(),
            verifier_info: vi.clone(),
        })
        .collect();

    // The global info the template reads carries the global constraints.
    let mut final_global_info = config.global_info.clone();
    if let Some(constraints) = config.global_constraints.get("constraints") {
        final_global_info.as_object_mut().map(|obj| obj.insert("globalConstraints".to_string(), constraints.clone()));
    }
    // One proof per airgroup, not an aggregation fan-in: the template never reads `agg_arity`.
    ctx.circuit_input = Some(CircuitInput {
        stark_infos,
        vadcop_info: final_global_info,
        basic_vk: basic_keys_recursive1,
        agg_vk: agg_keys_recursive2,
        publics: vec![],
        opts: CircomGenOptions {
            airgroup_id: None,
            has_compressor: false,
            has_recursion: false,
            is_final: true,
            agg_arity: 0,
            batch_size: 1,
        },
    });
    let hash = config.hash;
    ctx.settings = StarkSettings {
        initial_blowup_factor: Some(hf::final_blowup_factor(hash)),
        initial_folding_factor: Some(4),
        // Per family: poseidon's 22 is what its committed verifiers encode, blake3 affords 24.
        grinding_bits: Some(hf::final_grinding_bits(hash)),
        // The family's terminal FRI domain (log2) minus the blowup, as a log-degree bound: poseidon's
        // 5 - 4 = 1 is what its committed verifier encodes; a solved schedule takes it as a ceiling.
        final_degree: Some(hf::fri_terminal_degree(hash) - hf::final_blowup_factor(hash)),
        // Paid by the recursivef that verifies this proof: the family value every recursion layer takes.
        last_level_verification: hf::recursive_last_level_verification(hash),
        ..Default::default()
    };
    ctx.key = crate::pil::stark_setup::KeyIdentity { name: Some("vadcop_final".into()), ..Default::default() };

    let _span = tracing::info_span!("stage", t = "vadcop_final").entered();
    pipeline::run(&kinds::steps_for_vadcop_final(&kinds::VadcopFinalCfg { family }), &mut ctx)?;
    let (stark_info, verifier_info, _) = ctx.pil_info.take().context("vadcop_final produced no starkinfo")?;
    let const_root = ctx.const_root.context("vadcop_final produced no const root")?;
    Ok(FinalSetupResult { stark_info, verifier_info, const_root })
}

#[cfg(test)]
mod missing_const_tests {
    /// The const-tree block bails on a missing const file instead of warning and returning a zero root.
    #[test]
    fn missing_const_file_is_an_error() {
        let p = std::path::Path::new("/nonexistent/vadcop_final.const");
        assert!(crate::proving_key::util::require_const_file("vadcop_final", p).is_err());
    }
}
