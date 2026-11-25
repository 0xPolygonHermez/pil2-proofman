//! The vadcop_final proving key: one recursive2 verifier per airgroup and the global constraints
//! (`pipeline::kinds::steps_for_vadcop_final` lists the steps).

use std::fs;
use std::path::PathBuf;

use anyhow::{Context, Result};
use pil2_stark_recurser::stark2circom::CircomGenOptions;
use proofman_common::hash_family::{self as hf, HashFamily};
use serde_json::Value;

use crate::commands::common::parse_root_strings;

use crate::output::witness_gen::WitnessTracker;
use crate::proving_key::pipeline::{self, kinds, CircuitInput, PipelineCtx, Tools, VerifierInput};
use crate::types::stark_struct::StarkSettings;

/// Configuration for the final setup.
pub struct FinalSetupConfig<'a> {
    pub build_dir: &'a str,
    pub hash: HashFamily,
    pub global_info: &'a Value,
    pub global_constraints: &'a Value,
    pub tools: &'a Tools,
}

/// Result of the final setup.
pub struct FinalSetupResult {
    pub stark_info: Value,
    pub verifier_info: Value,
    pub const_root: [u64; 4],
}

/// Run the final setup (`genFinalSetup()`).
pub fn gen_final_setup(config: &FinalSetupConfig<'_>, witness_tracker: &WitnessTracker) -> Result<FinalSetupResult> {
    let build_dir = PathBuf::from(config.build_dir);
    let global_name = config.global_info.get("name").and_then(|v| v.as_str()).unwrap_or("pilout");
    let air_groups = config.global_info["air_groups"].as_array().context("globalInfo has no air_groups")?;

    let mut stark_infos = Vec::new();
    let mut verifier_infos = Vec::new();
    let mut agg_keys_recursive2: Vec<[String; 4]> = Vec::new();
    let mut basic_keys_recursive1 = Vec::new();
    let mut verifier_names = Vec::new();

    // Each airgroup's recursive2: its setup, its own root and the recursive1 roots it accepts.
    for ag in air_groups {
        let ag_name = ag.as_str().context("an airgroup without a name")?;
        let r2_dir = build_dir.join("provingKey").join(global_name).join(ag_name).join("recursive2");
        let read = |f: &str| -> Result<Value> {
            let p = r2_dir.join(f);
            let s = fs::read_to_string(&p).with_context(|| {
                format!(
                    "Recursive2 artifacts not found for airgroup '{ag_name}' ({}): run the recursive setup first",
                    p.display()
                )
            })?;
            Ok(serde_json::from_str(&s)?)
        };
        stark_infos.push(read("recursive2.starkinfo.json")?);
        verifier_infos.push(read("recursive2.verifierinfo.json")?);
        let vks = read("recursive2.vks.json")?;
        let root = |v: &Value, what: &str| parse_root_strings(v).with_context(|| format!("{ag_name} vks.json {what}"));
        agg_keys_recursive2.push(root(&vks["rootCRecursive2"], "rootCRecursive2")?);
        let r1 = vks["rootCRecursives1"]
            .as_array()
            .with_context(|| format!("{ag_name} vks.json has no rootCRecursives1"))?;
        basic_keys_recursive1
            .push(r1.iter().map(|vk| Ok(root(vk, "rootCRecursives1")?.to_vec())).collect::<Result<Vec<_>>>()?);
        verifier_names.push(format!("{ag_name}_recursive2.verifier.circom"));
    }

    let files_dir = build_dir.join("provingKey").join(global_name).join("vadcop_final");
    let family = config.hash;
    let mut ctx = PipelineCtx::new(build_dir, files_dir, "vadcop_final", family);
    ctx.tools = config.tools.clone();
    ctx.witness = Some(witness_tracker.clone());
    // One verifier per airgroup's recursive2, from the recursive2 starkinfo (not recursive1's).
    ctx.verifiers = stark_infos
        .iter()
        .zip(&verifier_infos)
        .zip(&verifier_names)
        .zip(&agg_keys_recursive2)
        .map(|(((si, vi), file), root)| VerifierInput {
            file: file.clone(),
            const_root: root.clone(),
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
        agg_vk: agg_keys_recursive2.iter().map(|k| k.to_vec()).collect(),
        publics: vec![],
        opts: CircomGenOptions {
            airgroup_id: None,
            has_compressor: false,
            has_recursion: false,
            agg_arity: 0,
            batch_size: 1,
        },
    });
    ctx.settings = StarkSettings {
        initial_blowup_factor: Some(hf::final_blowup_factor(family)),
        initial_folding_factor: Some(4),
        // Per family: poseidon's 22 is what its committed verifiers encode, blake3 affords 24.
        grinding_bits: Some(hf::final_grinding_bits(family)),
        // The family's terminal FRI domain (log2) minus the blowup, as a log-degree bound: poseidon's
        // 5 - 4 = 1 is what its committed verifier encodes; a solved schedule takes it as a ceiling.
        final_degree: Some(hf::fri_terminal_degree(family) - hf::final_blowup_factor(family)),
        // Paid by the recursivef that verifies this proof: the family value every recursion layer takes.
        last_level_verification: hf::recursive_last_level_verification(family),
        ..Default::default()
    };
    ctx.key = crate::pil::stark_setup::KeyIdentity { name: Some("vadcop_final".into()), ..Default::default() };

    let _span = tracing::info_span!("stage", t = "vadcop_final").entered();
    pipeline::run(&kinds::steps_for_vadcop_final(&kinds::VadcopFinalCfg { family }), &mut ctx)?;
    let (stark_info, verifier_info, _) = ctx.pil_info.take().context("vadcop_final produced no starkinfo")?;
    let const_root = ctx.const_root.context("vadcop_final produced no const root")?;
    Ok(FinalSetupResult { stark_info, verifier_info, const_root })
}
