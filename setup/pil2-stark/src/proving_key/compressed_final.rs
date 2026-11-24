//! Port of `generateCompressedFinalSetup.js`: generate the vadcop_final_compressed
//! circuit, compile it, and produce the proving key artifacts.
//!
//! The compressed final setup:
//! 1. Generates a verifier circom for vadcop_final using pil2circom
//! 2. Generates the compressed final circom via gencircom
//! 3. Compiles circom to R1CS + C++
//! 4. Converts R1CS to PIL (plonk2pil with "aggregation")
//! 5. Compiles PIL
//! 6. Runs starkSetup with specific compressed final settings
//! 7. Computes constant tree
//! 8. Writes verifier.rs for the compressed final circuit

use std::path::PathBuf;

use anyhow::Result;
use pil2_stark_recurser::stark2circom::CircomGenOptions;
use serde_json::Value;

use crate::output::witness_gen::WitnessTracker;
use crate::proving_key::pipeline::{self, kinds, CircuitInput, PipelineCtx, Tools, VerifierInput};
use proofman_common::hash_family::HashFamily;

/// Configuration for the compressed final setup.
pub struct CompressedFinalConfig<'a> {
    pub build_dir: &'a str,
    pub hash: &'a str,
    pub name: &'a str,
    pub const_root: &'a [String; 4],
    pub verification_keys: &'a [Vec<Vec<String>>],
    pub stark_info: &'a Value,
    pub verifier_info: &'a Value,

    // Tool paths
    pub circom_exec: &'a str,
    pub circuits_gl_path: &'a str,
    pub recurser_circuits_path: &'a str,
    pub std_pil_path: &'a str,
    pub recurser_pil_path: &'a str,
    pub circom_helpers_dir: &'a str,
    /// Low-degree test of the compressed proof; `None` is the family's (`compressed_final_stir`).
    pub low_degree_test: Option<crate::types::stark_struct::LowDegreeTestKind>,
}

/// Run the compressed final setup (`genCompressedFinalSetup()`).
pub fn gen_compressed_final_setup(config: &CompressedFinalConfig<'_>, witness_tracker: &WitnessTracker) -> Result<()> {
    let family: HashFamily = config.hash.parse().map_err(anyhow::Error::msg)?;
    const NAME: &str = "vadcop_final_compressed";
    let files_dir = PathBuf::from(config.build_dir).join("provingKey").join(config.name).join(NAME);
    let mut ctx = PipelineCtx::new(config.build_dir.into(), files_dir, NAME, family);
    ctx.tools = Tools {
        circom_exec: config.circom_exec.into(),
        libs: vec![config.recurser_circuits_path.into(), config.circuits_gl_path.into()],
        std_pil_path: config.std_pil_path.into(),
        recurser_pil_path: config.recurser_pil_path.into(),
        circom_helpers_dir: config.circom_helpers_dir.into(),
    };
    ctx.witness = Some(witness_tracker.clone());
    ctx.verifiers = vec![VerifierInput {
        file: "vadcop_final_stark.verifier.circom".into(),
        const_root: config.const_root.clone(),
        stark_info: config.stark_info.clone(),
        verifier_info: config.verifier_info.clone(),
    }];
    // Aggregates nothing: the template never reads `agg_arity`, and `gen_recursive2` rejects 0.
    ctx.circuit_input = Some(CircuitInput {
        stark_infos: vec![config.stark_info.clone()],
        vadcop_info: Value::Null,
        basic_vk: config.verification_keys.to_vec(),
        agg_vk: vec![],
        publics: vec![],
        opts: CircomGenOptions {
            airgroup_id: None,
            has_compressor: false,
            has_recursion: false,
            is_final: false,
            agg_arity: 0,
            batch_size: 1,
        },
    });
    ctx.key = crate::pil::stark_setup::KeyIdentity { name: Some("compressed_final".into()), ..Default::default() };
    let _span = tracing::info_span!("stage", t = NAME).entered();
    let cfg = kinds::CompressedFinalCfg { family, low_degree_test: config.low_degree_test };
    pipeline::run(&kinds::steps_for_compressed_final(&cfg), &mut ctx)
}

#[cfg(test)]
mod missing_const_tests {
    /// The const-tree block bails on a missing const file instead of warning and returning a zero root.
    #[test]
    fn missing_const_file_is_an_error() {
        let p = std::path::Path::new("/nonexistent/compressed_final.const");
        assert!(crate::proving_key::util::require_const_file("compressed_final", p).is_err());
    }
}
