//! The vadcop_final_compressed proving key: a proof of the vadcop_final proof, sized for proof
//! length (`pipeline::kinds::steps_for_compressed_final` lists the steps).

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
    pub hash: HashFamily,
    pub name: &'a str,
    pub const_root: &'a [String; 4],
    pub stark_info: &'a Value,
    pub verifier_info: &'a Value,
    pub tools: &'a Tools,
    /// Low-degree test of the compressed proof; `None` is STIR.
    pub low_degree_test: Option<crate::types::stark_struct::LowDegreeTestKind>,
}

/// Run the compressed final setup (`genCompressedFinalSetup()`).
pub fn gen_compressed_final_setup(config: &CompressedFinalConfig<'_>, witness_tracker: &WitnessTracker) -> Result<()> {
    let family = config.hash;
    const NAME: &str = "vadcop_final_compressed";
    let files_dir = PathBuf::from(config.build_dir).join("provingKey").join(config.name).join(NAME);
    let mut ctx = PipelineCtx::new(config.build_dir.into(), files_dir, NAME, family);
    ctx.tools = config.tools.clone();
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
        basic_vk: vec![],
        agg_vk: vec![],
        publics: vec![],
        opts: CircomGenOptions {
            airgroup_id: None,
            has_compressor: false,
            has_recursion: false,
            agg_arity: 0,
            batch_size: 1,
        },
    });
    ctx.key = crate::pil::stark_setup::KeyIdentity { name: Some("compressed_final".into()), ..Default::default() };
    let _span = tracing::info_span!("stage", t = NAME).entered();
    let cfg = kinds::CompressedFinalCfg { family, low_degree_test: config.low_degree_test };
    pipeline::run(&kinds::steps_for_compressed_final(&cfg), &mut ctx)
}
