//! `setup-recursive-test`: a raw user circom (no generated verifier) through the recursion's
//! tail, for the `test-recursive` CI example.

use std::path::PathBuf;

use anyhow::{bail, Result};

use crate::output::witness_gen::WitnessTracker;
use crate::proving_key::pipeline::{self, kinds, PipelineCtx, Tools};
use crate::proving_key::recursive::{recursive_blowup, recursive_stark_settings, RecursiveTemplate};
use proofman_common::hash_family::HashFamily;

/// What [`gen_recursive_test_setup`] builds.
pub struct RecursiveTestConfig<'a> {
    pub build_dir: &'a str,
    pub circom_path: &'a str,
    pub circom_name: &'a str,
    /// "compressor" or "aggregation".
    pub setup_type: &'a str,
    pub hash: HashFamily,
    pub tools: &'a Tools,
    pub blake3_lanes: Option<usize>,
    pub recursion_settings: &'a crate::types::stark_struct::StarkSettings,
    /// `None` keeps `recursive_blowup`.
    pub blowup: Option<usize>,
    pub min_n_bits: Option<usize>,
}

/// Run the recursive test setup from a user-provided circom file (`genRecursiveSetupTest()`).
///
/// Every output is named "Compressor"; the stark settings are the real recursion layer's of the
/// template being built, so a fixture reproduces the geometry it is meant to debug. Writes
/// globalInfo and globalConstraints at the end.
pub fn gen_recursive_test_setup(cfg: &RecursiveTestConfig<'_>, witness_tracker: &WitnessTracker) -> Result<()> {
    let family = cfg.hash;
    let template = match cfg.setup_type {
        "compressor" => RecursiveTemplate::Compressor,
        "aggregation" => RecursiveTemplate::Recursive2,
        other => bail!("Invalid setup type '{other}'. Must be one of: compressor, aggregation"),
    };
    // JS nameFile is always "Compressor" regardless of the setup type.
    const NAME_FILE: &str = "Compressor";
    let files_dir = PathBuf::from(cfg.build_dir)
        .join("provingKey")
        .join("build")
        .join(NAME_FILE)
        .join("airs")
        .join(NAME_FILE)
        .join("air");

    // `--blowup` overrides the template's; the constraint degree and finalDegree's twin follow it.
    let blowup = cfg.blowup.unwrap_or_else(|| recursive_blowup(template, family));
    let mut settings = recursive_stark_settings(template, family, cfg.recursion_settings);
    settings.initial_blowup_factor = Some(blowup);

    let mut ctx = PipelineCtx::new(cfg.build_dir.into(), files_dir, NAME_FILE, family);
    ctx.label = format!("Circuit '{}'", cfg.circom_name);
    ctx.tools = cfg.tools.clone();
    ctx.witness = Some(witness_tracker.clone());
    ctx.circom_file = Some(cfg.circom_path.into());
    ctx.settings = settings;
    // The test circuit is proven as air 0 of airgroup 0.
    ctx.key = crate::pil::stark_setup::KeyIdentity {
        name: Some(NAME_FILE.into()),
        circuit_type: Some(proofman_common::circuit_type_of(0, 1)),
        ..Default::default()
    };
    let steps = kinds::TestCfg { family, template, blowup, blake3_lanes: cfg.blake3_lanes, min_n_bits: cfg.min_n_bits };
    pipeline::run(&kinds::steps_for_test(&steps), &mut ctx)?;
    tracing::info!("files Generated Correctly");
    Ok(())
}
