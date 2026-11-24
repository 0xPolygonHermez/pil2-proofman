//! Port of `genRecursiveSetupTest` from `generateRecursiveSetup.js`.
//!
//! Unlike the production `gen_recursive_setup`, this function takes a raw circom
//! file as input (no pil2circom/gencircom steps) and is intended for the
//! `test-recursive` CI example and the `setup-recursive-test` subcommand.

use std::path::PathBuf;

use anyhow::{bail, Result};

use crate::output::witness_gen::WitnessTracker;
use crate::proving_key::pipeline::{self, kinds, PipelineCtx, Tools};
use crate::proving_key::recursive::{recursive_blowup, recursive_stark_settings, RecursiveTemplate};
use proofman_common::hash_family::HashFamily;

/// Run the recursive test setup from a user-provided circom file (`genRecursiveSetupTest()`).
///
/// Every output is named "Compressor"; the stark settings are the real recursion layer's of the
/// template being built (`setup_type`: "compressor" or "aggregation"), so a fixture reproduces the
/// geometry it is meant to debug. Writes globalInfo and globalConstraints at the end.
#[allow(clippy::too_many_arguments)]
pub fn gen_recursive_test_setup(
    build_dir: &str,
    circom_path: &str,
    circom_name: &str,
    setup_type: &str,
    hash: &str,
    circom_exec: &str,
    circuits_gl_path: &str,
    recurser_circuits_path: &str,
    std_pil_path: &str,
    recurser_pil_path: &str,
    circom_helpers_dir: &str,
    witness_tracker: &WitnessTracker,
    blake3_lanes: Option<usize>,
    recursion_settings: &crate::types::stark_struct::StarkSettings,
    blowup_override: Option<usize>,
    min_n_bits: Option<usize>,
) -> Result<()> {
    let family: HashFamily = hash.parse().map_err(anyhow::Error::msg)?;
    let template = match setup_type {
        "compressor" => RecursiveTemplate::Compressor,
        "aggregation" => RecursiveTemplate::Recursive2,
        _ => bail!("Invalid setup type '{}'. Must be one of: compressor, aggregation", setup_type),
    };
    // JS nameFile is always "Compressor" regardless of the setup type.
    const NAME_FILE: &str = "Compressor";
    let files_dir = PathBuf::from(build_dir)
        .join("provingKey")
        .join("build")
        .join(NAME_FILE)
        .join("airs")
        .join(NAME_FILE)
        .join("air");

    // `--blowup` overrides the template's; the constraint degree and finalDegree's twin follow it.
    let blowup = blowup_override.unwrap_or_else(|| recursive_blowup(template, family));
    let mut settings = recursive_stark_settings(template, family, recursion_settings);
    settings.initial_blowup_factor = Some(blowup);

    let mut ctx = PipelineCtx::new(build_dir.into(), files_dir, NAME_FILE, family);
    ctx.label = format!("Circuit '{circom_name}'");
    ctx.tools = Tools {
        circom_exec: circom_exec.into(),
        libs: vec![recurser_circuits_path.into(), circuits_gl_path.into()],
        std_pil_path: std_pil_path.into(),
        recurser_pil_path: recurser_pil_path.into(),
        circom_helpers_dir: circom_helpers_dir.into(),
    };
    ctx.witness = Some(witness_tracker.clone());
    ctx.circom_file = Some(circom_path.into());
    ctx.settings = settings;
    // The test circuit is proven as air 0 of airgroup 0.
    ctx.key = crate::pil::stark_setup::KeyIdentity {
        name: Some(NAME_FILE.into()),
        circuit_type: Some(proofman_common::circuit_type_of(0, 1)),
        ..Default::default()
    };
    let cfg = kinds::TestCfg { family, template, blowup, blake3_lanes, min_n_bits };
    pipeline::run(&kinds::steps_for_test(&cfg), &mut ctx)?;

    println!("files Generated Correctly");
    Ok(())
}

#[cfg(test)]
mod std_pil_path_tests {
    /// The generator takes `std_pil_path` from its caller; it does not read the env itself.
    #[test]
    fn std_pil_path_is_a_parameter() {
        let src = include_str!("recursive_test.rs");
        let needle = ["resolve_path_env(", "\"STD_PIL_PATH\""].concat();
        let body = src.split("#[cfg(test)]").next().unwrap();
        assert!(!body.contains(&needle), "generator must not resolve STD_PIL_PATH itself");
        assert!(body.contains("std_pil_path: &str"), "signature must carry std_pil_path");
    }
}
