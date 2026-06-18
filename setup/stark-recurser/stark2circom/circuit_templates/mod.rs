//! Circuit-level template generators (port of `gencircom.js`).
//!
//! Generates recursivef, recursive1/2, compressor, vadcop/final and solidity
//! verifier circuits from `starkInfo` and vadcop metadata.
//!
//! Tera templates live in `circuit_templates/tera/`.
//!
//! Entry point: [`gen_circom_circuit`]

pub mod stark_inputs;
pub(crate) mod vadcop_inputs;
pub(crate) mod calculate_hashes;
pub(crate) mod get_sha256_inputs;
pub(crate) mod verify_global_challenge;
pub(crate) mod verify_global_constraints;
pub mod templates;

use anyhow::{bail, Result};
use serde_json::Value;

pub use templates::{
    gen_recursivef, gen_recursion_final, gen_solidity, gen_iverifier, gen_final_compressed, gen_compressor,
    gen_recursive1, gen_recursive2, gen_vadcop_final,
};

// ── Public types ──────────────────────────────────────────────────────────────

/// Options forwarded from `GenCircomOptions` / CLI `--has*` flags.
#[derive(Debug, Default, Clone)]
pub struct CircomGenOptions {
    pub airgroup_id: Option<usize>,
    pub has_compressor: bool,
    pub has_recursion: bool,
    pub is_final: bool,
}

/// All inputs for a [`gen_circom_circuit`] call (mirrors `gencircom.js` parameters).
pub struct GenCircomCircuitInput<'a> {
    /// EJS template name (relative to stark-recurser root), used to select
    /// which Rust generator to call.
    pub template_name: &'a str,
    /// StarkInfo JSON objects, one per AIR (usually one for recursion templates).
    pub stark_infos: &'a [Value],
    /// Global/vadcop info JSON.
    pub vadcop_info: &'a Value,
    /// Verifier circom filenames (base names, not full paths).
    pub verifier_filenames: &'a [String],
    /// Per-airgroup per-air basic (recursive1) verification keys.
    pub basic_vk: &'a [Vec<Vec<String>>],
    /// Per-airgroup aggregation (recursive2) verification keys.
    pub agg_vk: &'a [Vec<String>],
    /// Optional publics info (JSON from `--publics-info`).
    pub publics: &'a [Value],
    pub options: &'a CircomGenOptions,
}

/// Generate a circom circuit string for the given template.
///
/// Mirrors `genCircom()` from `gencircom.js`.
pub fn gen_circom_circuit(input: &GenCircomCircuitInput<'_>) -> Result<String> {
    // The JS gencircom collapses single-element starkInfos arrays.
    let stark_info = if input.stark_infos.len() == 1 {
        &input.stark_infos[0]
    } else if input.stark_infos.is_empty() {
        &Value::Null
    } else {
        // Multi-starkInfo is used by VADCOP final; handled below per template.
        &Value::Null
    };

    let publics = if input.publics.len() == 1 {
        Some(&input.publics[0])
    } else if input.publics.is_empty() {
        None
    } else {
        bail!("gen_circom_circuit: multi-publics not yet supported");
    };

    let airgroup_id = input.options.airgroup_id.unwrap_or(0);

    match input.template_name {
        "src/recursion/templates/recursivef.circom.ejs" => {
            templates::gen_recursivef(stark_info, input.verifier_filenames, input.options)
        }
        "src/recursion/templates/final.circom.ejs" => {
            templates::gen_recursion_final(stark_info, input.verifier_filenames, publics, input.options)
        }
        "src/vadcop/templates/final_compressed.circom.ejs" => {
            templates::gen_final_compressed(stark_info, input.verifier_filenames, input.options)
        }
        "src/vadcop/templates/compressor.circom.ejs" => {
            templates::gen_compressor(stark_info, input.verifier_filenames, input.vadcop_info, input.options)
        }
        "src/vadcop/templates/recursive1.circom.ejs" => templates::gen_recursive1(
            stark_info,
            input.verifier_filenames,
            input.vadcop_info,
            airgroup_id,
            input.options,
        ),
        "src/vadcop/templates/recursive2.circom.ejs" => {
            // basic_vk[airgroup_id] contains per-air VKs for this airgroup
            let bvk = input.basic_vk.get(airgroup_id).cloned().unwrap_or_default();
            templates::gen_recursive2(
                stark_info,
                input.verifier_filenames,
                input.vadcop_info,
                airgroup_id,
                &bvk,
                input.options,
            )
        }
        "src/vadcop/templates/final.circom.ejs" => {
            // vadcop final uses all stark_infos
            templates::gen_vadcop_final(
                input.stark_infos,
                input.verifier_filenames,
                input.vadcop_info,
                input.basic_vk,
                input.agg_vk,
                input.options,
            )
        }
        other => {
            bail!("gen_circom_circuit: unsupported template '{}'", other)
        }
    }
}
