//! High-level wrappers for circom generation tools.
//!
//! `pil2circom` generates a per-air stark verifier circom using the in-process
//! Rust implementation (`gen_stark_verifier`).
//! `gen_circom` generates recursive/final circuit circom using the in-process
//! Rust generator (`gen_circom_circuit`).

use anyhow::{Context, Result};
use serde_json::Value;

use pil2_stark_recurser::stark2circom::{
    gen_circom_circuit, gen_stark_verifier, CircomGenOptions, GenCircomCircuitInput, StarkVerifierOptions,
};

// ── pil2circom ───────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct Pil2CircomOptions {
    pub skip_main: bool,
    pub verkey_input: bool,
    pub enable_input: bool,
    pub input_challenges: bool,
    pub hash: String,
}

impl Default for Pil2CircomOptions {
    fn default() -> Self {
        Self {
            skip_main: false,
            verkey_input: false,
            enable_input: false,
            input_challenges: false,
            hash: proofman_common::hash_family::DEFAULT_HASH_ID.to_string(),
        }
    }
}

/// Generate a stark verifier circom using the in-process Rust implementation.
pub fn pil2circom(
    const_root: &[String; 4],
    stark_info: &Value,
    verifier_info: &Value,
    opts: &Pil2CircomOptions,
) -> Result<String> {
    let rust_opts = StarkVerifierOptions {
        skip_main: opts.skip_main,
        verkey_input: opts.verkey_input,
        enable_input: opts.enable_input,
        input_challenges: opts.input_challenges,
        fri_queries_batch_size: None,
        multi_fri: false,
        hash: opts.hash.clone(),
    };
    let root: Option<&[String; 4]> = if opts.verkey_input { None } else { Some(const_root) };
    gen_stark_verifier(root, stark_info, verifier_info, &rust_opts).context("gen_stark_verifier failed")
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

/// Generate a circom circuit using the in-process Rust generator.
///
/// Returns an error if the template is not implemented.
pub fn gen_circom(input: &GenCircomInput<'_>) -> Result<String> {
    let rust_opts = CircomGenOptions {
        airgroup_id: input.options.airgroup_id.map(|x| x as usize),
        has_compressor: input.options.has_compressor,
        has_recursion: input.options.has_recursion,
        is_final: input.options.is_final,
    };
    let rust_input = GenCircomCircuitInput {
        template_name: input.template_name,
        stark_infos: input.stark_infos,
        vadcop_info: input.vadcop_info,
        verifier_filenames: input.verifier_filenames,
        basic_vk: input.basic_verification_keys,
        agg_vk: input.agg_verification_keys,
        publics: input.publics,
        options: &rust_opts,
    };
    gen_circom_circuit(&rust_input)
}
