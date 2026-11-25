//! Circuit-level generators: recursivef, recursive1/2, compressor, vadcop/final and the
//! Solidity verifiers, from `starkInfo` and vadcop metadata. A circom main is data over static
//! templates (`helper_circuits/`), and a Solidity contract is a static file with tokens (`sol/`).
//! Entry point: [`gen_circom_circuit`].

pub(crate) mod vadcop_inputs;
pub(crate) mod get_sha256_inputs;
pub(crate) mod verify_global_constraints;
pub mod templates;

use anyhow::{bail, Context, Result};
use serde_json::Value;

pub use templates::{
    gen_recursivef, gen_recursion_final, gen_solidity, gen_iverifier, gen_final_compressed, gen_compressor,
    gen_recursive1, gen_recursive2, gen_vadcop_final, SnarkVerifier,
};

/// `numProofValues` of globalInfo: an array (one count per stage) summed, or a scalar.
pub(crate) fn parse_num_proof_values(v: &Value) -> usize {
    match v.as_array() {
        Some(a) => a.iter().map(|e| e.as_u64().unwrap_or(0)).sum::<u64>() as usize,
        None => v.as_u64().unwrap_or(0) as usize,
    }
}

// ── Public types ──────────────────────────────────────────────────────────────

/// Options forwarded from `GenCircomOptions` / CLI `--has*` flags.
#[derive(Debug, Clone)]
pub struct CircomGenOptions {
    pub airgroup_id: Option<usize>,
    pub has_compressor: bool,
    pub has_recursion: bool,
    /// Proofs a `recursive2` circuit aggregates. Only that template reads it, and it rejects
    /// anything unsupported; others pass 0. Deliberately no `Default` — an unstated arity is
    /// how a wrong one would reach the circuit unnoticed.
    pub agg_arity: usize,
    /// Proofs one recursive1 verifies. 1 is the unbatched circuit.
    pub batch_size: usize,
}

/// Most proofs one recursive1 verifies. A deliberate cap, below the 26 the letter-named slots allow.
pub const MAX_RECURSIVE1_BATCH: usize = 3;

/// The circuit [`gen_circom_circuit`] renders.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CircuitTemplate {
    Recursivef,
    /// The final SNARK's circuit.
    Final,
    FinalCompressed,
    Compressor,
    Recursive1,
    Recursive2,
    VadcopFinal,
}

/// All inputs for a [`gen_circom_circuit`] call.
pub struct GenCircomCircuitInput<'a> {
    pub template: CircuitTemplate,
    /// StarkInfo JSON objects, one per AIR (one per airgroup for vadcop_final).
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

/// Generate the circom of `input.template`.
pub fn gen_circom_circuit(input: &GenCircomCircuitInput<'_>) -> Result<String> {
    // Every template but vadcop_final verifies one air.
    let stark_info = match input.stark_infos {
        [si] => si,
        _ => &Value::Null,
    };
    let publics = match input.publics {
        [] => None,
        [p] => Some(p),
        _ => bail!("gen_circom_circuit: multi-publics not yet supported"),
    };
    let airgroup_id = input.options.airgroup_id.unwrap_or(0);
    let files = input.verifier_filenames;
    match input.template {
        CircuitTemplate::Recursivef => templates::gen_recursivef(stark_info, files),
        CircuitTemplate::Final => templates::gen_recursion_final(stark_info, files, publics),
        CircuitTemplate::FinalCompressed => templates::gen_final_compressed(stark_info, files, input.options),
        CircuitTemplate::Compressor => templates::gen_compressor(stark_info, files, input.vadcop_info),
        CircuitTemplate::Recursive1 => {
            templates::gen_recursive1(stark_info, files, input.vadcop_info, airgroup_id, input.options)
        }
        CircuitTemplate::Recursive2 => {
            let bvk = input
                .basic_vk
                .get(airgroup_id)
                .with_context(|| format!("no recursive1 verification keys for airgroup {airgroup_id}"))?;
            templates::gen_recursive2(stark_info, files, input.vadcop_info, airgroup_id, bvk, input.options)
        }
        CircuitTemplate::VadcopFinal => {
            templates::gen_vadcop_final(input.stark_infos, files, input.vadcop_info, input.basic_vk, input.agg_vk)
        }
    }
}
