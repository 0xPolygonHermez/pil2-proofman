//! Circom / Solidity template generators.
//!
//! All non-trivial templates use **Tera** (embedded at compile time via
//! `include_str!`).  Sibling modules build the Tera context — including, when
//! relevant, pre-rendered code fragments emitted by the `Transcript` state
//! machine — and delegate the final Circom shape to a `.tera` file.
//!
//! Tera templates:
//! - `recursivef.circom.tera`        → [`gen_recursivef`]
//! - `recursion_final.circom.tera`   → [`gen_recursion_final`]
//! - `verifier.sol.tera`             → [`gen_solidity`]
//! - `iverifier.sol.tera`            → [`gen_iverifier`]
//! - `calculate_hashes.circom.tera`  → [`super::calculate_hashes::gen_calculate_hashes`]
//! - `get_sha256_inputs.circom.tera` → [`super::get_sha256_inputs::gen_get_sha256_inputs`]

use anyhow::{bail, Context, Result};
use serde_json::Value;
use tera::{Context as TeraCtx, Tera};

use super::get_sha256_inputs::gen_get_sha256_inputs;
use super::stark_inputs::{assign_stark_inputs, define_stark_inputs, EnableInput, StarkInputOptions};
use super::vadcop_inputs::{
    agg_types_consistency, agg_vadcop_inputs, assign_vadcop_inputs, basic_circuit_type, define_vadcop_inputs,
    init_vadcop_inputs, AggVadcopOptions, AssignVadcopOptions, VadcopSignals,
};
use super::calculate_hashes::gen_calculate_hashes;
use super::verify_global_challenge::gen_verify_global_challenge;
use super::verify_global_constraints::gen_verify_global_constraints;
use super::CircomGenOptions;
use crate::stark2circom::circom_verifier::bn128::uses_custom_templates;

// ── helpers ───────────────────────────────────────────────────────────────────

/// Parse `numProofValues` from a vadcop_info JSON value.
///
/// In the serialised globalInfo, `numProofValues` is emitted as a JSON array
/// (one entry per stage, e.g. `[2]`).  JavaScript coerces `[2] > 0` → `true`
/// and `[2].toString() === "2"`, so the JS templates work transparently.
/// Rust must do the coercion explicitly: sum the array elements, or fall back
/// to treating the value as a plain scalar.
fn parse_num_proof_values(v: &Value) -> usize {
    if let Some(arr) = v.as_array() {
        arr.iter().map(|e| e.as_u64().unwrap_or(0)).sum::<u64>() as usize
    } else {
        v.as_u64().unwrap_or(0) as usize
    }
}

// ── Embedded Tera templates ───────────────────────────────────────────────────

const RECURSIVEF_TMPL: &str = include_str!("tera/recursivef.circom.tera");
const FINAL_TMPL: &str = include_str!("tera/recursion_final.circom.tera");
const VERIFIER_SOL_TMPL: &str = include_str!("tera/verifier.sol.tera");
const IVERIFIER_SOL_TMPL: &str = include_str!("tera/iverifier.sol.tera");
const FINAL_COMPRESSED_TMPL: &str = include_str!("tera/final_compressed.circom.tera");
const COMPRESSOR_TMPL: &str = include_str!("tera/compressor.circom.tera");
const RECURSIVE1_TMPL: &str = include_str!("tera/recursive1.circom.tera");
const RECURSIVE1_BATCHED_TMPL: &str = include_str!("tera/recursive1_batched.circom.tera");
const RECURSIVE2_TMPL: &str = include_str!("tera/recursive2.circom.tera");
const VADCOP_FINAL_TMPL: &str = include_str!("tera/vadcop_final.circom.tera");

/// Render a single one-shot Tera template string (no inheritance / include).
pub(super) fn render(template_src: &str, ctx: &TeraCtx) -> Result<String> {
    let mut tera = Tera::default();
    tera.add_raw_template("t", template_src).context("Failed to parse Tera template")?;
    tera.render("t", ctx).context("Failed to render Tera template")
}

// ── recursivef ────────────────────────────────────────────────────────────────

/// Port of `src/recursion/templates/recursivef.circom.ejs`.
pub fn gen_recursivef(stark_info: &Value, verifier_filenames: &[String], _opts: &CircomGenOptions) -> Result<String> {
    let def_opts = StarkInputOptions { add_publics: true, is_final: false, parallel: false, shared_publics: false };
    let stark_signals = define_stark_inputs(stark_info, "", &def_opts);
    let stark_assign = assign_stark_inputs("sV", "", stark_info, &def_opts, &EnableInput::None);
    let n_publics = stark_info["nPublics"].as_u64().unwrap_or(0);

    let mut ctx = TeraCtx::new();
    ctx.insert("verifier_filenames", verifier_filenames);
    ctx.insert("stark_signals", &stark_signals);
    ctx.insert("stark_assign", &stark_assign);
    ctx.insert("has_publics", &(n_publics > 0));

    render(RECURSIVEF_TMPL, &ctx)
}

// ── recursion/final ───────────────────────────────────────────────────────────

/// Port of `src/recursion/templates/final.circom.ejs`.
pub fn gen_recursion_final(
    stark_info: &Value,
    verifier_filenames: &[String],
    publics: Option<&Value>,
    _opts: &CircomGenOptions,
) -> Result<String> {
    let def_opts = StarkInputOptions { add_publics: true, is_final: true, parallel: false, shared_publics: false };
    let stark_signals = define_stark_inputs(stark_info, "", &def_opts);
    let stark_assign = assign_stark_inputs("sV", "", stark_info, &def_opts, &EnableInput::None);

    // The circuit includes the recursivef's verifier: when that one uses custom templates, circom
    // wants the pragma here too, and the publics hash takes the verifier's LessThanGoldilocks.
    let custom_templates = uses_custom_templates(stark_info);
    let sha256_template = publics.map(|p| gen_get_sha256_inputs(p, custom_templates)).unwrap_or_default();
    let n_publics = stark_info["nPublics"].as_u64().unwrap_or(0) as usize;
    // Incoming `publics` = [rootCVadcopFinal(4) | is_vadcop_final_proof(1) | real publics].
    // rootC takes the first 4, the flag the next 1, and the remaining
    // `n_publics - 5` are the real publics that get hashed. The flag is verified
    // (it stays a public input) but is excluded from `publicsProof` / the hash.
    let n_publics_proof = n_publics.saturating_sub(5);

    let mut ctx = TeraCtx::new();
    ctx.insert("custom_templates", &custom_templates);
    ctx.insert("verifier_filenames", verifier_filenames);
    ctx.insert("sha256_template", &sha256_template);
    ctx.insert("stark_signals", &stark_signals);
    ctx.insert("stark_assign", &stark_assign);
    ctx.insert("n_publics_proof", &n_publics_proof);

    render(FINAL_TMPL, &ctx)
}

// ── Solidity contracts ────────────────────────────────────────────────────────

/// The verifier of the final SNARK that the project's Solidity verifier ([`gen_solidity`]) extends:
/// the contract that checks the final circuit's proof, where it is, and the proof its `verifyProof`
/// takes, with the publics hash as its one public.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SnarkVerifier {
    /// snarkjs's `FflonkVerifier`, in `FflonkVerifier.sol` beside the project's: a proof of 24
    /// `bytes32`.
    Fflonk,
    /// snarkjs's `PlonkVerifier`, in `PlonkVerifier.sol` beside the project's: a proof of 24
    /// `uint256`.
    Plonk,
    /// pilfflonk's `PilfflonkVerifier`, the verifier of the final circuit's vkey
    /// (`pilfflonk.verifier.sol`): a proof of `words` `bytes32`, the vkey's calldata
    /// (pilfflonk/docs/formats.md#calldata), whose number depends on the vkey.
    Pilfflonk {
        /// The path the project's verifier imports `pilfflonk.verifier.sol` from.
        source: String,
        /// The words of the `proof` argument of its `verifyProof`.
        words: u64,
    },
}

impl SnarkVerifier {
    /// The contract's name, without `Verifier`.
    fn name(&self) -> &'static str {
        match self {
            SnarkVerifier::Fflonk => "Fflonk",
            SnarkVerifier::Plonk => "Plonk",
            SnarkVerifier::Pilfflonk { .. } => "Pilfflonk",
        }
    }

    /// How the interface names the inherited verifier.
    fn label(&self) -> &'static str {
        match self {
            SnarkVerifier::Fflonk => "FFLONK",
            SnarkVerifier::Plonk => "PLONK",
            SnarkVerifier::Pilfflonk { .. } => "pilfflonk",
        }
    }

    /// The path the project's verifier imports the contract from.
    fn source(&self) -> String {
        match self {
            SnarkVerifier::Fflonk | SnarkVerifier::Plonk => format!("./{}Verifier.sol", self.name()),
            SnarkVerifier::Pilfflonk { source, .. } => source.clone(),
        }
    }

    /// The statement that decodes `proofBytes` into `proofDecoded`, the `proof` of `verifyProof`.
    fn proof_decode(&self) -> String {
        let (word, words) = match self {
            SnarkVerifier::Fflonk => ("bytes32", 24),
            SnarkVerifier::Plonk => ("uint256", 24),
            SnarkVerifier::Pilfflonk { words, .. } => ("bytes32", *words),
        };
        format!("{word}[{words}] memory proofDecoded = abi.decode(proofBytes, ({word}[{words}]));")
    }
}

/// Port of `src/recursion/contracts/verifier.sol.ejs`: the project's verifier, which extends
/// `verifier`. `version` is what the contract's `VERSION()` returns.
pub fn gen_solidity(
    name: &str,
    version: &str,
    root_c: &[u64; 4],
    publics: Option<&Value>,
    verifier: &SnarkVerifier,
) -> String {
    let camel = capitalise(name);

    let (has_program_vk, first_is_vk) = publics
        .and_then(|v| v["definitions"].as_array())
        .map(|defs| {
            let first = defs.first().and_then(|d| d["verificationKey"].as_bool()).unwrap_or(false);
            let last = defs.last().and_then(|d| d["verificationKey"].as_bool()).unwrap_or(false);
            (first || last, first)
        })
        .unwrap_or((false, false));

    let mut ctx = TeraCtx::new();
    ctx.insert("name", &camel);
    ctx.insert("snark", verifier.name());
    ctx.insert("snark_source", &verifier.source());
    ctx.insert("version", version);
    ctx.insert("root_c_0", &root_c[0]);
    ctx.insert("root_c_1", &root_c[1]);
    ctx.insert("root_c_2", &root_c[2]);
    ctx.insert("root_c_3", &root_c[3]);
    ctx.insert("has_program_vk", &has_program_vk);
    ctx.insert("first_is_vk", &first_is_vk);
    ctx.insert("proof_decode", &verifier.proof_decode());

    render(VERIFIER_SOL_TMPL, &ctx).unwrap_or_else(|e| panic!("verifier.sol template error: {e:#}"))
}

/// Port of `src/recursion/contracts/iverifier.sol.ejs`.
pub fn gen_iverifier(name: &str, publics: Option<&Value>, verifier: &SnarkVerifier) -> String {
    let camel = capitalise(name);

    let has_program_vk = publics
        .and_then(|v| v["definitions"].as_array())
        .map(|defs| {
            let first = defs.first().and_then(|d| d["verificationKey"].as_bool()).unwrap_or(false);
            let last = defs.last().and_then(|d| d["verificationKey"].as_bool()).unwrap_or(false);
            first || last
        })
        .unwrap_or(false);

    let mut ctx = TeraCtx::new();
    ctx.insert("name", &camel);
    ctx.insert("snark_label", verifier.label());
    ctx.insert("has_program_vk", &has_program_vk);

    render(IVERIFIER_SOL_TMPL, &ctx).unwrap_or_else(|e| panic!("iverifier.sol template error: {e:#}"))
}

// ── helpers ───────────────────────────────────────────────────────────────────

fn capitalise(s: &str) -> String {
    let mut chars = s.chars();
    match chars.next() {
        None => String::new(),
        Some(c) => c.to_uppercase().to_string() + chars.as_str(),
    }
}

// ── VADCOP templates ──────────────────────────────────────────────────────────

/// Port of `vadcop/templates/final_compressed.circom.ejs`.
pub fn gen_final_compressed(
    stark_info: &Value,
    verifier_filenames: &[String],
    opts: &CircomGenOptions,
) -> Result<String> {
    // The vadcop_final AIR carries the `is_vadcop_final_proof` flag as public @0,
    // ahead of the real publics. So the StarkVerifier consumes `n_publics` values,
    // but only the `n_publics - 1` after the flag are re-exposed by this circuit —
    // the flag is verified but stripped from the public interface. The template
    // hand-declares the publics (flag first, then `publics[n_real]`) and rebuilds
    // the full array for the verifier, so we emit stark_signals/stark_assign WITHOUT
    // their auto `publics` declaration/wiring.
    let n_publics = stark_info["nPublics"].as_u64().unwrap_or(0) as usize;
    let n_publics_real = n_publics.saturating_sub(1);
    let has_publics = if opts.has_recursion { n_publics > 4 } else { n_publics > 0 };

    let def_opts = StarkInputOptions { add_publics: false, is_final: false, parallel: false, shared_publics: false };

    let mut ctx = TeraCtx::new();
    ctx.insert("verifier_filenames", verifier_filenames);
    ctx.insert("stark_signals", &define_stark_inputs(stark_info, "", &def_opts));
    ctx.insert("stark_assign", &assign_stark_inputs("sV", "", stark_info, &def_opts, &EnableInput::None));
    ctx.insert("has_publics", &has_publics);
    ctx.insert("n_publics", &n_publics);
    ctx.insert("n_publics_real", &n_publics_real);

    render(FINAL_COMPRESSED_TMPL, &ctx)
}

/// Port of `vadcop/templates/compressor.circom.ejs`.
pub fn gen_compressor(
    stark_info: &Value,
    verifier_filenames: &[String],
    vadcop_info: &Value,
    _opts: &CircomGenOptions,
) -> Result<String> {
    let airgroup_id = stark_info["airgroupId"].as_u64().unwrap_or(0) as usize;
    let n_publics = vadcop_info["nPublics"].as_u64().unwrap_or(0) as usize;
    let num_proof_values = parse_num_proof_values(&vadcop_info["numProofValues"]);

    let def_opts = StarkInputOptions { add_publics: false, is_final: false, parallel: false, shared_publics: false };

    let mut pub_names: Vec<&str> = Vec::new();
    if n_publics > 0 {
        pub_names.push("publics");
    }
    if num_proof_values > 0 {
        pub_names.push("proofValues");
    }
    pub_names.push("globalChallenge");

    let mut ctx = TeraCtx::new();
    ctx.insert("verifier_filenames", verifier_filenames);
    ctx.insert("calculate_hashes", &gen_calculate_hashes(stark_info, vadcop_info));
    ctx.insert("has_publics", &(n_publics > 0));
    ctx.insert("n_publics", &n_publics);
    ctx.insert("has_proof_values", &(num_proof_values > 0));
    ctx.insert("num_proof_values", &num_proof_values);
    ctx.insert("stark_signals", &define_stark_inputs(stark_info, "", &def_opts));
    ctx.insert("vadcop_define", &define_vadcop_inputs(vadcop_info, airgroup_id, "sv", VadcopSignals::Output, None));
    ctx.insert("stark_assign", &assign_stark_inputs("sV", "", stark_info, &def_opts, &EnableInput::None));
    ctx.insert("vadcop_init", &init_vadcop_inputs("sV", "sv", "", airgroup_id, stark_info, vadcop_info, None));
    ctx.insert("pub_names", &pub_names.join(", "));

    render(COMPRESSOR_TMPL, &ctx)
}

/// Port of `vadcop/templates/recursive1.circom.ejs`; `batch_size > 1` renders k proofs of one air.
pub fn gen_recursive1(
    stark_info: &Value,
    verifier_filenames: &[String],
    vadcop_info: &Value,
    airgroup_id: usize,
    opts: &CircomGenOptions,
) -> Result<String> {
    let has_compressor = opts.has_compressor;
    let k = opts.batch_size.max(1);
    if k > super::MAX_RECURSIVE1_BATCH {
        bail!("gen_recursive1: batch_size {k} exceeds the cap of {}", super::MAX_RECURSIVE1_BATCH);
    }
    let n_publics = vadcop_info["nPublics"].as_u64().unwrap_or(0) as usize;
    let num_proof_values = parse_num_proof_values(&vadcop_info["numProofValues"]);

    let def_opts = StarkInputOptions { add_publics: false, is_final: false, parallel: false, shared_publics: false };
    // The k proofs are of one air and one run, so their verifiers read the template's publics.
    let assign_opts =
        StarkInputOptions { add_publics: !has_compressor, is_final: false, parallel: k > 1, shared_publics: true };

    let mut ctx = TeraCtx::new();
    ctx.insert("verifier_filenames", verifier_filenames);
    ctx.insert("has_compressor", &has_compressor);
    ctx.insert(
        "calculate_hashes",
        &if !has_compressor { gen_calculate_hashes(stark_info, vadcop_info) } else { String::new() },
    );
    ctx.insert("has_publics", &(n_publics > 0));
    ctx.insert("n_publics", &n_publics);
    ctx.insert("has_proof_values", &(num_proof_values > 0));
    ctx.insert("num_proof_values", &num_proof_values);

    // Only a single compressor slot re-publishes the set it was handed; anything folded is output.
    let mut publics_names: Vec<String> = Vec::new();
    let template = if k == 1 {
        let outer = if has_compressor { VadcopSignals::Input } else { VadcopSignals::Output };
        ctx.insert(
            "vadcop_define",
            &define_vadcop_inputs(vadcop_info, airgroup_id, "sv", outer, Some(&mut publics_names)),
        );
        ctx.insert("stark_signals", &define_stark_inputs(stark_info, "", &def_opts));
        ctx.insert("stark_assign", &assign_stark_inputs("sV", "", stark_info, &assign_opts, &EnableInput::None));
        ctx.insert(
            "vadcop_init_or_assign",
            &if !has_compressor {
                init_vadcop_inputs("sV", "sv", "", airgroup_id, stark_info, vadcop_info, None)
            } else {
                let av_opts = AssignVadcopOptions { add_prefix_agg_types: true, ..Default::default() };
                assign_vadcop_inputs("sV", vadcop_info, airgroup_id, "sv", "", &av_opts)
            },
        );
        RECURSIVE1_TMPL
    } else {
        ctx.insert(
            "vadcop_define_sv",
            &define_vadcop_inputs(vadcop_info, airgroup_id, "sv", VadcopSignals::Output, None),
        );
        let letters: Vec<String> = (0..k).map(|i| ((b'a' + i as u8) as char).to_string()).collect();
        let prefixes: Vec<String> = letters.iter().map(|l| format!("{l}_sv")).collect();
        // Compressor slots receive their values; basic slots mint them.
        let kind = if has_compressor { VadcopSignals::Input } else { VadcopSignals::Internal };
        let av_opts = AssignVadcopOptions { add_prefix_agg_types: true, set_enable_input: true, parallel: true };
        let slots: Vec<serde_json::Value> = letters
            .iter()
            .zip(prefixes.iter())
            .map(|(l, pfx)| {
                let comp = format!("v{}", l.to_uppercase());
                let vadcop_define = define_vadcop_inputs(vadcop_info, airgroup_id, pfx, kind, None);
                let enable =
                    if has_compressor { EnableInput::None } else { EnableInput::Expr(format!("1 - {l}_isNull")) };
                let vadcop_assign = if has_compressor {
                    assign_vadcop_inputs(&comp, vadcop_info, airgroup_id, pfx, l, &av_opts)
                } else {
                    init_vadcop_inputs(
                        &comp,
                        pfx,
                        l,
                        airgroup_id,
                        stark_info,
                        vadcop_info,
                        Some(&format!("{l}_isNull")),
                    )
                };
                serde_json::json!({
                    "lower": l,
                    "upper": l.to_uppercase(),
                    "prefix": pfx,
                    "vadcop_define": vadcop_define,
                    "stark_signals": define_stark_inputs(stark_info, l, &def_opts),
                    "stark_assign": assign_stark_inputs(&comp, l, stark_info, &assign_opts, &enable),
                    "vadcop_assign": vadcop_assign,
                })
            })
            .collect();
        ctx.insert("slots", &slots);
        ctx.insert(
            "agg_types_consistency",
            &agg_types_consistency(vadcop_info, airgroup_id, &prefixes, has_compressor),
        );
        let agg_opts = AggVadcopOptions { own_circuit_type: true, force_null: true };
        ctx.insert("agg_vadcop", &agg_vadcop_inputs(vadcop_info, airgroup_id, &prefixes, "sv", &agg_opts));
        ctx.insert("circuit_type", &basic_circuit_type(stark_info, vadcop_info));
        RECURSIVE1_BATCHED_TMPL
    };

    // Only a single compressor slot's sv_* are public inputs; batched slots stay private (recursive2's nPublics).
    let mut pub_names: Vec<String> = Vec::new();
    if has_compressor && k == 1 {
        pub_names.extend(publics_names);
    }
    if n_publics > 0 {
        pub_names.push("publics".into());
    }
    if num_proof_values > 0 {
        pub_names.push("proofValues".into());
    }
    pub_names.push("globalChallenge".into());
    pub_names.push("rootCAgg".into());
    ctx.insert("pub_names", &pub_names.join(", "));

    render(template, &ctx)
}

/// Port of `vadcop/templates/recursive2.circom.ejs`.
pub fn gen_recursive2(
    stark_info: &Value,
    verifier_filenames: &[String],
    vadcop_info: &Value,
    airgroup_id: usize,
    basic_vk: &[Vec<String>],
    opts: &CircomGenOptions,
) -> Result<String> {
    // `pub` in a library crate, so an external caller reaches this ahead of the CLI's
    // `--agg-arity` check. Reject anything outside the supported set before it becomes a
    // panicking slice index (`slot_prefixes[1..]` at 0) or nonsense slot names (past 'z').
    let agg_arity = opts.agg_arity;
    if !proofman_common::global_info::is_valid_aggregation_arity(agg_arity) {
        bail!(
            "gen_recursive2: unsupported aggregation arity {agg_arity}; valid values: {:?}",
            proofman_common::global_info::VALID_AGGREGATION_ARITIES
        );
    }

    let n_publics_raw = stark_info["nPublics"].as_u64().unwrap_or(0) as usize;
    let n_publics_vad = vadcop_info["nPublics"].as_u64().unwrap_or(0) as usize;
    let num_proof_values = parse_num_proof_values(&vadcop_info["numProofValues"]);
    let air_groups_len = vadcop_info["air_groups"].as_array().map(|a| a.len()).unwrap_or(0);
    let airs_0_len =
        vadcop_info["airs"].as_array().and_then(|a| a.first()).and_then(|v| v.as_array()).map(|a| a.len()).unwrap_or(0);
    let multi_air = air_groups_len > 1 || airs_0_len > 1;
    let airs_in_group = vadcop_info["airs"]
        .as_array()
        .and_then(|a| a.get(airgroup_id))
        .and_then(|v| v.as_array())
        .map(|a| a.len())
        .unwrap_or(0);

    let def_opts = StarkInputOptions { add_publics: false, is_final: false, parallel: false, shared_publics: false };
    let par_opts = StarkInputOptions { add_publics: false, is_final: false, parallel: true, shared_publics: false };
    let av_opts = AssignVadcopOptions { add_prefix_agg_types: true, set_enable_input: multi_air, parallel: true };

    // rootCBasics inline array assignments
    let rootc_basics: String = basic_vk
        .iter()
        .enumerate()
        .map(|(i, vk)| format!("    rootCBasics[{i}] = [{}];", vk.join(",")))
        .collect::<Vec<_>>()
        .join("\n");

    // Slot names come from the index, so N=3 still emits a/b/c and the generated
    // circom is unchanged.
    let slot_names: Vec<String> = (0..agg_arity).map(|i| ((b'a' + i as u8) as char).to_string()).collect();
    let slot_prefixes: Vec<String> = slot_names.iter().map(|s| format!("{s}_sv")).collect();

    let sel_fn = if multi_air { "SelectVerificationKeyNull" } else { "SelectVerificationKey" };

    let mut pub_names: Vec<&str> = Vec::new();
    if n_publics_vad > 0 {
        pub_names.push("publics");
    }
    if num_proof_values > 0 {
        pub_names.push("proofValues");
    }
    pub_names.push("globalChallenge");
    pub_names.push("rootCAgg");

    let mut ctx = TeraCtx::new();
    ctx.insert("verifier_filenames", verifier_filenames);
    ctx.insert("airs_in_group", &airs_in_group);
    ctx.insert("rootc_basics", &rootc_basics);
    ctx.insert("vadcop_define_sv", &define_vadcop_inputs(vadcop_info, airgroup_id, "sv", VadcopSignals::Output, None));
    ctx.insert("has_publics", &(n_publics_vad > 0));
    ctx.insert("n_publics", &n_publics_vad);
    ctx.insert("has_proof_values", &(num_proof_values > 0));
    ctx.insert("num_proof_values", &num_proof_values);
    // serde_json values, not tera::Context: a Context is not Serialize and cannot be
    // nested inside another Context.
    let slots: Vec<serde_json::Value> = slot_names
        .iter()
        .zip(slot_prefixes.iter())
        .map(|(name, prefix)| {
            let upper = name.to_uppercase();
            let comp = format!("v{upper}");
            serde_json::json!({
                "lower": name,
                "upper": upper,
                "prefix": prefix,
                "vadcop_define": define_vadcop_inputs(vadcop_info, airgroup_id, prefix, VadcopSignals::Input, None),
                "stark_signals": define_stark_inputs(stark_info, name, &def_opts),
                "stark_assign": assign_stark_inputs(&comp, name, stark_info, &par_opts, &EnableInput::None),
                "vadcop_assign": assign_vadcop_inputs(&comp, vadcop_info, airgroup_id, prefix, name, &av_opts),
            })
        })
        .collect();
    ctx.insert("slots", &slots);
    ctx.insert("agg_types_consistency", &agg_types_consistency(vadcop_info, airgroup_id, &slot_prefixes, false));
    ctx.insert("sel_fn", sel_fn);
    ctx.insert("agg_vadcop", &agg_vadcop_inputs(vadcop_info, airgroup_id, &slot_prefixes, "sv", &Default::default()));
    ctx.insert("n_publics_minus_4", &(n_publics_raw - 4));
    ctx.insert("pub_names", &pub_names.join(", "));

    render(RECURSIVE2_TMPL, &ctx)
}

/// Port of `vadcop/templates/final.circom.ejs`.
pub fn gen_vadcop_final(
    stark_infos: &[Value],
    verifier_filenames: &[String],
    vadcop_info: &Value,
    basic_vk: &[Vec<Vec<String>>],
    agg_vk: &[Vec<String>],
    _opts: &CircomGenOptions,
) -> Result<String> {
    let stark_info_0 = stark_infos.first().unwrap_or(&Value::Null);
    let agg_types = vadcop_info["aggTypes"].as_array().cloned().unwrap_or_default();
    let n_publics = vadcop_info["nPublics"].as_u64().unwrap_or(0) as usize;
    let num_proof_values = parse_num_proof_values(&vadcop_info["numProofValues"]);
    let proof_values_map = vadcop_info["proofValuesMap"].as_array().cloned().unwrap_or_default();
    let multi_air_groups = vadcop_info["air_groups"].as_array().map(|a| a.len()).unwrap_or(0) > 1;
    let air_groups_len = vadcop_info["air_groups"].as_array().map(|a| a.len()).unwrap_or(0);
    let airs_0_len =
        vadcop_info["airs"].as_array().and_then(|a| a.first()).and_then(|v| v.as_array()).map(|a| a.len()).unwrap_or(0);
    let multi_air = air_groups_len > 1 || airs_0_len > 1;

    let def_opts = StarkInputOptions { add_publics: false, is_final: false, parallel: false, shared_publics: false };
    let av_opts = AssignVadcopOptions { add_prefix_agg_types: true, set_enable_input: multi_air, parallel: false };

    // Pre-render per-airgroup define and assign sections
    let mut define_sections: Vec<String> = Vec::new();
    let mut assign_sections: Vec<String> = Vec::new();

    for (i, _) in agg_types.iter().enumerate() {
        let si = if multi_air_groups { stark_infos.get(i).unwrap_or(stark_info_0) } else { stark_info_0 };

        let mut section = String::new();
        section.push_str(&define_vadcop_inputs(vadcop_info, i, &format!("s{i}_sv"), VadcopSignals::Input, None));
        section.push_str(&define_stark_inputs(si, &format!("s{i}"), &def_opts));
        define_sections.push(section);

        let n_pub_raw = si["nPublics"].as_u64().unwrap_or(0) as usize;
        let agg_vk_i = agg_vk.get(i).cloned().unwrap_or_default();
        let basic_vk_i = basic_vk.get(i).cloned().unwrap_or_default();
        let airs_i_len = vadcop_info["airs"]
            .as_array()
            .and_then(|a| a.get(i))
            .and_then(|v| v.as_array())
            .map(|a| a.len())
            .unwrap_or(0);
        let sel_fn = if multi_air { "SelectVerificationKeyNull" } else { "SelectVerificationKey" };

        let vk_lines: String = basic_vk_i
            .iter()
            .enumerate()
            .map(|(j, bvk)| format!("    s{i}_sv_rootCBasics[{j}] = [{}];", bvk.join(",")))
            .collect::<Vec<_>>()
            .join("\n");

        let mut section = String::new();
        section.push_str(&assign_stark_inputs(&format!("sV{i}"), &format!("s{i}"), si, &def_opts, &EnableInput::None));
        section.push_str(&assign_vadcop_inputs(
            &format!("sV{i}"),
            vadcop_info,
            i,
            &format!("s{i}_sv"),
            &format!("s{i}"),
            &av_opts,
        ));
        section.push_str(&format!("    var s{i}_sv_rootCAgg[4] = [{}];\n", agg_vk_i.join(",")));
        section.push_str(&format!("    var s{i}_sv_rootCBasics[{airs_i_len}][4];\n\n"));
        section.push_str(&vk_lines);
        section.push('\n');
        section.push_str(&format!(
            "    sV{i}.rootC <== {sel_fn}({airs_i_len})(s{i}_sv_circuitType, s{i}_sv_rootCBasics, s{i}_sv_rootCAgg);\n"
        ));
        section.push_str(&format!(
            "    for (var i = 0; i < 4; i++) {{\n        sV{i}.publics[{} + i] <== s{i}_sv_rootCAgg[i];\n    }}\n",
            n_pub_raw - 4
        ));
        assign_sections.push(section);
    }

    // A null proof leaves its airgroup's values unconstrained, so the global checks must
    // see what an airgroup with no instances contributes, never the prover's values.
    // A single airgroup can never be null (the proof count below), so it needs no mask.
    let mask_null = multi_air && agg_types.len() > 1;
    let curve = vadcop_info["curve"].as_str().unwrap_or("None");
    let lattice_size = vadcop_info["latticeSize"].as_u64().unwrap_or(0) as usize;
    let src = |i: usize, what: &str| if mask_null { format!("s{i}_masked_{what}") } else { format!("s{i}_sv_{what}") };
    let mut null_sections: Vec<String> = Vec::new();
    if mask_null {
        for (i, ag) in agg_types.iter().enumerate() {
            let mut section = String::new();
            if curve == "None" {
                section.push_str(&format!(
                    "    signal s{i}_masked_stage1Hash[{lattice_size}];\n    for (var k = 0; k < {lattice_size}; k++) {{\n        s{i}_masked_stage1Hash[k] <== (1 - s{i}_sv_isNull) * s{i}_sv_stage1Hash[k];\n    }}\n"
                ));
            } else {
                section.push_str(&format!(
                    "    signal s{i}_masked_stage1Hash[2][5];\n    for (var k = 0; k < 2; k++) {{\n        for (var l = 0; l < 5; l++) {{\n            s{i}_masked_stage1Hash[k][l] <== (1 - s{i}_sv_isNull) * s{i}_sv_stage1Hash[k][l];\n        }}\n    }}\n"
                ));
            }
            let types = ag.as_array().cloned().unwrap_or_default();
            if !types.is_empty() {
                section.push_str(&format!("    signal s{i}_masked_airgroupvalues[{}][3];\n", types.len()));
                for (j, t) in types.iter().enumerate() {
                    // The identity of the aggregation: 0 for a sum, 1 for a product.
                    let v = format!("s{i}_sv_airgroupvalues[{j}]");
                    let first = if t["aggType"].as_u64() == Some(1) {
                        format!("{v}[0] + s{i}_sv_isNull * (1 - {v}[0])")
                    } else {
                        format!("(1 - s{i}_sv_isNull) * {v}[0]")
                    };
                    section.push_str(&format!(
                        "    s{i}_masked_airgroupvalues[{j}] <== [{first}, (1 - s{i}_sv_isNull) * {v}[1], (1 - s{i}_sv_isNull) * {v}[2]];\n"
                    ));
                }
            }
            null_sections.push(section);
        }
    }

    // (verifyGlobalConstraints input, source) for each airgroup that has airgroupvalues
    let airgroupvalues_wires: Vec<serde_json::Value> = agg_types
        .iter()
        .enumerate()
        .filter(|(_, ag)| ag.as_array().map(|a| a.len()).unwrap_or(0) > 0)
        .map(|(i, _)| serde_json::json!({ "dst": format!("s{i}_airgroupvalues"), "src": src(i, "airgroupvalues") }))
        .collect();

    let stage1hash_indices: Vec<usize> = (0..agg_types.len()).collect();
    let stage1hash_srcs: Vec<String> = stage1hash_indices.iter().map(|&i| src(i, "stage1Hash")).collect();

    let mut ctx = TeraCtx::new();
    ctx.insert("verifier_filenames", verifier_filenames);
    ctx.insert("verify_global_challenge", &gen_verify_global_challenge(stark_info_0, vadcop_info));
    ctx.insert("verify_global_constraints", &gen_verify_global_constraints(vadcop_info));
    ctx.insert("has_publics", &(n_publics > 0));
    ctx.insert("n_publics", &n_publics);
    ctx.insert("has_proof_values", &(num_proof_values > 0 || !proof_values_map.is_empty()));
    ctx.insert("num_proof_values", &num_proof_values);
    ctx.insert("define_sections", &define_sections);
    ctx.insert("assign_sections", &assign_sections);
    ctx.insert("stage1hash_indices", &stage1hash_indices);
    ctx.insert("stage1hash_srcs", &stage1hash_srcs);
    ctx.insert("airgroupvalues_wires", &airgroupvalues_wires);
    ctx.insert("multi_air", &multi_air);
    ctx.insert("null_sections", &null_sections);
    ctx.insert("n_airgroups", &agg_types.len());

    render(VADCOP_FINAL_TMPL, &ctx)
}

#[cfg(test)]
mod tests {
    use std::fs;
    use std::path::Path;
    use std::process::Command;

    use super::*;
    use crate::stark2circom::circom_verifier::{gen_stark_verifier_bn128, Pil2CircomOptions};
    use crate::plonk2pil::r1cs_types::read_r1cs_from_bytes;
    use proofman_fields::Bn128;
    use serde_json::json;

    /// A recursivef starkinfo, whose trees are custom or not, with what the final circuit and the
    /// head of its verifier read.
    fn recursivef_stark_info(merkle_tree_custom: bool) -> Value {
        json!({
            "starkStruct": {
                "verificationHashType": "BN128",
                "numQueries": 4,
                "nBits": 6,
                "nBitsExt": 10,
                "grindingBitsQueries": 0,
                "merkleTreeArity": 4,
                "merkleTreeCustom": merkle_tree_custom,
                "lastLevelVerification": 0,
                "hashCommits": false,
                "foldingFactors": [4],
                "logDegrees": [6, 2],
                "logDomainSizes": [10, 6]
            },
            "nStages": 2,
            "nPublics": 6,
            "nConstants": 3,
            "evMap": [],
            "cmPolsMap": [],
            "customCommits": [],
            "customCommitsMap": [],
            "challengesMap": [],
            "boundaries": [{"name": "everyRow"}],
            "airgroupValuesMap": [],
            "airValuesMap": [],
            "proofValuesMap": [],
            "mapSectionsN": {"cm1": 2, "cm2": 2, "cm3": 2},
            "openingPoints": [],
            "qDeg": 1
        })
    }

    /// The final circuit around a recursivef whose trees are custom or not.
    fn final_circuit(merkle_tree_custom: bool) -> String {
        let opts = CircomGenOptions {
            airgroup_id: None,
            has_compressor: false,
            has_recursion: false,
            is_final: true,
            agg_arity: 0,
            batch_size: 1,
        };
        let verifiers = ["recursivef.verifier.circom".to_string()];
        gen_recursion_final(&recursivef_stark_info(merkle_tree_custom), &verifiers, Some(&Value::Null), &opts).unwrap()
    }

    #[test]
    fn final_declares_custom_templates_for_custom_trees() {
        let out = final_circuit(true);
        assert!(
            out.starts_with(
                "pragma circom 2.1.0;\npragma custom_templates;\n\n    include \"recursivef.verifier.circom\";\n"
            ),
            "out:\n{out}"
        );
    }

    #[test]
    fn final_without_custom_trees_is_unchanged() {
        let out = final_circuit(false);
        assert!(
            out.starts_with("pragma circom 2.1.0;\n\n    include \"recursivef.verifier.circom\";\n"),
            "out:\n{out}"
        );
        assert!(!out.contains("custom_templates"), "out:\n{out}");
    }

    /// The project's verifier of a program whose publics end with its verification key, extending
    /// `verifier`.
    fn project_verifier(verifier: &SnarkVerifier) -> String {
        let publics = json!({"definitions": [{"name": "out"}, {"name": "rom_root", "verificationKey": true}]});
        gen_solidity("build", "v0", &[1, 2, 3, u64::MAX], Some(&publics), verifier)
    }

    /// The lines of `verifier`'s project verifier that are not FFLONK's.
    fn lines_unlike_fflonk(verifier: &SnarkVerifier) -> Vec<String> {
        let fflonk = project_verifier(&SnarkVerifier::Fflonk);
        let out = project_verifier(verifier);
        assert_eq!(out.lines().count(), fflonk.lines().count(), "out:\n{out}");
        out.lines()
            .zip(fflonk.lines())
            .filter(|(line, other)| line != other)
            .map(|(line, _)| line.trim().to_string())
            .collect()
    }

    /// snarkjs's verifiers are beside the project's, and take a proof of 24 words.
    #[test]
    fn the_snarkjs_verifiers_are_imported_from_beside_the_contract() {
        let fflonk = project_verifier(&SnarkVerifier::Fflonk);
        for line in [
            "import {FflonkVerifier} from \"./FflonkVerifier.sol\";",
            "contract BuildVerifier is FflonkVerifier, IBuildVerifier {",
            "bytes32[24] memory proofDecoded = abi.decode(proofBytes, (bytes32[24]));",
            "bool success = this.verifyProof(proofDecoded, [publicValuesDigest]);",
        ] {
            assert!(fflonk.contains(line), "{line} missing from:\n{fflonk}");
        }
        assert_eq!(
            lines_unlike_fflonk(&SnarkVerifier::Plonk),
            [
                "import {PlonkVerifier} from \"./PlonkVerifier.sol\";",
                "contract BuildVerifier is PlonkVerifier, IBuildVerifier {",
                "uint256[24] memory proofDecoded = abi.decode(proofBytes, (uint256[24]));",
            ]
        );
    }

    /// pilfflonk's verifier is imported from where the key has it, and takes a proof of the words of
    /// the vkey's calldata; the rest of the contract is FFLONK's.
    #[test]
    fn the_pilfflonk_verifier_is_imported_from_its_source_with_a_proof_of_its_words() {
        let source = "./provingKey/final/pilfflonk/pilfflonk.verifier.sol".to_string();
        assert_eq!(
            lines_unlike_fflonk(&SnarkVerifier::Pilfflonk { source, words: 47 }),
            [
                "import {PilfflonkVerifier} from \"./provingKey/final/pilfflonk/pilfflonk.verifier.sol\";",
                "contract BuildVerifier is PilfflonkVerifier, IBuildVerifier {",
                "bytes32[47] memory proofDecoded = abi.decode(proofBytes, (bytes32[47]));",
            ]
        );
    }

    /// The include graph of the final circuit around a recursivef with custom trees, which circom
    /// checks before anything else: every file on the way to a custom template declares
    /// `pragma custom_templates;` (CG04 otherwise), and no template is defined twice (T2008). The
    /// final circuit's head (its pragmas, the verifier's include and the publics hash, with its own
    /// includes) and the verifier's (its pragmas and includes), as the templates render them, are
    /// compiled over the libraries of the snark setup, with a main that instantiates
    /// `CustomPoseidon`, `LessThanGoldilocks` and two `RangeCheck`s, whose custom gates the r1cs
    /// has. The whole circuit, from a real recursivef, is compiled by setup-snark's
    /// `pilfflonk_final_circuit` test.
    ///
    /// Needs circomlib, from `npm install` in `setup/pil2-stark`; without it the test says so and
    /// passes.
    #[test]
    fn custom_final_includes_compile() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
        let circomlib = root.join("setup/pil2-stark/node_modules/circomlib/circuits");
        if !circomlib.is_dir() {
            eprintln!("skipped: {} not present (npm install in setup/pil2-stark)", circomlib.display());
            return;
        }
        let circom = root.join("setup/circom").join(if cfg!(target_os = "macos") { "circom_mac" } else { "circom" });
        let libraries = Path::new(env!("CARGO_MANIFEST_DIR")).join("stark2circom/circom_verifier");

        let stark_info = recursivef_stark_info(true);
        let verifier_info = json!({ "qVerifier": { "code": [] }, "queryVerifier": { "code": [] } });
        let opts = Pil2CircomOptions { skip_main: true, ..Pil2CircomOptions::default() };
        let root_c = ["1".to_string(), "0".to_string(), "0".to_string(), "0".to_string()];
        let verifier = gen_stark_verifier_bn128(Some(&root_c), &stark_info, &verifier_info, &opts).unwrap();
        let verifier_head = &verifier[..verifier.find("\ntemplate ").expect("a template in the verifier")];
        let circuit = final_circuit(true);
        let circuit_head = &circuit[..circuit.find("template Main()").expect("the final circuit's Main")];

        let dir = std::env::temp_dir().join(format!("custom_final_includes_{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        fs::write(dir.join("recursivef.verifier.circom"), verifier_head).unwrap();
        const MAIN: &str = r#"template Main() {
    signal input in[4];
    signal output hash[5] <== CustomPoseidon(4)(in, 0);
    signal output gl <== LessThanGoldilocks()(in[0]);
    RangeCheck(65)(in[1]);
    RangeCheck(154)(in[2]);
}

component main = Main();
"#;
        fs::write(dir.join("final.circom"), format!("{circuit_head}{MAIN}")).unwrap();

        // The libraries of `snark_setup.rs`'s final compile, in its order.
        let out = Command::new(&circom)
            .arg(dir.join("final.circom"))
            .args(["--O1", "--r1cs", "-o"])
            .arg(&dir)
            .arg("-l")
            .arg(libraries.join("helper_circuits"))
            .arg("-l")
            .arg(libraries.join("circuits.bn128"))
            .arg("-l")
            .arg(&circomlib)
            .output()
            .unwrap_or_else(|e| panic!("run {}: {e}", circom.display()));
        let stderr = String::from_utf8_lossy(&out.stderr).into_owned();
        assert!(out.status.success(), "circom failed:\n{stderr}{}", String::from_utf8_lossy(&out.stdout));

        // The custom templates are compiled as such, the custom ones of LessThanGoldilocks among
        // them: a Num2Bytes(64) on its in and one on in + 2^64 - p. RangeCheck(154) takes two, on
        // its low 80 bits and on the rest.
        let r1cs = read_r1cs_from_bytes::<Bn128>(&fs::read(dir.join("final.r1cs")).unwrap()).unwrap();
        let gate = |id: u32| {
            let gate = &r1cs.custom_gates[id as usize];
            let parameters: Vec<String> = gate.parameters.iter().map(Bn128::to_string).collect();
            format!("{}({})", gate.template_name, parameters.join(", "))
        };
        let mut uses: Vec<String> = r1cs.custom_gates_uses.iter().map(|gate_use| gate(gate_use.id)).collect();
        uses.sort();
        assert_eq!(
            uses,
            ["Num2Bytes(64)", "Num2Bytes(64)", "Num2Bytes(65)", "Num2Bytes(74)", "Num2Bytes(80)", "PoseidonT(5)"]
        );
        let _ = fs::remove_dir_all(&dir);
    }
}

#[cfg(test)]
mod recursive1_batch_tests {
    use super::*;

    /// Two airs in the group, so type 0 stays reserved for a null proof.
    fn vadcop() -> Value {
        serde_json::json!({
            "aggTypes": [[{"aggType": 0, "stage": 2}]], "air_groups": ["g"],
            "airs": [[{"name": "a"}, {"name": "b"}]], "curve": "None", "latticeSize": 368,
            "nPublics": 2, "numProofValues": [0], "hash": "blake3"
        })
    }

    fn stark_info() -> Value {
        serde_json::json!({
            "airId": 1, "airgroupId": 0, "nStages": 2, "nPublics": 12, "nConstants": 2,
            "mapSectionsN": { "cm1": 3, "cm2": 6, "cm3": 3 },
            "customCommits": [], "evMap": [{}], "airValuesMap": [], "challengesMap": [],
            "starkStruct": {
                "nBits": 10, "nBitsExt": 11, "merkleTreeArity": 2, "transcriptArity": 2,
                "merkleTreeCustom": false, "verificationHashType": "GL",
                "lastLevelVerification": 4, "hashCommits": true,
                "foldingFactors": [6], "logDegrees": [10, 4], "logDomainSizes": [11, 5],
                "numQueries": 8, "grindingBitsQueries": 24
            }
        })
    }

    fn gen(k: usize, has_compressor: bool) -> String {
        let opts = CircomGenOptions {
            airgroup_id: Some(0),
            has_compressor,
            has_recursion: false,
            is_final: false,
            agg_arity: 2,
            batch_size: k,
        };
        gen_recursive1(&stark_info(), &["v.circom".into()], &vadcop(), 0, &opts).unwrap()
    }

    /// k=1 renders the circuit that shipped before batching: one `sV`, no prefixes, no fold.
    #[test]
    fn one_slot_is_the_unbatched_circuit() {
        let out = gen(1, false);
        assert!(out.contains("component sV = StarkVerifier"), "{out}");
        assert!(!out.contains("a_sv_"), "no slot prefixes at k=1");
        assert!(!out.contains("agg_values.circom"), "no aggregation at k=1");
        assert!(out.contains("sv_aggregatedProofs <== 1;"));
    }

    #[test]
    fn each_slot_gets_its_own_verifier_switched_by_its_own_null_flag() {
        let out = gen(3, false);
        for (l, comp) in [("a", "vA"), ("b", "vB"), ("c", "vC")] {
            // `parallel`, like recursive2's slots: the k verifiers are independent.
            assert!(out.contains(&format!("component {comp} = parallel StarkVerifier")), "slot {l} verifier");
            assert!(out.contains(&format!("signal input {l}_isNull;")), "slot {l} flag");
            assert!(out.contains(&format!("{l}_isNull * ({l}_isNull - 1) === 0;")), "slot {l} binary");
            assert!(out.contains(&format!("{comp}.enable <== 1 - {l}_isNull;")), "slot {l} enable");
        }
        assert!(!out.contains("vD"), "only k verifiers");
    }

    /// Slots mint their types, so the consistency check must follow them.
    #[test]
    fn the_consistency_check_comes_after_the_slots_mint_their_types() {
        let out = gen(2, false);
        let mint = out.find("a_sv_aggregationTypes <== ").expect("the slot mints its types");
        let read = out.find("aggregationTypes[i] <== a_sv_aggregationTypes[i]").expect("the check reads them");
        assert!(mint < read, "the check at {read} reads what is only assigned at {mint}");
    }

    /// Every verifier reads the template's own `publics`; prefixing named a signal nothing declares.
    #[test]
    fn every_slot_reads_the_templates_own_publics() {
        let out = gen(3, false);
        assert!(out.contains("vA.publics[i] <== publics[i];"), "{out}");
        assert!(out.contains("vC.publics[i] <== publics[i];"));
        for l in ["a", "b", "c"] {
            assert!(!out.contains(&format!("{l}_publics[")), "slot {l} must not have publics of its own");
        }
    }

    #[test]
    fn the_fold_keeps_the_airs_circuit_type() {
        let out = gen(2, false);
        assert!(out.contains("sv_circuitType <== 3;"), "airId 1 + 2 in a multi-air group\n{out}");
        assert!(!out.contains("sv_circuitType <== 1;"), "must not claim the aggregated type");
    }

    /// A short batch has to prove, so the fold is null-aware and the count comes from the flags.
    #[test]
    fn a_short_batch_folds_through_the_null_templates() {
        let out = gen(2, false);
        assert!(out.contains("AggregateProofsNull(2)"), "{out}");
        assert!(out.contains("AggregateAirgroupValuesNull()"));
        assert!(out.contains("agg_values.circom"), "the null templates have to be included");
    }

    /// A valid recursive1 carrying nothing, and vadcop_final discards the count that would expose it.
    #[test]
    fn an_all_empty_batch_is_unprovable() {
        for has_compressor in [false, true] {
            let out = gen(2, has_compressor);
            assert!(out.contains("noProofs <== IsZero()(sv_aggregatedProofs);"), "{out}");
            assert!(out.contains("noProofs === 0;"));
        }
    }

    /// The values arrive with the proof, so the flag comes from the type rather than an input.
    #[test]
    fn the_compressor_path_takes_its_null_flag_from_the_proof() {
        let out = gen(2, true);
        assert!(out.contains("signal input a_sv_circuitType;"), "{out}");
        assert!(out.contains("a_sv_isNull <== IsZero()(a_sv_circuitType);"));
        assert!(out.contains("vA.enable <== 1 - a_sv_isNull;"));
        assert!(!out.contains("signal input a_isNull;"), "no separate flag when the type carries it");
        // Private, or `plonk2pil`'s nPublics runs past the starkinfo r1 borrows from r2.
        let main = out.rsplit("component main").next().unwrap();
        assert!(!main.contains("a_sv_"), "no slot value belongs in the public list: {main}");
    }

    #[test]
    fn past_the_cap_it_refuses() {
        let opts = CircomGenOptions {
            airgroup_id: Some(0),
            has_compressor: false,
            has_recursion: false,
            is_final: false,
            agg_arity: 2,
            batch_size: super::super::MAX_RECURSIVE1_BATCH + 1,
        };
        assert!(gen_recursive1(&stark_info(), &["v.circom".into()], &vadcop(), 0, &opts).is_err());
    }
}

#[cfg(test)]
mod solidity_version_tests {
    use super::*;

    fn zisk_publics() -> Value {
        serde_json::json!({
            "nPublics": 68,
            "definitions": [
                {"name": "rom_root", "initialPos": 0, "nValues": 4, "chunks": [1, 64], "verificationKey": true},
                {"name": "inputs", "initialPos": 4, "nValues": 64, "chunks": [1, 64]}
            ],
            "hasProgramVK": true
        })
    }

    #[test]
    fn the_verifier_returns_the_given_version() {
        let sol = gen_solidity("zisk", "v9.9.9-test", &[1, 2, 3, 4], Some(&zisk_publics()), &SnarkVerifier::Plonk);
        assert!(sol.contains(r#"return "v9.9.9-test";"#), "{sol}");
        assert!(sol.contains("contract ZiskVerifier is PlonkVerifier, IZiskVerifier"));
        assert!(sol.contains("must be a constant of the calling contract"));
    }

    /// The interface carries the key-binding rules a caller needs to verify safely.
    #[test]
    fn the_interface_documents_how_to_pin_the_keys() {
        let isol = gen_iverifier("zisk", Some(&zisk_publics()), &SnarkVerifier::Plonk);
        assert!(isol.contains("Both keys MUST be constants of the calling contract"), "{isol}");
        assert!(isol.contains("@param programVK"));
        assert!(isol.contains("@param rootCVadcopFinal"));
        assert!(isol.contains("the inherited raw PLONK verifier"));

        let fflonk = gen_iverifier("zisk", Some(&zisk_publics()), &SnarkVerifier::Fflonk);
        assert!(fflonk.contains("the inherited raw FFLONK verifier"));

        // Without a program key there is one key to pin, and no programVK parameter.
        let no_vk = gen_iverifier("zisk", None, &SnarkVerifier::Plonk);
        assert!(no_vk.contains("The root MUST be a constant of the calling contract"), "{no_vk}");
        assert!(!no_vk.contains("programVK"));
    }
}
