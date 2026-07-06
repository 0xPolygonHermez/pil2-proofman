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

use anyhow::{Context, Result};
use serde_json::Value;
use tera::{Context as TeraCtx, Tera};

use super::get_sha256_inputs::gen_get_sha256_inputs;
use super::stark_inputs::{assign_stark_inputs, define_stark_inputs, EnableInput, StarkInputOptions};
use super::vadcop_inputs::{
    agg_vadcop_inputs, assign_vadcop_inputs, define_vadcop_inputs, init_vadcop_inputs, AssignVadcopOptions,
};
use super::calculate_hashes::gen_calculate_hashes;
use super::verify_global_challenge::gen_verify_global_challenge;
use super::verify_global_constraints::gen_verify_global_constraints;
use super::CircomGenOptions;

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
    let def_opts = StarkInputOptions { add_publics: true, is_final: false, parallel: false };
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
    let def_opts = StarkInputOptions { add_publics: true, is_final: true, parallel: false };
    let stark_signals = define_stark_inputs(stark_info, "", &def_opts);
    let stark_assign = assign_stark_inputs("sV", "", stark_info, &def_opts, &EnableInput::None);

    let sha256_template = publics.map(gen_get_sha256_inputs).unwrap_or_default();
    let n_publics = stark_info["nPublics"].as_u64().unwrap_or(0) as usize;
    // Incoming `publics` = [rootCVadcopFinal(4) | is_vadcop_final_proof(1) | real publics].
    // rootC takes the first 4, the flag the next 1, and the remaining
    // `n_publics - 5` are the real publics that get hashed. The flag is verified
    // (it stays a public input) but is excluded from `publicsProof` / the hash.
    let n_publics_proof = n_publics.saturating_sub(5);

    let mut ctx = TeraCtx::new();
    ctx.insert("verifier_filenames", verifier_filenames);
    ctx.insert("sha256_template", &sha256_template);
    ctx.insert("stark_signals", &stark_signals);
    ctx.insert("stark_assign", &stark_assign);
    ctx.insert("n_publics_proof", &n_publics_proof);

    render(FINAL_TMPL, &ctx)
}

// ── Solidity contracts ────────────────────────────────────────────────────────

/// Port of `src/recursion/contracts/verifier.sol.ejs`.
pub fn gen_solidity(name: &str, root_c: &[u64; 4], publics: Option<&Value>, use_fflonk: bool) -> String {
    let camel = capitalise(name);
    let snark = if use_fflonk { "Fflonk" } else { "Plonk" };

    let (has_program_vk, first_is_vk) = publics
        .and_then(|v| v["definitions"].as_array())
        .map(|defs| {
            let first = defs.first().and_then(|d| d["verificationKey"].as_bool()).unwrap_or(false);
            let last = defs.last().and_then(|d| d["verificationKey"].as_bool()).unwrap_or(false);
            (first || last, first)
        })
        .unwrap_or((false, false));

    let proof_decode = if use_fflonk {
        "bytes32[24] memory proofDecoded = abi.decode(proofBytes, (bytes32[24]));"
    } else {
        "uint256[24] memory proofDecoded = abi.decode(proofBytes, (uint256[24]));"
    };

    let mut ctx = TeraCtx::new();
    ctx.insert("name", &camel);
    ctx.insert("snark", snark);
    ctx.insert("root_c_0", &root_c[0]);
    ctx.insert("root_c_1", &root_c[1]);
    ctx.insert("root_c_2", &root_c[2]);
    ctx.insert("root_c_3", &root_c[3]);
    ctx.insert("has_program_vk", &has_program_vk);
    ctx.insert("first_is_vk", &first_is_vk);
    ctx.insert("proof_decode", proof_decode);

    render(VERIFIER_SOL_TMPL, &ctx).unwrap_or_else(|e| panic!("verifier.sol template error: {e:#}"))
}

/// Port of `src/recursion/contracts/iverifier.sol.ejs`.
pub fn gen_iverifier(name: &str, publics: Option<&Value>) -> String {
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

    let def_opts = StarkInputOptions { add_publics: false, is_final: false, parallel: false };

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

    let def_opts = StarkInputOptions { add_publics: false, is_final: false, parallel: false };

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
    ctx.insert("vadcop_define", &define_vadcop_inputs(vadcop_info, airgroup_id, "sv", false, None));
    ctx.insert("stark_assign", &assign_stark_inputs("sV", "", stark_info, &def_opts, &EnableInput::None));
    ctx.insert("vadcop_init", &init_vadcop_inputs("sV", "sv", "", airgroup_id, stark_info, vadcop_info));
    ctx.insert("pub_names", &pub_names.join(", "));

    render(COMPRESSOR_TMPL, &ctx)
}

/// Port of `vadcop/templates/recursive1.circom.ejs`.
pub fn gen_recursive1(
    stark_info: &Value,
    verifier_filenames: &[String],
    vadcop_info: &Value,
    airgroup_id: usize,
    opts: &CircomGenOptions,
) -> Result<String> {
    let has_compressor = opts.has_compressor;
    let n_publics = vadcop_info["nPublics"].as_u64().unwrap_or(0) as usize;
    let num_proof_values = parse_num_proof_values(&vadcop_info["numProofValues"]);

    let def_opts = StarkInputOptions { add_publics: false, is_final: false, parallel: false };
    let assign_opts = StarkInputOptions { add_publics: !has_compressor, is_final: false, parallel: false };

    let mut publics_names: Vec<String> = Vec::new();
    let vadcop_define = define_vadcop_inputs(
        vadcop_info,
        airgroup_id,
        "sv",
        has_compressor,
        Some(&mut publics_names), // always collect; used in pub_names only when has_compressor
    );

    let vadcop_init_or_assign = if !has_compressor {
        init_vadcop_inputs("sV", "sv", "", airgroup_id, stark_info, vadcop_info)
    } else {
        let av_opts = AssignVadcopOptions { add_prefix_agg_types: true, ..Default::default() };
        assign_vadcop_inputs("sV", vadcop_info, airgroup_id, "sv", "", &av_opts)
    };

    // sv_* signals are inputs only when has_compressor (the compressor provides them).
    // When !has_compressor they are outputs — circom forbids outputs in the public list.
    let mut pub_names: Vec<String> = Vec::new();
    if has_compressor {
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

    let mut ctx = TeraCtx::new();
    ctx.insert("verifier_filenames", verifier_filenames);
    ctx.insert("has_compressor", &has_compressor);
    ctx.insert(
        "calculate_hashes",
        &if !has_compressor { gen_calculate_hashes(stark_info, vadcop_info) } else { String::new() },
    );
    ctx.insert("vadcop_define", &vadcop_define);
    ctx.insert("stark_signals", &define_stark_inputs(stark_info, "", &def_opts));
    ctx.insert("has_publics", &(n_publics > 0));
    ctx.insert("n_publics", &n_publics);
    ctx.insert("has_proof_values", &(num_proof_values > 0));
    ctx.insert("num_proof_values", &num_proof_values);
    ctx.insert("stark_assign", &assign_stark_inputs("sV", "", stark_info, &assign_opts, &EnableInput::None));
    ctx.insert("vadcop_init_or_assign", &vadcop_init_or_assign);
    ctx.insert("pub_names", &pub_names.join(", "));

    render(RECURSIVE1_TMPL, &ctx)
}

/// Port of `vadcop/templates/recursive2.circom.ejs`.
pub fn gen_recursive2(
    stark_info: &Value,
    verifier_filenames: &[String],
    vadcop_info: &Value,
    airgroup_id: usize,
    basic_vk: &[Vec<String>],
    _opts: &CircomGenOptions,
) -> Result<String> {
    let n_publics_raw = stark_info["nPublics"].as_u64().unwrap_or(0) as usize;
    let n_publics_vad = vadcop_info["nPublics"].as_u64().unwrap_or(0) as usize;
    let num_proof_values = parse_num_proof_values(&vadcop_info["numProofValues"]);
    let agg_types_len = vadcop_info["aggTypes"]
        .as_array()
        .and_then(|a| a.get(airgroup_id))
        .and_then(|v| v.as_array())
        .map(|a| a.len())
        .unwrap_or(0);
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

    let def_opts = StarkInputOptions { add_publics: false, is_final: false, parallel: false };
    let par_opts = StarkInputOptions { add_publics: false, is_final: false, parallel: true };
    let av_opts = AssignVadcopOptions { add_prefix_agg_types: true, set_enable_input: multi_air, parallel: true };

    // rootCBasics inline array assignments
    let rootc_basics: String = basic_vk
        .iter()
        .enumerate()
        .map(|(i, vk)| format!("    rootCBasics[{i}] = [{}];", vk.join(",")))
        .collect::<Vec<_>>()
        .join("\n");

    // aggregationTypes consistency block
    let agg_types_consistency = if agg_types_len > 0 {
        format!(
            "    signal aggregationTypes[{n}];\n    for(var i = 0; i < {n}; i++) {{\n        aggregationTypes[i] <== a_sv_aggregationTypes[i];\n        a_sv_aggregationTypes[i] === b_sv_aggregationTypes[i];\n        a_sv_aggregationTypes[i] === c_sv_aggregationTypes[i];\n    }}",
            n = agg_types_len
        )
    } else {
        String::new()
    };

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
    ctx.insert("vadcop_define_sv", &define_vadcop_inputs(vadcop_info, airgroup_id, "sv", false, None));
    ctx.insert("has_publics", &(n_publics_vad > 0));
    ctx.insert("n_publics", &n_publics_vad);
    ctx.insert("has_proof_values", &(num_proof_values > 0));
    ctx.insert("num_proof_values", &num_proof_values);
    ctx.insert("vadcop_define_a", &define_vadcop_inputs(vadcop_info, airgroup_id, "a_sv", true, None));
    ctx.insert("stark_signals_a", &define_stark_inputs(stark_info, "a", &def_opts));
    ctx.insert("vadcop_define_b", &define_vadcop_inputs(vadcop_info, airgroup_id, "b_sv", true, None));
    ctx.insert("stark_signals_b", &define_stark_inputs(stark_info, "b", &def_opts));
    ctx.insert("vadcop_define_c", &define_vadcop_inputs(vadcop_info, airgroup_id, "c_sv", true, None));
    ctx.insert("stark_signals_c", &define_stark_inputs(stark_info, "c", &def_opts));
    ctx.insert("agg_types_consistency", &agg_types_consistency);
    ctx.insert("stark_assign_a", &assign_stark_inputs("vA", "a", stark_info, &par_opts, &EnableInput::None));
    ctx.insert("stark_assign_b", &assign_stark_inputs("vB", "b", stark_info, &par_opts, &EnableInput::None));
    ctx.insert("stark_assign_c", &assign_stark_inputs("vC", "c", stark_info, &par_opts, &EnableInput::None));
    ctx.insert("vadcop_assign_a", &assign_vadcop_inputs("vA", vadcop_info, airgroup_id, "a_sv", "a", &av_opts));
    ctx.insert("vadcop_assign_b", &assign_vadcop_inputs("vB", vadcop_info, airgroup_id, "b_sv", "b", &av_opts));
    ctx.insert("vadcop_assign_c", &assign_vadcop_inputs("vC", vadcop_info, airgroup_id, "c_sv", "c", &av_opts));
    ctx.insert("sel_fn", sel_fn);
    ctx.insert("agg_vadcop", &agg_vadcop_inputs(vadcop_info, airgroup_id, "a_sv", "b_sv", "c_sv", "sv"));
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

    let def_opts = StarkInputOptions { add_publics: false, is_final: false, parallel: false };
    let av_opts = AssignVadcopOptions { add_prefix_agg_types: true, set_enable_input: multi_air, parallel: false };

    // Pre-render per-airgroup define and assign sections
    let mut define_sections: Vec<String> = Vec::new();
    let mut assign_sections: Vec<String> = Vec::new();

    for (i, _) in agg_types.iter().enumerate() {
        let si = if multi_air_groups { stark_infos.get(i).unwrap_or(stark_info_0) } else { stark_info_0 };

        let mut section = String::new();
        section.push_str(&define_vadcop_inputs(vadcop_info, i, &format!("s{i}_sv"), true, None));
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
            "    for (var i=0; i<4; i++) {{\n        sV{i}.publics[{} + i] <== s{i}_sv_rootCAgg[i];\n    }}\n",
            n_pub_raw - 4
        ));
        assign_sections.push(section);
    }

    // airgroupvalues wiring names for verifyGlobalConstraints
    let airgroupvalues_names: Vec<String> = agg_types
        .iter()
        .enumerate()
        .filter(|(_, ag)| ag.as_array().map(|a| a.len()).unwrap_or(0) > 0)
        .map(|(i, _)| format!("s{i}_sv_airgroupvalues"))
        .collect();

    let stage1hash_indices: Vec<usize> = (0..agg_types.len()).collect();

    let mut ctx = TeraCtx::new();
    ctx.insert("verifier_filenames", verifier_filenames);
    ctx.insert("verify_global_challenge", &gen_verify_global_challenge(stark_info_0, vadcop_info));
    ctx.insert("verify_global_constraints", &gen_verify_global_constraints(stark_info_0, vadcop_info));
    ctx.insert("has_publics", &(n_publics > 0));
    ctx.insert("n_publics", &n_publics);
    ctx.insert("has_proof_values", &(num_proof_values > 0 || !proof_values_map.is_empty()));
    ctx.insert("num_proof_values", &num_proof_values);
    ctx.insert("define_sections", &define_sections);
    ctx.insert("assign_sections", &assign_sections);
    ctx.insert("stage1hash_indices", &stage1hash_indices);
    ctx.insert("airgroupvalues_names", &airgroupvalues_names);
    ctx.insert("n_airgroups", &agg_types.len());

    render(VADCOP_FINAL_TMPL, &ctx)
}
