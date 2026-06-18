//! Rust port of `define_stark_inputs.circom.ejs` and `assign_stark_inputs.circom.ejs`.
//!
//! These two sub-templates appear inside every circuit template (recursivef,
//! recursion/final, compressor, recursive1, recursive2, vadcop/final) and declare
//! / wire all signal inputs for a StarkVerifier component.

use serde_json::Value;

// ── Math helpers ──────────────────────────────────────────────────────────────

/// Integer log2 for powers-of-two (arity is always a power of 2).
fn log2_usize(x: usize) -> usize {
    (x as f64).log2() as usize
}

/// Merkle tree siblings depth for BN128:
/// `floor((nBits - 1) / log2(arity)) + 1`
pub fn siblings_depth_bn128(n_bits: usize, arity: usize) -> usize {
    (n_bits.saturating_sub(1)) / log2_usize(arity) + 1
}

/// Merkle tree siblings depth for GL:
/// `ceil(nBits / log2(arity)) - lastLevelVerification`
pub fn siblings_depth_gl(n_bits: usize, arity: usize, last_level_verification: usize) -> usize {
    let lg = log2_usize(arity);
    let ceil = n_bits.div_ceil(lg);
    ceil.saturating_sub(last_level_verification)
}

/// `arity ** power` (integer, small exponents only).
pub fn arity_pow(arity: usize, power: usize) -> usize {
    arity.pow(power as u32)
}

// ── define_stark_inputs ───────────────────────────────────────────────────────

/// Options controlling signal generation (mirrors the JS `options` object
/// passed to the sub-template includes).
#[derive(Debug, Default, Clone)]
pub struct StarkInputOptions {
    /// Emit `signal input {prefix_}publics[nPublics]` when true.
    pub add_publics: bool,
    /// Enable `final` mode (BN128 root signal is scalar, not array[4], for final).
    pub is_final: bool,
    /// Emit `parallel` keyword on the `StarkVerifier` component (for recursive2).
    pub parallel: bool,
}

/// Generate the signal declarations for a StarkVerifier proof input block.
///
/// Ports `define_stark_inputs.circom.ejs`.
///
/// * `stark_info` – the full starkInfo JSON Value (from serialized `StarkInfoOutput`).
/// * `prefix`     – signal name prefix (empty string for the root circuit,
///   `"a_sv"` etc. for recursive2 multi-instance).
/// * `opts`       – generation options.
pub fn define_stark_inputs(stark_info: &Value, prefix: &str, opts: &StarkInputOptions) -> String {
    let mut out = String::new();
    let prefix_ = if prefix.is_empty() { String::new() } else { format!("{prefix}_") };

    let ss = &stark_info["starkStruct"];
    let hash_type = ss["verificationHashType"].as_str().unwrap_or("GL");
    let is_bn128 = hash_type == "BN128";
    let arity = ss["merkleTreeArity"].as_u64().unwrap_or(2) as usize;
    let llv = ss["lastLevelVerification"].as_u64().unwrap_or(0) as usize;
    let n_queries = ss["nQueries"].as_u64().unwrap_or(0) as usize;
    let n_constants = stark_info["nConstants"].as_u64().unwrap_or(0) as usize;
    let n_stages = stark_info["nStages"].as_u64().unwrap_or(0) as usize;
    let n_publics = stark_info["nPublics"].as_u64().unwrap_or(0) as usize;

    let steps: Vec<u64> = ss["steps"]
        .as_array()
        .map(|a| a.iter().map(|s| s["nBits"].as_u64().unwrap_or(0)).collect())
        .unwrap_or_default();

    let ev_map_len = stark_info["evMap"].as_array().map(|a| a.len()).unwrap_or(0);

    // Helpers for signal arrays (GL/BN128 root shape differs).
    let root_signal = |name: &str| -> String {
        if is_bn128 {
            format!("    signal input {prefix_}{name};\n")
        } else {
            format!("    signal input {prefix_}{name}[4];\n")
        }
    };

    // ── publics / airgroupvalues / airvalues ──────────────────────────────────
    if opts.add_publics && n_publics > 0 {
        out.push_str(&format!("    signal input {prefix_}publics[{n_publics}];\n"));
    }
    let agv_len = stark_info["airgroupValuesMap"].as_array().map(|a| a.len()).unwrap_or(0);
    if agv_len > 0 {
        out.push_str(&format!("    signal input {prefix_}airgroupvalues[{agv_len}][3];\n"));
    }
    let av_len = stark_info["airValuesMap"].as_array().map(|a| a.len()).unwrap_or(0);
    if av_len > 0 {
        out.push_str(&format!("    signal input {prefix_}airvalues[{av_len}][3];\n"));
    }

    // ── roots (one per stage + 1 extra) ──────────────────────────────────────
    for s in 1..=(n_stages + 1) {
        out.push_str(&root_signal(&format!("root{s}")));
    }

    // ── evals / s0 vals and siblings ─────────────────────────────────────────
    out.push_str(&format!("    signal input {prefix_}evals[{ev_map_len}][3];\n\n"));
    out.push_str(&format!("    signal input {prefix_}s0_valsC[{n_queries}][{n_constants}];\n"));

    // s0_siblingsC
    if is_bn128 {
        let d = siblings_depth_bn128(steps[0] as usize, arity);
        out.push_str(&format!("    signal input {prefix_}s0_siblingsC[{n_queries}][{d}][{arity}];\n"));
    } else {
        let d = siblings_depth_gl(steps[0] as usize, arity, llv);
        let leaf_w = (arity - 1) * 4;
        out.push_str(&format!("    signal input {prefix_}s0_siblingsC[{n_queries}][{d}][{leaf_w}];\n"));
        if llv > 0 {
            let last_count = arity_pow(arity, llv);
            out.push_str(&format!("    signal input {prefix_}s0_last_mt_levelsC[{last_count}][4];\n"));
        }
    }

    // custom commits
    let custom_commits = stark_info["customCommits"].as_array().cloned().unwrap_or_default();
    for cc in &custom_commits {
        let cc_name = cc["name"].as_str().unwrap_or("");
        let width = cc["stageWidths"].as_array().and_then(|a| a.first()).and_then(|v| v.as_u64()).unwrap_or(0) as usize;
        out.push_str(&format!("    signal input {prefix_}s0_vals_{cc_name}_0[{n_queries}][{width}];\n"));
        if is_bn128 {
            let d = siblings_depth_bn128(steps[0] as usize, arity);
            out.push_str(&format!("    signal input {prefix_}s0_siblings_{cc_name}_0[{n_queries}][{d}][{arity}];\n"));
        } else {
            let d = siblings_depth_gl(steps[0] as usize, arity, llv);
            let leaf_w = (arity - 1) * 4;
            out.push_str(&format!("    signal input {prefix_}s0_siblings_{cc_name}_0[{n_queries}][{d}][{leaf_w}];\n"));
            if llv > 0 {
                let last_count = arity_pow(arity, llv);
                out.push_str(&format!("    signal input {prefix_}s0_last_mt_levels_{cc_name}_0[{last_count}][4];\n"));
            }
        }
    }

    // per-stage cm vals/siblings
    let map_sections = &stark_info["mapSectionsN"];
    for s in 1..=(n_stages + 1) {
        let key = format!("cm{s}");
        let width = map_sections[&key].as_u64().unwrap_or(0) as usize;
        if width > 0 {
            out.push_str(&format!("    signal input {prefix_}s0_vals{s}[{n_queries}][{width}];\n"));
            if is_bn128 {
                let d = siblings_depth_bn128(steps[0] as usize, arity);
                out.push_str(&format!("    signal input {prefix_}s0_siblings{s}[{n_queries}][{d}][{arity}];\n"));
            } else {
                let d = siblings_depth_gl(steps[0] as usize, arity, llv);
                let leaf_w = (arity - 1) * 4;
                out.push_str(&format!("    signal input {prefix_}s0_siblings{s}[{n_queries}][{d}][{leaf_w}];\n"));
                if llv > 0 {
                    let last_count = arity_pow(arity, llv);
                    out.push_str(&format!("    signal input {prefix_}s0_last_mt_levels{s}[{last_count}][4];\n"));
                }
            }
        }
    }

    // FRI step roots and vals/siblings for steps 1..
    for s in 1..steps.len() {
        out.push_str(&root_signal(&format!("s{s}_root")));
    }
    for s in 1..steps.len() {
        let fold_vals = (1usize << (steps[s - 1] as usize - steps[s] as usize)) * 3;
        out.push_str(&format!("    signal input {prefix_}s{s}_vals[{n_queries}][{fold_vals}];\n"));
        if is_bn128 {
            let d = siblings_depth_bn128(steps[s] as usize, arity);
            out.push_str(&format!("    signal input {prefix_}s{s}_siblings[{n_queries}][{d}][{arity}];\n"));
        } else {
            let d = siblings_depth_gl(steps[s] as usize, arity, llv);
            let leaf_w = (arity - 1) * 4;
            out.push_str(&format!("    signal input {prefix_}s{s}_siblings[{n_queries}][{d}][{leaf_w}];\n"));
            if llv > 0 {
                let last_count = arity_pow(arity, llv);
                out.push_str(&format!("    signal input {prefix_}s{s}_last_mt_levels[{last_count}][4];\n"));
            }
        }
    }

    // Final polynomial
    let last_bits = steps.last().copied().unwrap_or(0) as usize;
    let final_pol_len = 1usize << last_bits;
    out.push_str(&format!("    signal input {prefix_}finalPol[{final_pol_len}][3];\n"));

    // Proof-of-work nonce (optional)
    let pow_bits = ss["powBits"].as_u64().unwrap_or(0);
    if pow_bits > 0 {
        out.push_str(&format!("    signal input {prefix_}nonce;\n"));
    }

    out
}

// ── assign_stark_inputs ───────────────────────────────────────────────────────

/// Options for the `enable` signal injection into the StarkVerifier component.
#[derive(Debug, Default, Clone)]
pub enum EnableInput {
    /// Do not emit an `enable` signal.
    #[default]
    None,
    /// Emit `{component_name}.enable <== 1;`
    One,
    /// Emit `{component_name}.enable <== {n};`
    Value(u64),
}

/// Generate the StarkVerifier component instantiation and signal wiring.
///
/// Ports `assign_stark_inputs.circom.ejs`.
///
/// * `component_name` – e.g. `"sV"`, `"vA"`, `"vB"`, `"vC"`.
/// * `prefix`         – signal name prefix.
/// * `stark_info`     – starkInfo JSON value.
/// * `opts`           – generation options (same as `define_stark_inputs`).
/// * `enable`         – optional `enable` signal injection.
pub fn assign_stark_inputs(
    component_name: &str,
    prefix: &str,
    stark_info: &Value,
    opts: &StarkInputOptions,
    enable: &EnableInput,
) -> String {
    let mut out = String::new();
    let prefix_ = if prefix.is_empty() { String::new() } else { format!("{prefix}_") };

    let ss = &stark_info["starkStruct"];
    let llv = ss["lastLevelVerification"].as_u64().unwrap_or(0) as usize;
    let n_stages = stark_info["nStages"].as_u64().unwrap_or(0) as usize;
    let n_publics = stark_info["nPublics"].as_u64().unwrap_or(0) as usize;
    let airgroup_id = stark_info["airgroupId"].as_u64();
    let steps: Vec<u64> = ss["steps"]
        .as_array()
        .map(|a| a.iter().map(|s| s["nBits"].as_u64().unwrap_or(0)).collect())
        .unwrap_or_default();
    let custom_commits = stark_info["customCommits"].as_array().cloned().unwrap_or_default();
    let map_sections = &stark_info["mapSectionsN"];

    // Component declaration
    let comp_type = match (airgroup_id, opts.is_final) {
        (Some(id), false) => {
            if opts.parallel {
                format!("parallel StarkVerifier{id}()")
            } else {
                format!("StarkVerifier{id}()")
            }
        }
        _ => {
            if opts.parallel {
                "parallel StarkVerifier()".into()
            } else {
                "StarkVerifier()".into()
            }
        }
    };
    out.push_str(&format!("    component {component_name} = {comp_type};\n"));

    // Publics
    if opts.add_publics && n_publics > 0 {
        out.push_str(&format!(
            "    for (var i=0; i< {n_publics}; i++) {{\n        \
             {component_name}.publics[i] <== {prefix_}publics[i];\n    }}\n"
        ));
    }

    // airgroupvalues / airvalues / proofvalues
    let agv_len = stark_info["airgroupValuesMap"].as_array().map(|a| a.len()).unwrap_or(0);
    if agv_len > 0 {
        out.push_str(&format!("    {component_name}.airgroupvalues <== {prefix_}airgroupvalues;\n"));
    }
    let av_len = stark_info["airValuesMap"].as_array().map(|a| a.len()).unwrap_or(0);
    if av_len > 0 {
        out.push_str(&format!("    {component_name}.airvalues <== {prefix_}airvalues;\n"));
    }
    let pv_len = stark_info["proofValuesMap"].as_array().map(|a| a.len()).unwrap_or(0);
    if pv_len > 0 {
        out.push_str(&format!("    {component_name}.proofvalues <== proofValues;\n"));
    }

    // roots
    for s in 1..=(n_stages + 1) {
        out.push_str(&format!("    {component_name}.root{s} <== {prefix_}root{s};\n"));
    }

    out.push_str(&format!("    {component_name}.evals <== {prefix_}evals;\n"));

    // s0_valsC / s0_siblingsC
    out.push_str(&format!("    {component_name}.s0_valsC <== {prefix_}s0_valsC;\n"));
    out.push_str(&format!("    {component_name}.s0_siblingsC <== {prefix_}s0_siblingsC;\n"));
    if llv > 0 {
        out.push_str(&format!("    {component_name}.s0_last_mt_levelsC <== {prefix_}s0_last_mt_levelsC;\n"));
    }

    // custom commits
    for cc in &custom_commits {
        let cc_name = cc["name"].as_str().unwrap_or("");
        out.push_str(&format!("    {component_name}.s0_vals_{cc_name}_0 <== {prefix_}s0_vals_{cc_name}_0;\n"));
        out.push_str(&format!("    {component_name}.s0_siblings_{cc_name}_0 <== {prefix_}s0_siblings_{cc_name}_0;\n"));
        if llv > 0 {
            out.push_str(&format!(
                "    {component_name}.s0_last_mt_levels_{cc_name}_0 <== {prefix_}s0_last_mt_levels_{cc_name}_0;\n"
            ));
        }
    }

    // per-stage cm
    for s in 1..=(n_stages + 1) {
        let key = format!("cm{s}");
        let width = map_sections[&key].as_u64().unwrap_or(0);
        if width > 0 {
            out.push_str(&format!("    {component_name}.s0_vals{s} <== {prefix_}s0_vals{s};\n"));
            out.push_str(&format!("    {component_name}.s0_siblings{s} <== {prefix_}s0_siblings{s};\n"));
            if llv > 0 {
                out.push_str(&format!(
                    "    {component_name}.s0_last_mt_levels{s} <== {prefix_}s0_last_mt_levels{s};\n"
                ));
            }
        }
    }

    // FRI step roots
    for s in 1..steps.len() {
        out.push_str(&format!("    {component_name}.s{s}_root <== {prefix_}s{s}_root;\n"));
    }
    // FRI step vals/siblings
    for s in 1..steps.len() {
        out.push_str(&format!("    {component_name}.s{s}_vals <== {prefix_}s{s}_vals;\n"));
        out.push_str(&format!("    {component_name}.s{s}_siblings <== {prefix_}s{s}_siblings;\n"));
        if llv > 0 {
            out.push_str(&format!("    {component_name}.s{s}_last_mt_levels <== {prefix_}s{s}_last_mt_levels;\n"));
        }
    }

    // finalPol
    out.push_str(&format!("    {component_name}.finalPol <== {prefix_}finalPol;\n"));

    // Proof-of-work nonce
    let pow_bits = ss["powBits"].as_u64().unwrap_or(0);
    if pow_bits > 0 {
        out.push_str(&format!("    {component_name}.nonce <== {prefix_}nonce;\n"));
    }

    // Optional enable signal
    match enable {
        EnableInput::None => {}
        EnableInput::One => out.push_str(&format!("    {component_name}.enable <== 1;\n")),
        EnableInput::Value(n) => out.push_str(&format!("    {component_name}.enable <== {n};\n")),
    }

    out
}
