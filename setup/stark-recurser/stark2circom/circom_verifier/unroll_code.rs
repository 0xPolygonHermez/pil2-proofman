//! Rust port of the EJS `unrollCode()` and `ref()` functions from
//! `circuits.gl/stark_verifier.circom.ejs`.
//!
//! `unroll_code` translates a slice of FPM "instructions" (from
//! `verifierInfo.qVerifier.code` or `verifierInfo.queryVerifier.code`) into
//! Circom signal-assignment lines.  It appends those lines to a `Vec<String>`
//! and returns the signal name of the last destination (matching the JS
//! `return ref(code[code.length-1].dest)` at the end of `unrollCode`).
//!
//! Each instruction JSON object has the shape:
//! ```json
//! { "op": "add"|"sub"|"mul"|"copy",
//!   "dest": { "type": "tmp", "id": N, "dim": 1|3, ... },
//!   "src":  [<ref>, <ref>]  // src[1] may be missing for "copy"
//! }
//! ```
//!
//! A `ref` object is one of:
//! | type            | fields used                        |
//! |-----------------|-------------------------------------|
//! | `"eval"`        | `id`                               |
//! | `"challenge"`   | `stage`, `stageId`                 |
//! | `"public"`      | `id`                               |
//! | `"x"`           | —                                  |
//! | `"Zi"`          | `boundaryId`, `dim` (forced 3)     |
//! | `"xDivXSubXi"`  | `id`                               |
//! | `"tmp"`         | `id`, `dim`                        |
//! | `"cm"`          | `id`  (→ cmPolsMap lookup)         |
//! | `"custom"`      | `commitId`, `id` (→ customCommitsMap) |
//! | `"const"`       | `id`                               |
//! | `"number"`      | `value`                            |
//! | `"airgroupvalue"` | `id`, `dim` (forced 3)           |
//! | `"airvalue"`    | `id`, `dim`                        |
//! | `"proofvalue"`  | `id`, `dim`                        |

use anyhow::{bail, Result};
use serde_json::Value;

// ── Context needed by ref() ───────────────────────────────────────────────────

/// Immutable context derived from `starkInfo` that `ref()` needs.
pub struct UnrollCtx<'a> {
    /// `starkInfo.nStages + 1` — the quotient-polynomial stage.
    pub q_stage: u64,
    /// `starkInfo.nStages + 2` — the evals/xi stage.
    pub evals_stage: u64,
    /// `starkInfo.nStages + 3` — the FRI stage.
    pub fri_stage: u64,
    /// `starkInfo.cmPolsMap` array.
    pub cm_pols_map: &'a [Value],
    /// `starkInfo.customCommits` array (for name lookup).
    pub custom_commits: &'a [Value],
    /// `starkInfo.customCommitsMap` — array of arrays (per-commit).
    pub custom_commits_map: &'a [Value],
    /// `starkInfo.boundaries` array (for Zi boundary name / offsetMin / offsetMax).
    pub boundaries: &'a [Value],
}

// ── ref() ─────────────────────────────────────────────────────────────────────

/// Translate a single operand JSON object into a Circom signal reference string.
///
/// `is_dest` — set when this is the destination of the instruction so that a
/// first-use `tmp` gains a `signal` declaration prefix.
/// `initialized` — set of `tmp` ids already declared in this scope.
pub fn ref_operand(r: &Value, is_dest: bool, initialized: &[u64], ctx: &UnrollCtx<'_>) -> Result<String> {
    let typ = r["type"].as_str().unwrap_or("");
    match typ {
        "eval" => {
            let id = r["id"].as_u64().unwrap_or(0);
            Ok(format!("evals[{id}]"))
        }
        "challenge" => {
            let stage = r["stage"].as_u64().unwrap_or(0);
            let stage_id = r["stageId"].as_u64().unwrap_or(0);
            if stage == ctx.q_stage {
                Ok("challengeQ".into())
            } else if stage == ctx.evals_stage {
                Ok("challengeXi".into())
            } else if stage == ctx.fri_stage {
                Ok(format!("challengesFRI[{stage_id}]"))
            } else {
                Ok(format!("challengesStage{stage}[{stage_id}]"))
            }
        }
        "public" => {
            let id = r["id"].as_u64().unwrap_or(0);
            Ok(format!("publics[{id}]"))
        }
        "x" => Ok("challengeXi".into()),
        "Zi" => {
            let boundary_id = r["boundaryId"].as_u64().unwrap_or(0) as usize;
            let boundary = ctx
                .boundaries
                .get(boundary_id)
                .ok_or_else(|| anyhow::anyhow!("Zi: invalid boundaryId {boundary_id}"))?;
            let name = boundary["name"].as_str().unwrap_or("");
            match name {
                "everyRow" => Ok("Zh".into()),
                "firstRow" => Ok("Zfirst".into()),
                "lastRow" => Ok("Zlast".into()),
                "everyFrame" => {
                    // Find the index among everyFrame boundaries that matches
                    // offsetMin + offsetMax of this particular boundary entry.
                    let offset_min = boundary["offsetMin"].as_u64().unwrap_or(0);
                    let offset_max = boundary["offsetMax"].as_u64().unwrap_or(0);
                    let frame_id = ctx
                        .boundaries
                        .iter()
                        .filter(|b| b["name"].as_str() == Some("everyFrame"))
                        .position(|b| {
                            b["offsetMin"].as_u64() == Some(offset_min) && b["offsetMax"].as_u64() == Some(offset_max)
                        })
                        .ok_or_else(|| anyhow::anyhow!("Zi everyFrame: no matching boundary"))?;
                    let idx = offset_min + offset_max - 1;
                    Ok(format!("Zframe{frame_id}[{idx}]"))
                }
                other => bail!("Zi: unknown boundary name '{other}'"),
            }
        }
        "xDivXSubXi" => {
            let id = r["id"].as_u64().unwrap_or(0);
            Ok(format!("xDivXSubXi[{id}]"))
        }
        "tmp" => {
            let id = r["id"].as_u64().unwrap_or(0);
            let dim = r["dim"].as_u64().unwrap_or(1);
            if is_dest && !initialized.contains(&id) {
                if dim == 1 {
                    Ok(format!("signal tmp_{id}"))
                } else {
                    Ok(format!("signal tmp_{id}[3]"))
                }
            } else {
                Ok(format!("tmp_{id}"))
            }
        }
        "cm" => {
            let id = r["id"].as_u64().unwrap_or(0) as usize;
            let pol = ctx.cm_pols_map.get(id).ok_or_else(|| anyhow::anyhow!("cm: id {id} out of cmPolsMap"))?;
            let stage = pol["stage"].as_u64().unwrap_or(0);
            let stage_id = pol["stageId"].as_u64().unwrap_or(0);
            Ok(format!("mapValues.cm{stage}_{stage_id}"))
        }
        "custom" => {
            let commit_id = r["commitId"].as_u64().unwrap_or(0) as usize;
            let id = r["id"].as_u64().unwrap_or(0) as usize;
            let commit_map_row = ctx
                .custom_commits_map
                .get(commit_id)
                .and_then(|v| v.as_array())
                .ok_or_else(|| anyhow::anyhow!("custom: commitId {commit_id} out of customCommitsMap"))?;
            let pol = commit_map_row
                .get(id)
                .ok_or_else(|| anyhow::anyhow!("custom: id {id} out of customCommitsMap[{commit_id}]"))?;
            let pol_stage = pol["stage"].as_u64().unwrap_or(0);
            let pol_stage_id = pol["stageId"].as_u64().unwrap_or(0);
            let name = ctx.custom_commits.get(commit_id).and_then(|c| c["name"].as_str()).unwrap_or("unknown");
            Ok(format!("mapValues.custom_{name}_{pol_stage}_{pol_stage_id}"))
        }
        "const" => {
            let id = r["id"].as_u64().unwrap_or(0);
            Ok(format!("consts[{id}]"))
        }
        "number" => {
            let v = r["value"].as_str().unwrap_or("0");
            Ok(v.to_string())
        }
        "airgroupvalue" => {
            let id = r["id"].as_u64().unwrap_or(0);
            Ok(format!("airgroupvalues[{id}]"))
        }
        "airvalue" => {
            let id = r["id"].as_u64().unwrap_or(0);
            let dim = r["dim"].as_u64().unwrap_or(1);
            if dim == 1 {
                Ok(format!("airvalues[{id}][0]"))
            } else {
                Ok(format!("airvalues[{id}]"))
            }
        }
        "proofvalue" => {
            let id = r["id"].as_u64().unwrap_or(0);
            let dim = r["dim"].as_u64().unwrap_or(1);
            if dim == 1 {
                Ok(format!("proofvalues[{id}][0]"))
            } else {
                Ok(format!("proofvalues[{id}]"))
            }
        }
        other => bail!("ref_operand: unknown type '{other}'"),
    }
}

// ── unroll_code() ─────────────────────────────────────────────────────────────

/// Translate a slice of FPM instructions into Circom lines.
///
/// `initialized` — tmp ids already declared before this call (cross-chunk
/// inputs that do NOT need a `signal` prefix).
///
/// Returns the circom signal name of the last instruction's destination
/// (matching JS `return ref(code[code.length-1].dest)`).
///
/// All produced lines are `push`ed into `out` (4-space indented, matching the
/// EJS output), but without a trailing newline on each line — callers join
/// with `\n`.
pub fn unroll_code(code: &[Value], initialized: &[u64], ctx: &UnrollCtx<'_>, out: &mut Vec<String>) -> Result<String> {
    // Track which tmps have been declared in this call so we don't double-declare.
    let mut declared: Vec<u64> = initialized.to_vec();

    for inst in code {
        let op = inst["op"].as_str().unwrap_or("");
        let dest = &inst["dest"];
        let src = &inst["src"];

        // Validate dest dimension.
        if dest["type"].as_str() == Some("tmp") {
            let dim = dest["dim"].as_u64().unwrap_or(0);
            if dim != 1 && dim != 3 {
                bail!("unroll_code: invalid dest dim {dim}");
            }
        }

        // Force dim=3 on Zi and airgroupvalue sources (matching EJS logic).
        let mut s0 = src.get(0).cloned().unwrap_or(Value::Null);
        let mut s1 = src.get(1).cloned().unwrap_or(Value::Null);
        if matches!(s0["type"].as_str(), Some("Zi") | Some("airgroupvalue")) {
            s0["dim"] = 3.into();
        }
        if matches!(s1["type"].as_str(), Some("Zi") | Some("airgroupvalue")) {
            s1["dim"] = 3.into();
        }

        let dest_str = ref_operand(dest, true, &declared, ctx)?;

        // After we compute dest_str, mark this tmp as declared.
        if dest["type"].as_str() == Some("tmp") {
            let id = dest["id"].as_u64().unwrap_or(0);
            if !declared.contains(&id) {
                declared.push(id);
            }
        }

        let line = match op {
            "add" => {
                let d0 = s0["dim"].as_u64().unwrap_or(1);
                let d1 = s1["dim"].as_u64().unwrap_or(1);
                let a = ref_operand(&s0, false, &declared, ctx)?;
                let b = ref_operand(&s1, false, &declared, ctx)?;
                match (d0, d1) {
                    (1, 1) => format!("    {dest_str} <== {a} + {b};"),
                    (1, 3) => format!("    {dest_str} <== [{a} + {b}[0], {b}[1],  {b}[2]];"),
                    (3, 1) => format!("    {dest_str} <== [{a}[0] + {b}, {a}[1], {a}[2]];"),
                    (3, 3) => format!("    {dest_str} <== [{a}[0] + {b}[0], {a}[1] + {b}[1], {a}[2] + {b}[2]];"),
                    _ => bail!("add: invalid src dims {d0},{d1}"),
                }
            }
            "sub" => {
                let d0 = s0["dim"].as_u64().unwrap_or(1);
                let d1 = s1["dim"].as_u64().unwrap_or(1);
                let a = ref_operand(&s0, false, &declared, ctx)?;
                let b = ref_operand(&s1, false, &declared, ctx)?;
                match (d0, d1) {
                    (1, 1) => format!("    {dest_str} <== {a} - {b};"),
                    (1, 3) => format!("    {dest_str} <== [{a} - {b}[0], -{b}[1], -{b}[2]];"),
                    (3, 1) => format!("    {dest_str} <== [{a}[0] - {b}, {a}[1], {a}[2]];"),
                    (3, 3) => format!("    {dest_str} <== [{a}[0] - {b}[0], {a}[1] - {b}[1], {a}[2] - {b}[2]];"),
                    _ => bail!("sub: invalid src dims {d0},{d1}"),
                }
            }
            "mul" => {
                let d0 = s0["dim"].as_u64().unwrap_or(1);
                let d1 = s1["dim"].as_u64().unwrap_or(1);
                let a = ref_operand(&s0, false, &declared, ctx)?;
                let b = ref_operand(&s1, false, &declared, ctx)?;
                match (d0, d1) {
                    (1, 1) => format!("    {dest_str} <== {a} * {b};"),
                    (1, 3) => format!("    {dest_str} <== [{a} * {b}[0], {a} * {b}[1], {a} * {b}[2]];"),
                    (3, 1) => format!("    {dest_str} <== [{a}[0] * {b}, {a}[1] * {b}, {a}[2] * {b}];"),
                    (3, 3) => format!("    {dest_str} <== CMul()({a}, {b});"),
                    _ => bail!("mul: invalid src dims {d0},{d1}"),
                }
            }
            "copy" => {
                let a = ref_operand(&s0, false, &declared, ctx)?;
                format!("    {dest_str} <== {a};")
            }
            other => bail!("unroll_code: unknown op '{other}'"),
        };
        out.push(line);
    }

    // Return the signal name (not declaration) of the last destination.
    if let Some(last) = code.last() {
        // Pass initialized=&declared so we get plain "tmp_N" not "signal tmp_N".
        ref_operand(&last["dest"], false, &declared, ctx)
    } else {
        Ok(String::new())
    }
}

// ── BN128 variant ─────────────────────────────────────────────────────────────

/// GL field prime (for normalising negative number values in BN128 unroll).
const GL_PRIME: u128 = 18_446_744_069_414_584_321;

/// Normalise a `number` value string (possibly negative) to a non-negative GL field element.
fn normalise_gl_number(raw: &str) -> u128 {
    if let Ok(v) = raw.parse::<i128>() {
        if v < 0 {
            (v.rem_euclid(GL_PRIME as i128)) as u128
        } else {
            v as u128
        }
    } else {
        0
    }
}

/// BN128-flavour code-generation for a slice of FPM instructions.
///
/// Produces Circom lines using `GLAdd`, `GLSub`, `GLMul`, `GLCAdd`, `GLCSub`,
/// `GLCMul`, `GLConst`, `GLConst3`, `GLC3`, `GLCopy`, `GLCCopy` as the BN128
/// EJS does, with `{maxNum}` annotations on every declared signal.
///
/// Returns the last destination signal name (without declaration prefix).
pub fn unroll_code_bn128(
    code: &[Value],
    initialized: &[u64],
    ctx: &UnrollCtx<'_>,
    out: &mut Vec<String>,
) -> Result<String> {
    let mut declared: Vec<u64> = initialized.to_vec();
    // Dedup table for number constants: (norm_val, dim) → signal_name
    let mut const_signals: std::collections::HashMap<(u128, u64), String> = Default::default();
    let mut const_cnt: usize = 0;
    // GLC3 wrap counter
    let mut value_cnt: usize = 0;

    // Inline helper: declare dest tmp signal if first use, return signal name.
    macro_rules! dest_sig {
        ($dest:expr) => {{
            if $dest["type"].as_str() == Some("tmp") {
                let id = $dest["id"].as_u64().unwrap_or(0);
                let dim = $dest["dim"].as_u64().unwrap_or(1);
                if !declared.contains(&id) {
                    declared.push(id);
                    let arr = if dim == 3 { "[3]" } else { "" };
                    out.push(format!("    signal {{maxNum}} tmp_{id}{arr};"));
                }
                format!("tmp_{id}")
            } else {
                ref_operand_bn128($dest, ctx)?
            }
        }};
    }

    for inst in code {
        let op = inst["op"].as_str().unwrap_or("");
        let dest = &inst["dest"];
        let src = &inst["src"];

        if dest["type"].as_str() == Some("tmp") {
            let dim = dest["dim"].as_u64().unwrap_or(0);
            if dim != 1 && dim != 3 {
                bail!("unroll_code_bn128: invalid dest dim {dim}");
            }
        }

        let mut s0 = src.get(0).cloned().unwrap_or(Value::Null);
        let mut s1 = src.get(1).cloned().unwrap_or(Value::Null);
        if matches!(s0["type"].as_str(), Some("Zi") | Some("airgroupvalue")) {
            s0["dim"] = 3.into();
        }
        if matches!(s1["type"].as_str(), Some("Zi") | Some("airgroupvalue")) {
            s1["dim"] = 3.into();
        }

        let dest_dim = dest["dim"].as_u64().unwrap_or(1);

        match op {
            "add" => {
                let d0 = s0["dim"].as_u64().unwrap_or(1);
                let d1 = s1["dim"].as_u64().unwrap_or(1);
                let dest_sig = dest_sig!(dest);
                let (a, b) = resolve_add_sub_pair(
                    &s0,
                    d0,
                    &s1,
                    d1,
                    dest_dim,
                    ctx,
                    out,
                    &mut const_signals,
                    &mut const_cnt,
                    &mut value_cnt,
                )?;
                if d0 == 1 && d1 == 1 {
                    out.push(format!("    {dest_sig} <== GLAdd()({a}, {b});"));
                } else {
                    out.push(format!("    {dest_sig} <== GLCAdd()({a}, {b});"));
                }
            }
            "sub" => {
                let d0 = s0["dim"].as_u64().unwrap_or(1);
                let d1 = s1["dim"].as_u64().unwrap_or(1);
                let dest_sig = dest_sig!(dest);
                // Special case: sub(3,1) where s1 is number 0 → GLCCopy
                if d0 == 3 && d1 == 1 && s1["type"].as_str() == Some("number") {
                    let raw = s1["value"].as_str().unwrap_or("0");
                    if normalise_gl_number(raw) == 0 {
                        let a = resolve_src_bn128(&s0, d0, out, &mut const_signals, &mut const_cnt, ctx)?;
                        out.push(format!("    {dest_sig} <== GLCCopy()({a});"));
                        continue;
                    }
                }
                let (a, b) = resolve_add_sub_pair(
                    &s0,
                    d0,
                    &s1,
                    d1,
                    dest_dim,
                    ctx,
                    out,
                    &mut const_signals,
                    &mut const_cnt,
                    &mut value_cnt,
                )?;
                if d0 == 1 && d1 == 1 {
                    out.push(format!("    {dest_sig} <== GLSub()({a}, {b});"));
                } else {
                    out.push(format!("    {dest_sig} <== GLCSub()({a}, {b});"));
                }
            }
            "mul" => {
                let d0 = s0["dim"].as_u64().unwrap_or(1);
                let d1 = s1["dim"].as_u64().unwrap_or(1);
                let dest_sig = dest_sig!(dest);
                let (a, b) = resolve_mul_pair(
                    &s0,
                    d0,
                    &s1,
                    d1,
                    dest_dim,
                    ctx,
                    out,
                    &mut const_signals,
                    &mut const_cnt,
                    &mut value_cnt,
                )?;
                if d0 == 1 && d1 == 1 {
                    out.push(format!("    {dest_sig} <== GLMul()({a}, {b});"));
                } else {
                    out.push(format!("    {dest_sig} <== GLCMul()({a}, {b});"));
                }
            }
            "copy" => {
                let d0 = s0["dim"].as_u64().unwrap_or(1);
                let dest_sig = dest_sig!(dest);
                let a = resolve_src_bn128(&s0, d0, out, &mut const_signals, &mut const_cnt, ctx)?;
                if d0 == 1 {
                    out.push(format!("    {dest_sig} <== GLCopy()({a});"));
                } else {
                    out.push(format!("    {dest_sig} <== {a};"));
                }
            }
            other => bail!("unroll_code_bn128: unknown op '{other}'"),
        }
    }

    if let Some(last) = code.last() {
        ref_operand_bn128(&last["dest"], ctx)
    } else {
        Ok(String::new())
    }
}

/// BN128-specific ref resolution: same fields as GL but no is_dest/decl tracking.
/// Numbers are NOT handled here — callers handle them separately via emit_const.
fn ref_operand_bn128(r: &Value, ctx: &UnrollCtx<'_>) -> Result<String> {
    let typ = r["type"].as_str().unwrap_or("");
    match typ {
        "eval" => Ok(format!("evals[{}]", r["id"].as_u64().unwrap_or(0))),
        "challenge" => {
            let stage = r["stage"].as_u64().unwrap_or(0);
            let stage_id = r["stageId"].as_u64().unwrap_or(0);
            if stage == ctx.q_stage {
                Ok("challengeQ".into())
            } else if stage == ctx.evals_stage {
                Ok("challengeXi".into())
            } else if stage == ctx.fri_stage {
                Ok(format!("challengesFRI[{stage_id}]"))
            } else {
                Ok(format!("challengesStage{stage}[{stage_id}]"))
            }
        }
        "public" => Ok(format!("publics[{}]", r["id"].as_u64().unwrap_or(0))),
        "x" => Ok("challengeXi".into()),
        "Zi" => {
            let boundary_id = r["boundaryId"].as_u64().unwrap_or(0) as usize;
            let boundary = ctx
                .boundaries
                .get(boundary_id)
                .ok_or_else(|| anyhow::anyhow!("Zi: invalid boundaryId {boundary_id}"))?;
            let name = boundary["name"].as_str().unwrap_or("");
            match name {
                "everyRow" => Ok("Zh".into()),
                "firstRow" => Ok("Zfirst".into()),
                "lastRow" => Ok("Zlast".into()),
                "everyFrame" => {
                    let offset_min = boundary["offsetMin"].as_u64().unwrap_or(0);
                    let offset_max = boundary["offsetMax"].as_u64().unwrap_or(0);
                    let frame_id = ctx
                        .boundaries
                        .iter()
                        .filter(|b| b["name"].as_str() == Some("everyFrame"))
                        .position(|b| {
                            b["offsetMin"].as_u64() == Some(offset_min) && b["offsetMax"].as_u64() == Some(offset_max)
                        })
                        .ok_or_else(|| anyhow::anyhow!("Zi everyFrame: no matching boundary"))?;
                    let idx = offset_min + offset_max - 1;
                    Ok(format!("Zframe{frame_id}[{idx}]"))
                }
                other => bail!("Zi: unknown boundary name '{other}'"),
            }
        }
        "xDivXSubXi" => Ok(format!("xDivXSubXi[{}]", r["id"].as_u64().unwrap_or(0))),
        "tmp" => {
            let id = r["id"].as_u64().unwrap_or(0);
            Ok(format!("tmp_{id}"))
        }
        "cm" => {
            let id = r["id"].as_u64().unwrap_or(0) as usize;
            let pol = ctx.cm_pols_map.get(id).ok_or_else(|| anyhow::anyhow!("cm: id {id} out of cmPolsMap"))?;
            let stage = pol["stage"].as_u64().unwrap_or(0);
            let stage_id = pol["stageId"].as_u64().unwrap_or(0);
            Ok(format!("mapValues.cm{stage}_{stage_id}"))
        }
        "custom" => {
            let commit_id = r["commitId"].as_u64().unwrap_or(0) as usize;
            let id = r["id"].as_u64().unwrap_or(0) as usize;
            let commit_map_row = ctx
                .custom_commits_map
                .get(commit_id)
                .and_then(|v| v.as_array())
                .ok_or_else(|| anyhow::anyhow!("custom: commitId {commit_id} out of range"))?;
            let pol = commit_map_row.get(id).ok_or_else(|| anyhow::anyhow!("custom: id {id} out of range"))?;
            let pol_stage = pol["stage"].as_u64().unwrap_or(0);
            let pol_stage_id = pol["stageId"].as_u64().unwrap_or(0);
            let name = ctx.custom_commits.get(commit_id).and_then(|c| c["name"].as_str()).unwrap_or("unknown");
            Ok(format!("mapValues.custom_{name}_{pol_stage}_{pol_stage_id}"))
        }
        "const" => Ok(format!("consts[{}]", r["id"].as_u64().unwrap_or(0))),
        "number" => {
            // Callers should handle number via emit_const; fall back to raw value string.
            let raw = r["value"].as_str().unwrap_or("0");
            Ok(normalise_gl_number(raw).to_string())
        }
        "airgroupvalue" => Ok(format!("airgroupvalues[{}]", r["id"].as_u64().unwrap_or(0))),
        "airvalue" => {
            let id = r["id"].as_u64().unwrap_or(0);
            let dim = r["dim"].as_u64().unwrap_or(1);
            Ok(if dim == 1 { format!("airvalues[{id}][0]") } else { format!("airvalues[{id}]") })
        }
        "proofvalue" => {
            let id = r["id"].as_u64().unwrap_or(0);
            let dim = r["dim"].as_u64().unwrap_or(1);
            Ok(if dim == 1 { format!("proofvalues[{id}][0]") } else { format!("proofvalues[{id}]") })
        }
        other => bail!("ref_operand_bn128: unknown type '{other}'"),
    }
}

/// Resolve a pair of operands for add/sub, inserting a GLC3 wrapper when dimensions mismatch.
#[allow(clippy::too_many_arguments)]
fn resolve_add_sub_pair(
    s0: &Value,
    d0: u64,
    s1: &Value,
    d1: u64,
    dest_dim: u64,
    ctx: &UnrollCtx<'_>,
    out: &mut Vec<String>,
    const_signals: &mut std::collections::HashMap<(u128, u64), String>,
    const_cnt: &mut usize,
    value_cnt: &mut usize,
) -> Result<(String, String)> {
    let raw0 = if s0["type"].as_str() == Some("number") {
        emit_const_inner(s0["value"].as_str().unwrap_or("0"), d0, out, const_signals, const_cnt)
    } else {
        ref_operand_bn128(s0, ctx)?
    };
    let raw1 = if s1["type"].as_str() == Some("number") {
        emit_const_inner(s1["value"].as_str().unwrap_or("0"), d1, out, const_signals, const_cnt)
    } else {
        ref_operand_bn128(s1, ctx)?
    };

    // Wrap if dimension mismatch (non-zero, non-const)
    let a = if d0 == 1
        && d1 == 3
        && (s0["type"].as_str() != Some("number") || {
            let n = normalise_gl_number(s0["value"].as_str().unwrap_or("0"));
            n != 0 && !const_signals.contains_key(&(n, dest_dim))
        }) {
        let sig = format!("value_{}", *value_cnt);
        *value_cnt += 1;
        out.push(format!("    signal {{maxNum}} {sig}[3] <== GLC3()({raw0});"));
        sig
    } else {
        raw0
    };

    let b = if d1 == 1
        && d0 == 3
        && (s1["type"].as_str() != Some("number") || {
            let n = normalise_gl_number(s1["value"].as_str().unwrap_or("0"));
            n != 0 && !const_signals.contains_key(&(n, dest_dim))
        }) {
        let sig = format!("value_{}", *value_cnt);
        *value_cnt += 1;
        out.push(format!("    signal {{maxNum}} {sig}[3] <== GLC3()({raw1});"));
        sig
    } else {
        raw1
    };

    Ok((a, b))
}

/// Resolve a pair for mul — same wrapping logic as add/sub.
#[allow(clippy::too_many_arguments)]
fn resolve_mul_pair(
    s0: &Value,
    d0: u64,
    s1: &Value,
    d1: u64,
    dest_dim: u64,
    ctx: &UnrollCtx<'_>,
    out: &mut Vec<String>,
    const_signals: &mut std::collections::HashMap<(u128, u64), String>,
    const_cnt: &mut usize,
    value_cnt: &mut usize,
) -> Result<(String, String)> {
    resolve_add_sub_pair(s0, d0, s1, d1, dest_dim, ctx, out, const_signals, const_cnt, value_cnt)
}

/// Emit a GLConst / GLConst3 signal (with dedup). Returns signal name.
fn emit_const_inner(
    val_raw: &str,
    dim: u64,
    out: &mut Vec<String>,
    const_signals: &mut std::collections::HashMap<(u128, u64), String>,
    const_cnt: &mut usize,
) -> String {
    let norm = normalise_gl_number(val_raw);
    let key = (norm, dim);
    if let Some(existing) = const_signals.get(&key) {
        return existing.clone();
    }
    let sig = format!("constValue_{}", *const_cnt);
    *const_cnt += 1;
    let arr = if dim == 3 { "[3]" } else { "" };
    let tpl = if dim == 3 { format!("GLConst3({norm})") } else { format!("GLConst({norm})") };
    out.push(format!("    signal {{maxNum}} {sig}{arr} <== {tpl}();"));
    const_signals.insert(key, sig.clone());
    sig
}

/// Resolve a src operand for BN128; number refs get emitted as GLConst signals.
fn resolve_src_bn128(
    r: &Value,
    expected_dim: u64,
    out: &mut Vec<String>,
    const_signals: &mut std::collections::HashMap<(u128, u64), String>,
    const_cnt: &mut usize,
    ctx: &UnrollCtx<'_>,
) -> Result<String> {
    if r["type"].as_str() == Some("number") {
        let raw = r["value"].as_str().unwrap_or("0");
        Ok(emit_const_inner(raw, expected_dim, out, const_signals, const_cnt))
    } else {
        ref_operand_bn128(r, ctx)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn make_ctx() -> (Vec<Value>, Vec<Value>, Vec<Value>, Vec<Value>) {
        // cmPolsMap: one entry with stage=1, stageId=0
        let cm_pols_map = vec![json!({"stage": 1, "stageId": 0})];
        // customCommits and customCommitsMap: empty
        let custom_commits: Vec<Value> = vec![];
        let custom_commits_map: Vec<Value> = vec![];
        // boundaries: everyRow at index 0
        let boundaries = vec![json!({"name": "everyRow"})];
        (cm_pols_map, custom_commits, custom_commits_map, boundaries)
    }

    #[test]
    fn test_add_1x1() {
        let (cm, cc, ccm, bounds) = make_ctx();
        let ctx = UnrollCtx {
            q_stage: 3,
            evals_stage: 4,
            fri_stage: 5,
            cm_pols_map: &cm,
            custom_commits: &cc,
            custom_commits_map: &ccm,
            boundaries: &bounds,
        };
        let code = vec![json!({
            "op": "add",
            "dest": {"type": "tmp", "id": 0, "dim": 1},
            "src": [
                {"type": "eval", "id": 1, "dim": 1},
                {"type": "eval", "id": 2, "dim": 1}
            ]
        })];
        let mut out = Vec::new();
        let last = unroll_code(&code, &[], &ctx, &mut out).unwrap();
        assert_eq!(out[0], "    signal tmp_0 <== evals[1] + evals[2];");
        assert_eq!(last, "tmp_0");
    }

    #[test]
    fn test_mul_3x3() {
        let (cm, cc, ccm, bounds) = make_ctx();
        let ctx = UnrollCtx {
            q_stage: 3,
            evals_stage: 4,
            fri_stage: 5,
            cm_pols_map: &cm,
            custom_commits: &cc,
            custom_commits_map: &ccm,
            boundaries: &bounds,
        };
        let code = vec![json!({
            "op": "mul",
            "dest": {"type": "tmp", "id": 5, "dim": 3},
            "src": [
                {"type": "eval", "id": 0, "dim": 3},
                {"type": "eval", "id": 1, "dim": 3}
            ]
        })];
        let mut out = Vec::new();
        unroll_code(&code, &[], &ctx, &mut out).unwrap();
        assert_eq!(out[0], "    signal tmp_5[3] <== CMul()(evals[0], evals[1]);");
    }

    #[test]
    fn test_tmp_initialized_no_redecl() {
        let (cm, cc, ccm, bounds) = make_ctx();
        let ctx = UnrollCtx {
            q_stage: 3,
            evals_stage: 4,
            fri_stage: 5,
            cm_pols_map: &cm,
            custom_commits: &cc,
            custom_commits_map: &ccm,
            boundaries: &bounds,
        };
        let code = vec![json!({
            "op": "copy",
            "dest": {"type": "tmp", "id": 7, "dim": 3},
            "src": [{"type": "tmp", "id": 3, "dim": 3}]
        })];
        let mut out = Vec::new();
        // tmp_7 is in initialized — so dest should NOT get "signal" prefix.
        unroll_code(&code, &[7], &ctx, &mut out).unwrap();
        assert_eq!(out[0], "    tmp_7 <== tmp_3;");
    }

    #[test]
    fn test_challenge_stages() {
        let (cm, cc, ccm, bounds) = make_ctx();
        let ctx = UnrollCtx {
            q_stage: 3,
            evals_stage: 4,
            fri_stage: 5,
            cm_pols_map: &cm,
            custom_commits: &cc,
            custom_commits_map: &ccm,
            boundaries: &bounds,
        };
        let code = vec![json!({
            "op": "copy",
            "dest": {"type": "tmp", "id": 1, "dim": 3},
            "src": [{"type": "challenge", "stage": 3, "stageId": 0, "dim": 3}]
        })];
        let mut out = Vec::new();
        unroll_code(&code, &[], &ctx, &mut out).unwrap();
        assert!(out[0].contains("challengeQ"), "got: {}", out[0]);

        let code2 = vec![json!({
            "op": "copy",
            "dest": {"type": "tmp", "id": 2, "dim": 3},
            "src": [{"type": "challenge", "stage": 2, "stageId": 1, "dim": 3}]
        })];
        let mut out2 = Vec::new();
        unroll_code(&code2, &[], &ctx, &mut out2).unwrap();
        assert!(out2[0].contains("challengesStage2[1]"), "got: {}", out2[0]);
    }

    #[test]
    fn test_zi_every_row() {
        let (cm, cc, ccm, bounds) = make_ctx();
        let ctx = UnrollCtx {
            q_stage: 3,
            evals_stage: 4,
            fri_stage: 5,
            cm_pols_map: &cm,
            custom_commits: &cc,
            custom_commits_map: &ccm,
            boundaries: &bounds,
        };
        let code = vec![json!({
            "op": "copy",
            "dest": {"type": "tmp", "id": 10, "dim": 3},
            "src": [{"type": "Zi", "boundaryId": 0, "dim": 3}]
        })];
        let mut out = Vec::new();
        unroll_code(&code, &[], &ctx, &mut out).unwrap();
        assert!(out[0].contains("Zh"), "got: {}", out[0]);
    }

    #[test]
    fn test_number_value() {
        let (cm, cc, ccm, bounds) = make_ctx();
        let ctx = UnrollCtx {
            q_stage: 3,
            evals_stage: 4,
            fri_stage: 5,
            cm_pols_map: &cm,
            custom_commits: &cc,
            custom_commits_map: &ccm,
            boundaries: &bounds,
        };
        let code = vec![json!({
            "op": "mul",
            "dest": {"type": "tmp", "id": 20, "dim": 1},
            "src": [
                {"type": "eval", "id": 0, "dim": 1},
                {"type": "number", "value": "42", "dim": 1}
            ]
        })];
        let mut out = Vec::new();
        unroll_code(&code, &[], &ctx, &mut out).unwrap();
        assert_eq!(out[0], "    signal tmp_20 <== evals[0] * 42;");
    }
}
