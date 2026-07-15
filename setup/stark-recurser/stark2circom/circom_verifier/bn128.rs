//! Rust port of `pil2circom.js` for the **BN128** verification hash type.
//!
//! Generates a `stark_verifier.circom` from `starkInfo` + `verifierInfo` + options
//! using the `circuits.bn128/stark_verifier.circom.ejs` template structure.
//!
//! Key differences from the GL generator (`pil2circom.rs`):
//!
//! - Hash primitive: `PoseidonEx` (or `CustomPoseidon` when `custom=true`)
//! - No expression chunking (`VerifyEvaluationsChunks`/`CalculateFRIPolChunks`)
//! - No `airgroupId` suffix on template names
//! - No batched query verification (`VerifyQueriesBatch`) — queries loop in `StarkVerifier`
//! - Has a `Main()` wrapper template (with SHA256 publics hash) that wraps `StarkVerifier`
//! - `component main = Main();` (not `StarkVerifier()`)
//! - `options.inputChallenges` does not apply to BN128
//! - Uses `GLConst`, `GLConst3`, `GLC3` inline templates

use anyhow::{bail, Result};
use serde_json::Value;
use tera::Tera;

use super::gl_field::{gl_exp, gl_inv, gl_mul, GL_SHIFT, GL_W};
use super::gl::Pil2CircomOptions;
use super::transcript_bn128::TranscriptBn128;
use super::unroll_code::{unroll_code_bn128, UnrollCtx};

// ── Top-level entry-point ─────────────────────────────────────────────────────

/// Embedded Tera template (BN128 stark verifier).
const TEMPLATE_BN128: &str = include_str!("tera/stark_verifier_bn128.circom.tera");

/// Generate a BN128 `stark_verifier.circom` entirely in Rust.
pub fn gen_stark_verifier_bn128(
    const_root: Option<&[String; 4]>,
    stark_info: &Value,
    verifier_info: &Value,
    opts: &Pil2CircomOptions,
) -> Result<String> {
    let hash_type = stark_info["starkStruct"]["verificationHashType"].as_str().unwrap_or("GL");
    if hash_type != "BN128" {
        bail!("gen_stark_verifier_bn128: expected BN128 hash type, got '{hash_type}'");
    }

    let ctx = build_tera_context_bn128(stark_info, verifier_info, opts, const_root)?;
    Tera::one_off(TEMPLATE_BN128, &ctx, false).map_err(|e| anyhow::anyhow!("Tera render error (BN128): {e}"))
}

// ── Tera context builder ──────────────────────────────────────────────────────

fn build_tera_context_bn128(
    si: &Value,
    vi: &Value,
    opts: &Pil2CircomOptions,
    const_root: Option<&[String; 4]>,
) -> Result<tera::Context> {
    let mut ctx = tera::Context::new();

    let ss = &si["starkStruct"];
    let n_stages = si["nStages"].as_u64().unwrap_or(0) as usize;
    let q_stage = n_stages + 1;
    let evals_stage = q_stage + 1;
    let fri_stage = evals_stage + 1;
    let arity = ss["merkleTreeArity"].as_u64().unwrap_or(16) as usize;
    let custom = ss["merkleTreeCustom"].as_bool().unwrap_or(false);
    let transcript_arity = if custom { arity } else { 16usize };
    let n_bits_arity = (arity as f64).log2().ceil() as usize;
    let n_queries = ss["nQueries"].as_u64().unwrap_or(0) as usize;
    let last_level_verification = ss["lastLevelVerification"].as_u64().unwrap_or(0) as usize;
    if last_level_verification > 0 && custom {
        bail!(
            "gen_stark_verifier_bn128: lastLevelVerification > 0 is not supported with \
             merkleTreeCustom (circuits.bn128/custom/merklehash.circom lacks the templates)"
        );
    }
    let s0_last_mt_size = if last_level_verification > 0 { arity.pow(last_level_verification as u32) } else { 0 };
    let n_bits = ss["nBits"].as_u64().unwrap_or(0) as usize;
    let n_bits_ext = ss["nBitsExt"].as_u64().unwrap_or(0) as usize;
    let pow_bits = ss["powBits"].as_u64().unwrap_or(0);
    let hash_commits = ss["hashCommits"].as_bool().unwrap_or(false);
    let q_deg = si["qDeg"].as_u64().unwrap_or(1) as usize;
    let steps: Vec<Value> = ss["steps"].as_array().map_or(vec![], |a| a.clone());
    let n_steps = steps.len();
    let step0_bits = steps.first().and_then(|s| s["nBits"].as_u64()).unwrap_or(0) as usize;
    let last_step_bits = steps.last().and_then(|s| s["nBits"].as_u64()).unwrap_or(0) as usize;
    let final_pol_size: usize = 1 << last_step_bits;
    let n_last_bits = last_step_bits;
    let max_deg_bits = (n_last_bits as isize - (n_bits_ext - n_bits) as isize).max(0) as usize;
    let max_deg_size: usize = 1 << max_deg_bits;
    let n_publics = si["nPublics"].as_u64().unwrap_or(0) as usize;
    let n_constants = si["nConstants"].as_u64().unwrap_or(0) as usize;
    let ev_map_len = si["evMap"].as_array().map_or(0, |a| a.len());
    let n_air_group_values = si["airgroupValuesMap"].as_array().map_or(0, |a| a.len());
    let challenges_map: Vec<Value> = si["challengesMap"].as_array().map_or(vec![], |a| a.clone());
    let cm_pols_map: Vec<Value> = si["cmPolsMap"].as_array().map_or(vec![], |a| a.clone());
    let custom_commits_json: Vec<Value> = si["customCommits"].as_array().map_or(vec![], |a| a.clone());
    let custom_commits_map_json: Vec<Value> = si["customCommitsMap"].as_array().map_or(vec![], |a| a.clone());
    let opening_points_json: Vec<Value> = si["openingPoints"].as_array().map_or(vec![], |a| a.clone());
    let boundaries_json: Vec<Value> = si["boundaries"].as_array().map_or(vec![], |a| a.clone());
    let map_sections_n = &si["mapSectionsN"];

    let sec_len = |key: &str| -> usize { map_sections_n[key].as_u64().unwrap_or(0) as usize };

    // N_Fields for BN128 (253-bit Poseidon outputs)
    let total_bits = n_queries * step0_bits;
    let n_fields = if total_bits == 0 { 0 } else { (total_bits - 1) / 253 + 1 };

    // Merkle dimensions (truncated by lastLevelVerification: all template
    // usages of merkle_levels_s0 are sibling-array dims).
    let merkle_levels_s0 = if step0_bits == 0 {
        0usize
    } else {
        (((step0_bits - 1) / n_bits_arity) + 1).saturating_sub(last_level_verification)
    };

    // ── Boundary flags ─────────────────────────────────────────────────────
    let has_first_row = boundaries_json.iter().any(|b| b["name"].as_str() == Some("firstRow"));
    let has_last_row = boundaries_json.iter().any(|b| b["name"].as_str() == Some("lastRow"));

    let zlast_root: String = if has_last_row {
        let mut root: u64 = 1;
        for _ in 0..((1u64 << n_bits) - 1) {
            root = gl_mul(root, GL_W[n_bits]);
        }
        root.to_string()
    } else {
        "0".to_string()
    };

    // ── everyFrame boundaries ───────────────────────────────────────────────
    let every_frames: Vec<serde_json::Value> = boundaries_json
        .iter()
        .filter(|b| b["name"].as_str() == Some("everyFrame"))
        .enumerate()
        .map(|(i, frame)| {
            let offset_min = frame["offsetMin"].as_u64().unwrap_or(0) as usize;
            let offset_max = frame["offsetMax"].as_u64().unwrap_or(0) as usize;
            let size = offset_min + offset_max;
            let mut ops: Vec<serde_json::Value> = Vec::new();
            let mut c = 0usize;
            for j in 0..offset_min {
                let mut root: u64 = 1;
                for _ in 0..j {
                    root = gl_mul(root, GL_W[n_bits]);
                }
                ops.push(serde_json::json!({
                    "c": c, "is_first": c == 0, "root": root.to_string()
                }));
                c += 1;
            }
            let back_exp = (1u64 << n_bits).saturating_sub((i + 1) as u64);
            let mut back_root: u64 = 1;
            for _ in 0..back_exp {
                back_root = gl_mul(back_root, GL_W[n_bits]);
            }
            for _ in 0..offset_max {
                ops.push(serde_json::json!({
                    "c": c, "is_first": c == 0, "root": back_root.to_string()
                }));
                c += 1;
            }
            serde_json::json!({
                "idx": i, "offset_min": offset_min,
                "offset_max": offset_max, "size": size, "ops": ops,
            })
        })
        .collect();

    // ── Per-stage challenge counts ──────────────────────────────────────────
    let challenges_per_stage: Vec<serde_json::Value> = (1..=n_stages)
        .filter_map(|stage| {
            let cnt = challenges_map.iter().filter(|c| c["stage"].as_u64() == Some(stage as u64)).count();
            if cnt > 0 {
                Some(serde_json::json!({ "stage": stage, "count": cnt }))
            } else {
                None
            }
        })
        .collect();

    // ── Stages info ─────────────────────────────────────────────────────────
    let stages_info: Vec<serde_json::Value> = (1..=q_stage)
        .map(|stage| {
            let cm_section_len = sec_len(&format!("cm{stage}"));
            serde_json::json!({
                "stage": stage,
                "cm_section_len": cm_section_len,
                "has_cm": cm_section_len > 0,
            })
        })
        .collect();

    // ── cm_pols_by_stage (for MapValues) ───────────────────────────────────
    let cm_pols_by_stage: Vec<serde_json::Value> = (1..=q_stage)
        .map(|stage| {
            let pols: Vec<serde_json::Value> = cm_pols_map
                .iter()
                .filter(|p| p["stage"].as_u64() == Some(stage as u64))
                .map(|p| {
                    serde_json::json!({
                        "stage_id": p["stageId"].as_u64().unwrap_or(0),
                        "dim": p["dim"].as_u64().unwrap_or(1),
                        "stage_pos": p["stagePos"].as_u64().unwrap_or(0),
                    })
                })
                .collect();
            serde_json::json!({ "stage": stage, "pols": pols })
        })
        .collect();

    // ── custom_commits ──────────────────────────────────────────────────────
    let custom_commits: Vec<serde_json::Value> = custom_commits_json
        .iter()
        .enumerate()
        .map(|(t, cc)| {
            let name = cc["name"].as_str().unwrap_or("").to_string();
            let stage_widths_count = cc["stageWidths"].as_array().map_or(0, |a| a.len());
            let section_len_0 = sec_len(&format!("{name}0"));
            let public_values: Vec<u64> = cc["publicValues"]
                .as_array()
                .map_or(vec![], |a| a.clone())
                .iter()
                .map(|pv| pv["idx"].as_u64().unwrap_or(0))
                .collect();
            let commit_map_arr =
                custom_commits_map_json.get(t).and_then(|v| v.as_array()).map_or(vec![], |a| a.clone());
            let pols_per_stage: Vec<serde_json::Value> = (0..stage_widths_count)
                .map(|l| {
                    let pols: Vec<serde_json::Value> = commit_map_arr
                        .iter()
                        .filter(|p| p["stage"].as_u64() == Some(l as u64))
                        .map(|p| {
                            serde_json::json!({
                                "stage_id": p["stageId"].as_u64().unwrap_or(0),
                                "dim": p["dim"].as_u64().unwrap_or(1),
                                "stage_pos": p["stagePos"].as_u64().unwrap_or(0),
                            })
                        })
                        .collect();
                    serde_json::json!({ "stage_idx": l, "pols": pols })
                })
                .collect();
            serde_json::json!({
                "name": name,
                "section_len_0": section_len_0,
                "stage_widths_count": stage_widths_count,
                "pols_per_stage": pols_per_stage,
                "public_values": public_values,
            })
        })
        .collect();

    // ── FRI step info ───────────────────────────────────────────────────────
    let fri_steps_info: Vec<serde_json::Value> = (0..n_steps)
        .map(|s| {
            let n_bits_s = steps[s]["nBits"].as_u64().unwrap_or(0) as usize;
            let prev_bits = if s == 0 { n_bits_s } else { steps[s - 1]["nBits"].as_u64().unwrap_or(0) as usize };
            let next_bits = if s < n_steps - 1 { steps[s + 1]["nBits"].as_u64().unwrap_or(0) as usize } else { 0 };
            let exponent = if s == 0 { 1usize } else { 1 << (prev_bits - n_bits_s) };
            let full_ml = if n_bits_s == 0 { 0usize } else { ((n_bits_s - 1) / n_bits_arity) + 1 };
            let ml = full_ml.saturating_sub(last_level_verification);
            let is_empty = last_level_verification > 0 && full_ml <= last_level_verification;
            let last_mt_size = if last_level_verification > 0 { arity.pow(last_level_verification as u32) } else { 0 };
            let mt_size = if n_bits_s == 0 { 1usize } else { 1 << n_bits_s };

            // e0 = inv(shift^(1 << (nBitsExt - prevStepBits)))
            // e1 = inv(shift^(1 << (nBitsExt - prevStepBits)) * w[prevStepBits])
            let (e0, e1) = if s == 0 {
                ("0".to_string(), "0".to_string())
            } else {
                let exp = 1u64 << (n_bits_ext - prev_bits) as u64;
                let e0_val = gl_inv(gl_exp(GL_SHIFT, exp));
                let e1_val = gl_inv(gl_mul(gl_exp(GL_SHIFT, exp), GL_W[prev_bits]));
                (e0_val.to_string(), e1_val.to_string())
            };
            let is_last = s == n_steps - 1;
            let next_pol = if is_last { "finalPol".to_string() } else { format!("s{}_vals_p", s + 1) };
            let val_size = if n_bits_s == 0 { 1usize } else { 1 << n_bits_s };
            serde_json::json!({
                "s": s,
                "n_bits": n_bits_s,
                "prev_bits": prev_bits,
                "next_bits": next_bits,
                "exponent": exponent,
                "val_size": val_size,
                "merkle_levels": ml,
                "is_empty": is_empty,
                "last_mt_size": last_mt_size,
                "mt_size": mt_size,
                "e0": e0,
                "e1": e1,
                "is_last": is_last,
                "next_pol": next_pol,
            })
        })
        .collect();

    // ── Opening points (VerifyQuery xDivXSubXi) ─────────────────────────────
    let opening_points: Vec<serde_json::Value> = opening_points_json
        .iter()
        .enumerate()
        .map(|(i, op)| {
            let opening = op.as_i64().unwrap_or(0);
            let abs_opening = opening.unsigned_abs() as usize;
            // Build the chain of multiplications: w0=1, w{j+1}=GLMul(root/invroot, w{j})
            serde_json::json!({
                "idx": i,
                "opening": opening,
                "abs_opening": abs_opening,
                "is_nonzero": opening != 0,
                "is_positive": opening > 0,
            })
        })
        .collect();

    // ── UnrollCtx ──────────────────────────────────────────────────────────
    let unroll_ctx = UnrollCtx {
        q_stage: q_stage as u64,
        evals_stage: evals_stage as u64,
        fri_stage: fri_stage as u64,
        cm_pols_map: &cm_pols_map,
        custom_commits: &custom_commits_json,
        custom_commits_map: &custom_commits_map_json,
        boundaries: &boundaries_json,
    };

    // ── Eval P code (VerifyEvaluations inline) ──────────────────────────────
    let q_verifier_code: Vec<Value> = vi["qVerifier"]["code"].as_array().map_or(vec![], |a| a.clone());
    let mut eval_p_lines: Vec<String> = Vec::new();
    let eval_p_last = unroll_code_bn128(&q_verifier_code, &[], &unroll_ctx, &mut eval_p_lines)?;
    let eval_p_code = eval_p_lines.join("\n");

    // ── Eval Q code (VerifyQuery inline) ────────────────────────────────────
    let query_verifier_code: Vec<Value> = vi["queryVerifier"]["code"].as_array().map_or(vec![], |a| a.clone());
    let mut eval_q_lines: Vec<String> = Vec::new();
    let eval_q_last = unroll_code_bn128(&query_verifier_code, &[], &unroll_ctx, &mut eval_q_lines)?;
    let eval_q_code = eval_q_lines.join("\n");

    // ── Q polynomial ev_id ──────────────────────────────────────────────────
    let ev_map: Vec<Value> = si["evMap"].as_array().map_or(vec![], |a| a.clone());
    let q_index = cm_pols_map
        .iter()
        .position(|p| p["stage"].as_u64() == Some(q_stage as u64) && p["stageId"].as_u64() == Some(0))
        .unwrap_or(0);
    let q_pol_ev_id = ev_map
        .iter()
        .position(|e| e["type"].as_str() == Some("cm") && e["id"].as_u64() == Some(q_index as u64))
        .unwrap_or(0);

    // ── constRoot string ────────────────────────────────────────────────────
    // For BN128 the verkey is a single scalar — the caller passes it in slot 0
    // (with slots 1-3 as "0"), so we only emit the first limb. The template uses
    // `signal rootC <== {{ const_root_str }};` which expects a single value.
    let const_root_str = const_root.map(|r| r[0].clone()).unwrap_or_else(|| "0".to_string());

    // ── Transcript code strings ─────────────────────────────────────────────
    let mut t_fri = TranscriptBn128::new(transcript_arity, custom, Some("friQueries".into()));
    t_fri.put("challengeFRIQueries", 3);
    if pow_bits > 0 {
        t_fri.put_single("nonce");
    }
    t_fri.get_permutations("queriesFRI", n_queries, step0_bits, n_fields);
    let calculate_fri_queries_code = t_fri.get_code();

    let mut t = TranscriptBn128::new(transcript_arity, custom, None);
    let mut transcript_publics_code = String::new();
    let mut transcript_evals_code = String::new();
    let mut transcript_last_pol_fri_code = String::new();

    t.put_single("rootC");
    if n_publics > 0 {
        if !hash_commits {
            t.put("publics", n_publics);
        } else {
            let mut t_pub = TranscriptBn128::new(transcript_arity, custom, Some("publics".into()));
            t_pub.put("publics", n_publics);
            t_pub.get_field_hash("publicsHash");
            transcript_publics_code = t_pub.get_code();
            t.put_single("publicsHash");
        }
    }
    for stage in 1..=n_stages {
        let cnt = challenges_map.iter().filter(|c| c["stage"].as_u64() == Some(stage as u64)).count();
        for j in 0..cnt {
            t.get_field(&format!("challengesStage{stage}[{j}]"));
        }
        t.put_single(&format!("root{stage}"));
    }
    t.get_field("challengeQ");
    t.put_single(&format!("root{q_stage}"));
    t.get_field("challengeXi");

    if !hash_commits {
        for i in 0..ev_map_len {
            t.put(&format!("evals[{i}]"), 3);
        }
    } else {
        let mut t_evals = TranscriptBn128::new(transcript_arity, custom, Some("evals".into()));
        for i in 0..ev_map_len {
            t_evals.put(&format!("evals[{i}]"), 3);
        }
        t_evals.get_field_hash("evalsHash");
        transcript_evals_code = t_evals.get_code();
        t.put_single("evalsHash");
    }
    t.get_field("challengesFRI[0]");
    t.get_field("challengesFRI[1]");
    for si_idx in 0..n_steps {
        t.get_field(&format!("challengesFRISteps[{si_idx}]"));
        if si_idx < n_steps - 1 {
            t.put_single(&format!("s{}_root", si_idx + 1));
        } else if !hash_commits {
            for j in 0..final_pol_size {
                t.put(&format!("finalPol[{j}]"), 3);
            }
        } else {
            let mut t_fp = TranscriptBn128::new(transcript_arity, custom, Some("lastPolFRI".into()));
            for j in 0..final_pol_size {
                t_fp.put(&format!("finalPol[{j}]"), 3);
            }
            t_fp.get_field_hash("lastPolFRIHash");
            transcript_last_pol_fri_code = t_fp.get_code();
            t.put_single("lastPolFRIHash");
        }
    }
    // Final: challengeFRIQueries field
    t.get_field("challengeFRIQueries");
    let transcript_code = t.get_code();

    // ── query vals joined (VerifySingleQuery → VerifyQuery) ─────────────────
    let mut query_vals_list_gl: Vec<String> = Vec::new();
    for stage in 1..=n_stages {
        if sec_len(&format!("cm{stage}")) > 0 {
            query_vals_list_gl.push(format!("s0_vals{stage}GL"));
        }
    }
    query_vals_list_gl.push(format!("s0_vals{q_stage}GL"));
    query_vals_list_gl.push("s0_valsCGL".into());
    for cc in &custom_commits_json {
        let name = cc["name"].as_str().unwrap_or("");
        query_vals_list_gl.push(format!("s0_vals_{name}_0GL"));
    }
    let query_vals_gl_joined = query_vals_list_gl.join(", ");
    let next_step0_bits = if n_steps > 1 { steps[1]["nBits"].as_u64().unwrap_or(0) as usize } else { 0 };
    let next_vals_pol_0 = if n_steps > 1 { "s1_vals_p" } else { "finalPol" };

    // ── challengeNames (Transcript output signals) ───────────────────────────
    let mut challenge_names: Vec<String> = Vec::new();
    for stage in 1..=n_stages {
        if challenges_map.iter().any(|c| c["stage"].as_u64() == Some(stage as u64)) {
            challenge_names.push(format!("challengesStage{stage}GL"));
        }
    }
    challenge_names.extend(["challengeQGL", "challengeXiGL", "challengesFRIGL"].iter().map(|s| s.to_string()));
    let challenge_names_joined = challenge_names.join(",");

    // ── transcript_call_inputs ───────────────────────────────────────────────
    // The template appends `root{q_stage}` and `evalsGL` after this list, so
    // do NOT include them here (otherwise the call args duplicate).
    let mut transcript_call_inputs: Vec<String> = Vec::new();
    if n_publics > 0 {
        transcript_call_inputs.push("publicsGL".into());
    }
    transcript_call_inputs.push("rootC".into());
    for stage in 1..=n_stages {
        transcript_call_inputs.push(format!("root{stage}"));
    }

    // ── si_roots ────────────────────────────────────────────────────────────
    let si_roots: Vec<String> = (1..n_steps).map(|s| format!("s{s}_root")).collect();
    let si_roots_joined = si_roots.join(",");

    // ── verifyEvalsInputs ────────────────────────────────────────────────────
    let mut verify_evals_inputs: Vec<String> = Vec::new();
    for stage in 1..=n_stages {
        if challenges_map.iter().any(|c| c["stage"].as_u64() == Some(stage as u64)) {
            verify_evals_inputs.push(format!("challengesStage{stage}GL"));
        }
    }
    verify_evals_inputs.extend(["challengeQGL", "challengeXiGL", "evalsGL"].iter().map(|s| s.to_string()));
    if n_publics > 0 {
        verify_evals_inputs.push("publicsGL".into());
    }
    if n_air_group_values > 0 {
        verify_evals_inputs.push("airgroupValuesGL".into());
    }
    verify_evals_inputs.push("enabled".into());
    let verify_evals_inputs_joined = verify_evals_inputs.join(", ");

    // ── Main-template signal wiring ──────────────────────────────────────────
    let sha256_bits = 160 + 64 * n_publics;
    let q_stage_cm_section_len = sec_len(&format!("cm{q_stage}"));

    // ── Insert all context ───────────────────────────────────────────────────
    ctx.insert("custom", &custom);
    ctx.insert("skip_main", &opts.skip_main);
    ctx.insert("verkey_input", &opts.verkey_input);
    ctx.insert("enable_input", &opts.enable_input);
    ctx.insert("n_stages", &n_stages);
    ctx.insert("q_stage", &q_stage);
    ctx.insert("n_queries", &n_queries);
    ctx.insert("step0_bits", &step0_bits);
    ctx.insert("n_bits", &n_bits);
    ctx.insert("n_bits_ext", &n_bits_ext);
    ctx.insert("pow_bits", &pow_bits);
    ctx.insert("q_deg", &q_deg);
    ctx.insert("hash_commits", &hash_commits);
    ctx.insert("n_publics", &n_publics);
    ctx.insert("n_constants", &n_constants);
    ctx.insert("ev_map_len", &ev_map_len);
    ctx.insert("n_air_group_values", &n_air_group_values);
    ctx.insert("n_steps", &n_steps);
    ctx.insert("arity", &arity);
    ctx.insert("n_bits_arity", &n_bits_arity);
    ctx.insert("merkle_levels_s0", &merkle_levels_s0);
    ctx.insert("last_level_verification", &last_level_verification);
    ctx.insert("last_level_verification_gt0", &(last_level_verification > 0));
    ctx.insert("s0_last_mt_size", &s0_last_mt_size);
    ctx.insert("transcript_arity", &transcript_arity);
    ctx.insert("final_pol_size", &final_pol_size);
    ctx.insert("n_last_bits", &n_last_bits);
    ctx.insert("max_deg_bits", &max_deg_bits);
    ctx.insert("max_deg_size", &max_deg_size);
    ctx.insert("n_fields", &n_fields);
    ctx.insert("has_first_row", &has_first_row);
    ctx.insert("has_last_row", &has_last_row);
    ctx.insert("zlast_root", &zlast_root);
    ctx.insert("q_pol_ev_id", &q_pol_ev_id);
    ctx.insert("const_root_str", &const_root_str);
    ctx.insert("gl_shift", &GL_SHIFT.to_string());
    ctx.insert("sha256_bits", &sha256_bits);
    ctx.insert("q_stage_cm_section_len", &q_stage_cm_section_len);
    ctx.insert("next_step0_bits", &next_step0_bits);
    ctx.insert("next_vals_pol_0", &next_vals_pol_0);
    let step0_bits_size = if step0_bits == 0 { 1usize } else { 1 << step0_bits };
    ctx.insert("step0_bits_size", &step0_bits_size);
    ctx.insert("query_vals_gl_joined", &query_vals_gl_joined);
    ctx.insert("challenge_names_joined", &challenge_names_joined);
    ctx.insert("transcript_call_inputs", &transcript_call_inputs);
    ctx.insert("si_roots_joined", &si_roots_joined);
    ctx.insert("verify_evals_inputs_joined", &verify_evals_inputs_joined);
    // Code strings
    ctx.insert("calculate_fri_queries_code", &calculate_fri_queries_code);
    ctx.insert("transcript_code", &transcript_code);
    ctx.insert("transcript_publics_code", &transcript_publics_code);
    ctx.insert("transcript_evals_code", &transcript_evals_code);
    ctx.insert("transcript_last_pol_fri_code", &transcript_last_pol_fri_code);
    ctx.insert("eval_p_code", &eval_p_code);
    ctx.insert("eval_p_last", &eval_p_last);
    ctx.insert("eval_q_code", &eval_q_code);
    ctx.insert("eval_q_last", &eval_q_last);
    // Iterables
    ctx.insert("challenges_per_stage", &challenges_per_stage);
    ctx.insert("stages_info", &stages_info);
    ctx.insert("cm_pols_by_stage", &cm_pols_by_stage);
    ctx.insert("custom_commits", &custom_commits);
    ctx.insert("fri_steps_info", &fri_steps_info);
    ctx.insert("opening_points", &opening_points);
    ctx.insert("every_frames", &every_frames);

    Ok(ctx)
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn minimal_stark_info(n_stages: u64, n_queries: usize, step0_bits: usize) -> Value {
        json!({
            "starkStruct": {
                "verificationHashType": "BN128",
                "nQueries": n_queries,
                "nBits": 16,
                "nBitsExt": 17,
                "powBits": 0,
                "merkleTreeArity": 16,
                "merkleTreeCustom": false,
                "lastLevelVerification": 0,
                "hashCommits": false,
                "steps": [{"nBits": step0_bits}]
            },
            "nStages": n_stages,
            "nPublics": 0,
            "evMap": [],
            "cmPolsMap": [],
            "customCommits": [],
            "customCommitsMap": [],
            "challengesMap": [],
            "boundaries": [{"name": "everyRow"}],
            "airgroupValuesMap": [],
            "airValuesMap": [],
            "proofValuesMap": [],
            "mapSectionsN": {},
            "openingPoints": [],
            "nConstants": 0,
            "qDeg": 1,
        })
    }

    fn minimal_verifier_info() -> Value {
        json!({
            "qVerifier": { "code": [] },
            "queryVerifier": { "code": [] }
        })
    }

    #[test]
    fn header_non_custom() {
        let si = minimal_stark_info(2, 10, 8);
        let vi = minimal_verifier_info();
        let opts = Pil2CircomOptions::default();
        let out = gen_stark_verifier_bn128(None, &si, &vi, &opts).unwrap();
        assert!(out.contains("include \"poseidon.circom\";"), "out:\n{out}");
        assert!(!out.contains("custom/"), "no custom includes:\n{out}");
        assert!(out.contains("include \"merklehash.circom\";"), "out:\n{out}");
    }

    #[test]
    fn header_custom() {
        let mut si = minimal_stark_info(2, 10, 8);
        si["starkStruct"]["merkleTreeCustom"] = json!(true);
        let vi = minimal_verifier_info();
        let opts = Pil2CircomOptions::default();
        let out = gen_stark_verifier_bn128(None, &si, &vi, &opts).unwrap();
        assert!(out.contains("include \"custom/poseidon.circom\";"), "out:\n{out}");
        assert!(out.contains("include \"custom/merklehash.circom\";"), "out:\n{out}");
    }

    #[test]
    fn header_skip_main_omits_sha256() {
        let si = minimal_stark_info(2, 10, 8);
        let vi = minimal_verifier_info();
        let opts = Pil2CircomOptions { skip_main: true, ..Pil2CircomOptions::default() };
        let out = gen_stark_verifier_bn128(None, &si, &vi, &opts).unwrap();
        assert!(!out.contains("sha256"), "sha256 should be absent:\n{out}");
        assert!(!out.contains("bitify.circom"), "bitify should be absent:\n{out}");
    }

    #[test]
    fn header_not_skip_main_includes_sha256() {
        let si = minimal_stark_info(2, 10, 8);
        let vi = minimal_verifier_info();
        let opts = Pil2CircomOptions::default();
        let out = gen_stark_verifier_bn128(None, &si, &vi, &opts).unwrap();
        assert!(out.contains("sha256/sha256.circom"), "out:\n{out}");
    }

    #[test]
    fn gl_const_templates_present() {
        let si = minimal_stark_info(2, 10, 8);
        let vi = minimal_verifier_info();
        let opts = Pil2CircomOptions::default();
        let out = gen_stark_verifier_bn128(None, &si, &vi, &opts).unwrap();
        assert!(out.contains("template GLConst(num)"), "out:\n{out}");
        assert!(out.contains("template GLConst3(num)"), "out:\n{out}");
        assert!(out.contains("template GLC3()"), "out:\n{out}");
    }

    fn llv_stark_info() -> Value {
        let mut si = minimal_stark_info(2, 4, 10);
        si["starkStruct"]["merkleTreeArity"] = json!(4);
        si["starkStruct"]["lastLevelVerification"] = json!(2);
        si["starkStruct"]["steps"] = json!([{"nBits": 10}, {"nBits": 6}]);
        si["mapSectionsN"] = json!({"cm1": 2, "cm2": 2, "cm3": 2});
        si
    }

    #[test]
    fn llv_emits_until_level_and_root_checks() {
        let si = llv_stark_info();
        let vi = minimal_verifier_info();
        let opts = Pil2CircomOptions::default();
        let out = gen_stark_verifier_bn128(None, &si, &vi, &opts).unwrap();
        // truncated sibling depth: full = (10-1)/2+1 = 5, minus llv 2 -> 3
        assert!(out.contains("s0_siblingsC[3][4]"), "out:\n{out}");
        // last-level inputs: arity^llv = 16, flat single elements
        assert!(out.contains("s0_last_levelsC[16]"), "out:\n{out}");
        assert!(out.contains("s0_last_levels1[16]"), "out:\n{out}");
        assert!(out.contains("s1_last_levels[16]"), "out:\n{out}");
        assert!(out.contains("VerifyMerkleHashUntilLevel("), "out:\n{out}");
        // root checks: s0 trees height 1<<10, fri step 1 height 1<<6
        assert!(out.contains("VerifyMerkleRoot(2, 4, 1024)"), "out:\n{out}");
        assert!(out.contains("VerifyMerkleRoot(2, 4, 64)"), "out:\n{out}");
    }

    #[test]
    fn llv_zero_output_has_no_last_levels() {
        let mut si = llv_stark_info();
        si["starkStruct"]["lastLevelVerification"] = json!(0);
        let vi = minimal_verifier_info();
        let opts = Pil2CircomOptions::default();
        let out = gen_stark_verifier_bn128(None, &si, &vi, &opts).unwrap();
        assert!(!out.contains("last_levels"), "out:\n{out}");
        assert!(!out.contains("UntilLevel"), "out:\n{out}");
        // full sibling depth restored
        assert!(out.contains("s0_siblingsC[5][4]"), "out:\n{out}");
    }

    #[test]
    fn llv_with_custom_bails() {
        let mut si = llv_stark_info();
        si["starkStruct"]["merkleTreeCustom"] = json!(true);
        let vi = minimal_verifier_info();
        let opts = Pil2CircomOptions::default();
        let result = gen_stark_verifier_bn128(None, &si, &vi, &opts);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("lastLevelVerification"));
    }

    #[test]
    fn wrong_hash_type_returns_error() {
        let mut si = minimal_stark_info(2, 10, 8);
        si["starkStruct"]["verificationHashType"] = json!("GL");
        let vi = minimal_verifier_info();
        let opts = Pil2CircomOptions::default();
        let result = gen_stark_verifier_bn128(None, &si, &vi, &opts);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("GL"));
    }

    #[test]
    fn component_main_emits_main_wrapper() {
        let si = minimal_stark_info(2, 10, 8);
        let vi = minimal_verifier_info();
        let opts = Pil2CircomOptions::default();
        let out = gen_stark_verifier_bn128(None, &si, &vi, &opts).unwrap();
        assert!(out.contains("component main = Main();"), "out:\n{out}");
    }
}
