//! Rust port of `pil2circom.js` — generates a GL `stark_verifier.circom` from
//! `starkInfo` + `verifierInfo` + options, entirely in-process.
//!
//! Architecture
//! ============
//! All "hard" computation (GL field arithmetic, Poseidon1 transcript state-machine,
//! expression chunking and unrolling) is done in Rust.  The results are collected
//! into a flat, serialisable [`TeraCtx`] struct which is passed to a Tera template
//! (`circuits.gl/stark_verifier.circom.tera`) that handles all structural loops,
//! conditionals and signal declarations.

use anyhow::{bail, Result};
use serde_json::Value;
use tera::Tera;

use super::expressions_chunks::get_expressions_chunks;
use super::gl_field::{gl_exp, gl_inv, GL_INV_W, GL_SHIFT, GL_W};
use crate::stark2circom::transcript::Transcript;
use super::unroll_code::{unroll_code, UnrollCtx};

// ── Public API re-exported from pil2-stark ────────────────────────────────────

#[derive(Debug, Clone)]
pub struct Pil2CircomOptions {
    pub skip_main: bool,
    pub verkey_input: bool,
    pub enable_input: bool,
    pub input_challenges: bool,
    pub fri_queries_batch_size: Option<usize>,
    pub multi_fri: bool,
    pub hash: String,
}

impl Default for Pil2CircomOptions {
    fn default() -> Self {
        Self {
            skip_main: false,
            verkey_input: false,
            enable_input: false,
            input_challenges: false,
            fri_queries_batch_size: None,
            multi_fri: false,
            hash: proofman_common::hash_family::DEFAULT_HASH_ID.to_string(),
        }
    }
}

// ── Top-level entry-point ─────────────────────────────────────────────────────

/// Embedded Tera template (GL stark verifier).
const TEMPLATE_GL: &str = include_str!("tera/stark_verifier_gl.circom.tera");

/// Generate a GL `stark_verifier.circom` entirely in Rust.
///
/// Equivalent to calling `node main_pil2circom.js -s starkinfo.json -i
/// verifierinfo.json [-v verkey.json] [--skipMain] [--verkeyInput]
/// [--enableInput] [--inputChallenges]`.
pub fn gen_stark_verifier_gl(
    const_root: Option<&[String; 4]>,
    stark_info: &Value,
    verifier_info: &Value,
    opts: &Pil2CircomOptions,
) -> Result<String> {
    let hash_type = stark_info["starkStruct"]["verificationHashType"].as_str().unwrap_or("GL");
    if hash_type != "GL" {
        bail!("gen_stark_verifier_gl: expected GL hash type, got '{hash_type}'");
    }

    let tera_ctx = build_tera_context(stark_info, verifier_info, opts, const_root)?;
    Tera::one_off(TEMPLATE_GL, &tera_ctx, false).map_err(|e| anyhow::anyhow!("Tera render error: {e}"))
}

// ── Tera context builder ──────────────────────────────────────────────────────

fn build_tera_context(
    si: &Value,
    vi: &Value,
    opts: &Pil2CircomOptions,
    const_root: Option<&[String; 4]>,
) -> Result<tera::Context> {
    use super::gl_field::gl_mul;
    let mut ctx = tera::Context::new();

    // ── Shorthand refs ───────────────────────────────────────────────────────
    let ss = &si["starkStruct"];
    let n_stages = si["nStages"].as_u64().unwrap_or(0) as usize;
    let q_stage = n_stages + 1;
    let arity = ss["merkleTreeArity"].as_u64().unwrap_or(16) as usize;
    let n_queries = ss["nQueries"].as_u64().unwrap_or(0) as usize;
    let n_bits = ss["nBits"].as_u64().unwrap_or(0) as usize;
    let n_bits_ext = ss["nBitsExt"].as_u64().unwrap_or(0) as usize;
    let pow_bits = ss["powBits"].as_u64().unwrap_or(0);
    let q_deg = si["qDeg"].as_u64().unwrap_or(1) as usize;
    let split_linear_hash = ss["splitLinearHash"].as_bool().unwrap_or(false);
    if split_linear_hash && opts.hash == "Poseidon1" {
        bail!(
            "gen_stark_verifier_gl: Poseidon1 + splitLinearHash is not supported \
             (no hash/poseidon1/merklehash_gpu.circom / linearhash_gpu.circom exist)"
        );
    }
    let hash_commits = ss["hashCommits"].as_bool().unwrap_or(false);
    let last_level_verification = ss["lastLevelVerification"].as_u64().unwrap_or(0);
    let multi_fri = opts.multi_fri;
    let steps: Vec<Value> = ss["steps"].as_array().map_or(vec![], |a| a.clone());
    let n_steps = steps.len();
    let step0_bits = steps.first().and_then(|s| s["nBits"].as_u64()).unwrap_or(0) as usize;
    let last_step_bits = steps.last().and_then(|s| s["nBits"].as_u64()).unwrap_or(0);
    let final_pol_size: u64 = 1 << last_step_bits;
    let n_last_bits = last_step_bits;
    let max_deg_bits = n_last_bits.saturating_sub((n_bits_ext - n_bits) as u64);
    let n_publics = si["nPublics"].as_u64().unwrap_or(0);
    let n_constants = si["nConstants"].as_u64().unwrap_or(0);
    let ev_map_len = si["evMap"].as_array().map_or(0, |a| a.len());
    let n_air_group_values = si["airgroupValuesMap"].as_array().map_or(0, |a| a.len());
    let n_air_values = si["airValuesMap"].as_array().map_or(0, |a| a.len());
    let n_proof_values = si["proofValuesMap"].as_array().map_or(0, |a| a.len());
    let challenges_map: Vec<Value> = si["challengesMap"].as_array().map_or(vec![], |a| a.clone());
    let air_values_map: Vec<Value> = si["airValuesMap"].as_array().map_or(vec![], |a| a.clone());
    let cm_pols_map: Vec<Value> = si["cmPolsMap"].as_array().map_or(vec![], |a| a.clone());
    let custom_commits_json: Vec<Value> = si["customCommits"].as_array().map_or(vec![], |a| a.clone());
    let custom_commits_map_json: Vec<Value> = si["customCommitsMap"].as_array().map_or(vec![], |a| a.clone());
    let opening_points_json: Vec<Value> = si["openingPoints"].as_array().map_or(vec![], |a| a.clone());
    let boundaries_json: Vec<Value> = si["boundaries"].as_array().map_or(vec![], |a| a.clone());
    let map_sections_n = &si["mapSectionsN"];

    // ── Helper: section size from mapSectionsN ────────────────────────────────
    let sec_len = |key: &str| -> u64 { map_sections_n[key].as_u64().unwrap_or(0) };

    // ── Template name suffix ──────────────────────────────────────────────────
    let id_suffix = match si.get("airgroupId").and_then(|v| v.as_u64()) {
        Some(aid) => aid.to_string(),
        None => String::new(),
    };
    let mk = |base: &str| -> String {
        if id_suffix.is_empty() {
            base.to_string()
        } else {
            format!("{base}{id_suffix}")
        }
    };

    // ── Merkle-tree dimensions ────────────────────────────────────────────────
    let log2_arity = (arity as f64).log2();
    let n_bits_arity = log2_arity.ceil() as u64;
    let siblings_width = ((arity - 1) * 4) as u64;
    let merkle_levels_s0 = (step0_bits as f64 / log2_arity).ceil() as u64;
    let n_bits_ext_size: u64 = 1 << n_bits_ext;

    // ── Batch sizing ──────────────────────────────────────────────────────────
    let batch_size = opts.fri_queries_batch_size.unwrap_or_else(|| n_queries.div_ceil(8).max(1));
    let batch_size = batch_size.min(n_queries).max(1);
    let full_batch_count = n_queries / batch_size;
    let last_batch_size = n_queries % batch_size;
    let last_batch_size_safe = last_batch_size.max(1);

    // ── n_fields ─────────────────────────────────────────────────────────────
    let total_bits = n_queries * step0_bits;
    let n_fields = if total_bits == 0 { 0 } else { (total_bits - 1) / 63 + 1 };

    // ── Per-stage challenge counts ────────────────────────────────────────────
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

    // ── Stages info ───────────────────────────────────────────────────────────
    let stages_info: Vec<serde_json::Value> = (1..=q_stage)
        .map(|stage| {
            let challenges_count = challenges_map.iter().filter(|c| c["stage"].as_u64() == Some(stage as u64)).count();
            let cm_section_len = sec_len(&format!("cm{stage}"));
            let air_values_in_stage: Vec<usize> = air_values_map
                .iter()
                .enumerate()
                .filter(|(_, av)| av["stage"].as_u64() == Some(stage as u64))
                .map(|(j, _)| j)
                .collect();
            serde_json::json!({
                "stage": stage,
                "challenges_count": challenges_count,
                "cm_section_len": cm_section_len,
                "has_cm": cm_section_len > 0,
                "air_values_in_stage": air_values_in_stage,
            })
        })
        .collect();

    // ── cm_pols_by_stage — for MapValues ─────────────────────────────────────
    let cm_pols_by_stage: Vec<serde_json::Value> = (1..=q_stage)
        .map(|stage| {
            let pols: Vec<serde_json::Value> = cm_pols_map
                .iter()
                .filter(|p| p["stage"].as_u64() == Some(stage as u64))
                .enumerate()
                .map(|(i, p)| {
                    serde_json::json!({
                        "idx": i,
                        "dim": p["dim"].as_u64().unwrap_or(1),
                        "stage_pos": p["stagePos"].as_u64().unwrap_or(0),
                    })
                })
                .collect();
            serde_json::json!({ "stage": stage, "pols": pols })
        })
        .collect();

    // ── custom_commits ────────────────────────────────────────────────────────
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
                        .enumerate()
                        .map(|(i, p)| {
                            serde_json::json!({
                                "idx": i,
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

    // ── FRI steps info ────────────────────────────────────────────────────────
    let fri_steps_info: Vec<serde_json::Value> = (0..n_steps)
        .map(|s| {
            let n_bits_s = steps[s]["nBits"].as_u64().unwrap_or(0);
            let prev_bits = if s == 0 { n_bits_s } else { steps[s - 1]["nBits"].as_u64().unwrap_or(0) };
            let next_bits = if s < n_steps - 1 { steps[s + 1]["nBits"].as_u64().unwrap_or(0) } else { 0 };
            let exponent = if s == 0 { 1u64 } else { 1u64 << (prev_bits - n_bits_s) };
            let full_ml = (n_bits_s as f64 / log2_arity).ceil() as u64;
            let ml = full_ml.saturating_sub(last_level_verification);
            let last_mt_size =
                if last_level_verification > 0 { (arity as u64).pow(last_level_verification as u32) } else { 0 };
            let e0 = if s == 0 {
                "0".to_string()
            } else {
                let exp = 1u64 << (n_bits_ext as u64 - prev_bits);
                gl_inv(gl_exp(GL_SHIFT, exp)).to_string()
            };
            let val_size = 1u64 << prev_bits.saturating_sub(n_bits_s);
            let mt_size = 1u64 << n_bits_s;
            serde_json::json!({
                "s": s,
                "n_bits": n_bits_s,
                "prev_bits": prev_bits,
                "next_bits": next_bits,
                "exponent": exponent,
                "merkle_levels": ml,
                "full_merkle_levels": full_ml,
                "val_size": val_size,
                "last_mt_size": last_mt_size,
                "mt_size": mt_size,
                "e0": e0,
                "is_last": s == n_steps - 1,
            })
        })
        .collect();

    // ── Boundary flags ────────────────────────────────────────────────────────
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

    // ── everyFrame boundaries ─────────────────────────────────────────────────
    let every_frames: Vec<serde_json::Value> = boundaries_json
        .iter()
        .filter(|b| b["name"].as_str() == Some("everyFrame"))
        .enumerate()
        .map(|(i, frame)| {
            let offset_min = frame["offsetMin"].as_u64().unwrap_or(0);
            let offset_max = frame["offsetMax"].as_u64().unwrap_or(0);
            let size = offset_min + offset_max;
            let mut ops: Vec<serde_json::Value> = Vec::new();
            let mut c = 0u64;
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
            let back_exp = (1u64 << n_bits).saturating_sub(i as u64 + 1);
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

    // ── Opening points ────────────────────────────────────────────────────────
    let opening_points: Vec<serde_json::Value> = opening_points_json
        .iter()
        .enumerate()
        .map(|(i, op)| {
            let opening = op.as_i64().unwrap_or(0);
            let abs_opening = opening.unsigned_abs();
            let mut root: u64 = 1;
            for _ in 0..abs_opening {
                root = gl_mul(root, if opening > 0 { GL_W[n_bits] } else { GL_INV_W[n_bits] });
            }
            serde_json::json!({ "idx": i, "opening": opening, "root": root.to_string() })
        })
        .collect();

    // ── UnrollCtx ────────────────────────────────────────────────────────────
    let unroll_ctx = UnrollCtx {
        q_stage: q_stage as u64,
        evals_stage: (q_stage + 1) as u64,
        fri_stage: (q_stage + 2) as u64,
        cm_pols_map: &cm_pols_map,
        custom_commits: &custom_commits_json,
        custom_commits_map: &custom_commits_map_json,
        boundaries: &boundaries_json,
    };

    // Helper: build chunk JSON list from ChunkedCode
    let build_chunk_list = |cc: &super::expressions_chunks::ChunkedCode| -> Vec<serde_json::Value> {
        cc.chunks
            .iter()
            .enumerate()
            .map(|(i, chunk)| {
                let mut lines: Vec<String> = Vec::new();
                let initialized: Vec<u64> = chunk.inputs.iter().chain(chunk.outputs.iter()).copied().collect();
                let _ = unroll_code(&chunk.code, &initialized, &unroll_ctx, &mut lines);
                let code = lines.join("\n");
                let inputs: Vec<serde_json::Value> =
                    chunk.inputs.iter().map(|&id| serde_json::json!({ "id": id, "dim": cc.tmps[&id].dim })).collect();
                let outputs: Vec<serde_json::Value> =
                    chunk.outputs.iter().map(|&id| serde_json::json!({ "id": id, "dim": cc.tmps[&id].dim })).collect();
                serde_json::json!({ "idx": i, "inputs": inputs, "outputs": outputs, "code": code })
            })
            .collect()
    };

    // ── Eval P chunks ─────────────────────────────────────────────────────────
    let q_verifier_code: Vec<Value> = vi["qVerifier"]["code"].as_array().map_or(vec![], |a| a.clone());
    let eval_p_raw = get_expressions_chunks(&q_verifier_code);
    let mut eval_p_chunks = build_chunk_list(&eval_p_raw);

    // ── Eval Q chunks ─────────────────────────────────────────────────────────
    let query_verifier_code: Vec<Value> = vi["queryVerifier"]["code"].as_array().map_or(vec![], |a| a.clone());
    let eval_q_raw = get_expressions_chunks(&query_verifier_code);
    let mut eval_q_chunks = build_chunk_list(&eval_q_raw);

    // ── inputsP — for VerifyEvaluations chunk calls ───────────────────────────
    let mut inputs_p: Vec<String> = Vec::new();
    for stage in 1..=n_stages {
        if challenges_map.iter().any(|c| c["stage"].as_u64() == Some(stage as u64)) {
            inputs_p.push(format!("challengesStage{stage}"));
        }
    }
    inputs_p.push("challengeQ".into());
    inputs_p.push("challengeXi".into());
    inputs_p.push("evals".into());
    if n_publics > 0 {
        inputs_p.push("publics".into());
    }
    if n_air_group_values > 0 {
        inputs_p.push("airgroupvalues".into());
    }
    if n_air_values > 0 {
        inputs_p.push("airvalues".into());
    }
    if n_proof_values > 0 {
        inputs_p.push("proofvalues".into());
    }
    inputs_p.push("Zh".into());
    if has_first_row {
        inputs_p.push("Zfirst".into());
    }
    if has_last_row {
        inputs_p.push("Zlast".into());
    }
    for fi in 0..every_frames.len() {
        inputs_p.push(format!("Zframe{fi}"));
    }

    // ── inputsQ — for CalculateFRIPolChunks calls ─────────────────────────────
    let mut inputs_q: Vec<String> = vec!["challengesFRI".into(), "evals".into()];
    for stage in 1..=n_stages {
        if sec_len(&format!("cm{stage}")) > 0 {
            inputs_q.push(format!("cm{stage}"));
        }
    }
    inputs_q.push(format!("cm{q_stage}"));
    inputs_q.push("consts".into());
    for cc in &custom_commits_json {
        let name = cc["name"].as_str().unwrap_or("");
        inputs_q.push(format!("custom_{name}_0"));
    }
    inputs_q.push("xDivXSubXi".into());

    // Annotate chunks with call-site strings
    for (i, chunk) in eval_p_chunks.iter_mut().enumerate() {
        let raw = &eval_p_raw.chunks[i];
        let mut call_ins = inputs_p.clone();
        call_ins.extend(raw.inputs.iter().map(|id| format!("tmp_{id}")));
        let call_outs: Vec<String> = raw.outputs.iter().map(|id| format!("tmp_{id}")).collect();
        chunk["call_inputs"] = serde_json::Value::String(call_ins.join(","));
        chunk["call_outputs"] = serde_json::Value::String(call_outs.join(","));
    }
    for (i, chunk) in eval_q_chunks.iter_mut().enumerate() {
        let raw = &eval_q_raw.chunks[i];
        let mut call_ins = inputs_q.clone();
        call_ins.extend(raw.inputs.iter().map(|id| format!("tmp_{id}")));
        let call_outs: Vec<String> = raw.outputs.iter().map(|id| format!("tmp_{id}")).collect();
        chunk["call_inputs"] = serde_json::Value::String(call_ins.join(","));
        chunk["call_outputs"] = serde_json::Value::String(call_outs.join(","));
    }

    // ── Q polynomial ev_id ────────────────────────────────────────────────────
    let ev_map: Vec<Value> = si["evMap"].as_array().map_or(vec![], |a| a.clone());
    let q_index = cm_pols_map
        .iter()
        .position(|p| p["stage"].as_u64() == Some(q_stage as u64) && p["stageId"].as_u64() == Some(0))
        .unwrap_or(0);
    let q_pol_ev_id = ev_map
        .iter()
        .position(|e| e["type"].as_str() == Some("cm") && e["id"].as_u64() == Some(q_index as u64))
        .unwrap_or(0);
    let last_dest_id_p =
        vi["qVerifier"]["code"].as_array().and_then(|a| a.last()).and_then(|i| i["dest"]["id"].as_u64()).unwrap_or(0);
    let last_dest_id_q = vi["queryVerifier"]["code"]
        .as_array()
        .and_then(|a| a.last())
        .and_then(|i| i["dest"]["id"].as_u64())
        .unwrap_or(0);

    // ── constRoot string ──────────────────────────────────────────────────────
    let const_root_str = const_root.map(|r| r.join(",")).unwrap_or_else(|| "0,0,0,0".to_string());

    // ── Transcript code strings ───────────────────────────────────────────────
    // Select the transcript hash family: a Poseidon2 proof verifies with the
    // Poseidon2 sponge, otherwise (Poseidon1 recursion) with the Poseidon1 sponge.
    let transcript_poseidon2 = opts.hash == "Poseidon2";
    let mut t_fri = Transcript::new(arity, Some("friQueries".into()));
    t_fri.set_drain_in_update_state(true);
    t_fri.set_poseidon2(transcript_poseidon2);
    t_fri.put("challengeFRIQueries", 3);
    if pow_bits > 0 {
        t_fri.put_single("nonce");
    }
    t_fri.get_permutations("queriesFRI", n_queries, step0_bits);
    let calculate_fri_queries_code = t_fri.get_code();

    let mut t = Transcript::new(arity, None);
    t.set_drain_in_update_state(true);
    t.set_poseidon2(transcript_poseidon2);
    let mut transcript_publics_code = String::new();
    let mut transcript_evals_code = String::new();
    let mut transcript_last_pol_fri_code = String::new();

    if !opts.input_challenges {
        t.put("rootC", 4);
        if n_publics > 0 {
            if !hash_commits {
                t.put("publics", n_publics as usize);
            } else {
                let mut t_pub = Transcript::new(arity, Some("publics".into()));
                t_pub.set_drain_in_update_state(true);
                t_pub.set_poseidon2(transcript_poseidon2);
                t_pub.put("publics", n_publics as usize);
                t_pub.get_state("publicsHash");
                transcript_publics_code = t_pub.get_code();
                t.put("publicsHash", 4);
            }
        }
        t.put("root1", 4);
    } else {
        t.put("globalChallenge", 3);
    }

    for i in 1..n_stages {
        let stage = (i + 1) as u64;
        let cnt = challenges_map.iter().filter(|c| c["stage"].as_u64() == Some(stage)).count();
        for j in 0..cnt {
            t.get_field(&format!("challengesStage{stage}[{j}]"));
        }
        t.put(&format!("root{stage}"), 4);
        for (j, av) in air_values_map.iter().enumerate() {
            if av["stage"].as_u64() == Some(stage) {
                t.put(&format!("airValues[{j}]"), 3);
            }
        }
    }

    t.get_field("challengeQ");
    t.put(&format!("root{q_stage}"), 4);
    t.get_field("challengeXi");

    // In hash_commits mode the EJS calls transcript.getCode() HERE (split 1) before
    // emitting the side evals transcript and continuing with FRI challenges.
    // Match that by capturing the stage code now and injecting `transcript_evals_code`
    // between stage and FRI in the template.
    let transcript_code_stage: String;
    if hash_commits {
        transcript_code_stage = t.get_code();
        let mut t_evals = Transcript::new(arity, Some("evals".into()));
        t_evals.set_drain_in_update_state(true);
        t_evals.set_poseidon2(transcript_poseidon2);
        for i in 0..ev_map_len {
            t_evals.put(&format!("evals[{i}]"), 3);
        }
        t_evals.get_state("evalsHash");
        transcript_evals_code = t_evals.get_code();
        t.put("evalsHash", 4);
    } else {
        // !hash_commits: evals go straight into the main transcript (no side transcript).
        for i in 0..ev_map_len {
            t.put(&format!("evals[{i}]"), 3);
        }
        transcript_code_stage = String::new(); // all code collected at the end
    }

    t.get_field("challengesFRI[0]");
    t.get_field("challengesFRI[1]");

    // transcript_code_fri_mid: FRI challenges up to (but not including) the final
    // challengesFRISteps[n_steps].  In hash_commits mode we split again just before
    // the lastPolFRI side transcript so that it appears before the drain+final-hash.
    let mut transcript_code_fri_mid = String::new();
    for si_idx in 0..n_steps {
        t.get_field(&format!("challengesFRISteps[{si_idx}]"));
        if si_idx < n_steps - 1 {
            t.put(&format!("s{}_root", si_idx + 1), 4);
        } else if !hash_commits {
            for j in 0..final_pol_size {
                t.put(&format!("finalPol[{j}]"), 3);
            }
        } else {
            // Split 2: capture FRI code up to challengesFRISteps[n_steps-1].
            transcript_code_fri_mid = t.get_code();
            let mut t_fp = Transcript::new(arity, Some("lastPolFRI".into()));
            t_fp.set_drain_in_update_state(true);
            t_fp.set_poseidon2(transcript_poseidon2);
            for j in 0..final_pol_size {
                t_fp.put(&format!("finalPol[{j}]"), 3);
            }
            t_fp.get_state("lastPolFRIHash");
            transcript_last_pol_fri_code = t_fp.get_code();
            t.put("lastPolFRIHash", 4);
        }
    }
    t.get_field(&format!("challengesFRISteps[{n_steps}]"));

    // In hash_commits mode: transcript_code is now only the final drain +
    // transcriptHash_N + challengesFRISteps[n_steps] (captured after split 2).
    // In !hash_commits mode: transcript_code holds everything (stage_code is empty).
    let transcript_code = t.get_code();

    // ── queryVals_joined ──────────────────────────────────────────────────────
    let mut query_vals_list: Vec<String> = Vec::new();
    for stage in 1..=n_stages {
        if sec_len(&format!("cm{stage}")) > 0 {
            query_vals_list.push(format!("s0_vals{stage}"));
        }
    }
    query_vals_list.push(format!("s0_vals{q_stage}"));
    query_vals_list.push("s0_valsC".into());
    for cc in &custom_commits_json {
        let name = cc["name"].as_str().unwrap_or("");
        query_vals_list.push(format!("s0_vals_{name}_0"));
    }
    let query_vals_joined = query_vals_list.join(", ");

    // ── challengeNames_joined for StarkVerifier transcript call ───────────────
    let mut challenge_names: Vec<String> = Vec::new();
    for stage in 1..=n_stages {
        if challenges_map.iter().any(|c| c["stage"].as_u64() == Some(stage as u64)) {
            challenge_names.push(format!("challengesStage{stage}"));
        }
    }
    challenge_names.extend(["challengeQ", "challengeXi", "challengesFRI"].iter().map(|s| s.to_string()));
    let challenge_names_joined = challenge_names.join(",");

    // ── si_roots_joined ───────────────────────────────────────────────────────
    let si_roots: Vec<String> = (1..n_steps).map(|s| format!("s{s}_root")).collect();
    let si_roots_joined = si_roots.join(",");

    // ── transcript_call_inputs ────────────────────────────────────────────────
    let mut transcript_call_inputs: Vec<String> = Vec::new();
    if !opts.input_challenges {
        if n_publics > 0 {
            transcript_call_inputs.push("publics".into());
        }
        transcript_call_inputs.push("rootC".into());
        transcript_call_inputs.push("root1".into());
    } else {
        transcript_call_inputs.push("globalChallenge".into());
    }
    if n_air_values > 0 {
        transcript_call_inputs.push("airvalues".into());
    }
    for stage in 2..=n_stages {
        transcript_call_inputs.push(format!("root{stage}"));
    }

    // ── verify_evals_inputs_joined ────────────────────────────────────────────
    let mut verify_evals_inputs: Vec<String> = Vec::new();
    for stage in 1..=n_stages {
        if challenges_map.iter().any(|c| c["stage"].as_u64() == Some(stage as u64)) {
            verify_evals_inputs.push(format!("challengesStage{stage}"));
        }
    }
    verify_evals_inputs.extend(["challengeQ", "challengeXi", "evals"].iter().map(|s| s.to_string()));
    if n_publics > 0 {
        verify_evals_inputs.push("publics".into());
    }
    if n_air_group_values > 0 {
        verify_evals_inputs.push("airgroupvalues".into());
    }
    if n_air_values > 0 {
        verify_evals_inputs.push("airvalues".into());
    }
    if n_proof_values > 0 {
        verify_evals_inputs.push("proofvalues".into());
    }
    verify_evals_inputs.push("enabled".into());
    let verify_evals_inputs_joined = verify_evals_inputs.join(", ");

    // ── next step bits (VerifyQuery call) ─────────────────────────────────────
    let next_step0_bits = if n_steps > 1 { steps[1]["nBits"].as_u64().unwrap_or(0) } else { 0 };
    let next_vals_pol_0 = if n_steps > 1 { "s1_vals_p" } else { "finalPol" };

    // ── Insert all context values ─────────────────────────────────────────────
    ctx.insert("split_linear_hash", &split_linear_hash);
    ctx.insert("n_stages", &n_stages);
    ctx.insert("q_stage", &q_stage);
    ctx.insert("n_queries", &n_queries);
    ctx.insert("step0_bits", &step0_bits);
    ctx.insert("n_bits", &n_bits);
    ctx.insert("n_bits_ext", &n_bits_ext);
    ctx.insert("n_bits_ext_size", &n_bits_ext_size);
    ctx.insert("pow_bits", &pow_bits);
    ctx.insert("q_deg", &q_deg);
    ctx.insert("hash_commits", &hash_commits);
    ctx.insert("last_level_verification", &last_level_verification);
    ctx.insert("last_level_verification_gt0", &(last_level_verification > 0));
    ctx.insert("multi_fri", &multi_fri);
    ctx.insert("skip_main", &opts.skip_main);
    ctx.insert("verkey_input", &opts.verkey_input);
    ctx.insert("enable_input", &opts.enable_input);
    ctx.insert("input_challenges", &opts.input_challenges);
    ctx.insert("hash", &opts.hash);
    ctx.insert("n_publics", &n_publics);
    ctx.insert("n_constants", &n_constants);
    ctx.insert("ev_map_len", &ev_map_len);
    ctx.insert("n_air_group_values", &n_air_group_values);
    ctx.insert("n_air_values", &n_air_values);
    ctx.insert("n_proof_values", &n_proof_values);
    ctx.insert("final_pol_size", &final_pol_size);
    ctx.insert("n_last_bits", &n_last_bits);
    ctx.insert("max_deg_bits", &max_deg_bits);
    let max_deg_size: u64 = 1u64 << max_deg_bits;
    ctx.insert("max_deg_size", &max_deg_size);
    ctx.insert("n_steps", &n_steps);
    ctx.insert("arity", &arity);
    ctx.insert("n_bits_arity", &n_bits_arity);
    ctx.insert("siblings_width", &siblings_width);
    ctx.insert("merkle_levels_s0", &merkle_levels_s0);
    ctx.insert("gl_shift", &GL_SHIFT.to_string());
    ctx.insert("const_root_str", &const_root_str);
    ctx.insert("batch_size", &batch_size);
    ctx.insert("full_batch_count", &full_batch_count);
    ctx.insert("last_batch_size", &last_batch_size);
    ctx.insert("last_batch_size_safe", &last_batch_size_safe);
    ctx.insert("n_fields", &n_fields);
    ctx.insert("has_first_row", &has_first_row);
    ctx.insert("has_last_row", &has_last_row);
    ctx.insert("zlast_root", &zlast_root);
    ctx.insert("q_pol_ev_id", &q_pol_ev_id);
    ctx.insert("last_dest_id_p", &last_dest_id_p);
    ctx.insert("last_dest_id_q", &last_dest_id_q);
    ctx.insert("next_step0_bits", &next_step0_bits);
    ctx.insert("next_vals_pol_0", &next_vals_pol_0);
    let q_stage_cm_section_len: u64 = sec_len(&format!("cm{q_stage}"));
    ctx.insert("q_stage_cm_section_len", &q_stage_cm_section_len);
    let s0_last_mt_size: u64 = fri_steps_info.first().and_then(|s| s["last_mt_size"].as_u64()).unwrap_or(0);
    ctx.insert("s0_last_mt_size", &s0_last_mt_size);
    ctx.insert("query_vals_joined", &query_vals_joined);
    ctx.insert("challenge_names_joined", &challenge_names_joined);
    ctx.insert("si_roots_joined", &si_roots_joined);
    ctx.insert("transcript_call_inputs", &transcript_call_inputs);
    ctx.insert("verify_evals_inputs_joined", &verify_evals_inputs_joined);
    // Template names
    ctx.insert("calculate_fri_queries_name", &mk("calculateFRIQueries"));
    ctx.insert("transcript_name", &mk("Transcript"));
    ctx.insert("verify_fri_name", &mk("VerifyFRI"));
    ctx.insert("verify_evaluations_name", &mk("VerifyEvaluations"));
    ctx.insert("map_values_name", &mk("MapValues"));
    ctx.insert("verify_query_name", &mk("VerifyQuery"));
    ctx.insert("verify_final_pol_name", &mk("VerifyFinalPol"));
    ctx.insert("verify_single_query_name", &mk("VerifySingleQuery"));
    ctx.insert("verify_queries_batch_name", &mk("VerifyQueriesBatch"));
    ctx.insert("verifier_name", &mk("StarkVerifier"));
    // Code strings
    ctx.insert("calculate_fri_queries_code", &calculate_fri_queries_code);
    ctx.insert("transcript_code", &transcript_code);
    ctx.insert("transcript_code_stage", &transcript_code_stage);
    ctx.insert("transcript_code_fri_mid", &transcript_code_fri_mid);
    ctx.insert("transcript_publics_code", &transcript_publics_code);
    ctx.insert("transcript_evals_code", &transcript_evals_code);
    ctx.insert("transcript_last_pol_fri_code", &transcript_last_pol_fri_code);
    // Iterables
    ctx.insert("challenges_per_stage", &challenges_per_stage);
    ctx.insert("stages_info", &stages_info);
    ctx.insert("cm_pols_by_stage", &cm_pols_by_stage);
    ctx.insert("custom_commits", &custom_commits);
    ctx.insert("eval_p_chunks", &eval_p_chunks);
    ctx.insert("eval_q_chunks", &eval_q_chunks);
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
                "verificationHashType": "GL",
                "nQueries": n_queries,
                "nBits": 16,
                "nBitsExt": 17,
                "powBits": 0,
                "merkleTreeArity": 4,
                "lastLevelVerification": 0,
                "splitLinearHash": false,
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
    fn names_without_airgroup_id() {
        let si = minimal_stark_info(2, 10, 8);
        let vi = minimal_verifier_info();
        let opts = Pil2CircomOptions::default();
        let out = gen_stark_verifier_gl(None, &si, &vi, &opts).unwrap();
        assert!(out.contains("template StarkVerifier("), "got:\n{out}");
        assert!(out.contains("template Transcript("), "got:\n{out}");
        assert!(out.contains("template calculateFRIQueries("), "got:\n{out}");
    }

    #[test]
    fn names_with_airgroup_id() {
        let mut si = minimal_stark_info(2, 10, 8);
        si["airgroupId"] = json!(42u64);
        let vi = minimal_verifier_info();
        let opts = Pil2CircomOptions::default();
        let out = gen_stark_verifier_gl(None, &si, &vi, &opts).unwrap();
        assert!(out.contains("template StarkVerifier42("), "got:\n{out}");
        assert!(out.contains("template Transcript42("), "got:\n{out}");
    }

    #[test]
    fn header_includes_merklehash() {
        let si = minimal_stark_info(2, 10, 8);
        let vi = minimal_verifier_info();
        // Poseidon2 → emits hash/poseidon2/merklehash.circom.
        let opts = Pil2CircomOptions { hash: "Poseidon2".to_string(), ..Default::default() };
        let out = gen_stark_verifier_gl(None, &si, &vi, &opts).unwrap();
        assert!(out.contains("pragma circom 2.1.0;"));
        assert!(out.contains("include \"hash/poseidon2/merklehash.circom\";"));
        assert!(out.contains("include \"hash/poseidon2/pow.circom\";"));
        assert!(!out.contains("merklehash_gpu"));
    }

    #[test]
    fn header_gpu_when_split_linear_hash() {
        let mut si = minimal_stark_info(2, 10, 8);
        si["starkStruct"]["splitLinearHash"] = json!(true);
        let vi = minimal_verifier_info();
        // splitLinearHash is only supported with Poseidon2.
        let opts = Pil2CircomOptions { hash: "Poseidon2".to_string(), ..Default::default() };
        let out = gen_stark_verifier_gl(None, &si, &vi, &opts).unwrap();
        assert!(out.contains("include \"hash/poseidon2/merklehash_gpu.circom\";"));
    }

    #[test]
    fn header_includes_poseidon1_when_hash_is_poseidon1() {
        let si = minimal_stark_info(2, 10, 8);
        let vi = minimal_verifier_info();
        let opts = Pil2CircomOptions { hash: "Poseidon1".to_string(), ..Default::default() };
        let out = gen_stark_verifier_gl(None, &si, &vi, &opts).unwrap();
        assert!(out.contains("include \"hash/poseidon1/merklehash.circom\";"));
        assert!(out.contains("include \"hash/poseidon1/pow.circom\";"));
        assert!(out.contains("include \"tree/treeselector8.circom\";"));
        assert!(!out.contains("hash/poseidon2/"));
        assert!(!out.contains("treeselector4"));
    }

    #[test]
    fn component_main_with_publics() {
        let mut si = minimal_stark_info(2, 10, 8);
        si["nPublics"] = json!(3u64);
        let vi = minimal_verifier_info();
        let opts = Pil2CircomOptions::default();
        let out = gen_stark_verifier_gl(None, &si, &vi, &opts).unwrap();
        assert!(out.contains("{public [publics]}"), "got:\n{out}");
        assert!(out.contains("StarkVerifier()"), "got:\n{out}");
    }

    #[test]
    fn component_main_no_publics() {
        let si = minimal_stark_info(2, 10, 8);
        let vi = minimal_verifier_info();
        let opts = Pil2CircomOptions::default();
        let out = gen_stark_verifier_gl(None, &si, &vi, &opts).unwrap();
        assert!(!out.contains("{public"), "got:\n{out}");
    }

    #[test]
    fn wrong_hash_type_returns_error() {
        let mut si = minimal_stark_info(2, 10, 8);
        si["starkStruct"]["verificationHashType"] = json!("BN128");
        let vi = minimal_verifier_info();
        let opts = Pil2CircomOptions::default();
        let result = gen_stark_verifier_gl(None, &si, &vi, &opts);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("BN128"));
    }
}
