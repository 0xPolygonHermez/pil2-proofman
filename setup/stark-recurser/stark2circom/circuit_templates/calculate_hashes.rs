//! Rust port of `vadcop/helpers/templates/calculate_hashes.circom.ejs`.
//!
//! Generates the `CalculateStage1Hash` circom template which computes a
//! Poseidon2 hash of `rootC + root1 + airValues` producing either an elliptic-
//! curve point (when `vadcopInfo.curve != "None"`) or a lattice vector.
//!
//! Implementation: the order-dependent `Transcript` state machine drives the
//! Poseidon2 chain in Rust, and the resulting code + collected output fields
//! are passed to a Tera template that owns the Circom shape.

use serde::Serialize;
use serde_json::Value;
use tera::Context as TeraCtx;

use crate::stark2circom::transcript::Transcript;

const CALCULATE_HASHES_TMPL: &str = include_str!("tera/calculate_hashes.circom.tera");

#[derive(Serialize)]
struct ChainRound {
    sig: String,
    base_next: usize, // == base + out_w; the destination offset for the copy loop
    inputs: Vec<String>,
    state: Vec<String>,
}

/// Port of `calculate_hashes.circom.ejs`.
///
/// Returns the complete `template CalculateStage1Hash() { ... }` block as a string.
pub fn gen_calculate_hashes(stark_info: &Value, vadcop_info: &Value) -> String {
    let arity = stark_info["starkStruct"]["merkleTreeArity"].as_u64().unwrap_or(2) as usize;
    let curve = vadcop_info["curve"].as_str().unwrap_or("None");
    let lattice_size = vadcop_info["latticeSize"].as_u64().unwrap_or(0) as usize;
    let is_ec = curve != "None";

    let air_values_map = stark_info["airValuesMap"].as_array().cloned().unwrap_or_default();
    let air_values_len = air_values_map.len();
    let has_stage1_air_values = air_values_map.iter().any(|v| v["stage"].as_u64() == Some(1));

    // The contribution hash must use the same family as the proof being recursed
    // (carried in globalInfo.hash). Poseidon2 ⇒ `Poseidon2(4, …)`, else Poseidon1
    // (`Poseidon(…)`). Defaults to Poseidon2 (the system default) when absent.
    let poseidon2 = vadcop_info.get("hash").and_then(|v| v.as_str()).map(|s| s == "Poseidon2").unwrap_or(true);

    // ── Drive the transcript (this part must stay imperative) ─────────────────
    let mut transcript = Transcript::new(arity, None);
    transcript.set_poseidon2(poseidon2);
    transcript.put("rootC", 4);
    transcript.put("root1", 4);
    for (j, av) in air_values_map.iter().enumerate() {
        if av["stage"].as_u64() == Some(1) {
            transcript.put_single(&format!("airValues[{j}][0]"));
        }
    }

    // Collect per-branch output fields and the chain rounds.
    let mut x_fields: Vec<String> = Vec::new();
    let mut y_fields: Vec<String> = Vec::new();
    let mut initial_fields: Vec<String> = Vec::new();
    let mut chain: Vec<ChainRound> = Vec::new();
    let mut out_w: usize = 0;

    if is_ec {
        for _ in 0..5 {
            x_fields.push(transcript.get_fields1_pub());
        }
        for _ in 0..5 {
            y_fields.push(transcript.get_fields1_pub());
        }
    } else {
        // GL width 16 (Poseidon1 or Poseidon2 per `poseidon2`): 12 inputs + 4 capacity → 16 cells.
        let stage1_count = air_values_map.iter().filter(|v| v["stage"].as_u64() == Some(1)).count();
        let input_w = 12;
        out_w = 16;
        let early_rounds = stage1_count.div_ceil(input_w) + 1;
        transcript.set_early_rounds_override(early_rounds, 4, out_w);

        for _ in 0..out_w {
            initial_fields.push(transcript.get_fields1_pub());
        }

        let n_rounds = lattice_size.div_ceil(out_w);
        for i in 0..n_rounds.saturating_sub(1) {
            let sig = format!("transcriptHash_{}", early_rounds + i);
            let base = i * out_w;
            let inputs: Vec<String> = (0..input_w).map(|j| format!("values[{}]", base + j)).collect();
            let state: Vec<String> = (0..4).map(|j| format!("values[{}]", base + input_w + j)).collect();
            chain.push(ChainRound { sig, base_next: base + out_w, inputs, state });
        }
    }

    // Snapshot the generated transcript code AFTER all put/get calls are done.
    let transcript_code = transcript.get_code();

    // Indices of air values with stage != 1 — those get `_ <== airValues[j];` drain lines.
    let unused_air_values: Vec<usize> = air_values_map
        .iter()
        .enumerate()
        .filter_map(|(j, av)| if av["stage"].as_u64() != Some(1) { Some(j) } else { None })
        .collect();

    // ── Build Tera context ───────────────────────────────────────────────────
    let mut ctx = TeraCtx::new();
    ctx.insert("has_stage1_air_values", &has_stage1_air_values);
    ctx.insert("air_values_len", &air_values_len);
    ctx.insert("is_ec", &is_ec);
    ctx.insert("lattice_size", &lattice_size);
    ctx.insert("transcript_code", &transcript_code);
    ctx.insert("unused_air_values", &unused_air_values);
    ctx.insert("poseidon2", &poseidon2);

    if is_ec {
        let curve_constants = &vadcop_info["curveConstants"];
        ctx.insert("x_fields", &x_fields);
        ctx.insert("y_fields", &y_fields);
        ctx.insert("curve_name", &curve);
        ctx.insert("curve_a", &curve_constants["A"].as_str().unwrap_or("0"));
        ctx.insert("curve_b", &curve_constants["B"].as_str().unwrap_or("0"));
        ctx.insert("curve_z", &curve_constants["Z"].as_str().unwrap_or("0"));
        ctx.insert("curve_c1", &curve_constants["C1"].as_str().unwrap_or("0"));
        ctx.insert("curve_c2", &curve_constants["C2"].as_str().unwrap_or("0"));
    } else {
        ctx.insert("arity", &arity);
        ctx.insert("out_w", &out_w);
        ctx.insert("initial_fields", &initial_fields);
        ctx.insert("chain", &chain);
    }

    super::templates::render(CALCULATE_HASHES_TMPL, &ctx).expect("calculate_hashes template render")
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn stark_info(arity: u64, air_values_map: Value) -> Value {
        json!({
            "starkStruct": { "merkleTreeArity": arity },
            "airValuesMap": air_values_map,
        })
    }

    #[test]
    fn lattice_no_air_values_poseidon2() {
        let stark = stark_info(2, json!([]));
        let vadcop = json!({
            "curve": "None",
            "latticeSize": 32,
            "hash": "Poseidon2",
        });
        let out = gen_calculate_hashes(&stark, &vadcop);

        assert!(out.contains("template CalculateStage1Hash()"));
        assert!(out.contains("signal output values[32]"));
        assert!(!out.contains("signal input airValues"));
        // Poseidon2 family must emit the Poseidon2 template, never the (undefined-here) Poseidon1 one.
        assert!(out.contains("Poseidon2(4, 16)("), "out:\n{out}");
        assert!(!out.contains("<== Poseidon(16)("), "must not emit Poseidon1 template:\n{out}");
        assert!(out.contains("values[0] <=="));
        // out_w = 16 → first round fills values[0..16]
        assert!(out.contains("values[15] <=="));
        assert!(!out.contains("HashToCurve"));
    }

    #[test]
    fn lattice_no_air_values_poseidon1() {
        let stark = stark_info(2, json!([]));
        let vadcop = json!({
            "curve": "None",
            "latticeSize": 32,
            "hash": "Poseidon1",
        });
        let out = gen_calculate_hashes(&stark, &vadcop);

        // Poseidon1 family emits the bare `Poseidon(nOuts)` template.
        assert!(out.contains("Poseidon(16)("), "out:\n{out}");
        assert!(!out.contains("Poseidon2(4, 16)("), "must not emit Poseidon2 template:\n{out}");
    }

    #[test]
    fn lattice_with_stage1_air_values() {
        let stark = stark_info(
            2,
            json!([
                { "stage": 1 },
                { "stage": 2 },
                { "stage": 1 },
                { "stage": 3 }
            ]),
        );
        let vadcop = json!({
            "curve": "None",
            "latticeSize": 32,
        });
        let out = gen_calculate_hashes(&stark, &vadcop);

        assert!(out.contains("signal input airValues[4][3]"));
        // Drains: indices 1 and 3 (stage != 1) get drain lines; 0 and 2 do not.
        assert!(out.contains("_ <== airValues[1]; // Unused air values at stage 1"));
        assert!(out.contains("_ <== airValues[3]; // Unused air values at stage 1"));
        assert!(!out.contains("_ <== airValues[0];"));
        assert!(!out.contains("_ <== airValues[2];"));
        assert_eq!(
            out.matches("// Unused air values at stage 1").count(),
            2,
            "exactly one drain line per non-stage-1 entry"
        );
    }

    #[test]
    fn ec_branch() {
        let stark = stark_info(2, json!([]));
        let vadcop = json!({
            "curve": "BN254",
            "latticeSize": 0,
            "curveConstants": {
                "A": "1, 2, 3, 4, 5",
                "B": "6, 7, 8, 9, 10",
                "Z": "11, 12, 13, 14, 15",
                "C1": "16, 17, 18, 19, 20",
                "C2": "21, 22, 23, 24, 25"
            }
        });
        let out = gen_calculate_hashes(&stark, &vadcop);

        assert!(out.contains("signal output P[2][5];"));
        assert!(out.contains("signal x[5] <== ["));
        assert!(out.contains("signal y[5] <== ["));
        assert!(out.contains("// Constants for the BN254 curve"));
        assert!(out.contains("var A[5] = [1, 2, 3, 4, 5];"));
        assert!(out.contains("var B[5] = [6, 7, 8, 9, 10];"));
        assert!(out.contains("var Z[5] = [11, 12, 13, 14, 15];"));
        assert!(out.contains("var C1[5] = [16, 17, 18, 19, 20];"));
        assert!(out.contains("var C2[5] = [21, 22, 23, 24, 25];"));
        assert!(out.contains("P <== HashToCurve(A, B, Z, C1, C2)(x,y);"));
        assert!(!out.contains("signal output values"));
    }
}
