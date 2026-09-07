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

use proofman_fields::BLAKE3_TRANSCRIPT_XOF_WORDS;

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
    let hash_family = vadcop_info.get("hash").and_then(|v| v.as_str()).unwrap_or("Poseidon2");

    // ── Drive the transcript (this part must stay imperative) ─────────────────
    let mut transcript = Transcript::new(None, hash_family);
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
    } else if hash_family == "blake3" {
        // The chain width must be the one the prover steps with, which is what
        // `TranscriptDyn::get_chain_state` returns -- a whole XOF block for blake3, not
        // the digest's four words. Eight per round for the same one compression, so the
        // chain runs at half the rounds. Both sides read this constant.
        out_w = BLAKE3_TRANSCRIPT_XOF_WORDS;
        for _ in 0..out_w {
            initial_fields.push(transcript.get_fields1_pub());
        }
        let n_rounds = lattice_size.div_ceil(out_w);
        for i in 0..n_rounds.saturating_sub(1) {
            let base = i * out_w;
            chain.push(ChainRound {
                sig: format!("latticeHash_{i}"),
                base_next: base + out_w,
                inputs: (0..out_w).map(|j| format!("values[{}]", base + j)).collect(),
                // Permute8 takes no capacity; the sponge form threads one.
                state: Vec::new(),
            });
        }
    } else {
        // Poseidon families: 12 inputs + 4 capacity → 16 cells.
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

    // The chain fills `values` in whole rounds, so a lattice that is not a multiple of
    // the round width would leave a tail unassigned -- circom would fail far from here.
    // The prover asserts the same thing (`contributions_size % w == 0`).
    assert!(
        is_ec || lattice_size.is_multiple_of(out_w),
        "calculate_hashes: latticeSize ({lattice_size}) must be a multiple of the {hash_family} \
         round width ({out_w})"
    );

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
    ctx.insert("hash_family", &hash_family);

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

    /// The blake3 chain, at the width the prover steps with.
    ///
    /// `calculate_internal_contributions` derives its round count from
    /// `get_chain_state().len()`, so the emitter must use the same width or the two build
    /// different `values` arrays. Nothing downstream catches that: the compressor
    /// publishes the chain as an unconstrained output. Both sides read
    /// BLAKE3_TRANSCRIPT_XOF_WORDS; this pins the emitter's half.
    #[test]
    fn lattice_blake3_rounds_match_the_prover() {
        const LATTICE: usize = 368;
        let stark = stark_info(2, json!([]));
        let vadcop = json!({ "curve": "None", "latticeSize": LATTICE, "hash": "blake3" });
        let out = gen_calculate_hashes(&stark, &vadcop);

        // One Blake3Permute8 per round, and the prover runs contributions_size / w - 1.
        let rounds = out.matches("Blake3Permute8()(").count();
        assert_eq!(rounds, LATTICE / BLAKE3_TRANSCRIPT_XOF_WORDS - 1, "chain rounds\n{out}");

        assert!(out.contains(&format!("signal output values[{LATTICE}]")));
        // A sponge round would thread a capacity; permute8 takes none.
        assert!(!out.contains("Poseidon"), "must not emit a sponge round:\n{out}");
        // The seed is the first whole block, so the round after it starts at out_w.
        assert!(out.contains(&format!("values[{BLAKE3_TRANSCRIPT_XOF_WORDS} + j]")), "out:\n{out}");
    }

    /// A lattice that is not a whole number of rounds would leave a tail of `values`
    /// undriven, which circom reports far from the cause. The prover asserts the same.
    #[test]
    #[should_panic(expected = "must be a multiple of")]
    fn lattice_not_a_multiple_of_the_round_width_is_rejected() {
        let stark = stark_info(2, json!([]));
        let vadcop = json!({ "curve": "None", "latticeSize": 12, "hash": "blake3" });
        let _ = gen_calculate_hashes(&stark, &vadcop);
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
