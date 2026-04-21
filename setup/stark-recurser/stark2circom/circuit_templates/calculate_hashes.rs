//! Rust port of `vadcop/helpers/templates/calculate_hashes.circom.ejs`.
//!
//! Generates the `CalculateStage1Hash` circom template which computes a
//! Poseidon2 hash of `rootC + root1 + airValues` producing either an elliptic-
//! curve point (when `vadcopInfo.curve != "None"`) or a lattice vector.

use serde_json::Value;

use crate::stark2circom::transcript::Transcript;

/// Port of `calculate_hashes.circom.ejs`.
///
/// Returns the complete `template CalculateStage1Hash() { ... }` block as a string.
pub fn gen_calculate_hashes(stark_info: &Value, vadcop_info: &Value) -> String {
    let mut out = String::new();

    let arity = stark_info["starkStruct"]["merkleTreeArity"].as_u64().unwrap_or(2) as usize;
    let curve = vadcop_info["curve"].as_str().unwrap_or("None");
    let lattice_size = vadcop_info["latticeSize"].as_u64().unwrap_or(0) as usize;

    let air_values_map = stark_info["airValuesMap"].as_array().cloned().unwrap_or_default();
    let has_stage1_air_values = air_values_map.iter().any(|v| v["stage"].as_u64() == Some(1));

    // ── Template header ───────────────────────────────────────────────────────
    out.push_str("template CalculateStage1Hash() {\n");
    out.push_str("    signal input rootC[4];\n");
    out.push_str("    signal input root1[4];\n");

    if has_stage1_air_values {
        let len = air_values_map.len();
        out.push_str(&format!("    signal input airValues[{len}][3];\n"));
    }

    if curve != "None" {
        out.push_str("    signal output P[2][5];\n");
    } else {
        out.push_str(&format!("    signal output values[{lattice_size}];\n"));
    }

    // ── Build transcript ──────────────────────────────────────────────────────
    // The JS puts rootC[4] + root1[4] + stage-1 airValues[j][0] into the transcript,
    // then calls getStateEC or getStateLattices to extract the output fields.
    let mut transcript = Transcript::new(arity, None);
    transcript.put("rootC", 4);
    transcript.put("root1", 4);
    for (j, av) in air_values_map.iter().enumerate() {
        if av["stage"].as_u64() == Some(1) {
            // JS: transcript.put(`airValues[${j}]`, 1)  →  _add1(`airValues[j][0]`)
            transcript.put_single(&format!("airValues[{j}][0]"));
        }
    }

    // ── Collect output fields before emitting code ────────────────────────────
    if curve != "None" {
        // getStateEC: x[5] and y[5] from transcript, then HashToCurve
        let mut x_fields = Vec::new();
        let mut y_fields = Vec::new();
        for _ in 0..5 {
            x_fields.push(transcript.get_fields1_pub());
        }
        for _ in 0..5 {
            y_fields.push(transcript.get_fields1_pub());
        }

        let code = transcript.get_code();
        out.push_str(&code);
        out.push('\n');

        // Emit unused air value drain lines
        for (j, av) in air_values_map.iter().enumerate() {
            if av["stage"].as_u64() != Some(1) {
                out.push_str(&format!("    _ <== airValues[{j}]; // Unused air values at stage 1\n"));
            }
        }

        let curve_name = vadcop_info["curve"].as_str().unwrap_or("Unknown");
        let curve_constants = &vadcop_info["curveConstants"];
        let a_str = curve_constants["A"].as_str().unwrap_or("0");
        let b_str = curve_constants["B"].as_str().unwrap_or("0");
        let z_str = curve_constants["Z"].as_str().unwrap_or("0");
        let c1_str = curve_constants["C1"].as_str().unwrap_or("0");
        let c2_str = curve_constants["C2"].as_str().unwrap_or("0");

        out.push_str(&format!("    signal x[5] <== [{}];\n", x_fields.join(", ")));
        out.push_str(&format!("    signal y[5] <== [{}];\n\n", y_fields.join(", ")));
        out.push_str(&format!("    // Constants for the {curve_name} curve\n"));
        out.push_str(&format!("    var A[5] = [{a_str}];\n"));
        out.push_str(&format!("    var B[5] = [{b_str}];\n"));
        out.push_str(&format!("    var Z[5] = [{z_str}];\n"));
        out.push_str(&format!("    var C1[5] = [{c1_str}];\n"));
        out.push_str(&format!("    var C2[5] = [{c2_str}];\n"));
        out.push_str("    P <== HashToCurve(A, B, Z, C1, C2)(x,y);\n");
    } else {
        // getStateLattices: JS fills values[0..output_width] from the first transcript
        // output, then chains: each subsequent Poseidon2 takes
        //   inputs  = values[i*out_w .. i*out_w + input_w]
        //   state   = values[i*out_w + input_w .. i*out_w + out_w]
        // and writes its output into values[(i+1)*out_w .. (i+2)*out_w].
        //
        // The JS updateState in calculate_hashes.ejs also uses a special used_vals:
        //   count = number of stage-1 airValues
        //   used_vals = (hCnt < ceil(count/(4*(arity-1))) + 1) ? 4 : out_w
        // meaning the very first hash(es) only expose 4 outputs (state) and drain the
        // rest; all subsequent ones expose the full out_w.
        let stage1_count = air_values_map.iter().filter(|v| v["stage"].as_u64() == Some(1)).count();
        let input_w = 4 * (arity - 1);
        let out_w = 4 * arity;
        // How many early rounds drain to 4 outputs.
        let early_rounds = stage1_count.div_ceil(input_w) + 1;
        transcript.set_early_rounds_override(early_rounds, 4, out_w);

        // Collect the first out_w output fields (all from round 0).
        let mut initial_fields: Vec<String> = Vec::new();
        for _ in 0..out_w {
            initial_fields.push(transcript.get_fields1_pub());
        }

        let code = transcript.get_code();
        out.push_str(&code);
        out.push('\n');

        // Emit unused air value drain lines
        for (j, av) in air_values_map.iter().enumerate() {
            if av["stage"].as_u64() != Some(1) {
                out.push_str(&format!("    _ <== airValues[{j}]; // Unused air values at stage 1\n"));
            }
        }

        // Assign values[0..out_w] from the initial transcript output.
        for (i, f) in initial_fields.iter().enumerate() {
            out.push_str(&format!("    values[{i}] <== {f};\n"));
        }

        // Chain subsequent rounds: n = ceil(lattice_size / out_w), emit n-1 extra hashes.
        let n_rounds = lattice_size.div_ceil(out_w);
        for i in 0..(n_rounds - 1) {
            let sig = format!("transcriptHash_{}", i + 1);
            let base = i * out_w;
            let inputs: Vec<String> = (0..input_w).map(|j| format!("values[{}]", base + j)).collect();
            let state: Vec<String> = (0..4).map(|j| format!("values[{}]", base + input_w + j)).collect();
            out.push_str(&format!(
                "    signal {sig}[{out_w}] <== Poseidon2({arity}, {out_w})([{inputs}], [{state}]);\n",
                inputs = inputs.join(", "),
                state = state.join(", "),
            ));
            out.push_str(&format!("    for (var j = 0; j < {out_w}; j++) {{\n"));
            out.push_str(&format!("        values[{} + j] <== {sig}[j];\n", base + out_w));
            out.push_str("    }\n\n");
        }
    }

    out.push_str("}\n");
    out
}
