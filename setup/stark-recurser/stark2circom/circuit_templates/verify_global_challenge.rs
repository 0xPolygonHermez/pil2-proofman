//! Rust port of `vadcop/helpers/templates/verify_global_challenge.circom.ejs`.

use serde_json::Value;

use crate::stark2circom::transcript::Transcript;

/// Port of `verify_global_challenge.circom.ejs`.
///
/// Returns the complete `template VerifyGlobalChallenges() { ... }` block.
pub fn gen_verify_global_challenge(stark_info: &Value, vadcop_info: &Value) -> String {
    let mut out = String::new();

    let arity = stark_info["starkStruct"]["merkleTreeArity"].as_u64().unwrap_or(2) as usize;
    let n_publics = vadcop_info["nPublics"].as_u64().unwrap_or(0) as usize;
    let curve = vadcop_info["curve"].as_str().unwrap_or("None");
    let lattice_size = vadcop_info["latticeSize"].as_u64().unwrap_or(0) as usize;

    let proof_values_map = vadcop_info["proofValuesMap"].as_array().cloned().unwrap_or_default();
    let agg_types_len = vadcop_info["aggTypes"].as_array().map(|a| a.len()).unwrap_or(0);
    let stage1_pv_count = proof_values_map.iter().filter(|v| v["stage"].as_u64() == Some(1)).count();

    // JS: ceil((nPublics + stage1ProofValues + 10*aggTypes.length) / (4*(arity-1)))
    // rounds before the "last" (final) round that outputs only 3 fields.
    let input_width = 4 * (arity - 1);
    let total_inputs = n_publics + stage1_pv_count + 10 * agg_types_len;
    let early_rounds = if input_width > 0 { total_inputs.div_ceil(input_width) } else { 1 };

    out.push_str("template VerifyGlobalChallenges() {\n\n");

    if n_publics > 0 {
        out.push_str(&format!("    signal input publics[{n_publics}];\n"));
    }
    if !proof_values_map.is_empty() {
        let pvm_len = proof_values_map.len();
        out.push_str(&format!("    signal input proofValues[{pvm_len}][3];\n"));
    }
    if curve != "None" {
        out.push_str(&format!("    signal input stage1Hash[{agg_types_len}][2][5];\n"));
    } else {
        out.push_str(&format!("    signal input stage1Hash[{agg_types_len}][{lattice_size}];\n"));
    }
    out.push_str("    \n");
    out.push_str("    signal input globalChallenge[3];\n");
    out.push_str("    signal calculatedGlobalChallenge[3];\n\n");

    let poseidon2 = vadcop_info.get("hash").and_then(|v| v.as_str()).map(|s| s == "Poseidon2").unwrap_or(true);

    // Build transcript with per-round used_vals override:
    // rounds < early_rounds → used_vals=4, final round → used_vals=3
    let mut transcript = Transcript::new(arity, None);
    transcript.set_poseidon2(poseidon2);
    transcript.set_early_rounds_override(early_rounds, 4, 3);

    if n_publics > 0 {
        transcript.put("publics", n_publics);
    }
    for (j, pv) in proof_values_map.iter().enumerate() {
        if pv["stage"].as_u64() == Some(1) {
            transcript.put_single(&format!("proofValues[{j}][0]"));
        }
        // unused ones get drain lines separately below
    }
    for k in 0..agg_types_len {
        if curve != "None" {
            transcript.put_2d(&format!("stage1Hash[{k}]"), 2, 5);
        } else {
            transcript.put(&format!("stage1Hash[{k}]"), lattice_size);
        }
    }
    transcript.get_field("calculatedGlobalChallenge");

    let code = transcript.get_code();
    out.push_str(&code);
    out.push('\n');

    // Drain unused proofValues
    for (j, pv) in proof_values_map.iter().enumerate() {
        if pv["stage"].as_u64() != Some(1) {
            out.push_str(&format!("    _ <== proofValues[{j}][0]; // Unused proof values at stage 1\n"));
        }
    }
    out.push('\n');

    out.push_str("    globalChallenge === calculatedGlobalChallenge;\n");
    out.push_str("}\n");
    out
}
