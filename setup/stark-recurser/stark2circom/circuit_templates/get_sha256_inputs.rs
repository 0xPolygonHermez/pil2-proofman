//! Rust port of `get_sha256_inputs.circom.ejs`.
//!
//! Generates the `getSha256Inputs` circom template that packs public inputs
//! into SHA-256 and returns a single field element output.

use serde_json::Value;

/// Generate the `getSha256Inputs` circom template (standalone file for `include`).
///
/// * `publics` — the publics info JSON (`publics_info.json` or `--publics-info` arg).
///   Must contain `.nPublics: usize` and `.definitions: [...]`.
///
/// Returns a circom fragment (without the `pragma` header, just the template body
/// and any `include` lines) suitable for inclusion at the top-level of another circom.
pub fn gen_get_sha256_inputs(publics: &Value) -> String {
    let n_publics = publics["nPublics"].as_u64().unwrap_or(0) as usize;

    // Compute total bits: sum over definitions of (nValues * chunks[0] * chunks[1]) + 256 (rootCVadcopFinal)
    let defs = publics["definitions"].as_array().cloned().unwrap_or_default();
    let data_bits: usize = defs
        .iter()
        .map(|d| {
            let n_values = d["nValues"].as_u64().unwrap_or(0) as usize;
            let n_chunks =
                d["chunks"].as_array().and_then(|a| a.first()).and_then(|v| v.as_u64()).unwrap_or(1) as usize;
            let bits_per_chunk =
                d["chunks"].as_array().and_then(|a| a.get(1)).and_then(|v| v.as_u64()).unwrap_or(64) as usize;
            n_values * n_chunks * bits_per_chunk
        })
        .sum();
    let total_bits = data_bits + 256; // + rootCVadcopFinal (4×64 bits)

    let mut out = String::new();

    out.push_str("include \"lessthangl.circom\";\n");
    out.push_str("include \"sha256/sha256.circom\";\n");
    out.push_str("include \"bitify.circom\";\n\n");

    out.push_str("template getSha256Inputs() {\n");
    out.push_str(&format!("    signal input publics[{n_publics}];\n\n"));
    out.push_str("    signal input rootCVadcopFinal[4];\n\n");
    out.push_str("    signal output publicsHash;\n\n");

    // Range-check each public
    out.push_str(&format!(
        "    for (var i = 0; i < {n_publics}; i++) {{\n        _<== LessThanGoldilocks()(publics[i]);\n    }}\n\n"
    ));

    out.push_str(&format!("    var totalBits = {total_bits};\n\n"));
    out.push_str(&format!("    component publicsHasher = Sha256({total_bits});\n\n"));

    // Per-definition bit packing
    let mut offset: usize = 0;
    for def in &defs {
        let n_values = def["nValues"].as_u64().unwrap_or(0) as usize;
        let n_chunks = def["chunks"].as_array().and_then(|a| a.first()).and_then(|v| v.as_u64()).unwrap_or(1) as usize;
        let bits_per_chunk =
            def["chunks"].as_array().and_then(|a| a.get(1)).and_then(|v| v.as_u64()).unwrap_or(64) as usize;
        let name = def["name"].as_str().unwrap_or("unknown");
        let initial_pos = def["initialPos"].as_u64().unwrap_or(0) as usize;
        let avoid_alias = def["avoidAlias"].as_bool().unwrap_or(false);
        let is_vk = def["verificationKey"].as_bool().unwrap_or(false);

        out.push_str(&format!("    var {name}Pos = {initial_pos};\n"));
        out.push_str(&format!("    var {name}BitsOffset = {offset};\n"));

        let bits_incr = n_values * n_chunks * bits_per_chunk;
        offset += bits_incr;

        if n_chunks == 1 {
            // Single-chunk: use Num2Bits directly
            let eff_bits = if avoid_alias { bits_per_chunk - 1 } else { bits_per_chunk };
            out.push_str(&format!("    component {name}Bits[{n_values}];\n"));
            out.push_str(&format!("    for(var i=0; i< {n_values}; i++) {{\n"));
            out.push_str(&format!("        {name}Bits[i] = Num2Bits({eff_bits});\n"));
            out.push_str(&format!("        {name}Bits[i].in <== publics[{name}Pos + i];\n"));
            out.push_str(&format!("        for (var j=0; j< {eff_bits}; j++) {{\n"));
            if is_vk {
                // Verification key: big-endian bit order (MSB first)
                out.push_str(&format!(
                    "            publicsHasher.in[{name}BitsOffset + i*{bits_per_chunk} + {bits_per_chunk} - 1 - j] <== {name}Bits[i].out[j];\n"
                ));
            } else {
                // Standard: byte-swapped within each byte
                out.push_str(&format!(
                    "            publicsHasher.in[{name}BitsOffset + i*{bits_per_chunk} + (j\\8)*8 + (7 - j%8)] <== {name}Bits[i].out[j];\n"
                ));
            }
            out.push_str("        }\n");
            if avoid_alias {
                out.push_str(&format!("        publicsHasher.in[{name}BitsOffset] <== 0;\n"));
            }
            out.push_str("    }\n");
        } else {
            // Multi-chunk
            out.push_str(&format!("    component {name}Bits[{n_values}][{n_chunks}];\n"));
            out.push_str(&format!("    for (var k = 0; k < {n_values}; k++) {{\n"));
            out.push_str(&format!("        for(var i=0; i< {n_chunks}; i++) {{\n"));
            out.push_str(&format!("            {name}Bits[k][i] = Num2Bits({bits_per_chunk});\n"));
            out.push_str(&format!("            {name}Bits[k][i].in <== publics[{name}Pos + k * {n_chunks} + i];\n"));
            out.push_str(&format!("            for(var j = 0; j < {bits_per_chunk}; j++) {{\n"));
            out.push_str(&format!(
                "                publicsHasher.in[{name}BitsOffset + (k*{n_chunks} + i)*{bits_per_chunk} + (j\\8)*8 + (7 - j%8)] <== {name}Bits[k][i].out[j];\n"
            ));
            out.push_str("            }\n");
            out.push_str("        }\n");
            out.push_str("    }\n");
        }
        out.push('\n');
    }

    // rootCVadcopFinal packing (4 × 64 bits, big-endian)
    out.push_str(&format!("    var rootCVadcopFinalBitsOffset = {offset};\n"));
    out.push_str("    component rootCVadcopFinalBits[4];\n");
    out.push_str("    for (var i = 0; i < 4; i++) {\n");
    out.push_str("        rootCVadcopFinalBits[i] = Num2Bits(64);\n");
    out.push_str("        rootCVadcopFinalBits[i].in <== rootCVadcopFinal[i];\n");
    out.push_str("        for (var j = 0; j < 64; j++) {\n");
    out.push_str("            publicsHasher.in[rootCVadcopFinalBitsOffset + i*64 + 63 - j] <== rootCVadcopFinalBits[i].out[j];\n");
    out.push_str("        }\n");
    out.push_str("    }\n\n");

    // Output: Sha256 → Bits2Num (256-bit, big-endian)
    out.push_str("    component b2nPublicsHash = Bits2Num(256);\n");
    out.push_str("    for (var i = 0; i < 256; i++) {\n");
    out.push_str("        b2nPublicsHash.in[i] <== publicsHasher.out[255-i];\n");
    out.push_str("    }\n\n");

    out.push_str("    publicsHash <== b2nPublicsHash.out;\n");
    out.push_str("}\n");

    out
}
