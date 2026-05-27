//! Rust port of `get_sha256_inputs.circom.ejs`.
//!
//! Generates the `getSha256Inputs` circom template that packs public inputs
//! into SHA-256 and returns a single field element output.
//!
//! Implementation: builds per-definition rows + cumulative bit offsets in Rust
//! and delegates the Circom shape to a Tera template.

use serde::Serialize;
use serde_json::Value;
use tera::Context as TeraCtx;

const GET_SHA256_INPUTS_TMPL: &str = include_str!("tera/get_sha256_inputs.circom.tera");

#[derive(Serialize)]
struct Sha256Def<'a> {
    name: &'a str,
    initial_pos: usize,
    bits_offset: usize,
    n_values: usize,
    n_chunks: usize,
    bits_per_chunk: usize,
    eff_bits: usize,
    is_vk: bool,
    avoid_alias: bool,
    single_chunk: bool,
}

/// Generate the `getSha256Inputs` circom template (standalone file for `include`).
///
/// * `publics` — the publics info JSON (`publics_info.json` or `--publics-info` arg).
///   Must contain `.nPublics: usize` and `.definitions: [...]`.
///
/// Returns a circom fragment (without the `pragma` header, just the template body
/// and any `include` lines) suitable for inclusion at the top-level of another circom.
pub fn gen_get_sha256_inputs(publics: &Value) -> String {
    let n_publics = publics["nPublics"].as_u64().unwrap_or(0) as usize;

    let empty = Vec::new();
    let defs_json = publics["definitions"].as_array().unwrap_or(&empty);

    let mut defs: Vec<Sha256Def<'_>> = Vec::with_capacity(defs_json.len());
    let mut offset: usize = 0;
    for d in defs_json {
        let name = d["name"].as_str().unwrap_or("unknown");
        let initial_pos = d["initialPos"].as_u64().unwrap_or(0) as usize;
        let n_values = d["nValues"].as_u64().unwrap_or(0) as usize;
        let n_chunks = d["chunks"].as_array().and_then(|a| a.first()).and_then(|v| v.as_u64()).unwrap_or(1) as usize;
        let bits_per_chunk =
            d["chunks"].as_array().and_then(|a| a.get(1)).and_then(|v| v.as_u64()).unwrap_or(64) as usize;
        let avoid_alias = d["avoidAlias"].as_bool().unwrap_or(false);
        let is_vk = d["verificationKey"].as_bool().unwrap_or(false);
        let single_chunk = n_chunks == 1;
        let eff_bits = if avoid_alias && single_chunk { bits_per_chunk - 1 } else { bits_per_chunk };

        defs.push(Sha256Def {
            name,
            initial_pos,
            bits_offset: offset,
            n_values,
            n_chunks,
            bits_per_chunk,
            eff_bits,
            is_vk,
            avoid_alias,
            single_chunk,
        });

        offset += n_values * n_chunks * bits_per_chunk;
    }
    let final_offset = offset;
    let total_bits = offset + 256; // + rootCVadcopFinal (4×64 bits)

    let mut ctx = TeraCtx::new();
    ctx.insert("n_publics", &n_publics);
    ctx.insert("total_bits", &total_bits);
    ctx.insert("final_offset", &final_offset);
    ctx.insert("defs", &defs);

    super::templates::render(GET_SHA256_INPUTS_TMPL, &ctx).expect("get_sha256_inputs template render")
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn run(publics: Value) -> String {
        gen_get_sha256_inputs(&publics)
    }

    #[test]
    fn single_chunk_def_no_vk() {
        let publics = json!({
            "nPublics": 4,
            "definitions": [
                {
                    "name": "myField",
                    "initialPos": 0,
                    "nValues": 4,
                    "chunks": [1, 64],
                    "avoidAlias": false,
                    "verificationKey": false
                }
            ]
        });
        let out = run(publics);
        assert!(out.contains("template getSha256Inputs()"));
        assert!(out.contains("signal input publics[4]"));
        assert!(out.contains("component myFieldBits[4]"));
        assert!(out.contains("myFieldBits[i] = Num2Bits(64)"));
        // standard (non-VK) byte-swapped indexing
        assert!(out.contains("(j\\8)*8 + (7 - j%8)"));
        // no big-endian VK form
        assert!(!out.contains("64 - 1 - j"));
        // no avoid_alias zero-pad
        assert!(!out.contains("publicsHasher.in[myFieldBitsOffset] <== 0;"));
    }

    #[test]
    fn single_chunk_vk_with_avoid_alias() {
        let publics = json!({
            "nPublics": 4,
            "definitions": [
                {
                    "name": "vk",
                    "initialPos": 0,
                    "nValues": 4,
                    "chunks": [1, 64],
                    "avoidAlias": true,
                    "verificationKey": true
                }
            ]
        });
        let out = run(publics);
        // avoid_alias trims to bits_per_chunk - 1
        assert!(out.contains("vkBits[i] = Num2Bits(63)"));
        // big-endian VK indexing
        assert!(out.contains("vkBitsOffset + i*64 + 64 - 1 - j"));
        // zero-pad line at the BitsOffset slot
        assert!(out.contains("publicsHasher.in[vkBitsOffset] <== 0;"));
    }

    #[test]
    fn multi_chunk_def() {
        let publics = json!({
            "nPublics": 4,
            "definitions": [
                {
                    "name": "multi",
                    "initialPos": 0,
                    "nValues": 2,
                    "chunks": [2, 32],
                    "avoidAlias": false,
                    "verificationKey": false
                }
            ]
        });
        let out = run(publics);
        assert!(out.contains("component multiBits[2][2]"));
        assert!(out.contains("for (var k = 0; k < 2; k++)"));
        assert!(out.contains("multiBits[k][i] = Num2Bits(32)"));
        assert!(out.contains("multiBits[k][i].in <== publics[multiPos + k * 2 + i]"));
        // no single-chunk form
        assert!(!out.contains("component multiBits[2];"));
    }

    #[test]
    fn full_template_skeleton() {
        let publics = json!({
            "nPublics": 8,
            "definitions": [
                {
                    "name": "a",
                    "initialPos": 0,
                    "nValues": 1,
                    "chunks": [1, 64],
                    "avoidAlias": false,
                    "verificationKey": false
                },
                {
                    "name": "b",
                    "initialPos": 1,
                    "nValues": 1,
                    "chunks": [1, 64],
                    "avoidAlias": false,
                    "verificationKey": true
                }
            ]
        });
        let out = run(publics);
        assert!(out.contains("include \"lessthangl.circom\";"));
        assert!(out.contains("include \"sha256/sha256.circom\";"));
        assert!(out.contains("include \"bitify.circom\";"));
        assert!(out.contains("template getSha256Inputs()"));
        // total = 1*1*64 + 1*1*64 + 256 = 384
        assert!(out.contains("var totalBits = 384;"));
        assert!(out.contains("component publicsHasher = Sha256(totalBits);"));
        // bits offsets: a at 0, b at 64, rootCVadcopFinal at 128
        assert!(out.contains("var aBitsOffset = 0;"));
        assert!(out.contains("var bBitsOffset = 64;"));
        assert!(out.contains("var rootCVadcopFinalBitsOffset = 128;"));
        assert!(out.contains("component rootCVadcopFinalBits[4]"));
        assert!(out.contains("component b2nPublicsHash = Bits2Num(256);"));
        assert!(out.contains("publicsHash <== b2nPublicsHash.out;"));
    }
}
