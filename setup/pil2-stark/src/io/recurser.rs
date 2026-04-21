//! High-level wrappers for circom generation tools.
//!
//! `pil2circom` runs both the Node.js and Rust implementations in parallel so
//! their outputs can be compared during the transition period.  The JS result
//! is always returned; any difference is logged as a warning.
//! `gen_circom` uses the in-process Rust generator exclusively — no JS fallback.

use std::fs;
use std::path::Path;

use anyhow::{Context, Result};
use serde_json::Value;

use stark_recurser::stark2circom::{gen_circom_circuit, gen_stark_verifier, CircomGenOptions, GenCircomCircuitInput, StarkVerifierOptions};
use crate::proving_key::recursive::run_pil2circom_js;

// ── pil2circom ───────────────────────────────────────────────────────────────

/// Normalise a circom line for comparison: trim trailing whitespace.
/// Empty lines are kept so line numbers stay meaningful.
fn normalise_lines(s: &str) -> Vec<&str> {
    s.lines().map(|l| l.trim_end()).collect()
}

/// Compare JS and Rust pil2circom outputs for a given air (identified by
/// `name`).  Both outputs are written to:
///   /tmp/pil2circom_<name>_js.circom
///   /tmp/pil2circom_<name>_rust.circom
/// Logs at INFO if semantically identical (after trailing-whitespace
/// normalisation), WARN with the first differing line otherwise.
fn compare_pil2circom_outputs(name: &str, js: &str, rust: &str) {
    // Sanitise name for use as a filename.
    let safe = name.replace(['/', '\\', ' ', ':'], "_");
    let js_path   = std::path::PathBuf::from(format!("/tmp/pil2circom_{safe}_js.circom"));
    let rust_path = std::path::PathBuf::from(format!("/tmp/pil2circom_{safe}_rust.circom"));

    if let Err(e) = fs::write(&js_path, js) {
        tracing::warn!("pil2circom[{name}]: could not write JS output: {e}");
    }
    if let Err(e) = fs::write(&rust_path, rust) {
        tracing::warn!("pil2circom[{name}]: could not write Rust output: {e}");
    }

    if js == rust {
        tracing::info!("pil2circom[{name}]: Rust and JS outputs are identical");
        return;
    }

    let js_lines   = normalise_lines(js);
    let rust_lines = normalise_lines(rust);

    if js_lines == rust_lines {
        tracing::info!("pil2circom[{name}]: Rust and JS outputs match (after whitespace normalisation)");
        return;
    }

    let first_diff = js_lines.iter().zip(rust_lines.iter()).enumerate()
        .find(|(_, (a, b))| a != b);

    match first_diff {
        Some((n, (js_line, rust_line))) => {
            tracing::warn!(
                "pil2circom[{name}]: outputs differ ({} vs {} lines). \
                 First difference at line {}:\n  JS  : {:?}\n  Rust: {:?}\n  \
                 diff {} {}",
                js_lines.len(), rust_lines.len(),
                n + 1, js_line, rust_line,
                js_path.display(), rust_path.display(),
            );
        }
        None => {
            tracing::warn!(
                "pil2circom[{name}]: outputs differ only in line count ({} vs {} lines)\n  \
                 diff {} {}",
                js_lines.len(), rust_lines.len(),
                js_path.display(), rust_path.display(),
            );
        }
    }
}


#[derive(Debug, Default)]
pub struct Pil2CircomOptions {
    /// Omit the `component main` line so the file can be `include`d.
    pub skip_main: bool,
    /// Pass the constant root as a public input instead of embedding it.
    pub verkey_input: bool,
    /// Emit an `enable` input signal.
    pub enable_input: bool,
    /// Pass challenges as public inputs.
    pub input_challenges: bool,
}

/// Generate a verifier circom by calling `node src/pil2circom/main_pil2circom.js`.
///
/// Writes temporary JSON files for the inputs, invokes the script, and returns
/// the output circom source as a `String`.
pub fn pil2circom(
    const_root: &[String; 4],
    stark_info: &Value,
    verifier_info: &Value,
    opts: &Pil2CircomOptions,
) -> Result<String> {
    let tmp_dir = tempfile::tempdir().context("Failed to create temp dir for pil2circom")?;
    let tmp = tmp_dir.path();

    let si_path = tmp.join("starkinfo.json");
    let vi_path = tmp.join("verifierinfo.json");
    let out_path = tmp.join("verifier.circom");

    fs::write(&si_path, serde_json::to_string(stark_info)?)?;
    fs::write(&vi_path, serde_json::to_string(verifier_info)?)?;

    // When not using verkey as a witness input, embed the constant root in the
    // circom by passing the file to pil2circom.  When verkey_input is true, no
    // verkey file is needed (it comes in as a public signal).
    let vk_path_buf;
    let verkey_path: Option<&Path> = if !opts.verkey_input {
        let hash_type = stark_info["starkStruct"]["verificationHashType"].as_str().unwrap_or("GL");
        let vk_json = if hash_type == "BN128" {
            // BN128: constRoot is a single scalar in const_root[0] (big decimal string).
            // The pil2circom EJS template renders it as: signal rootC <== <%- constRoot %>;
            // JSONbig in the JS will parse a JSON string as a JS string; EJS <%- x %> calls
            // x.toString() which renders the digits correctly as a circom literal.
            serde_json::json!({ "constRoot": const_root[0] })
        } else {
            // GL: constRoot is a 4-element array of u64 values.
            serde_json::json!({
                "constRoot": const_root
                    .iter()
                    .map(|s| s.parse::<u64>().unwrap_or(0))
                    .collect::<Vec<_>>()
            })
        };
        vk_path_buf = tmp.join("verkey.json");
        fs::write(&vk_path_buf, serde_json::to_string(&vk_json)?)?;
        Some(vk_path_buf.as_path())
    } else {
        None
    };

    run_pil2circom_js(
        &si_path,
        &vi_path,
        verkey_path,
        &out_path,
        opts.skip_main,
        opts.verkey_input,
        opts.enable_input,
        opts.input_challenges,
    )
    .context("pil2circom JS failed")?;

    let js_output = fs::read_to_string(&out_path).context("Failed to read pil2circom output")?;

    // ── Rust comparison ───────────────────────────────────────────────────────
    let rust_opts = StarkVerifierOptions {
        skip_main:              opts.skip_main,
        verkey_input:           opts.verkey_input,
        enable_input:           opts.enable_input,
        input_challenges:       opts.input_challenges,
        fri_queries_batch_size: None,
        multi_fri:              false,
    };
    let rust_root: Option<&[String; 4]> = if opts.verkey_input { None } else { Some(const_root) };
    match gen_stark_verifier(rust_root, stark_info, verifier_info, &rust_opts) {
        Ok(rust_output) => {
            let air_name = stark_info["name"].as_str().unwrap_or("unknown");
            compare_pil2circom_outputs(air_name, &js_output, &rust_output);
        }
        Err(e) => {
            tracing::warn!("pil2circom: Rust implementation failed: {e:#}");
        }
    }

    Ok(js_output)
}

// ── gen_circom ───────────────────────────────────────────────────────────────

/// Options controlling recursive/final circom generation.
#[derive(Debug, Default)]
pub struct GenCircomOptions {
    pub airgroup_id: Option<u64>,
    pub has_compressor: bool,
    pub has_recursion: bool,
    /// Selects the final-circuit template; the template path conveys this too,
    /// so this flag is informational and is not forwarded as a CLI flag.
    pub is_final: bool,
}

/// All inputs for a single `gen_circom` call.
pub struct GenCircomInput<'a> {
    /// EJS/Tera template path relative to the stark-recurser package root
    /// (e.g. `"src/vadcop/templates/final.circom.ejs"` or
    /// `"vadcop/final.circom.tera"`).
    pub template_name: &'a str,
    /// StarkInfo JSON objects, one per AIR.
    pub stark_infos: &'a [Value],
    /// The vadcop / global info JSON.
    pub vadcop_info: &'a Value,
    /// Verifier circom filenames (base names, not full paths).
    pub verifier_filenames: &'a [String],
    /// Per-airgroup, per-air constant roots for basic (recursive1) verification.
    pub basic_verification_keys: &'a [Vec<Vec<String>>],
    /// Per-airgroup constant roots for aggregation (recursive2) verification.
    pub agg_verification_keys: &'a [Vec<String>],
    /// Public inputs (currently unused by the script; reserved for future use).
    pub publics: &'a [Value],
    pub options: &'a GenCircomOptions,
}

/// Generate a circom circuit using the in-process Rust generator.
///
/// Returns an error if the template is not implemented.
pub fn gen_circom(input: &GenCircomInput<'_>) -> Result<String> {
    let rust_opts = CircomGenOptions {
        airgroup_id: input.options.airgroup_id.map(|x| x as usize),
        has_compressor: input.options.has_compressor,
        has_recursion: input.options.has_recursion,
        is_final: input.options.is_final,
    };
    let rust_input = GenCircomCircuitInput {
        template_name: input.template_name,
        stark_infos: input.stark_infos,
        vadcop_info: input.vadcop_info,
        verifier_filenames: input.verifier_filenames,
        basic_vk: input.basic_verification_keys,
        agg_vk: input.agg_verification_keys,
        publics: input.publics,
        options: &rust_opts,
    };
    gen_circom_circuit(&rust_input)
}
