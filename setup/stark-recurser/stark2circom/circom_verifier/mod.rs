//! Per-air stark verifier circom generator (port of `pil2circom.js`).
//!
//! Generates a `stark_verifier.circom` from `starkInfo` + `verifierInfo`.
//! Supports GL (Goldilocks/Poseidon2) and BN128 (SHA256/Keccak) hash types.
//!
//! Tera templates live in `circom_verifier/tera/`, circom library files in
//! `circom_verifier/circuits.gl/` and `circom_verifier/circuits.bn128/`.
//!
//! Entry points:
//! - [`gen_stark_verifier`]         — dispatches by `verificationHashType`
//! - [`gen_stark_verifier_gl`]      — Goldilocks (Poseidon1 or Poseidon2, selected by `opts.hash`)
//! - [`gen_stark_verifier_bn128`]   — BN128 hash

pub(crate) mod gl_field;
pub(crate) mod unroll_code;
pub(crate) mod expressions_chunks;
pub(crate) mod transcript_bn128;
pub(crate) mod gl;
pub(crate) mod bn128;
pub(crate) mod domain_zi;

use anyhow::{bail, Result};
use serde_json::Value;

pub use gl::Pil2CircomOptions;
pub(crate) use gl::gen_stark_verifier_gl;
pub(crate) use bn128::gen_stark_verifier_bn128;
pub use bn128::gen_stark_verifier_bn128_blake3;

/// Whether the air has a fixed-column group. A starkinfo without `nConstants` is malformed, not zero-fixed.
pub fn has_constants(si: &Value) -> Result<bool> {
    match si["nConstants"].as_u64() {
        Some(n) => Ok(n > 0),
        None => bail!("starkinfo has no numeric nConstants"),
    }
}

/// `has_constants(si)`, once a zero-fixed air is known to be verified in vadcop (`input_challenges`) mode with no rootC.
pub(crate) fn check_const_root(si: &Value, const_root: Option<&[String; 4]>, opts: &Pil2CircomOptions) -> Result<bool> {
    let has = has_constants(si)?;
    // the self-contained transcript seeds with rootC, so a zero-fixed air is verified only in vadcop mode
    if !has && (const_root.is_some() || opts.verkey_input || !opts.input_challenges) {
        bail!(
            "the air has no fixed columns (nConstants = 0): its verifier needs input_challenges, \
             no const root and no verkey input"
        );
    }
    Ok(has)
}

/// Generate a stark verifier circom (GL or BN128, selected by
/// `stark_info.starkStruct.verificationHashType`).
///
/// Pass `const_root = None` when `opts.verkey_input` is `true`.
pub fn gen_stark_verifier(
    const_root: Option<&[String; 4]>,
    stark_info: &Value,
    verifier_info: &Value,
    opts: &Pil2CircomOptions,
) -> Result<String> {
    let hash_type = stark_info["starkStruct"]["verificationHashType"].as_str().unwrap_or("GL");
    match hash_type {
        "GL" => gen_stark_verifier_gl(const_root, stark_info, verifier_info, opts),
        "BN128" => gen_stark_verifier_bn128(const_root, stark_info, verifier_info, opts),
        other => bail!("gen_stark_verifier: unsupported hash type '{other}'"),
    }
}

#[cfg(test)]
mod tests {
    use super::has_constants;
    use serde_json::json;

    #[test]
    fn has_constants_requires_the_key() {
        assert!(!has_constants(&json!({"nConstants": 0})).unwrap());
        assert!(has_constants(&json!({"nConstants": 2})).unwrap());
        assert!(has_constants(&json!({})).is_err());
    }
}
