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

use anyhow::{bail, Result};
use serde_json::Value;

pub use gl::Pil2CircomOptions;
pub(crate) use gl::gen_stark_verifier_gl;
pub(crate) use bn128::gen_stark_verifier_bn128;

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
