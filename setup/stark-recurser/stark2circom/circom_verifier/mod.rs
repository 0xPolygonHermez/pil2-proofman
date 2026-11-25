//! Per-air stark verifier circom generator (port of `pil2circom.js`).
//!
//! Generates a `stark_verifier.circom` from `starkInfo` + `verifierInfo`.
//! Supports GL (Poseidon1, Poseidon2, blake3) and BN128 (Poseidon, or blake3 over a GL STARK) verifiers.
//!
//! Both are the static library `helper_circuits/verifier/` plus per-key data and constraints: `lib_gl`
//! (FRI or STIR) and `lib_bn128` (FRI, Goldilocks emulated over BN128); they share the transcript.
//! Field and hash circuits live in `circom_verifier/circuits.gl/` and `circom_verifier/circuits.bn128/`.
//!
//! Entry points:
//! - [`gen_stark_verifier`]         — dispatches by `verificationHashType`
//! - [`gen_stark_verifier_gl`]      — Goldilocks (Poseidon1, Poseidon2 or blake3, selected by `opts.hash`)
//! - [`gen_stark_verifier_bn128`]   — BN128 hash

pub(crate) mod gl_field;
pub(crate) mod lib_bn128;
pub(crate) mod lib_params;
pub(crate) mod constraint_roots;
pub(crate) mod lib_gl;

use anyhow::{anyhow, bail, Result};
use pil_info::core::dag::format::ConstraintDag;
use pil_info::core::dag::lower_code::{LowerCtx, LowerTables};
use pil_info::FieldCfg;
use serde_json::Value;

pub(crate) use lib_gl::gen_stark_verifier_gl;
pub(crate) use lib_bn128::gen_stark_verifier_bn128;
pub use lib_bn128::gen_stark_verifier_bn128_blake3;

#[derive(Debug, Clone)]
pub struct Pil2CircomOptions {
    pub skip_main: bool,
    pub verkey_input: bool,
    pub enable_input: bool,
    pub input_challenges: bool,
    pub fri_queries_batch_size: Option<usize>,
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
            hash: proofman_common::hash_family::DEFAULT_HASH_ID.to_string(),
        }
    }
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

/// The validated constraint DAG of `vi` and the resolver of `si`.
pub(crate) fn dag_and_ctx(si: &Value, vi: &Value) -> Result<(ConstraintDag, LowerCtx)> {
    let dag = ConstraintDag::from_verifierinfo(vi, &FieldCfg::goldilocks()).map_err(|e| anyhow!("{e:?}"))?;
    let ctx = LowerTables::from_starkinfo(si).and_then(LowerCtx::new).map_err(|e| anyhow!("{e:?}"))?;
    Ok((dag, ctx))
}

#[cfg(test)]
mod render_reference {
    /// The final circuit of a recursivef and its BN128 verifier:
    ///   BN_SI=<starkinfo> BN_VI=<verifierinfo> BN_VK=<verkey.json> BN_HASH=<family> BN_PUB=<publics_info.json> BN_OUT=<dir>
    #[test]
    #[ignore]
    fn render_bn128_final() {
        use crate::stark2circom::circuit_templates::templates::gen_recursion_final;
        let env = |k: &str| std::env::var(k).unwrap_or_else(|_| panic!("{k} not set"));
        let read = |k: &str| std::fs::read_to_string(env(k)).unwrap_or_else(|e| panic!("{k}: {e}"));
        let si: serde_json::Value = serde_json::from_str(&read("BN_SI")).unwrap();
        let vi: serde_json::Value = serde_json::from_str(&read("BN_VI")).unwrap();
        let vk: serde_json::Value = serde_json::from_str(&read("BN_VK")).unwrap();
        let w = |v: &serde_json::Value| {
            v.as_u64().map(|x| x.to_string()).unwrap_or_else(|| v.as_str().unwrap().to_string())
        };
        let root: [String; 4] = match vk.as_array() {
            Some(a) => [w(&a[0]), w(&a[1]), w(&a[2]), w(&a[3])],
            None => [w(&vk), "0".into(), "0".into(), "0".into()],
        };
        let hash = env("BN_HASH");
        let opts = super::Pil2CircomOptions { hash: hash.clone(), skip_main: true, ..Default::default() };
        let v = if hash == "blake3" {
            super::gen_stark_verifier_bn128_blake3(Some(&root), &si, &vi, &opts)
        } else {
            super::gen_stark_verifier(Some(&root), &si, &vi, &opts)
        }
        .unwrap();
        let f = gen_recursion_final(
            &si,
            &["recursivef.verifier.circom".to_string()],
            Some(&serde_json::from_str(&read("BN_PUB")).unwrap()),
        )
        .unwrap();
        let dir = std::path::PathBuf::from(env("BN_OUT"));
        std::fs::write(dir.join("recursivef.verifier.circom"), v).unwrap();
        std::fs::write(dir.join("final.circom"), f).unwrap();
    }
}
