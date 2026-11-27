//! Per-air stark verifier circom generator (port of `pil2circom.js`).
//!
//! Generates a `stark_verifier.circom` from `starkInfo` + `verifierInfo`.
//! Supports GL (Poseidon1, Poseidon2, blake3) and BN128 (Poseidon, or blake3 over a GL STARK) verifiers.
//!
//! Both are the static library `helper_circuits/verifier/` (`StarkVerifierGL(k)`, `StarkVerifierBN128(k)`)
//! plus per key only data and constraints: `lib_gl` (FRI or STIR) and `lib_bn128` (FRI, Goldilocks
//! emulated over BN128). A key's verifier file holds its constraints; its data goes to the circuit's
//! main ([`KeyVerifier::data`], [`key_dispatch`]).
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

use anyhow::{anyhow, bail, Context, Result};
use pil_info::core::dag::format::ConstraintDag;
use pil_info::core::dag::lower_code::{ev_col_ref, LowerCtx, LowerTables};
use proofman_common::hash_family::HashFamily;
use proofman_common::{EvMapEType, PolMap, StarkInfo};
use pil_info::FieldCfg;
use serde_json::Value;

pub(crate) use lib_gl::{gen_stark_verifier_gl, key_suffix};
pub(crate) use lib_bn128::gen_stark_verifier_bn128;
pub use lib_bn128::gen_stark_verifier_bn128_blake3;
pub use lib_params::KeyData;

/// A key's verifier: the circom file (its family's include and its constraints) and its data, which
/// the main of the circuit that verifies it carries as `key_<name>(k)`.
#[derive(Debug)]
pub struct KeyVerifier {
    pub circom: String,
    pub data: KeyData,
}

#[derive(Debug, Clone)]
pub struct Pil2CircomOptions {
    pub verkey_input: bool,
    pub enable_input: bool,
    pub input_challenges: bool,
    pub fri_queries_batch_size: Option<usize>,
    pub hash: HashFamily,
}

impl Default for Pil2CircomOptions {
    fn default() -> Self {
        Self {
            verkey_input: false,
            enable_input: false,
            input_challenges: false,
            fri_queries_batch_size: None,
            hash: HashFamily::DEFAULT,
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
) -> Result<KeyVerifier> {
    let hash_type = stark_info["starkStruct"]["verificationHashType"].as_str();
    match hash_type.context("starkStruct.verificationHashType missing")? {
        "GL" => gen_stark_verifier_gl(const_root, stark_info, verifier_info, opts),
        "BN128" => gen_stark_verifier_bn128(const_root, stark_info, verifier_info, opts),
        other => bail!("gen_stark_verifier: unsupported hash type '{other}'"),
    }
}

/// The main's access to the keys it verifies, key `k` being `keys[k]`: their data `key_<name>(k)` and,
/// for GL keys, `ConstraintRootsAt(k)` (a BN128 key's `ConstraintEval` is called directly).
pub fn key_dispatch(keys: &[KeyData], gl: bool) -> String {
    if keys.is_empty() {
        return String::new();
    }
    let mut out = String::from(
        "// The data of the keys this circuit verifies (lib_params.rs): key_<name>(k) is key k's <name>.\n",
    );
    out += &lib_params::key_functions(keys);
    if gl {
        out += "
// Key k's constraints at xi.
template ConstraintRootsAt(k) {
    signal input evals[key_nEvals(k)][3];
    signal input challenges[key_nChallenges(k)][3];
    signal input publics[nonEmpty(key_nPublics(k))];
    signal input proofvalues[nonEmpty(key_nProofValues(k))][3];
    signal input airvalues[nonEmpty(key_nAirValues(k))][3];
    signal input airgroupvalues[nonEmpty(key_nAirgroupValues(k))][3];
    signal output roots[key_nRoots(k)][3];

";
        let call = |s: &str| {
            format!(
                "roots <== ConstraintRoots{s}()(evals, challenges, publics, proofvalues, airvalues, airgroupvalues);"
            )
        };
        // One key: its roots; more: one branch per key.
        match keys {
            [key] => out += &format!("    {}\n", call(&key.suffix)),
            _ => {
                for (k, key) in keys.iter().enumerate() {
                    out += &format!("    if (k == {k}) {{ {} }}\n", call(&key.suffix));
                }
            }
        }
        out += "}\n";
    }
    out
}

/// The validated constraint DAG of `vi` and the resolver of `si`.
pub(crate) fn dag_and_ctx(si: &StarkInfo, vi: &Value) -> Result<(ConstraintDag, LowerCtx)> {
    let dag = ConstraintDag::from_verifierinfo(vi, &FieldCfg::goldilocks()).map_err(|e| anyhow!("{e:?}"))?;
    let ctx = LowerCtx::new(lower_tables(si)?).map_err(|e| anyhow!("{e:?}"))?;
    Ok((dag, ctx))
}

/// The constraint DAG's resolver tables of a Goldilocks starkinfo.
pub fn lower_tables(si: &StarkInfo) -> Result<LowerTables> {
    let small = |x: u64| u32::try_from(x).with_context(|| format!("starkinfo value {x} past u32"));
    let ev_map = (si.ev_map.iter())
        .map(|e| {
            let ty = match e.type_ {
                EvMapEType::Cm => "cm",
                EvMapEType::Const => "const",
                EvMapEType::Custom => "custom",
            };
            let offset = *si.opening_points.get(e.opening_pos).context("evMap openingPos past openingPoints")?;
            ev_col_ref(ty, Some(small(e.commit_id)?), small(e.id)?, offset).context("evMap entry")
        })
        .collect::<Result<Vec<_>>>()?;
    let challenges = (si.challenges_map.iter().flatten())
        .map(|c| Ok((small(c.stage)?, small(c.stage_id)?, c.name.as_str())))
        .collect::<Result<Vec<_>>>()?;
    let stages = |m: &Option<Vec<PolMap>>| m.iter().flatten().map(|p| small(p.stage)).collect::<Result<Vec<u32>>>();
    let value_stages = [stages(&si.airvalues_map)?, stages(&si.proofvalues_map)?, stages(&si.airgroupvalues_map)?];
    LowerTables::new(ev_map, &challenges, small(si.n_publics)?, value_stages, &FieldCfg::goldilocks())
        .map_err(|e| anyhow!("{e:?}"))
}
