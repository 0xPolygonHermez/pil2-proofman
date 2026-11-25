//! The BN128 verifier (a recursivef's, in the final SNARK's circuit) of one key: its family's entry
//! (`verifier/bn128_open_<family>.circom`), data (`lib_params`) and constraints
//! (`ConstraintEval`). The verifier itself is the static `StarkVerifierBN128(k)`
//! (`helper_circuits/verifier/stark_verifier_bn128.circom`).

use anyhow::{bail, Context, Result};
use serde_json::Value;

use super::constraint_roots::constraint_eval_bn128;
use super::lib_params::{parse_stark_info, KeyData, KeyOpts};
use super::Pil2CircomOptions;

/// The BN128 verifier of a BN128 STARK (Poseidon trees, `verificationHashType` BN128).
pub fn gen_stark_verifier_bn128(
    const_root: Option<&[String; 4]>,
    stark_info: &Value,
    verifier_info: &Value,
    opts: &Pil2CircomOptions,
) -> Result<String> {
    gen(const_root, stark_info, verifier_info, opts, false)
}

/// The BN128 verifier of a Goldilocks STARK hashed with blake3: the recursivef of a blake3 key,
/// which the final SNARK's circuit verifies. Its arithmetic is the BN128 verifier's, its hashing the
/// Goldilocks verifier's: roots of four words, the GL transcript over circuits.gl's blake3 circuits
/// (field parts in `circuits.bn128/b3_field.circom`), binary Merkle paths with their last levels.
pub fn gen_stark_verifier_bn128_blake3(
    const_root: Option<&[String; 4]>,
    stark_info: &Value,
    verifier_info: &Value,
    opts: &Pil2CircomOptions,
) -> Result<String> {
    gen(const_root, stark_info, verifier_info, opts, true)
}

fn gen(
    const_root: Option<&[String; 4]>,
    si: &Value,
    vi: &Value,
    opts: &Pil2CircomOptions,
    blake3: bool,
) -> Result<String> {
    let info = parse_stark_info(si)?;
    let ss = &info.stark_struct;
    let custom = ss.merkle_tree_custom;
    let hash_type = &ss.verification_hash_type;
    if blake3 {
        if hash_type != "GL" || !opts.hash.is_blake3() {
            bail!("gen_stark_verifier_bn128_blake3: expected a Goldilocks STARK hashed with blake3");
        }
        if ss.merkle_tree_arity != 2 {
            bail!("gen_stark_verifier_bn128_blake3: blake3 trees are binary");
        }
        if !custom {
            bail!("gen_stark_verifier_bn128_blake3: the blake3 gates are custom templates (merkleTreeCustom)");
        }
    } else if hash_type != "BN128" {
        bail!("gen_stark_verifier_bn128: expected BN128 hash type, got '{hash_type}'");
    }
    ss.low_degree_test.fri().context("the BN128 verifier checks FRI, and this STARK opens with STIR")?;
    if !info.custom_commits.is_empty() {
        bail!("the BN128 verifier does not support custom commits");
    }
    // Its transcript absorbs none of them, so they would be free after the challenges.
    for (name, map) in [
        ("airvalues", &info.airvalues_map),
        ("airgroupvalues", &info.airgroupvalues_map),
        ("proofvalues", &info.proofvalues_map),
    ] {
        if map.as_ref().is_some_and(|m| !m.is_empty()) {
            bail!("the BN128 verifier does not support {name}: its transcript does not absorb them");
        }
    }
    if !blake3 && custom && ss.last_level_verification > 0 {
        bail!("lastLevelVerification > 0 with merkleTreeCustom: circuits.bn128/custom/merklehash.circom has no last-level templates");
    }
    let key = KeyData::new(
        &info,
        &KeyOpts {
            verkey_input: opts.verkey_input,
            enable_input: opts.enable_input,
            const_root: const_root.cloned(),
            ..Default::default()
        },
    )?;

    // The family's entry includes its field, hash, transcript and verifier files.
    let family = if blake3 {
        "blake3"
    } else if custom {
        "custom"
    } else {
        "poseidon"
    };
    let pragma = if custom { "pragma custom_templates;\n" } else { "" };
    let mut out = format!("pragma circom 2.1.0;\n{pragma}\ninclude \"verifier/bn128_open_{family}.circom\";\n\n");
    out += &key.functions("");
    let (dag, ctx) = super::dag_and_ctx(&info, vi)?;
    out += &format!("\n{}", constraint_eval_bn128(&dag, &ctx)?);
    Ok(out)
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use serde_json::json;

    pub(crate) fn stark_info(custom: bool, ll: u64) -> Value {
        json!({
            "keyVersion": proofman_common::KEY_VERSION,
            "starkStruct": {
                "verificationHashType": "BN128", "numQueries": 4, "nBits": 6, "nBitsExt": 10, "grindingBitsQueries": 0,
                "merkleTreeArity": 4, "merkleTreeCustom": custom, "lastLevelVerification": ll, "hashCommits": false,
                "transcriptArity": 4, "foldingFactors": [4], "logDegrees": [6, 2], "logDomainSizes": [10, 6]
            },
            "nStages": 2, "qDim": 3, "cExpId": 0, "security": {"regime": "test"}, "nPublics": 2, "nConstants": 2,
            "evMap": [{"type": "const", "id": 0, "prime": 0, "openingPos": 0}, {"type": "cm", "id": 0, "prime": 0, "openingPos": 0}],
            "cmPolsMap": [{"name": "q", "stage": 3, "stagePos": 0, "stageId": 0, "dim": 3}],
            "customCommits": [], "customCommitsMap": [],
            "challengesMap": [
                {"name": "std_vc", "stage": 3, "stageId": 0, "dim": 3}, {"name": "xi", "stage": 4, "stageId": 0, "dim": 3},
                {"name": "vf1", "stage": 5, "stageId": 0, "dim": 3}, {"name": "vf2", "stage": 5, "stageId": 1, "dim": 3}
            ],
            "boundaries": [{"name": "everyRow"}], "airgroupValuesMap": [], "airValuesMap": [], "proofValuesMap": [],
            "mapSectionsN": {"cm1": 0, "cm2": 0, "cm3": 3}, "openingPoints": [0], "qDeg": 1
        })
    }

    pub(crate) fn verifier_info() -> Value {
        json!({ "constraintDag": { "version": 1, "field": "gl", "extDim": 3, "nodes": [[5, 1, 0, 0]], "roots": [[0, 0, "c"]] } })
    }

    fn render(si: &Value) -> Result<String> {
        let root = ["7".to_string(), "0".into(), "0".into(), "0".into()];
        gen_stark_verifier_bn128(Some(&root), si, &verifier_info(), &Pil2CircomOptions::default())
    }

    #[test]
    fn per_key_output_is_includes_data_and_constraints() {
        let out = render(&stark_info(false, 0)).unwrap();
        assert!(
            out.starts_with("pragma circom 2.1.0;\n\ninclude \"verifier/bn128_open_poseidon.circom\";\n\nfunction "),
            "out:\n{out}"
        );
        assert!(out.contains("    tmp_0 <== evals[0];\n    signal {maxNum} tmp_1[3];\n    tmp_1 <== GLCMul()(tmp_0, Zh);\n\n    evalP <== tmp_1;\n}"), "out:\n{out}");
        assert!(out.contains("function sv_ROOT_C() { return [7, 0, 0, 0]; }"), "out:\n{out}");
        assert!(out.contains("function sv_OFF_ROOT() { return [0, 1, 2]; }"), "single-element roots:\n{out}");
        let templates: Vec<&str> = out.lines().filter(|l| l.starts_with("template ")).collect();
        assert_eq!(templates, ["template ConstraintEval() {"], "{out}");
        let custom = render(&stark_info(true, 0)).unwrap();
        assert!(
            custom.starts_with(
                "pragma circom 2.1.0;\npragma custom_templates;\n\ninclude \"verifier/bn128_open_custom.circom\";\n\n"
            ),
            "out:\n{custom}"
        );
    }

    #[test]
    fn last_levels_reach_the_data() {
        let out = render(&stark_info(false, 2)).unwrap();
        assert!(out.contains("function sv_lastLevels() { return 2; }"), "out:\n{out}");
        assert!(!out.contains("function sv_OFF_S0L() { return [-1, -1]; }"), "out:\n{out}");
        let none = render(&stark_info(false, 0)).unwrap();
        assert!(none.contains("function sv_OFF_S0L() { return [-1, -1]; }"), "out:\n{none}");
    }

    /// Custom trees of any arity (the sponge rate reads it), and a key with no fold step.
    #[test]
    fn custom_arity_and_no_fold_step() {
        let mut si = stark_info(true, 0);
        si["starkStruct"]["merkleTreeArity"] = json!(2);
        let out = render(&si).unwrap();
        assert!(out.contains("function sv_arity() { return 2; }"), "out:\n{out}");
        let mut si = stark_info(false, 0);
        si["starkStruct"]["logDomainSizes"] = json!([10]);
        si["starkStruct"]["logDegrees"] = json!([6]);
        si["starkStruct"]["foldingFactors"] = json!([]);
        let out = render(&si).unwrap();
        assert!(out.contains("function sv_nSteps() { return 1; }"), "out:\n{out}");
    }

    #[test]
    fn refuses_what_the_wrap_cannot_take() {
        let err = render(&stark_info(true, 2)).unwrap_err().to_string();
        assert!(err.contains("lastLevelVerification"), "{err}");
        let mut si = stark_info(false, 0);
        si["customCommits"] = json!([{"name": "rom", "publicValues": [{"idx": 0}], "stageWidths": [1]}]);
        assert!(render(&si).unwrap_err().to_string().contains("custom commits"));
        let mut si = stark_info(false, 0);
        si["starkStruct"]["verificationHashType"] = json!("GL");
        assert!(render(&si).unwrap_err().to_string().contains("GL"));
    }
}
