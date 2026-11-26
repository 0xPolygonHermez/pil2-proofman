//! The circom mains and the Solidity contracts. A main is data over static templates
//! (`helper_circuits/mains/`): its keys' dispatch
//! (`circom_verifier::key_dispatch`), the vadcop data `vd_*()` and a `component main` line. The
//! Solidity contracts are static too (`sol/`): [`gen_solidity`] and [`gen_iverifier`] fill in their
//! `__TOKEN__`s.

use anyhow::{bail, Result};
use serde_json::Value;

use super::get_sha256_inputs::{sha256_defs, ProgramVk};
use super::vadcop_inputs::{air_id_vars, vadcop_data};
use super::verify_global_constraints::gen_verify_global_constraints;
use super::CircomGenOptions;
use crate::stark2circom::circom_verifier::lib_params::parse_stark_info;
use proofman_common::proof_inputs::{proof_inputs, Origin};
use crate::stark2circom::circom_verifier::{key_dispatch, key_suffix};

// ── The static Solidity contracts ────────────────────────────────────────────

const VERIFIER_SOL: &str = include_str!("sol/verifier.sol");
const IVERIFIER_SOL: &str = include_str!("sol/iverifier.sol");

/// `src` with each `token` replaced by its value; every token must be in `src`.
fn fill(src: &str, tokens: &[(&str, &str)]) -> String {
    tokens.iter().fold(src.to_string(), |out, (token, value)| {
        assert!(out.contains(token), "the static contract has no {token}");
        out.replace(token, value)
    })
}

// ── circom mains ──────────────────────────────────────────────────────────────

/// A main's head: pragmas, the verifier files and `lib`, the keys' dispatch.
fn head(files: &[String], lib: &str, stark_infos: &[&Value], gl: bool, custom_pragma: bool) -> String {
    let mut out = String::from("pragma circom 2.1.0;\n");
    if custom_pragma {
        out += "pragma custom_templates;\n";
    }
    out += "\n";
    out += &files.iter().map(|f| format!("include \"{f}\";\n")).collect::<String>();
    out += &format!("include \"{lib}\";\n\n");
    let sfxs: Vec<String> = stark_infos.iter().map(|si| if gl { key_suffix(si) } else { String::new() }).collect();
    out + &key_dispatch(&sfxs, gl)
}

/// `component main {public [..]} = <call>;`.
fn main_line(publics: &[&str], call: &str) -> String {
    if publics.is_empty() {
        format!("\ncomponent main = {call};\n")
    } else {
        format!("\ncomponent main {{public [{}]}} = {call};\n", publics.join(", "))
    }
}

const GLOBALS: [&str; 4] = ["publics", "proofValues", "globalChallenge", "rootCAgg"];
const SV: [&str; 5] =
    ["sv_circuitType", "sv_aggregatedProofs", "sv_aggregationTypes", "sv_airgroupvalues", "sv_stage1Hash"];

/// The recursivef circuit: its verifier on the proof, rootC (and the publics) public.
pub fn gen_recursivef(stark_info: &Value, verifier_filenames: &[String]) -> Result<String> {
    let vd = vadcop_data(&Value::Null, &[], &[], 0, 0)?;
    Ok(head(verifier_filenames, "mains/recursivef.circom", &[stark_info], true, true)
        + &vd
        + &main_line(&["rootC", "publics"], "Recursivef()"))
}

/// The final circuit: the recursivef's BN128 verifier and the publics hash (`FinalSnark`, static).
/// Its witness is computed from a JSON zkin addressed by input name (`proof2json`, `gl_proof_zkin`),
/// so the main takes every proof input by its name and flattens them into `FinalSnark`'s proof.
pub fn gen_recursion_final(
    stark_info: &Value,
    verifier_filenames: &[String],
    publics: Option<&Value>,
) -> Result<String> {
    let sha = sha256_defs(publics)?;
    let info = parse_stark_info(stark_info)?;
    // rootCVadcopFinal(4) | is_vadcop_final_proof(1) | the program's publics.
    if info.n_publics < 5 {
        bail!("gen_recursion_final: the recursivef has {} publics, fewer than rootC and the flag", info.n_publics);
    }
    if sha.n_publics + 5 != info.n_publics as usize {
        bail!(
            "gen_recursion_final: the publics info has {} publics, the recursivef {} past rootC and the flag",
            sha.n_publics,
            info.n_publics - 5
        );
    }
    let (mut decl, mut wire, mut at) = (String::new(), String::new(), 0u64);
    for i in proof_inputs(&info).map_err(anyhow::Error::msg)? {
        let shape: String = i.dims.iter().map(|d| format!("[{d}]")).collect();
        decl += &format!("    signal input {}{shape};\n", i.name);
        match i.origin {
            Origin::Publics => wire += &format!("    f.publics <== {};\n", i.name),
            Origin::ProofValues => bail!("gen_recursion_final: the final circuit takes no proof values"),
            Origin::Proof | Origin::Unread => {
                // Row-major flattening: nested loops over the dims, the flat index from the loop variables.
                let vars: Vec<String> = (0..i.dims.len()).map(|d| format!("i{d}")).collect();
                let mut flat = String::from("0");
                for (v, d) in vars.iter().zip(&i.dims) {
                    flat = format!("({flat}) * {d} + {v}");
                }
                let idx: String = vars.iter().map(|v| format!("[{v}]")).collect();
                let mut line = format!("f.proof[{at} + {flat}] <== {}{idx};", i.name);
                for (v, d) in vars.iter().zip(&i.dims).rev() {
                    line = format!("for (var {v} = 0; {v} < {d}; {v}++) {{ {line} }}");
                }
                wire += &format!("    {line}\n");
                at += i.words();
            }
        }
    }
    Ok(head(verifier_filenames, "mains/final_snark.circom", &[stark_info], false, info.stark_struct.merkle_tree_custom)
        + &sha.defs
        + &format!(
            "\ntemplate Main() {{\n    signal output publicsHash;\n{decl}\n    component f = FinalSnark({}, {}, {});\n{wire}    publicsHash <== f.publicsHash;\n}}\n",
            sha.n_publics,
            sha.n_defs,
            sha.vk.map_or(-1, |vk| vk.pos as i64)
        )
        + &main_line(&[], "Main()"))
}

/// The compressed vadcop_final circuit. The incoming proof commits `[is_vadcop_final_proof | publics]`
/// as ONE public array; this proof re-exposes the flag-free rest as `publics_stripped`.
pub fn gen_final_compressed(
    stark_info: &Value,
    verifier_filenames: &[String],
    opts: &CircomGenOptions,
) -> Result<String> {
    let n_publics = stark_info["nPublics"].as_u64().unwrap_or(0) as usize;
    let has_publics = if opts.has_recursion { n_publics > 4 } else { n_publics > 0 };
    let (n, n_real) = if has_publics { (n_publics, n_publics.saturating_sub(1)) } else { (0, 0) };
    let vd = vadcop_data(&Value::Null, &[], &[], 0, 0)?;
    Ok(head(verifier_filenames, "mains/final_compressed.circom", &[stark_info], true, true)
        + &vd
        + &main_line(&[], &format!("FinalCompressed({n}, {n_real})")))
}

/// The compressor of a basic air.
pub fn gen_compressor(stark_info: &Value, verifier_filenames: &[String], vadcop_info: &Value) -> Result<String> {
    let (ag, air) = air_id_vars(stark_info)?;
    let vd = vadcop_data(vadcop_info, &[], &[], ag, air)?;
    Ok(head(verifier_filenames, "mains/compressor.circom", &[stark_info], true, true)
        + &vd
        + &main_line(&GLOBALS[..3], "Compressor()"))
}

/// A recursive1: one basic proof, one compressor proof, or `batch_size > 1` proofs of one air.
pub fn gen_recursive1(
    stark_info: &Value,
    verifier_filenames: &[String],
    vadcop_info: &Value,
    opts: &CircomGenOptions,
) -> Result<String> {
    let k = opts.batch_size.max(1);
    if k > super::MAX_RECURSIVE1_BATCH {
        bail!("gen_recursive1: batch_size {k} exceeds the cap of {}", super::MAX_RECURSIVE1_BATCH);
    }
    let (ag, air) = air_id_vars(stark_info)?;
    let vd = vadcop_data(vadcop_info, &[], &[], ag, air)?;
    // Only a single compressor slot's values are public inputs; batched slots stay private (recursive2's nPublics).
    let main = match (k, opts.has_compressor) {
        (1, false) => main_line(&GLOBALS, "Recursive1()"),
        (1, true) => main_line(&[&SV[..], &GLOBALS[..]].concat(), "Recursive1Compressed()"),
        (k, c) => main_line(&GLOBALS, &format!("Recursive1Batched({k}, {})", u8::from(c))),
    };
    Ok(head(verifier_filenames, "mains/recursive1.circom", &[stark_info], true, true) + &vd + &main)
}

/// A recursive2 over `opts.agg_arity` proofs of airgroup `airgroup_id`.
pub fn gen_recursive2(
    stark_info: &Value,
    verifier_filenames: &[String],
    vadcop_info: &Value,
    airgroup_id: usize,
    basic_vk: &[Vec<String>],
    opts: &CircomGenOptions,
) -> Result<String> {
    // `pub` in a library crate, so an external caller reaches this ahead of the CLI's `--agg-arity` check.
    let agg_arity = opts.agg_arity;
    if !proofman_common::global_info::is_valid_aggregation_arity(agg_arity) {
        bail!(
            "gen_recursive2: unsupported aggregation arity {agg_arity}; valid values: {:?}",
            proofman_common::global_info::VALID_AGGREGATION_ARITIES
        );
    }
    let mut basics = vec![Vec::new(); airgroup_id + 1];
    basics[airgroup_id] = basic_vk.to_vec();
    let vd = vadcop_data(vadcop_info, &basics, &[], airgroup_id as u64, 0)?;
    Ok(head(verifier_filenames, "mains/recursive2.circom", &[stark_info], true, true)
        + &vd
        + &main_line(&GLOBALS, &format!("Recursive2({agg_arity})")))
}

/// The vadcop_final over one recursive2 proof per airgroup, and the global constraints.
pub fn gen_vadcop_final(
    stark_infos: &[Value],
    verifier_filenames: &[String],
    vadcop_info: &Value,
    basic_vk: &[Vec<Vec<String>>],
    agg_vk: &[Vec<String>],
) -> Result<String> {
    let n_ag = vadcop_info["aggTypes"].as_array().map_or(0, Vec::len);
    if stark_infos.len() != n_ag.max(1) || agg_vk.len() < n_ag || basic_vk.len() < n_ag {
        bail!(
            "gen_vadcop_final: {n_ag} airgroups, {} starkinfos, {} recursive2 and {} recursive1 key sets",
            stark_infos.len(),
            agg_vk.len(),
            basic_vk.len()
        );
    }
    let sis: Vec<&Value> = stark_infos.iter().collect();
    let vd = vadcop_data(vadcop_info, basic_vk, agg_vk, 0, 0)?;
    Ok(head(verifier_filenames, "mains/vadcop_final.circom", &sis, true, true)
        + &vd
        + "\n"
        + &gen_verify_global_constraints(vadcop_info)?
        + &main_line(&GLOBALS[..1], &format!("VadcopFinal({n_ag})")))
}

// ── Solidity contracts ────────────────────────────────────────────────────────

/// The verifier of the final SNARK that the project's Solidity verifier ([`gen_solidity`]) extends:
/// the contract that checks the final circuit's proof, where it is, and the proof its `verifyProof`
/// takes, with the publics hash as its one public.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SnarkVerifier {
    /// snarkjs's `FflonkVerifier`, in `FflonkVerifier.sol` beside the project's: a proof of 24
    /// `bytes32`.
    Fflonk,
    /// snarkjs's `PlonkVerifier`, in `PlonkVerifier.sol` beside the project's: a proof of 24
    /// `uint256`.
    Plonk,
    /// pilfflonk's `PilfflonkVerifier`, the verifier of the final circuit's vkey
    /// (`pilfflonk.verifier.sol`): a proof of `words` `bytes32`, the vkey's calldata
    /// (pilfflonk/docs/formats.md#calldata), whose number depends on the vkey.
    Pilfflonk {
        /// The path the project's verifier imports `pilfflonk.verifier.sol` from.
        source: String,
        /// The words of the `proof` argument of its `verifyProof`.
        words: u64,
    },
}

impl SnarkVerifier {
    /// The contract's name, without `Verifier`.
    fn name(&self) -> &'static str {
        match self {
            SnarkVerifier::Fflonk => "Fflonk",
            SnarkVerifier::Plonk => "Plonk",
            SnarkVerifier::Pilfflonk { .. } => "Pilfflonk",
        }
    }

    /// How the interface names the inherited verifier.
    fn label(&self) -> &'static str {
        match self {
            SnarkVerifier::Fflonk => "FFLONK",
            SnarkVerifier::Plonk => "PLONK",
            SnarkVerifier::Pilfflonk { .. } => "pilfflonk",
        }
    }

    /// The path the project's verifier imports the contract from.
    fn source(&self) -> String {
        match self {
            SnarkVerifier::Fflonk | SnarkVerifier::Plonk => format!("./{}Verifier.sol", self.name()),
            SnarkVerifier::Pilfflonk { source, .. } => source.clone(),
        }
    }

    /// The statement that decodes `proofBytes` into `proofDecoded`, the `proof` of `verifyProof`.
    fn proof_decode(&self) -> String {
        let (word, words) = match self {
            SnarkVerifier::Fflonk => ("bytes32", 24),
            SnarkVerifier::Plonk => ("uint256", 24),
            SnarkVerifier::Pilfflonk { words, .. } => ("bytes32", *words),
        };
        format!("{word}[{words}] memory proofDecoded = abi.decode(proofBytes, ({word}[{words}]));")
    }
}

/// Port of `src/recursion/contracts/verifier.sol.ejs`: the project's verifier, which extends
/// `verifier`. `version` is what the contract's `VERSION()` returns.
pub fn gen_solidity(
    name: &str,
    version: &str,
    root_c: &[u64; 4],
    publics: Option<&Value>,
    verifier: &SnarkVerifier,
) -> Result<String> {
    let camel = capitalise(name);

    let vk = sha256_defs(publics)?.vk;
    let packed = match vk {
        Some(ProgramVk { first: true, .. }) => "programVK, publicValues, rootCVadcopFinal",
        Some(ProgramVk { first: false, .. }) => "publicValues, programVK, rootCVadcopFinal",
        None => "publicValues, rootCVadcopFinal",
    };
    let has_program_vk = vk.is_some();
    let root_c = root_c.map(|r| r.to_string());
    Ok(fill(
        VERIFIER_SOL,
        &[
            ("__SNARK_SOURCE__", &verifier.source()),
            ("__SNARK__", verifier.name()),
            ("__NAME__", &camel),
            ("__VERSION__", version),
            ("__ROOT_C_0__", &root_c[0]),
            ("__ROOT_C_1__", &root_c[1]),
            ("__ROOT_C_2__", &root_c[2]),
            ("__ROOT_C_3__", &root_c[3]),
            ("__PROGRAM_VK_PARAM__", if has_program_vk { "bytes32 programVK,\n        " } else { "" }),
            ("__PROGRAM_VK_ARG__", if has_program_vk { "programVK, " } else { "" }),
            ("__PACKED__", packed),
            ("__PROOF_DECODE__", &verifier.proof_decode()),
        ],
    ))
}

/// Port of `src/recursion/contracts/iverifier.sol.ejs`.
pub fn gen_iverifier(name: &str, publics: Option<&Value>, verifier: &SnarkVerifier) -> Result<String> {
    let camel = capitalise(name);

    let (pin, pin_value, fold_tree, vk_doc, vk_param) = if sha256_defs(publics)?.vk.is_some() {
        (
            "Both keys MUST be constants",
            "values",
            " For an aggregated proof both are the recurser's verkey, and pinning\n    /// them together is what \
             extends that recurser's leaf allow-list over the whole\n    /// fold tree.",
            "/// @param programVK The identity committed by the proof: the ROM root of the\n    /// RISC-V program for \
             a plain proof, the recursion domain for an aggregated one.\n    ",
            "bytes32 programVK,\n        ",
        )
    } else {
        ("The root MUST be a constant", "a value", "", "", "")
    };
    Ok(fill(
        IVERIFIER_SOL,
        &[
            ("__SNARK_LABEL__", verifier.label()),
            ("__NAME__", &camel),
            ("__PIN_VALUE__", pin_value),
            ("__PIN__", pin),
            ("__FOLD_TREE__", fold_tree),
            ("__PROGRAM_VK_DOC__", vk_doc),
            ("__PROGRAM_VK_PARAM__", vk_param),
        ],
    ))
}

fn capitalise(s: &str) -> String {
    let mut chars = s.chars();
    match chars.next() {
        None => String::new(),
        Some(c) => c.to_uppercase().to_string() + chars.as_str(),
    }
}

#[cfg(test)]
mod tests {
    use std::fs;
    use std::path::Path;
    use std::process::Command;

    use super::*;
    use crate::stark2circom::circom_verifier::{gen_stark_verifier_bn128, Pil2CircomOptions};
    use crate::plonk2pil::r1cs::types::read_r1cs_from_bytes;
    use proofman_fields::Bn128;
    use serde_json::json;

    /// A recursivef starkinfo, whose trees are custom or not, with what the final circuit and the
    /// head of its verifier read.
    pub(super) fn recursivef_stark_info(merkle_tree_custom: bool) -> Value {
        json!({
            "keyVersion": proofman_common::KEY_VERSION,
            "starkStruct": {
                "verificationHashType": "BN128",
                "numQueries": 4,
                "nBits": 6,
                "nBitsExt": 10,
                "grindingBitsQueries": 0,
                "merkleTreeArity": 4,
                "merkleTreeCustom": merkle_tree_custom,
                "lastLevelVerification": 0,
                "hashCommits": false,
                "transcriptArity": 4,
                "foldingFactors": [4],
                "logDegrees": [6, 2],
                "logDomainSizes": [10, 6]
            },
            "nStages": 2,
            "qDim": 3,
            "cExpId": 0,
            "security": {"regime": "test"},
            "nPublics": 5,
            "nConstants": 3,
            "evMap": [{"type": "const", "id": 0, "prime": 0, "openingPos": 0}, {"type": "cm", "id": 0, "prime": 0, "openingPos": 0}],
            "cmPolsMap": [{"name": "q", "stage": 3, "stagePos": 0, "stageId": 0, "dim": 3}],
            "customCommits": [],
            "customCommitsMap": [],
            "challengesMap": [
                {"name": "std_vc", "stage": 3, "stageId": 0}, {"name": "xi", "stage": 4, "stageId": 0},
                {"name": "vf1", "stage": 5, "stageId": 0}, {"name": "vf2", "stage": 5, "stageId": 1}
            ],
            "boundaries": [{"name": "everyRow"}],
            "airgroupValuesMap": [],
            "airValuesMap": [],
            "proofValuesMap": [],
            "mapSectionsN": {"cm1": 2, "cm2": 2, "cm3": 2},
            "openingPoints": [0],
            "qDeg": 1
        })
    }

    /// The final circuit around a recursivef whose trees are custom or not.
    fn final_circuit(merkle_tree_custom: bool) -> String {
        let verifiers = ["recursivef.verifier.circom".to_string()];
        gen_recursion_final(&recursivef_stark_info(merkle_tree_custom), &verifiers, Some(&Value::Null)).unwrap()
    }

    /// A custom-tree verifier wants the pragma on the main too; otherwise there is none.
    #[test]
    fn final_pragma_follows_custom_trees() {
        let out = final_circuit(true);
        assert!(
            out.starts_with(
                "pragma circom 2.1.0;\npragma custom_templates;\n\ninclude \"recursivef.verifier.circom\";\n"
            ),
            "out:\n{out}"
        );
        let out = final_circuit(false);
        assert!(out.starts_with("pragma circom 2.1.0;\n\ninclude \"recursivef.verifier.circom\";\n"), "out:\n{out}");
        assert!(!out.contains("custom_templates"), "out:\n{out}");
    }

    /// The final circuit's main: the publics definitions as data, the static FinalSnark.
    #[test]
    fn recursion_final_is_data_over_final_snark() {
        let si = recursivef_stark_info(false);
        let out = gen_recursion_final(&si, &["v.circom".into()], None).unwrap();
        assert!(out.contains("include \"mains/final_snark.circom\";"), "{out}");
        assert!(out.contains("function vd_DEFS() { return [0]; }"), "{out}");
        assert!(
            out.contains("component f = FinalSnark(0, 0, -1);") && out.ends_with("\ncomponent main = Main();\n"),
            "{out}"
        );
    }

    /// The main declares exactly the keys `proof2json` (after `dropEmpty`) and gen_recursivef_proof's
    /// publics emit: here with no grinding (the nonce is still sent), an empty stage 2 (its path is
    /// sent, its values are not) and one last level.
    #[test]
    fn final_main_declares_the_proof2json_keys() {
        let mut si = recursivef_stark_info(false);
        si["mapSectionsN"]["cm2"] = json!(0);
        si["starkStruct"]["lastLevelVerification"] = json!(1);
        let out = gen_recursion_final(&si, &["v.circom".into()], None).unwrap();
        let decls: Vec<&str> = out.lines().filter_map(|l| l.trim().strip_prefix("signal input ")).collect();
        assert_eq!(
            decls,
            [
                "publics[5];",
                "root1;",
                "root2;",
                "root3;",
                "evals[2][3];",
                "s0_valsC[4][3];",
                "s0_siblingsC[4][4][4];",
                "s0_last_levelsC[4];",
                "s0_vals1[4][2];",
                "s0_siblings1[4][4][4];",
                "s0_last_levels1[4];",
                "s0_siblings2[4][4][4];",
                "s0_last_levels2[4];",
                "s0_vals3[4][2];",
                "s0_siblings3[4][4][4];",
                "s0_last_levels3[4];",
                "s1_root;",
                "s1_vals[4][48];",
                "s1_siblings[4][2][4];",
                "s1_last_levels[4];",
                "finalPol[64][3];",
                "nonce;",
            ]
        );
    }

    /// publics_info with an output and a program VK, first or last.
    fn out_and_vk(vk_first: bool) -> Value {
        let out = json!({"name": "out", "initialPos": if vk_first { 4 } else { 0 }, "nValues": 1, "chunks": [1, 64]});
        let vk = json!({"name": "vk", "initialPos": if vk_first { 0 } else { 1 }, "nValues": 4, "chunks": [1, 64], "verificationKey": true});
        let defs = if vk_first { [vk, out] } else { [out, vk] };
        json!({"nPublics": 5, "definitions": defs})
    }

    /// FinalSnark needs rootC, the flag and as many publics as publics_info; it binds a program VK
    /// where it sits, and refuses one it cannot bind.
    #[test]
    fn final_binds_the_program_vk_where_it_sits() {
        let mut si = recursivef_stark_info(false);
        si["nPublics"] = json!(4);
        assert!(gen_recursion_final(&si, &["v.circom".into()], None).unwrap_err().to_string().contains("publics"));
        si["nPublics"] = json!(10);
        let err = gen_recursion_final(&si, &["v.circom".into()], None).unwrap_err().to_string();
        assert!(err.contains("0 publics, the recursivef 5"), "{err}");
        for (vk_first, pos) in [(true, ", 0);"), (false, ", 1);")] {
            let out = gen_recursion_final(&si, &["v.circom".into()], Some(&out_and_vk(vk_first))).unwrap();
            assert!(out.lines().any(|l| l.contains("FinalSnark(5, 2") && l.ends_with(pos)), "{out}");
        }
        let mut middle = out_and_vk(false);
        middle["definitions"].as_array_mut().unwrap().push(json!({"name": "x", "initialPos": 5, "nValues": 0}));
        let mut halves = out_and_vk(false);
        halves["definitions"][1]["chunks"] = json!([2, 32]);
        let mut two = out_and_vk(true);
        two["definitions"][1]["verificationKey"] = json!(true);
        for (publics, msg) in
            [(middle, "first or the last"), (halves, "chunks [1, 64]"), (two, "second verificationKey")]
        {
            let err = gen_recursion_final(&si, &["v.circom".into()], Some(&publics)).unwrap_err().to_string();
            assert!(err.contains(msg), "{err}");
        }
    }

    /// The project's verifier of a program whose publics end with its verification key, extending
    /// `verifier`.
    fn project_verifier(verifier: &SnarkVerifier) -> String {
        gen_solidity("build", "v0", &[1, 2, 3, u64::MAX], Some(&out_and_vk(false)), verifier).unwrap()
    }

    /// The lines of `verifier`'s project verifier that are not FFLONK's.
    fn lines_unlike_fflonk(verifier: &SnarkVerifier) -> Vec<String> {
        let fflonk = project_verifier(&SnarkVerifier::Fflonk);
        let out = project_verifier(verifier);
        assert_eq!(out.lines().count(), fflonk.lines().count(), "out:\n{out}");
        out.lines()
            .zip(fflonk.lines())
            .filter(|(line, other)| line != other)
            .map(|(line, _)| line.trim().to_string())
            .collect()
    }

    /// snarkjs's verifiers are beside the project's, and take a proof of 24 words.
    #[test]
    fn the_snarkjs_verifiers_are_imported_from_beside_the_contract() {
        let fflonk = project_verifier(&SnarkVerifier::Fflonk);
        for line in [
            "import {FflonkVerifier} from \"./FflonkVerifier.sol\";",
            "contract BuildVerifier is FflonkVerifier, IBuildVerifier {",
            "bytes32[24] memory proofDecoded = abi.decode(proofBytes, (bytes32[24]));",
            "bool success = this.verifyProof(proofDecoded, [publicValuesDigest]);",
        ] {
            assert!(fflonk.contains(line), "{line} missing from:\n{fflonk}");
        }
        assert_eq!(
            lines_unlike_fflonk(&SnarkVerifier::Plonk),
            [
                "import {PlonkVerifier} from \"./PlonkVerifier.sol\";",
                "contract BuildVerifier is PlonkVerifier, IBuildVerifier {",
                "uint256[24] memory proofDecoded = abi.decode(proofBytes, (uint256[24]));",
            ]
        );
    }

    /// pilfflonk's verifier is imported from where the key has it, and takes a proof of the words of
    /// the vkey's calldata; the rest of the contract is FFLONK's.
    #[test]
    fn the_pilfflonk_verifier_is_imported_from_its_source_with_a_proof_of_its_words() {
        let source = "./provingKey/final/pilfflonk/pilfflonk.verifier.sol".to_string();
        assert_eq!(
            lines_unlike_fflonk(&SnarkVerifier::Pilfflonk { source, words: 47 }),
            [
                "import {PilfflonkVerifier} from \"./provingKey/final/pilfflonk/pilfflonk.verifier.sol\";",
                "contract BuildVerifier is PilfflonkVerifier, IBuildVerifier {",
                "bytes32[47] memory proofDecoded = abi.decode(proofBytes, (bytes32[47]));",
            ]
        );
    }

    /// The whole final circuit around a recursivef with custom trees, as setup-snark compiles it: the
    /// static library, the key's data and constraints, the main's dispatch. circom checks the include
    /// graph (CG04: `pragma custom_templates;` on every file on the way to a custom template), every
    /// template call and signal access, and the r1cs has the custom gates of the Poseidon tree and of
    /// the publics hash.
    ///
    /// Needs circomlib, from `npm install` in `setup/pil2-stark`; without it the test says so and
    /// passes.
    #[test]
    fn custom_final_compiles() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
        let circomlib = root.join("setup/pil2-stark/node_modules/circomlib/circuits");
        if !circomlib.is_dir() {
            eprintln!("skipped: {} not present (npm install in setup/pil2-stark)", circomlib.display());
            return;
        }
        let circom = root.join("setup/circom").join(if cfg!(target_os = "macos") { "circom_mac" } else { "circom" });
        let libraries = Path::new(env!("CARGO_MANIFEST_DIR")).join("stark2circom/circom_verifier");

        let mut stark_info = recursivef_stark_info(true);
        stark_info["nPublics"] = json!(10);
        let verifier_info = json!({ "constraintDag": { "version": 1, "field": "gl", "extDim": 3, "nodes": [[5, 1, 0, 0]], "roots": [[0, 0, "c"]] } });
        let root_c = ["1".to_string(), "0".to_string(), "0".to_string(), "0".to_string()];
        let verifier =
            gen_stark_verifier_bn128(Some(&root_c), &stark_info, &verifier_info, &Pil2CircomOptions::default())
                .unwrap();
        // FinalSnark binds the VK (vkPos >= 0) where it sits: first and last.
        for vk_first in [true, false] {
            let circuit = gen_recursion_final(
                &stark_info,
                &["recursivef.verifier.circom".to_string()],
                Some(&out_and_vk(vk_first)),
            )
            .unwrap();

            let dir = std::env::temp_dir().join(format!("custom_final_{}_{vk_first}", std::process::id()));
            let _ = fs::remove_dir_all(&dir);
            fs::create_dir_all(&dir).unwrap();
            fs::write(dir.join("recursivef.verifier.circom"), &verifier).unwrap();
            fs::write(dir.join("final.circom"), circuit).unwrap();

            // The libraries of `snark_setup.rs`'s final compile, in its order.
            let out = Command::new(&circom)
                .arg(dir.join("final.circom"))
                .args(["--O1", "--r1cs", "-o"])
                .arg(&dir)
                .arg("-l")
                .arg(libraries.join("helper_circuits"))
                .arg("-l")
                .arg(libraries.join("circuits.bn128"))
                .arg("-l")
                .arg(&circomlib)
                .output()
                .unwrap_or_else(|e| panic!("run {}: {e}", circom.display()));
            let stderr = String::from_utf8_lossy(&out.stderr).into_owned();
            assert!(out.status.success(), "circom failed:\n{stderr}{}", String::from_utf8_lossy(&out.stdout));

            let r1cs = read_r1cs_from_bytes::<Bn128>(&fs::read(dir.join("final.r1cs")).unwrap()).unwrap();
            let mut names: Vec<String> = r1cs.custom_gates.iter().map(|g| g.template_name.clone()).collect();
            names.sort();
            names.dedup();
            assert!(names.contains(&"PoseidonT".to_string()) && names.contains(&"Num2Bytes".to_string()), "{names:?}");
            let _ = fs::remove_dir_all(&dir);
        }
    }
}

#[cfg(test)]
mod main_tests {
    use super::*;

    /// Two airs in the group, so type 0 stays reserved for a null proof.
    fn vadcop() -> Value {
        serde_json::json!({
            "aggTypes": [[{"aggType": 0, "stage": 2}]], "air_groups": ["g"],
            "airs": [[{"name": "a"}, {"name": "b"}]], "curve": "None", "latticeSize": 368,
            "nPublics": 2, "numProofValues": [0], "hash": "blake3"
        })
    }

    fn stark_info() -> Value {
        serde_json::json!({ "airId": 1, "airgroupId": 0, "circuitType": 5 })
    }

    fn opts(k: usize, has_compressor: bool) -> CircomGenOptions {
        CircomGenOptions { airgroup_id: Some(0), has_compressor, has_recursion: false, agg_arity: 2, batch_size: k }
    }

    fn gen(k: usize, has_compressor: bool) -> String {
        gen_recursive1(&stark_info(), &["v.circom".into()], &vadcop(), &opts(k, has_compressor)).unwrap()
    }

    /// A main is data: the verifier's include, the keys' dispatch, the vadcop data and the main line.
    #[test]
    fn mains_are_data_over_static_templates() {
        let out = gen(1, false);
        assert!(out.contains("include \"v.circom\";\ninclude \"mains/recursive1.circom\";"), "{out}");
        assert!(out.contains("function svd_nBits(k) { return sv0_nBits(); }"), "{out}");
        assert!(out.contains("roots <== ConstraintRoots0()(evals, challenges"), "{out}");
        assert!(out.contains("function vd_airId() { return 1; }"), "{out}");
        let templates: Vec<&str> = out.lines().filter(|l| l.starts_with("template ")).collect();
        assert_eq!(templates, ["template ConstraintRootsAt(k) {"], "{out}");
        assert!(out
            .ends_with("component main {public [publics, proofValues, globalChallenge, rootCAgg]} = Recursive1();\n"));
    }

    #[test]
    fn each_recursive1_shape_has_its_template() {
        assert!(gen(1, true).ends_with("component main {public [sv_circuitType, sv_aggregatedProofs, sv_aggregationTypes, sv_airgroupvalues, sv_stage1Hash, publics, proofValues, globalChallenge, rootCAgg]} = Recursive1Compressed();\n"));
        assert!(gen(2, false).ends_with("= Recursive1Batched(2, 0);\n"));
        assert!(gen(3, true).ends_with("= Recursive1Batched(3, 1);\n"));
    }

    #[test]
    fn past_the_cap_it_refuses() {
        let o = opts(super::super::MAX_RECURSIVE1_BATCH + 1, false);
        assert!(gen_recursive1(&stark_info(), &["v.circom".into()], &vadcop(), &o).is_err());
    }

    /// Each recursive2 accepts only its own airgroup's types, and so does each final slot.
    #[test]
    fn recursive2_and_final_select_by_airgroup() {
        let vadcop = serde_json::json!({
            "aggTypes": [[{"aggType": 0, "stage": 2}], [{"aggType": 0, "stage": 2}]], "air_groups": ["g0", "g1"],
            "airs": [[{"name": "a"}, {"name": "b"}], [{"name": "c"}]], "curve": "None", "latticeSize": 368,
            "nPublics": 2, "numProofValues": [0], "hash": "blake3"
        });
        let vk = vec!["1".to_string(), "2".into(), "3".into(), "4".into()];
        let si = |ag: u64| serde_json::json!({ "airId": 0, "airgroupId": ag });
        let o = CircomGenOptions {
            airgroup_id: Some(1),
            has_compressor: false,
            has_recursion: true,
            agg_arity: 2,
            batch_size: 1,
        };
        let rec2 = gen_recursive2(&si(1), &["v.circom".into()], &vadcop, 1, std::slice::from_ref(&vk), &o).unwrap();
        assert!(rec2.contains("function vd_airgroupId() { return 1; }"), "{rec2}");
        assert!(
            rec2.contains("function vd_nBasics(i) { if (i == 0) { return 0; } if (i == 1) { return 1; } return 0; }"),
            "{rec2}"
        );
        assert!(rec2.ends_with("= Recursive2(2);\n"), "{rec2}");

        let fin = gen_vadcop_final(
            &[si(0), si(1)],
            &["v0.circom".into(), "v1.circom".into()],
            &vadcop,
            &[vec![vk.clone(), vk.clone()], vec![vk.clone()]],
            &[vk.clone(), vk],
        )
        .unwrap();
        assert!(fin.contains("include \"mains/vadcop_final.circom\";"), "{fin}");
        assert!(
            fin.contains("function svd_nBits(k) { if (k == 0) { return sv0_nBits(); } return sv1_nBits(); }"),
            "{fin}"
        );
        assert!(fin.contains("template GlobalConstraints(nAg, maxAgg, nPub, nPV, nCh) {"), "{fin}");
        assert!(fin.ends_with("component main {public [publics]} = VadcopFinal(2);\n"), "{fin}");
    }
}

#[cfg(test)]
mod solidity_version_tests {
    use super::*;

    fn zisk_publics() -> Value {
        serde_json::json!({
            "nPublics": 68,
            "definitions": [
                {"name": "rom_root", "initialPos": 0, "nValues": 4, "chunks": [1, 64], "verificationKey": true},
                {"name": "inputs", "initialPos": 4, "nValues": 64, "chunks": [1, 64]}
            ],
            "hasProgramVK": true
        })
    }

    #[test]
    fn the_verifier_returns_the_given_version() {
        let sol =
            gen_solidity("zisk", "v9.9.9-test", &[1, 2, 3, 4], Some(&zisk_publics()), &SnarkVerifier::Plonk).unwrap();
        assert!(sol.contains(r#"return "v9.9.9-test";"#), "{sol}");
        assert!(sol.contains("contract ZiskVerifier is PlonkVerifier, IZiskVerifier"));
        assert!(sol.contains("must be a constant of the calling contract"));
    }

    /// The interface carries the key-binding rules a caller needs to verify safely.
    #[test]
    fn the_interface_documents_how_to_pin_the_keys() {
        let isol = gen_iverifier("zisk", Some(&zisk_publics()), &SnarkVerifier::Plonk).unwrap();
        assert!(isol.contains("Both keys MUST be constants of the calling contract"), "{isol}");
        assert!(isol.contains("@param programVK"));
        assert!(isol.contains("@param rootCVadcopFinal"));
        assert!(isol.contains("the inherited raw PLONK verifier"));

        let fflonk = gen_iverifier("zisk", Some(&zisk_publics()), &SnarkVerifier::Fflonk).unwrap();
        assert!(fflonk.contains("the inherited raw FFLONK verifier"));

        // Without a program key there is one key to pin, and no programVK parameter.
        let no_vk = gen_iverifier("zisk", None, &SnarkVerifier::Plonk).unwrap();
        assert!(no_vk.contains("The root MUST be a constant of the calling contract"), "{no_vk}");
        assert!(!no_vk.contains("programVK"));
    }
}
