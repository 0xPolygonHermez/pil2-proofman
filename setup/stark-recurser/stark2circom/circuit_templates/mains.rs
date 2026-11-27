//! The circom mains. A main is data over static templates (`helper_circuits/mains/`): its keys' data
//! and constraints dispatch (`circom_verifier::key_dispatch`), the vadcop data `vadcop_*()` and a
//! `component main` line.

use anyhow::{bail, Result};
use serde_json::Value;

use super::get_sha256_inputs::sha256_defs;
use super::vadcop_inputs::{air_id_vars, vadcop_data};
use super::CircomGenOptions;
use crate::stark2circom::circom_verifier::lib_params::parse_stark_info;
use proofman_common::proof_inputs::{proof_inputs, Origin};
use crate::stark2circom::circom_verifier::constraint_roots::global_constraints;
use crate::stark2circom::circom_verifier::{key_dispatch, KeyData};

/// A main's head: pragmas, the verifier files and `lib`, the keys' data and dispatch.
fn head(files: &[String], lib: &str, keys: &[KeyData], gl: bool, custom_pragma: bool) -> String {
    let mut out = String::from("pragma circom 2.1.0;\n");
    if custom_pragma {
        out += "pragma custom_templates;\n";
    }
    out += "\n";
    out += &files.iter().map(|f| format!("include \"{f}\";\n")).collect::<String>();
    out += &format!("include \"{lib}\";\n\n");
    out + &key_dispatch(keys, gl)
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
pub fn gen_recursivef(verifier_filenames: &[String], keys: &[KeyData]) -> Result<String> {
    let vd = vadcop_data(&Value::Null, &[], &[], 0, 0)?;
    Ok(head(verifier_filenames, "mains/recursivef.circom", keys, true, true)
        + &vd
        + &main_line(&["rootC", "publics"], "Recursivef()"))
}

/// The final circuit: the recursivef's BN128 verifier and the publics hash (`FinalSnark`, static).
/// Its witness is computed from a JSON zkin addressed by input name (`proof2json`, `gl_proof_zkin`),
/// so the main takes every proof input by its name and flattens them into `FinalSnark`'s proof.
pub fn gen_recursion_final(
    stark_info: &Value,
    verifier_filenames: &[String],
    keys: &[KeyData],
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
    Ok(head(verifier_filenames, "mains/final_snark.circom", keys, false, info.stark_struct.merkle_tree_custom)
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
    keys: &[KeyData],
    opts: &CircomGenOptions,
) -> Result<String> {
    let n_publics = stark_info["nPublics"].as_u64().unwrap_or(0) as usize;
    let has_publics = if opts.has_recursion { n_publics > 4 } else { n_publics > 0 };
    let (n, n_real) = if has_publics { (n_publics, n_publics.saturating_sub(1)) } else { (0, 0) };
    let vd = vadcop_data(&Value::Null, &[], &[], 0, 0)?;
    Ok(head(verifier_filenames, "mains/final_compressed.circom", keys, true, true)
        + &vd
        + &main_line(&[], &format!("FinalCompressed({n}, {n_real})")))
}

/// The compressor of a basic air.
pub fn gen_compressor(
    stark_info: &Value,
    verifier_filenames: &[String],
    keys: &[KeyData],
    vadcop_info: &Value,
) -> Result<String> {
    let (ag, air) = air_id_vars(stark_info)?;
    let vd = vadcop_data(vadcop_info, &[], &[], ag, air)?;
    // The compressor is recursive1 without its rootCAgg input.
    Ok(head(verifier_filenames, "mains/recursive1.circom", keys, true, true)
        + &vd
        + &main_line(&GLOBALS[..3], "Recursive1(0)"))
}

/// A recursive1: one basic proof, one compressor proof, or `batch_size > 1` proofs of one air.
pub fn gen_recursive1(
    stark_info: &Value,
    verifier_filenames: &[String],
    keys: &[KeyData],
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
        (1, false) => main_line(&GLOBALS, "Recursive1(1)"),
        (1, true) => main_line(&[&SV[..], &GLOBALS[..]].concat(), "Recursive1Compressed()"),
        (k, c) => main_line(&GLOBALS, &format!("Recursive1Batched({k}, {})", u8::from(c))),
    };
    Ok(head(verifier_filenames, "mains/recursive1.circom", keys, true, true) + &vd + &main)
}

/// A recursive2 over `opts.agg_arity` proofs of airgroup `airgroup_id`.
pub fn gen_recursive2(
    verifier_filenames: &[String],
    keys: &[KeyData],
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
    Ok(head(verifier_filenames, "mains/recursive2.circom", keys, true, true)
        + &vd
        + &main_line(&GLOBALS, &format!("Recursive2({agg_arity})")))
}

/// The vadcop_final over one recursive2 proof per airgroup, and the global constraints.
pub fn gen_vadcop_final(
    stark_infos: &[Value],
    verifier_filenames: &[String],
    keys: &[KeyData],
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
    let vd = vadcop_data(vadcop_info, basic_vk, agg_vk, 0, 0)?;
    Ok(head(verifier_filenames, "mains/vadcop_final.circom", keys, true, true)
        + &vd
        + "\n"
        + &global_constraints(vadcop_info)?
        + &main_line(&GLOBALS[..1], &format!("VadcopFinal({n_ag})")))
}

#[cfg(test)]
pub(super) mod tests {
    use std::fs;
    use std::path::Path;
    use std::process::Command;

    use super::*;
    use crate::stark2circom::circom_verifier::{
        gen_stark_verifier_bn128, gen_stark_verifier_bn128_blake3, KeyVerifier, Pil2CircomOptions,
    };
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
        gen_recursion_final(&recursivef_stark_info(merkle_tree_custom), &verifiers, &[], Some(&Value::Null)).unwrap()
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
        let out = gen_recursion_final(&si, &["v.circom".into()], &[], None).unwrap();
        assert!(out.contains("include \"mains/final_snark.circom\";"), "{out}");
        assert!(out.contains("function vadcop_DEFS() { return [0]; }"), "{out}");
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
        let out = gen_recursion_final(&si, &["v.circom".into()], &[], None).unwrap();
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
    pub(in crate::stark2circom::circuit_templates) fn out_and_vk(vk_first: bool) -> Value {
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
        assert!(gen_recursion_final(&si, &["v.circom".into()], &[], None).unwrap_err().to_string().contains("publics"));
        si["nPublics"] = json!(10);
        let err = gen_recursion_final(&si, &["v.circom".into()], &[], None).unwrap_err().to_string();
        assert!(err.contains("0 publics, the recursivef 5"), "{err}");
        for (vk_first, pos) in [(true, ", 0);"), (false, ", 1);")] {
            let out = gen_recursion_final(&si, &["v.circom".into()], &[], Some(&out_and_vk(vk_first))).unwrap();
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
            let err = gen_recursion_final(&si, &["v.circom".into()], &[], Some(&publics)).unwrap_err().to_string();
            assert!(err.contains(msg), "{err}");
        }
    }

    /// The whole final circuit around a recursivef, as setup-snark compiles it: the static library, the
    /// key's data and constraints, the main's dispatch. circom checks the include graph (CG04:
    /// `pragma custom_templates;` on every file on the way to a custom template), every template call
    /// and signal access. Returns the r1cs, for its custom gates and its size.
    ///
    /// Needs circomlib, from `npm install` in `setup/pil2-stark`; without it the test says so and
    /// passes (`None`).
    fn compile_final(
        label: &str,
        verifier: &KeyVerifier,
        stark_info: &Value,
        vk_first: bool,
    ) -> Option<crate::plonk2pil::r1cs::types::R1csFile<Bn128>> {
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
        let circomlib = root.join("setup/pil2-stark/node_modules/circomlib/circuits");
        if !circomlib.is_dir() {
            eprintln!("skipped: {} not present (npm install in setup/pil2-stark)", circomlib.display());
            return None;
        }
        let circom = root.join("setup/circom").join(if cfg!(target_os = "macos") { "circom_mac" } else { "circom" });
        let libraries = Path::new(env!("CARGO_MANIFEST_DIR")).join("stark2circom/circom_verifier");
        let files = ["recursivef.verifier.circom".to_string()];
        let keys = std::slice::from_ref(&verifier.data);
        let circuit = gen_recursion_final(stark_info, &files, keys, Some(&out_and_vk(vk_first))).unwrap();

        let dir = std::env::temp_dir().join(format!("final_{label}_{}_{vk_first}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        fs::write(dir.join("recursivef.verifier.circom"), &verifier.circom).unwrap();
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
        assert!(out.status.success(), "{label}: circom failed:\n{stderr}{}", String::from_utf8_lossy(&out.stdout));

        let r1cs = read_r1cs_from_bytes::<Bn128>(&fs::read(dir.join("final.r1cs")).unwrap()).unwrap();
        // Printed so a refactor of the library can compare sizes: `cargo test -- --nocapture compiles`.
        eprintln!(
            "BN128 final {label} vk_first={vk_first}: wires={} constraints={} custom_gate_uses={}",
            r1cs.header.n_vars,
            r1cs.header.n_constraints,
            r1cs.custom_gates_uses.len()
        );
        let _ = fs::remove_dir_all(&dir);
        Some(r1cs)
    }

    /// A recursivef with custom Poseidon trees: the r1cs has the custom gates of the Poseidon tree and
    /// of the publics hash.
    #[test]
    fn custom_final_compiles() {
        let mut stark_info = recursivef_stark_info(true);
        stark_info["nPublics"] = json!(10);
        let verifier_info = json!({ "constraintDag": { "version": 1, "field": "gl", "extDim": 3, "nodes": [[5, 1, 0, 0]], "roots": [[0, 0, "c"]] } });
        let root_c = ["1".to_string(), "0".to_string(), "0".to_string(), "0".to_string()];
        let verifier =
            gen_stark_verifier_bn128(Some(&root_c), &stark_info, &verifier_info, &Pil2CircomOptions::default())
                .unwrap();
        // FinalSnark binds the VK (vkPos >= 0) where it sits: first and last.
        for vk_first in [true, false] {
            let Some(r1cs) = compile_final("custom", &verifier, &stark_info, vk_first) else { return };
            let mut names: Vec<String> = r1cs.custom_gates.iter().map(|g| g.template_name.clone()).collect();
            names.sort();
            names.dedup();
            assert!(names.contains(&"PoseidonT".to_string()) && names.contains(&"Num2Bytes".to_string()), "{names:?}");
        }
    }

    /// The other two BN128 verifiers compile too: a recursivef with plain (circomlib) Poseidon trees, and
    /// one with blake3 trees over Goldilocks.
    #[test]
    fn plain_poseidon_and_blake3_finals_compile() {
        let verifier_info = json!({ "constraintDag": { "version": 1, "field": "gl", "extDim": 3, "nodes": [[5, 1, 0, 0]], "roots": [[0, 0, "c"]] } });
        let root_c = ["1".to_string(), "0".to_string(), "0".to_string(), "0".to_string()];

        let mut plain = recursivef_stark_info(false);
        plain["nPublics"] = json!(10);
        let verifier =
            gen_stark_verifier_bn128(Some(&root_c), &plain, &verifier_info, &Pil2CircomOptions::default()).unwrap();
        if compile_final("poseidon", &verifier, &plain, true).is_none() {
            return;
        }

        let mut blake3 = recursivef_stark_info(true);
        blake3["nPublics"] = json!(10);
        blake3["starkStruct"]["verificationHashType"] = json!("GL");
        blake3["starkStruct"]["merkleTreeArity"] = json!(2);
        let opts = Pil2CircomOptions { hash: proofman_common::hash_family::HashFamily::Blake3, ..Default::default() };
        let verifier = gen_stark_verifier_bn128_blake3(Some(&root_c), &blake3, &verifier_info, &opts).unwrap();
        compile_final("blake3", &verifier, &blake3, true);
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
        gen_recursive1(&stark_info(), &["v.circom".into()], &[key(0)], &vadcop(), &opts(k, has_compressor)).unwrap()
    }

    /// The data of a key of airgroup `ag` (the GL sample starkinfo, `4 + ag` publics).
    fn key(ag: u64) -> KeyData {
        use crate::stark2circom::circom_verifier::lib_gl::{gen_stark_verifier_gl, tests};
        let mut info = tests::stark_info();
        info["airgroupId"] = serde_json::json!(ag);
        info["nPublics"] = serde_json::json!(4 + ag);
        gen_stark_verifier_gl(None, &info, &tests::verifier_info(), &Default::default()).unwrap().data
    }

    /// A main is data: the verifier's include, the keys' data and dispatch, the vadcop data and the main line.
    #[test]
    fn mains_are_data_over_static_templates() {
        let out = gen(1, false);
        assert!(out.contains("include \"v.circom\";\ninclude \"mains/recursive1.circom\";"), "{out}");
        assert!(out.contains("function key_nBits(k) { return 16; }"), "{out}");
        assert!(out.contains("\n    roots <== ConstraintRoots0()(evals, challenges"), "{out}");
        assert!(!out.contains("if (k =="), "one key needs no branch on k:\n{out}");
        assert!(out.contains("function vadcop_airId() { return 1; }"), "{out}");
        let templates: Vec<&str> = out.lines().filter(|l| l.starts_with("template ")).collect();
        assert_eq!(templates, ["template ConstraintRootsAt(k) {"], "{out}");
        assert!(out
            .ends_with("component main {public [publics, proofValues, globalChallenge, rootCAgg]} = Recursive1(1);\n"));
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
        assert!(gen_recursive1(&stark_info(), &["v.circom".into()], &[], &vadcop(), &o).is_err());
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
        let rec2 = gen_recursive2(&["v.circom".into()], &[], &vadcop, 1, std::slice::from_ref(&vk), &o).unwrap();
        assert!(rec2.contains("function vadcop_airgroupId() { return 1; }"), "{rec2}");
        assert!(
            rec2.contains(
                "function vadcop_nBasics(i) { if (i == 0) { return 0; } if (i == 1) { return 1; } return 0; }"
            ),
            "{rec2}"
        );
        assert!(rec2.ends_with("= Recursive2(2);\n"), "{rec2}");

        // Key k is airgroup k: its data and its ConstraintRoots<k>.
        let fin = gen_vadcop_final(
            &[si(0), si(1)],
            &["v0.circom".into(), "v1.circom".into()],
            &[key(0), key(1)],
            &vadcop,
            &[vec![vk.clone(), vk.clone()], vec![vk.clone()]],
            &[vk.clone(), vk],
        )
        .unwrap();
        assert!(fin.contains("include \"mains/vadcop_final.circom\";"), "{fin}");
        assert!(fin.contains("function key_nPublics(k) { if (k == 0) { return 4; } return 5; }"), "{fin}");
        assert!(fin.contains("    if (k == 1) { roots <== ConstraintRoots1()("), "{fin}");
        assert!(fin.contains("template GlobalConstraints(nAg, maxAgg, nPub, nPV, nCh) {"), "{fin}");
        assert!(fin.ends_with("component main {public [publics]} = VadcopFinal(2);\n"), "{fin}");
    }
}
