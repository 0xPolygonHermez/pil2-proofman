//! The GL verifier (FRI or STIR) of one key: its constraints (`ConstraintRoots<sfx>`, the verifier file)
//! and its data (`lib_params`, for the main). Everything else is the static `StarkVerifierGL(k)`
//! (`helper_circuits/verifier/stark_verifier.circom`), which the circuit's main instantiates.

use anyhow::Result;
use serde_json::Value;

use super::constraint_roots::constraint_roots;
use super::lib_params::{parse_stark_info, KeyData, KeyOpts};
use super::{KeyVerifier, Pil2CircomOptions};
use proofman_common::hash_family::HashFamily;

/// The family's entry (`verifier/family_<dir>.circom`): its hash, selector and transcript files and the library.
fn includes(hash: HashFamily) -> String {
    let dir = match hash {
        HashFamily::Poseidon1 => "poseidon1",
        HashFamily::Poseidon2 => "poseidon2",
        HashFamily::Blake3 => "blake3",
    };
    format!("include \"verifier/family_{dir}.circom\";\n")
}

/// The suffix of a key's `ConstraintRoots`: its airgroup id, if the starkinfo has one.
pub(crate) fn key_suffix(si: &Value) -> String {
    si.get("airgroupId").and_then(Value::as_u64).map(|id| id.to_string()).unwrap_or_default()
}

/// Generate the GL verifier constraints and data of a starkinfo from its `constraintDag`.
pub(crate) fn gen_stark_verifier_gl(
    const_root: Option<&[String; 4]>,
    si: &Value,
    vi: &Value,
    opts: &Pil2CircomOptions,
) -> Result<KeyVerifier> {
    let sfx = key_suffix(si);
    let info = parse_stark_info(si)?;
    let (dag, ctx) = super::dag_and_ctx(&info, vi)?;
    let roots = constraint_roots(&sfx, &dag, &ctx)?;
    let data = KeyData::new(
        &info,
        &KeyOpts {
            input_challenges: opts.input_challenges,
            batch_size: opts.fri_queries_batch_size,
            roots: Some((roots.n_roots, roots.first_root_dim, ctx.tables().vc)),
            verkey_input: opts.verkey_input,
            enable_input: opts.enable_input,
            const_root: const_root.cloned(),
        },
        sfx,
    )?;
    let circom =
        format!("pragma circom 2.1.0;\npragma custom_templates;\n\n{}\n{}", includes(opts.hash), roots.template);
    Ok(KeyVerifier { circom, data })
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use serde_json::json;

    /// A FRI starkinfo with every input kind: 2 stages, a custom commit, last levels in clear.
    pub(crate) fn stark_info() -> Value {
        // Opening 1 is empty; columns: cm0 (base) at {0,2}, cm1 (ext) at {0}, const0 at {2}, custom at {0,2}.
        json!({
            "keyVersion": proofman_common::KEY_VERSION,
            "name": "Sample",
            "starkStruct": {
                "verificationHashType": "GL", "numQueries": 10, "nBits": 16, "nBitsExt": 17,
                "grindingBitsQueries": 8, "merkleTreeArity": 4, "transcriptArity": 4, "merkleTreeCustom": false,
                "lastLevelVerification": 1, "splitLinearHash": false, "hashCommits": false,
                "foldingFactors": [], "logDegrees": [], "logDomainSizes": [17, 14, 11]
            },
            "security": {"regime": "test"},
            "nStages": 2,
            "nPublics": 4,
            "nConstants": 1,
            "qDeg": 1,
            "qDim": 3,
            "cExpId": 0,
            "boundaries": [{"name": "everyRow"}],
            "challengesMap": [
                {"name": "std_alpha", "stage": 2, "stageId": 0},
                {"name": "std_vc", "stage": 3, "stageId": 0},
                {"name": "std_xi", "stage": 4, "stageId": 0},
                {"name": "std_vf1", "stage": 5, "stageId": 0},
                {"name": "std_vf2", "stage": 5, "stageId": 1}
            ],
            "airgroupValuesMap": [{"name": "g", "stage": 1}],
            "airValuesMap": [],
            "proofValuesMap": [],
            "mapSectionsN": {"cm1": 1, "cm2": 3, "cm3": 3, "rom0": 1},
            "openingPoints": [-1, 0, 1],
            "cmPolsMap": [
                {"name": "a", "stage": 1, "stagePos": 0, "stageId": 0, "dim": 1},
                {"name": "b", "stage": 2, "stagePos": 0, "stageId": 0, "dim": 3},
                {"name": "q", "stage": 3, "stagePos": 0, "stageId": 0, "dim": 3}
            ],
            "customCommits": [{"name": "rom", "publicValues": [{"idx": 0}, {"idx": 1}, {"idx": 2}, {"idx": 3}], "stageWidths": [1]}],
            "customCommitsMap": [[{"name": "r", "stage": 0, "stagePos": 0, "stageId": 0, "dim": 1}]],
            "evMap": [
                {"type": "cm", "id": 0, "prime": -1, "openingPos": 0},
                {"type": "cm", "id": 1, "prime": -1, "openingPos": 0},
                {"type": "custom", "id": 0, "prime": -1, "openingPos": 0, "commitId": 0},
                {"type": "cm", "id": 0, "prime": 1, "openingPos": 2},
                {"type": "const", "id": 0, "prime": 1, "openingPos": 2},
                {"type": "custom", "id": 0, "prime": 1, "openingPos": 2, "commitId": 0},
                {"type": "cm", "id": 2, "prime": 1, "openingPos": 2}
            ]
        })
    }

    /// Roots: cm0 * cm2' - alpha, and the quotient column read as is.
    pub(crate) fn verifier_info() -> Value {
        json!({ "constraintDag": {
            "version": 1, "field": "gl", "extDim": 3,
            "nodes": [[5, 0, 0, -1], [5, 0, 2, 1], [2, 0, 1], [6, 2, 0], [1, 2, 3]],
            "roots": [[4, 0, "c0"], [1, 0, "q"]]
        }})
    }

    /// The key's verifier file and, after it, the data its main carries.
    fn render(si: &Value, opts: &Pil2CircomOptions) -> String {
        let v = gen_stark_verifier_gl(None, si, &verifier_info(), opts).unwrap();
        format!("{}\n{}", v.circom, super::super::lib_params::key_functions(std::slice::from_ref(&v.data)))
    }

    fn opts(hash: HashFamily) -> Pil2CircomOptions {
        Pil2CircomOptions { hash, ..Default::default() }
    }

    fn has_all(out: &str, lines: &[&str]) {
        for l in lines {
            assert!(out.contains(l), "missing `{l}`:\n{out}");
        }
    }

    /// Per key only constraints in its file and data for the main: the verifier is the library's.
    #[test]
    fn per_key_output_is_data_and_constraints() {
        let v = gen_stark_verifier_gl(None, &stark_info(), &verifier_info(), &opts(HashFamily::Poseidon2)).unwrap();
        let templates: Vec<&str> = v.circom.lines().filter(|l| l.starts_with("template ")).collect();
        assert_eq!(templates, ["template parallel ConstraintRoots() {"], "{}", v.circom);
        assert!(!v.circom.contains("function "), "the verifier file holds no data:\n{}", v.circom);
        let out = render(&stark_info(), &opts(HashFamily::Poseidon2));
        assert_eq!(out.matches("function key_").count(), super::super::lib_params::FIELDS.len(), "{out}");
        has_all(
            &out,
            &["function key_STEPS(k) { return [17, 14, 11]; }", "include \"verifier/family_poseidon2.circom\";"],
        );
    }

    /// The flat proof's layout: publics and proof values are not in it, the inputs in proof2pointer order.
    #[test]
    fn proof_offsets_follow_proof2pointer() {
        let out = render(&stark_info(), &opts(HashFamily::Poseidon2));
        // airgroupvalues(1x3), roots 1..3 (3x4), evals (7x3), then the trees: const, custom, stages.
        has_all(
            &out,
            &[
                "function key_offAirgroupValues(k) { return 0; }",
                "function key_offAirValues(k) { return -1; }",
                "function key_OFF_ROOT(k) { return [3, 7, 11]; }",
                "function key_offEvals(k) { return 15; }",
                "function key_TREE_ROOT(k) { return [3, 7, 11, -1, -2]; }",
                "function key_TREE_PUB(k) { return [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 3]; }",
                "function key_ROOT_C(k) { return [0, 0, 0, 0]; }",
            ],
        );
        let off = |name: &str| -> Vec<i64> {
            let l = out.lines().find(|l| l.starts_with(&format!("function key_{name}(k) "))).unwrap();
            let v = &l[l.find("return ").unwrap() + 7..l.rfind(';').unwrap()];
            v.trim_matches(['[', ']']).split(", ").map(|x| x.parse().unwrap()).collect()
        };
        let (v, s, l) = (off("OFF_L0_VALUES"), off("OFF_L0_SIBLINGS"), off("OFF_L0_LAST_LEVELS"));
        // The const tree is the first in the proof: right after the 7 evaluations.
        assert_eq!(v[3], 15 + 21);
        assert!(v.iter().zip(&s).zip(&l).all(|((v, s), l)| v < s && s < l), "{v:?} {s:?} {l:?}");
        assert_eq!(off("proofSize")[0], off("offNonce")[0] + 1);
    }

    #[test]
    fn data_carries_the_protocol_flags() {
        let mut si = stark_info();
        si["circuitType"] = json!(37);
        si["starkStruct"]["hashCommits"] = json!(true);
        let o = Pil2CircomOptions { input_challenges: true, verkey_input: true, ..opts(HashFamily::Blake3) };
        let out = render(&si, &o);
        has_all(
            &out,
            &[
                "function key_hashCommits(k) { return 1; }",
                "function key_circuitType(k) { return 37; }",
                "function key_verkeyInput(k) { return 1; }",
                "include \"verifier/family_blake3.circom\";",
            ],
        );
    }

    #[test]
    fn basic_transcript_needs_stage2_and_circuit_type() {
        let o = Pil2CircomOptions { input_challenges: true, ..opts(HashFamily::Poseidon2) };
        let mut si = stark_info();
        si["circuitType"] = json!(37);
        si["nStages"] = json!(1);
        let err = gen_stark_verifier_gl(None, &si, &verifier_info(), &o).unwrap_err();
        assert!(err.to_string().contains("nStages"), "{err}");
        let err = gen_stark_verifier_gl(None, &stark_info(), &verifier_info(), &o).unwrap_err();
        assert!(err.to_string().contains("circuitType"), "{err}");
    }

    #[test]
    fn deep_groups_reach_the_data() {
        let out = render(&stark_info(), &opts(HashFamily::Poseidon2));
        // Openings -1 and 1 have evaluations; columns by first appearance: cm1_0, cm2_0, rom, const, cm3_0,
        // whose leaves sit at S0 offsets 0, 1, 8, 7, 4. Column groups {cm1_0, rom} at both, {cm2_0}, {const, cm3_0}.
        has_all(
            &out,
            &[
                "function key_nOpenings(k) { return 2; }",
                "function key_nPols(k) { return 5; }",
                "function key_nSums(k) { return 1; }",
                "function key_L0_TREE_START(k) { return [0, 1, 4, 7, 8, 9]; }",
                "function key_OPENING_NUM_EVALS(k) { return [3, 4]; }",
                "function key_EVAL_COLUMN(k) { return [0, 1, 2, 0, 3, 2, 4]; }",
                "function key_SUMS(k) { return [0, 1]; }",
                "function key_COLS(k) { return [2, 2, 0, 0, 1, 2, 8, 1, 0, 1, 1, 1, 3, 1, 2, 3, 7, 1, 4, 4, 3]; }",
            ],
        );
    }

    #[test]
    fn no_evaluations_is_an_error() {
        let mut si = stark_info();
        si["evMap"] = json!([]);
        let err = gen_stark_verifier_gl(None, &si, &verifier_info(), &opts(HashFamily::Poseidon2)).unwrap_err();
        assert!(err.to_string().contains("no evaluations"), "{err}");
    }

    #[test]
    fn needs_a_constraint_dag() {
        let err = gen_stark_verifier_gl(None, &stark_info(), &json!({}), &opts(HashFamily::Poseidon2)).unwrap_err();
        assert!(err.to_string().contains("constraintDag"), "{err}");
    }

    #[test]
    fn refuses_challenges_the_transcript_does_not_draw() {
        let mut si = stark_info();
        si["challengesMap"].as_array_mut().unwrap().push(json!({"name": "x", "stage": 1, "stageId": 0}));
        let err = gen_stark_verifier_gl(None, &si, &verifier_info(), &opts(HashFamily::Poseidon2)).unwrap_err();
        assert!(err.to_string().contains("not the order the verifier transcript draws"), "{err}");
    }

    #[test]
    fn names_with_airgroup_id() {
        let mut si = stark_info();
        si["airgroupId"] = json!(42u64);
        let v = gen_stark_verifier_gl(None, &si, &verifier_info(), &opts(HashFamily::Poseidon1)).unwrap();
        has_all(
            &v.circom,
            &["template parallel ConstraintRoots42() {", "include \"verifier/family_poseidon1.circom\";"],
        );
        assert_eq!(v.data.suffix, "42");
    }

    /// `stark_info()` with a 2-iteration STIR schedule.
    pub(crate) fn stir_info() -> Value {
        let mut si = stark_info();
        let ss = &mut si["starkStruct"];
        ss["lowDegreeTest"] = json!("STIR");
        ss["foldingFactors"] = json!([3, 3]);
        ss["logDegrees"] = json!([16, 13, 10]);
        ss["logDomainSizes"] = json!([17, 16, 15]);
        ss["numQueries"] = json!([10, 6]);
        ss["grindingBitsQueries"] = json!([4, 0]);
        si
    }

    /// STIR is the same library: the L_0 queries with T_0 as the one fold tree, then `StirVerifyRounds`.
    #[test]
    fn stir_key_is_data() {
        let out = render(&stir_info(), &opts(HashFamily::Poseidon2));
        has_all(
            &out,
            &[
                "function key_stir(k) { return 1; }",
                "function key_nQueries(k) { return 10; }",
                "function key_powBits(k) { return 4; }",
                "function key_STEPS(k) { return [17, 14]; }",
                "function key_STIR_NUM_QUERIES(k) { return [10, 6]; }",
                "function key_stirFinalBits(k) { return 10; }",
                "function key_nFolds(k) { return 2; }",
                "function key_offNonce(k) { return -1; }",
            ],
        );
        assert!(!out.contains("function key_offBetas(k) { return -1; }"), "{out}");
        assert!(!out.contains("function key_OFF_ANS(k) { return [-1]; }"), "{out}");
    }

    #[test]
    fn malformed_stir_schedule_is_an_error() {
        let mut si = stir_info();
        si["starkStruct"]["numQueries"] = json!([10]);
        let err = gen_stark_verifier_gl(None, &si, &verifier_info(), &opts(HashFamily::Poseidon2)).unwrap_err();
        assert!(err.to_string().contains("malformed STIR schedule"), "{err}");
    }

    #[test]
    fn refuses_split_linear_hash() {
        let mut si = stark_info();
        si["starkStruct"]["splitLinearHash"] = json!(true);
        let err = gen_stark_verifier_gl(None, &si, &verifier_info(), &opts(HashFamily::Poseidon2)).unwrap_err();
        assert!(err.to_string().contains("splitLinearHash"), "{err}");
    }

    /// A fold tree whose whole path is sent in clear has no siblings input (circom T3001 otherwise).
    #[test]
    fn empty_fold_tree_has_no_siblings() {
        let mut si = stark_info();
        si["starkStruct"]["logDomainSizes"] = json!([17, 14, 2]);
        let out = render(&si, &opts(HashFamily::Poseidon2));
        let fs = out.lines().find(|l| l.starts_with("function key_OFF_FOLD_SIBLINGS(k) ")).unwrap();
        assert!(fs.ends_with(", -1]; }") && !fs.contains("[-1"), "{fs}");
    }

    /// The library reads exactly the declared data (`key_<name>`, lib_params::FIELDS).
    #[test]
    fn library_reads_exactly_the_declared_fields() {
        let lib = [
            include_str!("helper_circuits/verifier/stark_verifier.circom"),
            include_str!("helper_circuits/verifier/stark_verifier_bn128.circom"),
            include_str!("helper_circuits/vadcop/values.circom"),
            include_str!("helper_circuits/vadcop/fold.circom"),
            include_str!("helper_circuits/mains/compressor.circom"),
            include_str!("helper_circuits/mains/recursive1.circom"),
            include_str!("helper_circuits/mains/recursive2.circom"),
            include_str!("helper_circuits/mains/vadcop_final.circom"),
            include_str!("helper_circuits/mains/final_snark.circom"),
        ]
        .concat();
        let mut used: Vec<&str> = lib
            .split("key_")
            .skip(1)
            .map(|r| &r[..r.find(|c: char| !c.is_ascii_alphanumeric() && c != '_').unwrap()])
            .filter(|n| !n.is_empty())
            .collect();
        used.sort();
        used.dedup();
        let mut fields = super::super::lib_params::FIELDS.to_vec();
        fields.sort();
        assert_eq!(used, fields);
    }
}
