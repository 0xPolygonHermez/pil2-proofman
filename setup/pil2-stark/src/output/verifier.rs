use anyhow::{anyhow, Result};
use std::fs;

use proofman_common::hash_family::{self, HashFamily};
use proofman_common::{EvMapEType, LowDegreeLayout};
use proofman_verifier::{AirConstraints, ColRef, ColType, LowerTables};
use crate::types::stark_info::{StarkInfo, VerifierInfo};
use crate::types::stark_struct::LowDegreeTest;

pub fn write_verifier_rust_file(
    path: &str,
    stark_info: &StarkInfo,
    verifier_info: &VerifierInfo,
    hash: HashFamily,
) -> Result<()> {
    // The native verifier parses no custom-commit openings; fail here, not at verification.
    anyhow::ensure!(
        stark_info.custom_commits.is_empty(),
        "{path}: the native Rust verifier does not support custom commits"
    );
    // Its transcript replay (stark_verify and stark_verify_stir) is laid out for three roots.
    anyhow::ensure!(
        stark_info.n_stages == 2,
        "{path}: the native Rust verifier supports nStages = 2 only (got {})",
        stark_info.n_stages
    );

    println!("> Writing rust verifier file");

    let rust_verifier = prepare_verifier_rust(stark_info, verifier_info, hash)?;
    fs::write(path, rust_verifier)?;

    Ok(())
}

/// The `challengesMap` position of challenge `stage_id` of stage `nStages + after` (xi: 2, vf1/vf2: 3).
fn deep_challenge_idx(stark_info: &StarkInfo, after: u64, stage_id: u64) -> Result<u64> {
    let stage = stark_info.n_stages + after;
    stark_info
        .challenges_map
        .iter()
        .position(|c| c.stage == stage && c.stage_id == stage_id)
        .map(|i| i as u64)
        .ok_or_else(|| anyhow::anyhow!("challengesMap has no challenge {stage_id} of stage {stage}"))
}

fn prepare_verifier_rust(stark_info: &StarkInfo, verifier_info: &VerifierInfo, hash: HashFamily) -> Result<String> {
    // Leaf + compression hashes follow the Merkle tree arity; the transcript hash
    // follows the (fixed, arity-4) transcript arity; grinding is a fixed-width hash.
    let merkle_arity = stark_info.stark_struct.merkle_tree_arity;
    let transcript_arity = stark_info.stark_struct.transcript_arity;
    let merkle_hash_type = hash_family::rust_hash_type(hash, merkle_arity as u64);
    // The transcript CONSTRUCTION, not a hash to wrap in a sponge: BLAKE3's is not a sponge.
    let transcript_type = hash_family::rust_transcript_type(hash, transcript_arity as u64);
    let grinding_type = hash_family::rust_grinding_type(hash);
    // The constraints are the verifierinfo's constraintDag, verbatim: the verifier crate checks and resolves it
    // against the tables (as C++ checkConstraintDag), here first so that a key it refuses fails the setup.
    let dag_json = serde_json::to_string(&verifier_info.constraint_dag)?;
    let tables = dag_tables(stark_info)?;
    AirConstraints::from_json(&dag_json, &tables).map_err(|e| anyhow!("the native verifier refuses the DAG: {e:?}"))?;

    let mut lines: Vec<String> = Vec::new();

    lines.push("use alloc::vec;".to_string());
    let mut hash_imports: Vec<&str> = Vec::new();
    for ht in [merkle_hash_type, grinding_type]
        .into_iter()
        .chain(hash_family::rust_transcript_imports(hash, transcript_arity as u64))
    {
        if !hash_imports.contains(&ht) {
            hash_imports.push(ht);
        }
    }
    let is_stir = matches!(stark_info.stark_struct.low_degree_test, LowDegreeTest::Stir(_));
    lines.push(format!("use proofman_fields::{{Goldilocks, {}}};", hash_imports.join(", ")));
    if is_stir {
        lines.push(
            "use crate::{stark_verify_stir, FriEvalGroup, FriEvalRef, StirParams, StirVerifierInfo};".to_string(),
        );
    } else {
        lines.push("use crate::{stark_verify, FriEvalGroup, FriEvalRef, VerifierInfo};".to_string());
    }
    lines.push("use crate::{AirConstraints, ColRef, ColType, LowerTables};".to_string());
    lines.push("use crate::VadcopFinalProof;".to_string());
    lines.push(String::new());

    // The constraint DAG and what it resolves against.
    lines.push(format!("const CONSTRAINT_DAG: &str = {dag_json:?};"));
    lines.push(String::new());
    lines.extend(dag_tables_rust(&tables));
    lines.push(String::new());

    // The shared STARK geometry, identical for both low-degree tests.

    let opening_points_str: Vec<String> = stark_info.opening_points.iter().map(|p| p.to_string()).collect();

    // The FRI query polynomial is structural, so the verifier evaluates it from
    // the evaluation map instead of an unrolled per-air function.
    let common = &stark_info.common;
    let group_strs: Vec<String> = proofman_common::deep::fri_ev_groups(common)
        .iter()
        .map(|g| {
            let refs: Vec<String> = g
                .refs
                .iter()
                .map(|r| format!("FriEvalRef::new({}, {}, {}, {})", r.bucket, r.offset, r.dim, r.col))
                .collect();
            format!("FriEvalGroup {{ opening: {}, refs: vec![{}] }}", g.opening, refs.join(", "))
        })
        .collect();
    let fri_ev_groups_str =
        format!("        fri_ev_groups: vec![\n            {}\n        ],", group_strs.join(",\n            "));

    // Find q_index: the evMap index of the cm polynomial at stage nStages+1, stageId 0
    let q_index = (stark_info.cm_pols_map.iter().position(|p| p.stage == stark_info.n_stages + 1 && p.stage_id == 0))
        .ok_or_else(|| anyhow!("no Q column in cmPolsMap"))?;
    let q_ev_index = (stark_info
        .ev_map
        .iter()
        .position(|ev| ev.ev_type == proofman_common::EvMapEType::Cm && ev.id == q_index as u64))
    .ok_or_else(|| anyhow!("the Q column is not opened in evMap"))?;

    let layout = common.proof_layout();
    let num_vals: Vec<String> = layout.num_vals.iter().map(|v| v.to_string()).collect();
    let xi_idx = deep_challenge_idx(stark_info, 2, 0)?;
    let n_challenges_total = stark_info.challenges_map.len() + common.stark_struct.low_degree_test.num_ldt_challenges();
    let (vf1_idx, vf2_idx) = (deep_challenge_idx(stark_info, 3, 0)?, deep_challenge_idx(stark_info, 3, 1)?);
    let vec_str = |v: &[u64]| -> String { v.iter().map(|b| b.to_string()).collect::<Vec<String>>().join(", ") };

    // verifier_info function
    lines.push("#[rustfmt::skip]".to_string());
    match (&stark_info.stark_struct.low_degree_test, &layout.ldt) {
        (LowDegreeTest::Fri(fri), LowDegreeLayout::Fri { steps, n_queries }) => {
            lines.push("fn verifier_info() -> VerifierInfo {".to_string());
            lines.push("    VerifierInfo {".to_string());
            lines.push(format!("        n_stages: {},", stark_info.n_stages));
            lines.push(format!("        n_constants: {},", layout.n_constants));
            lines.push(format!("        n_evals: {},", layout.n_evals));
            lines.push(format!("        n_bits: {},", stark_info.stark_struct.n_bits));
            lines.push(format!("        n_bits_ext: {},", layout.n_bits_ext));
            lines.push(format!("        arity: {},", layout.arity));
            lines.push(format!("        n_fri_queries: {},", n_queries));
            lines.push(format!("        n_fri_steps: {},", steps.len()));
            lines.push(format!("        n_challenges: {},", stark_info.challenges_map.len()));
            lines.push(format!("        xi_idx: {xi_idx},"));
            lines.push(format!("        vf1_idx: {vf1_idx},"));
            lines.push(format!("        vf2_idx: {vf2_idx},"));
            lines.push(format!("        n_challenges_total: {n_challenges_total},"));

            lines.push(format!("        fri_steps: vec![{}],", vec_str(steps)));

            lines.push(format!("        hash_commits: {},", layout.hash_commits));
            lines.push(format!("        last_level_verification: {},", layout.last_level_verification));
            lines.push(format!("        pow_bits: {},", fri.grinding_bits_queries));
            lines.push(fri_ev_groups_str.clone());

            lines.push(format!("        num_vals: vec![{}],", num_vals.join(", ")));
            lines.push(format!("        opening_points: vec![{}],", opening_points_str.join(", ")));
            lines.push(format!("        q_deg: {},", stark_info.q_deg));
            lines.push(format!("        q_index: {},", q_ev_index));

            lines.push("    }".to_string());
            lines.push("}\n".to_string());
        }
        (LowDegreeTest::Stir(_), LowDegreeLayout::Stir(stir)) => {
            lines.push("fn verifier_info() -> StirVerifierInfo {".to_string());
            lines.push("    StirVerifierInfo {".to_string());
            lines.push(format!("        n_stages: {},", stark_info.n_stages));
            lines.push(format!("        n_constants: {},", layout.n_constants));
            lines.push(format!("        n_evals: {},", layout.n_evals));
            lines.push(format!("        n_bits: {},", stark_info.stark_struct.n_bits));
            lines.push(format!("        n_bits_ext: {},", layout.n_bits_ext));
            lines.push(format!("        n_challenges: {},", stark_info.challenges_map.len()));
            lines.push(format!("        xi_idx: {xi_idx},"));
            lines.push(format!("        vf1_idx: {vf1_idx},"));
            lines.push(format!("        vf2_idx: {vf2_idx},"));
            lines.push(format!("        n_challenges_total: {n_challenges_total},"));

            lines.push(format!("        num_vals: vec![{}],", num_vals.join(", ")));
            lines.push(format!("        opening_points: vec![{}],", opening_points_str.join(", ")));
            lines.push(format!("        q_deg: {},", stark_info.q_deg));
            lines.push(format!("        q_index: {},", q_ev_index));
            lines.push(fri_ev_groups_str.clone());

            lines.push("        stir: StirParams {".to_string());
            lines.push(format!("            folding_factors: vec![{}],", vec_str(&stir.folding_factors)));
            lines.push(format!("            log_degrees: vec![{}],", vec_str(&stir.log_degrees)));
            lines.push(format!("            log_domain_sizes: vec![{}],", vec_str(&stir.log_domain_sizes)));
            lines.push(format!("            num_queries: vec![{}],", vec_str(&stir.num_queries)));
            lines.push(format!("            grinding_bits_queries: vec![{}],", vec_str(&stir.grinding_bits_queries)));
            lines.push(format!("            arity: {},", stir.arity));
            lines.push(format!("            last_level_verification: {},", stir.last_level_verification));
            lines.push(format!("            hash_commits: {},", stir.hash_commits));
            lines.push("        },".to_string());

            lines.push("    }".to_string());
            lines.push("}\n".to_string());
        }
        _ => anyhow::bail!("starkinfo and its proof layout disagree on the low-degree test"),
    }

    // The verifier holds its constraints, parsed, checked and resolved once; `verify` builds one per call.
    let generics = format!("{merkle_hash_type}, {merkle_hash_type}, {transcript_type}, {grinding_type}");
    let verify_fn = if is_stir { "stark_verify_stir" } else { "stark_verify" };
    let info_type = if is_stir { "StirVerifierInfo" } else { "VerifierInfo" };
    lines.extend([
        "/// The verifier with its constraint DAG parsed, checked and resolved once: build one, verify many proofs."
            .into(),
        "pub struct Verifier {".into(),
        format!("    info: {info_type},"),
        "    constraints: AirConstraints,".into(),
        "}\n".into(),
        "impl Verifier {".into(),
        "    /// `None` when the constraint DAG does not check against its tables.".into(),
        "    pub fn new() -> Option<Self> {".into(),
        "        let constraints = AirConstraints::from_json(CONSTRAINT_DAG, &dag_tables()).ok()?;".into(),
        "        Some(Verifier { info: verifier_info(), constraints })".into(),
        "    }\n".into(),
        "    pub fn verify(&self, proof: &VadcopFinalProof, vk: &[u64]) -> bool {".into(),
        "        self.verify_u64(&proof.proof_with_publics(), vk)".into(),
        "    }\n".into(),
        "    pub fn verify_u64(&self, proof: &[u64], vk: &[u64]) -> bool {".into(),
    ]);
    if stark_info.multiproof {
        lines.push(format!(
            "        let Some(flat) = crate::expand_proof::<{merkle_hash_type}, {merkle_hash_type}, {transcript_type}>(proof, vk, &self.info.proof_layout()) else {{"
        ));
        lines.push("            return false;".to_string());
        lines.push("        };".to_string());
        lines.push(format!("        {verify_fn}::<{generics}>(&flat, vk, &self.info, &self.constraints)"));
    } else {
        lines.push(format!("        {verify_fn}::<{generics}>(proof, vk, &self.info, &self.constraints)"));
    }
    lines.extend([
        "    }".to_string(),
        "}\n".into(),
        "/// One `Verifier` per call: hold a `Verifier` to verify more than one proof.".into(),
        "pub fn verify(proof: &VadcopFinalProof, vk: &[u64]) -> bool {".into(),
        "    Verifier::new().is_some_and(|v| v.verify(proof, vk))".into(),
        "}\n".into(),
        "pub fn verify_u64(proof: &[u64], vk: &[u64]) -> bool {".into(),
        "    Verifier::new().is_some_and(|v| v.verify_u64(proof, vk))".into(),
        "}\n".into(),
    ]);

    let expected_fn = if is_stir { "expected_stir_proof_size_bytes" } else { "expected_proof_size_bytes" };
    if stark_info.multiproof {
        lines.push("/// Upper bound: a multiproof never outgrows the flat proof by more than its tag.".to_string());
        lines.push("pub fn max_proof_bytes() -> usize {".to_string());
        lines.push(format!("    crate::{expected_fn}(&verifier_info()) + 8"));
    } else {
        lines.push("pub fn expected_proof_bytes() -> usize {".to_string());
        lines.push(format!("    crate::{expected_fn}(&verifier_info())"));
    }
    lines.push("}\n".to_string());

    Ok(lines.join("\n"))
}

/// The tables the native verifier resolves the DAG against: the starkinfo's evMap, challengesMap and nPublics (it
/// reads no value, so it has no value stages).
pub fn dag_tables(stark_info: &StarkInfo) -> Result<LowerTables> {
    let small = |v: u64| u32::try_from(v).map_err(|_| anyhow!("{v} does not fit a DAG index"));
    let ev_map = (stark_info.ev_map.iter())
        .map(|e| {
            let (col, commit_id) = match e.ev_type {
                EvMapEType::Cm => (ColType::Witness, None),
                EvMapEType::Const => (ColType::Fixed, None),
                EvMapEType::Custom => (ColType::Custom, Some(small(e.commit_id)?)),
            };
            Ok(ColRef { col, commit_id, id: small(e.id)?, offset: e.prime })
        })
        .collect::<Result<_>>()?;
    let challenges = (stark_info.challenges_map.iter())
        .map(|c| Ok((small(c.stage)?, small(c.stage_id)?, c.name.as_str())))
        .collect::<Result<Vec<_>>>()?;
    let n_publics = small(stark_info.n_publics)?;
    LowerTables::new(
        ev_map,
        &challenges,
        n_publics,
        Default::default(),
        pil_info::core::dag::format::DagField::Goldilocks,
    )
    .map_err(|e| anyhow!("the native verifier's tables: {e:?}"))
}

/// `fn dag_tables() -> LowerTables` rendering `t` (no value stages).
fn dag_tables_rust(t: &LowerTables) -> Vec<String> {
    let mut lines =
        vec!["#[rustfmt::skip]".to_string(), "fn dag_tables() -> LowerTables {".into(), "    LowerTables {".into()];
    lines.push("        ev_map: vec![".into());
    for e in &t.ev_map {
        lines.push(format!(
            "            ColRef {{ col: ColType::{:?}, commit_id: {:?}, id: {}, offset: {} }},",
            e.col, e.commit_id, e.id, e.offset
        ));
    }
    lines.push("        ],".into());
    lines.push("        challenges: vec![".into());
    for &(stage, stage_id) in &t.challenges {
        lines.push(format!("            ({stage}, {stage_id}),"));
    }
    lines.push("        ],".into());
    lines.push(format!("        vc: {},", t.vc));
    lines.push(format!("        n_publics: {},", t.n_publics));
    lines.push(format!("        ext_dim: {},", t.ext_dim));
    lines.push("        ..Default::default()".into());
    lines.push("    }".into());
    lines.push("}".into());
    lines
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::output::test_support::full_random;
    use pil_info::core::dag::format::{DagBoundary, DagNode, DagRoot};

    /// The rendered verifier carries the DAG verbatim and the tables it resolves against, and calls the DAG path.
    #[test]
    fn the_rendered_verifier_reads_the_dag() {
        let mut rendered = 0;
        for f in full_random(6) {
            let mut si = f.si.clone();
            si.custom_commits.clear();
            let out = match prepare_verifier_rust(&si, &f.vi, HashFamily::Blake3) {
                Ok(out) => out,
                // a random DAG may read a value, which the native verifier refuses
                Err(e) => {
                    assert!(format!("{e:#}").contains("value"), "{e:#}");
                    continue;
                }
            };
            let json = serde_json::to_string(&f.vi.constraint_dag).unwrap();
            assert!(out.contains(&format!("const CONSTRAINT_DAG: &str = {json:?};")), "{out}");
            assert!(out.contains(&format!("        n_publics: {},", si.n_publics)), "{out}");
            assert_eq!(out.matches("            ColRef {").count(), si.ev_map.len());
            assert_eq!(out.matches("            (").count(), si.challenges_map.len());
            assert!(out.contains("AirConstraints::from_json(CONSTRAINT_DAG, &dag_tables()).ok()?"), "{out}");
            assert!(out.contains("(proof, vk, &self.info, &self.constraints)"), "{out}");
            assert!(!out.contains("q_verify"), "{out}");
            rendered += 1;
        }
        assert!(rendered > 0, "every random key read a value");
    }

    #[test]
    fn the_setup_fails_on_a_dag_the_native_verifier_refuses() {
        let f = full_random(1).pop().expect("a random key");
        let (si, mut vi) = (f.si, f.vi);
        let dag = &mut vi.constraint_dag;
        dag.nodes.push(DagNode::AirValue { id: 0 });
        dag.roots.push(DagRoot {
            node: dag.nodes.len() as u32 - 1,
            boundary: DagBoundary::EveryRow,
            label: "av".into(),
        });
        let e = prepare_verifier_rust(&si, &vi, HashFamily::Blake3).expect_err("an air value is refused");
        assert!(format!("{e:#}").contains("value"), "{e:#}");
    }
}
