use anyhow::Result;
use std::fs;

use proofman_common::hash_family;
use crate::io::parser_args::get_parser_args;
use crate::types::stark_info::{StarkInfo, VerifierInfo};
use crate::types::stark_struct::LowDegreeTest;

pub fn write_verifier_rust_file(
    path: &str,
    stark_info: &StarkInfo,
    verifier_info: &VerifierInfo,
    vadcop_final_proof: bool,
    hash_id: &str,
) -> Result<()> {
    println!("> Writing rust verifier file");

    let rust_verifier = prepare_verifier_rust(stark_info, verifier_info, vadcop_final_proof, hash_id)?;
    fs::write(path, rust_verifier)?;

    Ok(())
}

fn prepare_verifier_rust(
    stark_info: &StarkInfo,
    verifier_info: &VerifierInfo,
    vadcop_final_proof: bool,
    hash_id: &str,
) -> Result<String> {
    // Leaf + compression hashes follow the Merkle tree arity; the transcript hash
    // follows the (fixed, arity-4) transcript arity; grinding is a fixed-width hash.
    let merkle_arity = stark_info.stark_struct.merkle_tree_arity;
    let transcript_arity = stark_info.stark_struct.transcript_arity;
    let merkle_hash_type = hash_family::rust_hash_type(hash_id, merkle_arity as u64);
    let transcript_hash_type = hash_family::rust_hash_type(hash_id, transcript_arity as u64);
    let grinding_type = hash_family::rust_grinding_type(hash_id);
    let mut numbers_q = Vec::new();
    let q_result =
        get_parser_args(stark_info, &verifier_info.q_verifier.code, &mut numbers_q, false, true, None, "q_verify")?;
    let verify_q_rust = q_result.verify_rust;
    let verify_q_helpers = q_result.verify_rust_helpers;

    let mut numbers_fri = Vec::new();
    let fri_result = get_parser_args(
        stark_info,
        &verifier_info.query_verifier.code,
        &mut numbers_fri,
        false,
        true,
        None,
        "query_verify",
    )?;
    let verify_fri_rust = fri_result.verify_rust;
    let verify_fri_helpers = fri_result.verify_rust_helpers;

    let mut lines: Vec<String> = Vec::new();

    lines.push("use alloc::vec;".to_string());
    lines.push("use alloc::vec::Vec;".to_string());
    lines.push("use alloc::string::ToString;".to_string());
    let mut hash_imports: Vec<&str> = Vec::new();
    for ht in [merkle_hash_type, transcript_hash_type, grinding_type] {
        if !hash_imports.contains(&ht) {
            hash_imports.push(ht);
        }
    }
    let is_stir = matches!(stark_info.stark_struct.low_degree_test, LowDegreeTest::Stir(_));
    lines
        .push(format!("use proofman_fields::{{Goldilocks, CubicExtensionField, Field, {}}};", hash_imports.join(", ")));
    if is_stir {
        lines.push("use crate::{stark_verify_stir, Boundary, StirParams, StirVerifierInfo};".to_string());
    } else {
        lines.push("use crate::{stark_verify, Boundary, VerifierInfo};".to_string());
    }
    if vadcop_final_proof {
        lines.push("use crate::VadcopFinalProof;".to_string());
    }
    lines.push(String::new());

    // q_verify helper functions (if chunked)
    for line in &verify_q_helpers {
        lines.push(line.clone());
    }

    // q_verify function
    lines.push("#[rustfmt::skip]".to_string());
    lines.push("#[allow(clippy::all)]".to_string());
    lines.push("fn q_verify(challenges: &[CubicExtensionField<Goldilocks>], evals: &[CubicExtensionField<Goldilocks>], _publics: &[Goldilocks], zi: &[CubicExtensionField<Goldilocks>]) -> CubicExtensionField<Goldilocks> {".to_string());
    for line in &verify_q_rust {
        lines.push(line.clone());
    }
    lines.push("}".to_string());
    lines.push(String::new());
    lines.push(String::new());

    // query_verify helper functions (if chunked)
    for line in &verify_fri_helpers {
        lines.push(line.clone());
    }

    // query_verify function
    lines.push("#[rustfmt::skip]".to_string());
    lines.push("#[allow(clippy::all)]".to_string());
    lines.push("fn query_verify(challenges: &[CubicExtensionField<Goldilocks>], evals: &[CubicExtensionField<Goldilocks>], vals: &[Vec<Goldilocks>], xdivxsub: &[CubicExtensionField<Goldilocks>]) -> CubicExtensionField<Goldilocks> {".to_string());
    for line in &verify_fri_rust {
        lines.push(line.clone());
    }
    lines.push("}\n".to_string());

    // The shared STARK geometry, identical for both low-degree tests.
    let mut num_vals: Vec<String> = Vec::new();
    for i in 0..stark_info.n_stages + 1 {
        let key = format!("cm{}", i + 1);
        let val = stark_info.map_sections_n.get(&key).copied().unwrap_or(0);
        num_vals.push(val.to_string());
    }

    let opening_points_str: Vec<String> = stark_info.opening_points.iter().map(|p| p.to_string()).collect();

    let mut boundary_strs: Vec<String> = Vec::new();
    for b in &stark_info.boundaries {
        let offset_min = match b.offset_min {
            Some(v) => v.to_string(),
            None => "None".to_string(),
        };
        let offset_max = match b.offset_max {
            Some(v) => v.to_string(),
            None => "None".to_string(),
        };
        boundary_strs.push(format!(
            "Boundary {{ name: \"{}\".to_string(), offset_min: {}, offset_max: {} }}",
            b.name, offset_min, offset_max
        ));
    }

    // Find q_index: the evMap index of the cm polynomial at stage nStages+1, stageId 0
    let q_index = stark_info.cm_pols_map.iter().position(|p| p.stage == stark_info.n_stages + 1 && p.stage_id == 0);
    let q_ev_index = if let Some(qi) = q_index {
        stark_info.ev_map.iter().position(|ev| ev.ev_type == "cm" && ev.id == qi as u64).unwrap_or(0)
    } else {
        0
    };

    let vec_str = |v: &[usize]| -> String { v.iter().map(|b| b.to_string()).collect::<Vec<String>>().join(", ") };

    // verifier_info function
    lines.push("#[rustfmt::skip]".to_string());
    match &stark_info.stark_struct.low_degree_test {
        LowDegreeTest::Fri(fri) => {
            lines.push("fn verifier_info() -> VerifierInfo {".to_string());
            lines.push("    VerifierInfo {".to_string());
            lines.push(format!("        n_stages: {},", stark_info.n_stages));
            lines.push(format!("        n_constants: {},", stark_info.n_constants));
            lines.push(format!("        n_evals: {},", stark_info.ev_map.len()));
            lines.push(format!("        n_bits: {},", stark_info.stark_struct.n_bits));
            lines.push(format!("        n_bits_ext: {},", stark_info.stark_struct.n_bits_ext));
            lines.push(format!("        arity: {},", merkle_arity));
            lines.push(format!("        n_fri_queries: {},", fri.num_queries));
            lines.push(format!("        n_fri_steps: {},", fri.log_domain_sizes.len()));
            lines.push(format!("        n_challenges: {},", stark_info.challenges_map.len()));
            // Stage challenges, one r^fold per fold (M = domains − 1) and the query seed.
            lines.push(format!(
                "        n_challenges_total: {},",
                stark_info.challenges_map.len() + fri.log_domain_sizes.len()
            ));

            lines.push(format!("        fri_steps: vec![{}],", vec_str(&fri.log_domain_sizes)));

            lines.push(format!("        hash_commits: {},", stark_info.stark_struct.hash_commits));
            lines
                .push(format!("        last_level_verification: {},", stark_info.stark_struct.last_level_verification));
            lines.push(format!("        pow_bits: {},", fri.grinding_bits_queries));

            lines.push(format!("        num_vals: vec![{}],", num_vals.join(", ")));
            lines.push(format!("        opening_points: vec![{}],", opening_points_str.join(", ")));
            lines.push(format!("        boundaries: vec![{}],", boundary_strs.join(", ")));
            lines.push(format!("        q_deg: {},", stark_info.q_deg));
            lines.push(format!("        q_index: {},", q_ev_index));

            lines.push("    }".to_string());
            lines.push("}\n".to_string());
        }
        LowDegreeTest::Stir(stir) => {
            let m = stir.num_iterations();
            lines.push("fn verifier_info() -> StirVerifierInfo {".to_string());
            lines.push("    StirVerifierInfo {".to_string());
            lines.push(format!("        n_stages: {},", stark_info.n_stages));
            lines.push(format!("        n_constants: {},", stark_info.n_constants));
            lines.push(format!("        n_evals: {},", stark_info.ev_map.len()));
            lines.push(format!("        n_bits: {},", stark_info.stark_struct.n_bits));
            lines.push(format!("        n_bits_ext: {},", stark_info.stark_struct.n_bits_ext));
            lines.push(format!("        n_challenges: {},", stark_info.challenges_map.len()));
            // r_fold_0, then (r_out, r_fold, r_comb) per iteration 1..M−1, then the M query
            // challenges — `num_ldt_challenges` in common/src/stark_info.rs.
            lines.push(format!(
                "        n_challenges_total: {},",
                stark_info.challenges_map.len() + 1 + (m - 1) * 3 + m
            ));

            lines.push(format!("        num_vals: vec![{}],", num_vals.join(", ")));
            lines.push(format!("        opening_points: vec![{}],", opening_points_str.join(", ")));
            lines.push(format!("        boundaries: vec![{}],", boundary_strs.join(", ")));
            lines.push(format!("        q_deg: {},", stark_info.q_deg));
            lines.push(format!("        q_index: {},", q_ev_index));

            lines.push("        stir: StirParams {".to_string());
            lines.push(format!("            folding_factors: vec![{}],", vec_str(&stir.folding_factors)));
            lines.push(format!("            log_degrees: vec![{}],", vec_str(&stir.log_degrees)));
            lines.push(format!("            log_domain_sizes: vec![{}],", vec_str(&stir.log_domain_sizes)));
            lines.push(format!("            num_queries: vec![{}],", vec_str(&stir.num_queries)));
            lines.push(format!("            grinding_bits_queries: vec![{}],", vec_str(&stir.grinding_bits_queries)));
            lines.push(format!("            arity: {},", merkle_arity));
            lines.push(format!(
                "            last_level_verification: {},",
                stark_info.stark_struct.last_level_verification
            ));
            lines.push(format!("            hash_commits: {},", stark_info.stark_struct.hash_commits));
            lines.push("        },".to_string());

            lines.push("    }".to_string());
            lines.push("}\n".to_string());
        }
    }

    // verify function. Generics: leaf, compression, transcript, grinding hashes.
    let generics = format!("{merkle_hash_type}, {merkle_hash_type}, {transcript_hash_type}, {grinding_type}");
    let verify_fn = if is_stir { "stark_verify_stir" } else { "stark_verify" };
    if vadcop_final_proof {
        lines.push("pub fn verify(proof: &VadcopFinalProof, vk: &[u64]) -> bool {".to_string());
        lines.push(format!(
            "    {verify_fn}::<{generics}>(&proof.proof_with_publics(), vk, &verifier_info(), q_verify, query_verify)"
        ));
        lines.push("}\n".to_string());
        lines.push("pub fn verify_u64(proof: &[u64], vk: &[u64]) -> bool {".to_string());
        lines.push(format!("    {verify_fn}::<{generics}>(proof, vk, &verifier_info(), q_verify, query_verify)"));
        lines.push("}\n".to_string());
    } else {
        lines.push("pub fn verify(proof: &[u64], vk: &[u64]) -> bool {".to_string());
        lines.push(format!("    {verify_fn}::<{generics}>(proof, vk, &verifier_info(), q_verify, query_verify)"));
        lines.push("}\n".to_string());
    }

    let expected_fn = if is_stir { "expected_stir_proof_size_bytes" } else { "expected_proof_size_bytes" };
    lines.push("pub fn expected_proof_bytes() -> usize {".to_string());
    lines.push(format!("    crate::{expected_fn}(&verifier_info())"));
    lines.push("}\n".to_string());

    Ok(lines.join("\n"))
}
