use anyhow::Result;
use std::fs;

use crate::io::parser_args::get_parser_args;
use crate::types::stark_info::{StarkInfo, VerifierInfo};

/// Generates a Rust source file containing `q_verify()`, `query_verify()`,
/// `verifier_info()`, and `verify()` functions for the STARK verifier.
///
/// Port of `writeVerifierRustFile` from binFile.js.
pub fn write_verifier_rust_file(
    path: &str,
    stark_info: &StarkInfo,
    verifier_info: &VerifierInfo,
    vadcop_final_proof: bool,
) -> Result<()> {
    println!("> Writing rust verifier file");

    let rust_verifier = prepare_verifier_rust(stark_info, verifier_info, vadcop_final_proof)?;
    fs::write(path, rust_verifier)?;

    Ok(())
}

fn prepare_verifier_rust(
    stark_info: &StarkInfo,
    verifier_info: &VerifierInfo,
    vadcop_final_proof: bool,
) -> Result<String> {
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

    let arity = stark_info.stark_struct.merkle_tree_arity;
    let poseidon_width = arity * 4;

    let mut lines: Vec<String> = Vec::new();

    // Imports
    lines.push(format!("use fields::{{Goldilocks, CubicExtensionField, Field, Poseidon{}}};", poseidon_width));
    lines.push("use crate::{Boundary, VerifierInfo, stark_verify};".to_string());
    if vadcop_final_proof {
        lines.push("use proofman_util::VadcopFinalProof;".to_string());
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

    // verifier_info function
    lines.push("#[rustfmt::skip]".to_string());
    lines.push("fn verifier_info() -> VerifierInfo {".to_string());
    lines.push("    VerifierInfo {".to_string());
    lines.push(format!("        n_stages: {},", stark_info.n_stages));
    lines.push(format!("        n_constants: {},", stark_info.n_constants));
    lines.push(format!("        n_evals: {},", stark_info.ev_map.len()));
    lines.push(format!("        n_bits: {},", stark_info.stark_struct.n_bits));
    lines.push(format!("        n_bits_ext: {},", stark_info.stark_struct.n_bits_ext));
    lines.push(format!("        arity: {},", arity));
    lines.push(format!("        n_fri_queries: {},", stark_info.stark_struct.n_queries));
    lines.push(format!("        n_fri_steps: {},", stark_info.stark_struct.steps.len()));
    lines.push(format!("        n_challenges: {},", stark_info.challenges_map.len()));
    lines.push(format!(
        "        n_challenges_total: {},",
        stark_info.challenges_map.len() + stark_info.stark_struct.steps.len() + 1
    ));

    let fri_steps_str: Vec<String> = stark_info.stark_struct.steps.iter().map(|s| s.n_bits.to_string()).collect();
    lines.push(format!("        fri_steps: vec![{}],", fri_steps_str.join(", ")));

    lines.push(format!("        hash_commits: {},", stark_info.stark_struct.hash_commits));
    lines.push(format!("        last_level_verification: {},", stark_info.stark_struct.last_level_verification));
    lines.push(format!("        pow_bits: {},", stark_info.stark_struct.pow_bits));

    let mut num_vals: Vec<String> = Vec::new();
    for i in 0..stark_info.n_stages + 1 {
        let key = format!("cm{}", i + 1);
        let val = stark_info.map_sections_n.get(&key).copied().unwrap_or(0);
        num_vals.push(val.to_string());
    }
    lines.push(format!("        num_vals: vec![{}],", num_vals.join(", ")));

    let opening_points_str: Vec<String> = stark_info.opening_points.iter().map(|p| p.to_string()).collect();
    lines.push(format!("        opening_points: vec![{}],", opening_points_str.join(", ")));

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
    lines.push(format!("        boundaries: vec![{}],", boundary_strs.join(", ")));

    lines.push(format!("        q_deg: {},", stark_info.q_deg));

    // Find q_index: the evMap index of the cm polynomial at stage nStages+1, stageId 0
    let q_index = stark_info.cm_pols_map.iter().position(|p| p.stage == stark_info.n_stages + 1 && p.stage_id == 0);
    let q_ev_index = if let Some(qi) = q_index {
        stark_info.ev_map.iter().position(|ev| ev.ev_type == "cm" && ev.id == qi as u64).unwrap_or(0)
    } else {
        0
    };
    lines.push(format!("        q_index: {},", q_ev_index));

    lines.push("    }".to_string());
    lines.push("}\n".to_string());

    // verify function
    if vadcop_final_proof {
        lines.push("pub fn verify(proof: &VadcopFinalProof, vk: &[u8]) -> bool {".to_string());
        lines.push(format!(
            "    stark_verify::<Poseidon{}, {}>(&proof.proof_with_publics(), vk, &verifier_info(), q_verify, query_verify)",
            poseidon_width, poseidon_width
        ));
        lines.push("}\n".to_string());
        lines.push("pub fn verify_bytes(proof: &[u8], vk: &[u8]) -> bool {".to_string());
        lines.push(format!(
            "    stark_verify::<Poseidon{}, {}>(proof, vk, &verifier_info(), q_verify, query_verify)",
            poseidon_width, poseidon_width
        ));
        lines.push("}\n".to_string());
    } else {
        lines.push("pub fn verify(proof: &[u8], vk: &[u8]) -> bool {".to_string());
        lines.push(format!(
            "    stark_verify::<Poseidon{}, {}>(proof, vk, &verifier_info(), q_verify, query_verify)",
            poseidon_width, poseidon_width
        ));
        lines.push("}\n".to_string());
    }

    Ok(lines.join("\n"))
}
