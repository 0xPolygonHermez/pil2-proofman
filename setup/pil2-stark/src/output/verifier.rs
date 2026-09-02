use anyhow::Result;
use std::fs;

use proofman_common::hash_family;
use crate::io::parser_args::get_parser_args;
use crate::types::stark_info::{StarkInfo, VerifierInfo};

pub fn write_verifier_rust_file(
    path: &str,
    stark_info: &StarkInfo,
    verifier_info: &VerifierInfo,
    vadcop_final_proof: bool,
    hash_id: &str,
) -> Result<()> {
    if !hash_family::supports_native_rust_verifier(hash_id) {
        tracing::warn!(
            "Skipping the native Rust verifier for {path}: the {hash_id} family has no \
             proofman_fields::Hash implementation, so no verifier can be emitted for it. The proving \
             key is complete for proving and for C++ verification; only this artifact is absent."
        );
        return Ok(());
    }

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
    // The transcript CONSTRUCTION, not a hash to wrap in a sponge: BLAKE3's is not a sponge.
    let transcript_type = hash_family::rust_transcript_type(hash_id, transcript_arity as u64);
    let grinding_type = hash_family::rust_grinding_type(hash_id);
    let mut numbers_q = Vec::new();
    let q_result =
        get_parser_args(stark_info, &verifier_info.q_verifier.code, &mut numbers_q, false, true, None, "q_verify")?;
    let verify_q_rust = q_result.verify_rust;
    let verify_q_helpers = q_result.verify_rust_helpers;

    let mut lines: Vec<String> = Vec::new();

    lines.push("use alloc::vec;".to_string());
    lines.push("use alloc::string::ToString;".to_string());
    let mut hash_imports: Vec<&str> = Vec::new();
    for ht in [merkle_hash_type, grinding_type]
        .into_iter()
        .chain(hash_family::rust_transcript_imports(hash_id, transcript_arity as u64))
    {
        if !hash_imports.contains(&ht) {
            hash_imports.push(ht);
        }
    }
    lines
        .push(format!("use proofman_fields::{{Goldilocks, CubicExtensionField, Field, {}}};", hash_imports.join(", ")));
    lines.push("use crate::{stark_verify, Boundary, FriEvalGroup, FriEvalRef, VerifierInfo};".to_string());
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

    // verifier_info function
    lines.push("#[rustfmt::skip]".to_string());
    lines.push("fn verifier_info() -> VerifierInfo {".to_string());
    lines.push("    VerifierInfo {".to_string());
    lines.push(format!("        n_stages: {},", stark_info.n_stages));
    lines.push(format!("        n_constants: {},", stark_info.n_constants));
    lines.push(format!("        n_evals: {},", stark_info.ev_map.len()));
    lines.push(format!("        n_bits: {},", stark_info.stark_struct.n_bits));
    lines.push(format!("        n_bits_ext: {},", stark_info.stark_struct.n_bits_ext));
    lines.push(format!("        arity: {},", merkle_arity));
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

    // The FRI query polynomial is structural, so the verifier evaluates it from
    // the evaluation map instead of an unrolled per-air function.
    let mut group_strs: Vec<String> = Vec::new();
    let mut next_eval = 0usize;
    for (o, opening) in stark_info.opening_points.iter().enumerate() {
        let mut ref_strs: Vec<String> = Vec::new();
        for (i, ev) in stark_info.ev_map.iter().enumerate() {
            if ev.prime != *opening {
                continue;
            }
            let (bucket, offset, dim) = match ev.ev_type.as_str() {
                "const" => (0u64, ev.id, 1u64),
                "cm" => {
                    let pol = &stark_info.cm_pols_map[ev.id as usize];
                    (pol.stage, pol.stage_pos, pol.dim)
                }
                "custom" => (stark_info.n_stages + 1 + ev.commit_id, ev.id, 1u64),
                other => panic!("Unknown evMap type: {other}"),
            };
            // The runtime indexes `evals` with a running counter.
            assert_eq!(i, next_eval, "evMap is not ordered by opening point");
            next_eval += 1;
            ref_strs.push(format!("FriEvalRef::new({bucket}, {offset}, {dim})"));
        }
        // Opening points from hint expressions have no evaluations.
        if ref_strs.is_empty() {
            continue;
        }
        group_strs.push(format!("FriEvalGroup {{ opening: {o}, refs: vec![{}] }}", ref_strs.join(", ")));
    }
    lines.push(format!("        fri_ev_groups: vec![\n            {}\n        ],", group_strs.join(",\n            ")));

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

    // verify function. Generics: leaf, compression, transcript, grinding hashes.
    let generics = format!("{merkle_hash_type}, {merkle_hash_type}, {transcript_type}, {grinding_type}");
    if vadcop_final_proof {
        lines.push("pub fn verify(proof: &VadcopFinalProof, vk: &[u64]) -> bool {".to_string());
        lines.push(format!(
            "    stark_verify::<{generics}>(&proof.proof_with_publics(), vk, &verifier_info(), q_verify)"
        ));
        lines.push("}\n".to_string());
        lines.push("pub fn verify_u64(proof: &[u64], vk: &[u64]) -> bool {".to_string());
        lines.push(format!("    stark_verify::<{generics}>(proof, vk, &verifier_info(), q_verify)"));
        lines.push("}\n".to_string());
    } else {
        lines.push("pub fn verify(proof: &[u64], vk: &[u64]) -> bool {".to_string());
        lines.push(format!("    stark_verify::<{generics}>(proof, vk, &verifier_info(), q_verify)"));
        lines.push("}\n".to_string());
    }

    lines.push("pub fn expected_proof_bytes() -> usize {".to_string());
    lines.push("    crate::expected_proof_size_bytes(&verifier_info())".to_string());
    lines.push("}\n".to_string());

    Ok(lines.join("\n"))
}
