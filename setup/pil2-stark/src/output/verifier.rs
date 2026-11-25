use anyhow::{anyhow, bail, ensure, Result};
use std::collections::HashMap;
use std::fs;

use pil_info::core::dag::lower_code::{lower_to_code, CodeOp, Fold, Instr, LowerCtx, Operand, VerifierCode};
use proofman_common::{hash_family, LowDegreeLayout};
use crate::types::goldilocks_u64;
use crate::types::stark_info::{StarkInfo, VerifierInfo};
use crate::types::stark_struct::LowDegreeTest;

pub fn write_verifier_rust_file(
    path: &str,
    stark_info: &StarkInfo,
    verifier_info: &VerifierInfo,
    hash_id: &str,
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

    let rust_verifier = prepare_verifier_rust(stark_info, verifier_info, hash_id)?;
    fs::write(path, rust_verifier)?;

    Ok(())
}

/// `challengesMap` position of vf1 (`stage_id` 0) or vf2 (1): the stage `nStages + 3` challenge with that
/// `stageId`, the same rule as the C++ `buildDeepTables`.
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

fn prepare_verifier_rust(stark_info: &StarkInfo, verifier_info: &VerifierInfo, hash_id: &str) -> Result<String> {
    // Leaf + compression hashes follow the Merkle tree arity; the transcript hash
    // follows the (fixed, arity-4) transcript arity; grinding is a fixed-width hash.
    let merkle_arity = stark_info.stark_struct.merkle_tree_arity;
    let transcript_arity = stark_info.stark_struct.transcript_arity;
    let merkle_hash_type = hash_family::rust_hash_type(hash_id, merkle_arity as u64);
    // The transcript CONSTRUCTION, not a hash to wrap in a sponge: BLAKE3's is not a sponge.
    let transcript_type = hash_family::rust_transcript_type(hash_id, transcript_arity as u64);
    let grinding_type = hash_family::rust_grinding_type(hash_id);
    let q = q_verify_of(stark_info, verifier_info)?;
    let (verify_q_rust, verify_q_helpers) = (q.verify_rust, q.verify_rust_helpers);

    let mut lines: Vec<String> = Vec::new();

    lines.push("use alloc::vec;".to_string());
    let mut hash_imports: Vec<&str> = Vec::new();
    for ht in [merkle_hash_type, grinding_type]
        .into_iter()
        .chain(hash_family::rust_transcript_imports(hash_id, transcript_arity as u64))
    {
        if !hash_imports.contains(&ht) {
            hash_imports.push(ht);
        }
    }
    let is_stir = matches!(stark_info.stark_struct.low_degree_test, LowDegreeTest::Stir(_));
    lines
        .push(format!("use proofman_fields::{{Goldilocks, CubicExtensionField, Field, {}}};", hash_imports.join(", ")));
    if is_stir {
        lines.push(
            "use crate::{stark_verify_stir, FriEvalGroup, FriEvalRef, StirParams, StirVerifierInfo};".to_string(),
        );
    } else {
        lines.push("use crate::{stark_verify, FriEvalGroup, FriEvalRef, VerifierInfo};".to_string());
    }
    lines.push("use crate::VadcopFinalProof;".to_string());
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

    // verify function. Generics: leaf, compression, transcript, grinding hashes.
    let generics = format!("{merkle_hash_type}, {merkle_hash_type}, {transcript_type}, {grinding_type}");
    let verify_fn = if is_stir { "stark_verify_stir" } else { "stark_verify" };
    if stark_info.multiproof {
        lines.push("pub fn verify(proof: &VadcopFinalProof, vk: &[u64]) -> bool {".to_string());
        lines.push("    verify_u64(&proof.proof_with_publics(), vk)".to_string());
        lines.push("}\n".to_string());
        lines.push("pub fn verify_u64(proof: &[u64], vk: &[u64]) -> bool {".to_string());
        lines.push("    let info = verifier_info();".to_string());
        lines.push(format!(
            "    let Some(flat) = crate::expand_proof::<{merkle_hash_type}, {merkle_hash_type}, {transcript_type}>(proof, vk, &info.proof_layout()) else {{"
        ));
        lines.push("        return false;".to_string());
        lines.push("    };".to_string());
        lines.push(format!("    {verify_fn}::<{generics}>(&flat, vk, &info, q_verify)"));
        lines.push("}\n".to_string());
    } else {
        lines.push("pub fn verify(proof: &VadcopFinalProof, vk: &[u64]) -> bool {".to_string());
        lines.push(format!(
            "    {verify_fn}::<{generics}>(&proof.proof_with_publics(), vk, &verifier_info(), q_verify)"
        ));
        lines.push("}\n".to_string());
        lines.push("pub fn verify_u64(proof: &[u64], vk: &[u64]) -> bool {".to_string());
        lines.push(format!("    {verify_fn}::<{generics}>(proof, vk, &verifier_info(), q_verify)"));
        lines.push("}\n".to_string());
    }

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

/// The Rust source of a generated `q_verify`: the chunk helpers and the function body.
#[derive(Debug, Clone, Default)]
pub struct VerifyRust {
    pub verify_rust_helpers: Vec<String>,
    pub verify_rust: Vec<String>,
}

/// The DAG lowered (STARK fold) and rendered as `q_verify`.
fn q_verify_of(stark_info: &StarkInfo, verifier_info: &VerifierInfo) -> Result<VerifyRust> {
    let ctx = LowerCtx::new(stark_info.lower_tables.clone()).map_err(|e| anyhow!("{e:?}"))?;
    let code = lower_to_code(&verifier_info.constraint_dag, &ctx, Fold::Stark).map_err(|e| anyhow!("{e:?}"))?;
    rust_verify_lines(stark_info, &code, "q_verify")
}

/// `(dim, slot)` of each instruction's destination and the slots per kind (base, ext): a slot is reused after
/// the last read of its temporary; sources are read before the destination is written.
fn dest_slots(code: &[Instr], ext: usize) -> (Vec<(usize, usize)>, [usize; 2]) {
    let mut last_read: HashMap<u32, usize> = HashMap::new();
    for (i, ins) in code.iter().enumerate() {
        for s in &ins.src {
            if let Operand::Tmp(k) = s.operand {
                last_read.insert(k, i);
            }
        }
    }
    let mut live: HashMap<u32, (usize, usize)> = HashMap::new();
    let (mut free, mut count): ([Vec<usize>; 2], [usize; 2]) = Default::default();
    let mut out = Vec::with_capacity(code.len());
    for (i, ins) in code.iter().enumerate() {
        for s in &ins.src {
            if let Operand::Tmp(k) = s.operand {
                if last_read[&k] == i {
                    if let Some((d, slot)) = live.remove(&k) {
                        free[usize::from(d == ext)].push(slot);
                    }
                }
            }
        }
        let kind = usize::from(ins.dim == ext);
        let slot = free[kind].pop().unwrap_or_else(|| {
            count[kind] += 1;
            count[kind] - 1
        });
        // a result never read (but the last) frees its slot at once
        if last_read.contains_key(&ins.dest) || i + 1 == code.len() {
            live.insert(ins.dest, (ins.dim, slot));
        } else {
            free[kind].push(slot);
        }
        out.push((ins.dim, slot));
    }
    (out, count)
}

/// `q_verify`'s Rust source: one assignment per instruction, chunked, temporaries packed by `dest_slots`.
pub fn rust_verify_lines(stark_info: &StarkInfo, code: &VerifierCode, function_name: &str) -> Result<VerifyRust> {
    let ext = crate::types::FIELD_EXTENSION;
    code.check_dims(ext).map_err(|e| anyhow!("{function_name}: {e:?}"))?;
    let (dests, count) = dest_slots(&code.code, ext);
    let mut slots: HashMap<u32, (usize, usize)> = HashMap::new();
    let mut lines: Vec<String> = Vec::with_capacity(code.code.len());
    let mut uses: Vec<Uses> = Vec::with_capacity(code.code.len());
    for (ins, &(dim, slot)) in code.code.iter().zip(&dests) {
        let mut u = Uses::default();
        let mut args: Vec<(String, usize)> = Vec::with_capacity(2);
        for s in &ins.src {
            let e = match s.operand {
                Operand::Tmp(k) => {
                    let (d, slot) = slots[&k];
                    format!("tmp_{d}[{slot}]")
                }
                Operand::Eval(k) => format!("evals[{k}]"),
                Operand::Challenge { id, .. } => format!("challenges[{id}]"),
                Operand::Number(ref v) => format!("Goldilocks::new({})", goldilocks_u64(v)?),
                Operand::Public(id) => {
                    ensure!((id as u64) < stark_info.n_publics, "tmp {}: public {id} is out of range", ins.dest);
                    format!("_publics[{id}]")
                }
                Operand::Zi { boundary } => {
                    ensure!(boundary == 0, "tmp {}: Zi {boundary}: q_verify has only the everyRow zerofier", ins.dest);
                    String::from("zi[0]")
                }
                Operand::ProofValue { .. } | Operand::AirValue { .. } | Operand::AirgroupValue { .. } => {
                    bail!("tmp {}: q_verify has no {:?} input", ins.dest, s.operand)
                }
            };
            u.add(&s.operand, s.dim == ext);
            args.push((e, s.dim));
        }
        slots.insert(ins.dest, (dim, slot));
        u.tmp[usize::from(dim == ext)] = true;
        let dest = format!("tmp_{dim}[{slot}]");
        // the scalar goes on the right: CubicExtensionField implements ext op base
        let line = match (ins.op, args.as_slice()) {
            (CodeOp::Copy, [a]) => format!("    {dest} = {};", a.0),
            (op, [a, b]) => {
                let swap = a.1 == 1 && b.1 == ext;
                let (x, y) = if swap { (&b.0, &a.0) } else { (&a.0, &b.0) };
                match op {
                    CodeOp::Add => format!("    {dest} = {x} + {y};"),
                    CodeOp::Mul => format!("    {dest} = {x} * {y};"),
                    CodeOp::Sub if swap => format!("    {dest} = {x}.sub_from_scalar({y});"),
                    _ => format!("    {dest} = {x} - {y};"),
                }
            }
            _ => unreachable!("operand count checked above"),
        };
        lines.push(line);
        uses.push(u);
    }
    let last = code.code.last().ok_or_else(|| anyhow!("{function_name}: no verifier code"))?;
    let &(dim, slot) = &slots[&last.dest];
    ensure!(dim == ext, "{function_name}: the result is not an extension element");

    let mut out = VerifyRust::default();
    let [count1d, count3d] = count;
    if count1d > 0 {
        out.verify_rust.push(format!("    let mut tmp_1 = vec![Goldilocks::ZERO; {count1d}];"));
    }
    if count3d > 0 {
        out.verify_rust.push(format!(
            "    let mut tmp_3 = vec![CubicExtensionField {{ value: [Goldilocks::ZERO, Goldilocks::ZERO, Goldilocks::ZERO] }}; {count3d}];"
        ));
    }
    if lines.len() <= RUST_VERIFIER_CHUNK_SIZE {
        out.verify_rust.extend(lines);
    } else {
        for (i, (body, us)) in
            lines.chunks(RUST_VERIFIER_CHUNK_SIZE).zip(uses.chunks(RUST_VERIFIER_CHUNK_SIZE)).enumerate()
        {
            let used = us.iter().fold(Uses::default(), |a, b| a.or(b));
            // a chunk that does not touch a tmp vector still takes it, underscored
            let (mut params, mut call) = (Vec::new(), Vec::new());
            for (k, n, ty, present) in [
                (0, "tmp_1", "&mut [Goldilocks]", count1d > 0),
                (1, "tmp_3", "&mut [CubicExtensionField<Goldilocks>]", count3d > 0),
            ] {
                if present {
                    let u = if used.tmp[k] { "" } else { "_" };
                    params.push(format!("{u}{n}: {ty}"));
                    call.push(format!("&mut {n}"));
                }
            }
            for (n, ty, present) in [
                ("challenges", "&[CubicExtensionField<Goldilocks>]", used.challenges),
                ("evals", "&[CubicExtensionField<Goldilocks>]", used.evals),
                ("_publics", "&[Goldilocks]", used.publics),
                ("zi", "&[CubicExtensionField<Goldilocks>]", used.zi),
            ] {
                if present {
                    params.push(format!("{n}: {ty}"));
                    call.push(n.to_string());
                }
            }
            let name = format!("{function_name}_chunk_{i}");
            out.verify_rust_helpers
                .extend(["#[inline(never)]", "#[rustfmt::skip]", "#[allow(clippy::all)]"].map(String::from));
            out.verify_rust_helpers.push(format!("fn {name}({}) {{", params.join(", ")));
            out.verify_rust_helpers.extend(body.iter().cloned());
            out.verify_rust_helpers.push("}\n".to_string());
            out.verify_rust.push(format!("    {name}({});", call.join(", ")));
        }
    }
    out.verify_rust.push(format!("    return tmp_3[{slot}];"));
    Ok(out)
}

/// The `q_verify` inputs and temporary vectors (`tmp[0]` base, `tmp[1]` ext) a block of lines reads or writes.
#[derive(Default, Clone, Copy)]
struct Uses {
    challenges: bool,
    evals: bool,
    publics: bool,
    zi: bool,
    tmp: [bool; 2],
}

impl Uses {
    fn add(&mut self, o: &Operand, ext: bool) {
        match o {
            Operand::Challenge { .. } => self.challenges = true,
            Operand::Eval(_) => self.evals = true,
            Operand::Public(_) => self.publics = true,
            Operand::Zi { .. } => self.zi = true,
            Operand::Tmp(_) => self.tmp[usize::from(ext)] = true,
            _ => {}
        }
    }

    fn or(self, o: &Uses) -> Uses {
        Uses {
            challenges: self.challenges | o.challenges,
            evals: self.evals | o.evals,
            publics: self.publics | o.publics,
            zi: self.zi | o.zi,
            tmp: [self.tmp[0] | o.tmp[0], self.tmp[1] | o.tmp[1]],
        }
    }
}

/// Number of verifier operations per `#[inline(never)]` helper chunk (matches JS).
const RUST_VERIFIER_CHUNK_SIZE: usize = 100;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::output::test_support::full_random;
    use pil_info::core::dag::format::{DagBoundary, DagNode, DagRoot};
    use pil_info::core::dag::lower_code::{CodeOp, Instr, Operand, Src, VerifierCode};

    fn tmp_assignments(lines: &[String]) -> Vec<String> {
        let is_tmp = |l: &&String| {
            let t = l.trim_start();
            (t.starts_with("tmp_1[") || t.starts_with("tmp_3["))
                && t.split(']').nth(1).is_some_and(|r| r.trim_start().starts_with('='))
        };
        lines.iter().filter(is_tmp).map(|l| l.trim_start().split(']').next().unwrap().to_string()).collect()
    }

    #[test]
    fn q_verify_one_assignment_per_op_node() {
        for f in full_random(3) {
            let dag = &f.vi.constraint_dag;
            let ctx = LowerCtx::new(f.si.lower_tables.clone()).unwrap();
            let code = lower_to_code(dag, &ctx, Fold::Stark).unwrap();
            // op nodes + fold instructions (copy of leaf roots, mul/add per root, final mul by Zi)
            assert_eq!(code.code.len(), code.tmp_used);
            let r = rust_verify_lines(&f.si, &code, "q_verify").unwrap();
            let all: Vec<String> = r.verify_rust_helpers.iter().chain(&r.verify_rust).cloned().collect();
            assert_eq!(tmp_assignments(&all).len(), code.code.len());
            // replay the packing: every read finds its temporary still in its slot
            let (dests, count) = dest_slots(&code.code, 3);
            let mut owner: HashMap<(usize, usize), u32> = HashMap::new();
            let mut at: HashMap<u32, (usize, usize)> = HashMap::new();
            for (ins, &d) in code.code.iter().zip(&dests) {
                for s in &ins.src {
                    if let Operand::Tmp(k) = s.operand {
                        assert_eq!(owner[&at[&k]], k, "tmp {k} overwritten before its read by {}", ins.dest);
                    }
                }
                owner.insert(d, ins.dest);
                at.insert(ins.dest, d);
            }
            assert!(count[0] + count[1] < code.code.len(), "no slot was reused");
        }
    }

    #[test]
    fn q_verify_chunks_at_100() {
        let ev = |i| Src { operand: Operand::Eval(i), dim: 3 };
        let n = 250;
        let code: Vec<Instr> = (0..n)
            .map(|k| {
                let prev = if k == 0 { ev(1) } else { Src { operand: Operand::Tmp(k - 1), dim: 3 } };
                Instr { op: CodeOp::Add, dest: k, dim: 3, src: vec![ev(0), prev] }
            })
            .collect();
        let vc = VerifierCode { tmp_used: n as usize, code };
        let si = StarkInfo { n_stages: 2, ..Default::default() };
        let r = rust_verify_lines(&si, &vc, "q_verify").unwrap();
        assert_eq!(r.verify_rust_helpers.iter().filter(|l| l.contains("fn q_verify_chunk_")).count(), 3);
        assert_eq!(tmp_assignments(&r.verify_rust_helpers).len(), n as usize);
    }

    #[test]
    fn q_verify_refuses_airvalue() {
        let f = full_random(1).pop().expect("a random key");
        assert!(q_verify_of(&f.si, &f.vi).is_ok(), "baseline: the unmodified key renders");
        let (mut si, mut vi) = (f.si, f.vi);
        if si.lower_tables.air_values_stage.is_empty() {
            si.lower_tables.air_values_stage = vec![2];
        }
        let dag = &mut vi.constraint_dag;
        dag.nodes.push(DagNode::AirValue { id: 0 });
        dag.roots.push(DagRoot {
            node: dag.nodes.len() as u32 - 1,
            boundary: DagBoundary::EveryRow,
            label: "av".into(),
        });
        let e = q_verify_of(&si, &vi).expect_err("an airvalue cannot be supplied to q_verify");
        assert!(format!("{e:#}").to_lowercase().contains("airvalue"), "{e:#}");
    }

    #[test]
    fn q_verify_refuses_non_every_row_root() {
        let f = full_random(1).pop().expect("a random key");
        let mut vi = f.vi;
        vi.constraint_dag.roots[0].boundary = DagBoundary::FirstRow;
        let e = q_verify_of(&f.si, &vi).expect_err("a STARK root is everyRow");
        assert!(format!("{e:#}").contains("everyRow"), "{e:#}");
    }

    // Dimension corruption: an ext-typed instruction over a base-field operand, and the reverse.
    #[test]
    fn q_verify_refuses_dim_mismatch() {
        let si = StarkInfo { n_stages: 2, n_publics: 1, ..Default::default() };
        let src = |operand, dim| Src { operand, dim };
        let run = |dim, a: Src, b: Src| {
            let code = vec![Instr { op: CodeOp::Mul, dest: 0, dim, src: vec![a, b] }];
            rust_verify_lines(&si, &VerifierCode { tmp_used: 1, code }, "q_verify")
        };
        // a public is base-field: typed ext it disagrees with its slot
        assert!(run(3, src(Operand::Public(0), 3), src(Operand::Eval(0), 3)).is_err());
        // an eval is ext: typed base, and a base-typed result over ext operands
        assert!(run(3, src(Operand::Public(0), 1), src(Operand::Eval(0), 1)).is_err());
        assert!(run(1, src(Operand::Eval(0), 3), src(Operand::Eval(1), 3)).is_err());
    }
}
