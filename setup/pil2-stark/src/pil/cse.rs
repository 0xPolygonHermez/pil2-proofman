//! Common-subexpression elimination for straight-line expression code.
//!
//! Relies on the code being SSA — each operation writes a fresh `tmp` and reads
//! only already-defined ones — which reduces deduplication to a hash-cons.

use std::collections::HashMap;

use crate::types::output::{CodeEntry, CodeRef};

pub struct CseResult {
    pub code: Vec<CodeEntry>,
    pub removed: usize,
}

/// A source operand reduced to its identity, for hash-consing.
#[derive(PartialEq, Eq, Hash, Clone, PartialOrd, Ord)]
enum Operand {
    Tmp(usize),
    Leaf(Box<LeafKey>),
}

/// Every field that distinguishes one non-`tmp` operand from another. `exp_id` is
/// excluded: it is provenance never read for semantics, and stays true across a merge.
#[derive(PartialEq, Eq, Hash, Clone, PartialOrd, Ord)]
struct LeafKey {
    ref_type: String,
    id: usize,
    dim: usize,
    prime: Option<i64>,
    value: Option<String>,
    stage: Option<usize>,
    stage_id: Option<usize>,
    commit_id: Option<usize>,
    opening: Option<i64>,
    boundary_id: Option<usize>,
    airgroup_id: Option<usize>,
}

fn operand(r: &CodeRef) -> Operand {
    if r.ref_type == "tmp" {
        return Operand::Tmp(r.id);
    }
    Operand::Leaf(Box::new(LeafKey {
        ref_type: r.ref_type.clone(),
        id: r.id,
        dim: r.dim,
        prime: r.prime,
        value: r.value.clone(),
        stage: r.stage,
        stage_id: r.stage_id,
        commit_id: r.commit_id,
        opening: r.opening,
        boundary_id: r.boundary_id,
        airgroup_id: r.airgroup_id,
    }))
}

/// Removes operations recomputing an earlier value, redirecting consumers at the
/// survivor. The final operation always survives: it names the block's result.
pub fn cse_code(code: &[CodeEntry]) -> CseResult {
    // Dropped tmp id -> surviving tmp id holding the same value.
    let mut alias: HashMap<usize, usize> = HashMap::new();
    let mut seen: HashMap<(String, Vec<Operand>), usize> = HashMap::new();
    let mut out: Vec<CodeEntry> = Vec::with_capacity(code.len());

    for (i, instr) in code.iter().enumerate() {
        let is_last = i + 1 == code.len();

        let mut rewritten = instr.clone();
        for r in &mut rewritten.src {
            if r.ref_type == "tmp" {
                if let Some(&canon) = alias.get(&r.id) {
                    r.id = canon;
                }
            }
        }

        let mut key_srcs: Vec<Operand> = rewritten.src.iter().map(operand).collect();
        // Only `add` and `mul` are commutative; `sub` operand order is meaningful.
        if rewritten.op == "add" || rewritten.op == "mul" {
            key_srcs.sort();
        }
        let key = (rewritten.op.clone(), key_srcs);

        if rewritten.dest.ref_type == "tmp" && !is_last {
            if let Some(&canon) = seen.get(&key) {
                alias.insert(rewritten.dest.id, canon);
                continue;
            }
            seen.insert(key, rewritten.dest.id);
        }
        out.push(rewritten);
    }

    let removed = code.len() - out.len();
    CseResult { code: out, removed }
}

#[cfg(test)]
mod tests {
    use super::*;

    const P: u128 = 0xFFFF_FFFF_0000_0001; // Goldilocks

    fn r(ref_type: &str, id: usize) -> CodeRef {
        CodeRef {
            ref_type: ref_type.to_string(),
            id,
            dim: 3,
            prime: None,
            value: None,
            stage: None,
            stage_id: None,
            commit_id: None,
            opening: None,
            boundary_id: None,
            airgroup_id: None,
            exp_id: None,
        }
    }

    fn tmp(id: usize) -> CodeRef {
        r("tmp", id)
    }

    fn eval(id: usize) -> CodeRef {
        r("eval", id)
    }

    fn op(name: &str, dest: CodeRef, src: Vec<CodeRef>) -> CodeEntry {
        CodeEntry { op: name.to_string(), dest, src }
    }

    /// `mul(eval0, eval1)` computed twice collapses to one operation, and the
    /// consumer of the second copy is redirected at the first.
    #[test]
    fn removes_a_repeated_operation_and_redirects_its_consumer() {
        let code = vec![
            op("mul", tmp(10), vec![eval(0), eval(1)]),
            op("mul", tmp(11), vec![eval(0), eval(1)]),
            op("add", tmp(12), vec![tmp(10), tmp(11)]),
        ];

        let result = cse_code(&code);

        assert_eq!(result.removed, 1);
        assert_eq!(result.code.len(), 2);
        let last = result.code.last().unwrap();
        assert_eq!(last.op, "add");
        assert_eq!(last.src[0].id, 10);
        assert_eq!(last.src[1].id, 10);
    }

    /// The block's result is identified by the last operation's dest, so a
    /// redundant final operation must survive even though its value is a repeat.
    #[test]
    fn never_drops_the_final_operation() {
        let code = vec![op("mul", tmp(10), vec![eval(0), eval(1)]), op("mul", tmp(11), vec![eval(0), eval(1)])];

        let result = cse_code(&code);

        assert_eq!(result.removed, 0);
        assert_eq!(result.code[1].dest.id, 11);
    }

    /// `sub` is not commutative, so swapped operands are different values.
    #[test]
    fn does_not_merge_sub_with_swapped_operands() {
        let code = vec![
            op("sub", tmp(10), vec![eval(0), eval(1)]),
            op("sub", tmp(11), vec![eval(1), eval(0)]),
            op("add", tmp(12), vec![tmp(10), tmp(11)]),
        ];

        assert_eq!(cse_code(&code).removed, 0);
    }

    /// `add` and `mul` are commutative, so swapped operands are the same value.
    #[test]
    fn merges_commutative_ops_with_swapped_operands() {
        let code = vec![
            op("mul", tmp(10), vec![eval(0), eval(1)]),
            op("mul", tmp(11), vec![eval(1), eval(0)]),
            op("add", tmp(12), vec![tmp(10), tmp(11)]),
        ];

        let result = cse_code(&code);

        assert_eq!(result.removed, 1);
        assert_eq!(result.code.last().unwrap().src[0].id, 10);
        assert_eq!(result.code.last().unwrap().src[1].id, 10);
    }

    /// Operands that differ only in a discriminating field are different values.
    #[test]
    fn does_not_merge_operands_differing_only_in_stage() {
        let mut staged = eval(0);
        staged.stage = Some(2);
        let code = vec![
            op("mul", tmp(10), vec![eval(0), eval(1)]),
            op("mul", tmp(11), vec![staged, eval(1)]),
            op("add", tmp(12), vec![tmp(10), tmp(11)]),
        ];

        assert_eq!(cse_code(&code).removed, 0);
    }

    #[test]
    fn handles_an_empty_block() {
        let result = cse_code(&[]);
        assert_eq!(result.removed, 0);
        assert!(result.code.is_empty());
    }

    /// Evaluates a straight-line block in the base field. Two distinct
    /// expressions over the inputs agree at a random point only with
    /// negligible probability, so this detects an unsound merge.
    fn eval_block(code: &[CodeEntry], inputs: &[u64]) -> u64 {
        let mut tmps: HashMap<usize, u128> = HashMap::new();
        let mut last = 0u128;
        for instr in code {
            let val = |r: &CodeRef| -> u128 {
                if r.ref_type == "tmp" {
                    tmps[&r.id]
                } else {
                    inputs[r.id % inputs.len()] as u128
                }
            };
            let (a, b) = (val(&instr.src[0]), val(&instr.src[1]));
            let out = match instr.op.as_str() {
                "add" => (a + b) % P,
                "sub" => (a + P - b) % P,
                "mul" => (a * b) % P,
                other => panic!("unexpected op {other}"),
            };
            tmps.insert(instr.dest.id, out);
            last = out;
        }
        last as u64
    }

    /// Builds a deterministic pseudorandom SSA block that reuses a narrow
    /// operand pool and re-emits earlier operations with operands swapped, so
    /// both genuine repeats and non-commutative near-misses actually occur.
    fn random_block(n_ops: usize, seed: u64) -> Vec<CodeEntry> {
        let mut state = seed;
        let mut next = || {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (state >> 33) as usize
        };
        let ops = ["add", "sub", "mul"];
        let mut code: Vec<CodeEntry> = Vec::new();
        let mut defined: Vec<usize> = Vec::new();
        for i in 0..n_ops {
            let dest_id = 1000 + i;
            let k = next();
            if k % 4 == 0 && !code.is_empty() {
                let prev = &code[k % code.len()];
                let (pa, pb) = (prev.src[1].clone(), prev.src[0].clone());
                let pop = prev.op.clone();
                code.push(op(&pop, tmp(dest_id), vec![pa, pb]));
            } else {
                let pick = |k: usize, defined: &Vec<usize>| -> CodeRef {
                    if defined.is_empty() || k % 3 == 0 {
                        eval(k % 6)
                    } else {
                        tmp(defined[k % defined.len().min(5)])
                    }
                };
                let a = pick(next(), &defined);
                let b = pick(next(), &defined);
                code.push(op(ops[next() % 3], tmp(dest_id), vec![a, b]));
            }
            defined.push(dest_id);
        }
        code
    }

    #[test]
    fn cse_preserves_the_value_the_block_computes() {
        let inputs: Vec<u64> = (0..6).map(|i| 0x9E37_79B9_7F4A_7C15u64.wrapping_mul(i + 1) % P as u64).collect();

        let mut total_removed = 0;
        for seed in 0..40u64 {
            let code = random_block(400, seed);
            let result = cse_code(&code);
            total_removed += result.removed;

            assert_eq!(
                eval_block(&code, &inputs),
                eval_block(&result.code, &inputs),
                "seed {seed}: CSE changed the computed value"
            );
        }
        assert!(total_removed > 0, "test is vacuous: CSE removed nothing");
    }
}
