//! Host-side equivalence check: evaluate two IRs of the same expression on
//! identical random operand values and compare the outputs exactly. Every
//! non-tmp operand (column, constant, challenge, ...) gets one random value
//! per assignment; `Pow{base,j}` is derived from the value of `Ch{base}` so
//! the powers table is modelled exactly as the launcher computes it.

use crate::field::{self, F3, Rng};
use crate::ir::{Ir, Operand};
use std::collections::HashMap;

struct Env {
    rng: Rng,
    leaves: HashMap<Operand, F3>,
    /// hoisted-invariants table of the IR being evaluated (word index -> value)
    tab: HashMap<u64, F3>,
}

impl Env {
    fn value(&mut self, opnd: &Operand, tmps: &HashMap<u64, F3>) -> F3 {
        match opnd {
            Operand::Tmp { id, .. } => *tmps.get(id).unwrap_or_else(|| panic!("use of undefined tmp t{id}")),
            Operand::Num(v) => F3::base(*v),
            Operand::Pow { base, j } => {
                let ch = self.value(&Operand::Ch { base: *base }, tmps);
                field::pow3(ch, *j)
            }
            Operand::Tab { idx, .. } => *self.tab.get(idx).unwrap_or_else(|| panic!("use of unset table word {idx}")),
            leaf => {
                let dim = leaf.dim();
                if let Some(v) = self.leaves.get(leaf) {
                    return *v;
                }
                let v = self.rng.f3(dim);
                self.leaves.insert(leaf.clone(), v);
                v
            }
        }
    }
}

/// Evaluate `ir` under `env`; returns the value of the explicit output store
/// (or of the last tmp when the expression ends in one) and its dim.
fn eval(ir: &Ir, env: &mut Env) -> (F3, u64) {
    // The per-launch table program first, exactly as the table kernel runs it.
    env.tab.clear();
    let mut ttmps: HashMap<u64, F3> = HashMap::new();
    for instr in &ir.tab {
        let a = env.value(&instr.a, &ttmps);
        let b = env.value(&instr.b, &ttmps);
        let v = field::apply(&instr.op, a, instr.a.dim(), b, instr.b.dim());
        ttmps.insert(instr.dst_id.expect("table op without dst"), v);
    }
    for &(tmp, idx, _dim) in &ir.tab_out {
        env.tab.insert(idx, ttmps[&tmp]);
    }
    let mut tmps: HashMap<u64, F3> = HashMap::new();
    let mut out: Option<(F3, u64)> = None;
    for instr in &ir.instrs {
        let a = env.value(&instr.a, &tmps);
        let b = env.value(&instr.b, &tmps);
        let v = field::apply(&instr.op, a, instr.a.dim(), b, instr.b.dim());
        if instr.dst_is_tmp {
            tmps.insert(instr.dst_id.unwrap(), v);
            out = Some((v, instr.ddim));
        } else {
            out = Some((v, instr.ddim));
        }
    }
    out.expect("empty IR")
}

/// `Ok(())` if both IRs produce identical outputs on `rounds` random
/// assignments; `Err` with the first mismatch otherwise.
pub fn equivalent(original: &Ir, optimized: &Ir, rounds: u64) -> Result<(), String> {
    for round in 0..rounds {
        let mut env = Env { rng: Rng(0x5eed_0000 + round), leaves: HashMap::new(), tab: HashMap::new() };
        let (vo, do_) = eval(original, &mut env);
        let (vn, dn) = eval(optimized, &mut env);
        // A base-field value stored into a dim-3 output is zero-padded, so
        // compare as F3 after normalizing dims.
        let norm = |v: F3, d: u64| if d == 1 { F3::base(v.a) } else { v };
        if norm(vo, do_) != norm(vn, dn) {
            return Err(format!("round {round}: original={vo:?} (dim {do_}) optimized={vn:?} (dim {dn})"));
        }
    }
    Ok(())
}
