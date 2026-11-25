//! Tier A: every r1cs constraint is implied by the plonk equations the fixed columns enforce.
//!
//! Soundness (the Lean target). Take any trace satisfying the AIR and set each r1cs wire `x` to
//! `v(var(x))`, the value of its copy class: `var` is the class the exec map places `x` in (verified
//! single by [`super::fixed::labels`]), wire 0 is the constant 1, and an unplaced wire takes the class of
//! a placed one through a pure copy `x = y` of the r1cs, which then holds by construction ("defining").
//! An addition wire `a = c_l·l + c_r·r` is substituted only after its definition is found among the
//! enforced equations, so the substitution is an identity of the trace. Every other r1cs constraint,
//! rewritten over classes, must equal `λ·E` for an enforced equation `E` expanded the same way; `E`
//! holds, so the constraint holds. Hints (map, additions) only steer the search: a wrong one fails it.

use std::collections::{BTreeMap, HashMap, HashSet, VecDeque};

use proofman_common::exec_format::ExecFile;

use super::fixed::{Classes, Columns, Enforced};
use super::poly::{normalize, Lin, Mono, Poly, ONE};
use super::CheckError;
use crate::plonk2pil::field::PlonkField;
use crate::plonk2pil::r1cs::types::{LinearCombination, R1csFile};

/// No class.
pub const NONE: u32 = u32::MAX - 1;
/// A wire with no class: `FREE | wire`, never equal to a class (cells < 2^31, checked).
pub const FREE: u32 = 1 << 31;

/// The class of every r1cs wire, and the copies used to define the unplaced ones.
pub struct Vars {
    pub var: Vec<u32>,
    pub defining: Vec<bool>,
}

impl Vars {
    /// `var(x)`, or a variable no equation has if `x` has no class.
    pub fn of(&self, x: u32) -> u32 {
        match self.var[x as usize] {
            NONE => FREE | x,
            v => v,
        }
    }
}

fn nz<F: PlonkField>(lc: &LinearCombination<F>) -> impl Iterator<Item = (u32, F)> + '_ {
    lc.iter().filter(|(_, c)| !c.is_zero()).map(|(w, c)| (*w, *c))
}

/// `k` if `lc` is the constant `k` (wire 0 only, or empty).
fn constant<F: PlonkField>(lc: &LinearCombination<F>) -> Option<F> {
    let mut k = F::ZERO;
    for (w, c) in nz(lc) {
        if w != 0 {
            return None;
        }
        k += c;
    }
    Some(k)
}

/// The constraint as a linear form `L = 0`, when `A` or `B` is constant.
fn linear<F: PlonkField>(
    a: &LinearCombination<F>,
    b: &LinearCombination<F>,
    c: &LinearCombination<F>,
) -> Option<BTreeMap<u32, F>> {
    let (k, other) = match (constant(a), constant(b)) {
        (Some(k), _) => (k, b),
        (_, Some(k)) => (k, a),
        _ => return None,
    };
    let mut l: BTreeMap<u32, F> = BTreeMap::new();
    for (w, x) in nz(other) {
        *l.entry(w).or_insert(F::ZERO) += k * x;
    }
    for (w, x) in nz(c) {
        *l.entry(w).or_insert(F::ZERO) -= x;
    }
    l.retain(|_, x| !x.is_zero());
    Some(l)
}

/// `var` from the labels, then the unplaced wires through the r1cs's pure copies (breadth first).
pub fn wire_vars<F: PlonkField>(r1cs: &R1csFile<F>, label: &[u32]) -> Vars {
    let n = r1cs.header.n_vars as usize;
    let mut var: Vec<u32> = (0..n)
        .map(|w| if w == 0 { ONE } else { label.get(w).copied().filter(|&l| l != u32::MAX).unwrap_or(NONE) })
        .collect();
    let mut adj: Vec<Vec<(u32, u32)>> = vec![Vec::new(); n];
    for (i, c) in r1cs.constraints.iter().enumerate() {
        let Some(l) = linear(&c.a, &c.b, &c.c) else { continue };
        let t: Vec<(u32, F)> = l.into_iter().collect();
        if let [(x, cx), (y, cy)] = t[..] {
            if x != 0 && y != 0 && cx == -cy && (var[x as usize] == NONE || var[y as usize] == NONE) {
                adj[x as usize].push((y, i as u32));
                adj[y as usize].push((x, i as u32));
            }
        }
    }
    let mut defining = vec![false; r1cs.constraints.len()];
    let mut q: VecDeque<u32> = (0..n as u32).filter(|&w| var[w as usize] != NONE).collect();
    while let Some(x) = q.pop_front() {
        for &(y, i) in &adj[x as usize] {
            if var[y as usize] == NONE {
                var[y as usize] = var[x as usize];
                defining[i as usize] = true;
                q.push_back(y);
            }
        }
    }
    Vars { var, defining }
}

/// Substitution of the verified addition classes.
struct Defs<F>(HashMap<u32, Lin<F>>);

impl<F: PlonkField> Defs<F> {
    fn lin(&self, v: u32, stack: &mut Vec<u32>) -> Lin<F> {
        match self.0.get(&v) {
            Some(d) if !stack.contains(&v) => {
                stack.push(v);
                let mut t = Vec::new();
                for &(u, c) in &d.0 {
                    if u == ONE {
                        t.push((ONE, c));
                    } else {
                        t.extend(self.lin(u, stack).0.into_iter().map(|(w, x)| (w, x * c)));
                    }
                }
                stack.pop();
                Lin::new(t)
            }
            _ => Lin::var(v),
        }
    }

    fn expand(&self, p: &Poly<F>) -> Poly<F> {
        if self.0.is_empty() {
            return p.clone();
        }
        let mut out: Vec<(Mono, F)> = Vec::new();
        let mut st = Vec::new();
        for &((a, b), c) in &p.0 {
            match (a, b) {
                (ONE, ONE) => out.push(((ONE, ONE), c)),
                (a, ONE) => Poly::push_lin(&mut out, c, &self.lin(a, &mut st)),
                (a, b) => Poly::push_prod(&mut out, c, &self.lin(a, &mut st), &self.lin(b, &mut st)),
            }
        }
        Poly::new(out)
    }
}

/// What tier A saw.
#[derive(Debug, Clone, Default)]
pub struct TierA {
    pub constraints: usize,
    pub equations: usize,
    pub unique: usize,
    pub additions: usize,
}

fn lin_of<F: PlonkField>(lc: &LinearCombination<F>, vars: &Vars) -> Lin<F> {
    Lin::new(nz(lc).map(|(w, c)| (vars.of(w), c)).collect())
}

/// `A·B − C` over classes.
pub fn r1cs_poly<F: PlonkField>(c: &crate::plonk2pil::r1cs::types::R1csConstraint<F>, vars: &Vars) -> Poly<F> {
    let mut t = Vec::new();
    Poly::push_prod(&mut t, F::ONE, &lin_of(&c.a, vars), &lin_of(&c.b, vars));
    Poly::push_lin(&mut t, -F::ONE, &lin_of(&c.c, vars));
    Poly::new(t)
}

const BATCH: usize = 1 << 12;

/// Every enforced equation mapped by `f`, the non-zero ones normalized into `sink`; returns the count.
fn equations<F: PlonkField>(
    enf: &Enforced<F>,
    cols: &mut Columns<F>,
    cls: &Classes,
    f: impl Fn(Poly<F>) -> Poly<F>,
    mut sink: impl FnMut(Poly<F>),
) -> Result<usize, CheckError> {
    let (mut batch, mut n) = (Vec::with_capacity(BATCH), 0);
    for r in enf.refs() {
        n += 1;
        let p = f(enf.equation(r, cols, cls)?);
        if !p.is_zero() {
            batch.push(p);
        }
        if batch.len() == BATCH {
            normalize(&mut batch);
            batch.drain(..).for_each(&mut sink);
        }
    }
    normalize(&mut batch);
    batch.into_iter().for_each(sink);
    Ok(n)
}

/// Tier A over the decoded equations `enf`.
pub fn check<F: PlonkField>(
    r1cs: &R1csFile<F>,
    exec: &ExecFile<F>,
    label: &[u32],
    vars: &Vars,
    enf: &Enforced<F>,
    cols: &mut Columns<F>,
    cls: &Classes,
) -> Result<TierA, CheckError> {
    let n_vars = r1cs.header.n_vars;
    // additions: a = c_l·l + c_r·r, accepted once enforced, only for classes no r1cs wire is in
    let r1cs_classes: HashSet<u32> = vars.var.iter().copied().filter(|&v| v != NONE && v != ONE).collect();
    let var_of = |w: u32| {
        if w < n_vars {
            vars.of(w)
        } else {
            label.get(w as usize).copied().filter(|&l| l != u32::MAX).unwrap_or(FREE | w)
        }
    };
    let (mut cand, mut cand_polys) = (Vec::new(), Vec::new());
    for (i, add) in exec.additions.iter().enumerate() {
        let Some(&ca) = label.get((n_vars + i as u32) as usize).filter(|&&l| l != u32::MAX) else { continue };
        if r1cs_classes.contains(&ca) {
            continue;
        }
        let d = Lin::new(vec![(var_of(add.wires[0]), add.coeffs[0]), (var_of(add.wires[1]), add.coeffs[1])]);
        let mut t = vec![((ca, ONE), F::ONE)];
        Poly::push_lin(&mut t, -F::ONE, &d);
        let p = Poly::new(t);
        if !p.is_zero() {
            cand.push((ca, d));
            cand_polys.push(p);
        }
    }
    normalize(&mut cand_polys);
    let mut wanted: HashMap<Poly<F>, Vec<usize>> = HashMap::new();
    for (i, p) in cand_polys.into_iter().enumerate() {
        wanted.entry(p).or_default().push(i);
    }
    let mut found = vec![false; cand.len()];
    equations(
        enf,
        cols,
        cls,
        |p| p,
        |p| {
            for &i in wanted.get(&p).into_iter().flatten() {
                found[i] = true;
            }
        },
    )?;
    let mut defs = Defs(HashMap::new());
    for ((ca, d), _) in cand.into_iter().zip(found).filter(|x| x.1) {
        defs.0.entry(ca).or_insert(d);
    }
    // expanded forms; a definition expands to 0 and drops out
    let mut exp: HashSet<Poly<F>> = HashSet::new();
    let equations = equations(
        enf,
        cols,
        cls,
        |p| defs.expand(&p),
        |p| {
            exp.insert(p);
        },
    )?;
    // the r1cs, each `λ·E` for an enforced `E`
    let mut batch: Vec<(usize, Poly<F>)> = Vec::with_capacity(BATCH);
    let flush = |batch: &mut Vec<(usize, Poly<F>)>| -> Result<(), CheckError> {
        let (idx, mut ps): (Vec<usize>, Vec<Poly<F>>) = batch.drain(..).unzip();
        normalize(&mut ps);
        match idx.into_iter().zip(ps).find(|(_, p)| !exp.contains(p)) {
            Some((i, _)) => {
                Err(CheckError::NotImplied { r1cs: i as u32, why: "no enforced equation implies it".into() })
            }
            None => Ok(()),
        }
    };
    for (i, c) in r1cs.constraints.iter().enumerate() {
        if vars.defining[i] {
            continue;
        }
        let q = r1cs_poly(c, vars);
        if q.is_zero() {
            continue;
        }
        let free = |v: u32| v != ONE && v & FREE != 0;
        if let Some(&((a, b), _)) = q.0.iter().find(|((a, b), _)| free(*a) || free(*b)) {
            let w = if free(a) { a } else { b } & !FREE;
            return Err(CheckError::NotImplied { r1cs: i as u32, why: format!("wire {w} has no cell") });
        }
        batch.push((i, q));
        if batch.len() == BATCH {
            flush(&mut batch)?;
        }
    }
    flush(&mut batch)?;
    Ok(TierA { constraints: r1cs.constraints.len(), equations, unique: exp.len(), additions: defs.0.len() })
}
