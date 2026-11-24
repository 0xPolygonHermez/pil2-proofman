//! Greedy intermediate-polynomial search over `ir::degree_of`; returns the set only.

use std::collections::{BTreeMap, BTreeSet};

use crate::cfg::DegreePolicy;
use crate::core::im::{boundary_degree, check_im, im_order};
use crate::core::ir::{degree_of, degree_table, dim_table, ColType, ConstraintSystem, Expr, ExprId};
use crate::error::PilInfoError;

type Set = BTreeSet<ExprId>;
/// `None` = infeasible; the degree is what the search believes, never trusted (check_im re-derives it).
type Found = (Option<Set>, i64);

/// The cheapest feasible degree's set, scored with `check_im`'s derived qDeg (old: `calculate_intermediate_polynomials`).
pub fn search_im(cs: &ConstraintSystem, policy: &DegreePolicy, q_dim: usize) -> Result<Set, PilInfoError> {
    let max = policy.max_constraint_degree().max(2);
    let stark_min_q = matches!(policy, DegreePolicy::FromBlowup { .. });
    let mut best: Option<(i64, Set)> = None;
    let deg = degree_table(cs);
    let dims = dim_table(cs).map_err(|u| PilInfoError::InvalidPilout(u.0))?;
    for d in 2..=max {
        let Some(set) = (Search { cs, deg: &deg, abs_max: d, memo: BTreeMap::new() }).roots() else { continue };
        let Ok(im) = im_order(cs, &set) else { continue };
        let Ok(checked) = check_im(cs, &im, d as u32, stark_min_q) else { continue };
        let im_cols: usize = im.exprs.iter().map(|&i| dims[i as usize] as usize).sum();
        let cost = policy.cost(im.exprs.len(), im_cols, i64::from(checked.q_deg), q_dim);
        let empty = set.is_empty();
        if best.as_ref().is_none_or(|(c, _)| cost < *c) {
            best = Some((cost, set));
        }
        // a degree needing no intermediates repeats for every larger one
        if empty {
            break;
        }
    }
    best.map(|(_, s)| s).ok_or(PilInfoError::NoFeasibleDegree { max })
}

/// One degree's search; the memo is keyed on everything the result depends on but `abs_max`, which is fixed.
struct Search<'a> {
    cs: &'a ConstraintSystem,
    deg: &'a [u32],
    abs_max: usize,
    memo: BTreeMap<(ExprId, usize, Set), Found>,
}

impl Search<'_> {
    /// Roots in constraint order, threading the set, each reached as the old cExp's `exp` reference.
    fn roots(&mut self) -> Option<Set> {
        let mut set = Set::new();
        for c in &self.cs.constraints {
            let root = match boundary_degree(&c.boundary) {
                0 => c.e.clone(),
                // the boundary selector as a degree-1 leaf: only its degree matters to the search
                _ => Expr::Mul(Box::new(c.e.clone()), Box::new(selector())),
            };
            set = self.expr(&root, &set, self.abs_max).0?;
        }
        Some(set)
    }

    fn target(&mut self, id: ExprId, set: &Set, max_deg: usize) -> Found {
        let key = (id, max_deg, set.clone());
        if let Some(r) = self.memo.get(&key) {
            return r.clone();
        }
        let r = match self.cs.exprs.get(id as usize) {
            Some(e) => self.expr(e, set, max_deg),
            None => (None, -1),
        };
        self.memo.insert(key, r.clone());
        r
    }

    fn expr(&mut self, e: &Expr, set: &Set, max_deg: usize) -> Found {
        match e {
            Expr::Add(a, b) | Expr::Sub(a, b) => {
                let (Some(s1), d1) = self.expr(a, set, max_deg) else { return (None, -1) };
                let (Some(s2), d2) = self.expr(b, &s1, max_deg) else { return (None, -1) };
                (Some(s2), d1.max(d2).max(0))
            }
            // the old arena lowers neg to mul(-1, x), whose degree-0 leaf is skipped
            Expr::Neg(a) => self.expr(a, set, max_deg),
            Expr::Mul(a, b) => self.mul(e, a, b, set, max_deg),
            Expr::Ref(id) => {
                if max_deg < 1 {
                    return (None, -1);
                }
                if set.contains(id) {
                    return (Some(set.clone()), 1);
                }
                match self.target(*id, set, self.abs_max) {
                    (Some(mut s), d) if d > max_deg as i64 => {
                        s.insert(*id);
                        (Some(s), 1)
                    }
                    r => r,
                }
            }
            leaf if degree_of(leaf, self.deg) == 0 => (Some(set.clone()), 0),
            _ if max_deg < 1 => (None, -1),
            _ => (Some(set.clone()), 1),
        }
    }

    fn mul(&mut self, e: &Expr, a: &Expr, b: &Expr, set: &Set, max_deg: usize) -> Found {
        if is_const_leaf(a) {
            return self.expr(b, set, max_deg);
        }
        if is_const_leaf(b) {
            return self.expr(a, set, max_deg);
        }
        let here = degree_of(e, self.deg) as usize;
        if here <= max_deg {
            return (Some(set.clone()), here as i64);
        }
        let mut best: Found = (None, -1);
        for l in 0..=max_deg {
            let (Some(s1), d1) = self.expr(a, set, l) else { continue };
            let (Some(s2), d2) = self.expr(b, &s1, max_deg - l) else { continue };
            if best.0.as_ref().is_none_or(|p| s2.len() < p.len()) {
                let done = s2.len() == set.len();
                best = (Some(s2), d1 + d2);
                // cannot do better than the starting set
                if done {
                    break;
                }
            }
        }
        best
    }
}

/// A degree-0 leaf (not a composite, `Neg` being the old `mul(-1, x)`, nor a `Ref`): a product by it is skipped.
fn is_const_leaf(e: &Expr) -> bool {
    !matches!(e, Expr::Add(..) | Expr::Sub(..) | Expr::Mul(..) | Expr::Neg(_) | Expr::Ref(_)) && degree_of(e, &[]) == 0
}

fn selector() -> Expr {
    Expr::Col { col: ColType::Fixed, commit_id: None, id: u32::MAX, offset: 0 }
}
