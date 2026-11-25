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
    let deg = degree_table(cs).map_err(|u| PilInfoError::InvalidPilout(u.0))?;
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

/// The state of a product's split search: `a` within `l`, `b` within `max - l`, the best set so far.
struct Split<'e> {
    a: &'e Expr,
    b: &'e Expr,
    set: Set,
    max: usize,
    l: usize,
    best: Found,
}

/// A suspended search call, resumed with the result of the call it made (the search is iterative: no recursion
/// over the expression depth).
enum Cont<'e> {
    /// The lhs of an add/sub is done: search the rhs from its set.
    AddRhs { b: &'e Expr, max: usize },
    /// Both operands are done: the larger degree, at least 0.
    AddDone { d1: i64 },
    /// A `Ref`'s target is done: promote it when above `max`.
    Promote { id: ExprId, max: usize },
    /// A target's body is done: memoise it.
    Memo { key: (ExprId, usize, Set) },
    /// `a` of a split is done.
    SplitLhs(Split<'e>),
    /// `b` of a split is done.
    SplitRhs(Split<'e>, i64),
}

/// Search `e` from a set within a degree, or return a result.
enum Step<'e> {
    Call(&'e Expr, Set, usize),
    Ret(Found),
}

const NONE: Found = (None, -1);

impl<'a> Search<'a> {
    /// Roots in constraint order, threading the set, each reached as the old cExp's `exp` reference.
    fn roots(&mut self) -> Option<Set> {
        let mut set = Set::new();
        for c in &self.cs.constraints {
            let root = match boundary_degree(&c.boundary) {
                0 => c.e.clone(),
                // the boundary selector as a degree-1 leaf: only its degree matters to the search
                _ => Expr::Mul(Box::new(c.e.clone()), Box::new(selector())),
            };
            set = self.expr(&root, set, self.abs_max).0?;
        }
        Some(set)
    }

    /// `e` searched from `set` within `max_deg`: the set it needs and the degree it reads as.
    fn expr<'e>(&mut self, e: &'e Expr, set: Set, max_deg: usize) -> Found
    where
        'a: 'e,
    {
        let mut stack = Vec::new();
        let mut step = self.enter(e, set, max_deg, &mut stack);
        loop {
            step = match step {
                Step::Call(e, set, max) => self.enter(e, set, max, &mut stack),
                Step::Ret(r) => match stack.pop() {
                    None => return r,
                    Some(k) => self.resume(k, r, &mut stack),
                },
            };
        }
    }

    fn enter<'e>(&mut self, e: &'e Expr, set: Set, max_deg: usize, stack: &mut Vec<Cont<'e>>) -> Step<'e>
    where
        'a: 'e,
    {
        let cs: &'a ConstraintSystem = self.cs;
        match e {
            Expr::Add(a, b) | Expr::Sub(a, b) => {
                stack.push(Cont::AddRhs { b, max: max_deg });
                Step::Call(a, set, max_deg)
            }
            // the old arena lowers neg to mul(-1, x), whose degree-0 leaf is skipped
            Expr::Neg(a) => Step::Call(a, set, max_deg),
            Expr::Mul(a, b) if is_const_leaf(a) => Step::Call(b, set, max_deg),
            Expr::Mul(a, b) if is_const_leaf(b) => Step::Call(a, set, max_deg),
            Expr::Mul(a, b) => match degree_of(e, self.deg).map(|d| d as usize) {
                Err(_) => Step::Ret(NONE),
                Ok(here) if here <= max_deg => Step::Ret((Some(set), here as i64)),
                Ok(_) => Self::split(Split { a, b, set, max: max_deg, l: 0, best: NONE }, stack),
            },
            Expr::Ref(_) if max_deg < 1 => Step::Ret(NONE),
            Expr::Ref(id) if set.contains(id) => Step::Ret((Some(set), 1)),
            Expr::Ref(id) => {
                stack.push(Cont::Promote { id: *id, max: max_deg });
                let key = (*id, self.abs_max, set);
                if let Some(r) = self.memo.get(&key) {
                    return Step::Ret(r.clone());
                }
                match cs.exprs.get(*id as usize) {
                    Some(body) => {
                        let set = key.2.clone();
                        stack.push(Cont::Memo { key });
                        Step::Call(body, set, self.abs_max)
                    }
                    None => {
                        self.memo.insert(key, NONE);
                        Step::Ret(NONE)
                    }
                }
            }
            leaf if !matches!(leaf, Expr::Col { .. }) => Step::Ret((Some(set), 0)),
            _ if max_deg < 1 => Step::Ret(NONE),
            _ => Step::Ret((Some(set), 1)),
        }
    }

    fn resume<'e>(&mut self, k: Cont<'e>, r: Found, stack: &mut Vec<Cont<'e>>) -> Step<'e> {
        match (k, r) {
            (Cont::AddRhs { b, max }, (Some(s1), d1)) => {
                stack.push(Cont::AddDone { d1 });
                Step::Call(b, s1, max)
            }
            (Cont::AddDone { d1 }, (Some(s2), d2)) => Step::Ret((Some(s2), d1.max(d2).max(0))),
            (Cont::AddRhs { .. } | Cont::AddDone { .. }, _) => Step::Ret(NONE),
            (Cont::Promote { id, max }, (Some(mut s), d)) if d > max as i64 => {
                s.insert(id);
                Step::Ret((Some(s), 1))
            }
            (Cont::Promote { .. }, r) => Step::Ret(r),
            (Cont::Memo { key }, r) => {
                self.memo.insert(key, r.clone());
                Step::Ret(r)
            }
            (Cont::SplitLhs(m), (Some(s1), d1)) => {
                let (b, rest) = (m.b, m.max - m.l);
                stack.push(Cont::SplitRhs(m, d1));
                Step::Call(b, s1, rest)
            }
            (Cont::SplitLhs(m), _) => Self::split(Split { l: m.l + 1, ..m }, stack),
            (Cont::SplitRhs(mut m, d1), (Some(s2), d2)) if m.best.0.as_ref().is_none_or(|p| s2.len() < p.len()) => {
                // cannot do better than the starting set
                let done = s2.len() == m.set.len();
                m.best = (Some(s2), d1 + d2);
                if done {
                    return Step::Ret(m.best);
                }
                Self::split(Split { l: m.l + 1, ..m }, stack)
            }
            (Cont::SplitRhs(m, _), _) => Self::split(Split { l: m.l + 1, ..m }, stack),
        }
    }

    /// Tries split `m.l` of a product, or returns the best one once every split is tried.
    fn split<'e>(m: Split<'e>, stack: &mut Vec<Cont<'e>>) -> Step<'e> {
        if m.l > m.max {
            return Step::Ret(m.best);
        }
        let (a, set, l) = (m.a, m.set.clone(), m.l);
        stack.push(Cont::SplitLhs(m));
        Step::Call(a, set, l)
    }
}

/// A degree-0 leaf (not a composite, `Neg` being the old `mul(-1, x)`, nor a `Ref`): a product by it is skipped.
fn is_const_leaf(e: &Expr) -> bool {
    !matches!(e, Expr::Add(..) | Expr::Sub(..) | Expr::Mul(..) | Expr::Neg(_) | Expr::Ref(_) | Expr::Col { .. })
}

fn selector() -> Expr {
    Expr::Col { col: ColType::Fixed, commit_id: None, id: u32::MAX, offset: 0 }
}
