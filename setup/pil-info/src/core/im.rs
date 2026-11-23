//! ImChoice, order, substitute and check_im (derives qDeg).

use std::collections::{BTreeMap, BTreeSet};

use crate::core::from_pilout::{self, PiloutError};
use crate::core::ir::{degree, degree_with, map_expr, visit, Boundary, ColType, ConstraintSystem, Expr, ExprId};

/// The promoted expressions, in today's order (ascending expression id) until Task 9.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ImChoice {
    pub exprs: Vec<ExprId>,
}

/// `root_degrees` per constraint, `im_degrees` per promoted expression (in `exprs` order), both substituted.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Checked {
    pub q_deg: u32,
    pub root_degrees: Vec<u32>,
    pub im_degrees: Vec<u32>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ImError {
    NotPromotable { expr: u32 },
    RootTooHigh { constraint: usize, degree: u32, max: u32 },
    ImTooHigh { expr: u32, degree: u32, max: u32 },
    Unsorted,
    Duplicate { expr: u32 },
    BadRef(PiloutError),
}

pub fn im_order(set: &BTreeSet<ExprId>) -> Result<ImChoice, ImError> {
    let im = ImChoice { exprs: set.iter().copied().collect() };
    check_order(&im)?;
    Ok(im)
}

fn check_order(im: &ImChoice) -> Result<(), ImError> {
    for w in im.exprs.windows(2) {
        if w[0] == w[1] {
            return Err(ImError::Duplicate { expr: w[0] });
        }
        if w[0] > w[1] {
            return Err(ImError::Unsorted);
        }
    }
    Ok(())
}

/// The witness column id of the `k`-th promoted expression: after every pilout witness column.
pub fn im_column_id(cs: &ConstraintSystem, k: usize) -> u32 {
    (cs.witness.len() + k) as u32
}

/// The degree a boundary's selector adds to its root (the old cExp's `mul(e, Zi)`).
pub fn boundary_degree(b: &Boundary) -> u32 {
    u32::from(*b != Boundary::EveryRow)
}

/// `Ref` resolved; a promoted `Ref` becomes the im column. Inlines the DAG: for tests and references only.
pub fn substitute(cs: &ConstraintSystem, im: &ImChoice, e: &Expr) -> Expr {
    substitute_in(cs, im, e, &mut BTreeMap::new())
}

fn substitute_in(cs: &ConstraintSystem, im: &ImChoice, e: &Expr, memo: &mut BTreeMap<ExprId, Expr>) -> Expr {
    map_expr(e, &mut |n| {
        let Expr::Ref(id) = n else { return None };
        if let Some(k) = im.exprs.iter().position(|x| x == id) {
            return Some(Expr::Col { col: ColType::Witness, commit_id: None, id: im_column_id(cs, k), offset: 0 });
        }
        if let Some(r) = memo.get(id) {
            return Some(r.clone());
        }
        memo.insert(*id, Expr::Const(0u32.into())); // a cycle (rejected at ingestion) reads as 0, as in `degree`
        let r = cs.exprs.get(*id as usize).map_or(Expr::Const(0u32.into()), |t| substitute_in(cs, im, t, memo));
        memo.insert(*id, r.clone());
        Some(r)
    })
}

/// Every `Ref` in range and the expressions acyclic, so `degree_with` reads no Ref as 0.
fn check_refs(cs: &ConstraintSystem) -> Result<(), ImError> {
    from_pilout::check_refs(&cs.exprs).map_err(ImError::BadRef)?;
    for c in &cs.constraints {
        let mut bad = None;
        visit(&c.e, &mut |n| {
            if let Expr::Ref(to) = n {
                if *to as usize >= cs.exprs.len() {
                    bad = Some(*to);
                }
            }
        });
        if let Some(to) = bad {
            return Err(ImError::BadRef(PiloutError::BadOperand {
                expr: to,
                why: "constraint expression out of range",
            }));
        }
    }
    Ok(())
}

pub fn check_im(cs: &ConstraintSystem, im: &ImChoice, max_deg: u32, stark_min_q: bool) -> Result<Checked, ImError> {
    check_order(im)?;
    check_refs(cs)?;
    for &id in &im.exprs {
        let promotable = cs.exprs.get(id as usize).is_some_and(|e| degree(cs, e) > 0);
        if !promotable {
            return Err(ImError::NotPromotable { expr: id });
        }
    }
    let mut root_degrees = Vec::with_capacity(cs.constraints.len());
    for (k, c) in cs.constraints.iter().enumerate() {
        let d = degree_with(cs, &c.e, &im.exprs).saturating_add(boundary_degree(&c.boundary));
        if d > max_deg {
            return Err(ImError::RootTooHigh { constraint: k, degree: d, max: max_deg });
        }
        root_degrees.push(d);
    }
    let mut im_degrees = Vec::with_capacity(im.exprs.len());
    for &id in &im.exprs {
        // the body is not a Ref to itself, so only the OTHER promoted ones read as columns
        let d = cs.exprs.get(id as usize).map_or(0, |e| degree_with(cs, e, &im.exprs));
        if d > max_deg {
            return Err(ImError::ImTooHigh { expr: id, degree: d, max: max_deg });
        }
        im_degrees.push(d);
    }
    // the im roots `col_k - e_k` are quotient roots too
    let top = root_degrees.iter().chain(&im_degrees).copied().max().unwrap_or(0);
    let q_deg = top.saturating_sub(1);
    Ok(Checked { q_deg: if stark_min_q { q_deg.max(1) } else { q_deg }, root_degrees, im_degrees })
}
