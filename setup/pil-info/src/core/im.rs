//! ImChoice, order, substitute and check_im (derives qDeg).

use std::collections::{BTreeMap, BTreeSet};

use crate::core::from_pilout::{self, PiloutError};
use crate::core::ir::{
    degree_of, degree_table, degree_table_with, map_expr, stage_table, visit, Boundary, ColType, ConstraintSystem,
    Expr, ExprId, Unresolved,
};

/// The promoted expressions in canonical order: by stage, then by first appearance in the constraints.
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
    Unreached { expr: u32 },
    BadRef(PiloutError),
    Unresolved(Unresolved),
}

/// `set` by (stage, first appearance in the constraints' pre-order walk, each `Ref` entered once); one no
/// constraint reaches is `Unreached`.
pub fn im_order(cs: &ConstraintSystem, set: &BTreeSet<ExprId>) -> Result<ImChoice, ImError> {
    let stage = stage_table(cs).map_err(ImError::Unresolved)?;
    let mut seen = vec![false; cs.exprs.len()];
    let mut first = BTreeMap::new();
    let mut stack: Vec<&Expr> = cs.constraints.iter().rev().map(|c| &c.e).collect();
    while let Some(e) = stack.pop() {
        match e {
            Expr::Add(a, b) | Expr::Sub(a, b) | Expr::Mul(a, b) => stack.extend([&**b, &**a]),
            Expr::Neg(a) => stack.push(a),
            Expr::Ref(id) if !seen.get(*id as usize).copied().unwrap_or(true) => {
                seen[*id as usize] = true;
                first.insert(*id, first.len());
                stack.push(&cs.exprs[*id as usize]);
            }
            _ => {}
        }
    }
    let key = |&id: &ExprId| match (stage.get(id as usize), first.get(&id)) {
        (Some(s), Some(f)) => Ok((*s, *f, id)),
        _ => Err(ImError::Unreached { expr: id }),
    };
    let mut keyed = set.iter().map(key).collect::<Result<Vec<_>, _>>()?;
    keyed.sort();
    Ok(ImChoice { exprs: keyed.into_iter().map(|k| k.2).collect() })
}

/// `im` is duplicate-free and in `im_order`.
fn check_order(cs: &ConstraintSystem, im: &ImChoice) -> Result<(), ImError> {
    let set: BTreeSet<ExprId> = im.exprs.iter().copied().collect();
    if set.len() != im.exprs.len() {
        let mut dup = im.exprs.clone();
        dup.sort();
        let expr = dup.windows(2).find(|w| w[0] == w[1]).map_or(0, |w| w[0]);
        return Err(ImError::Duplicate { expr });
    }
    if im_order(cs, &set)? != *im {
        return Err(ImError::Unsorted);
    }
    Ok(())
}

/// The witness column id of the `k`-th promoted expression: after every pilout witness column.
pub fn im_column_id(cs: &ConstraintSystem, k: usize) -> u32 {
    (cs.witness.len() + k) as u32 // below the ingestion limit
}

/// The degree a boundary's selector adds to its root (the old cExp's `mul(e, Zi)`).
pub fn boundary_degree(b: &Boundary) -> u32 {
    u32::from(*b != Boundary::EveryRow)
}

/// `Ref` resolved; a promoted `Ref` becomes the im column. Inlines the DAG: for tests and references only.
pub fn substitute(cs: &ConstraintSystem, im: &ImChoice, e: &Expr) -> Result<Expr, Unresolved> {
    let col = |id: &ExprId| {
        let k = im.exprs.iter().position(|x| x == id)?;
        Some(Expr::Col { col: ColType::Witness, commit_id: None, id: im_column_id(cs, k), offset: 0 })
    };
    let sub = |e: &Expr, t: &[Expr]| {
        let mut err = None;
        let out = map_expr(e, &mut |n| {
            let Expr::Ref(id) = n else { return None };
            Some(col(id).or_else(|| t.get(*id as usize).cloned()).unwrap_or_else(|| {
                err.get_or_insert(Unresolved(format!("expression {id}")));
                Expr::Const(0u32.into())
            }))
        });
        err.map_or(Ok(out), Err)
    };
    let mut t = Vec::with_capacity(cs.exprs.len());
    for e in &cs.exprs {
        let v = sub(e, &t)?;
        t.push(v);
    }
    sub(e, &t)
}

/// Every `Ref` in range and pointing below its expression (the numbering `from_pilout` gives).
fn check_refs(cs: &ConstraintSystem) -> Result<(), ImError> {
    let bad = |what| ImError::BadRef(from_pilout::bad(cs.exprs.len(), what));
    for (i, e) in cs.exprs.iter().enumerate() {
        let mut ok = true;
        visit(e, &mut |n| ok &= !matches!(n, Expr::Ref(t) if *t as usize >= i));
        if !ok {
            return Err(bad("an expression reference not below its expression"));
        }
    }
    for c in &cs.constraints {
        let mut ok = true;
        visit(&c.e, &mut |n| ok &= !matches!(n, Expr::Ref(t) if *t as usize >= cs.exprs.len()));
        if !ok {
            return Err(bad("constraint expression out of range"));
        }
    }
    Ok(())
}

pub fn check_im(cs: &ConstraintSystem, im: &ImChoice, max_deg: u32, stark_min_q: bool) -> Result<Checked, ImError> {
    check_refs(cs)?;
    let plain = degree_table(cs).map_err(ImError::Unresolved)?;
    for &id in &im.exprs {
        if plain.get(id as usize).is_none_or(|d| *d == 0) {
            return Err(ImError::NotPromotable { expr: id });
        }
    }
    check_order(cs, im)?;
    let read = degree_table_with(cs, &im.exprs).map_err(ImError::Unresolved)?;
    let mut root_degrees = Vec::with_capacity(cs.constraints.len());
    for (k, c) in cs.constraints.iter().enumerate() {
        let d = degree_of(&c.e, &read).map_err(ImError::Unresolved)?.saturating_add(boundary_degree(&c.boundary));
        if d > max_deg {
            return Err(ImError::RootTooHigh { constraint: k, degree: d, max: max_deg });
        }
        root_degrees.push(d);
    }
    let mut im_degrees = Vec::with_capacity(im.exprs.len());
    for &id in &im.exprs {
        // the body, not `Ref(id)`: only the OTHER promoted ones read as columns
        let d = degree_of(&cs.exprs[id as usize], &read).map_err(ImError::Unresolved)?;
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
