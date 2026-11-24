//! Opening points and the evaluation map (the only evMap order).

use std::cmp::Reverse;
use std::collections::{BTreeMap, BTreeSet};

use crate::cfg::Opening;
use crate::core::im::{im_column_id, ImChoice};
use crate::core::ir::{visit, ColRef, ColType, ConstraintSystem, Expr, ExprId};
use crate::core::layout::Layout;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EvEntry {
    pub pol: ColRef,
    pub opening_pos: u32,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Openings {
    pub opening_points: Vec<i64>,
    pub ev_map: Vec<EvEntry>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum OpeningsError {
    Duplicate { pol: ColRef },
    Unresolved { what: String },
}

/// Points: 0 plus every row offset read by a root, an im expression or an extra root, sorted numerically.
/// `ev_map`: the columns the roots read (im columns at offset 0, Q pieces at 0 under Deep), in today's order.
/// An `extra_roots` expression adds points only, never evaluations.
pub fn openings(
    cs: &ConstraintSystem,
    im: &ImChoice,
    lay: &Layout,
    extra_roots: &[Expr],
) -> Result<Openings, OpeningsError> {
    // today's evMap walks every root through every Ref, promoted or not
    let ims: Vec<Expr> = im.exprs.iter().map(|&e| Expr::Ref(e)).collect();
    let mut evs: BTreeSet<ColRef> = BTreeSet::new();
    reach(cs, cs.constraints.iter().map(|c| &c.e).chain(&ims), |k| {
        evs.insert(k);
    })?;
    for k in 0..im.exprs.len() {
        let id = im_column_id(cs, k);
        if !lay.cm_pols_map.get(id as usize).is_some_and(|p| p.im_pol) {
            return Err(OpeningsError::Unresolved { what: format!("im column {id} is not in the layout") });
        }
        evs.insert(cm(id));
    }
    let mut points: BTreeSet<i64> = evs.iter().map(|k| k.offset).collect();
    points.insert(0);
    reach(cs, extra_roots.iter(), |k| {
        points.insert(k.offset);
    })?;
    if cs.opening == Opening::Deep {
        let q_stage = cs.n_stages + 1;
        for (id, _) in lay.cm_pols_map.iter().enumerate().filter(|(_, p)| p.stage == q_stage) {
            if !evs.insert(cm(id as u32)) {
                return Err(OpeningsError::Duplicate { pol: cm(id as u32) });
            }
        }
    }
    let pos: BTreeMap<i64, u32> = points.iter().enumerate().map(|(i, o)| (*o, i as u32)).collect();
    let mut ev_map: Vec<EvEntry> = evs.into_iter().map(|pol| EvEntry { pol, opening_pos: pos[&pol.offset] }).collect();
    // today's order: opening, then custom > const > cm, then id
    let type_key = |e: &EvEntry| match e.pol.col {
        ColType::Witness => 0,
        ColType::Fixed => 1,
        ColType::Custom => e.pol.commit_id.unwrap_or(0) as u64 + 2,
    };
    ev_map.sort_by_key(|e| (e.opening_pos, Reverse(type_key(e)), e.pol.id, e.pol.offset));
    Ok(Openings { opening_points: points.into_iter().collect(), ev_map })
}

/// `max(ev_map.len(), n_openings + n_distinct_pols - 1)`, counting only openings that have evaluations.
pub fn fri_batch_size(ops: &Openings) -> u32 {
    let n_open = ops.opening_points.iter().filter(|o| ops.ev_map.iter().any(|e| e.pol.offset == **o)).count();
    let n_pols = ops.ev_map.iter().map(|e| (e.pol.col, e.pol.commit_id, e.pol.id)).collect::<BTreeSet<_>>().len();
    ops.ev_map.len().max((n_open + n_pols).saturating_sub(1)) as u32
}

fn cm(id: u32) -> ColRef {
    ColRef { col: ColType::Witness, commit_id: None, id, offset: 0 }
}

/// Every column read by `roots`, through Refs, each Ref walked once; an unknown Ref or column is an error.
fn reach<'a>(
    cs: &'a ConstraintSystem,
    roots: impl Iterator<Item = &'a Expr>,
    mut f: impl FnMut(ColRef),
) -> Result<(), OpeningsError> {
    let mut seen: BTreeSet<ExprId> = BTreeSet::new();
    let mut work: Vec<&'a Expr> = roots.collect();
    let mut bad: Option<String> = None;
    while let Some(e) = work.pop() {
        visit(e, &mut |n| match *n {
            Expr::Ref(id) => match cs.exprs.get(id as usize) {
                Some(t) if seen.insert(id) => work.push(t),
                Some(_) => {}
                None => bad = Some(format!("expression {id}")),
            },
            Expr::Col { col, commit_id, id, offset } => {
                // witness ids stop before the im and Q columns
                let n = match col {
                    ColType::Witness => Some(cs.witness.len()),
                    ColType::Fixed => Some(cs.fixed.len()),
                    ColType::Custom => commit_id.and_then(|c| cs.custom.get(c as usize)).map(Vec::len),
                };
                if n.is_none_or(|n| id as usize >= n) {
                    bad = Some(format!("{col:?} {commit_id:?} column {id} at {offset}"));
                }
                f(ColRef { col, commit_id, id, offset });
            }
            _ => {}
        });
        if let Some(what) = bad.take() {
            return Err(OpeningsError::Unresolved { what });
        }
    }
    Ok(())
}
