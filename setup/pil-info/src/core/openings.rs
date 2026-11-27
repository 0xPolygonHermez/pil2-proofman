//! Opening points and the evaluation map (the only evMap order).

use std::cmp::Reverse;
use std::collections::{BTreeMap, BTreeSet};

use crate::cfg::Opening;
use crate::core::im::{im_column_id, ImChoice};
use crate::core::ir::{visit, ColRef, ColType, ConstraintSystem, Expr, ExprId, HintField, HintValue};
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

/// Points: 0 plus every row offset read by a root, an im expression or a hint expression, sorted numerically.
/// `ev_map`: the columns the roots read (im columns at offset 0, Q pieces at 0 under Deep), in today's order.
/// A hint adds points only, never evaluations.
pub fn openings(cs: &ConstraintSystem, im: &ImChoice, lay: &Layout) -> Result<Openings, OpeningsError> {
    // today's evMap walks every root through every Ref, promoted or not
    let ims: Vec<Expr> = im.exprs.iter().map(|&e| Expr::Ref(e)).collect();
    let mut evs: BTreeSet<ColRef> = BTreeSet::new();
    reach(cs, cs.constraints.iter().map(|c| &c.e).chain(&ims), |k| {
        evs.insert(k);
    })?;
    for k in 0..im.exprs.len() {
        let id = im_column_id(cs, k);
        if !lay.cm_pols_map.get(id as usize).is_some_and(|p| p.exp_id.is_some()) {
            return Err(OpeningsError::Unresolved { what: format!("im column {id} is not in the layout") });
        }
        evs.insert(cm(id));
    }
    let mut points: BTreeSet<i64> = evs.iter().map(|k| k.offset).collect();
    points.insert(0);
    reach(cs, hint_exprs(cs).iter(), |k| {
        points.insert(k.offset);
    })?;
    if cs.opening == Opening::Deep {
        let q_stage = cs.n_stages + 1;
        for (id, _) in lay.cm_pols_map.iter().enumerate().filter(|(_, p)| p.stage == q_stage) {
            let pol = cm(id as u32); // below the ingestion limit
            if !evs.insert(pol) {
                return Err(OpeningsError::Duplicate { pol });
            }
        }
    }
    let pos: BTreeMap<i64, u32> = points.iter().enumerate().map(|(i, o)| (*o, i as u32)).collect();
    // today's order: opening, then custom > const > cm, then id
    let mut keyed = Vec::with_capacity(evs.len());
    for pol in evs {
        let type_key = match (pol.col, pol.commit_id) {
            (ColType::Witness, _) => 0,
            (ColType::Fixed, _) => 1,
            (ColType::Custom, Some(c)) => u64::from(c) + 2,
            (ColType::Custom, None) => {
                return Err(OpeningsError::Unresolved { what: "a custom column without a commit".into() })
            }
        };
        let missing = || OpeningsError::Unresolved { what: format!("opening point {}", pol.offset) };
        let opening_pos = *pos.get(&pol.offset).ok_or_else(missing)?;
        keyed.push(((opening_pos, Reverse(type_key), pol.id, pol.offset), EvEntry { pol, opening_pos }));
    }
    keyed.sort_by_key(|(k, _)| *k);
    Ok(Openings { opening_points: points.into_iter().collect(), ev_map: keyed.into_iter().map(|(_, e)| e).collect() })
}

/// The expressions the hints name (their `Ref` operands): the ones the prover keeps.
fn hint_exprs(cs: &ConstraintSystem) -> Vec<Expr> {
    fn walk(fields: &[HintField], out: &mut Vec<Expr>) {
        for f in fields {
            match &f.value {
                HintValue::Expr(e @ Expr::Ref(_)) => out.push(e.clone()),
                HintValue::Array(a) => walk(a, out),
                HintValue::Expr(_) | HintValue::Str(_) => {}
            }
        }
    }
    let mut out = Vec::new();
    cs.hints.iter().for_each(|h| walk(&h.fields, &mut out));
    out
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
