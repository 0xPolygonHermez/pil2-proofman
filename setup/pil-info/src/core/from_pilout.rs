//! Strict pilout -> ConstraintSystem; rejects, never repairs.

use num_bigint::BigUint;
use pil2_pilout::pilout::{self as pb, constraint, expression as ex, operand, SymbolType};

use super::ir::{
    visit, Boundary, ColType, Column, Constraint, ConstraintSystem, CustomCommit, Expr, Hint, Lengths, NamedValue,
};
use crate::cfg::{FieldCfg, Opening};

/// Largest id a symbol table may reach; keeps a hostile id from allocating the world.
const MAX_SLOTS: usize = 1 << 24;

#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub enum PiloutError {
    #[error("constraint {constraint} has no expression_idx")]
    MissingExpressionIdx { constraint: usize },
    #[error("constraint {constraint} has no body")]
    EmptyConstraint { constraint: usize },
    #[error("constraint {constraint} (`{line}`) is {boundary}, and the STARK takes only everyRow")]
    BoundaryNotEveryRow { constraint: usize, line: String, boundary: String },
    #[error("expression {expr} refers to expression {to}, which does not exist")]
    DanglingRef { expr: u32, to: u32 },
    #[error("expression {expr} refers to itself through its references")]
    CyclicRef { expr: u32 },
    #[error("the pilout has no air {air} in airgroup {airgroup}")]
    UnknownAir { airgroup: usize, air: usize },
    #[error("expression {expr}: {why}")]
    BadOperand { expr: u32, why: &'static str },
}

pub(crate) fn bad(expr: usize, why: &'static str) -> PiloutError {
    PiloutError::BadOperand { expr: expr as u32, why }
}

pub(crate) fn const_of(be: &[u8]) -> Expr {
    Expr::Const(BigUint::from_bytes_be(be))
}

/// Every `Ref` of `exprs` resolves, and no expression reaches itself (iterative DFS, no recursion).
pub(crate) fn check_refs(exprs: &[Expr]) -> Result<(), PiloutError> {
    let refs: Vec<Vec<u32>> = exprs
        .iter()
        .map(|e| {
            let mut r = Vec::new();
            visit(e, &mut |n| {
                if let Expr::Ref(t) = n {
                    r.push(*t);
                }
            });
            r
        })
        .collect();
    for (i, rs) in refs.iter().enumerate() {
        if let Some(&to) = rs.iter().find(|&&t| t as usize >= exprs.len()) {
            return Err(PiloutError::DanglingRef { expr: i as u32, to });
        }
    }
    // 0 = unvisited, 1 = on the stack, 2 = done
    let mut state = vec![0u8; exprs.len()];
    for root in 0..exprs.len() {
        if state[root] != 0 {
            continue;
        }
        let mut stack = vec![(root, 0usize)];
        state[root] = 1;
        while let Some(&(n, k)) = stack.last() {
            if let Some(&t) = refs[n].get(k) {
                let t = t as usize;
                if let Some(top) = stack.last_mut() {
                    top.1 += 1;
                }
                match state[t] {
                    1 => return Err(PiloutError::CyclicRef { expr: t as u32 }),
                    0 => {
                        state[t] = 1;
                        stack.push((t, 0));
                    }
                    _ => {}
                }
            } else {
                state[n] = 2;
                stack.pop();
            }
        }
    }
    Ok(())
}

/// Number of cells of a symbol: 1, or the product of its array lengths.
pub(crate) fn symbol_len(s: &pb::Symbol) -> usize {
    if s.dim == 0 {
        1
    } else {
        s.lengths.iter().fold(1usize, |a, l| a.saturating_mul(*l as usize))
    }
}

/// Indices of the `k`-th cell of `s`, last index fastest; `None` for a scalar.
fn cell_lengths(s: &pb::Symbol, k: usize) -> Lengths {
    if s.dim == 0 {
        return None;
    }
    let mut rest = k;
    let mut idx = vec![0u32; s.lengths.len()];
    for (i, l) in s.lengths.iter().enumerate().rev() {
        let l = (*l as usize).max(1);
        idx[i] = (rest % l) as u32;
        rest /= l;
    }
    Some(idx)
}

fn cell(s: &pb::Symbol, k: usize) -> (String, Lengths) {
    (s.name.clone(), cell_lengths(s, k))
}

/// Place `name` at `idx`, padding with `fill`.
pub(crate) fn place<T: Clone>(v: &mut Vec<T>, idx: usize, fill: T, val: T) -> Result<(), PiloutError> {
    if idx >= MAX_SLOTS {
        return Err(bad(idx, "symbol id out of range"));
    }
    if v.len() <= idx {
        v.resize(idx + 1, fill);
    }
    v[idx] = val;
    Ok(())
}

/// First column id of `stage` in a commit with `widths` (the only column-id rule).
pub(crate) fn stage_base(widths: &[u32], stage: u32) -> u32 {
    widths.iter().take(stage.saturating_sub(1) as usize).sum()
}

/// What the operands of one air may refer to.
struct Tables<'a> {
    stage_widths: &'a [u32],
    custom_widths: &'a [Vec<u32>],
    n_fixed: usize,
    n_periodic: usize,
    n_air_values: usize,
    n_airgroup_values: usize,
    pilout: &'a pb::PilOut,
}

fn within(ok: bool, parent: usize, why: &'static str) -> Result<(), PiloutError> {
    if ok {
        Ok(())
    } else {
        Err(bad(parent, why))
    }
}

/// `idx` is a column of stage `stage` in a commit of `widths` (stage 0 is not a witness stage).
fn check_staged_col(widths: &[u32], stage: u32, idx: u32, parent: usize) -> Result<(), PiloutError> {
    let w = (stage as usize).checked_sub(1).and_then(|s| widths.get(s));
    within(w.is_some_and(|w| idx < *w), parent, "witness column outside the stage widths")
}

fn lower_operand(o: &Option<pb::Operand>, parent: usize, t: &Tables) -> Result<Expr, PiloutError> {
    let inner = o.as_ref().and_then(|o| o.operand.as_ref()).ok_or_else(|| bad(parent, "missing operand"))?;
    Ok(match inner {
        operand::Operand::Expression(r) => Expr::Ref(r.idx),
        operand::Operand::Constant(c) => const_of(&c.value),
        operand::Operand::WitnessCol(w) => {
            check_staged_col(t.stage_widths, w.stage, w.col_idx, parent)?;
            Expr::Col {
                col: ColType::Witness,
                commit_id: None,
                id: w.col_idx + stage_base(t.stage_widths, w.stage),
                offset: w.row_offset as i64,
            }
        }
        operand::Operand::CustomCol(c) => {
            let w = t.custom_widths.get(c.commit_id as usize).ok_or_else(|| bad(parent, "unknown custom commit"))?;
            within(c.stage == 0, parent, "custom column of a stage other than 0")?;
            within(w.first().is_some_and(|w| c.col_idx < *w), parent, "custom column outside the stage widths")?;
            Expr::Col {
                col: ColType::Custom,
                commit_id: Some(c.commit_id),
                id: c.col_idx + stage_base(w, c.stage),
                offset: c.row_offset as i64,
            }
        }
        operand::Operand::FixedCol(f) => {
            within((f.idx as usize) < t.n_fixed, parent, "fixed column out of range")?;
            Expr::Col { col: ColType::Fixed, commit_id: None, id: f.idx, offset: f.row_offset as i64 }
        }
        operand::Operand::PeriodicCol(p) => {
            within((p.idx as usize) < t.n_periodic, parent, "periodic column out of range")?;
            Expr::Col { col: ColType::Fixed, commit_id: None, id: p.idx, offset: p.row_offset as i64 }
        }
        operand::Operand::PublicValue(p) => {
            check_public(t.pilout, p.idx, parent)?;
            Expr::Public(p.idx)
        }
        operand::Operand::AirGroupValue(a) => {
            within((a.idx as usize) < t.n_airgroup_values, parent, "airgroup value out of range")?;
            Expr::AirgroupValue { airgroup: None, id: a.idx }
        }
        operand::Operand::AirValue(a) => {
            within((a.idx as usize) < t.n_air_values, parent, "air value out of range")?;
            Expr::AirValue(a.idx)
        }
        operand::Operand::Challenge(c) => {
            check_challenge(&t.pilout.num_challenges, c.stage, c.idx, parent)?;
            Expr::Challenge { stage: c.stage, stage_id: c.idx }
        }
        operand::Operand::ProofValue(p) => {
            check_proof_value(t.pilout, p.idx, parent)?;
            Expr::ProofValue(p.idx)
        }
    })
}

pub(crate) fn check_public(p: &pb::PilOut, idx: u32, parent: usize) -> Result<(), PiloutError> {
    within(idx < p.num_public_values, parent, "public value out of range")
}

/// The proof-value index is bounded by the sum over stages (per-stage vs global idx is unconfirmed).
pub(crate) fn check_proof_value(p: &pb::PilOut, idx: u32, parent: usize) -> Result<(), PiloutError> {
    within(idx < p.num_proof_values.iter().sum(), parent, "proof value out of range")
}

pub(crate) fn check_challenge(num_challenges: &[u32], stage: u32, idx: u32, parent: usize) -> Result<(), PiloutError> {
    let n = (stage as usize).checked_sub(1).and_then(|s| num_challenges.get(s));
    within(n.is_some_and(|n| idx < *n), parent, "challenge out of range")
}

/// An operation with its operands, the same for an air's and a global expression.
pub(crate) enum Op<'a, O> {
    Add(&'a Option<O>, &'a Option<O>),
    Sub(&'a Option<O>, &'a Option<O>),
    Mul(&'a Option<O>, &'a Option<O>),
    Neg(&'a Option<O>),
}

/// The only Add/Sub/Mul/Neg lowering; `lo` lowers one operand.
pub(crate) fn lower_op<O>(
    op: Op<O>,
    mut lo: impl FnMut(&Option<O>) -> Result<Expr, PiloutError>,
) -> Result<Expr, PiloutError> {
    Ok(match op {
        Op::Add(a, b) => Expr::Add(Box::new(lo(a)?), Box::new(lo(b)?)),
        Op::Sub(a, b) => Expr::Sub(Box::new(lo(a)?), Box::new(lo(b)?)),
        Op::Mul(a, b) => Expr::Mul(Box::new(lo(a)?), Box::new(lo(b)?)),
        Op::Neg(a) => Expr::Neg(Box::new(lo(a)?)),
    })
}

fn lower_expression(e: &pb::Expression, i: usize, t: &Tables) -> Result<Expr, PiloutError> {
    let op = match e.operation.as_ref().ok_or_else(|| bad(i, "expression without an operation"))? {
        ex::Operation::Add(a) => Op::Add(&a.lhs, &a.rhs),
        ex::Operation::Sub(a) => Op::Sub(&a.lhs, &a.rhs),
        ex::Operation::Mul(a) => Op::Mul(&a.lhs, &a.rhs),
        ex::Operation::Neg(a) => Op::Neg(&a.value),
    };
    lower_op(op, |o| lower_operand(o, i, t))
}

fn lower_constraint(k: usize, c: &pb::Constraint, n_exprs: usize, opening: Opening) -> Result<Constraint, PiloutError> {
    use constraint::Constraint as C;
    let inner = c.constraint.as_ref().ok_or(PiloutError::EmptyConstraint { constraint: k })?;
    let (idx, line, boundary, name) = match inner {
        C::EveryRow(x) => (&x.expression_idx, &x.debug_line, Boundary::EveryRow, "everyRow"),
        C::FirstRow(x) => (&x.expression_idx, &x.debug_line, Boundary::FirstRow, "firstRow"),
        C::LastRow(x) => (&x.expression_idx, &x.debug_line, Boundary::LastRow, "lastRow"),
        C::EveryFrame(x) => (
            &x.expression_idx,
            &x.debug_line,
            Boundary::EveryFrame { min: x.offset_min, max: x.offset_max },
            "everyFrame",
        ),
    };
    let label = line.clone().unwrap_or_default();
    let idx = idx.as_ref().ok_or(PiloutError::MissingExpressionIdx { constraint: k })?.idx;
    if opening == Opening::Deep && boundary != Boundary::EveryRow {
        return Err(PiloutError::BoundaryNotEveryRow { constraint: k, line: label, boundary: name.to_string() });
    }
    if idx as usize >= n_exprs {
        return Err(PiloutError::BadOperand { expr: idx, why: "constraint expression out of range" });
    }
    Ok(Constraint { e: Expr::Ref(idx), boundary, label })
}

/// Columns of kind `kind` for this air (and custom commit `commit`), in symbol order, arrays expanded.
fn columns(
    p: &pb::PilOut,
    airgroup: usize,
    air: usize,
    kind: SymbolType,
    commit: Option<u32>,
    widths: &[u32],
    field: &FieldCfg,
) -> Result<Vec<Column>, PiloutError> {
    let mut out = Vec::new();
    for s in &p.symbols {
        if s.r#type != kind as i32 || s.air_group_id != Some(airgroup as u32) || s.air_id != Some(air as u32) {
            continue;
        }
        if kind == SymbolType::CustomCol && s.commit_id != commit {
            continue;
        }
        let stage = s.stage.unwrap_or(0);
        if kind == SymbolType::CustomCol && stage != 0 {
            return Err(bad(s.id as usize, "custom column of a stage other than 0"));
        }
        let dim = if stage <= 1 { 1 } else { field.ext_dim() as u32 };
        let base: u32 =
            if kind == SymbolType::FixedCol { 0 } else { widths.iter().take(stage.saturating_sub(1) as usize).sum() };
        for k in 0..symbol_len(s) {
            if out.len() >= MAX_SLOTS {
                return Err(bad(s.id as usize, "too many columns"));
            }
            let lengths = cell_lengths(s, k);
            out.push(Column { name: s.name.clone(), stage, dim, id: base + s.id + k as u32, lengths });
        }
    }
    Ok(out)
}

/// `mk(symbol, cell)` of the symbols of `kind` (arrays expanded) placed at their ids; `scope` is
/// (airgroup, optional air).
fn named<T: Clone>(
    p: &pb::PilOut,
    kind: SymbolType,
    scope: Option<(usize, Option<usize>)>,
    fill: T,
    mk: impl Fn(&pb::Symbol, usize) -> T,
) -> Result<Vec<T>, PiloutError> {
    let mut v: Vec<T> = Vec::new();
    for s in &p.symbols {
        let in_scope =
            scope.is_none_or(|(g, a)| s.air_group_id == Some(g as u32) && a.is_none_or(|a| s.air_id == Some(a as u32)));
        if s.r#type != kind as i32 || !in_scope {
            continue;
        }
        for k in 0..symbol_len(s) {
            place(&mut v, s.id as usize + k, fill.clone(), mk(s, k))?;
        }
    }
    Ok(v)
}

/// Pairs the declared stages with the symbol cells; a symbol past the table is an error, an unnamed value gets "".
fn staged(cells: Vec<(String, Lengths)>, stages: &[u32]) -> Result<Vec<NamedValue>, PiloutError> {
    within(cells.len() <= stages.len(), 0, "a value symbol beyond the declared values")?;
    let at = |i: usize| cells.get(i).cloned().unwrap_or_default();
    Ok(stages.iter().enumerate().map(|(i, st)| NamedValue { name: at(i).0, stage: *st, lengths: at(i).1 }).collect())
}

/// Lowers air (`airgroup`, `air`) of `pilout`; reads the protobuf directly.
pub fn from_pilout(
    pilout: &pb::PilOut,
    airgroup: usize,
    air: usize,
    field: &FieldCfg,
    opening: Opening,
) -> Result<ConstraintSystem, PiloutError> {
    let (ag, a) = pilout
        .air_groups
        .get(airgroup)
        .and_then(|g| g.airs.get(air).map(|a| (g, a)))
        .ok_or(PiloutError::UnknownAir { airgroup, air })?;
    let custom_widths: Vec<Vec<u32>> = a.custom_commits.iter().map(|c| c.stage_widths.clone()).collect();
    let tables = Tables {
        stage_widths: &a.stage_widths,
        custom_widths: &custom_widths,
        n_fixed: a.fixed_cols.len(),
        n_periodic: a.periodic_cols.len(),
        n_air_values: a.air_values.len(),
        n_airgroup_values: ag.air_group_values.len(),
        pilout,
    };
    let exprs = a
        .expressions
        .iter()
        .enumerate()
        .map(|(i, e)| lower_expression(e, i, &tables))
        .collect::<Result<Vec<_>, _>>()?;
    check_refs(&exprs)?;
    let constraints = a
        .constraints
        .iter()
        .enumerate()
        .map(|(k, c)| lower_constraint(k, c, exprs.len(), opening))
        .collect::<Result<Vec<_>, _>>()?;

    let witness = columns(pilout, airgroup, air, SymbolType::WitnessCol, None, &a.stage_widths, field)?;
    let fixed = columns(pilout, airgroup, air, SymbolType::FixedCol, None, &[], field)?;
    let custom = (0..custom_widths.len())
        .map(|c| columns(pilout, airgroup, air, SymbolType::CustomCol, Some(c as u32), &[], field))
        .collect::<Result<Vec<_>, _>>()?;

    let challenges = pilout
        .symbols
        .iter()
        .filter(|s| s.r#type == SymbolType::Challenge as i32)
        .map(|s| (s.stage.unwrap_or(1), s.id, s.name.clone()))
        .collect();
    let unnamed = NamedValue { stage: 1, ..Default::default() };
    let proof_values = named(pilout, SymbolType::ProofValue, None, unnamed.clone(), |s, k| {
        let (name, lengths) = cell(s, k);
        NamedValue { name, stage: s.stage.unwrap_or(1), lengths }
    })?;
    let air_stages: Vec<u32> = a.air_values.iter().map(|v| v.stage).collect();
    let agv_stages: Vec<u32> = ag.air_group_values.iter().map(|v| v.stage).collect();
    let hints = pilout
        .hints
        .iter()
        .filter(|h| h.air_id == Some(air as u32) && h.air_group_id == Some(airgroup as u32))
        .map(|h| Hint { name: h.name.clone(), fields: h.hint_fields.clone() })
        .collect();
    within(!pilout.num_challenges.is_empty(), 0, "the pilout declares no challenge stages")?;
    let custom_commits = a
        .custom_commits
        .iter()
        .map(|c| CustomCommit {
            name: c.name.clone().unwrap_or_default(),
            public_values: c.public_values.iter().map(|v| v.idx).collect(),
            stage_widths: c.stage_widths.clone(),
        })
        .collect();
    let publics = named(pilout, SymbolType::PublicValue, None, unnamed, |s, k| {
        let (name, lengths) = cell(s, k);
        NamedValue { name, stage: 1, lengths }
    })?;
    let air_values = staged(
        named(pilout, SymbolType::AirValue, Some((airgroup, Some(air))), Default::default(), cell)?,
        &air_stages,
    )?;
    let airgroup_values = staged(
        named(pilout, SymbolType::AirGroupValue, Some((airgroup, None)), Default::default(), cell)?,
        &agv_stages,
    )?;

    Ok(ConstraintSystem {
        field: field.clone(),
        opening,
        n_stages: pilout.num_challenges.len() as u32,
        n_rows_bits: a.num_rows.map_or(0, |n| n.checked_ilog2().unwrap_or(0)),
        exprs,
        constraints,
        witness,
        fixed,
        custom,
        challenges,
        publics,
        proof_values,
        air_values,
        airgroup_values,
        hints,
        custom_commits,
        airgroup_agg_types: ag.air_group_values.iter().map(|v| v.agg_type).collect(),
    })
}
