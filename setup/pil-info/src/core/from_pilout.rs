//! Strict pilout -> ConstraintSystem; rejects, never repairs.

use num_bigint::BigUint;
use pil2_pilout::pilout as pb;

use super::ir::{
    map_expr, visit, Boundary, ColType, Column, Constraint, ConstraintSystem, CustomCommit, Expr, Hint, HintField,
    HintValue, Lengths, NamedValue,
};
use super::pilout_core::{self as pc, AggType, AirGroupValue, PiloutCore, SymbolKind};
pub(crate) use super::pilout_core::bad;
pub use super::pilout_core::PiloutError;
use crate::cfg::{FieldCfg, Opening};

/// The ingestion limit on every table of an air (and on any symbol id); below it ids and counts fit u32.
const MAX_SLOTS: usize = 1 << 24;

fn overflow(what: impl Into<String>) -> PiloutError {
    PiloutError::Overflow { what: what.into() }
}

pub(crate) fn const_of(be: &[u8]) -> Expr {
    Expr::Const(BigUint::from_bytes_be(be))
}

fn refs_of(e: &Expr) -> Vec<u32> {
    let mut r = Vec::new();
    visit(e, &mut |n| {
        if let Expr::Ref(t) = n {
            r.push(*t);
        }
    });
    r
}

/// Every `Ref` of `exprs` resolves and no expression reaches itself; returns a post-order (iterative DFS).
fn post_order(exprs: &[Expr]) -> Result<Vec<usize>, PiloutError> {
    let refs: Vec<Vec<u32>> = exprs.iter().map(refs_of).collect();
    for (i, rs) in refs.iter().enumerate() {
        if let Some(&to) = rs.iter().find(|&&t| t as usize >= exprs.len()) {
            return Err(PiloutError::DanglingRef { expr: i, to });
        }
    }
    // 0 = unvisited, 1 = on the stack, 2 = done
    let mut state = vec![0u8; exprs.len()];
    let mut order = Vec::with_capacity(exprs.len());
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
                    1 => return Err(PiloutError::CyclicRef { expr: t }),
                    0 => {
                        state[t] = 1;
                        stack.push((t, 0));
                    }
                    _ => {}
                }
            } else {
                state[n] = 2;
                order.push(n);
                stack.pop();
            }
        }
    }
    Ok(order)
}

/// Every `Ref` of `exprs` resolves, and no expression reaches itself.
pub(crate) fn check_refs(exprs: &[Expr]) -> Result<(), PiloutError> {
    post_order(exprs).map(|_| ())
}

/// `new[old]`: by depth, then by body with each `Ref` read as its target's structural class; only identical
/// expressions keep their input order, so the numbering is a function of structure and topological.
fn canonical_numbering(exprs: &[Expr]) -> Result<Vec<u32>, PiloutError> {
    let mut depth = vec![0usize; exprs.len()];
    for i in post_order(exprs)? {
        depth[i] = refs_of(&exprs[i]).iter().map(|&t| depth[t as usize] + 1).max().unwrap_or(0);
    }
    let mut by_depth = vec![Vec::new(); depth.iter().max().map_or(0, |d| d + 1)];
    (0..exprs.len()).for_each(|i| by_depth[depth[i]].push(i));
    let (mut class, mut new) = (vec![0u32; exprs.len()], vec![0u32; exprs.len()]);
    let (mut n_class, mut next) = (0u32, 0u32);
    for level in by_depth {
        // the targets are shallower, so their classes are final
        let key = |i: usize| {
            map_expr(&exprs[i], &mut |n| if let Expr::Ref(t) = n { Some(Expr::Ref(class[*t as usize])) } else { None })
        };
        let mut keyed: Vec<(Expr, usize)> = level.into_iter().map(|i| (key(i), i)).collect();
        keyed.sort();
        for (k, (body, i)) in keyed.iter().enumerate() {
            if k > 0 && keyed[k - 1].0 != *body {
                n_class += 1;
            }
            // below the ingestion limit
            (class[*i], new[*i], next) = (n_class, next, next + 1);
        }
        n_class += 1;
    }
    Ok(new)
}

/// `e` with each `Ref(old)` read as `Ref(new[old])`.
fn renumber(e: &Expr, new: &[u32]) -> Expr {
    map_expr(e, &mut |n| if let Expr::Ref(t) = n { Some(Expr::Ref(new[*t as usize])) } else { None })
}

/// Number of cells of a symbol: 1, or the product of its array lengths.
fn symbol_len(s: &pc::Symbol) -> usize {
    if s.dim == 0 {
        1
    } else {
        s.lengths.iter().fold(1usize, |a, l| a.saturating_mul(*l as usize))
    }
}

/// Indices of the `k`-th cell of `s`, last index fastest; `None` for a scalar.
fn cell_lengths(s: &pc::Symbol, k: usize) -> Lengths {
    if s.dim == 0 {
        return None;
    }
    let mut rest = k;
    let mut idx = vec![0u32; s.lengths.len()];
    for (i, l) in s.lengths.iter().enumerate().rev() {
        let l = (*l).max(1);
        // `rest % l < l`, a u32
        idx[i] = (rest % l as usize) as u32;
        rest /= l as usize;
    }
    Some(idx)
}

fn cell(s: &pc::Symbol, k: usize) -> (String, Lengths) {
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
pub fn stage_base(widths: &[u32], stage: u32) -> Result<u32, PiloutError> {
    let mut below = widths.iter().take(stage.saturating_sub(1) as usize);
    below.try_fold(0u32, |a, w| a.checked_add(*w)).ok_or_else(|| overflow("a stage's first column id"))
}

/// What the operands of one air may refer to.
struct Tables<'a> {
    stage_widths: &'a [u32],
    custom_commits: &'a [pc::CustomCommit],
    n_fixed: u32,
    n_air_values: usize,
    n_airgroup_values: usize,
    num_challenges: &'a [u32],
    n_publics: u32,
    num_proof_values: &'a [u32],
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

fn lower_operand(o: &pc::Operand, parent: usize, t: &Tables) -> Result<Expr, PiloutError> {
    use pc::Operand as O;
    Ok(match o {
        O::Expression(idx) => Expr::Ref(*idx),
        O::Constant(c) => Expr::Const(c.clone()),
        O::WitnessCol(w) => {
            check_staged_col(t.stage_widths, w.stage, w.col_idx, parent)?;
            Expr::Col {
                col: ColType::Witness,
                commit_id: None,
                // both below the ingestion limit
                id: stage_base(t.stage_widths, w.stage)? + w.col_idx,
                offset: i64::from(w.row_offset),
            }
        }
        O::CustomCol(c) => {
            let w = t.custom_commits.get(c.commit_id as usize).ok_or_else(|| bad(parent, "unknown custom commit"))?;
            within(c.stage == 0, parent, "custom column of a stage other than 0")?;
            let first = w.stage_widths.first();
            within(first.is_some_and(|w| c.col_idx < *w), parent, "custom column outside the stage widths")?;
            Expr::Col {
                col: ColType::Custom,
                commit_id: Some(c.commit_id),
                id: c.col_idx,
                offset: i64::from(c.row_offset),
            }
        }
        O::FixedCol(f) => {
            within(f.idx < t.n_fixed, parent, "fixed column out of range")?;
            Expr::Col { col: ColType::Fixed, commit_id: None, id: f.idx, offset: i64::from(f.row_offset) }
        }
        O::PeriodicCol(_) => return Err(bad(parent, "a periodic column operand, which the STARK does not read")),
        O::PublicValue(idx) => {
            check_public(t.n_publics, *idx, parent)?;
            Expr::Public(*idx)
        }
        O::AirGroupValue(idx) => {
            within((*idx as usize) < t.n_airgroup_values, parent, "airgroup value out of range")?;
            Expr::AirgroupValue { airgroup: None, id: *idx }
        }
        O::AirValue(idx) => {
            within((*idx as usize) < t.n_air_values, parent, "air value out of range")?;
            Expr::AirValue(*idx)
        }
        O::Challenge(c) => {
            check_challenge(t.num_challenges, c.stage, c.idx, parent)?;
            Expr::Challenge { stage: c.stage, stage_id: c.idx }
        }
        O::ProofValue(p) => {
            check_proof_value(t.num_proof_values, p.idx, parent)?;
            Expr::ProofValue(p.idx)
        }
    })
}

pub(crate) fn check_public(n_publics: u32, idx: u32, parent: usize) -> Result<(), PiloutError> {
    within(idx < n_publics, parent, "public value out of range")
}

/// The proof-value index is global: bounded by the sum over stages.
pub(crate) fn check_proof_value(per_stage: &[u32], idx: u32, parent: usize) -> Result<(), PiloutError> {
    within(u64::from(idx) < per_stage.iter().map(|n| u64::from(*n)).sum(), parent, "proof value out of range")
}

pub(crate) fn check_challenge(num_challenges: &[u32], stage: u32, idx: u32, parent: usize) -> Result<(), PiloutError> {
    let n = (stage as usize).checked_sub(1).and_then(|s| num_challenges.get(s));
    within(n.is_some_and(|n| idx < *n), parent, "challenge out of range")
}

/// An operation with its operands, the same for an air's and a global expression.
pub(crate) enum Op<'a, O> {
    Add(&'a O, &'a O),
    Sub(&'a O, &'a O),
    Mul(&'a O, &'a O),
    Neg(&'a O),
}

/// The only Add/Sub/Mul/Neg lowering; `lo` lowers one operand.
pub(crate) fn lower_op<O>(op: Op<O>, mut lo: impl FnMut(&O) -> Result<Expr, PiloutError>) -> Result<Expr, PiloutError> {
    Ok(match op {
        Op::Add(a, b) => Expr::Add(Box::new(lo(a)?), Box::new(lo(b)?)),
        Op::Sub(a, b) => Expr::Sub(Box::new(lo(a)?), Box::new(lo(b)?)),
        Op::Mul(a, b) => Expr::Mul(Box::new(lo(a)?), Box::new(lo(b)?)),
        Op::Neg(a) => Expr::Neg(Box::new(lo(a)?)),
    })
}

fn lower_expression(e: &pc::Expression, i: usize, t: &Tables) -> Result<Expr, PiloutError> {
    let op = match e {
        pc::Expression::Add([a, b]) => Op::Add(a, b),
        pc::Expression::Sub([a, b]) => Op::Sub(a, b),
        pc::Expression::Mul([a, b]) => Op::Mul(a, b),
        pc::Expression::Neg(a) => Op::Neg(a),
    };
    lower_op(op, |o| lower_operand(o, i, t))
}

fn lower_constraint(k: usize, c: &pc::Constraint, n_exprs: usize, opening: Opening) -> Result<Constraint, PiloutError> {
    // an absent debug line is an empty label
    let label = c.debug_line.clone().unwrap_or_default();
    if opening == Opening::Deep && c.kind != Boundary::EveryRow {
        return Err(PiloutError::BoundaryNotEveryRow { constraint: k, line: label, boundary: c.kind.name().into() });
    }
    if c.expression_idx as usize >= n_exprs {
        return Err(bad(c.expression_idx as usize, "constraint expression out of range"));
    }
    Ok(Constraint { e: Expr::Ref(c.expression_idx), boundary: c.kind.clone(), label })
}

fn lower_hint_field(f: &pc::HintField, hint: &str, t: &Tables) -> Result<HintField, PiloutError> {
    let value = match &f.value {
        pc::HintValue::String(s) => HintValue::Str(s.clone()),
        pc::HintValue::Operand(o) => HintValue::Expr(lower_operand(o, 0, t).map_err(|e| match e {
            PiloutError::BadOperand { why, .. } => PiloutError::BadHint { hint: hint.to_string(), why },
            e => e,
        })?),
        pc::HintValue::Array(a) => {
            HintValue::Array(a.iter().map(|f| lower_hint_field(f, hint, t)).collect::<Result<_, _>>()?)
        }
    };
    Ok(HintField { name: f.name.clone(), value })
}

fn renumber_hint(f: HintField, new: &[u32]) -> HintField {
    let value = match f.value {
        HintValue::Expr(e) => HintValue::Expr(renumber(&e, new)),
        HintValue::Array(a) => HintValue::Array(a.into_iter().map(|f| renumber_hint(f, new)).collect()),
        s @ HintValue::Str(_) => s,
    };
    HintField { value, ..f }
}

/// This air's symbols (`air` None: of its airgroup only) of kind `kind`.
fn of_air(core: &PiloutCore, kind: SymbolKind, airgroup: u32, air: Option<u32>) -> impl Iterator<Item = &pc::Symbol> {
    core.symbols
        .iter()
        .filter(move |s| s.kind == kind && s.air_group_id == Some(airgroup) && air.is_none_or(|a| s.air_id == Some(a)))
}

/// Columns of `symbols`, arrays expanded, ids from `base` of each symbol's stage.
fn columns<'a>(
    symbols: impl Iterator<Item = &'a pc::Symbol>,
    stage: impl Fn(&pc::Symbol) -> Result<u32, PiloutError>,
    base: impl Fn(u32) -> Result<u32, PiloutError>,
    field: &FieldCfg,
) -> Result<Vec<Column>, PiloutError> {
    let mut out = Vec::new();
    for s in symbols {
        let stage = stage(s)?;
        let dim = if stage <= 1 { 1 } else { field.ext_dim() as u32 };
        let n = symbol_len(s);
        let fits = |x: usize| x.saturating_add(n) <= MAX_SLOTS;
        within(fits(s.id as usize) && fits(out.len()), s.id as usize, "columns past the ingestion limit")?;
        // the base, the id and the cell are each below the ingestion limit
        let first = base(stage)? + s.id;
        for k in 0..n {
            out.push(Column { name: s.name.clone(), stage, dim, id: first + k as u32, lengths: cell_lengths(s, k) });
        }
    }
    Ok(out)
}

fn declared_stage(s: &pc::Symbol) -> Result<u32, PiloutError> {
    s.stage.ok_or_else(|| PiloutError::MissingStage { symbol: s.name.clone() })
}

/// `mk(symbol, cell)` of `symbols` (arrays expanded) placed at their ids; a hole is an error.
fn named<'a, T: Clone>(
    symbols: impl Iterator<Item = &'a pc::Symbol>,
    mk: impl Fn(&pc::Symbol, usize) -> Result<T, PiloutError>,
) -> Result<Vec<T>, PiloutError> {
    let mut v: Vec<Option<T>> = Vec::new();
    for s in symbols {
        for k in 0..symbol_len(s) {
            place(&mut v, (s.id as usize).saturating_add(k), None, Some(mk(s, k)?))?;
        }
    }
    v.into_iter().enumerate().map(|(i, x)| x.ok_or_else(|| bad(i, "a value with no symbol"))).collect()
}

fn global(core: &PiloutCore, kind: SymbolKind) -> impl Iterator<Item = &pc::Symbol> {
    core.symbols.iter().filter(move |s| s.kind == kind)
}

/// Pairs the declared stages with the symbol cells; a symbol past the table is an error, a trailing unnamed value gets "".
fn staged(cells: Vec<(String, Lengths)>, stages: &[u32]) -> Result<Vec<NamedValue>, PiloutError> {
    within(cells.len() <= stages.len(), 0, "a value symbol beyond the declared values")?;
    let at = |i: usize| cells.get(i).cloned().unwrap_or_default();
    Ok(stages.iter().enumerate().map(|(i, st)| NamedValue { name: at(i).0, stage: *st, lengths: at(i).1 }).collect())
}

/// Lowers air (`airgroup`, `air`) of `pilout`: `from_pilout_core` over that air and `from_pb`'s globals.
pub fn from_pilout(
    pilout: &pb::PilOut,
    airgroup: usize,
    air: usize,
    field: &FieldCfg,
    opening: Opening,
) -> Result<ConstraintSystem, PiloutError> {
    let unknown = PiloutError::UnknownAir { airgroup, air };
    let ag = pilout.air_groups.get(airgroup).ok_or(unknown.clone())?;
    let a = pc::air_of(ag.airs.get(air).ok_or(unknown)?, airgroup, air)?;
    lower(&PiloutCore::globals_of(pilout)?, &pc::air_group_values(ag)?, &a, airgroup, air, field, opening)
}

/// Lowers air (`airgroup`, `air`) of `core`.
pub fn from_pilout_core(
    core: &PiloutCore,
    airgroup: usize,
    air: usize,
    field: &FieldCfg,
    opening: Opening,
) -> Result<ConstraintSystem, PiloutError> {
    let unknown = PiloutError::UnknownAir { airgroup, air };
    let ag = core.air_groups.get(airgroup).ok_or(unknown.clone())?;
    lower(core, &ag.air_group_values, ag.airs.get(air).ok_or(unknown)?, airgroup, air, field, opening)
}

/// The ingestion limit: every table of `a` and of `core` below `MAX_SLOTS`.
fn check_limits(core: &PiloutCore, a: &pc::Air) -> Result<(), PiloutError> {
    let sum = |w: &[u32]| w.iter().map(|x| *x as usize).sum::<usize>();
    let tables = [
        a.expressions.len(),
        a.constraints.len(),
        sum(&a.stage_widths),
        a.n_fixed_cols as usize,
        a.n_periodic_cols as usize,
        a.air_values.len(),
        a.custom_commits.len(),
        core.num_challenges.len(),
    ];
    let custom = a.custom_commits.iter().map(|c| sum(&c.stage_widths));
    if tables.into_iter().chain(custom).any(|n| n >= MAX_SLOTS) {
        return Err(overflow("a table of the air past the ingestion limit"));
    }
    Ok(())
}

/// Lowers `a` of airgroup values `agvs`; of `core` only the pilout-wide tables, symbols and hints are read.
fn lower(
    core: &PiloutCore,
    agvs: &[AirGroupValue],
    a: &pc::Air,
    airgroup: usize,
    air: usize,
    field: &FieldCfg,
    opening: Opening,
) -> Result<ConstraintSystem, PiloutError> {
    check_limits(core, a)?;
    let unknown = || PiloutError::UnknownAir { airgroup, air };
    let (g32, a32) = (u32::try_from(airgroup).map_err(|_| unknown())?, u32::try_from(air).map_err(|_| unknown())?);
    if a.constraints.is_empty() {
        return Err(PiloutError::NoConstraints { airgroup, air });
    }
    let tables = Tables {
        stage_widths: &a.stage_widths,
        custom_commits: &a.custom_commits,
        n_fixed: a.n_fixed_cols,
        n_air_values: a.air_values.len(),
        n_airgroup_values: agvs.len(),
        num_challenges: &core.num_challenges,
        n_publics: core.num_public_values,
        num_proof_values: &core.num_proof_values,
    };
    let rows = a.num_rows;
    if !rows.is_power_of_two() {
        return Err(PiloutError::NumRowsNotPowerOfTwo { airgroup, air, rows });
    }
    let exprs = a
        .expressions
        .iter()
        .enumerate()
        .map(|(i, e)| lower_expression(e, i, &tables))
        .collect::<Result<Vec<_>, _>>()?;
    let new = canonical_numbering(&exprs)?;
    let mut numbered = vec![Expr::Const(BigUint::default()); exprs.len()];
    exprs.iter().zip(&new).for_each(|(e, &k)| numbered[k as usize] = renumber(e, &new));
    let exprs = numbered;
    let constraints = a
        .constraints
        .iter()
        .enumerate()
        .map(|(k, c)| lower_constraint(k, c, exprs.len(), opening).map(|c| Constraint { e: renumber(&c.e, &new), ..c }))
        .collect::<Result<Vec<_>, _>>()?;

    let air_syms = |kind| of_air(core, kind, g32, Some(a32));
    let witness = columns(air_syms(SymbolKind::WitnessCol), declared_stage, |s| stage_base(&a.stage_widths, s), field)?;
    // fixed and custom columns are stage 0; a custom column declaring another stage is refused
    let stage0 = |s: &pc::Symbol| match s.stage {
        None | Some(0) => Ok(0),
        Some(_) => Err(bad(s.id as usize, "custom column of a stage other than 0")),
    };
    let fixed = columns(air_syms(SymbolKind::FixedCol), |_| Ok(0), |_| Ok(0), field)?;
    for s in air_syms(SymbolKind::CustomCol) {
        let c = s.commit_id.ok_or_else(|| PiloutError::MissingCommitId { symbol: s.name.clone() })?;
        within((c as usize) < a.custom_commits.len(), s.id as usize, "custom column of an unknown commit")?;
    }
    let custom = (0..a.custom_commits.len())
        .map(|c| {
            let syms = air_syms(SymbolKind::CustomCol).filter(move |s| s.commit_id.is_some_and(|x| x as usize == c));
            columns(syms, stage0, |_| Ok(0), field)
        })
        .collect::<Result<Vec<_>, _>>()?;

    let challenges = global(core, SymbolKind::Challenge)
        .map(|s| Ok((declared_stage(s)?, s.id, s.name.clone())))
        .collect::<Result<Vec<_>, PiloutError>>()?;
    let proof_values = named(global(core, SymbolKind::ProofValue), |s, k| {
        let (name, lengths) = cell(s, k);
        Ok(NamedValue { name, stage: declared_stage(s)?, lengths })
    })?;
    let hints = core
        .hints
        .iter()
        .filter(|h| h.air_id == Some(a32) && h.air_group_id == Some(g32))
        .map(|h| {
            // only what `Hint::entries` reads: a leading array alone, else every field
            let read = match h.fields.first() {
                Some(f @ pc::HintField { value: pc::HintValue::Array(_), .. }) => std::slice::from_ref(f),
                _ => &h.fields[..],
            };
            let lower = |f| lower_hint_field(f, &h.name, &tables).map(|f| renumber_hint(f, &new));
            let fields = read.iter().map(lower).collect::<Result<_, _>>()?;
            Ok(Hint { name: h.name.clone(), fields })
        })
        .collect::<Result<Vec<_>, PiloutError>>()?;
    let n_publics = core.num_public_values;
    let custom_commits = a
        .custom_commits
        .iter()
        .map(|c| match c.public_values.iter().find(|&&idx| idx >= n_publics) {
            Some(&idx) => Err(PiloutError::CommitPublic { commit: c.name.clone(), idx, declared: n_publics }),
            None => Ok(CustomCommit {
                name: c.name.clone(),
                public_values: c.public_values.clone(),
                stage_widths: c.stage_widths.clone(),
            }),
        })
        .collect::<Result<_, _>>()?;
    let publics = named(global(core, SymbolKind::PublicValue), |s, k| {
        let (name, lengths) = cell(s, k);
        Ok(NamedValue { name, stage: 1, lengths })
    })?;
    if publics.len() != n_publics as usize {
        return Err(PiloutError::PublicCount { declared: n_publics, named: publics.len() });
    }
    let cells = |kind, air| named(of_air(core, kind, g32, air), |s, k| Ok(cell(s, k)));
    let air_values = staged(cells(SymbolKind::AirValue, Some(a32))?, &a.air_values)?;
    let agv_stages: Vec<u32> = agvs.iter().map(|v| v.stage).collect();
    let airgroup_values = staged(cells(SymbolKind::AirGroupValue, None)?, &agv_stages)?;

    // n_stages = max(highest declared witness/value stage, numChallenges.len(), 1); extra stages have no challenges
    let declared = witness
        .iter()
        .map(|c| c.stage)
        .chain([&air_values, &airgroup_values, &proof_values].into_iter().flat_map(|v| v.iter().map(|x| x.stage)));
    let n_stages = declared.fold((core.num_challenges.len() as u32).max(1), u32::max);
    if n_stages as usize >= MAX_SLOTS {
        return Err(overflow("the stage count past the ingestion limit"));
    }

    Ok(ConstraintSystem {
        field: field.clone(),
        opening,
        n_stages,
        n_rows_bits: rows.trailing_zeros(),
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
        airgroup_agg_types: agvs
            .iter()
            .map(|v| match v.agg_type {
                AggType::Sum => 0,
                AggType::Prod => 1,
            })
            .collect(),
    })
}
