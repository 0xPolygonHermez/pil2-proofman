//! Typed IR expressions and hints -> the expression arena and hint infos the code generator reads.

use crate::core::global::GlobalSystem;
use crate::core::ir::{col_dim, value_dim, ColType, ConstraintSystem, Expr, Hint, HintField, HintValue};
use crate::error::{PilInfoError, Result};
use crate::expr::expression::{ExprChild, Expression};
use crate::types::pilout_info::{HintFieldEntry, HintFieldValue, HintInfo};

/// Formats one leaf (neither an operation nor a `Ref`).
pub type Leaf<'a> = dyn Fn(&Expr) -> Result<Expression> + 'a;

fn invalid(why: String) -> PilInfoError {
    PilInfoError::InvalidPilout(why)
}

fn node(op: &str) -> Expression {
    Expression { op: op.into(), ..Default::default() }
}

/// `e` as an operand; a `Ref` to `add/sub(x, 0)` with `x` not a `Ref` reads as `x`.
fn operand(exprs: &[Expr], leaf: &Leaf, e: &Expr) -> Result<Expression> {
    let Expr::Ref(id) = e else { return leaf(e) };
    let target = exprs.get(*id as usize).ok_or_else(|| invalid(format!("a reference to expression {id}")))?;
    if let Expr::Add(x, z) | Expr::Sub(x, z) = target {
        if !matches!(**x, Expr::Ref(_)) && matches!(&**z, Expr::Const(c) if c.bits() == 0) {
            return leaf(x);
        }
    }
    Ok(Expression { id: Some(*id as usize), ..node("exp") })
}

/// One arena entry per expression, in index order, its operands inline.
pub fn from_ir(exprs: &[Expr], leaf: &Leaf) -> Result<Vec<Expression>> {
    let child = |x: &Expr| operand(exprs, leaf, x).map(|e| ExprChild::Inline(Box::new(e)));
    let entry = |(i, e): (usize, &Expr)| {
        let (op, values) = match e {
            Expr::Add(a, b) => ("add", vec![child(a)?, child(b)?]),
            Expr::Sub(a, b) => ("sub", vec![child(a)?, child(b)?]),
            Expr::Mul(a, b) => ("mul", vec![child(a)?, child(b)?]),
            Expr::Neg(a) => ("neg", vec![child(a)?]),
            _ => return Err(invalid(format!("expression {i} is not an operation"))),
        };
        Ok(Expression { values, ..node(op) })
    };
    exprs.iter().enumerate().map(entry).collect()
}

/// The hints' fields (`Hint::entries`), a hint without fields skipped; an `exp` operand marks its entry `keep`.
pub fn hints(hints: &[Hint], exprs: &[Expr], leaf: &Leaf, arena: &mut [Expression]) -> Result<Vec<HintInfo>> {
    let mut out = Vec::new();
    for h in hints.iter().filter(|h| !h.fields.is_empty()) {
        let mut fields = Vec::new();
        for f in h.entries() {
            let (value, lengths) = hint_value(f, exprs, leaf, arena)?;
            let values = match (value, &lengths) {
                (HintFieldValue::Array(a), Some(_)) => *a,
                (v, _) => vec![v],
            };
            fields.push(HintFieldEntry { name: f.name.clone().unwrap_or_default(), values, lengths });
        }
        out.push(HintInfo { name: h.name.clone(), fields });
    }
    Ok(out)
}

/// A field's value; an array's lengths are its own, then the first nested length met at each depth.
fn hint_value(
    f: &HintField,
    exprs: &[Expr],
    leaf: &Leaf,
    arena: &mut [Expression],
) -> Result<(HintFieldValue, Option<Vec<usize>>)> {
    Ok(match &f.value {
        HintValue::Array(a) => {
            let (mut values, mut lengths) = (Vec::with_capacity(a.len()), Vec::new());
            for sub in a {
                let (v, sub_lengths) = hint_value(sub, exprs, leaf, arena)?;
                values.push(v);
                if lengths.is_empty() {
                    lengths.push(a.len());
                }
                for (k, &l) in sub_lengths.iter().flatten().enumerate() {
                    if k + 1 >= lengths.len() {
                        lengths.resize(k + 2, 0);
                    }
                    if lengths[k + 1] == 0 {
                        lengths[k + 1] = l;
                    }
                }
            }
            (HintFieldValue::Array(Box::new(values)), Some(lengths))
        }
        HintValue::Expr(e) => {
            let v = operand(exprs, leaf, e)?;
            if v.op == "exp" {
                if let Some(target) = v.id.and_then(|id| arena.get_mut(id)) {
                    target.keep = Some(true);
                }
            }
            (HintFieldValue::Single(Box::new(v)), None)
        }
        HintValue::Str(s) => {
            (HintFieldValue::Single(Box::new(Expression { value: Some(s.clone()), ..node("string") })), None)
        }
    })
}

/// `count[s]` = how many of `stages` are stage `s + 1`, over the first `n` stages.
fn per_stage(stages: impl Iterator<Item = u32>, n: usize) -> Vec<usize> {
    let mut v = vec![0; n];
    stages.for_each(|s| {
        if let Some(w) = (s as usize).checked_sub(1).and_then(|i| v.get_mut(i)) {
            *w += 1;
        }
    });
    v
}

/// The id of a challenge: its index within its stage after the earlier stages' counts.
fn challenge(per_stage: &[usize], stage: u32, stage_id: u32) -> Expression {
    let before: usize = per_stage.iter().take((stage as usize).saturating_sub(1)).sum();
    let (stage, stage_id) = (stage as usize, stage_id as usize);
    Expression { stage, stage_id: Some(stage_id), id: Some(stage_id + before), ..node("challenge") }
}

fn number(e: &Expr) -> Option<Expression> {
    match e {
        Expr::Const(c) => Some(Expression { value: Some(c.to_string()), ..node("number") }),
        Expr::Public(id) => Some(Expression { id: Some(*id as usize), stage: 1, ..node("public") }),
        _ => None,
    }
}

fn value(op: &str, id: u32, stage: u32, ext: usize) -> Expression {
    Expression { id: Some(id as usize), stage: stage as usize, dim: value_dim(stage, ext), ..node(op) }
}

/// The leaves of an air's expressions and hints.
pub fn air_leaf(cs: &ConstraintSystem) -> impl Fn(&Expr) -> Result<Expression> + '_ {
    let ext = cs.field.ext_dim();
    let n = cs.n_stages as usize;
    let widths = per_stage(cs.witness.iter().map(|c| c.stage), n);
    let challenges = per_stage(cs.challenges.iter().map(|c| c.0), n);
    let mut witness_stage = vec![None; cs.witness.len()];
    for c in &cs.witness {
        if let Some(s) = witness_stage.get_mut(c.id as usize) {
            *s = Some(c.stage);
        }
    }
    let stage_of = |v: &[crate::core::ir::NamedValue], id: u32| v.get(id as usize).map_or(0, |v| v.stage);
    move |e: &Expr| {
        if let Some(x) = number(e) {
            return Ok(x);
        }
        let col = |op: &str, id: u32, stage: u32, offset: i64| Expression {
            id: Some(id as usize),
            row_offset: Some(offset),
            stage: stage as usize,
            dim: col_dim(stage, ext),
            ..node(op)
        };
        Ok(match *e {
            Expr::Col { col: ColType::Witness, id, offset, .. } => {
                let stage = witness_stage.get(id as usize).copied().flatten();
                let stage = stage.ok_or_else(|| invalid(format!("witness column {id} has no symbol")))?;
                let base: usize = widths.iter().take((stage as usize).saturating_sub(1)).sum();
                let stage_id =
                    (id as usize).checked_sub(base).ok_or_else(|| invalid(format!("witness column {id}")))?;
                Expression { stage_id: Some(stage_id), ..col("cm", id, stage, offset) }
            }
            Expr::Col { col: ColType::Fixed, id, offset, .. } => col("const", id, 0, offset),
            Expr::Col { col: ColType::Custom, commit_id: Some(c), id, offset }
                if (c as usize) < cs.custom_commits.len() =>
            {
                Expression { stage_id: Some(id as usize), commit_id: Some(c as usize), ..col("custom", id, 0, offset) }
            }
            Expr::Challenge { stage, stage_id } => challenge(&challenges, stage, stage_id),
            Expr::ProofValue(id) => {
                let v = cs.proof_values.get(id as usize).ok_or_else(|| invalid(format!("proof value {id}")))?;
                value("proofvalue", id, v.stage, ext)
            }
            Expr::AirValue(id) => value("airvalue", id, stage_of(&cs.air_values, id), ext),
            Expr::AirgroupValue { airgroup: None, id } => {
                value("airgroupvalue", id, stage_of(&cs.airgroup_values, id), ext)
            }
            _ => return Err(invalid(format!("{e:?} is not an operand of an air"))),
        })
    }
}

/// The leaves of the global expressions and hints; a hint's airgroup and air values have no airgroup.
pub fn global_leaf(gs: &GlobalSystem) -> impl Fn(&Expr) -> Result<Expression> + '_ {
    let ext = gs.field.ext_dim();
    let challenges: Vec<usize> = gs.num_challenges.iter().map(|&n| n as usize).collect();
    move |e: &Expr| {
        if let Some(x) = number(e) {
            return Ok(x);
        }
        Ok(match *e {
            Expr::AirgroupValue { airgroup: Some(g), id } => {
                let stage = gs.airgroup_value_stages.get(g as usize).and_then(|v| v.get(id as usize)).copied();
                Expression { airgroup_id: Some(g as usize), ..value("airgroupvalue", id, stage.unwrap_or(0), ext) }
            }
            Expr::AirgroupValue { airgroup: None, id } => {
                Expression { id: Some(id as usize), dim: ext, ..node("airgroupvalue") }
            }
            Expr::AirValue(id) => Expression { id: Some(id as usize), ..node("airvalue") },
            Expr::Challenge { stage, stage_id } => challenge(&challenges, stage, stage_id),
            Expr::ProofValue(id) => {
                let stage =
                    gs.proof_value_stages.get(id as usize).ok_or_else(|| invalid(format!("proof value {id}")))?;
                value("proofvalue", id, *stage, ext)
            }
            _ => return Err(invalid(format!("{e:?} is not a global operand"))),
        })
    }
}
