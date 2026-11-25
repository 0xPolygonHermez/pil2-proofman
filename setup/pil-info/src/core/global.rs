//! Global constraints and hints: `PiloutCore`'s global tables lowered to the typed `Expr`.

use pil2_pilout::pilout as pb;

use super::from_pilout::{
    bad, check_challenge, check_proof_value, check_public, check_refs, hint_entries, lower_op, proof_values, Op,
    PiloutError,
};
use super::ir::{Expr, ExprId, Hint, HintField, HintValue};
use super::pilout_core::{self as pc, AirGroupValue, GlobalExpression, GlobalOperand, PiloutCore};
use crate::cfg::FieldCfg;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GlobalConstraint {
    pub expr: ExprId,
    pub label: String,
}

/// Global expressions (pilout order), constraints and hints in the same typed `Expr`, with the tables they read.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GlobalSystem {
    pub field: FieldCfg,
    pub exprs: Vec<Expr>,
    pub constraints: Vec<GlobalConstraint>,
    /// The global hints (no airgroup, no air); an airgroup value without airgroup is a hint's.
    pub hints: Vec<Hint>,
    pub num_challenges: Vec<u32>,
    /// Each airgroup's values.
    pub airgroup_values: Vec<Vec<AirGroupValue>>,
    /// Stage of each proof value, by id.
    pub proof_value_stages: Vec<u32>,
}

/// What the global operands are checked against.
struct Tables<'a> {
    core: &'a PiloutCore,
    field: &'a FieldCfg,
    proof_value_stages: &'a [u32],
}

fn lower_operand(o: &GlobalOperand, parent: usize, t: &Tables) -> Result<Expr, PiloutError> {
    let core = t.core;
    Ok(match o {
        GlobalOperand::Expression(r) => Expr::Ref(*r),
        GlobalOperand::Constant(c) => Expr::Const(c % t.field.modulus()),
        GlobalOperand::PublicValue(idx) => {
            check_public(core.num_public_values, *idx, parent)?;
            Expr::Public(*idx)
        }
        GlobalOperand::AirGroupValue(a) => {
            let g = core.air_groups.get(a.air_group_id as usize).ok_or_else(|| bad(parent, "unknown airgroup"))?;
            if a.idx as usize >= g.air_group_values.len() {
                return Err(bad(parent, "airgroup value out of range"));
            }
            Expr::AirgroupValue { airgroup: Some(a.air_group_id), id: a.idx }
        }
        GlobalOperand::Challenge(c) => {
            check_challenge(&core.num_challenges, c.stage, c.idx, parent)?;
            Expr::Challenge { stage: c.stage, stage_id: c.idx }
        }
        GlobalOperand::ProofValue(v) => {
            check_proof_value(t.proof_value_stages, v.stage, v.idx, parent)?;
            Expr::ProofValue(v.idx)
        }
    })
}

fn lower_expression(e: &GlobalExpression, i: usize, t: &Tables) -> Result<Expr, PiloutError> {
    let op = match e {
        GlobalExpression::Add([a, b]) => Op::Add(a, b),
        GlobalExpression::Sub([a, b]) => Op::Sub(a, b),
        GlobalExpression::Mul([a, b]) => Op::Mul(a, b),
        GlobalExpression::Neg(a) => Op::Neg(a),
    };
    lower_op(op, |o| lower_operand(o, i, t))
}

/// A global hint operand: an air operand read globally (`Expression` is a global expression, an airgroup value
/// one of some airgroup's); columns and air values are refused.
fn lower_hint_operand(o: &pc::Operand, hint: &str, t: &Tables) -> Result<Expr, PiloutError> {
    let core = t.core;
    let refuse = |why| PiloutError::BadHint { hint: hint.to_string(), why };
    let at = |r: Result<(), PiloutError>| r.map_err(|_| refuse("operand out of range"));
    Ok(match o {
        pc::Operand::Expression(idx) if (*idx as usize) < core.expressions.len() => Expr::Ref(*idx),
        pc::Operand::Expression(_) => return Err(refuse("a global expression the pilout does not have")),
        pc::Operand::Constant(c) => Expr::Const(c % t.field.modulus()),
        pc::Operand::PublicValue(idx) => {
            at(check_public(core.num_public_values, *idx, 0))?;
            Expr::Public(*idx)
        }
        pc::Operand::AirGroupValue(idx) => {
            let known = core.air_groups.iter().any(|g| (*idx as usize) < g.air_group_values.len());
            at(if known { Ok(()) } else { Err(bad(0, "airgroup value out of range")) })?;
            Expr::AirgroupValue { airgroup: None, id: *idx }
        }
        pc::Operand::Challenge(c) => {
            at(check_challenge(&core.num_challenges, c.stage, c.idx, 0))?;
            Expr::Challenge { stage: c.stage, stage_id: c.idx }
        }
        pc::Operand::ProofValue(v) => {
            at(check_proof_value(t.proof_value_stages, v.stage, v.idx, 0))?;
            Expr::ProofValue(v.idx)
        }
        pc::Operand::AirValue(_) => return Err(refuse("an air value in a global hint")),
        pc::Operand::WitnessCol(_)
        | pc::Operand::CustomCol(_)
        | pc::Operand::FixedCol(_)
        | pc::Operand::PeriodicCol(_) => return Err(refuse("a column operand in a global hint")),
    })
}

fn lower_hint_field(f: &pc::HintField, hint: &str, t: &Tables) -> Result<HintField, PiloutError> {
    let value = match &f.value {
        pc::HintValue::String(s) => HintValue::Str(s.clone()),
        pc::HintValue::Operand(o) => HintValue::Expr(lower_hint_operand(o, hint, t)?),
        pc::HintValue::Array(a) => {
            HintValue::Array(a.iter().map(|f| lower_hint_field(f, hint, t)).collect::<Result<_, _>>()?)
        }
    };
    Ok(HintField { name: f.name.clone(), value })
}

/// Same strict rules as `from_pilout`: an out-of-range operand, or any dangling or cyclic reference is an error.
pub fn global_from_core(core: &PiloutCore, field: &FieldCfg) -> Result<GlobalSystem, PiloutError> {
    let proof_value_stages: Vec<u32> = proof_values(core)?.iter().map(|v| v.stage).collect();
    let t = Tables { core, field, proof_value_stages: &proof_value_stages };
    let exprs =
        core.expressions.iter().enumerate().map(|(i, e)| lower_expression(e, i, &t)).collect::<Result<Vec<_>, _>>()?;
    check_refs(&exprs)?;
    let constraints = (core.constraints.iter())
        .map(|c| {
            if c.expression_idx as usize >= exprs.len() {
                return Err(bad(c.expression_idx as usize, "constraint expression out of range"));
            }
            Ok(GlobalConstraint { expr: c.expression_idx, label: c.debug_line.clone().unwrap_or_default() })
        })
        .collect::<Result<Vec<_>, _>>()?;
    let global_hint = |h: &&pc::Hint| h.air_group_id.is_none() && h.air_id.is_none() && !h.fields.is_empty();
    let hints = (core.hints.iter().filter(global_hint))
        .map(|h| {
            let fields = hint_entries(&h.fields).iter().map(|f| lower_hint_field(f, &h.name, &t));
            Ok(Hint { name: h.name.clone(), fields: fields.collect::<Result<_, _>>()? })
        })
        .collect::<Result<Vec<_>, PiloutError>>()?;
    Ok(GlobalSystem {
        field: field.clone(),
        exprs,
        constraints,
        hints,
        num_challenges: core.num_challenges.clone(),
        airgroup_values: core.air_groups.iter().map(|g| g.air_group_values.clone()).collect(),
        proof_value_stages,
    })
}

/// `global_from_core` over the pilout's global tables.
pub fn global_from_pilout(pilout: &pb::PilOut, field: &FieldCfg) -> Result<GlobalSystem, PiloutError> {
    let mut core = PiloutCore::globals_of(pilout)?;
    core.air_groups = (pilout.air_groups.iter())
        .map(|g| {
            Ok(pc::AirGroup { name: g.name.clone(), air_group_values: pc::air_group_values(g)?, airs: Vec::new() })
        })
        .collect::<Result<_, PiloutError>>()?;
    global_from_core(&core, field)
}
