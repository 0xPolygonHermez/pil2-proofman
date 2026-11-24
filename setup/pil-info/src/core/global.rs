//! Global constraints.
//! TODO: `output::global_constraints` still formats globalConstraints.json from `pb` (format_global_*); moving it
//! here needs the global hints typed (not in `GlobalSystem`) and the JSON kept byte-identical. Tests call this only.

use pil2_pilout::pilout::{self as pb, global_expression as gx, global_operand as go};

use super::from_pilout::{
    bad, check_challenge, check_proof_value, check_public, check_refs, const_of, lower_op, Op, PiloutError,
};
use super::ir::Expr;
use crate::cfg::FieldCfg;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GlobalConstraint {
    pub e: Expr,
    pub label: String,
}

/// Global expressions and constraints in the same typed `Expr`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GlobalSystem {
    pub field: FieldCfg,
    pub exprs: Vec<Expr>,
    pub constraints: Vec<GlobalConstraint>,
}

fn lower_operand(o: &Option<pb::GlobalOperand>, parent: usize, p: &pb::PilOut) -> Result<Expr, PiloutError> {
    let inner = o.as_ref().and_then(|o| o.operand.as_ref()).ok_or_else(|| bad(parent, "missing operand"))?;
    let public = |idx: u32| check_public(p.num_public_values, idx, parent).map(|()| Expr::Public(idx));
    Ok(match inner {
        go::Operand::Expression(r) => Expr::Ref(r.idx),
        go::Operand::Constant(c) => const_of(&c.value),
        go::Operand::PublicValue(v) => public(v.idx)?,
        go::Operand::PublicTableAggregatedValue(v) => public(v.idx)?,
        go::Operand::PublicTableColumn(v) => public(v.idx)?,
        go::Operand::AirGroupValue(a) => {
            let g = p.air_groups.get(a.air_group_id as usize).ok_or_else(|| bad(parent, "unknown airgroup"))?;
            if a.idx as usize >= g.air_group_values.len() {
                return Err(bad(parent, "airgroup value out of range"));
            }
            Expr::AirgroupValue { airgroup: Some(a.air_group_id), id: a.idx }
        }
        go::Operand::Challenge(c) => {
            check_challenge(&p.num_challenges, c.stage, c.idx, parent)?;
            Expr::Challenge { stage: c.stage, stage_id: c.idx }
        }
        go::Operand::ProofValue(v) => {
            check_proof_value(&p.num_proof_values, v.idx, parent)?;
            Expr::ProofValue(v.idx)
        }
    })
}

fn lower_expression(e: &pb::GlobalExpression, i: usize, p: &pb::PilOut) -> Result<Expr, PiloutError> {
    let op = match e.operation.as_ref().ok_or_else(|| bad(i, "expression without an operation"))? {
        gx::Operation::Add(a) => Op::Add(&a.lhs, &a.rhs),
        gx::Operation::Sub(a) => Op::Sub(&a.lhs, &a.rhs),
        gx::Operation::Mul(a) => Op::Mul(&a.lhs, &a.rhs),
        gx::Operation::Neg(a) => Op::Neg(&a.value),
    };
    lower_op(op, |o| lower_operand(o, i, p))
}

/// Same strict rules as `from_pilout`: a constraint without an expression, an out-of-range operand, or any dangling or cyclic reference is an error.
pub fn global_from_pilout(pilout: &pb::PilOut, field: &FieldCfg) -> Result<GlobalSystem, PiloutError> {
    let exprs = pilout
        .expressions
        .iter()
        .enumerate()
        .map(|(i, e)| lower_expression(e, i, pilout))
        .collect::<Result<Vec<_>, _>>()?;
    check_refs(&exprs)?;
    let constraints = pilout
        .constraints
        .iter()
        .enumerate()
        .map(|(k, c)| {
            let idx = c.expression_idx.as_ref().ok_or(PiloutError::MissingExpressionIdx { constraint: k })?.idx;
            if idx as usize >= exprs.len() {
                return Err(bad(idx as usize, "constraint expression out of range"));
            }
            Ok(GlobalConstraint { e: Expr::Ref(idx), label: c.debug_line.clone().unwrap_or_default() })
        })
        .collect::<Result<Vec<_>, _>>()?;
    Ok(GlobalSystem { field: field.clone(), exprs, constraints })
}
