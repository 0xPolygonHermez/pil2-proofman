use indexmap::IndexMap;
use num_bigint::BigUint;

use pil2_pilout::pilout::{
    self as pb, constraint, expression as expr_mod, global_expression as gexpr_mod, global_operand, hint_field,
    operand, SymbolType,
};

use crate::cfg::FieldCfg;
use crate::error::{PilInfoError, Result};
use crate::expr::expression::{ExprChild, Expression, ExpressionArena};

// ---------------------------------------------------------------------------
// Intermediate result types
// ---------------------------------------------------------------------------

/// A formatted constraint extracted from the protobuf.
#[derive(Debug, Clone)]
pub struct ConstraintInfo {
    pub boundary: String,
    pub e: usize,
    pub line: Option<String>,
    pub offset_min: Option<u32>,
    pub offset_max: Option<u32>,
    pub stage: Option<usize>,
    pub im_pol: bool,
}

/// A formatted symbol extracted from the protobuf.
#[derive(Debug, Clone)]
pub struct SymbolInfo {
    pub name: String,
    pub sym_type: String,
    pub stage: Option<usize>,
    pub dim: usize,
    pub id: Option<usize>,
    pub pol_id: Option<usize>,
    pub stage_id: Option<usize>,
    pub air_id: Option<usize>,
    pub airgroup_id: Option<usize>,
    pub commit_id: Option<usize>,
    pub lengths: Option<Vec<usize>>,
    pub idx: Option<usize>,
    pub stage_pos: Option<usize>,
    pub im_pol: bool,
    pub exp_id: Option<usize>,
}

/// A single hint field value (leaf or nested array).
#[derive(Debug, Clone)]
pub enum HintFieldValue {
    /// A leaf: an expression node (op="exp", op="string", etc.)
    Single(Box<Expression>),
    /// A nested array of hint field values
    Array(Box<Vec<HintFieldValue>>),
}

/// A named hint field with value(s) and optional dimension lengths.
#[derive(Debug, Clone)]
pub struct HintFieldEntry {
    pub name: String,
    pub values: Vec<HintFieldValue>,
    pub lengths: Option<Vec<usize>>,
}

/// A formatted hint.
#[derive(Debug, Clone)]
pub struct HintInfo {
    pub name: String,
    pub fields: Vec<HintFieldEntry>,
}

/// Custom commit metadata.
#[derive(Debug, Clone)]
pub struct CustomCommitInfo {
    pub name: String,
    pub stage_widths: Vec<u32>,
    pub public_values: Vec<u32>,
}

/// Aggregate result from `get_pilout_info`.
#[derive(Debug)]
pub struct SetupResult {
    pub name: String,
    pub air_id: usize,
    pub airgroup_id: usize,

    pub pil_power: u32,
    pub n_stages: usize,
    pub n_constants: usize,
    pub n_publics: usize,
    pub n_commitments: usize,

    pub cm_pols_map: Vec<SymbolInfo>,
    pub const_pols_map: Vec<SymbolInfo>,
    pub challenges_map: Vec<SymbolInfo>,
    pub publics_map: Vec<SymbolInfo>,
    pub proof_values_map: Vec<SymbolInfo>,
    pub airgroup_values_map: Vec<SymbolInfo>,
    pub air_values_map: Vec<SymbolInfo>,

    pub map_sections_n: IndexMap<String, usize>,

    pub custom_commits: Vec<CustomCommitInfo>,
    pub custom_commits_map: Vec<Vec<SymbolInfo>>,
    pub air_group_values: Vec<pb::AirGroupValue>,

    pub expressions: Vec<Expression>,
    pub constraints: Vec<ConstraintInfo>,
    pub symbols: Vec<SymbolInfo>,
    pub hints: Vec<HintInfo>,

    /// Number of witness columns in stage 1 that are not intermediate polynomials.
    pub n_commitments_stage1: usize,
    /// Intermediate polynomial expression strings: (base_field, extended_field).
    pub im_pols_info: (Vec<String>, Vec<String>),
    /// Sorted opening points (lex string order matching JS Array.sort()).
    pub opening_points: Vec<i64>,
}

// ---------------------------------------------------------------------------
// Byte buffer -> big-integer string (mirrors JS `ProtoOut.buf2bint`)
// ---------------------------------------------------------------------------

/// Convert a big-endian byte buffer of any length to a decimal string (an empty buffer is 0).
fn buf_to_bigint_string(buf: &[u8]) -> String {
    BigUint::from_bytes_be(buf).to_string()
}

// ---------------------------------------------------------------------------
// Arena-based expression formatting context
// ---------------------------------------------------------------------------

/// Context for converting protobuf expressions into arena-indexed Expressions.
struct FormatCtx<'a> {
    field: &'a FieldCfg,
    air_expressions: &'a [pb::Expression],
    stage_widths: &'a [u32],
    num_challenges: &'a [u32],
    air_values: &'a [pb::AirValue],
    air_group_values: &'a [pb::AirGroupValue],
    custom_commits: &'a [pb::CustomCommit],
    arena: Vec<Expression>,
}

impl<'a> FormatCtx<'a> {
    /// Format a protobuf Operand into an inline Expression (not pushed to arena).
    /// Matches JS behavior where child operands are inline objects.
    ///
    /// Fails on a custom column of a custom commit the air does not have.
    fn format_operand_inline(&mut self, op: &operand::Operand) -> Result<Expression> {
        let formatted = match op {
            operand::Operand::Expression(expr_ref) => {
                let id = expr_ref.idx as usize;
                // Optimization: unwrap add/sub(X, 0) where LHS is not an expression ref
                if let Some(inner_expr) = self.air_expressions.get(id) {
                    if let Some(ref operation) = inner_expr.operation {
                        if let Some(lhs) = zero_rhs_lhs(operation) {
                            return self.format_operand_inline(lhs);
                        }
                    }
                }
                Expression { op: "exp".to_string(), id: Some(id), ..Default::default() }
            }
            operand::Operand::Constant(c) => {
                let value = buf_to_bigint_string(&c.value);
                Expression { op: "number".to_string(), value: Some(value), ..Default::default() }
            }
            operand::Operand::WitnessCol(wc) => {
                let stage_id = wc.col_idx as usize;
                let row_offset = wc.row_offset as i64;
                let stage = wc.stage as usize;
                let id = stage_id
                    + self.stage_widths.iter().take(stage.saturating_sub(1)).map(|w| *w as usize).sum::<usize>();
                let dim = if stage <= 1 { 1 } else { self.field.ext_dim() };
                Expression {
                    op: "cm".to_string(),
                    id: Some(id),
                    stage_id: Some(stage_id),
                    row_offset: Some(row_offset),
                    stage,
                    dim,
                    ..Default::default()
                }
            }
            operand::Operand::CustomCol(cc) => {
                let commit_id = cc.commit_id as usize;
                let Some(custom_commit) = self.custom_commits.get(commit_id) else {
                    return Err(PilInfoError::InvalidPilout(format!(
                        "a custom column of custom commit {commit_id}, and the air has {} custom commits",
                        self.custom_commits.len()
                    )));
                };
                let custom_stage_widths = &custom_commit.stage_widths;
                let stage_id = cc.col_idx as usize;
                let row_offset = cc.row_offset as i64;
                let stage = cc.stage as usize;
                let id = stage_id
                    + custom_stage_widths.iter().take(stage.saturating_sub(1)).map(|w| *w as usize).sum::<usize>();
                let dim = if stage <= 1 { 1 } else { self.field.ext_dim() };
                Expression {
                    op: "custom".to_string(),
                    id: Some(id),
                    stage_id: Some(stage_id),
                    row_offset: Some(row_offset),
                    stage,
                    dim,
                    commit_id: Some(commit_id),
                    ..Default::default()
                }
            }
            operand::Operand::FixedCol(fc) => {
                let id = fc.idx as usize;
                let row_offset = fc.row_offset as i64;
                Expression {
                    op: "const".to_string(),
                    id: Some(id),
                    row_offset: Some(row_offset),
                    stage: 0,
                    dim: 1,
                    ..Default::default()
                }
            }
            operand::Operand::PublicValue(pv) => {
                let id = pv.idx as usize;
                Expression { op: "public".to_string(), id: Some(id), stage: 1, ..Default::default() }
            }
            operand::Operand::AirGroupValue(agv) => {
                let id = agv.idx as usize;
                let stage = self.air_group_values.get(id).map(|v| v.stage as usize).unwrap_or(0);
                let dim = if stage == 1 { 1 } else { self.field.ext_dim() };
                Expression { op: "airgroupvalue".to_string(), id: Some(id), dim, stage, ..Default::default() }
            }
            operand::Operand::AirValue(av) => {
                let id = av.idx as usize;
                let stage = self.air_values.get(id).map(|v| v.stage as usize).unwrap_or(0);
                let dim = if stage == 1 { 1 } else { self.field.ext_dim() };
                Expression { op: "airvalue".to_string(), id: Some(id), stage, dim, ..Default::default() }
            }
            operand::Operand::Challenge(ch) => {
                let stage_id_val = ch.idx as usize;
                let stage = ch.stage as usize;
                let id = stage_id_val
                    + self.num_challenges.iter().take(stage.saturating_sub(1)).map(|c| *c as usize).sum::<usize>();
                Expression {
                    op: "challenge".to_string(),
                    stage,
                    stage_id: Some(stage_id_val),
                    id: Some(id),
                    ..Default::default()
                }
            }
            operand::Operand::ProofValue(pv) => {
                let id = pv.idx as usize;
                let stage = pv.stage as usize;
                let dim = if stage == 1 { 1 } else { self.field.ext_dim() };
                Expression { op: "proofvalue".to_string(), id: Some(id), stage, dim, ..Default::default() }
            }
            operand::Operand::PeriodicCol(pc) => {
                let id = pc.idx as usize;
                let row_offset = pc.row_offset as i64;
                Expression {
                    op: "const".to_string(),
                    id: Some(id),
                    row_offset: Some(row_offset),
                    stage: 0,
                    dim: 1,
                    ..Default::default()
                }
            }
        };
        Ok(formatted)
    }

    /// Format an `Option<&Operand>`, a child of expression `parent`, into an inline ExprChild.
    ///
    /// Fails on a reference to an expression the air does not have.
    fn format_operand_child(&mut self, parent: usize, operand: Option<&pb::Operand>) -> Result<ExprChild> {
        match operand.and_then(|o| o.operand.as_ref()) {
            Some(op) => {
                if let operand::Operand::Expression(expr_ref) = op {
                    let n = self.air_expressions.len();
                    if expr_ref.idx as usize >= n {
                        return Err(PilInfoError::InvalidPilout(format!(
                            "expression {parent} refers to expression {}, and the air has {n}",
                            expr_ref.idx
                        )));
                    }
                }
                Ok(ExprChild::Inline(Box::new(self.format_operand_inline(op)?)))
            }
            None => Ok(ExprChild::Inline(Box::new(Expression {
                op: "number".to_string(),
                value: Some("0".to_string()),
                ..Default::default()
            }))),
        }
    }

    /// Format a single top-level protobuf Expression, the air's expression `i` (add/sub/mul/neg
    /// with children). Children are stored as inline ExprChild values (not pushed to arena).
    fn format_expression_node(&mut self, i: usize, expr: &pb::Expression) -> Result<Expression> {
        let operation = match &expr.operation {
            Some(op) => op,
            None => {
                return Ok(Expression { op: "number".to_string(), value: Some("0".to_string()), ..Default::default() });
            }
        };

        let formatted = match operation {
            expr_mod::Operation::Add(add) => {
                let lhs = self.format_operand_child(i, add.lhs.as_ref())?;
                let rhs = self.format_operand_child(i, add.rhs.as_ref())?;
                Expression { op: "add".to_string(), values: vec![lhs, rhs], ..Default::default() }
            }
            expr_mod::Operation::Sub(sub) => {
                let lhs = self.format_operand_child(i, sub.lhs.as_ref())?;
                let rhs = self.format_operand_child(i, sub.rhs.as_ref())?;
                Expression { op: "sub".to_string(), values: vec![lhs, rhs], ..Default::default() }
            }
            expr_mod::Operation::Mul(mul) => {
                let lhs = self.format_operand_child(i, mul.lhs.as_ref())?;
                let rhs = self.format_operand_child(i, mul.rhs.as_ref())?;
                Expression { op: "mul".to_string(), values: vec![lhs, rhs], ..Default::default() }
            }
            expr_mod::Operation::Neg(neg) => {
                let val = self.format_operand_child(i, neg.value.as_ref())?;
                Expression { op: "neg".to_string(), values: vec![val], ..Default::default() }
            }
        };
        Ok(formatted)
    }
}

/// The LHS of add/sub(X, const(0)) where X is not an expression reference: what such an
/// expression is unwrapped to, inline, instead of an arena index.
fn zero_rhs_lhs(operation: &expr_mod::Operation) -> Option<&operand::Operand> {
    let (lhs_operand, rhs_operand) = match operation {
        expr_mod::Operation::Add(add) => (add.lhs.as_ref()?, add.rhs.as_ref()?),
        expr_mod::Operation::Sub(sub) => (sub.lhs.as_ref()?, sub.rhs.as_ref()?),
        _ => return None,
    };

    let lhs_op = lhs_operand.operand.as_ref()?;
    let rhs_op = rhs_operand.operand.as_ref()?;

    if matches!(lhs_op, operand::Operand::Expression(_)) {
        return None;
    }

    if let operand::Operand::Constant(c) = rhs_op {
        let val = buf_to_bigint_string(&c.value);
        if val == "0" {
            return Some(lhs_op);
        }
    }

    None
}

// ---------------------------------------------------------------------------
// format_expressions: public API
// ---------------------------------------------------------------------------

/// Format all Air-level protobuf expressions into a flat `Vec<Expression>`.
///
/// Top-level expressions occupy indices 0..N-1. Child operands are stored
/// as inline `ExprChild::Inline` values within each expression, matching
/// the JS representation where child nodes are nested objects.
///
/// Fails on a reference to an expression or a custom commit the air does not have, and on
/// expressions that refer to each other in a cycle.
pub fn format_expressions(
    air_expressions: &[pb::Expression],
    stage_widths: &[u32],
    num_challenges: &[u32],
    air_values: &[pb::AirValue],
    air_group_values: &[pb::AirGroupValue],
    custom_commits: &[pb::CustomCommit],
    field: &FieldCfg,
) -> Result<Vec<Expression>> {
    let n = air_expressions.len();

    let mut ctx = FormatCtx {
        field,
        air_expressions,
        stage_widths,
        num_challenges,
        air_values,
        air_group_values,
        custom_commits,
        arena: Vec::with_capacity(n),
    };

    // Reserve the first N slots with placeholders.
    for _ in 0..n {
        ctx.arena.push(Expression { op: "__placeholder__".to_string(), ..Default::default() });
    }

    // Now format each top-level expression. Children get pushed at indices >= N.
    for (i, air_expr) in air_expressions.iter().enumerate() {
        let formatted = ctx.format_expression_node(i, air_expr)?;
        ctx.arena[i] = formatted;
    }

    check_reference_cycles(&ctx.arena, "expression")?;
    Ok(ctx.arena)
}

// ---------------------------------------------------------------------------
// Reference cycles
// ---------------------------------------------------------------------------

/// The expressions `e` refers to, `exp` operands at any depth of its children, in the order of
/// its operands.
fn references(e: &Expression) -> Vec<usize> {
    let mut refs = Vec::new();
    let mut pending: Vec<&ExprChild> = e.values.iter().rev().collect();
    while let Some(child) = pending.pop() {
        match child {
            ExprChild::Id(id) => refs.push(*id),
            ExprChild::Inline(inner) => {
                if inner.op == "exp" {
                    refs.extend(inner.id);
                }
                pending.extend(inner.values.iter().rev());
            }
        }
    }
    refs
}

/// Refuses formatted expressions that refer to each other in a cycle, which a pilout cannot mean:
/// an expression would be defined by itself. Every pass follows the references recursively, and a
/// cycle recursed until the stack overflowed, which aborts the process. `what` names the
/// expressions in the error: `"expression"` or `"global expression"`.
///
/// A depth-first search with an explicit stack, so that a long chain of references cannot
/// overflow the stack here either.
fn check_reference_cycles(expressions: &[Expression], what: &str) -> Result<()> {
    #[derive(Clone, Copy, PartialEq, Eq)]
    enum Visit {
        New,
        OnPath,
        Done,
    }
    let refs: Vec<Vec<usize>> = expressions.iter().map(references).collect();
    let mut visit = vec![Visit::New; expressions.len()];
    for root in 0..expressions.len() {
        if visit[root] != Visit::New {
            continue;
        }
        // The path from `root`, each expression with the index of the next reference to follow.
        let mut path: Vec<(usize, usize)> = vec![(root, 0)];
        visit[root] = Visit::OnPath;
        while let Some(top) = path.last_mut() {
            let node = top.0;
            let Some(&next) = refs[node].get(top.1) else {
                visit[node] = Visit::Done;
                path.pop();
                continue;
            };
            top.1 += 1;
            // An index beyond the expressions is refused where it is formatted; here it is no
            // cycle.
            match visit.get(next) {
                Some(Visit::New) => {
                    visit[next] = Visit::OnPath;
                    path.push((next, 0));
                }
                Some(Visit::OnPath) => {
                    let start = path.iter().position(|&(e, _)| e == next).unwrap_or(0);
                    let cycle: Vec<String> =
                        path[start..].iter().map(|&(e, _)| e.to_string()).chain([next.to_string()]).collect();
                    return Err(PilInfoError::InvalidPilout(format!(
                        "{what} {next} refers to itself, through the references {}",
                        cycle.join(" → ")
                    )));
                }
                Some(Visit::Done) | None => {}
            }
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// format_global_expressions
// ---------------------------------------------------------------------------

/// Context for converting global protobuf expressions into Expressions.
struct GlobalFormatCtx<'a> {
    field: &'a FieldCfg,
    global_expressions: &'a [pb::GlobalExpression],
    num_challenges: &'a [u32],
    air_groups: &'a [pb::AirGroup],
    arena: Vec<Expression>,
}

impl<'a> GlobalFormatCtx<'a> {
    /// Format a GlobalOperand into an inline Expression.
    fn format_global_operand_inline(&mut self, op: &global_operand::Operand) -> Expression {
        match op {
            global_operand::Operand::Expression(expr_ref) => {
                let id = expr_ref.idx as usize;
                // Optimization: unwrap add/sub(X, 0) where LHS is not an expression ref
                if let Some(inner_expr) = self.global_expressions.get(id) {
                    if let Some(ref operation) = inner_expr.operation {
                        if let Some(unwrapped) = self.try_unwrap_zero_rhs_global(operation) {
                            return unwrapped;
                        }
                    }
                }
                Expression { op: "exp".to_string(), id: Some(id), ..Default::default() }
            }
            global_operand::Operand::Constant(c) => {
                let value = buf_to_bigint_string(&c.value);
                Expression { op: "number".to_string(), value: Some(value), ..Default::default() }
            }
            global_operand::Operand::PublicValue(pv) => {
                let id = pv.idx as usize;
                Expression { op: "public".to_string(), id: Some(id), stage: 1, ..Default::default() }
            }
            global_operand::Operand::AirGroupValue(agv) => {
                let id = agv.idx as usize;
                let air_group_id = agv.air_group_id as usize;
                // In global mode, look up stage from the airgroup's airGroupValues
                let stage = self
                    .air_groups
                    .get(air_group_id)
                    .and_then(|ag| ag.air_group_values.get(id))
                    .map(|v| v.stage as usize)
                    .unwrap_or(0);
                let dim = if stage == 1 { 1 } else { self.field.ext_dim() };
                Expression {
                    op: "airgroupvalue".to_string(),
                    id: Some(id),
                    airgroup_id: Some(air_group_id),
                    dim,
                    stage,
                    ..Default::default()
                }
            }
            global_operand::Operand::Challenge(ch) => {
                let stage_id_val = ch.idx as usize;
                let stage = ch.stage as usize;
                let id = stage_id_val
                    + self.num_challenges.iter().take(stage.saturating_sub(1)).map(|c| *c as usize).sum::<usize>();
                Expression {
                    op: "challenge".to_string(),
                    stage,
                    stage_id: Some(stage_id_val),
                    id: Some(id),
                    ..Default::default()
                }
            }
            global_operand::Operand::ProofValue(pv) => {
                let id = pv.idx as usize;
                let stage = pv.stage as usize;
                let dim = if stage == 1 { 1 } else { self.field.ext_dim() };
                Expression { op: "proofvalue".to_string(), id: Some(id), stage, dim, ..Default::default() }
            }
            global_operand::Operand::PublicTableAggregatedValue(ptav) => {
                let id = ptav.idx as usize;
                Expression { op: "public".to_string(), id: Some(id), stage: 1, ..Default::default() }
            }
            global_operand::Operand::PublicTableColumn(ptc) => {
                let id = ptc.idx as usize;
                Expression { op: "public".to_string(), id: Some(id), stage: 1, ..Default::default() }
            }
        }
    }

    /// Format an `Option<&GlobalOperand>`, a child of global expression `parent`, into an inline
    /// ExprChild.
    ///
    /// Fails on a reference to a global expression the pilout does not have.
    fn format_global_operand_child(&mut self, parent: usize, operand: Option<&pb::GlobalOperand>) -> Result<ExprChild> {
        match operand.and_then(|o| o.operand.as_ref()) {
            Some(op) => {
                if let global_operand::Operand::Expression(expr_ref) = op {
                    let n = self.global_expressions.len();
                    if expr_ref.idx as usize >= n {
                        return Err(PilInfoError::InvalidPilout(format!(
                            "global expression {parent} refers to global expression {}, and the pilout has {n}",
                            expr_ref.idx
                        )));
                    }
                }
                Ok(ExprChild::Inline(Box::new(self.format_global_operand_inline(op))))
            }
            None => Ok(ExprChild::Inline(Box::new(Expression {
                op: "number".to_string(),
                value: Some("0".to_string()),
                ..Default::default()
            }))),
        }
    }

    /// Try to unwrap add/sub(X, const(0)) for global expressions.
    fn try_unwrap_zero_rhs_global(&mut self, operation: &gexpr_mod::Operation) -> Option<Expression> {
        let (lhs_operand, rhs_operand) = match operation {
            gexpr_mod::Operation::Add(add) => (add.lhs.as_ref()?, add.rhs.as_ref()?),
            gexpr_mod::Operation::Sub(sub) => (sub.lhs.as_ref()?, sub.rhs.as_ref()?),
            _ => return None,
        };

        let lhs_op = lhs_operand.operand.as_ref()?;
        let rhs_op = rhs_operand.operand.as_ref()?;

        if matches!(lhs_op, global_operand::Operand::Expression(_)) {
            return None;
        }

        if let global_operand::Operand::Constant(c) = rhs_op {
            let val = buf_to_bigint_string(&c.value);
            if val == "0" {
                return Some(self.format_global_operand_inline(lhs_op));
            }
        }

        None
    }

    /// Format a single top-level GlobalExpression, the pilout's global expression `i`.
    fn format_global_expression_node(&mut self, i: usize, expr: &pb::GlobalExpression) -> Result<Expression> {
        let operation = match &expr.operation {
            Some(op) => op,
            None => {
                return Ok(Expression { op: "number".to_string(), value: Some("0".to_string()), ..Default::default() });
            }
        };

        let formatted = match operation {
            gexpr_mod::Operation::Add(add) => {
                let lhs = self.format_global_operand_child(i, add.lhs.as_ref())?;
                let rhs = self.format_global_operand_child(i, add.rhs.as_ref())?;
                Expression { op: "add".to_string(), values: vec![lhs, rhs], ..Default::default() }
            }
            gexpr_mod::Operation::Sub(sub) => {
                let lhs = self.format_global_operand_child(i, sub.lhs.as_ref())?;
                let rhs = self.format_global_operand_child(i, sub.rhs.as_ref())?;
                Expression { op: "sub".to_string(), values: vec![lhs, rhs], ..Default::default() }
            }
            gexpr_mod::Operation::Mul(mul) => {
                let lhs = self.format_global_operand_child(i, mul.lhs.as_ref())?;
                let rhs = self.format_global_operand_child(i, mul.rhs.as_ref())?;
                Expression { op: "mul".to_string(), values: vec![lhs, rhs], ..Default::default() }
            }
            gexpr_mod::Operation::Neg(neg) => {
                let val = self.format_global_operand_child(i, neg.value.as_ref())?;
                Expression { op: "neg".to_string(), values: vec![val], ..Default::default() }
            }
        };
        Ok(formatted)
    }
}

/// Format all pilout-level global expressions into a flat `Vec<Expression>`.
///
/// Mirrors JS `formatExpressions(pilout, true)` for global mode. Fails on a reference to a global
/// expression the pilout does not have, and on global expressions that refer to each other in a
/// cycle.
pub fn format_global_expressions(
    global_expressions: &[pb::GlobalExpression],
    num_challenges: &[u32],
    air_groups: &[pb::AirGroup],
    field: &FieldCfg,
) -> Result<Vec<Expression>> {
    let n = global_expressions.len();

    let mut ctx =
        GlobalFormatCtx { field, global_expressions, num_challenges, air_groups, arena: Vec::with_capacity(n) };

    // Reserve the first N slots with placeholders.
    for _ in 0..n {
        ctx.arena.push(Expression { op: "__placeholder__".to_string(), ..Default::default() });
    }

    // Now format each top-level expression.
    for (i, gexpr) in global_expressions.iter().enumerate() {
        let formatted = ctx.format_global_expression_node(i, gexpr)?;
        ctx.arena[i] = formatted;
    }

    check_reference_cycles(&ctx.arena, "global expression")?;
    Ok(ctx.arena)
}

/// Format global constraints from pilout.
///
/// Each global constraint has an expression index and a debug line.
/// Boundary is always "finalProof" for global constraints.
pub fn format_global_constraints(constraints: &[pb::GlobalConstraint]) -> Vec<ConstraintInfo> {
    constraints
        .iter()
        .filter_map(|c| {
            let expr_idx = c.expression_idx.as_ref()?.idx as usize;
            Some(ConstraintInfo {
                boundary: "finalProof".to_string(),
                e: expr_idx,
                line: c.debug_line.clone(),
                offset_min: None,
                offset_max: None,
                stage: None,
                im_pol: false,
            })
        })
        .collect()
}

/// Refuses a constraint of `constraints` whose expression is not one of the `n_expressions` of
/// its air, or of the pilout for the global constraints; `whose` says which in the error.
pub fn check_constraint_expressions(constraints: &[ConstraintInfo], n_expressions: usize, whose: &str) -> Result<()> {
    match constraints.iter().enumerate().find(|(_, c)| c.e >= n_expressions) {
        Some((i, c)) => Err(PilInfoError::InvalidPilout(format!(
            "constraint {i} of {whose} is expression {}, and there are {n_expressions}",
            c.e
        ))),
        None => Ok(()),
    }
}

/// Format global symbols (symbols not tied to a specific air).
///
/// In global mode, filters out AIR_VALUE, CUSTOM_COL, FIXED_COL, WITNESS_COL.
pub fn format_global_symbols(all_symbols: &[pb::Symbol], _num_challenges: &[u32], field: &FieldCfg) -> Vec<SymbolInfo> {
    let mut result = Vec::new();

    for s in all_symbols {
        let stype = s.r#type;
        // Skip IM_COL (type 0) and air-specific types in global mode
        if stype == SymbolType::ImCol as i32
            || stype == SymbolType::AirValue as i32
            || stype == SymbolType::CustomCol as i32
            || stype == SymbolType::FixedCol as i32
            || stype == SymbolType::WitnessCol as i32
        {
            continue;
        }

        if stype == SymbolType::ProofValue as i32 {
            let stage = s.stage.unwrap_or(1) as usize;
            let dim = if stage == 1 { 1 } else { field.ext_dim() };
            if s.dim == 0 {
                result.push(SymbolInfo {
                    name: s.name.clone(),
                    sym_type: "proofvalue".to_string(),
                    stage: Some(stage),
                    dim,
                    id: Some(s.id as usize),
                    pol_id: None,
                    stage_id: None,
                    air_id: None,
                    airgroup_id: None,
                    commit_id: None,
                    lengths: None,
                    idx: None,
                    stage_pos: None,
                    im_pol: false,
                    exp_id: None,
                });
            } else {
                generate_multi_array_symbols(&mut result, &[], s, "proofvalue", stage, dim, s.id as usize, 0);
            }
        } else if stype == SymbolType::Challenge as i32 {
            let stage = s.stage.unwrap_or(1) as usize;
            // Challenge ID: count preceding challenge symbols
            let id = all_symbols
                .iter()
                .filter(|si| {
                    si.r#type == SymbolType::Challenge as i32
                        && (si.stage.unwrap_or(1) < s.stage.unwrap_or(1) || (si.stage == s.stage && si.id < s.id))
                })
                .count();
            result.push(SymbolInfo {
                name: s.name.clone(),
                sym_type: "challenge".to_string(),
                stage: Some(stage),
                dim: field.ext_dim(),
                id: Some(id),
                pol_id: None,
                stage_id: Some(s.id as usize),
                air_id: None,
                airgroup_id: None,
                commit_id: None,
                lengths: None,
                idx: None,
                stage_pos: None,
                im_pol: false,
                exp_id: None,
            });
        } else if stype == SymbolType::PublicValue as i32 {
            if s.dim == 0 || s.lengths.is_empty() {
                result.push(SymbolInfo {
                    name: s.name.clone(),
                    sym_type: "public".to_string(),
                    stage: Some(1),
                    dim: 1,
                    id: Some(s.id as usize),
                    pol_id: None,
                    stage_id: None,
                    air_id: None,
                    airgroup_id: None,
                    commit_id: None,
                    lengths: None,
                    idx: None,
                    stage_pos: None,
                    im_pol: false,
                    exp_id: None,
                });
            } else {
                generate_multi_array_symbols(&mut result, &[], s, "public", 1, 1, s.id as usize, 0);
            }
        } else if stype == SymbolType::AirGroupValue as i32 {
            // In global mode, stage is undefined (not set from airGroupValues)
            let dim = field.ext_dim();
            if s.dim == 0 {
                result.push(SymbolInfo {
                    name: s.name.clone(),
                    sym_type: "airgroupvalue".to_string(),
                    stage: None,
                    dim,
                    id: Some(s.id as usize),
                    pol_id: None,
                    stage_id: None,
                    air_id: None,
                    airgroup_id: s.air_group_id.map(|v| v as usize),
                    commit_id: None,
                    lengths: None,
                    idx: None,
                    stage_pos: None,
                    im_pol: false,
                    exp_id: None,
                });
            } else {
                generate_multi_array_symbols(&mut result, &[], s, "airgroupvalue", 0, dim, s.id as usize, 0);
            }
        } else if stype == SymbolType::PeriodicCol as i32 {
            // Skip periodic cols in global mode
            continue;
        } else if stype == SymbolType::PublicTable as i32 {
            // Skip public tables in global mode
            continue;
        }
    }

    result
}

/// Format global hints (hints without air_id and airgroup_id).
///
/// Uses the same hint formatting as air-level hints but processes only
/// global hints from the pilout.
///
/// Fails on a hint field without a value.
pub fn format_global_hints(
    pilout: &pb::PilOut,
    expressions: &mut [Expression],
    field: &FieldCfg,
) -> Result<Vec<HintInfo>> {
    // Filter hints that are global (no airGroupId and no airId)
    let global_hints: Vec<&pb::Hint> =
        pilout.hints.iter().filter(|h| h.air_group_id.is_none() && h.air_id.is_none()).collect();

    let mut hints = Vec::new();

    for raw_hint in &global_hints {
        let hint_name = raw_hint.name.clone();

        // JS: rawHints[i].hintFields[0].hintFieldArray.hintFields
        let inner_fields = if let Some(first_hf) = raw_hint.hint_fields.first() {
            if let Some(hint_field::Value::HintFieldArray(arr)) = &first_hf.value {
                &arr.hint_fields[..]
            } else {
                &raw_hint.hint_fields[..]
            }
        } else {
            continue;
        };

        let mut fields = Vec::new();
        for hint_field in inner_fields {
            let name = hint_field.name.clone().unwrap_or_default();
            let (values, lengths) = process_global_hint_field(&hint_name, hint_field, pilout, expressions, field)?;
            let entry = if lengths.is_none() {
                HintFieldEntry { name, values: vec![values], lengths: None }
            } else {
                HintFieldEntry {
                    name,
                    values: match values {
                        HintFieldValue::Array(arr) => *arr,
                        single => vec![single],
                    },
                    lengths,
                }
            };
            fields.push(entry);
        }
        hints.push(HintInfo { name: hint_name, fields });
    }

    Ok(hints)
}

/// Recursively process a global hint field of hint `hint`.
///
/// Global hint fields use regular Operand types (not GlobalOperand),
/// but with global-mode resolution for airGroupValue.
fn process_global_hint_field(
    hint: &str,
    hint_field: &pb::HintField,
    pilout: &pb::PilOut,
    expressions: &mut [Expression],
    field: &FieldCfg,
) -> Result<(HintFieldValue, Option<Vec<usize>>)> {
    let processed = match &hint_field.value {
        Some(hint_field::Value::HintFieldArray(arr)) => {
            let fields = &arr.hint_fields;
            let mut result_fields = Vec::new();
            let mut lengths: Vec<usize> = Vec::new();

            for sub_field in fields {
                let (values, sub_lengths) = process_global_hint_field(hint, sub_field, pilout, expressions, field)?;
                result_fields.push(values);

                if lengths.is_empty() {
                    lengths.push(fields.len());
                }

                if let Some(sub) = sub_lengths {
                    for (k, &sub_len) in sub.iter().enumerate() {
                        if k + 1 >= lengths.len() {
                            lengths.resize(k + 2, 0);
                        }
                        if lengths[k + 1] == 0 {
                            lengths[k + 1] = sub_len;
                        }
                    }
                }
            }

            (HintFieldValue::Array(Box::new(result_fields)), Some(lengths))
        }
        Some(hint_field::Value::Operand(op_msg)) => {
            if let Some(ref op) = op_msg.operand {
                let value = format_global_hint_operand(op, pilout, field);
                // If the value is an "exp" reference, mark keep=true
                if value.op == "exp" {
                    if let Some(id) = value.id {
                        if id < expressions.len() {
                            expressions[id].keep = Some(true);
                        }
                    }
                }
                (HintFieldValue::Single(Box::new(value)), None)
            } else {
                (
                    HintFieldValue::Single(Box::new(Expression {
                        op: "number".to_string(),
                        value: Some("0".to_string()),
                        ..Default::default()
                    })),
                    None,
                )
            }
        }
        Some(hint_field::Value::StringValue(s)) => (
            HintFieldValue::Single(Box::new(Expression {
                op: "string".to_string(),
                value: Some(s.clone()),
                ..Default::default()
            })),
            None,
        ),
        None => return Err(PilInfoError::HintFieldWithoutValue { hint: hint.to_string() }),
    };
    Ok(processed)
}

/// Format a regular Operand in global hint context.
///
/// Global hints use the regular Operand type but some fields
/// (like airGroupValue) need global-mode resolution.
fn format_global_hint_operand(op: &operand::Operand, pilout: &pb::PilOut, field: &FieldCfg) -> Expression {
    match op {
        operand::Operand::Expression(expr_ref) => {
            let id = expr_ref.idx as usize;
            // Optimization: unwrap add/sub(X, 0) where LHS is not an expression ref
            // Mirrors JS formatExpression behavior for expression references
            if let Some(gexpr) = pilout.expressions.get(id) {
                if let Some(ref operation) = gexpr.operation {
                    if let Some(unwrapped) = try_unwrap_global_hint_zero_rhs(operation, pilout, field) {
                        return unwrapped;
                    }
                }
            }
            Expression { op: "exp".to_string(), id: Some(id), ..Default::default() }
        }
        operand::Operand::Constant(c) => {
            let value = buf_to_bigint_string(&c.value);
            Expression { op: "number".to_string(), value: Some(value), ..Default::default() }
        }
        operand::Operand::PublicValue(pv) => {
            let id = pv.idx as usize;
            Expression { op: "public".to_string(), id: Some(id), stage: 1, ..Default::default() }
        }
        operand::Operand::AirGroupValue(agv) => {
            let id = agv.idx as usize;
            // In global mode for hints, airGroupValue doesn't have airGroupId
            // in the Operand type (only GlobalOperand has it).
            // The stage comes from the expression context.
            Expression { op: "airgroupvalue".to_string(), id: Some(id), dim: field.ext_dim(), ..Default::default() }
        }
        operand::Operand::Challenge(ch) => {
            let stage_id_val = ch.idx as usize;
            let stage = ch.stage as usize;
            let id = stage_id_val
                + pilout.num_challenges.iter().take(stage.saturating_sub(1)).map(|c| *c as usize).sum::<usize>();
            Expression {
                op: "challenge".to_string(),
                stage,
                stage_id: Some(stage_id_val),
                id: Some(id),
                ..Default::default()
            }
        }
        operand::Operand::ProofValue(pv) => {
            let id = pv.idx as usize;
            let stage = pv.stage as usize;
            let dim = if stage == 1 { 1 } else { field.ext_dim() };
            Expression { op: "proofvalue".to_string(), id: Some(id), stage, dim, ..Default::default() }
        }
        operand::Operand::AirValue(av) => {
            let id = av.idx as usize;
            Expression { op: "airvalue".to_string(), id: Some(id), ..Default::default() }
        }
        operand::Operand::WitnessCol(_)
        | operand::Operand::FixedCol(_)
        | operand::Operand::PeriodicCol(_)
        | operand::Operand::CustomCol(_) => {
            // These should not appear in global hints
            Expression { op: "number".to_string(), value: Some("0".to_string()), ..Default::default() }
        }
    }
}

/// Try to unwrap add/sub(X, const(0)) for global expression references in hints.
///
/// When a hint field operand is an expression reference, and the referenced
/// global expression is add/sub(LHS, 0) where LHS is not an expression ref,
/// return the LHS operand directly instead of the expression reference.
fn try_unwrap_global_hint_zero_rhs(
    operation: &gexpr_mod::Operation,
    pilout: &pb::PilOut,
    field: &FieldCfg,
) -> Option<Expression> {
    let (lhs_operand, rhs_operand) = match operation {
        gexpr_mod::Operation::Add(add) => (add.lhs.as_ref()?, add.rhs.as_ref()?),
        gexpr_mod::Operation::Sub(sub) => (sub.lhs.as_ref()?, sub.rhs.as_ref()?),
        _ => return None,
    };

    let lhs_op = lhs_operand.operand.as_ref()?;
    let rhs_op = rhs_operand.operand.as_ref()?;

    // Don't unwrap if LHS is itself an expression reference
    if matches!(lhs_op, global_operand::Operand::Expression(_)) {
        return None;
    }

    if let global_operand::Operand::Constant(c) = rhs_op {
        let val = buf_to_bigint_string(&c.value);
        if val == "0" {
            // Convert the GlobalOperand LHS to an Expression
            return Some(convert_global_operand_to_expression(lhs_op, pilout, field));
        }
    }

    None
}

/// Convert a GlobalOperand to an Expression for hint field processing.
fn convert_global_operand_to_expression(
    op: &global_operand::Operand,
    pilout: &pb::PilOut,
    field: &FieldCfg,
) -> Expression {
    match op {
        global_operand::Operand::Expression(expr_ref) => {
            let id = expr_ref.idx as usize;
            // Recursively try to unwrap
            if let Some(gexpr) = pilout.expressions.get(id) {
                if let Some(ref operation) = gexpr.operation {
                    if let Some(unwrapped) = try_unwrap_global_hint_zero_rhs(operation, pilout, field) {
                        return unwrapped;
                    }
                }
            }
            Expression { op: "exp".to_string(), id: Some(id), ..Default::default() }
        }
        global_operand::Operand::Constant(c) => {
            let value = buf_to_bigint_string(&c.value);
            Expression { op: "number".to_string(), value: Some(value), ..Default::default() }
        }
        global_operand::Operand::PublicValue(pv) => {
            let id = pv.idx as usize;
            Expression { op: "public".to_string(), id: Some(id), stage: 1, ..Default::default() }
        }
        global_operand::Operand::AirGroupValue(agv) => {
            let id = agv.idx as usize;
            let air_group_id = agv.air_group_id as usize;
            let stage = pilout
                .air_groups
                .get(air_group_id)
                .and_then(|ag| ag.air_group_values.get(id))
                .map(|v| v.stage as usize)
                .unwrap_or(0);
            let dim = if stage == 1 { 1 } else { field.ext_dim() };
            Expression {
                op: "airgroupvalue".to_string(),
                id: Some(id),
                airgroup_id: Some(air_group_id),
                dim,
                stage,
                ..Default::default()
            }
        }
        global_operand::Operand::Challenge(ch) => {
            let stage_id_val = ch.idx as usize;
            let stage = ch.stage as usize;
            let id = stage_id_val
                + pilout.num_challenges.iter().take(stage.saturating_sub(1)).map(|c| *c as usize).sum::<usize>();
            Expression {
                op: "challenge".to_string(),
                stage,
                stage_id: Some(stage_id_val),
                id: Some(id),
                ..Default::default()
            }
        }
        global_operand::Operand::ProofValue(pv) => {
            let id = pv.idx as usize;
            let stage = pv.stage as usize;
            let dim = if stage == 1 { 1 } else { field.ext_dim() };
            Expression { op: "proofvalue".to_string(), id: Some(id), stage, dim, ..Default::default() }
        }
        global_operand::Operand::PublicTableAggregatedValue(ptav) => {
            let id = ptav.idx as usize;
            Expression { op: "public".to_string(), id: Some(id), stage: 1, ..Default::default() }
        }
        global_operand::Operand::PublicTableColumn(ptc) => {
            let id = ptc.idx as usize;
            Expression { op: "public".to_string(), id: Some(id), stage: 1, ..Default::default() }
        }
    }
}

// ---------------------------------------------------------------------------
// format_constraints
// ---------------------------------------------------------------------------

/// Format constraints from protobuf, mirroring JS `formatConstraints`.
pub fn format_constraints(constraints: &[pb::Constraint]) -> Vec<ConstraintInfo> {
    constraints
        .iter()
        .filter_map(|c| {
            let inner = c.constraint.as_ref()?;
            match inner {
                constraint::Constraint::FirstRow(fr) => Some(ConstraintInfo {
                    boundary: "firstRow".to_string(),
                    e: fr.expression_idx.as_ref().map(|e| e.idx as usize).unwrap_or(0),
                    line: fr.debug_line.clone(),
                    offset_min: None,
                    offset_max: None,
                    stage: None,
                    im_pol: false,
                }),
                constraint::Constraint::LastRow(lr) => Some(ConstraintInfo {
                    boundary: "lastRow".to_string(),
                    e: lr.expression_idx.as_ref().map(|e| e.idx as usize).unwrap_or(0),
                    line: lr.debug_line.clone(),
                    offset_min: None,
                    offset_max: None,
                    stage: None,
                    im_pol: false,
                }),
                constraint::Constraint::EveryRow(er) => Some(ConstraintInfo {
                    boundary: "everyRow".to_string(),
                    e: er.expression_idx.as_ref().map(|e| e.idx as usize).unwrap_or(0),
                    line: er.debug_line.clone(),
                    offset_min: None,
                    offset_max: None,
                    stage: None,
                    im_pol: false,
                }),
                constraint::Constraint::EveryFrame(ef) => Some(ConstraintInfo {
                    boundary: "everyFrame".to_string(),
                    e: ef.expression_idx.as_ref().map(|e| e.idx as usize).unwrap_or(0),
                    line: ef.debug_line.clone(),
                    offset_min: Some(ef.offset_min),
                    offset_max: Some(ef.offset_max),
                    stage: None,
                    im_pol: false,
                }),
            }
        })
        .collect()
}

// ---------------------------------------------------------------------------
// format_symbols
// ---------------------------------------------------------------------------

/// Format symbols from pilout, mirroring JS `formatSymbols`.
///
/// Fails on a custom column of a stage other than 0.
pub fn format_symbols(
    all_symbols: &[pb::Symbol],
    _num_challenges: &[u32],
    air_group_values: &[pb::AirGroupValue],
    air_values: &[pb::AirValue],
    field: &FieldCfg,
) -> Result<Vec<SymbolInfo>> {
    let mut result = Vec::new();

    for s in all_symbols {
        let stype = s.r#type;
        // Skip IM_COL (type 0)
        if stype == SymbolType::ImCol as i32 {
            continue;
        }

        if stype == SymbolType::FixedCol as i32
            || stype == SymbolType::WitnessCol as i32
            || stype == SymbolType::CustomCol as i32
        {
            let stage = s.stage.unwrap_or(0) as usize;
            if stype == SymbolType::CustomCol as i32 && stage != 0 {
                return Err(PilInfoError::CustomColumnStage { name: s.name.clone(), stage });
            }

            let type_str = if stype == SymbolType::FixedCol as i32 {
                "fixed"
            } else if stype == SymbolType::CustomCol as i32 {
                "custom"
            } else {
                "witness"
            };

            let dim = if stage <= 1 { 1 } else { field.ext_dim() };
            let pol_id = compute_pol_id(all_symbols, s);

            if s.dim == 0 {
                let mut sym = SymbolInfo {
                    name: s.name.clone(),
                    sym_type: type_str.to_string(),
                    stage: Some(stage),
                    dim,
                    pol_id: Some(pol_id),
                    stage_id: Some(s.id as usize),
                    air_id: s.air_id.map(|v| v as usize),
                    airgroup_id: s.air_group_id.map(|v| v as usize),
                    id: None,
                    commit_id: None,
                    lengths: None,
                    idx: None,
                    stage_pos: None,
                    im_pol: false,
                    exp_id: None,
                };
                if stype == SymbolType::CustomCol as i32 {
                    sym.commit_id = s.commit_id.map(|v| v as usize);
                }
                result.push(sym);
            } else {
                generate_multi_array_symbols(&mut result, &[], s, type_str, stage, dim, pol_id, 0);
            }
        } else if stype == SymbolType::ProofValue as i32 {
            let stage = s.stage.unwrap_or(1) as usize;
            let dim = if stage == 1 { 1 } else { field.ext_dim() };

            if s.dim == 0 {
                result.push(SymbolInfo {
                    name: s.name.clone(),
                    sym_type: "proofvalue".to_string(),
                    stage: Some(stage),
                    dim,
                    id: Some(s.id as usize),
                    pol_id: None,
                    stage_id: None,
                    air_id: None,
                    airgroup_id: None,
                    commit_id: None,
                    lengths: None,
                    idx: None,
                    stage_pos: None,
                    im_pol: false,
                    exp_id: None,
                });
            } else {
                generate_multi_array_symbols(&mut result, &[], s, "proofvalue", stage, dim, s.id as usize, 0);
            }
        } else if stype == SymbolType::Challenge as i32 {
            let stage = s.stage.unwrap_or(1) as usize;
            let id = all_symbols
                .iter()
                .filter(|si| {
                    si.r#type == SymbolType::Challenge as i32 && {
                        let si_stage = si.stage.unwrap_or(0) as usize;
                        si_stage < stage || (si_stage == stage && si.id < s.id)
                    }
                })
                .count();

            result.push(SymbolInfo {
                name: s.name.clone(),
                sym_type: "challenge".to_string(),
                stage: Some(stage),
                dim: field.ext_dim(),
                id: Some(id),
                stage_id: Some(s.id as usize),
                pol_id: None,
                air_id: None,
                airgroup_id: None,
                commit_id: None,
                lengths: None,
                idx: None,
                stage_pos: None,
                im_pol: false,
                exp_id: None,
            });
        } else if stype == SymbolType::PublicValue as i32 {
            if s.dim == 0 {
                result.push(SymbolInfo {
                    name: s.name.clone(),
                    sym_type: "public".to_string(),
                    stage: Some(1),
                    dim: 1,
                    id: Some(s.id as usize),
                    pol_id: None,
                    stage_id: None,
                    air_id: None,
                    airgroup_id: None,
                    commit_id: None,
                    lengths: None,
                    idx: None,
                    stage_pos: None,
                    im_pol: false,
                    exp_id: None,
                });
            } else {
                generate_multi_array_symbols(&mut result, &[], s, "public", 1, 1, s.id as usize, 0);
            }
        } else if stype == SymbolType::AirGroupValue as i32 {
            let stage = air_group_values.get(s.id as usize).map(|v| v.stage as usize);

            if s.dim == 0 {
                let mut sym = SymbolInfo {
                    name: s.name.clone(),
                    sym_type: "airgroupvalue".to_string(),
                    stage,
                    dim: field.ext_dim(),
                    id: Some(s.id as usize),
                    airgroup_id: s.air_group_id.map(|v| v as usize),
                    pol_id: None,
                    stage_id: None,
                    air_id: None,
                    commit_id: None,
                    lengths: None,
                    idx: None,
                    stage_pos: None,
                    im_pol: false,
                    exp_id: None,
                };
                if stage.is_none() || stage == Some(0) {
                    sym.stage = None;
                }
                result.push(sym);
            } else {
                generate_multi_array_symbols(
                    &mut result,
                    &[],
                    s,
                    "airgroupvalue",
                    stage.unwrap_or(0),
                    field.ext_dim(),
                    s.id as usize,
                    0,
                );
            }
        } else if stype == SymbolType::AirValue as i32 {
            let stage = air_values.get(s.id as usize).map(|v| v.stage as usize).unwrap_or(0);
            let dim = if stage != 1 { field.ext_dim() } else { 1 };

            if s.dim == 0 {
                result.push(SymbolInfo {
                    name: s.name.clone(),
                    sym_type: "airvalue".to_string(),
                    stage: Some(stage),
                    dim,
                    id: Some(s.id as usize),
                    airgroup_id: s.air_group_id.map(|v| v as usize),
                    pol_id: None,
                    stage_id: None,
                    air_id: None,
                    commit_id: None,
                    lengths: None,
                    idx: None,
                    stage_pos: None,
                    im_pol: false,
                    exp_id: None,
                });
            } else {
                generate_multi_array_symbols(&mut result, &[], s, "airvalue", stage, dim, s.id as usize, 0);
            }
        }
        // Other types (PeriodicCol, PublicTable) are skipped
    }

    Ok(result)
}

/// Compute the polId for a fixed/witness/custom column symbol.
fn compute_pol_id(all_symbols: &[pb::Symbol], s: &pb::Symbol) -> usize {
    let mut pol_id: usize = 0;
    for si in all_symbols {
        if si.r#type != s.r#type || si.air_id != s.air_id || si.air_group_id != s.air_group_id {
            continue;
        }
        let si_stage = si.stage.unwrap_or(0);
        let s_stage = s.stage.unwrap_or(0);
        if !(si_stage < s_stage || (si_stage == s_stage && si.id < s.id)) {
            continue;
        }
        if s.r#type == SymbolType::CustomCol as i32 && s.commit_id != si.commit_id {
            continue;
        }
        if si.dim == 0 {
            pol_id += 1;
        } else {
            pol_id += si.lengths.iter().map(|l| *l as usize).product::<usize>();
        }
    }
    pol_id
}

/// Recursively generate symbols for multi-dimensional arrays.
#[allow(clippy::too_many_arguments)]
fn generate_multi_array_symbols(
    symbols: &mut Vec<SymbolInfo>,
    indexes: &[usize],
    sym: &pb::Symbol,
    type_str: &str,
    stage: usize,
    dim: usize,
    pol_id: usize,
    shift: usize,
) -> usize {
    if indexes.len() == sym.lengths.len() {
        let mut symbol = SymbolInfo {
            name: sym.name.clone(),
            lengths: Some(indexes.to_vec()),
            idx: Some(shift),
            sym_type: type_str.to_string(),
            pol_id: Some(pol_id + shift),
            id: Some(pol_id + shift),
            stage_id: Some(sym.id as usize + shift),
            stage: Some(stage),
            dim,
            air_id: sym.air_id.map(|v| v as usize),
            airgroup_id: sym.air_group_id.map(|v| v as usize),
            commit_id: None,
            stage_pos: None,
            im_pol: false,
            exp_id: None,
        };
        if sym.commit_id.is_some() {
            symbol.commit_id = sym.commit_id.map(|v| v as usize);
        }
        symbols.push(symbol);
        return shift + 1;
    }

    let len = sym.lengths[indexes.len()] as usize;
    let mut current_shift = shift;
    for i in 0..len {
        let mut new_indexes = indexes.to_vec();
        new_indexes.push(i);
        current_shift =
            generate_multi_array_symbols(symbols, &new_indexes, sym, type_str, stage, dim, pol_id, current_shift);
    }
    current_shift
}

// ---------------------------------------------------------------------------
// format_hints
// ---------------------------------------------------------------------------

/// Format hints from protobuf, mirroring JS `formatHints`.
///
/// Fails on a hint field without a value, or on an operand of a custom commit the air does not
/// have.
#[allow(clippy::too_many_arguments)]
pub fn format_hints(
    raw_hints: &[pb::Hint],
    air_expressions: &[pb::Expression],
    stage_widths: &[u32],
    num_challenges: &[u32],
    air_values: &[pb::AirValue],
    air_group_values: &[pb::AirGroupValue],
    custom_commits: &[pb::CustomCommit],
    expressions: &mut [Expression],
    field: &FieldCfg,
) -> Result<Vec<HintInfo>> {
    let mut hints = Vec::new();

    for raw_hint in raw_hints {
        let hint_name = raw_hint.name.clone();

        // JS: rawHints[i].hintFields[0].hintFieldArray.hintFields
        let inner_fields = if let Some(first_hf) = raw_hint.hint_fields.first() {
            if let Some(hint_field::Value::HintFieldArray(arr)) = &first_hf.value {
                &arr.hint_fields[..]
            } else {
                &raw_hint.hint_fields[..]
            }
        } else {
            continue;
        };

        let mut fields = Vec::new();
        for hint_field in inner_fields {
            let name = hint_field.name.clone().unwrap_or_default();
            let (values, lengths) = process_hint_field(
                &hint_name,
                hint_field,
                air_expressions,
                stage_widths,
                num_challenges,
                air_values,
                air_group_values,
                custom_commits,
                expressions,
                field,
            )?;
            let entry = if lengths.is_none() {
                HintFieldEntry { name, values: vec![values], lengths: None }
            } else {
                HintFieldEntry {
                    name,
                    values: match values {
                        HintFieldValue::Array(arr) => *arr,
                        single => vec![single],
                    },
                    lengths,
                }
            };
            fields.push(entry);
        }
        hints.push(HintInfo { name: hint_name, fields });
    }

    Ok(hints)
}

/// Recursively process a hint field of hint `hint`.
#[allow(clippy::too_many_arguments)]
fn process_hint_field(
    hint: &str,
    hint_field: &pb::HintField,
    air_expressions: &[pb::Expression],
    stage_widths: &[u32],
    num_challenges: &[u32],
    air_values: &[pb::AirValue],
    air_group_values: &[pb::AirGroupValue],
    custom_commits: &[pb::CustomCommit],
    expressions: &mut [Expression],
    field: &FieldCfg,
) -> Result<(HintFieldValue, Option<Vec<usize>>)> {
    let processed = match &hint_field.value {
        Some(hint_field::Value::HintFieldArray(arr)) => {
            let fields = &arr.hint_fields;
            let mut result_fields = Vec::new();
            let mut lengths: Vec<usize> = Vec::new();

            for sub_field in fields {
                let (values, sub_lengths) = process_hint_field(
                    hint,
                    sub_field,
                    air_expressions,
                    stage_widths,
                    num_challenges,
                    air_values,
                    air_group_values,
                    custom_commits,
                    expressions,
                    field,
                )?;
                result_fields.push(values);

                if lengths.is_empty() {
                    lengths.push(fields.len());
                }

                if let Some(sub) = sub_lengths {
                    for (k, &sub_len) in sub.iter().enumerate() {
                        if k + 1 >= lengths.len() {
                            lengths.resize(k + 2, 0);
                        }
                        if lengths[k + 1] == 0 {
                            lengths[k + 1] = sub_len;
                        }
                    }
                }
            }

            (HintFieldValue::Array(Box::new(result_fields)), Some(lengths))
        }
        Some(hint_field::Value::Operand(operand)) => {
            if let Some(ref op) = operand.operand {
                // Build a temporary FormatCtx just for this operand.
                // Hint field operands produce standalone Expression objects
                // (they are not inserted into the main expression arena).
                let mut ctx = FormatCtx {
                    field,
                    air_expressions,
                    stage_widths,
                    num_challenges,
                    air_values,
                    air_group_values,
                    custom_commits,
                    arena: Vec::new(),
                };
                let value = ctx.format_operand_inline(op)?;

                // If the value is an "exp" reference, mark keep=true
                if value.op == "exp" {
                    if let Some(id) = value.id {
                        if id < expressions.len() {
                            expressions[id].keep = Some(true);
                        }
                    }
                }
                (HintFieldValue::Single(Box::new(value)), None)
            } else {
                (
                    HintFieldValue::Single(Box::new(Expression {
                        op: "number".to_string(),
                        value: Some("0".to_string()),
                        ..Default::default()
                    })),
                    None,
                )
            }
        }
        Some(hint_field::Value::StringValue(s)) => (
            HintFieldValue::Single(Box::new(Expression {
                op: "string".to_string(),
                value: Some(s.clone()),
                ..Default::default()
            })),
            None,
        ),
        None => return Err(PilInfoError::HintFieldWithoutValue { hint: hint.to_string() }),
    };
    Ok(processed)
}

// ---------------------------------------------------------------------------
// get_pilout_info: main orchestrator
// ---------------------------------------------------------------------------

/// Extract pilout info for a single air, mirroring JS `getPiloutInfo`.
///
/// Fails on an air the pilout does not have, and on the parts of the air that refer to nothing
/// or that the passes cannot process.
pub fn get_pilout_info(
    pilout: &pb::PilOut,
    airgroup_id: usize,
    air_id: usize,
    field: &FieldCfg,
) -> Result<SetupResult> {
    let found = pilout.air_groups.get(airgroup_id).and_then(|ag| ag.airs.get(air_id).map(|air| (ag, air)));
    let Some((airgroup, air)) = found else {
        return Err(PilInfoError::NoSuchAir { airgroup_id, air_id });
    };

    let air_name = air.name.clone().unwrap_or_default();
    let num_rows = air.num_rows.unwrap_or(0);
    let pil_power = if num_rows > 0 { (num_rows as f64).log2() as u32 } else { 0 };

    let constraints = format_constraints(&air.constraints);
    check_constraint_expressions(&constraints, air.expressions.len(), &format!("air {air_name}"))?;

    let mut expressions = format_expressions(
        &air.expressions,
        &air.stage_widths,
        &pilout.num_challenges,
        &air.air_values,
        &airgroup.air_group_values,
        &air.custom_commits,
        field,
    )?;

    // Gather symbols for this air from the global pilout symbols list
    let air_symbols: Vec<pb::Symbol> = pilout
        .symbols
        .iter()
        .filter(|sym| {
            sym.air_group_id.is_none()
                || (sym.air_group_id == Some(airgroup_id as u32)
                    && (sym.air_id.is_none() || sym.air_id == Some(air_id as u32)))
        })
        .cloned()
        .collect();

    let mut all_symbols =
        format_symbols(&air_symbols, &pilout.num_challenges, &airgroup.air_group_values, &air.air_values, field)?;

    // Filter: keep only witness/fixed that match this air
    all_symbols.retain(|s| {
        if s.sym_type == "witness" || s.sym_type == "fixed" {
            s.air_id == Some(air_id) && s.airgroup_id == Some(airgroup_id)
        } else {
            true
        }
    });

    let n_commitments = all_symbols
        .iter()
        .filter(|s| s.sym_type == "witness" && s.air_id == Some(air_id) && s.airgroup_id == Some(airgroup_id))
        .count();

    let n_constants = all_symbols
        .iter()
        .filter(|s| s.sym_type == "fixed" && s.air_id == Some(air_id) && s.airgroup_id == Some(airgroup_id))
        .count();

    let n_publics = all_symbols.iter().filter(|s| s.sym_type == "public").count();

    let n_stages = if !pilout.num_challenges.is_empty() {
        pilout.num_challenges.len()
    } else {
        all_symbols.iter().filter_map(|s| s.stage).max().unwrap_or(0)
    };

    // Filter hints for this air (strict match, same as JS)
    let air_hints: Vec<pb::Hint> = pilout
        .hints
        .iter()
        .filter(|h| h.air_id == Some(air_id as u32) && h.air_group_id == Some(airgroup_id as u32))
        .cloned()
        .collect();

    let hints = format_hints(
        &air_hints,
        &air.expressions,
        &air.stage_widths,
        &pilout.num_challenges,
        &air.air_values,
        &airgroup.air_group_values,
        &air.custom_commits,
        &mut expressions,
        field,
    )?;

    // Build custom commits info
    let mut map_sections_n = IndexMap::new();
    map_sections_n.insert("const".to_string(), 0);

    let mut custom_commits_info = Vec::new();
    let mut custom_commits_map: Vec<Vec<SymbolInfo>> = Vec::new();

    for cc in &air.custom_commits {
        let cc_name = cc.name.clone().unwrap_or_default();
        custom_commits_info.push(CustomCommitInfo {
            name: cc_name.clone(),
            stage_widths: cc.stage_widths.clone(),
            public_values: cc.public_values.iter().map(|pv| pv.idx).collect(),
        });
        custom_commits_map.push(Vec::new());

        for (j, &width) in cc.stage_widths.iter().enumerate() {
            if width > 0 {
                map_sections_n.insert(format!("{}{}", cc_name, j), 0);
            }
        }
    }

    Ok(SetupResult {
        name: air_name,
        air_id,
        airgroup_id,
        pil_power,
        n_stages,
        n_constants,
        n_publics,
        n_commitments,
        cm_pols_map: Vec::new(),
        const_pols_map: Vec::new(),
        challenges_map: Vec::new(),
        publics_map: Vec::new(),
        proof_values_map: Vec::new(),
        airgroup_values_map: Vec::new(),
        air_values_map: Vec::new(),
        map_sections_n,
        custom_commits: custom_commits_info,
        custom_commits_map,
        air_group_values: airgroup.air_group_values.clone(),
        expressions,
        constraints,
        symbols: all_symbols,
        hints,
        n_commitments_stage1: 0,
        im_pols_info: (Vec::new(), Vec::new()),
        opening_points: Vec::new(),
    })
}

// ---------------------------------------------------------------------------
// Expression arena helpers
// ---------------------------------------------------------------------------

/// Convert a flat `Vec<Expression>` into an `ExpressionArena`.
pub fn build_arena(exprs: Vec<Expression>) -> ExpressionArena {
    let mut arena = ExpressionArena::new();
    for e in exprs {
        arena.push(e);
    }
    arena
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn buf_to_bigint_string_reads_big_endian() {
        assert_eq!(buf_to_bigint_string(&[]), "0");
        assert_eq!(buf_to_bigint_string(&[0]), "0");
        assert_eq!(buf_to_bigint_string(&[0x01, 0x00]), "256");
        // Goldilocks p − 1, the widest value a STARK pilout holds.
        assert_eq!(buf_to_bigint_string(&[0xff, 0xff, 0xff, 0xff, 0, 0, 0, 0]), "18446744069414584320");
    }

    #[test]
    fn buf_to_bigint_string_keeps_values_wider_than_128_bits() {
        // BN254 r − 1: 32 bytes, which a u128 accumulator silently truncated.
        let r_minus_one: [u8; 32] = [
            0x30, 0x64, 0x4e, 0x72, 0xe1, 0x31, 0xa0, 0x29, 0xb8, 0x50, 0x45, 0xb6, 0x81, 0x81, 0x58, 0x5d, 0x28, 0x33,
            0xe8, 0x48, 0x79, 0xb9, 0x70, 0x91, 0x43, 0xe1, 0xf5, 0x93, 0xf0, 0x00, 0x00, 0x00,
        ];
        assert_eq!(
            buf_to_bigint_string(&r_minus_one),
            "21888242871839275222246405745257275088548364400416034343698204186575808495616"
        );
        // Leading zero bytes do not change the value.
        let mut padded = vec![0u8; 8];
        padded.extend_from_slice(&r_minus_one);
        assert_eq!(buf_to_bigint_string(&padded), buf_to_bigint_string(&r_minus_one));
    }
}
