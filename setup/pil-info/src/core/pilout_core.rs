//! The core's input: the pilout subset it reads (docs/superpowers/specs/2026-11-24-pilout-core-input.md).

use num_bigint::BigUint;
use pil2_pilout::pilout::{self as pb, constraint, expression as ex, hint_field, operand, SymbolType};
use serde::{Deserialize, Serialize};

use super::ir::Boundary;

#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub enum PiloutError {
    #[error("constraint {constraint} has no expression_idx")]
    MissingExpressionIdx { constraint: usize },
    #[error("constraint {constraint} has no body")]
    EmptyConstraint { constraint: usize },
    #[error("constraint {constraint} (`{line}`) is {boundary}, and the STARK takes only everyRow")]
    BoundaryNotEveryRow { constraint: usize, line: String, boundary: String },
    #[error("expression {expr} refers to expression {to}, which does not exist")]
    DanglingRef { expr: usize, to: u32 },
    #[error("expression {expr} refers to itself through its references")]
    CyclicRef { expr: usize },
    #[error("the pilout has no air {air} in airgroup {airgroup}")]
    UnknownAir { airgroup: usize, air: usize },
    #[error("no pilout expression has canonical id {id}")]
    NoExpression { id: u32 },
    #[error("expression {expr}: {why}")]
    BadOperand { expr: usize, why: &'static str },
    #[error("hint `{hint}`: {why}")]
    BadHint { hint: String, why: &'static str },
    #[error("air {air} of airgroup {airgroup} declares no num_rows")]
    MissingNumRows { airgroup: usize, air: usize },
    #[error("air {air} of airgroup {airgroup} has {rows} rows, not a power of two")]
    NumRowsNotPowerOfTwo { airgroup: usize, air: usize, rows: u32 },
    #[error("symbol `{symbol}` declares no stage")]
    MissingStage { symbol: String },
    #[error("custom column `{symbol}` declares no commit_id")]
    MissingCommitId { symbol: String },
    #[error("air {air} of airgroup {airgroup} has no constraints")]
    NoConstraints { airgroup: usize, air: usize },
    #[error("the pilout declares {declared} public values and its symbols name {named}")]
    PublicCount { declared: u32, named: usize },
    #[error("custom commit `{commit}` binds public {idx}, past the {declared} public values")]
    CommitPublic { commit: String, idx: u32, declared: u32 },
    #[error("{what} does not fit in 32 bits")]
    Overflow { what: String },
    #[error("pilout core input: {why}")]
    Input { why: String },
}

pub(crate) fn bad(expr: usize, why: &'static str) -> PiloutError {
    PiloutError::BadOperand { expr, why }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct PiloutCore {
    pub num_challenges: Vec<u32>,
    pub num_proof_values: Vec<u32>,
    pub num_public_values: u32,
    pub air_groups: Vec<AirGroup>,
    pub symbols: Vec<Symbol>,
    pub hints: Vec<Hint>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct AirGroup {
    pub air_group_values: Vec<AirGroupValue>,
    pub airs: Vec<Air>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct AirGroupValue {
    pub agg_type: AggType,
    pub stage: u32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum AggType {
    Sum,
    Prod,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct Air {
    pub num_rows: u32,
    pub stage_widths: Vec<u32>,
    pub n_fixed_cols: u32,
    pub n_periodic_cols: u32,
    /// Stage of each air value.
    pub air_values: Vec<u32>,
    pub custom_commits: Vec<CustomCommit>,
    pub expressions: Vec<Expression>,
    pub constraints: Vec<Constraint>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct CustomCommit {
    pub name: String,
    pub public_values: Vec<u32>,
    pub stage_widths: Vec<u32>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum Expression {
    Add([Operand; 2]),
    Sub([Operand; 2]),
    Mul([Operand; 2]),
    Neg(Operand),
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum Operand {
    Expression(u32),
    #[serde(with = "decimal")]
    Constant(BigUint),
    WitnessCol(StagedCol),
    CustomCol(CustomCol),
    FixedCol(Col),
    PeriodicCol(Col),
    PublicValue(u32),
    AirGroupValue(u32),
    AirValue(u32),
    Challenge(StagedId),
    ProofValue(StagedId),
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct Col {
    pub idx: u32,
    pub row_offset: i32,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct StagedCol {
    pub stage: u32,
    pub col_idx: u32,
    pub row_offset: i32,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct CustomCol {
    pub commit_id: u32,
    pub stage: u32,
    pub col_idx: u32,
    pub row_offset: i32,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct StagedId {
    pub stage: u32,
    pub idx: u32,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct Constraint {
    pub kind: Boundary,
    pub expression_idx: u32,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub debug_line: Option<String>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum SymbolKind {
    FixedCol,
    WitnessCol,
    ProofValue,
    AirGroupValue,
    PublicValue,
    Challenge,
    AirValue,
    CustomCol,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct Symbol {
    pub name: String,
    #[serde(rename = "type")]
    pub kind: SymbolKind,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub air_group_id: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub air_id: Option<u32>,
    pub id: u32,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stage: Option<u32>,
    pub dim: u32,
    pub lengths: Vec<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub commit_id: Option<u32>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct Hint {
    pub name: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub air_group_id: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub air_id: Option<u32>,
    pub fields: Vec<HintField>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct HintField {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    pub value: HintValue,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum HintValue {
    String(String),
    Operand(Operand),
    Array(Vec<HintField>),
}

/// A field element as a decimal string.
mod decimal {
    use num_bigint::BigUint;
    use serde::{de::Error, Deserialize, Deserializer, Serializer};

    pub fn serialize<S: Serializer>(v: &BigUint, s: S) -> Result<S::Ok, S::Error> {
        s.serialize_str(&v.to_str_radix(10))
    }
    pub fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<BigUint, D::Error> {
        let s = String::deserialize(d)?;
        let digits = !s.is_empty() && s.bytes().all(|b| b.is_ascii_digit());
        digits
            .then(|| BigUint::parse_bytes(s.as_bytes(), 10))
            .flatten()
            .ok_or_else(|| D::Error::custom("not a decimal"))
    }
}

fn input(why: String) -> PiloutError {
    PiloutError::Input { why }
}

impl PiloutCore {
    /// The only protobuf reader on the trusted path: a structural copy, every absent required field an error.
    pub fn from_pb(p: &pb::PilOut) -> Result<Self, PiloutError> {
        let mut air_groups = Vec::with_capacity(p.air_groups.len());
        for (g, ag) in p.air_groups.iter().enumerate() {
            let airs = ag.airs.iter().enumerate().map(|(a, x)| air_of(x, g, a)).collect::<Result<_, _>>()?;
            air_groups.push(AirGroup { air_group_values: air_group_values(ag)?, airs });
        }
        Ok(PiloutCore { air_groups, ..Self::globals_of(p)? })
    }

    /// `from_pb` without the airgroups: the pilout-wide tables, every symbol and every hint.
    pub(crate) fn globals_of(p: &pb::PilOut) -> Result<Self, PiloutError> {
        Ok(PiloutCore {
            num_challenges: p.num_challenges.clone(),
            num_proof_values: p.num_proof_values.clone(),
            num_public_values: p.num_public_values,
            air_groups: Vec::new(),
            symbols: p.symbols.iter().filter_map(|s| symbol_of(s).transpose()).collect::<Result<_, _>>()?,
            hints: p.hints.iter().map(hint_of).collect::<Result<_, _>>()?,
        })
    }

    pub fn from_json(s: &str) -> Result<Self, PiloutError> {
        serde_json::from_str(s).map_err(|e| input(e.to_string()))
    }

    pub fn to_json(&self) -> Result<String, PiloutError> {
        serde_json::to_string(self).map_err(|e| input(e.to_string()))
    }
}

pub(crate) fn air_group_values(ag: &pb::AirGroup) -> Result<Vec<AirGroupValue>, PiloutError> {
    let value = |v: &pb::AirGroupValue| Ok(AirGroupValue { agg_type: agg_type(v.agg_type)?, stage: v.stage });
    ag.air_group_values.iter().map(value).collect()
}

fn agg_type(v: i32) -> Result<AggType, PiloutError> {
    match pb::AggregationType::try_from(v) {
        Ok(pb::AggregationType::Sum) => Ok(AggType::Sum),
        Ok(pb::AggregationType::Prod) => Ok(AggType::Prod),
        Err(_) => Err(input(format!("unknown aggregation type {v}"))),
    }
}

pub(crate) fn air_of(a: &pb::Air, airgroup: usize, air: usize) -> Result<Air, PiloutError> {
    let len = |n: usize, what: &str| u32::try_from(n).map_err(|_| input(format!("too many {what}")));
    let expressions = a.expressions.iter().enumerate().map(|(i, e)| expression_of(e, i)).collect::<Result<_, _>>()?;
    let constraints = a.constraints.iter().enumerate().map(|(k, c)| constraint_of(c, k)).collect::<Result<_, _>>()?;
    Ok(Air {
        num_rows: a.num_rows.ok_or(PiloutError::MissingNumRows { airgroup, air })?,
        stage_widths: a.stage_widths.clone(),
        n_fixed_cols: len(a.fixed_cols.len(), "fixed columns")?,
        n_periodic_cols: len(a.periodic_cols.len(), "periodic columns")?,
        air_values: a.air_values.iter().map(|v| v.stage).collect(),
        custom_commits: a
            .custom_commits
            .iter()
            .map(|c| {
                let name = c.name.clone().ok_or_else(|| input("a custom commit without a name".into()))?;
                let public_values = c.public_values.iter().map(|v| v.idx).collect();
                Ok(CustomCommit { name, public_values, stage_widths: c.stage_widths.clone() })
            })
            .collect::<Result<_, PiloutError>>()?,
        expressions,
        constraints,
    })
}

fn expression_of(e: &pb::Expression, i: usize) -> Result<Expression, PiloutError> {
    let o = |x: &Option<pb::Operand>| x.as_ref().and_then(operand_of).ok_or_else(|| bad(i, "missing operand"));
    Ok(match e.operation.as_ref().ok_or_else(|| bad(i, "expression without an operation"))? {
        ex::Operation::Add(a) => Expression::Add([o(&a.lhs)?, o(&a.rhs)?]),
        ex::Operation::Sub(a) => Expression::Sub([o(&a.lhs)?, o(&a.rhs)?]),
        ex::Operation::Mul(a) => Expression::Mul([o(&a.lhs)?, o(&a.rhs)?]),
        ex::Operation::Neg(a) => Expression::Neg(o(&a.value)?),
    })
}

fn operand_of(o: &pb::Operand) -> Option<Operand> {
    use operand::Operand as O;
    Some(match o.operand.as_ref()? {
        O::Expression(r) => Operand::Expression(r.idx),
        O::Constant(c) => Operand::Constant(BigUint::from_bytes_be(&c.value)),
        O::WitnessCol(w) => {
            Operand::WitnessCol(StagedCol { stage: w.stage, col_idx: w.col_idx, row_offset: w.row_offset })
        }
        O::CustomCol(c) => Operand::CustomCol(CustomCol {
            commit_id: c.commit_id,
            stage: c.stage,
            col_idx: c.col_idx,
            row_offset: c.row_offset,
        }),
        O::FixedCol(f) => Operand::FixedCol(Col { idx: f.idx, row_offset: f.row_offset }),
        O::PeriodicCol(f) => Operand::PeriodicCol(Col { idx: f.idx, row_offset: f.row_offset }),
        O::PublicValue(v) => Operand::PublicValue(v.idx),
        O::AirGroupValue(v) => Operand::AirGroupValue(v.idx),
        O::AirValue(v) => Operand::AirValue(v.idx),
        O::Challenge(c) => Operand::Challenge(StagedId { stage: c.stage, idx: c.idx }),
        O::ProofValue(v) => Operand::ProofValue(StagedId { stage: v.stage, idx: v.idx }),
    })
}

fn constraint_of(c: &pb::Constraint, k: usize) -> Result<Constraint, PiloutError> {
    use constraint::Constraint as C;
    let (kind, e, line) = match c.constraint.as_ref().ok_or(PiloutError::EmptyConstraint { constraint: k })? {
        C::EveryRow(x) => (Boundary::EveryRow, &x.expression_idx, &x.debug_line),
        C::FirstRow(x) => (Boundary::FirstRow, &x.expression_idx, &x.debug_line),
        C::LastRow(x) => (Boundary::LastRow, &x.expression_idx, &x.debug_line),
        C::EveryFrame(x) => {
            (Boundary::EveryFrame { min: x.offset_min, max: x.offset_max }, &x.expression_idx, &x.debug_line)
        }
    };
    let expression_idx = e.as_ref().map(|e| e.idx).ok_or(PiloutError::MissingExpressionIdx { constraint: k })?;
    Ok(Constraint { kind, expression_idx, debug_line: line.clone() })
}

/// `None` for a symbol kind the core does not read (imCol, periodicCol, publicTable).
fn symbol_of(s: &pb::Symbol) -> Result<Option<Symbol>, PiloutError> {
    let kind = match SymbolType::try_from(s.r#type) {
        Ok(SymbolType::FixedCol) => SymbolKind::FixedCol,
        Ok(SymbolType::WitnessCol) => SymbolKind::WitnessCol,
        Ok(SymbolType::ProofValue) => SymbolKind::ProofValue,
        Ok(SymbolType::AirGroupValue) => SymbolKind::AirGroupValue,
        Ok(SymbolType::PublicValue) => SymbolKind::PublicValue,
        Ok(SymbolType::Challenge) => SymbolKind::Challenge,
        Ok(SymbolType::AirValue) => SymbolKind::AirValue,
        Ok(SymbolType::CustomCol) => SymbolKind::CustomCol,
        Ok(SymbolType::ImCol | SymbolType::PeriodicCol | SymbolType::PublicTable) => return Ok(None),
        Err(_) => return Err(input(format!("symbol `{}` has unknown type {}", s.name, s.r#type))),
    };
    Ok(Some(Symbol {
        name: s.name.clone(),
        kind,
        air_group_id: s.air_group_id,
        air_id: s.air_id,
        id: s.id,
        stage: s.stage,
        dim: s.dim,
        lengths: s.lengths.clone(),
        commit_id: s.commit_id,
    }))
}

fn hint_of(h: &pb::Hint) -> Result<Hint, PiloutError> {
    let fields = h.hint_fields.iter().map(|f| hint_field_of(f, &h.name)).collect::<Result<_, _>>()?;
    Ok(Hint { name: h.name.clone(), air_group_id: h.air_group_id, air_id: h.air_id, fields })
}

fn hint_field_of(f: &pb::HintField, hint: &str) -> Result<HintField, PiloutError> {
    let missing = || input(format!("a field of hint `{hint}` has no value"));
    let value = match f.value.as_ref().ok_or_else(missing)? {
        hint_field::Value::StringValue(s) => HintValue::String(s.clone()),
        hint_field::Value::Operand(o) => HintValue::Operand(operand_of(o).ok_or_else(missing)?),
        hint_field::Value::HintFieldArray(a) => {
            HintValue::Array(a.hint_fields.iter().map(|f| hint_field_of(f, hint)).collect::<Result<_, _>>()?)
        }
    };
    Ok(HintField { name: f.name.clone(), value })
}
