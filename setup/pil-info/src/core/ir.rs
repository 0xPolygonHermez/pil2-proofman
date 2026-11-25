//! Typed Expr, ConstraintSystem, the only traversal and degree.

use std::collections::BTreeMap;

use num_bigint::BigUint;
use serde::{Deserialize, Serialize};

use super::pilout_core::AggType;
use crate::cfg::{FieldCfg, Opening};

/// Index into the lowered expression table: numbered canonically and topologically (every `Ref` points below).
pub type ExprId = u32;
/// A field element, reduced mod the field's modulus.
pub type Fe = BigUint;

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ColType {
    Witness,
    Fixed,
    Custom,
}

/// One column read: the key of the evMap, the DAG leaves and `Env`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ColRef {
    pub col: ColType,
    pub commit_id: Option<u32>,
    pub id: u32,
    pub offset: i64,
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Expr {
    Add(Box<Expr>, Box<Expr>),
    Sub(Box<Expr>, Box<Expr>),
    Mul(Box<Expr>, Box<Expr>),
    Neg(Box<Expr>),
    Const(Fe),
    /// `id` is the stage-offset column id (witness/custom) or the fixed index; `commit_id` is set for custom.
    Col {
        col: ColType,
        commit_id: Option<u32>,
        id: u32,
        offset: i64,
    },
    Challenge {
        stage: u32,
        stage_id: u32,
    },
    Public(u32),
    ProofValue(u32),
    AirValue(u32),
    /// `airgroup` is None in an air's expressions (its own airgroup) and set in global ones.
    AirgroupValue {
        airgroup: Option<u32>,
        id: u32,
    },
    /// Named sub-expression of the pilout; resolved by `im::substitute`.
    Ref(ExprId),
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum Boundary {
    EveryRow,
    FirstRow,
    LastRow,
    EveryFrame {
        #[serde(rename = "offsetMin")]
        min: u32,
        #[serde(rename = "offsetMax")]
        max: u32,
    },
}

impl Boundary {
    pub fn name(&self) -> &'static str {
        match self {
            Boundary::EveryRow => "everyRow",
            Boundary::FirstRow => "firstRow",
            Boundary::LastRow => "lastRow",
            Boundary::EveryFrame { .. } => "everyFrame",
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Constraint {
    pub e: Expr,
    pub boundary: Boundary,
    pub label: String,
}

/// Indices of an array symbol's cell; `None` for a scalar symbol.
pub type Lengths = Option<Vec<u32>>;

/// A column in pilout order; `name` is its symbol's, `lengths` its cell within it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Column {
    pub name: String,
    pub stage: u32,
    pub dim: u32,
    pub id: u32,
    pub lengths: Lengths,
}

/// A custom commit as the air declares it.
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct CustomCommit {
    pub name: String,
    #[serde(serialize_with = "idx_objects")]
    pub public_values: Vec<u32>,
    pub stage_widths: Vec<u32>,
}

/// Public value ids as the starkinfo writes them, `[{"idx": i}]`.
fn idx_objects<S: serde::Serializer>(v: &[u32], s: S) -> Result<S::Ok, S::Error> {
    s.collect_seq(v.iter().map(|idx| serde_json::json!({ "idx": idx })))
}

/// A public, proof value, air value or airgroup value.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize)]
pub struct NamedValue {
    pub name: String,
    pub stage: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub lengths: Lengths,
    /// An airgroup value's aggregation; `Sum` for the other values.
    #[serde(skip)]
    pub agg_type: AggType,
}

/// A hint field: a string, one operand of an expression, or an array of fields.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum HintValue {
    Str(String),
    Expr(Expr),
    Array(Vec<HintField>),
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct HintField {
    pub name: Option<String>,
    pub value: HintValue,
}

/// A pilout hint, its operands lowered like the expressions; `fields` are a leading array's elements, else all.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Hint {
    pub name: String,
    pub fields: Vec<HintField>,
}

/// A lookup the system cannot resolve (an id past its table).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Unresolved(pub String);

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ConstraintSystem {
    /// The air's name; empty when the pilout gives none.
    pub name: String,
    pub field: FieldCfg,
    pub opening: Opening,
    pub n_stages: u32,
    pub n_rows_bits: u32,
    pub exprs: Vec<Expr>,
    pub constraints: Vec<Constraint>,
    pub witness: Vec<Column>,
    pub fixed: Vec<Column>,
    pub custom: Vec<Vec<Column>>,
    pub challenges: Vec<(u32, u32, String)>,
    /// Publics are stage 1.
    pub publics: Vec<NamedValue>,
    pub proof_values: Vec<NamedValue>,
    pub air_values: Vec<NamedValue>,
    pub airgroup_values: Vec<NamedValue>,
    pub hints: Vec<Hint>,
    pub custom_commits: Vec<CustomCommit>,
}

impl ConstraintSystem {
    /// everyRow, then each other boundary at its first constraint: pilfflonk's `boundaryId` order.
    pub fn boundaries(&self) -> Vec<Boundary> {
        let mut out = vec![Boundary::EveryRow];
        for c in &self.constraints {
            if !out.contains(&c.boundary) {
                out.push(c.boundary.clone());
            }
        }
        out
    }
}

/// Dimension of a column of `stage`: base field in stages 0 and 1, else the extension.
pub fn col_dim(stage: u32, ext: usize) -> usize {
    if stage <= 1 {
        1
    } else {
        ext
    }
}

/// Dimension of an air, airgroup or proof value of `stage`: base field in stage 1 only.
pub fn value_dim(stage: u32, ext: usize) -> usize {
    if stage == 1 {
        1
    } else {
        ext
    }
}

/// Values for `eval_expr`; a missing entry reads as 0.
#[derive(Clone, Debug, Default)]
pub struct Env {
    pub cols: BTreeMap<ColRef, Fe>,
    pub challenges: BTreeMap<(u32, u32), Fe>,
    pub publics: BTreeMap<u32, Fe>,
    pub proof_values: BTreeMap<u32, Fe>,
    pub air_values: BTreeMap<u32, Fe>,
    pub airgroup_values: BTreeMap<(Option<u32>, u32), Fe>,
}

/// One pass in index order: `node` reads a `Ref` from the entries before it (lowering numbers topologically).
fn table<T, E>(cs: &ConstraintSystem, mut node: impl FnMut(&Expr, &[T]) -> Result<T, E>) -> Result<Vec<T>, E> {
    let mut t = Vec::with_capacity(cs.exprs.len());
    for e in &cs.exprs {
        let v = node(e, &t)?;
        t.push(v);
    }
    Ok(t)
}

/// An operation with its operands' values, or a leaf other than `Ref`.
enum Node<'a, T> {
    Add(T, T),
    Sub(T, T),
    Mul(T, T),
    Neg(T),
    Leaf(&'a Expr),
}

/// THE fold behind degree, dim, stage and eval: `f` values a node from its operands', `Ref(i)` reads `t[i]`.
fn fold<T: Clone>(e: &Expr, t: &[T], f: &impl Fn(Node<T>) -> Result<T, Unresolved>) -> Result<T, Unresolved> {
    match e {
        Expr::Add(a, b) => f(Node::Add(fold(a, t, f)?, fold(b, t, f)?)),
        Expr::Sub(a, b) => f(Node::Sub(fold(a, t, f)?, fold(b, t, f)?)),
        Expr::Mul(a, b) => f(Node::Mul(fold(a, t, f)?, fold(b, t, f)?)),
        Expr::Neg(a) => f(Node::Neg(fold(a, t, f)?)),
        // past `t` breaks the topological numbering `from_pilout` guarantees
        Expr::Ref(id) => t
            .get(*id as usize)
            .cloned()
            .ok_or_else(|| Unresolved(format!("expression {id} read before it is numbered"))),
        leaf => f(Node::Leaf(leaf)),
    }
}

/// The largest operand's value, `leaf` for a leaf (dim and stage).
fn max_of(n: Node<u32>, leaf: impl Fn(&Expr) -> Result<u32, Unresolved>) -> Result<u32, Unresolved> {
    match n {
        Node::Add(a, b) | Node::Sub(a, b) | Node::Mul(a, b) => Ok(a.max(b)),
        Node::Neg(a) => Ok(a),
        Node::Leaf(e) => leaf(e),
    }
}

/// `f` of each witness column, by id.
fn by_witness_id(cs: &ConstraintSystem, f: impl Fn(&Column) -> u32) -> Vec<Option<u32>> {
    let mut v = vec![None; cs.witness.iter().map(|c| c.id as usize + 1).max().unwrap_or(0)];
    cs.witness.iter().for_each(|c| v[c.id as usize] = Some(f(c)));
    v
}

fn witness_at(witness: &[Option<u32>], id: u32) -> Result<u32, Unresolved> {
    witness.get(id as usize).copied().flatten().ok_or_else(|| Unresolved(format!("witness column {id}")))
}

/// THE degree rule: columns 1, constants/challenges/values 0, `Ref(i)` = `t[i]`.
pub fn degree_of(e: &Expr, t: &[u32]) -> Result<u32, Unresolved> {
    fold(e, t, &|n| {
        Ok(match n {
            Node::Add(a, b) | Node::Sub(a, b) => a.max(b),
            Node::Mul(a, b) => a.saturating_add(b),
            Node::Neg(a) => a,
            Node::Leaf(l) => u32::from(matches!(l, Expr::Col { .. })),
        })
    })
}

/// Degree of each expression as a `Ref` reads it: a `promoted` one is its column (1).
pub fn degree_table_with(cs: &ConstraintSystem, promoted: &[ExprId]) -> Result<Vec<u32>, Unresolved> {
    // `t.len()` is the index of `e`
    table(cs, |e, t| if promoted.contains(&(t.len() as u32)) { Ok(1) } else { degree_of(e, t) })
}

/// Degree of every expression, one pass in index order.
pub fn degree_table(cs: &ConstraintSystem) -> Result<Vec<u32>, Unresolved> {
    degree_table_with(cs, &[])
}

/// `degree_of` over a fresh table; a pass over many expressions builds the table once instead.
pub fn degree(cs: &ConstraintSystem, e: &Expr) -> Result<u32, Unresolved> {
    degree_of(e, &degree_table(cs)?)
}

/// Dimension of `e`: the largest of its leaves (witness/values by stage, challenges extension), `Ref(i)` = `t[i]`.
fn dim_of(cs: &ConstraintSystem, witness: &[Option<u32>], e: &Expr, t: &[u32]) -> Result<u32, Unresolved> {
    let ext = cs.field.ext_dim() as u32;
    let by_stage = |v: &[NamedValue], i: u32, what: &str| match v.get(i as usize) {
        Some(v) => Ok(value_dim(v.stage, ext as usize) as u32),
        None => Err(Unresolved(format!("{what} {i}"))),
    };
    let leaf = |e: &Expr| match e {
        Expr::Col { col: ColType::Witness, id, .. } => witness_at(witness, *id),
        // fixed and custom columns are stage 0 (custom of another stage is rejected at ingestion)
        Expr::Col { .. } | Expr::Const(_) | Expr::Public(_) => Ok(1),
        Expr::AirValue(i) => by_stage(&cs.air_values, *i, "air value"),
        Expr::ProofValue(i) => by_stage(&cs.proof_values, *i, "proof value"),
        Expr::AirgroupValue { airgroup: None, id } => by_stage(&cs.airgroup_values, *id, "airgroup value"),
        // challenges and another airgroup's values
        _ => Ok(ext),
    };
    fold(e, t, &|n| max_of(n, leaf))
}

/// Dimension of every expression, one pass in index order.
pub fn dim_table(cs: &ConstraintSystem) -> Result<Vec<u32>, Unresolved> {
    let witness = by_witness_id(cs, |c| c.dim);
    table(cs, |e, t| dim_of(cs, &witness, e, t))
}

/// Dimension of `e` over a fresh table.
pub fn dim(cs: &ConstraintSystem, e: &Expr) -> Result<u32, Unresolved> {
    dim_of(cs, &by_witness_id(cs, |c| c.dim), e, &dim_table(cs)?)
}

/// Stage of `e`: the largest of its leaves' (witness by column, values by their stage, challenges theirs).
fn stage_of(cs: &ConstraintSystem, witness: &[Option<u32>], e: &Expr, t: &[u32]) -> Result<u32, Unresolved> {
    let at = |v: &[NamedValue], i: u32, what: &str| {
        v.get(i as usize).map(|x| x.stage).ok_or_else(|| Unresolved(format!("{what} {i}")))
    };
    let leaf = |e: &Expr| match e {
        Expr::Col { col: ColType::Witness, id, .. } => witness_at(witness, *id),
        Expr::Challenge { stage, .. } => Ok(*stage),
        Expr::Public(_) => Ok(1),
        Expr::ProofValue(i) => at(&cs.proof_values, *i, "proof value"),
        Expr::AirValue(i) => at(&cs.air_values, *i, "air value"),
        Expr::AirgroupValue { airgroup: None, id } => at(&cs.airgroup_values, *id, "airgroup value"),
        // fixed/custom columns, constants and another airgroup's values
        _ => Ok(0),
    };
    fold(e, t, &|n| max_of(n, leaf))
}

/// Stage of every expression, one pass in index order.
pub fn stage_table(cs: &ConstraintSystem) -> Result<Vec<u32>, Unresolved> {
    let witness = by_witness_id(cs, |c| c.stage);
    table(cs, |e, t| stage_of(cs, &witness, e, t))
}

/// THE traversal: pre-order; `f` returning `Some` replaces the node (its result is not revisited).
pub fn map_expr(e: &Expr, f: &mut impl FnMut(&Expr) -> Option<Expr>) -> Expr {
    if let Some(r) = f(e) {
        return r;
    }
    match e {
        Expr::Add(a, b) => Expr::Add(Box::new(map_expr(a, f)), Box::new(map_expr(b, f))),
        Expr::Sub(a, b) => Expr::Sub(Box::new(map_expr(a, f)), Box::new(map_expr(b, f))),
        Expr::Mul(a, b) => Expr::Mul(Box::new(map_expr(a, f)), Box::new(map_expr(b, f))),
        Expr::Neg(a) => Expr::Neg(Box::new(map_expr(a, f))),
        leaf => leaf.clone(),
    }
}

/// Pre-order read-only walk over every node.
pub fn visit(e: &Expr, f: &mut impl FnMut(&Expr)) {
    f(e);
    match e {
        Expr::Add(a, b) | Expr::Sub(a, b) | Expr::Mul(a, b) => {
            visit(a, f);
            visit(b, f);
        }
        Expr::Neg(a) => visit(a, f),
        _ => {}
    }
}

/// Value of `e` under `env` (a missing entry reads as 0), `Ref(i)` = `t[i]`.
fn eval_of(cs: &ConstraintSystem, env: &Env, e: &Expr, t: &[Fe]) -> Result<Fe, Unresolved> {
    let p = cs.field.modulus();
    let get = |m: &BTreeMap<u32, Fe>, k: &u32| m.get(k).cloned().unwrap_or_default();
    let leaf = |e: &Expr| match e {
        Expr::Const(c) => c.clone(),
        Expr::Col { col, commit_id, id, offset } => {
            let k = ColRef { col: *col, commit_id: *commit_id, id: *id, offset: *offset };
            env.cols.get(&k).cloned().unwrap_or_default()
        }
        Expr::Challenge { stage, stage_id } => env.challenges.get(&(*stage, *stage_id)).cloned().unwrap_or_default(),
        Expr::Public(i) => get(&env.publics, i),
        Expr::ProofValue(i) => get(&env.proof_values, i),
        Expr::AirValue(i) => get(&env.air_values, i),
        Expr::AirgroupValue { airgroup, id } => env.airgroup_values.get(&(*airgroup, *id)).cloned().unwrap_or_default(),
        // operations and `Ref` never reach a leaf
        _ => Fe::default(),
    };
    fold(e, t, &|n| {
        Ok(match n {
            Node::Add(a, b) => (a + b) % p,
            Node::Sub(a, b) => (a + p - b % p) % p,
            Node::Mul(a, b) => (a * b) % p,
            Node::Neg(a) => (p - a % p) % p,
            Node::Leaf(l) => leaf(l),
        })
    })
}

/// Value of every expression under `env`, one pass in index order.
pub fn eval_table(cs: &ConstraintSystem, env: &Env) -> Result<Vec<Fe>, Unresolved> {
    table(cs, |e, t| eval_of(cs, env, e, t))
}

/// Reference evaluator over the field (tests only), over a fresh table.
pub fn eval_expr(cs: &ConstraintSystem, e: &Expr, env: &Env) -> Result<Fe, Unresolved> {
    eval_of(cs, env, e, &eval_table(cs, env)?)
}
