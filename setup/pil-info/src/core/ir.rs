//! Typed Expr, ConstraintSystem, the only traversal and degree.

use std::collections::BTreeMap;

use num_bigint::BigUint;
use pil2_pilout::pilout as pb;

use crate::cfg::{FieldCfg, Opening};

/// Index into the pilout expression table (input-determined, never an arena push order).
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

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Expr {
    Add(Box<Expr>, Box<Expr>),
    Sub(Box<Expr>, Box<Expr>),
    Mul(Box<Expr>, Box<Expr>),
    Neg(Box<Expr>),
    Const(BigUint),
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

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Boundary {
    EveryRow,
    FirstRow,
    LastRow,
    EveryFrame { min: u32, max: u32 },
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
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CustomCommit {
    pub name: String,
    pub public_values: Vec<u32>,
    pub stage_widths: Vec<u32>,
}

/// A public, proof value, air value or airgroup value.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct NamedValue {
    pub name: String,
    pub stage: u32,
    pub lengths: Lengths,
}

/// A pilout hint, passed through untyped until a task needs its fields.
#[derive(Clone, Debug, PartialEq)]
pub struct Hint {
    pub name: String,
    pub fields: Vec<pb::HintField>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ConstraintSystem {
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
    /// `aggType` of each of the airgroup's values.
    pub airgroup_agg_types: Vec<i32>,
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

/// THE degree function: columns 1, constants/challenges/values 0, `Ref` = degree of its target.
pub fn degree(cs: &ConstraintSystem, e: &Expr) -> u32 {
    degree_with(cs, e, &[])
}

/// `degree` with each `promoted` Ref read as a column (degree 1), without inlining the DAG.
pub fn degree_with(cs: &ConstraintSystem, e: &Expr, promoted: &[ExprId]) -> u32 {
    degree_in(cs, e, &mut promoted.iter().map(|&id| (id, 1)).collect())
}

fn degree_in(cs: &ConstraintSystem, e: &Expr, memo: &mut BTreeMap<ExprId, u32>) -> u32 {
    match e {
        Expr::Add(a, b) | Expr::Sub(a, b) => degree_in(cs, a, memo).max(degree_in(cs, b, memo)),
        Expr::Mul(a, b) => degree_in(cs, a, memo).saturating_add(degree_in(cs, b, memo)),
        Expr::Neg(a) => degree_in(cs, a, memo),
        Expr::Col { .. } => 1,
        Expr::Const(_)
        | Expr::Challenge { .. }
        | Expr::Public(_)
        | Expr::ProofValue(_)
        | Expr::AirValue(_)
        | Expr::AirgroupValue { .. } => 0,
        Expr::Ref(id) => {
            if let Some(d) = memo.get(id) {
                return *d;
            }
            memo.insert(*id, 0); // a cycle (rejected at ingestion) reads as 0 rather than looping
            let d = cs.exprs.get(*id as usize).map_or(0, |t| degree_in(cs, t, memo));
            memo.insert(*id, d);
            d
        }
    }
}

/// Dimension of `e`: the largest of its leaves (witness/values by stage, challenges extension); memoised like `degree`.
pub fn dim(cs: &ConstraintSystem, e: &Expr) -> u32 {
    let mut witness = Vec::new();
    for c in &cs.witness {
        let i = c.id as usize;
        if witness.len() <= i {
            witness.resize(i + 1, 1);
        }
        witness[i] = c.dim;
    }
    dim_in(cs, e, &witness, &mut BTreeMap::new())
}

fn dim_in(cs: &ConstraintSystem, e: &Expr, witness: &[u32], memo: &mut BTreeMap<ExprId, u32>) -> u32 {
    let ext = cs.field.ext_dim() as u32;
    let by_stage = |s: Option<&NamedValue>| if s.is_some_and(|v| v.stage == 1) { 1 } else { ext };
    match e {
        Expr::Add(a, b) | Expr::Sub(a, b) | Expr::Mul(a, b) => {
            dim_in(cs, a, witness, memo).max(dim_in(cs, b, witness, memo))
        }
        Expr::Neg(a) => dim_in(cs, a, witness, memo),
        Expr::Col { col: ColType::Witness, id, .. } => witness.get(*id as usize).copied().unwrap_or(1),
        // fixed and custom columns are stage 0 (custom of another stage is rejected at ingestion)
        Expr::Col { .. } | Expr::Const(_) | Expr::Public(_) => 1,
        Expr::Challenge { .. } | Expr::AirgroupValue { airgroup: Some(_), .. } => ext,
        Expr::AirValue(i) => by_stage(cs.air_values.get(*i as usize)),
        Expr::ProofValue(i) => by_stage(cs.proof_values.get(*i as usize)),
        Expr::AirgroupValue { airgroup: None, id } => by_stage(cs.airgroup_values.get(*id as usize)),
        Expr::Ref(id) => {
            if let Some(d) = memo.get(id) {
                return *d;
            }
            memo.insert(*id, 1);
            let d = cs.exprs.get(*id as usize).map_or(1, |t| dim_in(cs, t, witness, memo));
            memo.insert(*id, d);
            d
        }
    }
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

/// Reference evaluator over the field (tests only); a cyclic `Ref` reads as 0.
pub fn eval_expr(cs: &ConstraintSystem, e: &Expr, env: &Env) -> Fe {
    eval_in(cs, e, env, &mut BTreeMap::new())
}

fn eval_in(cs: &ConstraintSystem, e: &Expr, env: &Env, memo: &mut BTreeMap<ExprId, Fe>) -> Fe {
    let p = cs.field.modulus();
    let get = |m: &BTreeMap<u32, Fe>, k: &u32| m.get(k).cloned().unwrap_or_default();
    match e {
        Expr::Add(a, b) => (eval_in(cs, a, env, memo) + eval_in(cs, b, env, memo)) % p,
        Expr::Sub(a, b) => (eval_in(cs, a, env, memo) + p - eval_in(cs, b, env, memo) % p) % p,
        Expr::Mul(a, b) => (eval_in(cs, a, env, memo) * eval_in(cs, b, env, memo)) % p,
        Expr::Neg(a) => (p - eval_in(cs, a, env, memo) % p) % p,
        Expr::Const(c) => c % p,
        Expr::Col { col, commit_id, id, offset } => {
            let k = ColRef { col: *col, commit_id: *commit_id, id: *id, offset: *offset };
            env.cols.get(&k).cloned().unwrap_or_default()
        }
        Expr::Challenge { stage, stage_id } => env.challenges.get(&(*stage, *stage_id)).cloned().unwrap_or_default(),
        Expr::Public(i) => get(&env.publics, i),
        Expr::ProofValue(i) => get(&env.proof_values, i),
        Expr::AirValue(i) => get(&env.air_values, i),
        Expr::AirgroupValue { airgroup, id } => env.airgroup_values.get(&(*airgroup, *id)).cloned().unwrap_or_default(),
        Expr::Ref(id) => {
            if let Some(v) = memo.get(id) {
                return v.clone();
            }
            memo.insert(*id, Fe::default());
            let v = cs.exprs.get(*id as usize).map_or_else(Fe::default, |t| eval_in(cs, t, env, memo));
            memo.insert(*id, v.clone());
            v
        }
    }
}
