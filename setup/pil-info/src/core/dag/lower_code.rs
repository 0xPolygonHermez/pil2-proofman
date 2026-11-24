//! DAG -> straight-line verifier code; `LowerCtx` is the only column/challenge/value resolver.

use std::cell::RefCell;
use std::collections::BTreeMap;

use crate::core::dag::format::{ConstraintDag, DagBoundary, DagError, DagNode};
use crate::core::fold::fold_roots;
use crate::core::ir::ColRef;

/// What a DAG's references resolve against; `ev_map[k]` is evaluation `k`, `challenges[k]` is `(stage, stage_id)`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LowerTables {
    pub ev_map: Vec<ColRef>,
    pub challenges: Vec<(u32, u32)>,
    /// Index of `std_vc` in `challenges`.
    pub vc: usize,
    pub n_publics: u32,
    /// Stage of each value, indexed by its map id.
    pub air_values_stage: Vec<u32>,
    pub proof_values_stage: Vec<u32>,
    pub airgroup_values_stage: Vec<u32>,
    pub ext_dim: usize,
}

/// `LowerTables` with unique keys, indexed: the only resolver.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LowerCtx {
    t: LowerTables,
    ev: BTreeMap<ColRef, usize>,
    challenges: BTreeMap<(u32, u32), usize>,
}

/// A resolved leaf: what lowering and evaluation read.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Operand {
    Tmp(u32),
    Eval(u32),
    Challenge { id: u32, stage: u32, stage_id: u32 },
    Number(String),
    Public(u32),
    ProofValue { id: u32, stage: u32 },
    AirValue { id: u32, stage: u32 },
    AirgroupValue { id: u32, stage: u32 },
    Zi,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CodeOp {
    Add,
    Sub,
    Mul,
    Copy,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Src {
    pub operand: Operand,
    pub dim: usize,
}

/// `tmp[dest] = op(src..)`, of dimension `dim`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Instr {
    pub op: CodeOp,
    pub dest: u32,
    pub dim: usize,
    pub src: Vec<Src>,
}

/// The core's verifier code (qVerifier vocabulary); `emit::verifier` converts it to the JSON entry.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VerifierCode {
    pub tmp_used: usize,
    pub code: Vec<Instr>,
}

impl LowerCtx {
    /// Rejects a repeated evMap entry or challenge.
    pub fn new(t: LowerTables) -> Result<Self, DagError> {
        fn index<K: Ord + Copy + std::fmt::Debug>(keys: &[K], what: &str) -> Result<BTreeMap<K, usize>, DagError> {
            let mut m = BTreeMap::new();
            for (i, k) in keys.iter().enumerate() {
                if m.insert(*k, i).is_some() {
                    return Err(DagError::DuplicateKey { what: format!("{what} {k:?}") });
                }
            }
            Ok(m)
        }
        let ev = index(&t.ev_map, "evMap entry")?;
        let challenges = index(&t.challenges, "challenge")?;
        Ok(LowerCtx { t, ev, challenges })
    }

    pub fn tables(&self) -> &LowerTables {
        &self.t
    }

    pub fn ev_index(&self, k: &ColRef) -> Option<usize> {
        self.ev.get(k).copied()
    }

    pub fn challenge_index(&self, stage: u32, stage_id: u32) -> Option<usize> {
        self.challenges.get(&(stage, stage_id)).copied()
    }

    /// Dimension of leaf `n` (at index `i`): 1 for const/public, by stage for values, ext for the rest.
    pub fn dim(&self, i: u32, n: &DagNode) -> Result<usize, DagError> {
        let t = &self.t;
        let by_stage = |s: &[u32], id: &u32, what: &str| match s.get(*id as usize) {
            Some(1) => Ok(1),
            Some(_) => Ok(t.ext_dim),
            None => Err(DagError::Unresolved { node: i, what: format!("{what} {id}") }),
        };
        match n {
            DagNode::Const { .. } | DagNode::Public { .. } => Ok(1),
            DagNode::ProofValue { id } => by_stage(&t.proof_values_stage, id, "proof value"),
            DagNode::AirValue { id } => by_stage(&t.air_values_stage, id, "air value"),
            DagNode::AirgroupValue { id } => by_stage(&t.airgroup_values_stage, id, "airgroup value"),
            _ => Ok(t.ext_dim),
        }
    }

    /// THE resolver: leaf `node` (at index `i`) as an operand; `None` for an operation node.
    pub fn resolve(&self, i: u32, n: &DagNode) -> Result<Option<Operand>, DagError> {
        let (miss, t) = (|what: String| DagError::Unresolved { node: i, what }, &self.t);
        let stage =
            |s: &[u32], id: u32, what: &str| s.get(id as usize).copied().ok_or_else(|| miss(format!("{what} {id}")));
        let op = match *n {
            DagNode::Add { .. } | DagNode::Sub { .. } | DagNode::Mul { .. } | DagNode::Neg { .. } => return Ok(None),
            DagNode::Const { ref value } => Operand::Number(value.clone()),
            DagNode::Col { col, commit_id, id, offset } => {
                let k = self.ev_index(&ColRef { col, commit_id, id, offset });
                Operand::Eval(k.ok_or_else(|| miss(format!("{col:?} column {id} at offset {offset}")))? as u32)
            }
            DagNode::Challenge { stage, stage_id } => {
                let k = self.challenge_index(stage, stage_id);
                let id = k.ok_or_else(|| miss(format!("challenge {stage}.{stage_id}")))? as u32;
                Operand::Challenge { id, stage, stage_id }
            }
            DagNode::Public { id } if id < t.n_publics => Operand::Public(id),
            DagNode::Public { id } => return Err(miss(format!("public {id}"))),
            DagNode::ProofValue { id } => {
                Operand::ProofValue { id, stage: stage(&t.proof_values_stage, id, "proof value")? }
            }
            DagNode::AirValue { id } => Operand::AirValue { id, stage: stage(&t.air_values_stage, id, "air value")? },
            DagNode::AirgroupValue { id } => {
                Operand::AirgroupValue { id, stage: stage(&t.airgroup_values_stage, id, "airgroup value")? }
            }
        };
        Ok(Some(op))
    }

    /// `std_vc` as an operand.
    pub fn vc_operand(&self) -> Result<Operand, DagError> {
        let vc = self.t.vc;
        let &(stage, stage_id) =
            self.t.challenges.get(vc).ok_or(DagError::Unresolved { node: 0, what: "std_vc".into() })?;
        Ok(Operand::Challenge { id: vc as u32, stage, stage_id })
    }
}

/// How the roots combine into the quotient; piece 3 adds Shplonk.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Fold {
    Stark,
}

/// The root nodes in order; a STARK root must be everyRow.
pub fn stark_roots(dag: &ConstraintDag) -> Result<Vec<u32>, DagError> {
    if dag.roots.is_empty() {
        return Err(DagError::Empty);
    }
    dag.roots
        .iter()
        .map(|r| match r.boundary {
            DagBoundary::EveryRow => Ok(r.node),
            _ => Err(DagError::NonCanonical { node: r.node, why: "a STARK root is everyRow" }),
        })
        .collect()
}

/// One instruction per operation node in node order, then the fold (`core::fold::fold_roots`) times `Zi`.
pub fn lower_to_code(dag: &ConstraintDag, ctx: &LowerCtx, fold: Fold) -> Result<VerifierCode, DagError> {
    let Fold::Stark = fold;
    let ext = ctx.tables().ext_dim;
    let mut code: Vec<Instr> = Vec::new();
    let mut val: Vec<Src> = Vec::with_capacity(dag.nodes.len());
    for (i, n) in dag.nodes.iter().enumerate() {
        let i = i as u32;
        if let Some(operand) = ctx.resolve(i, n)? {
            val.push(Src { operand, dim: ctx.dim(i, n)? });
            continue;
        }
        let get = |c: u32| val.get(c as usize).cloned().ok_or(DagError::ForwardRef { node: i });
        let (op, src) = match *n {
            DagNode::Add { a, b } => (CodeOp::Add, vec![get(a)?, get(b)?]),
            DagNode::Sub { a, b } => (CodeOp::Sub, vec![get(a)?, get(b)?]),
            DagNode::Mul { a, b } => (CodeOp::Mul, vec![get(a)?, get(b)?]),
            DagNode::Neg { a } => (CodeOp::Sub, vec![Src { operand: Operand::Number("0".into()), dim: 1 }, get(a)?]),
            _ => return Err(DagError::NonCanonical { node: i, why: "a leaf did not resolve" }),
        };
        val.push(emit(&mut code, op, src));
    }
    let vc = Src { operand: ctx.vc_operand()?, dim: ext };
    let mut roots = Vec::new();
    for node in stark_roots(dag)? {
        let v = val.get(node as usize).cloned().ok_or(DagError::ForwardRef { node })?;
        // a leaf root is copied into a temporary first, as the old code did
        roots.push(if matches!(v.operand, Operand::Tmp(_)) { v } else { emit(&mut code, CodeOp::Copy, vec![v]) });
    }
    let code = RefCell::new(code);
    let acc = fold_roots(
        roots,
        |acc| emit(&mut code.borrow_mut(), CodeOp::Mul, vec![vc.clone(), acc]),
        |acc, c| emit(&mut code.borrow_mut(), CodeOp::Add, vec![acc, c]),
    )
    .ok_or(DagError::Empty)?;
    let mut code = code.into_inner();
    emit(&mut code, CodeOp::Mul, vec![acc, Src { operand: Operand::Zi, dim: ext }]);
    Ok(VerifierCode { tmp_used: code.len(), code })
}

/// Appends `tmp[k] = op(src)` with `k` the next temporary; returns it as an operand.
fn emit(code: &mut Vec<Instr>, op: CodeOp, src: Vec<Src>) -> Src {
    let dest = code.len() as u32;
    let dim = src.iter().map(|s| s.dim).max().unwrap_or(1);
    code.push(Instr { op, dest, dim, src });
    Src { operand: Operand::Tmp(dest), dim }
}
