//! DAG -> straight-line verifier code; `LowerCtx` is the only column/challenge/value resolver.

use alloc::collections::BTreeMap;
use alloc::format;
use alloc::string::String;
use alloc::vec;
use alloc::vec::Vec;
use core::cell::RefCell;
use core::ops::{Add, Mul, Neg, Sub};

use crate::format::{ColRef, ColType, ConstraintDag, DagBoundary, DagError, DagField, DagNode};
use crate::{fold_roots, value_dim};

/// What a DAG's references resolve against; `ev_map[k]` is evaluation `k`, `challenges[k]` is `(stage, stage_id)`.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
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

/// A resolved leaf: what lowering and evaluation read; `Zi` is the inverse zerofier of a boundary
/// (pilfflonk's `boundaryId`; 0, everyRow, for the STARK).
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
    Zi { boundary: u32 },
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

/// An evMap entry (`cm`, `const` or `custom` with its commit) as the column it opens; `None` for another type.
pub fn ev_col_ref(ty: &str, commit_id: Option<u32>, id: u32, offset: i64) -> Option<ColRef> {
    let (col, commit_id) = match ty {
        "cm" => (ColType::Witness, None),
        "const" => (ColType::Fixed, None),
        "custom" => (ColType::Custom, Some(commit_id?)),
        _ => return None,
    };
    Some(ColRef { col, commit_id, id, offset })
}

/// Index of the challenge named `std_vc`, which must be named exactly once.
fn std_vc_index<'a>(names: impl IntoIterator<Item = &'a str>) -> Result<usize, DagError> {
    let mut at = names.into_iter().enumerate().filter(|(_, n)| *n == "std_vc").map(|(i, _)| i);
    match (at.next(), at.next()) {
        (_, Some(_)) => Err(DagError::DuplicateKey { what: "challenge std_vc".into() }),
        (Some(i), None) => Ok(i),
        (None, None) => Err(DagError::Unresolved { node: 0, what: "no challenge std_vc".into() }),
    }
}

/// Stage of each air, proof and airgroup value, by map id.
pub type ValueStages = [Vec<u32>; 3];

impl LowerTables {
    /// The tables from typed maps: the evMap columns, the challenges `(stage, stageId, name)`, nPublics and the
    /// value stages over `field`.
    pub fn new(
        ev_map: Vec<ColRef>,
        challenges: &[(u32, u32, &str)],
        n_publics: u32,
        [air_values_stage, proof_values_stage, airgroup_values_stage]: ValueStages,
        field: impl Into<DagField>,
    ) -> Result<Self, DagError> {
        Ok(LowerTables {
            vc: std_vc_index(challenges.iter().map(|c| c.2))?,
            ev_map,
            challenges: challenges.iter().map(|c| (c.0, c.1)).collect(),
            n_publics,
            air_values_stage,
            proof_values_stage,
            airgroup_values_stage,
            ext_dim: field.into().ext_dim(),
        })
    }
}

impl VerifierCode {
    /// The one dimension rule of the lowered code's consumers: an operand has its kind's dimension (a temporary its
    /// definition's, a value 1 at stage 1), a result the max of its operands'; each temporary is written once, before
    /// it is read.
    pub fn check_dims(&self, ext_dim: usize) -> Result<(), DagError> {
        let mut defs: BTreeMap<u32, usize> = BTreeMap::new();
        for ins in &self.code {
            let bad = |what: String| DagError::Dimension { tmp: ins.dest, what };
            let n_src = if ins.op == CodeOp::Copy { 1 } else { 2 };
            if ins.src.len() != n_src {
                return Err(bad(format!("{:?} takes {n_src} operands, got {}", ins.op, ins.src.len())));
            }
            for s in &ins.src {
                let d = match s.operand {
                    Operand::Tmp(k) => {
                        *defs.get(&k).ok_or_else(|| bad(format!("reads tmp {k} before it is written")))?
                    }
                    Operand::Eval(_) | Operand::Challenge { .. } | Operand::Zi { .. } => ext_dim,
                    Operand::Number(_) | Operand::Public(_) => 1,
                    Operand::ProofValue { stage, .. }
                    | Operand::AirValue { stage, .. }
                    | Operand::AirgroupValue { stage, .. } => value_dim(stage, ext_dim),
                };
                if s.dim != d {
                    return Err(bad(format!("operand {:?} typed dim {} but has dim {d}", s.operand, s.dim)));
                }
            }
            let dim = ins.src.iter().map(|s| s.dim).max().unwrap_or(1);
            if ins.dim != dim {
                return Err(bad(format!("typed dim {} over operands of dim {dim}", ins.dim)));
            }
            if defs.insert(ins.dest, dim).is_some() {
                return Err(bad("assigned twice".into()));
            }
        }
        Ok(())
    }
}

impl LowerCtx {
    /// Rejects a repeated evMap entry or challenge.
    pub fn new(t: LowerTables) -> Result<Self, DagError> {
        fn index<K: Ord + Copy + core::fmt::Debug>(keys: &[K], what: &str) -> Result<BTreeMap<K, usize>, DagError> {
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
            Some(&stage) => Ok(value_dim(stage, t.ext_dim)),
            None => Err(DagError::Unresolved { node: i, what: format!("{what} {id}") }),
        };
        match n {
            DagNode::Const { .. } | DagNode::Public { .. } => Ok(1),
            DagNode::ProofValue { id } => by_stage(&t.proof_values_stage, id, "proof value"),
            DagNode::AirValue { id } => by_stage(&t.air_values_stage, id, "air value"),
            DagNode::AirgroupValue { airgroup_id: None, id } => {
                by_stage(&t.airgroup_values_stage, id, "airgroup value")
            }
            DagNode::AirgroupValue { airgroup_id: Some(g), id } => {
                Err(DagError::Unresolved { node: i, what: format!("airgroup value {id} of airgroup {g} in an air") })
            }
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
            DagNode::AirgroupValue { airgroup_id: None, id } => {
                Operand::AirgroupValue { id, stage: stage(&t.airgroup_values_stage, id, "airgroup value")? }
            }
            DagNode::AirgroupValue { airgroup_id: Some(g), id } => {
                return Err(miss(format!("airgroup value {id} of airgroup {g} in an air")))
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

/// How the roots combine into the quotient.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Fold<'a> {
    Stark,
    /// pilfflonk's: each non-everyRow root times its boundary's `Zi` (`boundaries` in `boundaryId` order).
    Shplonk {
        boundaries: &'a [DagBoundary],
    },
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

/// One instruction per operation node in node order (the only traversal); `val[i]` is node `i`'s value.
fn lower_nodes(dag: &ConstraintDag, ctx: &LowerCtx) -> Result<(Vec<Instr>, Vec<Src>), DagError> {
    if dag.ext_dim as usize != ctx.tables().ext_dim {
        return Err(DagError::NonCanonical { node: 0, why: "the DAG's extDim is not the lowering context's" });
    }
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
    Ok((code, val))
}

/// The node code and every STARK root as an operand, unfolded (the circom verifier folds them itself).
pub fn lower_roots(dag: &ConstraintDag, ctx: &LowerCtx) -> Result<(Vec<Instr>, Vec<Src>), DagError> {
    let (code, val) = lower_nodes(dag, ctx)?;
    let roots = stark_roots(dag)?
        .into_iter()
        .map(|node| val.get(node as usize).cloned().ok_or(DagError::ForwardRef { node }))
        .collect::<Result<Vec<_>, _>>()?;
    Ok((code, roots))
}

/// One instruction per operation node in node order, then the fold (`core::fold::fold_roots`) times `Zi`.
/// Shplonk first multiplies each non-everyRow root by its boundary's `Zi`.
pub fn lower_to_code(dag: &ConstraintDag, ctx: &LowerCtx, fold: Fold<'_>) -> Result<VerifierCode, DagError> {
    let ext = ctx.tables().ext_dim;
    let (roots, every_row) = match fold {
        Fold::Stark => (stark_roots(dag)?.into_iter().map(|n| (n, None)).collect(), 0),
        Fold::Shplonk { boundaries } => shplonk_roots(dag, boundaries)?,
    };
    let zi = |boundary: u32| Src { operand: Operand::Zi { boundary }, dim: ext };
    let (mut code, val) = lower_nodes(dag, ctx)?;
    let vc = Src { operand: ctx.vc_operand()?, dim: ext };
    let mut folded = Vec::with_capacity(roots.len());
    for (node, boundary) in roots {
        let v = val.get(node as usize).cloned().ok_or(DagError::ForwardRef { node })?;
        folded.push(match boundary {
            Some(b) => emit(&mut code, CodeOp::Mul, vec![v, zi(b)]),
            None if matches!(v.operand, Operand::Tmp(_)) => v,
            // a leaf root is copied into a temporary first, as the old code did
            None => emit(&mut code, CodeOp::Copy, vec![v]),
        });
    }
    let code = RefCell::new(code);
    let acc = fold_roots(
        folded,
        |acc| emit(&mut code.borrow_mut(), CodeOp::Mul, vec![vc.clone(), acc]),
        |acc, c| emit(&mut code.borrow_mut(), CodeOp::Add, vec![acc, c]),
    )
    .ok_or(DagError::Empty)?;
    let mut code = code.into_inner();
    emit(&mut code, CodeOp::Mul, vec![acc, zi(every_row)]);
    Ok(VerifierCode { tmp_used: code.len(), code })
}

/// Each root with its `boundaries` index unless everyRow, and everyRow's index (the final `Zi`).
type FoldRoots = (Vec<(u32, Option<u32>)>, u32);

fn shplonk_roots(dag: &ConstraintDag, boundaries: &[DagBoundary]) -> Result<FoldRoots, DagError> {
    let first = dag.roots.first().ok_or(DagError::Empty)?;
    let id = |node: u32, b: &DagBoundary| match boundaries.iter().position(|x| x == b) {
        Some(k) => Ok(k as u32),
        None => Err(DagError::Unresolved { node, what: format!("boundary {b:?}") }),
    };
    let every_row = id(first.node, &DagBoundary::EveryRow)?;
    let roots = dag
        .roots
        .iter()
        .map(|r| match r.boundary {
            DagBoundary::EveryRow => Ok((r.node, None)),
            ref b => Ok((r.node, Some(id(r.node, b)?))),
        })
        .collect::<Result<_, DagError>>()?;
    Ok((roots, every_row))
}

/// Appends `tmp[k] = op(src)` with `k` the next temporary; returns it as an operand.
fn emit(code: &mut Vec<Instr>, op: CodeOp, src: Vec<Src>) -> Src {
    let dest = code.len() as u32;
    let dim = src.iter().map(|s| s.dim).max().unwrap_or(1);
    code.push(Instr { op, dest, dim, src });
    Src { operand: Operand::Tmp(dest), dim }
}

/// An air's DAG with every leaf resolved once and mapped by its consumer (say, a constant to a field element): what a
/// verifier evaluates at each proof without parsing, checking or resolving again.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ResolvedDag<L> {
    nodes: Vec<Resolved<L>>,
    roots: Vec<u32>,
    vc: L,
}

#[derive(Clone, Debug, PartialEq, Eq)]
enum Resolved<L> {
    Add(u32, u32),
    Sub(u32, u32),
    Mul(u32, u32),
    Neg(u32),
    Leaf(L),
}

impl LowerCtx {
    /// `dag` (its roots everyRow) with each leaf resolved and `std_vc`, then mapped by `leaf`.
    pub fn resolve_dag<L>(
        &self,
        dag: &ConstraintDag,
        mut leaf: impl FnMut(u32, Operand) -> Result<L, DagError>,
    ) -> Result<ResolvedDag<L>, DagError> {
        if dag.ext_dim as usize != self.t.ext_dim {
            return Err(DagError::NonCanonical { node: 0, why: "the DAG's extDim is not the lowering context's" });
        }
        let roots = stark_roots(dag)?;
        let mut nodes = Vec::with_capacity(dag.nodes.len());
        for (i, n) in dag.nodes.iter().enumerate() {
            let i = i as u32;
            nodes.push(match (self.resolve(i, n)?, n) {
                (Some(op), _) => Resolved::Leaf(leaf(i, op)?),
                (None, &DagNode::Add { a, b }) => Resolved::Add(a, b),
                (None, &DagNode::Sub { a, b }) => Resolved::Sub(a, b),
                (None, &DagNode::Mul { a, b }) => Resolved::Mul(a, b),
                (None, &DagNode::Neg { a }) => Resolved::Neg(a),
                (None, _) => return Err(DagError::NonCanonical { node: i, why: "a leaf did not resolve" }),
            });
        }
        let vc = leaf(0, self.vc_operand()?)?;
        Ok(ResolvedDag { nodes, roots, vc })
    }
}

impl<L> ResolvedDag<L> {
    /// `fold_roots` of the roots' values with `std_vc`, `value` reading each leaf; `None` when a leaf has none.
    pub fn eval<T>(&self, mut value: impl FnMut(&L) -> Option<T>) -> Option<T>
    where
        T: Copy + Add<Output = T> + Sub<Output = T> + Mul<Output = T> + Neg<Output = T>,
    {
        let v = self.node_values(&mut value)?;
        let vc = value(&self.vc)?;
        fold_roots(self.roots.iter().map(|&r| v[r as usize]), |acc| vc * acc, |acc, c| acc + c)
    }

    /// Each root's value, in root order, unfolded; `None` when a leaf has none.
    pub fn eval_roots<T>(&self, mut value: impl FnMut(&L) -> Option<T>) -> Option<Vec<T>>
    where
        T: Copy + Add<Output = T> + Sub<Output = T> + Mul<Output = T> + Neg<Output = T>,
    {
        let v = self.node_values(&mut value)?;
        Some(self.roots.iter().map(|&r| v[r as usize]).collect())
    }

    /// The DAG was validated (children precede their parents, roots in range), so indexing cannot fail.
    fn node_values<T>(&self, value: &mut impl FnMut(&L) -> Option<T>) -> Option<Vec<T>>
    where
        T: Copy + Add<Output = T> + Sub<Output = T> + Mul<Output = T> + Neg<Output = T>,
    {
        let mut v: Vec<T> = Vec::with_capacity(self.nodes.len());
        for n in &self.nodes {
            let x = match *n {
                Resolved::Add(a, b) => v[a as usize] + v[b as usize],
                Resolved::Sub(a, b) => v[a as usize] - v[b as usize],
                Resolved::Mul(a, b) => v[a as usize] * v[b as usize],
                Resolved::Neg(a) => -v[a as usize],
                Resolved::Leaf(ref l) => value(l)?,
            };
            v.push(x);
        }
        Some(v)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Two roots over a public and std_vc: each root's value, and the fold is Horner over them.
    #[test]
    fn eval_roots_are_what_eval_folds() {
        let t = LowerTables { challenges: vec![(3, 0)], vc: 0, n_publics: 1, ext_dim: 3, ..Default::default() };
        let root = |node| crate::format::DagRoot { node, boundary: DagBoundary::EveryRow, label: String::new() };
        let dag = ConstraintDag {
            version: ConstraintDag::VERSION,
            field: "gl".into(),
            ext_dim: 3,
            nodes: vec![
                DagNode::Public { id: 0 },
                DagNode::Const { value: "5".into() },
                DagNode::Mul { a: 0, b: 1 },
                DagNode::Neg { a: 0 },
            ],
            roots: vec![root(2), root(3)],
        };
        let r = LowerCtx::new(t).unwrap().resolve_dag(&dag, |_, op| Ok(op)).unwrap();
        let value = |op: &Operand| match op {
            Operand::Public(0) => Some(7i64),
            Operand::Number(v) => v.parse().ok(),
            Operand::Challenge { .. } => Some(10),
            _ => None,
        };
        assert_eq!(r.eval_roots(value), Some(vec![35, -7]));
        assert_eq!(r.eval(value), Some(10 * 35 - 7));
    }
}
