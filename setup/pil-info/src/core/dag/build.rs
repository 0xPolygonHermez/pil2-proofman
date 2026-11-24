//! The DAG builder: hash-consed, post-order, constraint roots then `im_e - e` roots.

use std::collections::BTreeMap;

use crate::core::dag::format::{field_name, ColType, ConstraintDag, DagBoundary, DagError, DagNode, DagRoot};
use crate::core::im::{im_column_id, ImChoice};
use crate::core::ir::{Boundary, ConstraintSystem, Expr, ExprId};

/// Walks the substituted roots with a memo per `ExprId` (never inlines a `Ref`); validates before returning.
pub fn build_dag(cs: &ConstraintSystem, im: &ImChoice) -> Result<ConstraintDag, DagError> {
    let promoted = im.exprs.iter().enumerate().map(|(k, &id)| (id, im_column_id(cs, k))).collect();
    let mut b = Builder { cs, promoted, nodes: Vec::new(), index: BTreeMap::new(), memo: BTreeMap::new() };
    let mut roots = Vec::with_capacity(cs.constraints.len() + im.exprs.len());
    for c in &cs.constraints {
        let node = b.expr(&c.e)?;
        roots.push(DagRoot { node, boundary: boundary(&c.boundary), label: c.label.clone() });
    }
    for (k, &id) in im.exprs.iter().enumerate() {
        let col = b.node(DagNode::Col { col: ColType::Witness, commit_id: None, id: im_column_id(cs, k), offset: 0 });
        // the body itself, not `Ref(id)`, which would read as the column
        let body = b.target(id)?;
        let e = b.expr(body)?;
        let node = b.node(DagNode::Sub { a: col, b: e });
        roots.push(DagRoot { node, boundary: DagBoundary::EveryRow, label: format!("im.{k} expression {id}") });
    }
    let dag = ConstraintDag {
        version: ConstraintDag::VERSION,
        field: field_name(&cs.field)?.to_string(),
        ext_dim: cs.field.ext_dim() as u32,
        nodes: b.nodes,
        roots,
    };
    dag.validate(&cs.field)?;
    Ok(dag)
}

fn boundary(b: &Boundary) -> DagBoundary {
    match *b {
        Boundary::EveryRow => DagBoundary::EveryRow,
        Boundary::FirstRow => DagBoundary::FirstRow,
        Boundary::LastRow => DagBoundary::LastRow,
        Boundary::EveryFrame { min, max } => DagBoundary::EveryFrame { min: min.into(), max: max.into() },
    }
}

struct Builder<'a> {
    cs: &'a ConstraintSystem,
    /// Promoted expression -> its im column id.
    promoted: BTreeMap<ExprId, u32>,
    nodes: Vec<DagNode>,
    index: BTreeMap<DagNode, u32>,
    /// `None` while the expression is being walked (a cycle).
    memo: BTreeMap<ExprId, Option<u32>>,
}

impl<'a> Builder<'a> {
    /// THE node constructor: sorts commutative children, then returns the existing equal node or appends.
    fn node(&mut self, n: DagNode) -> u32 {
        let n = match n {
            DagNode::Add { a, b } => DagNode::Add { a: a.min(b), b: a.max(b) },
            DagNode::Mul { a, b } => DagNode::Mul { a: a.min(b), b: a.max(b) },
            n => n,
        };
        if let Some(&i) = self.index.get(&n) {
            return i;
        }
        let i = self.nodes.len() as u32;
        self.nodes.push(n.clone());
        self.index.insert(n, i);
        i
    }

    fn target(&self, id: ExprId) -> Result<&'a Expr, DagError> {
        let cs = self.cs;
        cs.exprs.get(id as usize).ok_or_else(|| unbuildable(format!("expression {id} out of range")))
    }

    fn expr(&mut self, e: &Expr) -> Result<u32, DagError> {
        let leaf = match e {
            Expr::Add(a, b) | Expr::Sub(a, b) | Expr::Mul(a, b) => {
                let (a, b) = (self.expr(a)?, self.expr(b)?);
                return Ok(self.node(match e {
                    Expr::Add(..) => DagNode::Add { a, b },
                    Expr::Sub(..) => DagNode::Sub { a, b },
                    _ => DagNode::Mul { a, b },
                }));
            }
            Expr::Neg(a) => {
                let a = self.expr(a)?;
                return Ok(self.node(DagNode::Neg { a }));
            }
            Expr::Ref(id) => return self.reference(*id),
            Expr::Const(c) => DagNode::Const { value: (c % self.cs.field.modulus()).to_string() },
            Expr::Col { col, commit_id, id, offset } => {
                DagNode::Col { col: *col, commit_id: *commit_id, id: *id, offset: *offset }
            }
            Expr::Challenge { stage, stage_id } => DagNode::Challenge { stage: *stage, stage_id: *stage_id },
            Expr::Public(id) => DagNode::Public { id: *id },
            Expr::ProofValue(id) => DagNode::ProofValue { id: *id },
            Expr::AirValue(id) => DagNode::AirValue { id: *id },
            Expr::AirgroupValue { airgroup: None, id } => DagNode::AirgroupValue { id: *id },
            Expr::AirgroupValue { airgroup: Some(g), id } => {
                return Err(unbuildable(format!("airgroup value {id} of airgroup {g} in an air")))
            }
        };
        Ok(self.node(leaf))
    }

    /// A promoted `Ref` is its im column; any other is walked once and memoised.
    fn reference(&mut self, id: ExprId) -> Result<u32, DagError> {
        if let Some(&col) = self.promoted.get(&id) {
            return Ok(self.node(DagNode::Col { col: ColType::Witness, commit_id: None, id: col, offset: 0 }));
        }
        match self.memo.get(&id) {
            Some(Some(n)) => return Ok(*n),
            Some(None) => return Err(unbuildable(format!("expression {id} is cyclic"))),
            None => {}
        }
        self.memo.insert(id, None);
        let n = self.expr(self.target(id)?)?;
        self.memo.insert(id, Some(n));
        Ok(n)
    }
}

fn unbuildable(what: String) -> DagError {
    DagError::Unbuildable { what }
}
