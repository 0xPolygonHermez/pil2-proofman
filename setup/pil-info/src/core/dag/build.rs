//! The DAG builder: hash-consed, post-order, constraint roots then `im_e - e` roots.

use std::collections::BTreeMap;

use crate::core::dag::format::{ColType, ConstraintDag, DagField, DagBoundary, DagError, DagNode, DagRoot};
use crate::core::global::GlobalSystem;
use crate::core::im::{im_column_id, ImChoice};
use crate::core::ir::{Boundary, ConstraintSystem, Expr, ExprId};
use crate::cfg::FieldCfg;

/// Walks the substituted roots with a memo per `ExprId` (never inlines a `Ref`); validates before returning.
pub fn build_dag(cs: &ConstraintSystem, im: &ImChoice) -> Result<ConstraintDag, DagError> {
    let promoted = im.exprs.iter().enumerate().map(|(k, &id)| (id, im_column_id(cs, k))).collect();
    let mut b = Builder::new(&cs.exprs, promoted, false);
    let mut roots = Vec::with_capacity(cs.constraints.len() + im.exprs.len());
    for c in &cs.constraints {
        let node = b.expr(&c.e)?;
        roots.push(DagRoot { node, boundary: dag_boundary(&c.boundary), label: c.label.clone() });
    }
    for (k, &id) in im.exprs.iter().enumerate() {
        let col = b.node(DagNode::Col { col: ColType::Witness, commit_id: None, id: im_column_id(cs, k), offset: 0 });
        // the body itself, not `Ref(id)`, which would read as the column
        let body = b.target(id)?;
        let e = b.expr(body)?;
        let node = b.node(DagNode::Sub { a: col, b: e });
        roots.push(DagRoot { node, boundary: DagBoundary::EveryRow, label: format!("im.{k} expression {id}") });
    }
    finish(b.nodes, roots, &cs.field)
}

/// The global constraints of `gs`, one `Global` root each (labelled with its pilout line), with the air builder's
/// memo, hash-consing and node order; `None` when there are none.
pub fn build_global_dag(gs: &GlobalSystem) -> Result<Option<ConstraintDag>, DagError> {
    if gs.constraints.is_empty() {
        return Ok(None);
    }
    let mut b = Builder::new(&gs.exprs, BTreeMap::new(), true);
    let mut roots = Vec::with_capacity(gs.constraints.len());
    for c in &gs.constraints {
        // the body, so that the root is the constraint's own expression and not a `Ref` to it
        let body = b.target(c.expr)?;
        let node = b.expr(body)?;
        roots.push(DagRoot { node, boundary: DagBoundary::Global, label: c.label.clone() });
    }
    finish(b.nodes, roots, &gs.field).map(Some)
}

fn finish(nodes: Vec<DagNode>, roots: Vec<DagRoot>, field: &FieldCfg) -> Result<ConstraintDag, DagError> {
    let dag = ConstraintDag {
        version: ConstraintDag::VERSION,
        field: DagField::from(field).name().to_string(),
        ext_dim: field.ext_dim() as u32,
        nodes,
        roots,
    };
    dag.validate(field)?;
    Ok(dag)
}

pub fn dag_boundary(b: &Boundary) -> DagBoundary {
    match *b {
        Boundary::EveryRow => DagBoundary::EveryRow,
        Boundary::FirstRow => DagBoundary::FirstRow,
        Boundary::LastRow => DagBoundary::LastRow,
        Boundary::EveryFrame { min, max } => DagBoundary::EveryFrame { min: min.into(), max: max.into() },
    }
}

struct Builder<'a> {
    exprs: &'a [Expr],
    /// Promoted expression -> its im column id.
    promoted: BTreeMap<ExprId, u32>,
    /// Building the global DAG: airgroup values name their airgroup, columns and air values are refused.
    global: bool,
    nodes: Vec<DagNode>,
    index: BTreeMap<DagNode, u32>,
    /// `None` while the expression is being walked (a cycle).
    memo: BTreeMap<ExprId, Option<u32>>,
}

impl<'a> Builder<'a> {
    fn new(exprs: &'a [Expr], promoted: BTreeMap<ExprId, u32>, global: bool) -> Self {
        Builder { exprs, promoted, global, nodes: Vec::new(), index: BTreeMap::new(), memo: BTreeMap::new() }
    }

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
        let exprs = self.exprs;
        exprs.get(id as usize).ok_or_else(|| unbuildable(format!("expression {id} out of range")))
    }

    /// Post-order over `e` with an explicit stack (the recursion's node order; depth is not bounded by the stack).
    fn expr(&mut self, e: &'a Expr) -> Result<u32, DagError> {
        enum Step<'e> {
            Enter(&'e Expr),
            Op(&'e Expr),
            Done(ExprId),
        }
        let (mut todo, mut vals) = (vec![Step::Enter(e)], Vec::new());
        let pop = |v: &mut Vec<u32>| v.pop().ok_or_else(|| unbuildable("the walk lost an operand".into()));
        while let Some(step) = todo.pop() {
            match step {
                Step::Enter(n @ (Expr::Add(a, b) | Expr::Sub(a, b) | Expr::Mul(a, b))) => {
                    todo.extend([Step::Op(n), Step::Enter(b), Step::Enter(a)]);
                }
                Step::Enter(n @ Expr::Neg(a)) => todo.extend([Step::Op(n), Step::Enter(a)]),
                Step::Enter(Expr::Ref(id)) => {
                    if let Some(&col) = self.promoted.get(id) {
                        vals.push(self.node(DagNode::Col {
                            col: ColType::Witness,
                            commit_id: None,
                            id: col,
                            offset: 0,
                        }));
                        continue;
                    }
                    match self.memo.get(id) {
                        Some(Some(n)) => vals.push(*n),
                        Some(None) => return Err(unbuildable(format!("expression {id} is cyclic"))),
                        None => {
                            self.memo.insert(*id, None);
                            todo.extend([Step::Done(*id), Step::Enter(self.target(*id)?)]);
                        }
                    }
                }
                Step::Enter(leaf) => {
                    let n = self.leaf(leaf)?;
                    vals.push(self.node(n));
                }
                Step::Op(Expr::Neg(_)) => {
                    let a = pop(&mut vals)?;
                    vals.push(self.node(DagNode::Neg { a }));
                }
                Step::Op(op) => {
                    let (b, a) = (pop(&mut vals)?, pop(&mut vals)?);
                    vals.push(self.node(match op {
                        Expr::Add(..) => DagNode::Add { a, b },
                        Expr::Sub(..) => DagNode::Sub { a, b },
                        _ => DagNode::Mul { a, b },
                    }));
                }
                // a walked, non-promoted `Ref` is memoised once
                Step::Done(id) => {
                    let n = *vals.last().ok_or_else(|| unbuildable("the walk lost an operand".into()))?;
                    self.memo.insert(id, Some(n));
                }
            }
        }
        pop(&mut vals)
    }

    fn leaf(&self, e: &Expr) -> Result<DagNode, DagError> {
        if self.global && matches!(e, Expr::Col { .. } | Expr::AirValue(_)) {
            return Err(unbuildable(format!("{e:?} in a global constraint")));
        }
        Ok(match e {
            Expr::Const(c) => DagNode::Const { value: c.to_string() },
            Expr::Col { col, commit_id, id, offset } => {
                DagNode::Col { col: *col, commit_id: *commit_id, id: *id, offset: *offset }
            }
            Expr::Challenge { stage, stage_id } => DagNode::Challenge { stage: *stage, stage_id: *stage_id },
            Expr::Public(id) => DagNode::Public { id: *id },
            Expr::ProofValue(id) => DagNode::ProofValue { id: *id },
            Expr::AirValue(id) => DagNode::AirValue { id: *id },
            Expr::AirgroupValue { airgroup, id } if airgroup.is_some() == self.global => {
                DagNode::AirgroupValue { airgroup_id: *airgroup, id: *id }
            }
            Expr::AirgroupValue { airgroup: Some(g), id } => {
                return Err(unbuildable(format!("airgroup value {id} of airgroup {g} in an air")))
            }
            Expr::AirgroupValue { airgroup: None, id } => {
                return Err(unbuildable(format!("airgroup value {id} without its airgroup in a global constraint")))
            }
            _ => return Err(unbuildable("an operation where a leaf was expected".into())),
        })
    }
}

fn unbuildable(what: String) -> DagError {
    DagError::Unbuildable { what }
}
