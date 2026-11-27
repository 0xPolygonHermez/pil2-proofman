//! What a global DAG's leaves read: the one resolver of the global verifiers (circom `GlobalConstraints`, C++
//! `verifyGlobalConstraints`).
//!
//! Every value is an extension element; a stage-1 proof or airgroup value is a base element (its first limb, the
//! others read as zero), as in an air. Challenges are numbered across stages: `(stage, stage_id)` is
//! `sum(num_challenges[..stage - 1]) + stage_id`.

use crate::core::dag::format::{ConstraintDag, DagError, DagNode};
use crate::core::global::GlobalSystem;
use crate::core::ir::value_dim;

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct GlobalTables {
    pub n_publics: u32,
    /// Challenges per stage, stage 1 first.
    pub num_challenges: Vec<u32>,
    /// Stage of each proof value, by id.
    pub proof_values_stage: Vec<u32>,
    /// Stage of each airgroup's values, by airgroup and id.
    pub airgroup_values_stage: Vec<Vec<u32>>,
    pub ext_dim: usize,
}

/// A resolved global leaf.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum GlobalLeaf {
    Number(String),
    Public(u32),
    /// The flat challenge index.
    Challenge(u32),
    ProofValue {
        id: u32,
        stage: u32,
    },
    AirgroupValue {
        airgroup: u32,
        id: u32,
        stage: u32,
    },
}

impl GlobalTables {
    pub fn of(gs: &GlobalSystem) -> Self {
        GlobalTables {
            n_publics: gs.num_publics,
            num_challenges: gs.num_challenges.clone(),
            proof_values_stage: gs.proof_value_stages.clone(),
            airgroup_values_stage: gs.airgroup_values.iter().map(|g| g.iter().map(|v| v.stage).collect()).collect(),
            ext_dim: gs.field.ext_dim(),
        }
    }

    /// Leaf `n` (at index `i`) with its references checked; `None` for an operation node.
    pub fn resolve(&self, i: u32, n: &DagNode) -> Result<Option<GlobalLeaf>, DagError> {
        let miss = |what: String| DagError::Unresolved { node: i, what };
        Ok(Some(match *n {
            DagNode::Add { .. } | DagNode::Sub { .. } | DagNode::Mul { .. } | DagNode::Neg { .. } => return Ok(None),
            DagNode::Const { ref value } => GlobalLeaf::Number(value.clone()),
            DagNode::Public { id } if id < self.n_publics => GlobalLeaf::Public(id),
            DagNode::Challenge { stage, stage_id } => {
                let k = stage.checked_sub(1).map(|k| k as usize);
                match k.and_then(|k| self.num_challenges.get(k).map(|&n| (k, n))) {
                    Some((k, n)) if stage_id < n => {
                        GlobalLeaf::Challenge(self.num_challenges[..k].iter().sum::<u32>() + stage_id)
                    }
                    _ => return Err(miss(format!("challenge {stage}.{stage_id}"))),
                }
            }
            DagNode::ProofValue { id } => match self.proof_values_stage.get(id as usize) {
                Some(&stage) => GlobalLeaf::ProofValue { id, stage },
                None => return Err(miss(format!("proof value {id}"))),
            },
            DagNode::AirgroupValue { airgroup_id: Some(g), id } => {
                match self.airgroup_values_stage.get(g as usize).and_then(|v| v.get(id as usize)) {
                    Some(&stage) => GlobalLeaf::AirgroupValue { airgroup: g, id, stage },
                    None => return Err(miss(format!("airgroup value {id} of airgroup {g}"))),
                }
            }
            ref n => return Err(miss(format!("{n:?} in a global DAG"))),
        }))
    }

    /// Dimension of a resolved leaf: 1 for a number, a public or a stage-1 value, the extension's otherwise.
    pub fn dim(&self, leaf: &GlobalLeaf) -> usize {
        match *leaf {
            GlobalLeaf::Number(_) | GlobalLeaf::Public(_) => 1,
            GlobalLeaf::Challenge(_) => self.ext_dim,
            GlobalLeaf::ProofValue { stage, .. } | GlobalLeaf::AirgroupValue { stage, .. } => {
                value_dim(stage, self.ext_dim)
            }
        }
    }

    /// Every leaf of a global `dag` resolved (index = node; `None` for an operation node), the DAG checked first.
    pub fn resolve_all(&self, dag: &ConstraintDag) -> Result<Vec<Option<GlobalLeaf>>, DagError> {
        if !dag.is_global() {
            return Err(DagError::NonCanonical { node: 0, why: "not a global DAG" });
        }
        if dag.ext_dim as usize != self.ext_dim {
            return Err(DagError::NonCanonical { node: 0, why: "the DAG's extDim is not the tables'" });
        }
        dag.nodes.iter().enumerate().map(|(i, n)| self.resolve(i as u32, n)).collect()
    }
}
