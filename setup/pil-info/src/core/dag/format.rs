//! Constraint DAG types, validation and JSON codec.

use std::collections::BTreeMap;
use std::fmt;

use num_bigint::BigUint;
use serde::de::{Error as _, IgnoredAny, SeqAccess, Visitor};
use serde::ser::{Error as _, SerializeSeq};
use serde::{Deserialize, Deserializer, Serialize, Serializer};

use crate::cfg::FieldCfg;
pub use crate::core::ir::ColType;

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum DagNode {
    Add { a: u32, b: u32 },
    Sub { a: u32, b: u32 },
    Mul { a: u32, b: u32 },
    Neg { a: u32 },
    Const { value: String },
    Col { col: ColType, commit_id: Option<u32>, id: u32, offset: i64 },
    Challenge { stage: u32, stage_id: u32 },
    Public { id: u32 },
    ProofValue { id: u32 },
    AirValue { id: u32 },
    AirgroupValue { id: u32 },
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum DagBoundary {
    EveryRow,
    FirstRow,
    LastRow,
    EveryFrame { min: i64, max: i64 },
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DagRoot {
    pub node: u32,
    pub boundary: DagBoundary,
    pub label: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ConstraintDag {
    pub version: u32,
    pub field: String,
    pub ext_dim: u32,
    pub nodes: Vec<DagNode>,
    pub roots: Vec<DagRoot>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum DagError {
    Empty,
    Version { found: u32 },
    ForwardRef { node: u32 },
    Duplicate { node: u32, of: u32 },
    Unreachable { node: u32 },
    ConstOutOfRange { node: u32 },
    NonCanonical { node: u32, why: &'static str },
    Unresolved { node: u32, what: String },
    // the builder met an expression it cannot place
    Unbuildable { what: String },
    // a resolver table repeats a key
    DuplicateKey { what: String },
}

impl ConstraintDag {
    pub const VERSION: u32 = 1;

    /// Every canonical-form rule except the node order.
    pub fn validate(&self, f: &FieldCfg) -> Result<(), DagError> {
        if self.version != Self::VERSION {
            return Err(DagError::Version { found: self.version });
        }
        if self.nodes.is_empty() || self.roots.is_empty() {
            return Err(DagError::Empty);
        }
        if self.field != field_name(f)? || self.ext_dim as usize != f.ext_dim() {
            return Err(DagError::NonCanonical { node: 0, why: "field or extDim is not the setup's" });
        }
        if u32::try_from(self.nodes.len()).is_err() {
            return Err(DagError::NonCanonical { node: u32::MAX, why: "more than 2^32 nodes" });
        }
        let mut seen: BTreeMap<&DagNode, u32> = BTreeMap::new();
        for (i, n) in self.nodes.iter().enumerate() {
            let i = i as u32; // < nodes.len(), checked above
            if n.children().iter().any(|&c| c >= i) {
                return Err(DagError::ForwardRef { node: i });
            }
            check_node(i, n, f)?;
            if let Some(&of) = seen.get(n) {
                return Err(DagError::Duplicate { node: i, of });
            }
            seen.insert(n, i);
        }
        let mut reached = vec![false; self.nodes.len()];
        for r in &self.roots {
            *reached.get_mut(r.node as usize).ok_or(DagError::ForwardRef { node: r.node })? = true;
        }
        // children precede parents, so one reverse sweep marks everything reachable
        for i in (0..self.nodes.len()).rev() {
            if reached[i] {
                for c in self.nodes[i].children() {
                    reached[c as usize] = true;
                }
            }
        }
        match reached.iter().position(|r| !r) {
            Some(i) => Err(DagError::Unreachable { node: i as u32 }), // < nodes.len()
            None => Ok(()),
        }
    }

    /// Number of operation nodes (add, sub, mul, neg).
    pub fn op_count(&self) -> usize {
        self.nodes.iter().filter(|n| !n.children().is_empty()).count()
    }
}

impl DagNode {
    /// The operand node indices (empty for a leaf).
    pub fn children(&self) -> Vec<u32> {
        match *self {
            DagNode::Add { a, b } | DagNode::Sub { a, b } | DagNode::Mul { a, b } => vec![a, b],
            DagNode::Neg { a } => vec![a],
            _ => Vec::new(),
        }
    }
}

/// The DAG's `field` key for a field configuration; a field the format does not name is an error.
pub fn field_name(f: &FieldCfg) -> Result<&'static str, DagError> {
    if *f == FieldCfg::goldilocks() {
        Ok("gl")
    } else if *f == FieldCfg::bn128() {
        Ok("bn128")
    } else {
        Err(DagError::NonCanonical { node: 0, why: "a field the DAG format does not name" })
    }
}

fn check_node(i: u32, n: &DagNode, f: &FieldCfg) -> Result<(), DagError> {
    let bad = |why| Err(DagError::NonCanonical { node: i, why });
    match n {
        DagNode::Add { a, b } | DagNode::Mul { a, b } if a > b => bad("commutative children are not sorted"),
        DagNode::Const { value } => {
            let digits = !value.is_empty() && value.bytes().all(|c| c.is_ascii_digit());
            if !digits || (value.len() > 1 && value.starts_with('0')) {
                return bad("constant is not a canonical decimal");
            }
            match BigUint::parse_bytes(value.as_bytes(), 10) {
                Some(v) if v < *f.modulus() => Ok(()),
                _ => Err(DagError::ConstOutOfRange { node: i }),
            }
        }
        DagNode::Col { col: ColType::Custom, commit_id: None, .. } => bad("custom column without commitId"),
        DagNode::Col { col: ColType::Witness | ColType::Fixed, commit_id: Some(_), .. } => {
            bad("commitId on a non-custom column")
        }
        _ => Ok(()),
    }
}

// ---- JSON: nodes and roots are arrays (see the DAG format spec) -------------------------------

const OP_ADD: u32 = 0;
const OP_SUB: u32 = 1;
const OP_MUL: u32 = 2;
const OP_NEG: u32 = 3;
const OP_CONST: u32 = 4;
const OP_COL: u32 = 5;
const OP_CHALLENGE: u32 = 6;
const OP_PUBLIC: u32 = 7;
const OP_PROOFVALUE: u32 = 8;
const OP_AIRVALUE: u32 = 9;
const OP_AIRGROUPVALUE: u32 = 10;

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct DagOut<'a> {
    version: u32,
    field: &'a str,
    ext_dim: u32,
    nodes: Vec<NodeOut<'a>>,
    roots: Vec<RootOut<'a>>,
}

struct NodeOut<'a>(&'a DagNode);
struct RootOut<'a>(&'a DagRoot);

impl Serialize for NodeOut<'_> {
    fn serialize<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        let mut q = s.serialize_seq(None)?;
        match self.0 {
            DagNode::Add { a, b } | DagNode::Sub { a, b } | DagNode::Mul { a, b } => {
                let op = match self.0 {
                    DagNode::Add { .. } => OP_ADD,
                    DagNode::Sub { .. } => OP_SUB,
                    _ => OP_MUL,
                };
                q.serialize_element(&op)?;
                q.serialize_element(a)?;
                q.serialize_element(b)?;
            }
            DagNode::Neg { a } => {
                q.serialize_element(&OP_NEG)?;
                q.serialize_element(a)?;
            }
            DagNode::Const { value } => {
                q.serialize_element(&OP_CONST)?;
                q.serialize_element(value)?;
            }
            DagNode::Col { col, commit_id, id, offset } => {
                q.serialize_element(&OP_COL)?;
                let t: u32 = match col {
                    ColType::Witness => 0,
                    ColType::Fixed => 1,
                    ColType::Custom => 2,
                };
                q.serialize_element(&t)?;
                q.serialize_element(id)?;
                q.serialize_element(offset)?;
                match (col, commit_id) {
                    (ColType::Custom, Some(c)) => q.serialize_element(c)?,
                    (ColType::Custom, None) | (_, Some(_)) => {
                        return Err(S::Error::custom("commitId is set exactly on custom columns"))
                    }
                    _ => {}
                }
            }
            DagNode::Challenge { stage, stage_id } => {
                q.serialize_element(&OP_CHALLENGE)?;
                q.serialize_element(stage)?;
                q.serialize_element(stage_id)?;
            }
            DagNode::Public { id }
            | DagNode::ProofValue { id }
            | DagNode::AirValue { id }
            | DagNode::AirgroupValue { id } => {
                let op = match self.0 {
                    DagNode::Public { .. } => OP_PUBLIC,
                    DagNode::ProofValue { .. } => OP_PROOFVALUE,
                    DagNode::AirValue { .. } => OP_AIRVALUE,
                    _ => OP_AIRGROUPVALUE,
                };
                q.serialize_element(&op)?;
                q.serialize_element(id)?;
            }
        }
        q.end()
    }
}

impl Serialize for RootOut<'_> {
    fn serialize<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        if self.0.boundary != DagBoundary::EveryRow {
            return Err(S::Error::custom("only everyRow roots are serialisable"));
        }
        (self.0.node, 0u32, &self.0.label).serialize(s)
    }
}

impl Serialize for ConstraintDag {
    fn serialize<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        DagOut {
            version: self.version,
            field: &self.field,
            ext_dim: self.ext_dim,
            nodes: self.nodes.iter().map(NodeOut).collect(),
            roots: self.roots.iter().map(RootOut).collect(),
        }
        .serialize(s)
    }
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct DagIn {
    version: u32,
    field: String,
    ext_dim: u32,
    nodes: Vec<NodeIn>,
    roots: Vec<(u32, u32, String)>,
}

struct NodeIn(DagNode);

impl<'de> Deserialize<'de> for NodeIn {
    fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        d.deserialize_seq(NodeVisitor)
    }
}

struct NodeVisitor;

impl<'de> Visitor<'de> for NodeVisitor {
    type Value = NodeIn;

    fn expecting(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.write_str("a constraint DAG node [opcode, ...fields]")
    }

    fn visit_seq<A: SeqAccess<'de>>(self, mut q: A) -> Result<NodeIn, A::Error> {
        fn next<'de, T: Deserialize<'de>, A: SeqAccess<'de>>(q: &mut A, what: &str) -> Result<T, A::Error> {
            q.next_element()?.ok_or_else(|| A::Error::custom(format!("node misses its {what}")))
        }
        let op: u32 = next(&mut q, "opcode")?;
        let node = match op {
            OP_ADD => DagNode::Add { a: next(&mut q, "a")?, b: next(&mut q, "b")? },
            OP_SUB => DagNode::Sub { a: next(&mut q, "a")?, b: next(&mut q, "b")? },
            OP_MUL => DagNode::Mul { a: next(&mut q, "a")?, b: next(&mut q, "b")? },
            OP_NEG => DagNode::Neg { a: next(&mut q, "a")? },
            OP_CONST => DagNode::Const { value: next(&mut q, "value")? },
            OP_COL => {
                let col = match next::<u32, _>(&mut q, "type")? {
                    0 => ColType::Witness,
                    1 => ColType::Fixed,
                    2 => ColType::Custom,
                    t => return Err(A::Error::custom(format!("unknown column type {t}"))),
                };
                let (id, offset) = (next(&mut q, "id")?, next(&mut q, "offset")?);
                let commit_id = if col == ColType::Custom { Some(next(&mut q, "commitId")?) } else { None };
                DagNode::Col { col, commit_id, id, offset }
            }
            OP_CHALLENGE => DagNode::Challenge { stage: next(&mut q, "stage")?, stage_id: next(&mut q, "stageId")? },
            OP_PUBLIC => DagNode::Public { id: next(&mut q, "id")? },
            OP_PROOFVALUE => DagNode::ProofValue { id: next(&mut q, "id")? },
            OP_AIRVALUE => DagNode::AirValue { id: next(&mut q, "id")? },
            OP_AIRGROUPVALUE => DagNode::AirgroupValue { id: next(&mut q, "id")? },
            _ => return Err(A::Error::custom(format!("unknown opcode {op}"))),
        };
        if q.next_element::<IgnoredAny>()?.is_some() {
            return Err(A::Error::custom(format!("opcode {op} has too many fields")));
        }
        Ok(NodeIn(node))
    }
}

impl<'de> Deserialize<'de> for ConstraintDag {
    fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        let raw = DagIn::deserialize(d)?;
        let roots = raw
            .roots
            .into_iter()
            .map(|(node, boundary, label)| match boundary {
                0 => Ok(DagRoot { node, boundary: DagBoundary::EveryRow, label }),
                b => Err(D::Error::custom(format!("unknown root boundary {b} (only 0 = everyRow)"))),
            })
            .collect::<Result<_, _>>()?;
        Ok(ConstraintDag {
            version: raw.version,
            field: raw.field,
            ext_dim: raw.ext_dim,
            nodes: raw.nodes.into_iter().map(|n| n.0).collect(),
            roots,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn gl() -> FieldCfg {
        FieldCfg::goldilocks()
    }
    const GL_P: &str = "18446744069414584321";

    fn col(id: u32, offset: i64) -> DagNode {
        DagNode::Col { col: ColType::Witness, commit_id: None, id, offset }
    }
    fn root(node: u32) -> DagRoot {
        DagRoot { node, boundary: DagBoundary::EveryRow, label: format!("r{node}") }
    }
    fn dag(nodes: Vec<DagNode>, roots: Vec<DagRoot>) -> ConstraintDag {
        ConstraintDag { version: 1, field: "gl".into(), ext_dim: 3, nodes, roots }
    }
    /// col0 * col1 as the single root: nodes 0,1 are leaves, node 2 the product.
    fn product() -> ConstraintDag {
        dag(vec![col(0, 0), col(1, 0), DagNode::Mul { a: 0, b: 1 }], vec![root(2)])
    }

    #[test]
    fn json_round_trip_all_kinds() {
        // the spec example parses to exactly these nodes and roots
        let spec = r#"{"version": 1, "field": "gl", "extDim": 3,
            "nodes": [[5,0,3,0], [6,3,0], [2,0,1], [5,0,4,0], [1,3,2]],
            "roots": [[2,0,"c0 line 12"], [4,0,"im.0 ImPol"]]}"#;
        let d: ConstraintDag = serde_json::from_str(spec).unwrap();
        assert_eq!(
            d.nodes,
            vec![
                col(3, 0),
                DagNode::Challenge { stage: 3, stage_id: 0 },
                DagNode::Mul { a: 0, b: 1 },
                col(4, 0),
                DagNode::Sub { a: 3, b: 2 },
            ]
        );
        assert_eq!((d.version, d.field.as_str(), d.ext_dim), (1, "gl", 3));
        let labels: Vec<(u32, &str)> = d.roots.iter().map(|r| (r.node, r.label.as_str())).collect();
        assert_eq!(labels, vec![(2, "c0 line 12"), (4, "im.0 ImPol")]);
        assert!(d.roots.iter().all(|r| r.boundary == DagBoundary::EveryRow));
        d.validate(&gl()).unwrap();

        // every node kind, with its normative array shape
        let all = dag(
            vec![
                col(1, -1),                                                                  // 0
                DagNode::Col { col: ColType::Fixed, commit_id: None, id: 2, offset: 1 },     // 1
                DagNode::Col { col: ColType::Custom, commit_id: Some(7), id: 3, offset: 0 }, // 2
                DagNode::Const { value: "42".into() },                                       // 3
                DagNode::Challenge { stage: 2, stage_id: 1 },                                // 4
                DagNode::Public { id: 5 },                                                   // 5
                DagNode::ProofValue { id: 6 },                                               // 6
                DagNode::AirValue { id: 7 },                                                 // 7
                DagNode::AirgroupValue { id: 8 },                                            // 8
                DagNode::Add { a: 0, b: 1 },                                                 // 9
                DagNode::Sub { a: 2, b: 3 },                                                 // 10
                DagNode::Mul { a: 4, b: 5 },                                                 // 11
                DagNode::Neg { a: 6 },                                                       // 12
                DagNode::Add { a: 7, b: 8 },                                                 // 13
            ],
            vec![root(9), root(10), root(11), root(12), root(13)],
        );
        let v = serde_json::to_value(&all).unwrap();
        let want = serde_json::json!({
            "version": 1, "field": "gl", "extDim": 3,
            "nodes": [[5,0,1,-1], [5,1,2,1], [5,2,3,0,7], [4,"42"], [6,2,1], [7,5], [8,6], [9,7], [10,8],
                      [0,0,1], [1,2,3], [2,4,5], [3,6], [0,7,8]],
            "roots": [[9,0,"r9"], [10,0,"r10"], [11,0,"r11"], [12,0,"r12"], [13,0,"r13"]],
        });
        assert_eq!(v, want);
        let back: ConstraintDag = serde_json::from_str(&serde_json::to_string(&all).unwrap()).unwrap();
        assert_eq!(back, all);
    }

    #[test]
    fn rejects_empty() {
        assert_eq!(dag(vec![], vec![]).validate(&gl()), Err(DagError::Empty));
        assert_eq!(dag(vec![col(0, 0)], vec![]).validate(&gl()), Err(DagError::Empty));
        product().validate(&gl()).unwrap();
    }

    #[test]
    fn rejects_wrong_version() {
        let mut d = product();
        d.version = 2;
        assert_eq!(d.validate(&gl()), Err(DagError::Version { found: 2 }));
        d.version = 0;
        assert_eq!(d.validate(&gl()), Err(DagError::Version { found: 0 }));
    }

    #[test]
    fn rejects_forward_reference() {
        // node 0 reads node 1; and a node reading itself
        let d = dag(vec![DagNode::Add { a: 0, b: 1 }, col(0, 0)], vec![root(0)]);
        assert_eq!(d.validate(&gl()), Err(DagError::ForwardRef { node: 0 }));
        let d = dag(vec![col(0, 0), DagNode::Neg { a: 1 }], vec![root(1)]);
        assert_eq!(d.validate(&gl()), Err(DagError::ForwardRef { node: 1 }));
        // a root past the end is not a valid index either
        let d = dag(vec![col(0, 0)], vec![root(5)]);
        assert!(d.validate(&gl()).is_err());
    }

    #[test]
    fn rejects_duplicate() {
        let d = dag(vec![col(0, 0), col(0, 0), DagNode::Add { a: 0, b: 1 }], vec![root(2)]);
        assert_eq!(d.validate(&gl()), Err(DagError::Duplicate { node: 1, of: 0 }));
    }

    #[test]
    fn rejects_unsorted_commutative() {
        for (bad, ok) in [
            (DagNode::Add { a: 1, b: 0 }, DagNode::Add { a: 0, b: 1 }),
            (DagNode::Mul { a: 1, b: 0 }, DagNode::Mul { a: 0, b: 1 }),
        ] {
            let d = dag(vec![col(0, 0), col(1, 0), bad], vec![root(2)]);
            assert!(matches!(d.validate(&gl()), Err(DagError::NonCanonical { node: 2, .. })), "{d:?}");
            dag(vec![col(0, 0), col(1, 0), ok], vec![root(2)]).validate(&gl()).unwrap();
        }
    }

    #[test]
    fn accepts_sub_with_swapped_operands() {
        let d = dag(vec![col(0, 0), col(1, 0), DagNode::Sub { a: 1, b: 0 }], vec![root(2)]);
        d.validate(&gl()).unwrap();
    }

    #[test]
    fn rejects_unreachable() {
        let d = dag(vec![col(0, 0), col(1, 0)], vec![root(0)]);
        assert_eq!(d.validate(&gl()), Err(DagError::Unreachable { node: 1 }));
        // reachable only through an unreachable parent is still unreachable
        let d = dag(vec![col(0, 0), col(1, 0), DagNode::Neg { a: 1 }, col(2, 0)], vec![root(0), root(3)]);
        assert!(matches!(d.validate(&gl()), Err(DagError::Unreachable { .. })));
    }

    #[test]
    fn rejects_noncanonical_const() {
        let one = |v: &str| dag(vec![DagNode::Const { value: v.into() }], vec![root(0)]);
        one("0").validate(&gl()).unwrap();
        one("7").validate(&gl()).unwrap();
        assert!(matches!(one("07").validate(&gl()), Err(DagError::NonCanonical { node: 0, .. })));
        assert!(matches!(one("00").validate(&gl()), Err(DagError::NonCanonical { node: 0, .. })));
        assert!(matches!(one("").validate(&gl()), Err(DagError::NonCanonical { node: 0, .. })));
        assert!(matches!(one("0x7").validate(&gl()), Err(DagError::NonCanonical { node: 0, .. })));
        assert!(matches!(one("-1").validate(&gl()), Err(DagError::NonCanonical { node: 0, .. })));
        // p itself and above is out of range over Goldilocks, p - 1 is the largest element
        assert_eq!(one(GL_P).validate(&gl()), Err(DagError::ConstOutOfRange { node: 0 }));
        one("18446744069414584320").validate(&gl()).unwrap();
        // the same value is an element of BN128
        let mut bn = one(GL_P);
        bn.field = "bn128".into();
        bn.ext_dim = 1;
        bn.validate(&FieldCfg::bn128()).unwrap();
    }

    #[test]
    fn rejects_commit_id_on_cm() {
        for t in [ColType::Witness, ColType::Fixed] {
            let d = dag(vec![DagNode::Col { col: t, commit_id: Some(0), id: 0, offset: 0 }], vec![root(0)]);
            assert!(matches!(d.validate(&gl()), Err(DagError::NonCanonical { node: 0, .. })), "{t:?}");
        }
        // a custom column needs its commit
        let d = dag(vec![DagNode::Col { col: ColType::Custom, commit_id: None, id: 0, offset: 0 }], vec![root(0)]);
        assert!(matches!(d.validate(&gl()), Err(DagError::NonCanonical { node: 0, .. })));
        let d = dag(vec![DagNode::Col { col: ColType::Custom, commit_id: Some(1), id: 0, offset: 0 }], vec![root(0)]);
        d.validate(&gl()).unwrap();
    }

    #[test]
    fn rejects_unknown_opcode_or_arity() {
        let with = |nodes: &str| {
            let j = format!(r#"{{"version":1,"field":"gl","extDim":3,"nodes":{nodes},"roots":[[0,0,"r"]]}}"#);
            serde_json::from_str::<ConstraintDag>(&j)
        };
        with("[[5,0,0,0]]").unwrap();
        assert!(with("[[11,0]]").is_err(), "unknown opcode");
        assert!(with("[[99]]").is_err(), "unknown opcode");
        assert!(with("[[]]").is_err(), "no opcode");
        assert!(with("[[0,1]]").is_err(), "add needs two operands");
        assert!(with("[[0,0,0,0]]").is_err(), "add takes two operands");
        assert!(with("[[3,0,1]]").is_err(), "neg takes one operand");
        assert!(with("[[5,0,3]]").is_err(), "col needs an offset");
        assert!(with("[[5,0,3,0,7]]").is_err(), "a cm column has no commit id");
        assert!(with("[[5,2,3,0]]").is_err(), "a custom column needs its commit id");
        assert!(with("[[5,3,0,0]]").is_err(), "unknown column type");
        assert!(with("[[4,7]]").is_err(), "a constant is a decimal string");
        assert!(with("[[6,1]]").is_err(), "challenge needs two fields");
    }

    #[test]
    fn op_count_counts_operation_nodes() {
        assert_eq!(product().op_count(), 1);
        let spec = dag(
            vec![col(3, 0), DagNode::Challenge { stage: 3, stage_id: 0 }, DagNode::Mul { a: 0, b: 1 }, col(4, 0)]
                .into_iter()
                .chain([DagNode::Sub { a: 3, b: 2 }])
                .collect(),
            vec![root(2), root(4)],
        );
        assert_eq!(spec.op_count(), 2);
    }
}
