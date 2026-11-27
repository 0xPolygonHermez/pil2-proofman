//! The air's constraints as the setup's `constraintDag` (verifierinfo.json), checked and resolved once, then
//! evaluated at xi: Horner(roots, std_vc) * Zi, as C++ `evalConstraintDag`.
//!
//! The checks are pil-info's `ConstraintDag::from_verifierinfo` (format, canonical form, reachability) and the
//! resolution is its `LowerCtx` (every column, challenge and public found exactly once). This verifier reads no
//! proof, air or airgroup value: a DAG with one is refused.

use alloc::collections::{BTreeMap, BTreeSet};
use alloc::string::String;
use alloc::vec::Vec;

use proofman_fields::{CubicExtensionField, Field, Goldilocks, PrimeField64};
use serde::de::IgnoredAny;
use serde::Deserialize;

type E3 = CubicExtensionField<Goldilocks>;

/// An evMap entry: column type (0 witness, 1 fixed, 2 custom), id, row offset and custom commit (0 otherwise).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct EvKey {
    pub col: u8,
    pub id: u64,
    pub offset: i64,
    pub commit_id: u64,
}

/// A challengesMap entry.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ChallengeKey {
    pub stage: u64,
    pub stage_id: u64,
    pub name: String,
}

/// What the DAG's leaves resolve against: the starkinfo's evMap, challengesMap and nPublics.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct DagTables {
    pub ev_map: Vec<EvKey>,
    pub challenges: Vec<ChallengeKey>,
    pub n_publics: u64,
}

/// A resolved node: an operation over earlier nodes or a leaf's slot.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Node {
    Add(u32, u32),
    Sub(u32, u32),
    Mul(u32, u32),
    Neg(u32),
    Number(u64),
    Eval(u32),
    Challenge(u32),
    Public(u32),
}

/// A checked and resolved constraint DAG; only `from_json` builds one.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ConstraintDag {
    nodes: Vec<Node>,
    roots: Vec<u32>,
    /// `std_vc`'s index in the challenges.
    vc: u32,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct DagJson {
    version: u32,
    field: String,
    ext_dim: u32,
    nodes: Vec<Vec<FieldJson>>,
    roots: Vec<(u32, u32, IgnoredAny)>,
}

#[derive(Deserialize)]
#[serde(untagged)]
enum FieldJson {
    Int(i64),
    Str(String),
}

/// The node as its format reads it (before resolution): the key of the uniqueness check.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum Raw {
    Add(u32, u32),
    Sub(u32, u32),
    Mul(u32, u32),
    Neg(u32),
    Const(u64),
    Col(EvKey),
    Challenge(u64, u64),
    Public(u64),
    Value(i64, u64),
}

const OP_COL: i64 = 5;
const OP_AIRGROUPVALUE: i64 = 10;
const CUSTOM: i64 = 2;

impl Raw {
    fn children(&self) -> [Option<u32>; 2] {
        match *self {
            Raw::Add(a, b) | Raw::Sub(a, b) | Raw::Mul(a, b) => [Some(a), Some(b)],
            Raw::Neg(a) => [Some(a), None],
            _ => [None, None],
        }
    }
}

fn parse_node(f: &[FieldJson]) -> Result<Raw, &'static str> {
    let int = |k: usize| match f.get(k) {
        Some(FieldJson::Int(v)) => Ok(*v),
        _ => Err("a node field is not an integer"),
    };
    let unsigned = |k: usize| int(k).and_then(|v| u64::try_from(v).map_err(|_| "a negative index"));
    let node = |k: usize| unsigned(k).and_then(|v| u32::try_from(v).map_err(|_| "a node index past 2^32"));
    let op = int(0)?;
    let arity = match op {
        0..=2 | 6 => 3,
        3 | 4 | 7..=9 => 2,
        OP_COL if int(1)? == CUSTOM => 5,
        OP_COL => 4,
        // the third field (the airgroup) is a global DAG's
        OP_AIRGROUPVALUE if f.len() == 3 => return Err("an airgroup value naming its airgroup in an air"),
        OP_AIRGROUPVALUE => 2,
        _ => return Err("unknown opcode"),
    };
    if f.len() != arity {
        return Err("wrong number of fields for its opcode");
    }
    Ok(match op {
        0 => Raw::Add(node(1)?, node(2)?),
        1 => Raw::Sub(node(1)?, node(2)?),
        2 => Raw::Mul(node(1)?, node(2)?),
        3 => Raw::Neg(node(1)?),
        4 => match &f[1] {
            FieldJson::Str(s) => Raw::Const(canonical(s)?),
            _ => return Err("a constant is not a decimal string"),
        },
        OP_COL => {
            let col = u8::try_from(unsigned(1)?).ok().filter(|&t| t <= 2).ok_or("unknown column type")?;
            let commit_id = if col == 2 { unsigned(4)? } else { 0 };
            Raw::Col(EvKey { col, id: unsigned(2)?, offset: int(3)?, commit_id })
        }
        6 => Raw::Challenge(unsigned(1)?, unsigned(2)?),
        7 => Raw::Public(unsigned(1)?),
        _ => Raw::Value(op, unsigned(1)?),
    })
}

/// A canonical decimal (digits, no leading zero) below the Goldilocks prime.
fn canonical(s: &str) -> Result<u64, &'static str> {
    if s.is_empty() || !s.bytes().all(|c| c.is_ascii_digit()) || (s.len() > 1 && s.starts_with('0')) {
        return Err("a constant is not a canonical decimal");
    }
    s.parse::<u64>().ok().filter(|&v| v < Goldilocks::ORDER_U64).ok_or("a constant out of range")
}

impl ConstraintDag {
    /// The DAG of `json` (verifierinfo's `constraintDag`), checked and resolved against `t`.
    pub fn from_json(json: &str, t: &DagTables) -> Result<Self, &'static str> {
        let d: DagJson = serde_json::from_str(json).map_err(|_| "constraintDag is not the DAG format")?;
        if d.version != 1 {
            return Err("unsupported DAG version");
        }
        if d.field != "gl" || d.ext_dim != 3 {
            return Err("field must be gl with extDim 3");
        }
        if d.nodes.is_empty() || d.roots.is_empty() {
            return Err("no nodes or no roots");
        }
        if u32::try_from(d.nodes.len()).is_err() {
            return Err("more than 2^32 nodes");
        }

        let mut ev: BTreeMap<EvKey, u32> = BTreeMap::new();
        for (k, e) in t.ev_map.iter().enumerate() {
            if ev.insert(*e, k as u32).is_some() {
                return Err("a repeated evMap entry");
            }
        }
        let mut challenges: BTreeMap<(u64, u64), u32> = BTreeMap::new();
        for (k, c) in t.challenges.iter().enumerate() {
            if challenges.insert((c.stage, c.stage_id), k as u32).is_some() {
                return Err("a repeated challenge");
            }
        }
        let mut vc = t.challenges.iter().enumerate().filter(|(_, c)| c.name == "std_vc").map(|(k, _)| k as u32);
        let vc = match (vc.next(), vc.next()) {
            (Some(k), None) => k,
            _ => return Err("std_vc is not named exactly once"),
        };

        let mut seen: BTreeSet<Raw> = BTreeSet::new();
        let mut nodes = Vec::with_capacity(d.nodes.len());
        for (i, f) in d.nodes.iter().enumerate() {
            let raw = parse_node(f)?;
            if raw.children().iter().flatten().any(|&c| c as usize >= i) {
                return Err("a forward or self reference");
            }
            if let Raw::Add(a, b) | Raw::Mul(a, b) = raw {
                if a > b {
                    return Err("commutative children are not sorted");
                }
            }
            nodes.push(match raw {
                Raw::Add(a, b) => Node::Add(a, b),
                Raw::Sub(a, b) => Node::Sub(a, b),
                Raw::Mul(a, b) => Node::Mul(a, b),
                Raw::Neg(a) => Node::Neg(a),
                Raw::Const(v) => Node::Number(v),
                Raw::Col(k) => Node::Eval(*ev.get(&k).ok_or("a column the evMap does not open")?),
                Raw::Challenge(s, id) => Node::Challenge(*challenges.get(&(s, id)).ok_or("an unknown challenge")?),
                Raw::Public(id) if id < t.n_publics => Node::Public(id as u32),
                Raw::Public(_) => return Err("a public out of range"),
                Raw::Value(..) => return Err("a proof, air or airgroup value: this verifier reads none"),
            });
            if !seen.insert(raw) {
                return Err("a repeated node");
            }
        }

        let mut roots = Vec::with_capacity(d.roots.len());
        for &(node, boundary, _) in &d.roots {
            if boundary != 0 {
                return Err("a root that is not everyRow");
            }
            if node as usize >= nodes.len() {
                return Err("a root past the nodes");
            }
            roots.push(node);
        }
        // children precede parents, so one reverse sweep marks everything reachable
        let mut reached = alloc::vec![false; nodes.len()];
        for &r in &roots {
            reached[r as usize] = true;
        }
        for i in (0..nodes.len()).rev() {
            if reached[i] {
                if let Node::Add(a, b) | Node::Sub(a, b) | Node::Mul(a, b) = nodes[i] {
                    reached[a as usize] = true;
                    reached[b as usize] = true;
                } else if let Node::Neg(a) = nodes[i] {
                    reached[a as usize] = true;
                }
            }
        }
        if reached.contains(&false) {
            return Err("a node no root reaches");
        }
        Ok(ConstraintDag { nodes, roots, vc })
    }

    /// Horner(roots, challenges[std_vc]) * zi; `None` when an input is shorter than a leaf reads.
    pub fn eval(&self, challenges: &[E3], evals: &[E3], publics: &[Goldilocks], zi: E3) -> Option<E3> {
        let embed = |g: Goldilocks| E3 { value: [g, Goldilocks::ZERO, Goldilocks::ZERO] };
        let mut v: Vec<E3> = Vec::with_capacity(self.nodes.len());
        for n in &self.nodes {
            let x = match *n {
                Node::Add(a, b) => v[a as usize] + v[b as usize],
                Node::Sub(a, b) => v[a as usize] - v[b as usize],
                Node::Mul(a, b) => v[a as usize] * v[b as usize],
                Node::Neg(a) => -v[a as usize],
                Node::Number(c) => embed(Goldilocks::new(c)),
                Node::Eval(k) => *evals.get(k as usize)?,
                Node::Challenge(k) => *challenges.get(k as usize)?,
                Node::Public(k) => embed(*publics.get(k as usize)?),
            };
            v.push(x);
        }
        let vc = *challenges.get(self.vc as usize)?;
        let mut acc = v[self.roots[0] as usize];
        for &r in &self.roots[1..] {
            acc = vc * acc + v[r as usize];
        }
        Some(acc * zi)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::format;
    use alloc::string::ToString;

    /// evMap: witness 0 at rows 0 and -1, fixed 0; challenges 2.0 (std_alpha) and 3.0 (std_vc); one public.
    fn tables() -> DagTables {
        let ev = |col, offset| EvKey { col, id: 0, offset, commit_id: 0 };
        let ch = |stage, name: &str| ChallengeKey { stage, stage_id: 0, name: name.to_string() };
        DagTables {
            ev_map: alloc::vec![ev(0, 0), ev(0, -1), ev(1, 0)],
            challenges: alloc::vec![ch(2, "std_alpha"), ch(3, "std_vc")],
            n_publics: 1,
        }
    }

    fn json(nodes: &str, roots: &str) -> String {
        format!(r#"{{"version":1,"field":"gl","extDim":3,"nodes":{nodes},"roots":{roots}}}"#)
    }

    fn e3(a: u64, b: u64, c: u64) -> E3 {
        E3 { value: [Goldilocks::new(a), Goldilocks::new(b), Goldilocks::new(c)] }
    }

    /// c0 = w0 * alpha - 7, c1 = w0' + fixed0 - public0: (vc * c0 + c1) * zi.
    #[test]
    fn horner_of_the_roots_times_zi() {
        let nodes = r#"[[5,0,0,0],[6,2,0],[2,0,1],[4,"7"],[1,2,3],[5,0,0,-1],[5,1,0,0],[0,5,6],[7,0],[1,7,8]]"#;
        let dag = ConstraintDag::from_json(&json(nodes, r#"[[4,0,"c0"],[9,0,"c1"]]"#), &tables()).unwrap();
        let (evals, challenges) = ([e3(1, 2, 3), e3(4, 5, 6), e3(7, 8, 9)], [e3(10, 11, 12), e3(13, 14, 15)]);
        let (publics, zi) = ([Goldilocks::new(16)], e3(17, 18, 19));
        let embed = |v| e3(v, 0, 0);
        let c0 = evals[0] * challenges[0] - embed(7);
        let c1 = evals[1] + evals[2] - embed(16);
        assert_eq!(dag.eval(&challenges, &evals, &publics, zi), Some((challenges[1] * c0 + c1) * zi));
        assert_eq!(dag.eval(&challenges, &evals[..2], &publics, zi), None, "evals shorter than the DAG reads");
        assert_eq!(dag.eval(&challenges, &evals, &[], zi), None, "no publics");
    }

    #[test]
    fn refusals() {
        let refused = |nodes: &str, roots: &str| ConstraintDag::from_json(&json(nodes, roots), &tables()).unwrap_err();
        let r0 = r#"[[0,0,"r"]]"#;
        assert_eq!(
            refused(r#"[[5,0,0,0],[5,0,0,-1],[0,1,0]]"#, r#"[[2,0,"r"]]"#),
            "commutative children are not sorted"
        );
        assert_eq!(refused(r#"[[3,0]]"#, r0), "a forward or self reference");
        assert_eq!(refused(r#"[[5,0,0,0],[5,0,0,0],[0,0,1]]"#, r#"[[2,0,"r"]]"#), "a repeated node");
        assert_eq!(refused(r#"[[5,0,0,0],[7,0]]"#, r0), "a node no root reaches");
        assert_eq!(refused(r#"[[4,"07"]]"#, r0), "a constant is not a canonical decimal");
        assert_eq!(refused(r#"[[4,"18446744069414584321"]]"#, r0), "a constant out of range");
        assert_eq!(refused(r#"[[7,0]]"#, r#"[[0,1,"r"]]"#), "a root that is not everyRow");
        assert_eq!(refused(r#"[[10,0,1]]"#, r0), "an airgroup value naming its airgroup in an air");
        for value in ["[[8,0]]", "[[9,0]]", "[[10,0]]"] {
            assert_eq!(refused(value, r0), "a proof, air or airgroup value: this verifier reads none");
        }
        assert_eq!(refused(r#"[[5,0,1,0]]"#, r0), "a column the evMap does not open");
        assert_eq!(refused(r#"[[6,2,1]]"#, r0), "an unknown challenge");
        assert_eq!(refused(r#"[[7,1]]"#, r0), "a public out of range");
        assert_eq!(refused(r#"[[11,0]]"#, r0), "unknown opcode");
        assert_eq!(refused(r#"[[5,0,0]]"#, r0), "wrong number of fields for its opcode");
        assert_eq!(refused(r#"[[7,0]]"#, r#"[[1,0,"r"]]"#), "a root past the nodes");
        let mut t = tables();
        t.challenges[0].name = "std_vc".to_string();
        assert_eq!(ConstraintDag::from_json(&json("[[7,0]]", r0), &t).unwrap_err(), "std_vc is not named exactly once");
        let other = r#"{"version":2,"field":"gl","extDim":3,"nodes":[[7,0]],"roots":[[0,0,"r"]]}"#;
        assert_eq!(ConstraintDag::from_json(other, &tables()).unwrap_err(), "unsupported DAG version");
    }
}
