//! The global constraints of a vadcop_final: `GlobalConstraints(nAg, maxAgg, nPub, nPV, nCh)` and its
//! `GlobalConstraint{c}_chunk{i}` parts (≤ 50 instructions each, to keep circom compile times down),
//! over the padded `airgroupvalues[nAg][maxAgg][3]` that `VadcopFinal` (final_vadcop.circom) builds.

use std::collections::{HashMap, HashSet};

use anyhow::{bail, Context, Result};
use serde::Deserialize;
use serde_json::Value;

use super::parse_num_proof_values;

const MIN_CHUNK_SIZE: usize = 50;

/// The inputs every chunk and `GlobalConstraints` take, in call order.
const INPUTS: &str = "    signal input airgroupvalues[nAg][maxAgg][3];
    signal input publics[nPub];
    signal input proofValues[nPV][3];
    signal input challenges[nCh][3];
";
const PARAMS: &str = "nAg, maxAgg, nPub, nPV, nCh";

/// One global constraint: its instructions; the last one's dest must be 0.
#[derive(Deserialize)]
struct GlobalConstraint {
    code: Vec<Inst>,
}

#[derive(Deserialize)]
struct Inst {
    op: Op,
    dest: Ref,
    src: Vec<Ref>,
}

#[derive(Deserialize, Clone, Copy)]
#[serde(rename_all = "lowercase")]
enum Op {
    Add,
    Sub,
    Mul,
    Copy,
}

#[derive(Deserialize)]
struct Ref {
    dim: u64,
    #[serde(flatten)]
    kind: Kind,
}

#[derive(Deserialize)]
#[serde(tag = "type", rename_all = "lowercase")]
enum Kind {
    Public {
        id: u64,
    },
    Proofvalue {
        id: u64,
    },
    Tmp {
        id: u64,
    },
    Number {
        value: Number,
    },
    Challenge {
        id: u64,
    },
    Airgroupvalue {
        #[serde(rename = "airgroupId")]
        airgroup_id: u64,
        id: u64,
    },
}

#[derive(Deserialize)]
#[serde(untagged)]
enum Number {
    Str(String),
    Int(u64),
}

/// The sizes the references are checked against.
struct Shape {
    n_agg: Vec<u64>,
    n_publics: u64,
    n_proof_values: u64,
    n_challenges: u64,
}

impl Ref {
    fn tmp(&self) -> Option<u64> {
        match self.kind {
            Kind::Tmp { id } => Some(id),
            _ => None,
        }
    }

    /// A circom expression; `decl` declares a tmp dest the first time it is written.
    fn expr(&self, decl: bool) -> String {
        let idx = if self.dim == 1 { "[0]" } else { "" };
        match &self.kind {
            Kind::Public { id } => format!("publics[{id}]"),
            Kind::Proofvalue { id } => format!("proofValues[{id}]{idx}"),
            Kind::Challenge { id } => format!("challenges[{id}]{idx}"),
            Kind::Airgroupvalue { airgroup_id, id } => format!("airgroupvalues[{airgroup_id}][{id}]{idx}"),
            Kind::Tmp { id } if decl => format!("signal tmp_{id}{}", if self.dim == 1 { "" } else { "[3]" }),
            Kind::Tmp { id } => format!("tmp_{id}"),
            Kind::Number { value: Number::Int(n) } => n.to_string(),
            Kind::Number { value: Number::Str(s) } => s.clone(),
        }
    }

    /// Refuses what would not be the value the constraint names: a bad dim, a number that is not a
    /// decimal literal, an index past its array (an airgroup value past its airgroup reads padding).
    fn check(&self, s: &Shape) -> Result<()> {
        if self.dim != 1 && self.dim != 3 {
            bail!("dim {} is neither 1 nor 3", self.dim);
        }
        let (id, n, what) = match &self.kind {
            Kind::Public { id } => (*id, s.n_publics, "publics"),
            Kind::Proofvalue { id } => (*id, s.n_proof_values, "proofValues"),
            Kind::Challenge { id } => (*id, s.n_challenges, "stage-2 challenges"),
            Kind::Airgroupvalue { airgroup_id, id } => {
                let n = s.n_agg.get(*airgroup_id as usize).with_context(|| format!("no airgroup {airgroup_id}"))?;
                (*id, *n, "airgroupvalues")
            }
            Kind::Number { value: Number::Str(v) } if v.is_empty() || !v.bytes().all(|b| b.is_ascii_digit()) => {
                bail!("number {v:?} is not a decimal literal")
            }
            Kind::Tmp { .. } | Kind::Number { .. } => return Ok(()),
        };
        if id >= n {
            bail!("{what}[{id}] is out of range ({n})");
        }
        Ok(())
    }
}

/// The circom line of one instruction.
fn line(inst: &Inst, declared: &mut HashSet<u64>) -> Result<String> {
    let [a, b] = match (inst.op, inst.src.as_slice()) {
        (Op::Copy, [a]) => return Ok(format!("    {} <== {};\n", dest(&inst.dest, declared), a.expr(false))),
        (Op::Add | Op::Sub | Op::Mul, [a, b]) => [a, b],
        _ => bail!("an instruction has {} sources", inst.src.len()),
    };
    let (x, y) = (a.expr(false), b.expr(false));
    let rhs = match (inst.op, a.dim, b.dim) {
        (Op::Add, 1, 1) => format!("{x} + {y}"),
        (Op::Add, 1, 3) => format!("[{x} + {y}[0], {y}[1], {y}[2]]"),
        (Op::Add, 3, 1) => format!("[{x}[0] + {y}, {x}[1], {x}[2]]"),
        (Op::Add, 3, 3) => format!("[{x}[0] + {y}[0], {x}[1] + {y}[1], {x}[2] + {y}[2]]"),
        (Op::Sub, 1, 1) => format!("{x} - {y}"),
        (Op::Sub, 1, 3) => format!("[{x} - {y}[0], -{y}[1], -{y}[2]]"),
        (Op::Sub, 3, 1) => format!("[{x}[0] - {y}, {x}[1], {x}[2]]"),
        (Op::Sub, 3, 3) => format!("[{x}[0] - {y}[0], {x}[1] - {y}[1], {x}[2] - {y}[2]]"),
        (Op::Mul, 1, 1) => format!("{x} * {y}"),
        (Op::Mul, 1, 3) => format!("[{x} * {y}[0], {x} * {y}[1], {x} * {y}[2]]"),
        (Op::Mul, 3, 1) => format!("[{x}[0] * {y}, {x}[1] * {y}, {x}[2] * {y}]"),
        (Op::Mul, 3, 3) => format!("CMul()({x}, {y})"),
        _ => unreachable!("dims are checked"),
    };
    let d = dest(&inst.dest, declared);
    Ok(format!("    {d} <== {rhs};\n"))
}

fn dest(r: &Ref, declared: &mut HashSet<u64>) -> String {
    r.expr(r.tmp().is_some_and(|id| declared.insert(id)))
}

struct Chunk<'a> {
    code: &'a [Inst],
    inputs: Vec<u64>,
    outputs: Vec<u64>,
}

/// `code` in chunks of < MIN_CHUNK_SIZE instructions, with the tmps that cross into (inputs) and out of
/// (outputs) each.
fn split_chunks(code: &[Inst]) -> Vec<Chunk<'_>> {
    let mut last_use: HashMap<u64, usize> = HashMap::new();
    for (i, inst) in code.iter().enumerate() {
        for id in std::iter::once(&inst.dest).chain(&inst.src).filter_map(Ref::tmp) {
            last_use.insert(id, i);
        }
    }
    let mut chunks = Vec::new();
    let mut start = 0;
    let (mut live, mut prev_live) = (HashSet::new(), HashSet::new());
    let (mut inputs, mut outputs) = (HashSet::new(), HashSet::new());
    let sorted = |s: &HashSet<u64>| {
        let mut v: Vec<u64> = s.iter().copied().collect();
        v.sort();
        v
    };
    for (i, inst) in code.iter().enumerate() {
        if let Some(id) = inst.dest.tmp() {
            live.insert(id);
            outputs.insert(id);
        }
        for id in inst.src.iter().filter_map(Ref::tmp) {
            let last = last_use.get(&id) == Some(&i);
            if last {
                live.remove(&id);
                outputs.remove(&id);
            }
            if prev_live.contains(&id) {
                inputs.insert(id);
                if last {
                    prev_live.remove(&id);
                }
            }
        }
        if i + 2 - start >= MIN_CHUNK_SIZE || i + 1 == code.len() {
            chunks.push(Chunk { code: &code[start..=i], inputs: sorted(&inputs), outputs: sorted(&outputs) });
            start = i + 1;
            prev_live.extend(live.drain());
            outputs.clear();
            inputs.clear();
        }
    }
    chunks
}

/// The global constraints of globalInfo `vi` (with its `globalConstraints`, absent meaning none), and
/// `vd_nGlobalChallenges()`, the stage-2 challenges `VadcopFinal` draws for them.
pub fn gen_verify_global_constraints(vi: &Value) -> Result<String> {
    let constraints: Vec<GlobalConstraint> = match vi.get("globalConstraints") {
        Some(c) => serde_json::from_value(c.clone()).context("globalConstraints")?,
        None => Vec::new(),
    };
    let agg_types = vi["aggTypes"].as_array().context("globalInfo has no aggTypes")?;
    let shape = Shape {
        n_agg: agg_types.iter().map(|a| a.as_array().map_or(0, Vec::len) as u64).collect(),
        n_publics: vi["nPublics"].as_u64().context("globalInfo has no nPublics")?,
        n_proof_values: parse_num_proof_values(&vi["numProofValues"]) as u64,
        // numChallenges[1]: stage 2's; a reference past it is refused.
        n_challenges: vi["numChallenges"].get(1).and_then(Value::as_u64).unwrap_or(0),
    };

    let mut out = format!("function vd_nGlobalChallenges() {{ return {}; }}\n\n", shape.n_challenges);
    let mut main = String::new();
    for (c, gc) in constraints.iter().enumerate() {
        let ctx = || format!("global constraint {c}");
        let last = gc.code.last().with_context(|| format!("{}: no instructions", ctx()))?;
        let mut dims: HashMap<u64, u64> = HashMap::new();
        for r in gc.code.iter().flat_map(|i| std::iter::once(&i.dest).chain(&i.src)) {
            r.check(&shape).with_context(ctx)?;
        }
        for inst in &gc.code {
            if let Some(id) = inst.dest.tmp() {
                dims.insert(id, inst.dest.dim);
            }
        }
        let sig = |io: &str, id: &u64| {
            format!("    signal {io}tmp_{id}{};\n", if dims.get(id) == Some(&1) { "" } else { "[3]" })
        };
        for (ci, chunk) in split_chunks(&gc.code).iter().enumerate() {
            out += &format!("template GlobalConstraint{c}_chunk{ci}({PARAMS}) {{\n{INPUTS}");
            out += &chunk.inputs.iter().map(|id| sig("input ", id)).collect::<String>();
            out += &chunk.outputs.iter().map(|id| sig("output ", id)).collect::<String>();
            let mut declared: HashSet<u64> = chunk.inputs.iter().chain(&chunk.outputs).copied().collect();
            for inst in chunk.code {
                out += &line(inst, &mut declared).with_context(ctx)?;
            }
            out += "}\n";

            main += &chunk.outputs.iter().map(|id| sig("", id)).collect::<String>();
            let args: Vec<String> = ["airgroupvalues", "publics", "proofValues", "challenges"]
                .iter()
                .map(|s| s.to_string())
                .chain(chunk.inputs.iter().map(|id| format!("tmp_{id}")))
                .collect();
            let outs: Vec<String> = chunk.outputs.iter().map(|id| format!("tmp_{id}")).collect();
            main += &format!(
                "    ({}) <== GlobalConstraint{c}_chunk{ci}({PARAMS})({});\n",
                outs.join(", "),
                args.join(", ")
            );
        }
        let id = last.dest.tmp().with_context(|| format!("{}: its last instruction writes no tmp", ctx()))?;
        if last.dest.dim == 1 {
            main += &format!("    tmp_{id} === 0;\n");
        } else {
            main += &format!("    tmp_{id}[0] === 0;\n    tmp_{id}[1] === 0;\n    tmp_{id}[2] === 0;\n");
        }
    }
    Ok(out + &format!("template GlobalConstraints({PARAMS}) {{\n{INPUTS}{main}}}\n"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn tmp(id: u64, dim: u64) -> Value {
        json!({"type": "tmp", "id": id, "dim": dim})
    }

    fn info(constraints: Value) -> Value {
        json!({"aggTypes": [[{"aggType": 0}], [{"aggType": 0}, {"aggType": 1}]], "nPublics": 2,
               "numProofValues": [1], "numChallenges": [0, 2], "globalConstraints": constraints})
    }

    /// 60 dim-3 instructions chain across two chunks; a dim-1 constraint checks one `=== 0`.
    #[test]
    fn chunks_chain_and_the_last_dest_is_zero() {
        let mut code = vec![json!({"op": "mul", "dest": tmp(0, 3), "src": [
            {"type": "airgroupvalue", "airgroupId": 1, "id": 1, "dim": 3}, {"type": "challenge", "id": 1, "dim": 3}]})];
        for i in 1..60u64 {
            let (op, src) = (["add", "sub", "mul"][i as usize % 3], json!({"type": "public", "id": i % 2, "dim": 1}));
            code.push(json!({"op": op, "dest": tmp(i, 3), "src": [tmp(i - 1, 3), src]}));
        }
        let binary = json!([
            {"op": "sub", "dest": tmp(0, 1), "src": [{"type": "number", "value": "1", "dim": 1}, {"type": "proofvalue", "id": 0, "dim": 1}]},
            {"op": "mul", "dest": tmp(1, 1), "src": [{"type": "proofvalue", "id": 0, "dim": 1}, tmp(0, 1)]}
        ]);
        let out = gen_verify_global_constraints(&info(json!([{"code": code}, {"code": binary}]))).unwrap();
        for l in [
            "function vd_nGlobalChallenges() { return 2; }",
            "    signal tmp_0[3] <== CMul()(airgroupvalues[1][1], challenges[1]);",
            "template GlobalConstraint0_chunk1(nAg, maxAgg, nPub, nPV, nCh) {",
            "    signal input tmp_48[3];",
            "    (tmp_48) <== GlobalConstraint0_chunk0(nAg, maxAgg, nPub, nPV, nCh)(airgroupvalues, publics, proofValues, challenges);",
            "    (tmp_59) <== GlobalConstraint0_chunk1(nAg, maxAgg, nPub, nPV, nCh)(airgroupvalues, publics, proofValues, challenges, tmp_48);",
            "    tmp_59[0] === 0;\n    tmp_59[1] === 0;\n    tmp_59[2] === 0;\n",
            "    signal tmp_0 <== 1 - proofValues[0][0];",
            "    tmp_1 <== proofValues[0][0] * tmp_0;\n}",
            "    tmp_1 === 0;\n}",
        ] {
            assert!(out.contains(l), "missing `{l}`:\n{out}");
        }
        assert!(!out.contains("GlobalConstraint0_chunk2"), "{out}");
    }

    #[test]
    fn refuses_references_past_their_arrays() {
        let refused = |r: Value| {
            let c = json!([{"code": [{"op": "copy", "dest": tmp(0, 3), "src": [r]}]}]);
            gen_verify_global_constraints(&info(c)).unwrap_err().root_cause().to_string()
        };
        assert!(refused(json!({"type": "airgroupvalue", "airgroupId": 0, "id": 1, "dim": 3})).contains("out of range"));
        assert!(refused(json!({"type": "challenge", "id": 2, "dim": 3})).contains("out of range"));
        assert!(refused(json!({"type": "number", "value": "1); x <== (2", "dim": 1})).contains("decimal"));
        assert!(refused(json!({"type": "nope", "id": 0, "dim": 1})).contains("nope"));
    }
}
