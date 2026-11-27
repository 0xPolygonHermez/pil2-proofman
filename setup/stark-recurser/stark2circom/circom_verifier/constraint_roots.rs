//! The air's constraints at xi from the constraint DAG, one signal per operation node. GL:
//! `ConstraintRoots<suffix>`, one root each (`lower_roots`); the fold, Zi and the Q check are the
//! library's (`verifier/stark_verifier.circom`). BN128: `ConstraintEval`, the roots folded with
//! `lower_to_code` in lazily reduced `GL*` templates (`verifier/stark_verifier_bn128.circom`).
//! Global constraints: `GlobalConstraints` of vadcop_final from the global DAG, each root `=== 0`.

use std::collections::BTreeSet;

use anyhow::{anyhow, bail, Result};
use pil_info::core::dag::format::{ConstraintDag, DagNode};
use pil_info::core::dag::global::{GlobalLeaf, GlobalTables};
use pil_info::core::dag::lower_code::{
    lower_roots, lower_to_code, CodeOp, Fold, Instr, LowerCtx, Operand, Src, VerifierCode,
};

// ── Goldilocks: `ConstraintRoots` ─────────────────────────────────────────────

/// The generated template plus what the params record about it.
pub(crate) struct ConstraintRoots {
    pub template: String,
    pub n_roots: usize,
    pub first_root_dim: usize,
}

/// A value-map entry: limb 0 at stage 1 (base field), all three after.
fn gl_value(map: &str, id: u32, stage: u32) -> String {
    if stage == 1 {
        format!("{map}[{id}][0]")
    } else {
        format!("{map}[{id}]")
    }
}

/// The circom expression of an operand.
fn gl_operand(src: &Src) -> Result<String> {
    Ok(match &src.operand {
        Operand::Tmp(k) => format!("tmp_{k}"),
        Operand::Eval(k) => format!("evals[{k}]"),
        Operand::Challenge { id, .. } => format!("challenges[{id}]"),
        Operand::Number(v) => v.clone(),
        Operand::Public(id) => format!("publics[{id}]"),
        Operand::ProofValue { id, stage } => gl_value("proofvalues", *id, *stage),
        Operand::AirValue { id, stage } => gl_value("airvalues", *id, *stage),
        Operand::AirgroupValue { id, stage } => gl_value("airgroupvalues", *id, *stage),
        Operand::Zi { .. } => bail!("Zi belongs to the fold, not to a constraint"),
    })
}

/// The right-hand side of `x op y` over operands of dimensions `dx`, `dy`, in unroll_code's forms.
fn gl_rhs(op: CodeOp, x: &str, dx: usize, y: &str, dy: usize) -> Option<String> {
    Some(match (op, dx, dy) {
        (CodeOp::Add, 1, 1) => format!("{x} + {y}"),
        (CodeOp::Add, 1, 3) => format!("[{x} + {y}[0], {y}[1],  {y}[2]]"),
        (CodeOp::Add, 3, 1) => format!("[{x}[0] + {y}, {x}[1], {x}[2]]"),
        (CodeOp::Add, 3, 3) => format!("[{x}[0] + {y}[0], {x}[1] + {y}[1], {x}[2] + {y}[2]]"),
        (CodeOp::Sub, 1, 1) => format!("{x} - {y}"),
        (CodeOp::Sub, 1, 3) => format!("[{x} - {y}[0], -{y}[1], -{y}[2]]"),
        (CodeOp::Sub, 3, 1) => format!("[{x}[0] - {y}, {x}[1], {x}[2]]"),
        (CodeOp::Sub, 3, 3) => format!("[{x}[0] - {y}[0], {x}[1] - {y}[1], {x}[2] - {y}[2]]"),
        (CodeOp::Mul, 1, 1) => format!("{x} * {y}"),
        (CodeOp::Mul, 1, 3) => format!("[{x} * {y}[0], {x} * {y}[1], {x} * {y}[2]]"),
        (CodeOp::Mul, 3, 1) => format!("[{x}[0] * {y}, {x}[1] * {y}, {x}[2] * {y}]"),
        (CodeOp::Mul, 3, 3) => format!("CMul()({x}, {y})"),
        _ => return None,
    })
}

/// One instruction, in unroll_code's forms so the constraints are today's.
// The former templated verifier's forms over lower_roots' stream (counts measured identical).
fn gl_instr(i: &Instr) -> Result<String> {
    let [a, b] = &i.src[..] else { bail!("tmp_{} has {} operands", i.dest, i.src.len()) };
    let (x, y) = (gl_operand(a)?, gl_operand(b)?);
    let Some(rhs) = gl_rhs(i.op, &x, a.dim, &y, b.dim) else {
        bail!("tmp_{}: no circom form for {:?} of dims {}, {}", i.dest, i.op, a.dim, b.dim)
    };
    let decl = if i.dim == 1 { format!("signal tmp_{}", i.dest) } else { format!("signal tmp_{}[3]", i.dest) };
    Ok(format!("    {decl} <== {rhs};\n"))
}

/// `template parallel ConstraintRoots{suffix}()`, fixed signature (empty inputs padded to one entry).
pub(crate) fn constraint_roots(suffix: &str, dag: &ConstraintDag, ctx: &LowerCtx) -> Result<ConstraintRoots> {
    let (code, roots) = lower_roots(dag, ctx).map_err(|e| anyhow!("constraintDag: {e:?}"))?;
    let code = VerifierCode { tmp_used: code.len(), code };
    code.check_dims(3).map_err(|e| anyhow!("constraintDag: {e:?}"))?;
    let t = ctx.tables();
    let len1 = |n: usize| n.max(1);
    let mut out = format!(
        "// The constraints at xi, one root each; the fold, Zi and the Q check are verifier/stark_verifier.circom's.\n\
         template parallel ConstraintRoots{suffix}() {{\n\
         \x20   signal input evals[{}][3];\n\
         \x20   signal input challenges[{}][3];\n\
         \x20   signal input publics[{}];\n\
         \x20   signal input proofvalues[{}][3];\n\
         \x20   signal input airvalues[{}][3];\n\
         \x20   signal input airgroupvalues[{}][3];\n\
         \x20   signal output roots[{}][3];\n\n",
        t.ev_map.len(),
        len1(t.challenges.len()),
        len1(t.n_publics as usize),
        len1(t.proof_values_stage.len()),
        len1(t.air_values_stage.len()),
        len1(t.airgroup_values_stage.len()),
        roots.len(),
    );
    for i in &code.code {
        out += &gl_instr(i)?;
    }
    out.push('\n');
    for (k, r) in roots.iter().enumerate() {
        let x = gl_operand(r)?;
        out += &if r.dim == 3 {
            format!("    roots[{k}] <== {x};\n")
        } else {
            format!("    roots[{k}] <== [{x}, 0, 0];\n")
        };
    }
    out += "}\n";
    let first_root_dim = roots.first().map_or(3, |r| r.dim);
    Ok(ConstraintRoots { template: out, n_roots: roots.len(), first_root_dim })
}

// ── Global constraints: `GlobalConstraints` ───────────────────────────────────

/// The tables a vadcop_final reads its global values by, from its globalInfo.
fn global_tables(gi: &serde_json::Value) -> Result<GlobalTables> {
    let u32s = |v: &serde_json::Value| v.as_u64().and_then(|n| u32::try_from(n).ok());
    let stages =
        |list: &serde_json::Value| -> Option<Vec<u32>> { list.as_array()?.iter().map(|v| u32s(&v["stage"])).collect() };
    Ok(GlobalTables {
        n_publics: u32s(&gi["nPublics"]).ok_or_else(|| anyhow!("globalInfo has no nPublics"))?,
        num_challenges: (gi["numChallenges"].as_array().into_iter().flatten())
            .map(|n| u32s(n).ok_or_else(|| anyhow!("numChallenges: {n}")))
            .collect::<Result<_>>()?,
        proof_values_stage: match gi.get("proofValuesMap") {
            Some(m) => stages(m).ok_or_else(|| anyhow!("proofValuesMap without stages"))?,
            None => Vec::new(),
        },
        airgroup_values_stage: (gi["aggTypes"].as_array().ok_or_else(|| anyhow!("globalInfo has no aggTypes"))?)
            .iter()
            .map(|g| stages(g).ok_or_else(|| anyhow!("aggTypes without stages")))
            .collect::<Result<_>>()?,
        ext_dim: 3,
    })
}

/// A global leaf as a circom expression over `GlobalConstraints`' inputs (a stage-1 value is its limb 0).
fn global_operand(leaf: &GlobalLeaf, t: &GlobalTables) -> Result<String> {
    Ok(match *leaf {
        GlobalLeaf::Number(ref v) => v.clone(),
        GlobalLeaf::Public(id) => format!("publics[{id}]"),
        // vadcop_final draws only the stage-2 challenges (`GlobalChallenges`): the input holds those
        GlobalLeaf::Challenge(k) => {
            let first = *t.num_challenges.first().unwrap_or(&0);
            let n = *t.num_challenges.get(1).unwrap_or(&0);
            if k < first || k - first >= n {
                bail!("global challenge {k} is not a stage-2 challenge");
            }
            format!("challenges[{}]", k - first)
        }
        GlobalLeaf::ProofValue { id, stage } => gl_value("proofValues", id, stage),
        GlobalLeaf::AirgroupValue { airgroup, id, stage } => {
            gl_value(&format!("airgroupvalues[{airgroup}]"), id, stage)
        }
    })
}

/// Operation nodes per `GlobalConstraintsPart` template: one template of a few thousand signals takes the witness
/// compiler three times as long (measured: 37 s against 13 s for 5000 nodes).
const GLOBAL_PART_SIZE: usize = 50;

const GLOBAL_INPUTS: &str = "    signal input airgroupvalues[nAg][maxAgg][3];
    signal input publics[nPub];
    signal input proofValues[nPV][3];
    signal input challenges[nCh][3];
";
const GLOBAL_PARAMS: &str = "nAg, maxAgg, nPub, nPV, nCh";

/// `vadcop_nGlobalChallenges()` and `template GlobalConstraints(nAg, maxAgg, nPub, nPV, nCh)` of globalInfo `gi`:
/// each root of its `globalConstraintDag` `=== 0` (none without the key). The operation nodes, one signal each,
/// are computed in `GlobalConstraintsPart{k}` templates of `GLOBAL_PART_SIZE` nodes, in node order; a part takes
/// the earlier nodes it reads and returns those read after it.
pub(crate) fn global_constraints(gi: &serde_json::Value) -> Result<String> {
    let t = global_tables(gi)?;
    let f = pil_info::FieldCfg::goldilocks();
    let dag = ConstraintDag::from_global_info(gi, &f).map_err(|e| anyhow!("globalConstraintDag: {e:?}"))?;
    let n_challenges = t.num_challenges.get(1).copied().unwrap_or(0);
    let mut out = format!("function vadcop_nGlobalChallenges() {{ return {n_challenges}; }}\n\n");
    let main_head = format!("// The global constraints, one value per root, each zero.\ntemplate GlobalConstraints({GLOBAL_PARAMS}) {{\n{GLOBAL_INPUTS}\n");
    let Some(dag) = dag else { return Ok(out + &main_head + "}\n") };
    let leaves = t.resolve_all(&dag).map_err(|e| anyhow!("globalConstraintDag: {e:?}"))?;

    // each node as (circom expression, dimension); an operation node is the signal `node_{i}`
    let mut val: Vec<(String, usize)> = Vec::with_capacity(dag.nodes.len());
    let mut lines: Vec<(usize, String)> = Vec::new(); // (operation node, its right-hand side)
    for (i, (n, leaf)) in dag.nodes.iter().zip(&leaves).enumerate() {
        if let Some(l) = leaf {
            val.push((global_operand(l, &t)?, t.dim(l)));
            continue;
        }
        let zero = ("0".to_string(), 1);
        let (op, (x, dx), (y, dy)) = match *n {
            DagNode::Add { a, b } => (CodeOp::Add, &val[a as usize], &val[b as usize]),
            DagNode::Sub { a, b } => (CodeOp::Sub, &val[a as usize], &val[b as usize]),
            DagNode::Mul { a, b } => (CodeOp::Mul, &val[a as usize], &val[b as usize]),
            DagNode::Neg { a } => (CodeOp::Sub, &zero, &val[a as usize]),
            _ => bail!("node {i} is neither an operation nor a resolved leaf"),
        };
        let rhs = gl_rhs(op, x, *dx, y, *dy).ok_or_else(|| anyhow!("node {i}: dims {dx}, {dy}"))?;
        let dim = (*dx).max(*dy);
        lines.push((i, rhs));
        val.push((format!("node_{i}"), dim));
    }

    // part of each operation node, and the parts' inputs and outputs
    let mut part = vec![usize::MAX; dag.nodes.len()];
    for (k, &(i, _)) in lines.iter().enumerate() {
        part[i] = k / GLOBAL_PART_SIZE;
    }
    let n_parts = lines.len().div_ceil(GLOBAL_PART_SIZE);
    let (mut inputs, mut outputs) = (vec![BTreeSet::new(); n_parts], vec![BTreeSet::new(); n_parts]);
    for &(i, _) in &lines {
        for c in dag.nodes[i].children() {
            let c = c as usize;
            if part[c] != usize::MAX && part[c] != part[i] {
                inputs[part[i]].insert(c);
                outputs[part[c]].insert(c);
            }
        }
    }
    for r in &dag.roots {
        let r = r.node as usize;
        if part[r] != usize::MAX {
            outputs[part[r]].insert(r);
        }
    }

    let decl = |io: &str, i: usize| format!("    signal {io}node_{i}{};\n", if val[i].1 == 3 { "[3]" } else { "" });
    let mut main = main_head;
    for (k, chunk) in lines.chunks(GLOBAL_PART_SIZE).enumerate() {
        if outputs[k].is_empty() {
            bail!("global constraint part {k} returns nothing: a node no root reaches");
        }
        out += &format!("template GlobalConstraintsPart{k}({GLOBAL_PARAMS}) {{\n{GLOBAL_INPUTS}");
        out += &inputs[k].iter().map(|&i| decl("input ", i)).collect::<String>();
        out += &outputs[k].iter().map(|&i| decl("output ", i)).collect::<String>();
        for (i, rhs) in chunk {
            out += &if outputs[k].contains(i) {
                format!("    node_{i}")
            } else {
                decl("", *i).trim_end_matches(";\n").to_string()
            };
            out += &format!(" <== {rhs};\n");
        }
        out += "}\n\n";

        main += &outputs[k].iter().map(|&i| decl("", i)).collect::<String>();
        let args = ["airgroupvalues", "publics", "proofValues", "challenges"].map(String::from);
        let args: Vec<String> = args.into_iter().chain(inputs[k].iter().map(|i| format!("node_{i}"))).collect();
        let outs: Vec<String> = outputs[k].iter().map(|i| format!("node_{i}")).collect();
        main +=
            &format!("    ({}) <== GlobalConstraintsPart{k}({GLOBAL_PARAMS})({});\n", outs.join(", "), args.join(", "));
    }
    main.push('\n');
    for r in &dag.roots {
        let (x, dim) = &val[r.node as usize];
        main += &if *dim == 3 {
            format!("    {x}[0] === 0;\n    {x}[1] === 0;\n    {x}[2] === 0;\n")
        } else {
            format!("    {x} === 0;\n")
        };
    }
    Ok(out + &main + "}\n")
}

// ── BN128: `ConstraintEval` ───────────────────────────────────────────────────

/// A number as a Goldilocks element (the DAG may spell it negative).
fn gl_number(v: &str) -> Result<u128> {
    const P: i128 = 0xFFFF_FFFF_0000_0001;
    let n: i128 = v.parse().map_err(|_| anyhow!("constraint number {v:?} is not an integer"))?;
    Ok(n.rem_euclid(P) as u128)
}

/// BN128 forms of the lowered code: `GLConst`s shared per (value, dim), a base operand lifted by
/// `GLC3` next to an extension one (a zero is a `GLConst3` instead), as the former templated verifier emitted.
#[derive(Default)]
struct Bn128Code {
    lines: Vec<String>,
    consts: std::collections::HashMap<(u128, usize), String>,
    n_values: usize,
    /// Temporaries that are another signal: `x + 0` and `x - 0`, which the DAG keeps.
    alias: std::collections::HashMap<u32, String>,
}

impl Bn128Code {
    fn constant(&mut self, n: u128, dim: usize) -> String {
        if let Some(s) = self.consts.get(&(n, dim)) {
            return s.clone();
        }
        let s = format!("constValue_{}", self.consts.len());
        let (arr, t) = if dim == 3 { ("[3]", "GLConst3") } else { ("", "GLConst") };
        self.lines.push(format!("    signal {{maxNum}} {s}{arr} <== {t}({n})();"));
        self.consts.insert((n, dim), s.clone());
        s
    }

    fn is_zero(src: &Src) -> Result<bool> {
        Ok(matches!(&src.operand, Operand::Number(v) if gl_number(v)? == 0))
    }

    /// `src` as a signal, read by an op of dimension `dim` (a zero takes that dimension).
    fn raw(&mut self, src: &Src, dim: usize) -> Result<String> {
        Ok(match &src.operand {
            Operand::Number(v) => {
                let n = gl_number(v)?;
                self.constant(n, if n == 0 { dim } else { src.dim })
            }
            Operand::Tmp(k) => self.alias.get(k).cloned().unwrap_or_else(|| format!("tmp_{k}")),
            Operand::Eval(k) => format!("evals[{k}]"),
            Operand::Challenge { id, .. } => format!("challenges[{id}]"),
            Operand::Public(k) => format!("publics[{k}]"),
            Operand::Zi { boundary: 0 } => "Zh".into(),
            v => bail!("the BN128 verifier does not take {v:?} (its transcript does not bind it)"),
        })
    }

    /// A non-zero base value next to an extension one, lifted.
    fn lift(&mut self, src: &Src, name: String, other: usize) -> Result<String> {
        if src.dim == 1 && other == 3 && !Self::is_zero(src)? {
            let v = format!("value_{}", self.n_values);
            self.n_values += 1;
            self.lines.push(format!("    signal {{maxNum}} {v}[3] <== GLC3()({name});"));
            return Ok(v);
        }
        Ok(name)
    }

    /// `x + 0`, `0 + x` or `x - 0` at x's own dimension: the op's operand that is x itself.
    fn identity_operand<'a>(i: &'a Instr) -> Result<Option<&'a Src>> {
        let [a, b] = &i.src[..] else { return Ok(None) };
        let candidates: &[(&Src, &Src)] = match i.op {
            CodeOp::Add => &[(a, b), (b, a)],
            CodeOp::Sub => &[(a, b)],
            _ => &[],
        };
        for &(x, z) in candidates {
            if x.dim == i.dim && Self::is_zero(z)? {
                return Ok(Some(x));
            }
        }
        Ok(None)
    }

    fn instr(&mut self, i: &Instr) -> Result<()> {
        if let Some(x) = Self::identity_operand(i)? {
            let name = self.raw(x, x.dim)?;
            self.alias.insert(i.dest, name);
            return Ok(());
        }
        let arr = if i.dim == 3 { "[3]" } else { "" };
        self.lines.push(format!("    signal {{maxNum}} tmp_{}{arr};", i.dest));
        let d = format!("    tmp_{} <== ", i.dest);
        let line = match (i.op, &i.src[..]) {
            (CodeOp::Copy, [a]) if a.dim == 1 => format!("{d}GLCopy()({});", self.raw(a, 1)?),
            (CodeOp::Copy, [a]) => format!("{d}{};", self.raw(a, 3)?),
            (op, [a, b]) => {
                let (x, y) = (self.raw(a, b.dim)?, self.raw(b, a.dim)?);
                let (x, y) = (self.lift(a, x, b.dim)?, self.lift(b, y, a.dim)?);
                let t = match op {
                    CodeOp::Add => "Add",
                    CodeOp::Sub => "Sub",
                    CodeOp::Mul => "Mul",
                    CodeOp::Copy => bail!("tmp_{}: a copy of two operands", i.dest),
                };
                format!("{d}GL{}{t}()({x}, {y});", if i.dim == 1 { "" } else { "C" })
            }
            _ => bail!("tmp_{}: {:?} of {} operands", i.dest, i.op, i.src.len()),
        };
        self.lines.push(line);
        Ok(())
    }
}

/// `template ConstraintEval()`: the folded constraints at xi times Zh, as `evalP` (BN128 verifier).
pub(crate) fn constraint_eval_bn128(dag: &ConstraintDag, ctx: &LowerCtx) -> Result<String> {
    let code = lower_to_code(dag, ctx, Fold::Stark).map_err(|e| anyhow!("constraintDag: {e:?}"))?;
    code.check_dims(3).map_err(|e| anyhow!("constraintDag: {e:?}"))?;
    let code = code.code;
    let mut b = Bn128Code::default();
    for i in &code {
        b.instr(i)?;
    }
    let t = ctx.tables();
    let last = code.last().map_or(0, |i| i.dest);
    let last = b.alias.get(&last).cloned().unwrap_or_else(|| format!("tmp_{last}"));
    // Always declared (zero entries without publics): the static verifier wires it.
    let publics = format!("    signal input {{maxNum}} publics[{}];\n", t.n_publics);
    Ok(format!(
        "// The constraints at xi, folded, times Zh.\ntemplate ConstraintEval() {{\n    signal input {{maxNum}} challenges[{}][3];\n    signal input {{maxNum}} evals[{}][3];\n{publics}    signal input {{maxNum}} Zh[3];\n    signal output {{maxNum}} evalP[3];\n\n{}\n\n    evalP <== {last};\n}}\n",
        t.challenges.len(),
        t.ev_map.len(),
        b.lines.join("\n")
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use pil_info::core::dag::format::{ColType, DagBoundary, DagNode, DagRoot};
    use pil_info::core::dag::lower_code::LowerTables;
    use pil_info::core::ir::ColRef;

    fn ctx() -> LowerCtx {
        let col = |id| ColRef { col: ColType::Witness, commit_id: None, id, offset: 0 };
        LowerCtx::new(LowerTables {
            ev_map: vec![col(0), col(1)],
            challenges: vec![(2, 0), (3, 0)],
            vc: 1,
            n_publics: 1,
            air_values_stage: vec![1],
            proof_values_stage: vec![],
            airgroup_values_stage: vec![2],
            ext_dim: 3,
        })
        .unwrap()
    }

    fn dag(nodes: Vec<DagNode>, roots: &[u32]) -> ConstraintDag {
        let roots = roots.iter().map(|&node| DagRoot { node, boundary: DagBoundary::EveryRow, label: String::new() });
        ConstraintDag { version: ConstraintDag::VERSION, field: "gl".into(), ext_dim: 3, nodes, roots: roots.collect() }
    }

    fn cm(id: u32) -> DagNode {
        DagNode::Col { col: ColType::Witness, commit_id: None, id, offset: 0 }
    }

    /// 3 op nodes, 2 roots: one signal per op node, the roots as the fixed-size output.
    #[test]
    fn one_signal_per_op_node() {
        let d = dag(
            vec![
                cm(0),
                cm(1),
                DagNode::Mul { a: 0, b: 1 },
                DagNode::Challenge { stage: 2, stage_id: 0 },
                DagNode::Add { a: 2, b: 3 },
                DagNode::Sub { a: 4, b: 0 },
            ],
            &[5, 2],
        );
        let r = constraint_roots("0", &d, &ctx()).unwrap();
        assert_eq!(r.n_roots, 2);
        assert_eq!(r.template.matches(" <== ").count(), 3 + 2, "{}", r.template);
        assert!(r.template.contains("signal tmp_0[3] <== CMul()(evals[0], evals[1]);"), "{}", r.template);
        assert!(r.template.contains("signal output roots[2][3];"), "{}", r.template);
        assert!(r.template.contains("roots[0] <== tmp_2;"), "{}", r.template);
        assert!(r.template.contains("roots[1] <== tmp_0;"), "{}", r.template);
        assert!(!r.template.contains("Zh") && !r.template.contains("Zfirst"), "{}", r.template);
    }

    /// A base-field root leaves as [c, 0, 0] and the params learn its dimension.
    #[test]
    fn base_field_first_root() {
        let d = dag(
            vec![DagNode::Public { id: 0 }, DagNode::AirValue { id: 0 }, DagNode::Mul { a: 0, b: 1 }, cm(0)],
            &[2, 3],
        );
        let r = constraint_roots("0", &d, &ctx()).unwrap();
        assert_eq!(r.first_root_dim, 1);
        assert!(r.template.contains("signal tmp_0 <== publics[0] * airvalues[0][0];"), "{}", r.template);
        assert!(r.template.contains("roots[0] <== [tmp_0, 0, 0];"), "{}", r.template);
        assert!(r.template.contains("roots[1] <== evals[0];"), "{}", r.template);
    }

    #[test]
    fn refuses_a_non_every_row_root() {
        let mut d = dag(vec![cm(0)], &[0]);
        d.roots[0].boundary = DagBoundary::FirstRow;
        let Err(e) = constraint_roots("0", &d, &ctx()) else { panic!("a firstRow root must be refused") };
        assert!(e.to_string().contains("everyRow"), "{e}");
    }

    /// globalInfo with two airgroups (stages [1, 2] and [2]), 2 publics, proof values of stages 1 and 2, and
    /// challenges [1, 2]: the stage-2 ones are vadcop_final's `challenges`.
    fn global_info(nodes: serde_json::Value, roots: serde_json::Value) -> serde_json::Value {
        serde_json::json!({
            "nPublics": 2, "numChallenges": [1, 2],
            "proofValuesMap": [{"stage": 1}, {"stage": 2}],
            "aggTypes": [[{"aggType": 0, "stage": 1}, {"aggType": 0, "stage": 2}], [{"aggType": 1, "stage": 2}]],
            "globalConstraintDag": {"version": 1, "field": "gl", "extDim": 3, "nodes": nodes, "roots": roots},
        })
    }

    /// One signal per operation node, leaves inline (stage-1 values as limb 0), each root `=== 0` at its dimension.
    #[test]
    fn global_constraints_from_the_dag() {
        let gi = global_info(
            serde_json::json!([
                [10, 0, 0], // 0: airgroup 0 value 0 (stage 1)
                [10, 0, 1], // 1: airgroup 1 value 0
                [6, 2, 1],  // 2: challenge 2.1
                [2, 1, 2],  // 3: 1 * 2
                [0, 0, 3],  // 4: 0 + 3
                [7, 1],     // 5: public 1
                [8, 0],     // 6: proof value 0 (stage 1)
                [1, 5, 6],  // 7: 5 - 6
                [3, 7],     // 8: -7
            ]),
            serde_json::json!([[4, 1, "line 1"], [8, 1, "line 2"]]),
        );
        let out = global_constraints(&gi).unwrap();
        for l in [
            "function vadcop_nGlobalChallenges() { return 2; }",
            "template GlobalConstraintsPart0(nAg, maxAgg, nPub, nPV, nCh) {",
            "    signal output node_4[3];\n    signal output node_8;\n",
            "    signal node_3[3] <== CMul()(airgroupvalues[1][0], challenges[1]);",
            "    node_4 <== [airgroupvalues[0][0][0] + node_3[0], node_3[1],  node_3[2]];",
            "    signal node_7 <== publics[1] - proofValues[0][0];",
            "    node_8 <== 0 - node_7;",
            "template GlobalConstraints(nAg, maxAgg, nPub, nPV, nCh) {",
            "    signal node_4[3];\n    signal node_8;\n    (node_4, node_8) <== GlobalConstraintsPart0(nAg, maxAgg, nPub, nPV, nCh)(airgroupvalues, publics, proofValues, challenges);",
            "    node_4[0] === 0;\n    node_4[1] === 0;\n    node_4[2] === 0;\n",
            "    node_8 === 0;\n}\n",
        ] {
            assert!(out.contains(l), "missing `{l}`:\n{out}");
        }
        assert_eq!(out.matches(" <== ").count(), 5, "{out}");
    }

    /// 120 chained nodes over three parts: a part takes the earlier nodes it reads and returns the later-read ones.
    #[test]
    fn global_constraints_in_parts() {
        // node 0: public 0; node 1: public 1; node k >= 2: node k-1 * node 0 (k odd) or + node 1 (k even)
        let mut nodes = vec![serde_json::json!([7, 0]), serde_json::json!([7, 1])];
        for k in 2..122u32 {
            let (op, other) = if k % 2 == 1 { (2, 0) } else { (0, 1) };
            nodes.push(serde_json::json!([op, other.min(k - 1), other.max(k - 1)]));
        }
        let gi = global_info(serde_json::Value::Array(nodes), serde_json::json!([[121, 1, "chain"], [51, 1, "mid"]]));
        let out = global_constraints(&gi).unwrap();
        for l in [
            "template GlobalConstraintsPart2(nAg, maxAgg, nPub, nPV, nCh) {",
            // part 0 holds nodes 2..51: node 51 is read by part 1 and is a root
            "    (node_51) <== GlobalConstraintsPart0(nAg, maxAgg, nPub, nPV, nCh)(airgroupvalues, publics, proofValues, challenges);",
            "    (node_101) <== GlobalConstraintsPart1(nAg, maxAgg, nPub, nPV, nCh)(airgroupvalues, publics, proofValues, challenges, node_51);",
            "    (node_121) <== GlobalConstraintsPart2(nAg, maxAgg, nPub, nPV, nCh)(airgroupvalues, publics, proofValues, challenges, node_101);",
            "    signal input node_51;\n    signal output node_101;\n",
            "    signal node_52 <== publics[1] + node_51;",
            "    node_121 === 0;\n    node_51 === 0;\n}\n",
        ] {
            assert!(out.contains(l), "missing `{l}`:\n{out}");
        }
        assert!(!out.contains("GlobalConstraintsPart3"), "{out}");
    }

    #[test]
    fn global_constraints_refuse_what_vadcop_final_does_not_read() {
        let refused = |leaf: serde_json::Value| {
            let gi = global_info(serde_json::json!([leaf]), serde_json::json!([[0, 1, ""]]));
            global_constraints(&gi).unwrap_err().to_string()
        };
        assert!(refused(serde_json::json!([6, 1, 0])).contains("stage-2"), "a stage-1 challenge");
        assert!(refused(serde_json::json!([6, 2, 2])).contains("Unresolved"), "past stage 2's challenges");
        assert!(refused(serde_json::json!([10, 1, 1])).contains("Unresolved"), "past airgroup 1's values");
        assert!(refused(serde_json::json!([10, 2, 0])).contains("Unresolved"), "an unknown airgroup");
        assert!(refused(serde_json::json!([7, 2])).contains("Unresolved"), "past the publics");
        assert!(refused(serde_json::json!([10, 0])).contains("NonCanonical"), "an air's airgroup value");
        assert!(refused(serde_json::json!([5, 0, 0, 0])).contains("NonCanonical"), "a column");
    }

    #[test]
    fn no_global_constraints_no_body() {
        let mut gi = global_info(serde_json::json!([]), serde_json::json!([]));
        gi.as_object_mut().unwrap().remove("globalConstraintDag");
        let out = global_constraints(&gi).unwrap();
        assert!(out.ends_with("    signal input challenges[nCh][3];\n\n}\n"), "{out}");
    }

    /// `x - 0` on a base value is `x` itself: no GLSub against a zero constant, whose bound
    /// used to be 0 and wrapped the quotient off x mod GL.
    #[test]
    fn bn128_minus_zero_is_the_operand() {
        let zero = DagNode::Const { value: "0".into() };
        let d = dag(
            vec![DagNode::Public { id: 0 }, zero, DagNode::Sub { a: 0, b: 1 }, cm(0), DagNode::Mul { a: 2, b: 3 }],
            &[4],
        );
        let out = constraint_eval_bn128(&d, &ctx()).unwrap();
        assert!(!out.contains("GLSub") && !out.contains("GLCSub"), "{out}");
        assert!(out.contains("GLC3()(publics[0])"), "x - 0 reads x: {out}");
    }

    /// The one dimension rule refuses a DAG over another extension (here degree 2).
    #[test]
    fn bn128_refuses_a_dimension_mismatch() {
        let mut t = ctx().tables().clone();
        t.ext_dim = 2;
        let mut d = dag(vec![cm(0), cm(1), DagNode::Mul { a: 0, b: 1 }], &[2]);
        d.ext_dim = 2;
        let e = constraint_eval_bn128(&d, &LowerCtx::new(t).unwrap()).unwrap_err();
        assert!(e.to_string().contains("Dimension"), "{e}");
    }
}
