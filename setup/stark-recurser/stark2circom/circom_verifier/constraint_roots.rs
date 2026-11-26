//! The air's constraints at xi from the constraint DAG, one signal per operation node. GL:
//! `ConstraintRoots<suffix>`, one root each (`lower_roots`); the fold, Zi and the Q check are the
//! library's (`verifier/stark_verifier.circom`). BN128: `ConstraintEval`, the roots folded with
//! `lower_to_code` in lazily reduced `GL*` templates (`verifier/stark_verifier_bn128.circom`).

use anyhow::{anyhow, bail, Result};
use pil_info::core::dag::format::ConstraintDag;
use pil_info::core::dag::lower_code::{
    lower_roots, lower_to_code, CodeOp, Fold, Instr, LowerCtx, Operand, Src, VerifierCode,
};

/// The generated template plus what the params record about it.
pub(crate) struct ConstraintRoots {
    pub template: String,
    pub n_roots: usize,
    pub first_root_dim: usize,
}

/// A value-map entry: limb 0 at stage 1 (base field), all three after.
fn value(map: &str, id: u32, stage: u32) -> String {
    if stage == 1 {
        format!("{map}[{id}][0]")
    } else {
        format!("{map}[{id}]")
    }
}

/// The circom expression of an operand.
fn operand(src: &Src) -> Result<String> {
    Ok(match &src.operand {
        Operand::Tmp(k) => format!("tmp_{k}"),
        Operand::Eval(k) => format!("evals[{k}]"),
        Operand::Challenge { id, .. } => format!("challenges[{id}]"),
        Operand::Number(v) => v.clone(),
        Operand::Public(id) => format!("publics[{id}]"),
        Operand::ProofValue { id, stage } => value("proofvalues", *id, *stage),
        Operand::AirValue { id, stage } => value("airvalues", *id, *stage),
        Operand::AirgroupValue { id, stage } => value("airgroupvalues", *id, *stage),
        Operand::Zi { .. } => bail!("Zi belongs to the fold, not to a constraint"),
    })
}

/// One instruction, in unroll_code's forms so the constraints are today's.
// The former templated verifier's forms over lower_roots' stream (counts measured identical).
fn instr(i: &Instr) -> Result<String> {
    let [a, b] = &i.src[..] else { bail!("tmp_{} has {} operands", i.dest, i.src.len()) };
    let (x, y) = (operand(a)?, operand(b)?);
    let rhs = match (i.op, a.dim, b.dim) {
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
        (op, da, db) => bail!("tmp_{}: no circom form for {op:?} of dims {da}, {db}", i.dest),
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
        out += &instr(i)?;
    }
    out.push('\n');
    for (k, r) in roots.iter().enumerate() {
        let x = operand(r)?;
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

    /// `x - 0` on a base value is `x` itself: no GLSub against a zero constant, whose bound
    /// used to be 0 and wrapped the quotient off x mod GL.
    #[test]
    fn bn128_minus_zero_is_the_operand() {
        let zero = DagNode::Const { value: "0".into() };
        let d = dag(vec![DagNode::Public { id: 0 }, zero, DagNode::Sub { a: 0, b: 1 }, cm(0), DagNode::Mul { a: 2, b: 3 }], &[4]);
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
