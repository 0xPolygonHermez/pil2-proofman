//! The DEEP polynomial f_0 at a query point, emitted from the starkinfo's opening groups
//! (`proofman_common::deep`) for both circom backends.
//!
//! f = Σ_G S_G·(Σ_{o∈G} A_o) − Σ_o A_o·K_o, with A_o = vf1^{e_o}·xDivXSubXi[rank of o] (e_o = number of later
//! non-empty openings), S_G = Σ_{c∈G} vf2^c·p_c over the columns opened at exactly the set G, and
//! K_o = Σ_{c∈o} vf2^c·evals[evalPos]: the grouped form the native provers and verifiers compute.
//!
//! `friConsts` (computed once per proof) is `[vf1, vf2, vf2^0..vf2^{nPols-1}, vf1^{e_o} per non-empty
//! opening, K_o per non-empty opening]`.

use std::collections::HashMap;

use anyhow::{bail, Context, Result};
use proofman_common::deep::{col_groups, fri_ev_groups};
use proofman_common::{EvMap, EvMapEType, StarkInfo};
use serde_json::Value;

/// A circom expression and its dimension (1: base field, 3: extension).
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Operand {
    pub name: String,
    pub dim: u8,
}

impl Operand {
    fn new(name: impl Into<String>, dim: u8) -> Self {
        Self { name: name.into(), dim }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) enum OpKind {
    Add,
    Sub,
    Mul,
}

/// `dest = a op b`; `dest` is a fresh signal of dimension max(a.dim, b.dim).
#[derive(Debug, Clone)]
pub(crate) struct Op {
    pub kind: OpKind,
    pub dest: String,
    pub a: Operand,
    pub b: Operand,
}

impl Op {
    pub fn dest_dim(&self) -> u8 {
        self.a.dim.max(self.b.dim)
    }
}

#[derive(Default)]
struct Ops {
    ops: Vec<Op>,
    prefix: &'static str,
}

impl Ops {
    fn push(&mut self, kind: OpKind, a: Operand, b: Operand) -> Operand {
        let dest = format!("{}{}", self.prefix, self.ops.len());
        let out = Operand::new(dest.clone(), a.dim.max(b.dim));
        self.ops.push(Op { kind, dest, a, b });
        out
    }

    /// Left-to-right sum; a single term is returned as is.
    fn sum(&mut self, terms: Vec<Operand>) -> Option<Operand> {
        terms.into_iter().reduce(|acc, t| self.push(OpKind::Add, acc, t))
    }
}

/// The DEEP evaluation split into its once-per-proof and per-query halves.
pub(crate) struct DeepPlan {
    /// Inputs `challengesDeep` and `evals`; produces the operands of `const_outputs`.
    pub once: Vec<Op>,
    /// `friConsts[i] <== const_outputs[i]`.
    pub const_outputs: Vec<Operand>,
    /// Inputs `friConsts`, `xDivXSubXi`, `mapValues.*` and `consts`.
    pub query: Vec<Op>,
    /// f at the query point.
    pub result: Operand,
    /// Opening points (indices into `openingPoints`) with evaluations; `xDivXSubXi[r]` is the r-th.
    pub openings: Vec<usize>,
}

impl DeepPlan {
    pub fn n_fri_consts(&self) -> usize {
        self.const_outputs.len()
    }

    /// The `opening_points` entries with evaluations, each tagged with its rank there: `xDivXSubXi` is
    /// only emitted for those, indexed by rank.
    pub fn ranked_openings(&self, opening_points: &[Value]) -> Vec<Value> {
        self.openings
            .iter()
            .enumerate()
            .map(|(rank, &o)| {
                let mut p = opening_points[o].clone();
                p["rank"] = rank.into();
                p
            })
            .collect()
    }

    /// The five `deep_*` template variables; the backends differ only in the rendered code strings.
    pub fn insert_ctx(&self, ctx: &mut tera::Context, once_code: &str, const_assign: &str, query_code: &str) {
        ctx.insert("deep_once_code", once_code);
        ctx.insert("deep_const_assign", const_assign);
        ctx.insert("deep_query_code", query_code);
        ctx.insert("deep_result", &self.result.name);
        ctx.insert("n_fri_consts", &self.n_fri_consts());
    }
}

/// The opened polynomial behind one evaluation, as `MapValues` (or the fixed leaf) names it.
fn pol_operand(ev: &EvMap, info: &StarkInfo) -> Result<Operand> {
    let (id, commit) = (ev.id as usize, ev.commit_id as usize);
    Ok(match ev.type_ {
        EvMapEType::Const => Operand::new(format!("consts[{id}]"), 1),
        EvMapEType::Cm => {
            let pol = info.cm_pols_map.as_ref().and_then(|m| m.get(id)).with_context(|| format!("cmPolsMap[{id}]"))?;
            Operand::new(format!("mapValues.cm{}_{}", pol.stage, pol.stage_id), pol.dim as u8)
        }
        EvMapEType::Custom => {
            let pol = info.custom_commits_map.get(commit).and_then(|m| m.as_ref()?.get(id));
            let pol = pol.with_context(|| format!("customCommitsMap[{commit}][{id}]"))?;
            let name = &info.custom_commits.get(commit).with_context(|| format!("customCommits[{commit}]"))?.name;
            Operand::new(format!("mapValues.custom_{name}_{}_{}", pol.stage, pol.stage_id), 1)
        }
    })
}

/// The DEEP ops of a starkinfo (JSON), in the op order and sharing of `deep_poly.rs`'s expression.
pub(crate) fn plan(si: &Value) -> Result<DeepPlan> {
    let info: StarkInfo = serde_json::from_value(si.clone()).context("starkinfo")?;
    // An empty evMap would leave f undefined (the circuit must never check lowValues against 0).
    if info.ev_map.is_empty() {
        bail!("evMap has no evaluations, so the DEEP polynomial is undefined");
    }
    let groups = fri_ev_groups(&info);
    let cgroups = col_groups(&groups);
    let n = groups.len();
    let n_pols = groups.iter().flat_map(|g| &g.refs).map(|r| r.col as usize + 1).max().unwrap_or(0);

    // The k-th ref is evMap[k] (fri_ev_groups walks evMap in order), which names the column.
    let mut pols: HashMap<u16, Operand> = HashMap::new();
    let mut eval_pos: Vec<Vec<usize>> = Vec::with_capacity(n);
    let mut k = 0;
    for g in &groups {
        let mut pos = Vec::with_capacity(g.refs.len());
        for r in &g.refs {
            let ev = &info.ev_map[k];
            if let std::collections::hash_map::Entry::Vacant(e) = pols.entry(r.col) {
                e.insert(pol_operand(ev, &info)?);
            }
            pos.push(k);
            k += 1;
        }
        eval_pos.push(pos);
    }

    let vf2_idx = |c: usize| 2 + c;
    let vf1_idx = |r: usize| 2 + n_pols + r;
    let k_idx = |r: usize| 2 + n_pols + n + r;
    let fc = |i: usize| Operand::new(format!("friConsts[{i}]"), 3);
    let one = Operand::new("1", 1);

    // ── Once per proof ──
    let mut once = Ops { prefix: "deepC_", ..Default::default() };
    let vf1 = Operand::new("challengesDeep[0]", 3);
    let vf2 = Operand::new("challengesDeep[1]", 3);
    let powers = |base: &Operand, count: usize, once: &mut Ops| {
        let mut p = vec![one.clone(), base.clone()];
        while p.len() < count {
            let last = p.last().unwrap().clone();
            p.push(once.push(OpKind::Mul, last, base.clone()));
        }
        p.truncate(count);
        p
    };
    let vf2_pow = powers(&vf2, n_pols, &mut once);
    let vf1_pow = powers(&vf1, n, &mut once);
    let mut k_vals = Vec::with_capacity(n);
    for (g, pos) in groups.iter().zip(&eval_pos) {
        let terms = g
            .refs
            .iter()
            .zip(pos)
            .map(|(r, &k)| {
                let ev = Operand::new(format!("evals[{k}]"), 3);
                if r.col == 0 {
                    ev
                } else {
                    once.push(OpKind::Mul, vf2_pow[r.col as usize].clone(), ev)
                }
            })
            .collect();
        k_vals.push(once.sum(terms).expect("a group has refs"));
    }
    let mut const_outputs = vec![vf1.clone(), vf2.clone()];
    const_outputs.extend(vf2_pow.iter().cloned());
    const_outputs.extend((0..n).map(|r| vf1_pow[n - 1 - r].clone()));
    const_outputs.extend(k_vals);

    // ── Per query ──
    let mut query = Ops { prefix: "deepQ_", ..Default::default() };
    let rank: HashMap<usize, usize> = groups.iter().enumerate().map(|(r, g)| (g.opening as usize, r)).collect();
    let a: Vec<Operand> = (0..n)
        .map(|r| {
            let xdiv = Operand::new(format!("xDivXSubXi[{r}]"), 3);
            if r == n - 1 {
                xdiv
            } else {
                query.push(OpKind::Mul, fc(vf1_idx(r)), xdiv)
            }
        })
        .collect();
    // Left-to-right sums of A over each group's openings, sharing common prefixes.
    let mut a_sums: HashMap<Vec<usize>, Operand> = HashMap::new();
    let mut terms = Vec::with_capacity(cgroups.len());
    for cg in &cgroups {
        let ranks: Vec<usize> = cg.openings.iter().map(|o| rank[o]).collect();
        let mut acc = a[ranks[0]].clone();
        for j in 1..ranks.len() {
            acc = match a_sums.get(&ranks[..=j]) {
                Some(s) => s.clone(),
                None => {
                    let s = query.push(OpKind::Add, acc, a[ranks[j]].clone());
                    a_sums.insert(ranks[..=j].to_vec(), s.clone());
                    s
                }
            };
        }
        let cols = cg
            .cols
            .iter()
            .map(|r| {
                let p = pols[&r.col].clone();
                if r.col == 0 {
                    p
                } else {
                    query.push(OpKind::Mul, fc(vf2_idx(r.col as usize)), p)
                }
            })
            .collect();
        let s = query.sum(cols).expect("a column group has columns");
        terms.push(query.push(OpKind::Mul, s, acc));
    }
    let mut result = query.sum(terms).expect("groups imply terms");
    for (r, a_r) in a.iter().enumerate() {
        let m = query.push(OpKind::Mul, a_r.clone(), fc(k_idx(r)));
        result = query.push(OpKind::Sub, result, m);
    }

    let openings = groups.iter().map(|g| g.opening as usize).collect();
    Ok(DeepPlan { once: once.ops, const_outputs, query: query.ops, result, openings })
}

/// GL backend: native extension arithmetic, `CMul` for an extension product.
pub(crate) fn render_gl(ops: &[Op]) -> String {
    let mut out = String::new();
    for op in ops {
        let (a, b) = (&op.a.name, &op.b.name);
        let decl = if op.dest_dim() == 1 { format!("signal {}", op.dest) } else { format!("signal {}[3]", op.dest) };
        let rhs = match (op.kind, op.a.dim, op.b.dim) {
            (OpKind::Mul, 3, 3) => format!("CMul()({a}, {b})"),
            (OpKind::Mul, 1, 3) => format!("[{a} * {b}[0], {a} * {b}[1], {a} * {b}[2]]"),
            (OpKind::Mul, 3, 1) => format!("[{a}[0] * {b}, {a}[1] * {b}, {a}[2] * {b}]"),
            (OpKind::Mul, _, _) => format!("{a} * {b}"),
            (kind @ (OpKind::Add | OpKind::Sub), da, db) => {
                let s = if kind == OpKind::Add { "+" } else { "-" };
                let limb = |x: &str, d: u8, i: usize| match (d, i) {
                    (3, _) => format!("{x}[{i}]"),
                    (_, 0) => x.to_string(),
                    _ => "0".into(),
                };
                if da == 1 && db == 1 {
                    format!("{a} {s} {b}")
                } else {
                    let l: Vec<String> = (0..3).map(|i| format!("{} {s} {}", limb(a, da, i), limb(b, db, i))).collect();
                    format!("[{}]", l.join(", "))
                }
            }
        };
        out.push_str(&format!("    {decl} <== {rhs};\n"));
    }
    out
}

/// BN128 backend: emulated Goldilocks with lazy reduction (`{maxNum}` tags); a base-field operand
/// meeting an extension one is lifted with `GLC3`, as `unroll_code_bn128` did.
pub(crate) fn render_bn128(ops: &[Op]) -> String {
    let mut out = String::new();
    let mut lifted = 0usize;
    for op in ops {
        let mut lift = |x: &Operand, other: u8, out: &mut String| {
            if x.dim == 1 && other == 3 {
                let sig = format!("{}lift{lifted}", op.dest.split('_').next().unwrap_or("deep"));
                lifted += 1;
                out.push_str(&format!("    signal {{maxNum}} {sig}[3] <== GLC3()({});\n", x.name));
                sig
            } else {
                x.name.clone()
            }
        };
        let a = lift(&op.a, op.b.dim, &mut out);
        let b = lift(&op.b, op.a.dim, &mut out);
        let arr = if op.dest_dim() == 3 { "[3]" } else { "" };
        let c = if op.dest_dim() == 3 { "C" } else { "" };
        let t = match op.kind {
            OpKind::Add => "Add",
            OpKind::Sub => "Sub",
            OpKind::Mul => "Mul",
        };
        out.push_str(&format!("    signal {{maxNum}} {}{arr};\n    {} <== GL{c}{t}()({a}, {b});\n", op.dest, op.dest));
    }
    out
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use proofman_fields::{CubicExtensionField, Goldilocks};
    use serde_json::json;

    type F3 = CubicExtensionField<Goldilocks>;

    fn f3(a: u64, b: u64, c: u64) -> F3 {
        CubicExtensionField { value: [Goldilocks::new(a), Goldilocks::new(b), Goldilocks::new(c)] }
    }

    /// A full starkinfo (GL hash, 2 stages). Opening 1 is empty; columns: cm0 (base) at {0,2}, cm1 (ext)
    /// at {0}, const0 at {2}, custom at {0,2}.
    pub(crate) fn sample_stark_info() -> Value {
        json!({
            "starkStruct": {
                "verificationHashType": "GL", "numQueries": 10, "nBits": 16, "nBitsExt": 17,
                "grindingBitsQueries": 0, "merkleTreeArity": 4, "transcriptArity": 4, "merkleTreeCustom": false,
                "lastLevelVerification": 0, "splitLinearHash": false, "hashCommits": false,
                "foldingFactors": [], "logDegrees": [], "logDomainSizes": [8]
            },
            "security": {"regime": "test"},
            "nStages": 2,
            "nPublics": 4,
            "nConstants": 1,
            "qDeg": 1,
            "qDim": 3,
            "cExpId": 0,
            "boundaries": [{"name": "everyRow"}],
            "challengesMap": [],
            "airgroupValuesMap": [],
            "airValuesMap": [],
            "proofValuesMap": [],
            "mapSectionsN": {"cm1": 1, "cm2": 3, "cm3": 3, "rom0": 1},
            "openingPoints": [-1, 0, 1],
            "cmPolsMap": [
                {"name": "a", "stage": 1, "stagePos": 0, "stageId": 0, "dim": 1},
                {"name": "b", "stage": 2, "stagePos": 0, "stageId": 0, "dim": 3},
                {"name": "q", "stage": 3, "stagePos": 0, "stageId": 0, "dim": 3}
            ],
            "customCommits": [{"name": "rom", "publicValues": [{"idx": 0}, {"idx": 1}, {"idx": 2}, {"idx": 3}], "stageWidths": [1]}],
            "customCommitsMap": [[{"name": "r", "stage": 0, "stagePos": 0, "stageId": 0, "dim": 1}]],
            "evMap": [
                {"type": "cm", "id": 0, "prime": -1, "openingPos": 0},
                {"type": "cm", "id": 1, "prime": -1, "openingPos": 0},
                {"type": "custom", "id": 0, "prime": -1, "openingPos": 0, "commitId": 0},
                {"type": "cm", "id": 0, "prime": 1, "openingPos": 2},
                {"type": "const", "id": 0, "prime": 1, "openingPos": 2},
                {"type": "custom", "id": 0, "prime": 1, "openingPos": 2, "commitId": 0},
                {"type": "cm", "id": 2, "prime": 1, "openingPos": 2}
            ]
        })
    }

    fn eval_ops(ops: &[Op], env: &mut HashMap<String, F3>) {
        for op in ops {
            let (a, b) = (env[&op.a.name], env[&op.b.name]);
            let v = match op.kind {
                OpKind::Add => a + b,
                OpKind::Sub => a - b,
                OpKind::Mul => a * b,
            };
            env.insert(op.dest.clone(), v);
        }
    }

    /// The emitted ops equal the plain DEEP sum Σ_o vf1^{e_o}·xdiv_o·Σ_{c∈o} vf2^c·(p_c − eval).
    #[test]
    fn plan_matches_the_deep_sum() {
        let si = sample_stark_info();
        let plan = plan(&si).unwrap();
        assert_eq!(plan.openings, vec![0, 2]);
        // 2 + nPols (5) + 2 * nOpenings (2)
        assert_eq!(plan.n_fri_consts(), 2 + 5 + 4);

        let (vf1, vf2) = (f3(3, 5, 7), f3(11, 13, 17));
        let evals: Vec<F3> = (0..7).map(|i| f3(100 + i, 200 + i, 300 + i)).collect();
        let xdiv = [f3(21, 22, 23), F3::zero(), f3(31, 32, 33)];
        let p: HashMap<&str, F3> = [
            ("mapValues.cm1_0", f3(41, 0, 0)),
            ("mapValues.cm2_0", f3(42, 43, 44)),
            ("mapValues.custom_rom_0_0", f3(45, 0, 0)),
            ("consts[0]", f3(46, 0, 0)),
            ("mapValues.cm3_0", f3(47, 48, 49)),
        ]
        .into();

        let mut env: HashMap<String, F3> = HashMap::new();
        env.insert("1".into(), F3::one());
        env.insert("challengesDeep[0]".into(), vf1);
        env.insert("challengesDeep[1]".into(), vf2);
        for (k, e) in evals.iter().enumerate() {
            env.insert(format!("evals[{k}]"), *e);
        }
        eval_ops(&plan.once, &mut env);
        let consts: Vec<F3> = plan.const_outputs.iter().map(|o| env[&o.name]).collect();
        let mut env: HashMap<String, F3> = HashMap::new();
        for (i, c) in consts.iter().enumerate() {
            env.insert(format!("friConsts[{i}]"), *c);
        }
        for (r, &o) in plan.openings.iter().enumerate() {
            env.insert(format!("xDivXSubXi[{r}]"), xdiv[o]);
        }
        for (k, v) in &p {
            env.insert(k.to_string(), *v);
        }
        eval_ops(&plan.query, &mut env);
        let got = env[&plan.result.name];

        // cols by first appearance: cm1_0=0, cm2_0=1, custom=2, consts[0]=3, cm3_0=4.
        let col = |name: &str| {
            ["mapValues.cm1_0", "mapValues.cm2_0", "mapValues.custom_rom_0_0", "consts[0]", "mapValues.cm3_0"]
                .iter()
                .position(|n| *n == name)
                .unwrap() as u64
        };
        let pw = |b: F3, e: u64| (0..e).fold(F3::one(), |acc, _| acc * b);
        let opening0 = ["mapValues.cm1_0", "mapValues.cm2_0", "mapValues.custom_rom_0_0"];
        let opening2 = ["mapValues.cm1_0", "consts[0]", "mapValues.custom_rom_0_0", "mapValues.cm3_0"];
        let mut want = F3::zero();
        for (e, (o, names, first)) in [(0usize, &opening0[..], 0usize), (2, &opening2[..], 3)].into_iter().enumerate() {
            let mut inner = F3::zero();
            for (i, n) in names.iter().enumerate() {
                inner += pw(vf2, col(n)) * (p[n] - evals[first + i]);
            }
            want += pw(vf1, 1 - e as u64) * xdiv[o] * inner;
        }
        assert_eq!(got, want);
    }

    /// Each kind renders its own operator: a base-field Mul must not fall into the Add/Sub arm.
    #[test]
    fn render_gl_renders_each_kind() {
        let op = |kind, da, db| Op { kind, dest: "d".into(), a: Operand::new("a", da), b: Operand::new("b", db) };
        let rendered = |kind, da, db| render_gl(&[op(kind, da, db)]);
        assert_eq!(rendered(OpKind::Mul, 1, 1), "    signal d <== a * b;\n");
        assert_eq!(rendered(OpKind::Add, 1, 1), "    signal d <== a + b;\n");
        assert_eq!(rendered(OpKind::Sub, 1, 1), "    signal d <== a - b;\n");
        assert_eq!(rendered(OpKind::Mul, 3, 3), "    signal d[3] <== CMul()(a, b);\n");
        assert_eq!(rendered(OpKind::Sub, 1, 3), "    signal d[3] <== [a - b[0], 0 - b[1], 0 - b[2]];\n");
    }

    /// Both backends: (name, hash, merkle arity, generator, DEEP template).
    type Gen = fn(&Value, &Value, &super::super::Pil2CircomOptions) -> Result<String>;
    fn backends() -> [(&'static str, &'static str, u64, Gen, &'static str); 2] {
        [
            (
                "GL",
                "Poseidon2",
                4,
                |si, vi, o| super::super::gl::gen_stark_verifier_gl(None, si, vi, o),
                "DeepConstants",
            ),
            (
                "BN128",
                "Poseidon",
                16,
                |si, vi, o| super::super::bn128::gen_stark_verifier_bn128(None, si, vi, o),
                "CalculateFRIConstants",
            ),
        ]
    }

    #[test]
    fn no_evaluations_is_an_error() {
        for (ty, hash, arity, gen, _) in backends() {
            let mut si = sample_stark_info();
            si["starkStruct"]["verificationHashType"] = json!(ty);
            si["starkStruct"]["merkleTreeArity"] = json!(arity);
            si["evMap"] = json!([]);
            let opts = super::super::Pil2CircomOptions { hash: hash.to_string(), ..Default::default() };
            let err = gen(&si, &json!({ "qVerifier": { "code": [] } }), &opts).unwrap_err();
            assert!(err.to_string().contains("no evaluations"), "{ty}: {err}");
        }
    }

    #[test]
    fn deep_from_groups_renders() {
        for (ty, hash, arity, gen, template) in backends() {
            let mut si = sample_stark_info();
            si["starkStruct"]["verificationHashType"] = json!(ty);
            si["starkStruct"]["merkleTreeArity"] = json!(arity);
            let opts = super::super::Pil2CircomOptions { hash: hash.to_string(), ..Default::default() };
            let out = gen(&si, &json!({ "qVerifier": { "code": [] } }), &opts).unwrap();
            assert!(out.contains(&format!("template {template}(")), "{ty}: no {template} template:\n{out}");
            assert!(!out.contains("CalculateDeepPolChunks"), "{ty}: chunked DEEP code is back:\n{out}");
            let assigned = |r: usize| out.matches(&format!("xDivXSubXi[{r}] <==")).count();
            assert_eq!(
                (assigned(0), assigned(1), assigned(2)),
                (1, 1, 0),
                "{ty}: one xDivXSubXi per non-empty opening"
            );
            assert!(out.contains("xDivXSubXi[2][3];"), "{ty}: declared only for the non-empty openings:\n{out}");
            // friConsts = [vf1, vf2, vf2^0..vf2^4, vf1^e x2, K x2]
            assert!(out.contains("friConsts[11][3]"), "{ty}: friConsts size:\n{out}");
            assert!(out.contains("mapValues.custom_rom_0_0"), "{ty}: custom column not read:\n{out}");
        }
    }

    #[test]
    fn plan_without_evaluations_is_an_error() {
        let mut si = sample_stark_info();
        si["evMap"] = json!([]);
        let Err(err) = plan(&si) else { panic!("an empty evMap must not plan") };
        assert!(err.to_string().contains("no evaluations"), "{err}");
    }
}
