//! Goldilocks-cubic evaluator of a constraint DAG and of lowered verifier code (the DAG's parity oracle).

use anyhow::{anyhow, bail, Result};
use proofman_fields::{CubicExtensionField, Goldilocks};

use pil_info::core::dag::format::{ConstraintDag, DagNode};
use pil_info::core::dag::lower_code::{stark_roots, LowerCtx, LowerTables, Operand};
use pil_info::core::fold::fold_roots;
use pil_info::core::ir::{ColRef, ColType};

use crate::types::stark_info::{CodeOperation, OpType, PolMap, StarkInfo, VerifierInfo};
use crate::types::{goldilocks_u64, GOLDILOCKS_MODULUS};

pub type E3 = CubicExtensionField<Goldilocks>;

/// Everything a DAG or its lowered code reads, in the order of the starkinfo maps.
#[derive(Clone, Debug)]
pub struct GlInputs {
    pub evals: Vec<E3>,
    pub challenges: Vec<E3>,
    pub publics: Vec<Goldilocks>,
    pub proof_values: Vec<Goldilocks>,
    pub air_values: Vec<Goldilocks>,
    pub airgroup_values: Vec<Goldilocks>,
    pub zi: E3,
}

/// The resolver context of `si`: its evMap, challengesMap (`std_vc` by name) and value stages.
pub fn lower_ctx(si: &StarkInfo) -> Result<LowerCtx> {
    let ev_map = si
        .ev_map
        .iter()
        .map(|e| {
            let (col, commit_id) = match e.ev_type.as_str() {
                "custom" => (ColType::Custom, Some(e.commit_id as u32)),
                "const" => (ColType::Fixed, None),
                "cm" => (ColType::Witness, None),
                t => bail!("evMap entry of unknown type {t:?}"),
            };
            Ok(ColRef { col, commit_id, id: e.id as u32, offset: e.prime })
        })
        .collect::<Result<Vec<_>>>()?;
    let challenges: Vec<(u32, u32)> = si.challenges_map.iter().map(|c| (c.stage as u32, c.stage_id as u32)).collect();
    // a missing `std_vc` is past the end, so lowering and evaluation report it unresolved
    let vc = si.challenges_map.iter().position(|c| c.name == "std_vc").unwrap_or(challenges.len());
    let stages = |m: &[PolMap]| m.iter().map(|p| p.stage as u32).collect();
    let t = LowerTables {
        ev_map,
        challenges,
        vc,
        n_publics: si.n_publics as u32,
        air_values_stage: stages(&si.air_values_map),
        proof_values_stage: stages(&si.proof_values_map),
        airgroup_values_stage: stages(&si.airgroup_values_map),
        ext_dim: 3,
    };
    LowerCtx::new(t).map_err(|e| anyhow!("{e:?}"))
}

/// splitmix64.
struct Rng(u64);
impl Rng {
    fn gl(&mut self) -> Goldilocks {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        Goldilocks::new((z ^ (z >> 31)) % GOLDILOCKS_MODULUS)
    }
    fn e3(&mut self) -> E3 {
        E3::from_array(&[self.gl(), self.gl(), self.gl()])
    }
}

/// Flat length of a value map: 1 per stage-1 entry, 3 per other.
fn flat_len(m: &[PolMap]) -> usize {
    m.iter().map(|p| if p.stage == 1 { 1 } else { 3 }).sum()
}

/// Seeded inputs sized by `si`'s maps.
pub fn random_inputs(si: &StarkInfo, seed: u64) -> GlInputs {
    let mut r = Rng(seed);
    let mut base = |n: usize| (0..n).map(|_| r.gl()).collect::<Vec<_>>();
    let publics = base(si.n_publics as usize);
    let proof_values = base(flat_len(&si.proof_values_map));
    let air_values = base(flat_len(&si.air_values_map));
    let airgroup_values = base(flat_len(&si.airgroup_values_map));
    GlInputs {
        evals: (0..si.ev_map.len()).map(|_| r.e3()).collect(),
        challenges: (0..si.challenges_map.len()).map(|_| r.e3()).collect(),
        publics,
        proof_values,
        air_values,
        airgroup_values,
        zi: r.e3(),
    }
}

fn embed(g: Goldilocks) -> E3 {
    E3::from_array(&[g, Goldilocks::new(0), Goldilocks::new(0)])
}

/// Value `id` of a flat value buffer laid out by `stages` (stage 1 takes 1 element, others 3).
fn value(stages: &[u32], flat: &[Goldilocks], id: u32, what: &str) -> Result<E3> {
    let id = id as usize;
    let at: usize = stages.iter().take(id).map(|&s| if s == 1 { 1 } else { 3 }).sum();
    let stage = *stages.get(id).ok_or_else(|| anyhow!("{what} {id} is not in its map"))?;
    let n = if stage == 1 { 1 } else { 3 };
    let v = flat.get(at..at + n).ok_or_else(|| anyhow!("{what} {id} is outside its inputs"))?;
    Ok(if n == 1 { embed(v[0]) } else { E3::from_array(v) })
}

fn number(s: &str) -> Result<E3> {
    Ok(embed(Goldilocks::new(goldilocks_u64(s)?)))
}

fn at<T: Copy>(v: &[T], id: u64, what: &str) -> Result<T> {
    v.get(id as usize).copied().ok_or_else(|| anyhow!("{what} {id} is outside its inputs"))
}

/// The value of an operand the DAG and its code share.
fn operand(o: &Operand, ctx: &LowerCtx, inp: &GlInputs) -> Result<E3> {
    let t = ctx.tables();
    Ok(match *o {
        Operand::Tmp(_) => bail!("a temporary is not a DAG leaf"),
        Operand::Eval(id) => at(&inp.evals, id.into(), "evaluation")?,
        Operand::Challenge { id, .. } => at(&inp.challenges, id.into(), "challenge")?,
        Operand::Number(ref v) => number(v)?,
        Operand::Public(id) => embed(at(&inp.publics, id.into(), "public")?),
        Operand::ProofValue { id, .. } => value(&t.proof_values_stage, &inp.proof_values, id, "proof value")?,
        Operand::AirValue { id, .. } => value(&t.air_values_stage, &inp.air_values, id, "air value")?,
        Operand::AirgroupValue { id, .. } => {
            value(&t.airgroup_values_stage, &inp.airgroup_values, id, "airgroup value")?
        }
        Operand::Zi { .. } => inp.zi,
    })
}

/// `fold_roots` of the roots' values times `inp.zi`.
pub fn eval_dag(dag: &ConstraintDag, ctx: &LowerCtx, inp: &GlInputs) -> Result<E3> {
    let mut v: Vec<E3> = Vec::with_capacity(dag.nodes.len());
    for (i, n) in dag.nodes.iter().enumerate() {
        let get = |c: u32| v.get(c as usize).copied().ok_or_else(|| anyhow!("node {i} reads node {c} ahead of it"));
        let x = match *n {
            DagNode::Add { a, b } => get(a)? + get(b)?,
            DagNode::Sub { a, b } => get(a)? - get(b)?,
            DagNode::Mul { a, b } => get(a)? * get(b)?,
            DagNode::Neg { a } => -get(a)?,
            _ => match ctx.resolve(i as u32, n).map_err(|e| anyhow!("{e:?}"))? {
                Some(o) => operand(&o, ctx, inp)?,
                None => bail!("node {i} is neither an operation nor a leaf"),
            },
        };
        v.push(x);
    }
    let vc = operand(&ctx.vc_operand().map_err(|e| anyhow!("{e:?}"))?, ctx, inp)?;
    let roots = stark_roots(dag)
        .map_err(|e| anyhow!("{e:?}"))?
        .into_iter()
        .map(|r| v.get(r as usize).copied().ok_or_else(|| anyhow!("root {r} is not a node")))
        .collect::<Result<Vec<_>>>()?;
    let c = fold_roots(roots, |acc| vc * acc, |acc, c| acc + c).ok_or_else(|| anyhow!("the DAG has no roots"))?;
    Ok(c * inp.zi)
}

/// The value of the last instruction of `code`.
pub fn eval_code(code: &[CodeOperation], ctx: &LowerCtx, inp: &GlInputs) -> Result<E3> {
    let mut tmp: Vec<E3> = Vec::new();
    for (k, c) in code.iter().enumerate() {
        let src = c
            .src
            .iter()
            .map(|s| {
                let o = match s.op_type {
                    OpType::Tmp => return at(&tmp, s.id, "temporary"),
                    OpType::Number => return Ok(embed(Goldilocks::new(s.value))),
                    OpType::Eval => Operand::Eval(s.id as u32),
                    OpType::Challenge => Operand::Challenge { id: s.id as u32, stage: 0, stage_id: 0 },
                    OpType::Public => Operand::Public(s.id as u32),
                    OpType::Proofvalue => Operand::ProofValue { id: s.id as u32, stage: 0 },
                    OpType::Airvalue => Operand::AirValue { id: s.id as u32, stage: 0 },
                    OpType::Airgroupvalue => Operand::AirgroupValue { id: s.id as u32, stage: 0 },
                    OpType::Zi => Operand::Zi { boundary: 0 },
                    t => bail!("instruction {k} reads a {}", t.to_str()),
                };
                operand(&o, ctx, inp)
            })
            .collect::<Result<Vec<E3>>>()?;
        let x = match (c.op.as_str(), src.as_slice()) {
            ("add", [a, b]) => *a + *b,
            ("sub", [a, b]) => *a - *b,
            ("mul", [a, b]) => *a * *b,
            ("copy", [a]) => *a,
            (op, s) => bail!("instruction {k}: {op} of {} operands", s.len()),
        };
        if c.dest.op_type != OpType::Tmp {
            bail!("instruction {k} does not write a temporary");
        }
        let d = c.dest.id as usize;
        if tmp.len() <= d {
            tmp.resize(d + 1, E3::zero());
        }
        tmp[d] = x;
    }
    let last = code.last().ok_or_else(|| anyhow!("the code is empty"))?;
    at(&tmp, last.dest.id, "temporary")
}

/// TODO(piece 3): delete once qVerifier comes from `lower_to_code`. `dag` against `vi`'s qVerifier code on three seeds; the error says "differs" on a mismatch.
/// (`dag` is an added parameter: `si` and `vi` alone carry no DAG.)
pub fn check_dag_parity(dag: &ConstraintDag, si: &StarkInfo, vi: &VerifierInfo) -> Result<()> {
    let ctx = lower_ctx(si)?;
    for seed in 0..3 {
        let inp = random_inputs(si, seed);
        let (d, c) = (eval_dag(dag, &ctx, &inp)?, eval_code(&vi.q_verifier.code, &ctx, &inp)?);
        if d != c {
            bail!("the constraint DAG differs from the qVerifier code at seed {seed}: {d} vs {c}");
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::collections::{BTreeSet, HashMap};

    use num_bigint::BigUint;
    use pil_info::core::dag::build::build_dag;
    use pil_info::core::dag::format::{DagBoundary, DagNode, DagRoot};
    use pil_info::core::dag::lower_code::{lower_to_code, Fold};
    use pil_info::core::im::im_column_id;
    use pil_info::core::ir::{eval_expr, Env, Expr};
    use pil_info::emit::verifier::expression_code;
    use pil_info::pil::gen_code;
    use proofman_fields::{Field, PrimeField64};

    use super::*;
    use crate::output::test_support::{common, full, full_random};
    use crate::types::stark_info::{CustomCommit, EvMap, OpType, PolMap};

    const P: u64 = 0xFFFF_FFFF_0000_0001;

    fn pol(name: &str, stage: u64, dim: u64, stage_id: u64) -> PolMap {
        PolMap { name: name.into(), stage, dim, stage_id, ..Default::default() }
    }

    /// 4 cm + 2 const + 1 custom column at a few offsets, 3 challenges (the last is `std_vc`), 3 publics, 2+2+1 values.
    fn tiny_si() -> StarkInfo {
        let mut si = StarkInfo { n_stages: 2, n_publics: 3, ..Default::default() };
        si.cm_pols_map = (0..4).map(|i| pol(&format!("c{i}"), 1, 1, i)).collect();
        si.const_pols_map = (0..2).map(|i| pol(&format!("k{i}"), 0, 1, i)).collect();
        si.custom_commits = vec![CustomCommit { name: "rom".into() }];
        si.challenges_map = vec![pol("alpha", 2, 3, 0), pol("beta", 2, 3, 1), pol("std_vc", 3, 3, 0)];
        si.proof_values_map = vec![pol("pv0", 1, 1, 0), pol("pv1", 2, 3, 0)];
        si.air_values_map = vec![pol("av0", 1, 1, 0), pol("av1", 2, 3, 0)];
        si.airgroup_values_map = vec![pol("gv0", 2, 3, 0)];
        si.opening_points = vec![0, 1, -1];
        let pos = |p: i64| si.opening_points.iter().position(|&x| x == p).unwrap() as u64;
        let mut ev = Vec::new();
        for id in 0..4 {
            for p in [0, 1, -1] {
                ev.push(EvMap { ev_type: "cm".into(), id, prime: p, commit_id: 0, opening_pos: pos(p) });
            }
        }
        for id in 0..2 {
            for p in [0, 1] {
                ev.push(EvMap { ev_type: "const".into(), id, prime: p, commit_id: 0, opening_pos: pos(p) });
            }
        }
        ev.push(EvMap { ev_type: "custom".into(), id: 0, prime: 0, commit_id: 0, opening_pos: 0 });
        si.ev_map = ev;
        si
    }

    /// A seeded canonical DAG over `tiny_si`: hash-consed, sorted commutative children, every sink a root.
    fn random_dag(seed: u64) -> ConstraintDag {
        let mut rng = common::Rng(seed);
        let mut nodes: Vec<DagNode> = Vec::new();
        let mut index: HashMap<DagNode, u32> = HashMap::new();
        let mut used: BTreeSet<u32> = BTreeSet::new();
        let mut push = |n: DagNode, nodes: &mut Vec<DagNode>| -> u32 {
            *index.entry(n.clone()).or_insert_with(|| {
                nodes.push(n);
                nodes.len() as u32 - 1
            })
        };
        for _ in 0..8 + rng.below(8) {
            let leaf = match rng.below(10) {
                0..=3 => {
                    let off = [0, 1, -1][rng.below(3) as usize];
                    DagNode::Col { col: ColType::Witness, commit_id: None, id: rng.below(4) as u32, offset: off }
                }
                4 => DagNode::Col {
                    col: ColType::Fixed,
                    commit_id: None,
                    id: rng.below(2) as u32,
                    offset: rng.below(2) as i64,
                },
                5 => DagNode::Col { col: ColType::Custom, commit_id: Some(0), id: 0, offset: 0 },
                6 => DagNode::Const { value: [0, 1, rng.next() % P][rng.below(3) as usize].to_string() },
                7 => DagNode::Challenge { stage: 2, stage_id: rng.below(2) as u32 },
                8 => DagNode::Public { id: rng.below(3) as u32 },
                _ => match rng.below(3) {
                    0 => DagNode::ProofValue { id: rng.below(2) as u32 },
                    1 => DagNode::AirValue { id: rng.below(2) as u32 },
                    _ => DagNode::AirgroupValue { id: 0 },
                },
            };
            push(leaf, &mut nodes);
        }
        for _ in 0..10 + rng.below(20) {
            let n = nodes.len() as u64;
            let (a, b) = (rng.below(n) as u32, rng.below(n) as u32);
            let (lo, hi) = (a.min(b), a.max(b));
            let op = match rng.below(8) {
                0..=2 => DagNode::Add { a: lo, b: hi },
                3..=4 => DagNode::Mul { a: lo, b: hi },
                5..=6 => DagNode::Sub { a, b },
                _ => DagNode::Neg { a },
            };
            match &op {
                DagNode::Add { a, b } | DagNode::Mul { a, b } | DagNode::Sub { a, b } => used.extend([*a, *b]),
                DagNode::Neg { a } => {
                    used.insert(*a);
                }
                _ => {}
            }
            push(op, &mut nodes);
        }
        let roots = (0..nodes.len() as u32)
            .filter(|i| !used.contains(i))
            .map(|node| DagRoot { node, boundary: DagBoundary::EveryRow, label: format!("r{node}") })
            .collect();
        ConstraintDag { version: 1, field: "gl".into(), ext_dim: 3, nodes, roots }
    }

    fn code_of(dag: &ConstraintDag, si: &StarkInfo) -> Vec<CodeOperation> {
        let entry = expression_code(&lower_to_code(dag, &lower_ctx(si).unwrap(), Fold::Stark).unwrap());
        VerifierInfo::try_from(&gen_code::VerifierInfo { q_verifier: entry, constraint_dag: None })
            .unwrap()
            .q_verifier
            .code
    }

    #[test]
    fn lowered_code_matches_dag() {
        let si = tiny_si();
        for seed in 0..50 {
            let dag = random_dag(seed);
            dag.validate(&pil_info::FieldCfg::goldilocks()).unwrap();
            let ops = code_of(&dag, &si);
            let (inp, ctx) = (random_inputs(&si, seed), lower_ctx(&si).unwrap());
            let (got, want) = (eval_code(&ops, &ctx, &inp).unwrap(), eval_dag(&dag, &ctx, &inp).unwrap());
            assert_eq!(got, want, "seed {seed}");
        }
    }

    #[test]
    fn lower_ctx_rejects_unknown_or_repeated_entries() {
        let mut si = tiny_si();
        si.ev_map[0].ev_type = "cmm".into();
        assert!(lower_ctx(&si).is_err(), "an unknown evMap type");
        let mut si = tiny_si();
        si.ev_map.push(si.ev_map[0].clone());
        assert!(lower_ctx(&si).is_err(), "a repeated evMap entry");
        let mut si = tiny_si();
        si.challenges_map.push(si.challenges_map[0].clone());
        assert!(lower_ctx(&si).is_err(), "a repeated challenge");
    }

    fn const_pilout() -> pil2_pilout::pilout::PilOut {
        use common::*;
        let e = vec![mul(witness(0, 0), witness(1, 0)), sub(exp(0), constant("7")), mul(exp(1), witness(2, 0))];
        pilout(&["a", "b", "c", "d"], &[], e, vec![every_row(2), every_row(1)])
    }

    #[test]
    fn parity_accepts_matching_code() {
        let f = full(const_pilout()).expect("feasible");
        let dag = build_dag(&f.run.cs, &f.run.im).unwrap();
        check_dag_parity(&dag, &f.si, &f.vi).unwrap();
    }

    #[test]
    fn parity_rejects_changed_constant() {
        let f = full(const_pilout()).expect("feasible");
        let dag = build_dag(&f.run.cs, &f.run.im).unwrap();
        let mut vi = f.vi.clone();
        let n = vi.q_verifier.code.iter_mut().flat_map(|o| o.src.iter_mut()).find(|s| s.op_type == OpType::Number);
        n.expect("the qVerifier reads the constant 7").value += 1;
        let err = check_dag_parity(&dag, &f.si, &vi).unwrap_err();
        assert!(err.to_string().contains("differs"), "{err}");
    }

    #[test]
    fn parity_rejects_reordered_roots() {
        let f = full(const_pilout()).expect("feasible");
        let mut dag = build_dag(&f.run.cs, &f.run.im).unwrap();
        assert!(dag.roots.len() >= 2, "two constraint roots");
        dag.roots.swap(0, 1);
        let err = check_dag_parity(&dag, &f.si, &f.vi).unwrap_err();
        assert!(err.to_string().contains("differs"), "{err}");
    }

    fn fe(g: Goldilocks) -> BigUint {
        BigUint::from(g.as_canonical_u64())
    }
    fn base(v: &BigUint) -> E3 {
        let mut e = E3::zero();
        e[0] = Goldilocks::new(u64::try_from(v).unwrap());
        e
    }

    #[test]
    fn dag_equals_expression_eval() {
        let runs = full_random(20);
        assert_eq!(runs.len(), 20, "not enough feasible random pilouts");
        for (k, f) in runs.iter().enumerate() {
            let (cs, im, si) = (&f.run.cs, &f.run.im, &f.si);
            let dag = build_dag(cs, im).unwrap();
            // base-field values only, so the field-of-BigUint reference can follow
            let mut inp = random_inputs(si, k as u64);
            inp.evals.iter_mut().for_each(|e| *e = base(&fe(e[0])));
            inp.challenges.iter_mut().for_each(|e| *e = base(&fe(e[0])));
            inp.zi = base(&fe(inp.zi[0]));
            let mut env = Env::default();
            let ev_of = |id: u64, prime: i64| {
                si.ev_map.iter().position(|e| e.ev_type == "cm" && e.id == id && e.prime == prime)
            };
            for (j, ev) in si.ev_map.iter().enumerate().filter(|(_, e)| e.ev_type == "cm") {
                if (ev.id as usize) < cs.witness.len() {
                    let k = ColRef { col: ColType::Witness, commit_id: None, id: ev.id as u32, offset: ev.prime };
                    env.cols.insert(k, fe(inp.evals[j][0]));
                }
            }
            // an im column is its defining expression's value
            for (m, &e) in im.exprs.iter().enumerate() {
                let id = im_column_id(cs, m);
                let v = eval_expr(cs, &Expr::Ref(e), &env).unwrap();
                env.cols.insert(ColRef { col: ColType::Witness, commit_id: None, id, offset: 0 }, v.clone());
                if let Some(j) = ev_of(id as u64, 0) {
                    inp.evals[j] = base(&v);
                }
            }
            let p = BigUint::from(P);
            let vc_at = si.challenges_map.iter().position(|c| c.name == "std_vc").unwrap();
            let vc = fe(inp.challenges[vc_at][0]);
            let mut roots: Vec<BigUint> = cs.constraints.iter().map(|c| eval_expr(cs, &c.e, &env).unwrap()).collect();
            for (m, &e) in im.exprs.iter().enumerate() {
                let col = Expr::Col { col: ColType::Witness, commit_id: None, id: im_column_id(cs, m), offset: 0 };
                roots.push(eval_expr(cs, &Expr::Sub(Box::new(col), Box::new(Expr::Ref(e))), &env).unwrap());
            }
            let mut it = roots.into_iter();
            let first = it.next().unwrap();
            let acc = it.fold(first, |acc, c| (&vc * acc + c) % &p);
            let want = acc * fe(inp.zi[0]) % &p;
            let got = eval_dag(&dag, &lower_ctx(si).unwrap(), &inp).unwrap();
            assert_eq!((fe(got[0]), got[1].is_zero(), got[2].is_zero()), (want, true, true), "pilout {k}");
        }
    }
}
