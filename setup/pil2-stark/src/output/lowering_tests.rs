//! The lowering's value oracle (a Goldilocks-cubic DAG evaluator) and its tests.

mod eval {
    use std::collections::HashMap;

    use anyhow::{anyhow, bail, Result};
    use proofman_fields::{CubicExtensionField, Goldilocks};

    use pil_info::core::dag::format::{ConstraintDag, DagNode};
    use pil_info::core::dag::lower_code::{stark_roots, CodeOp, LowerCtx, LowerTables, Operand, VerifierCode};
    use pil_info::core::fold::fold_roots;

    use crate::types::{goldilocks_u64, GOLDILOCKS_MODULUS};

    pub type E3 = CubicExtensionField<Goldilocks>;

    /// Everything a DAG or its lowered code reads, in the order of the resolver tables.
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

    use crate::output::test_support::common::Rng;

    fn gl(r: &mut Rng) -> Goldilocks {
        Goldilocks::new(r.next() % GOLDILOCKS_MODULUS)
    }

    fn e3(r: &mut Rng) -> E3 {
        E3::from_array(&[gl(r), gl(r), gl(r)])
    }

    /// Flat length of a value map: `value_dim` of each entry.
    fn flat_len(stages: &[u32]) -> usize {
        stages.iter().map(|&s| pil_info::core::ir::value_dim(s, 3)).sum()
    }

    /// Seeded inputs sized by `t`.
    pub fn random_inputs(t: &LowerTables, seed: u64) -> GlInputs {
        let mut r = Rng(seed);
        let mut base = |n: usize| (0..n).map(|_| gl(&mut r)).collect::<Vec<_>>();
        let publics = base(t.n_publics as usize);
        let proof_values = base(flat_len(&t.proof_values_stage));
        let air_values = base(flat_len(&t.air_values_stage));
        let airgroup_values = base(flat_len(&t.airgroup_values_stage));
        GlInputs {
            evals: (0..t.ev_map.len()).map(|_| e3(&mut r)).collect(),
            challenges: (0..t.challenges.len()).map(|_| e3(&mut r)).collect(),
            publics,
            proof_values,
            air_values,
            airgroup_values,
            zi: e3(&mut r),
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

    /// The value of the last instruction of lowered `code`.
    pub fn eval_code(code: &VerifierCode, ctx: &LowerCtx, inp: &GlInputs) -> Result<E3> {
        let mut tmp: HashMap<u32, E3> = HashMap::new();
        let mut last = None;
        for ins in &code.code {
            let src = (ins.src.iter())
                .map(|s| match s.operand {
                    Operand::Tmp(k) => tmp.get(&k).copied().ok_or_else(|| anyhow!("tmp {k} read before written")),
                    ref o => operand(o, ctx, inp),
                })
                .collect::<Result<Vec<E3>>>()?;
            let x = match (ins.op, src.as_slice()) {
                (CodeOp::Add, [a, b]) => *a + *b,
                (CodeOp::Sub, [a, b]) => *a - *b,
                (CodeOp::Mul, [a, b]) => *a * *b,
                (CodeOp::Copy, [a]) => *a,
                (op, s) => bail!("{op:?} of {} operands", s.len()),
            };
            tmp.insert(ins.dest, x);
            last = Some(x);
        }
        last.ok_or_else(|| anyhow!("the code is empty"))
    }
}

mod tests {
    use std::collections::{BTreeSet, HashMap};

    use num_bigint::BigUint;
    use pil_info::core::dag::build::build_dag;
    use pil_info::core::dag::format::{DagBoundary, DagNode, DagRoot};
    use pil_info::core::dag::format::ConstraintDag;
    use pil_info::core::dag::lower_code::{lower_to_code, Fold};
    use proofman_fields::Goldilocks;
    use pil_info::core::im::im_column_id;
    use pil_info::core::ir::{eval_expr, ColRef, ColType, Env, Expr};
    use proofman_fields::{Field, PrimeField64};

    use super::eval::*;
    use crate::output::test_support::{common, full_random};
    use pil_info::core::dag::lower_code::{LowerCtx, LowerTables};

    const P: u64 = 0xFFFF_FFFF_0000_0001;

    /// 4 cm + 2 const + 1 custom column at a few offsets, 3 challenges (the last is `std_vc`), 3 publics, 2+2+1 values.
    fn tiny_ctx() -> LowerCtx {
        let col = |col, commit_id, id, offset| ColRef { col, commit_id, id, offset };
        let mut ev_map: Vec<ColRef> =
            (0..4).flat_map(|id| [0, 1, -1].map(|p| col(ColType::Witness, None, id, p))).collect();
        ev_map.extend((0..2).flat_map(|id| [0, 1].map(|p| col(ColType::Fixed, None, id, p))));
        ev_map.push(col(ColType::Custom, Some(0), 0, 0));
        LowerCtx::new(LowerTables {
            ev_map,
            challenges: vec![(2, 0), (2, 1), (3, 0)],
            vc: 2,
            n_publics: 3,
            air_values_stage: vec![1, 2],
            proof_values_stage: vec![1, 2],
            airgroup_values_stage: vec![2],
            ext_dim: 3,
        })
        .unwrap()
    }

    /// A seeded canonical DAG over `tiny_ctx`: hash-consed, sorted commutative children, every sink a root.
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

    #[test]
    fn lowered_code_matches_dag() {
        let ctx = tiny_ctx();
        for seed in 0..50 {
            let dag = random_dag(seed);
            dag.validate(&pil_info::FieldCfg::goldilocks()).unwrap();
            let inp = random_inputs(ctx.tables(), seed);
            let ops = lower_to_code(&dag, &ctx, Fold::Stark).unwrap();
            let (got, want) = (eval_code(&ops, &ctx, &inp).unwrap(), eval_dag(&dag, &ctx, &inp).unwrap());
            assert_eq!(got, want, "seed {seed}");
        }
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
            let mut inp = random_inputs(&si.lower_tables, k as u64);
            inp.evals.iter_mut().for_each(|e| *e = base(&fe(e[0])));
            inp.challenges.iter_mut().for_each(|e| *e = base(&fe(e[0])));
            inp.zi = base(&fe(inp.zi[0]));
            let mut env = Env::default();
            let ev_of = |id: u64, prime: i64| {
                si.ev_map
                    .iter()
                    .position(|e| e.ev_type == proofman_common::EvMapEType::Cm && e.id == id && e.prime == prime)
            };
            for (j, ev) in si.ev_map.iter().enumerate().filter(|(_, e)| e.ev_type == proofman_common::EvMapEType::Cm) {
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
            let got = eval_dag(&dag, &LowerCtx::new(si.lower_tables.clone()).unwrap(), &inp).unwrap();
            assert_eq!((fe(got[0]), got[1].is_zero(), got[2].is_zero()), (want, true, true), "pilout {k}");
        }
    }
}
