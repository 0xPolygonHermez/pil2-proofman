//! The global constraint DAG against the former global code (pil_code_gen over the global expressions, which
//! globalConstraints.json carried until the DAG replaced it), on seeded values.

use std::collections::HashMap;

use anyhow::{anyhow, bail, Result};
use num_bigint::BigUint;
use proofman_fields::{CubicExtensionField, Goldilocks};
use serde_json::Value;

use pil_info::core::dag::build::build_global_dag;
use pil_info::core::dag::format::{ConstraintDag, DagNode};
use pil_info::core::dag::global::{GlobalLeaf, GlobalTables};
use pil_info::core::global::{global_from_pilout, GlobalConstraint, GlobalSystem};
use pil_info::core::ir::Expr;
use pil_info::core::pilout_core::{AggType, AirGroupValue};
use pil_info::emit::arena;
use pil_info::expr::helpers::add_info_expressions;
use pil_info::pil::codegen::{build_code, pil_code_gen, CodeGenCtx};
use pil_info::FieldCfg;

use crate::output::test_support::common::Rng;
use crate::types::{goldilocks_u64, GOLDILOCKS_MODULUS};

type E3 = CubicExtensionField<Goldilocks>;

/// Every value with three random limbs; a stage-1 value is read as its first limb by both sides.
struct Inputs {
    publics: Vec<Goldilocks>,
    challenges: Vec<E3>,
    proof_values: Vec<E3>,
    airgroup_values: Vec<Vec<E3>>,
}

fn gl(r: &mut Rng) -> Goldilocks {
    Goldilocks::new(r.next() % GOLDILOCKS_MODULUS)
}

fn e3(r: &mut Rng) -> E3 {
    E3::from_array(&[gl(r), gl(r), gl(r)])
}

fn embed(g: Goldilocks) -> E3 {
    E3::from_array(&[g, Goldilocks::new(0), Goldilocks::new(0)])
}

fn base_if(dim1: bool, v: E3) -> E3 {
    if dim1 {
        embed(v.value[0])
    } else {
        v
    }
}

fn inputs(t: &GlobalTables, seed: u64) -> Inputs {
    let mut r = Rng(seed);
    Inputs {
        publics: (0..t.n_publics).map(|_| gl(&mut r)).collect(),
        challenges: (0..t.num_challenges.iter().sum::<u32>()).map(|_| e3(&mut r)).collect(),
        proof_values: t.proof_values_stage.iter().map(|_| e3(&mut r)).collect(),
        airgroup_values: t.airgroup_values_stage.iter().map(|g| g.iter().map(|_| e3(&mut r)).collect()).collect(),
    }
}

/// Each root's value, leaves read through `GlobalTables` (the verifiers' semantics).
fn eval_dag(dag: &ConstraintDag, t: &GlobalTables, inp: &Inputs) -> Result<Vec<E3>> {
    let leaves = t.resolve_all(dag).map_err(|e| anyhow!("{e:?}"))?;
    let mut v: Vec<E3> = Vec::with_capacity(dag.nodes.len());
    for (n, leaf) in dag.nodes.iter().zip(&leaves) {
        let x = match (n, leaf) {
            (DagNode::Add { a, b }, _) => v[*a as usize] + v[*b as usize],
            (DagNode::Sub { a, b }, _) => v[*a as usize] - v[*b as usize],
            (DagNode::Mul { a, b }, _) => v[*a as usize] * v[*b as usize],
            (DagNode::Neg { a }, _) => -v[*a as usize],
            (_, Some(l)) => {
                let x = match *l {
                    GlobalLeaf::Number(ref s) => embed(Goldilocks::new(goldilocks_u64(s)?)),
                    GlobalLeaf::Public(id) => embed(inp.publics[id as usize]),
                    GlobalLeaf::Challenge(k) => inp.challenges[k as usize],
                    GlobalLeaf::ProofValue { id, .. } => inp.proof_values[id as usize],
                    GlobalLeaf::AirgroupValue { airgroup, id, .. } => {
                        inp.airgroup_values[airgroup as usize][id as usize]
                    }
                };
                base_if(t.dim(l) == 1, x)
            }
            (n, None) => bail!("{n:?} did not resolve"),
        };
        v.push(x);
    }
    Ok(dag.roots.iter().map(|r| v[r.node as usize]).collect())
}

/// The former code of each global constraint, as globalConstraints.json had it.
fn former_code(gs: &GlobalSystem) -> Vec<Value> {
    let leaf = arena::global_leaf(gs);
    let mut expressions = arena::from_ir(&gs.exprs, &leaf).unwrap();
    for c in &gs.constraints {
        add_info_expressions(&mut expressions, c.expr as usize, &gs.field).unwrap();
    }
    let mut ctx = CodeGenCtx::new(0, 0, gs.num_challenges.len().max(1), "n");
    let mut out = Vec::new();
    for c in &gs.constraints {
        pil_code_gen(&mut ctx, &[], &expressions, c.expr as usize, 0).unwrap();
        let block = build_code(&mut ctx).unwrap();
        ctx.tmp_used = block.tmp_used;
        out.push(serde_json::to_value(&block.code).unwrap());
    }
    out
}

/// One former global constraint's code: the value of its last instruction.
fn eval_code(code: &[Value], inp: &Inputs) -> Result<E3> {
    let mut tmp: HashMap<u64, E3> = HashMap::new();
    let mut last = None;
    for ins in code {
        let read = |r: &Value| -> Result<E3> {
            let id = r["id"].as_u64().unwrap_or(0) as usize;
            let dim1 = r["dim"] == 1;
            Ok(match r["type"].as_str().unwrap_or("") {
                "tmp" => *tmp.get(&(id as u64)).ok_or_else(|| anyhow!("tmp {id} read before written"))?,
                "public" => embed(inp.publics[id]),
                "number" => {
                    let s = match &r["value"] {
                        Value::String(s) => s.clone(),
                        v => v.to_string(),
                    };
                    embed(Goldilocks::new(goldilocks_u64(&s)?))
                }
                "challenge" => inp.challenges[id],
                "proofvalue" => base_if(dim1, inp.proof_values[id]),
                "airgroupvalue" => {
                    let g = r["airgroupId"].as_u64().ok_or_else(|| anyhow!("airgroupvalue without airgroupId"))?;
                    base_if(dim1, inp.airgroup_values[g as usize][id])
                }
                t => bail!("operand type {t}"),
            })
        };
        let src =
            ins["src"].as_array().ok_or_else(|| anyhow!("no src"))?.iter().map(read).collect::<Result<Vec<_>>>()?;
        let x = match (ins["op"].as_str().unwrap_or(""), src.as_slice()) {
            ("add", [a, b]) => *a + *b,
            ("sub", [a, b]) => *a - *b,
            ("mul", [a, b]) => *a * *b,
            ("copy", [a]) => *a,
            (op, s) => bail!("{op} of {} operands", s.len()),
        };
        tmp.insert(ins["dest"]["id"].as_u64().ok_or_else(|| anyhow!("dest without id"))?, x);
        last = Some(x);
    }
    last.ok_or_else(|| anyhow!("empty code"))
}

/// The DAG's roots and the former code agree on `seeds` input sets, constraint by constraint.
fn assert_same_values(gs: &GlobalSystem, seeds: std::ops::Range<u64>) {
    let dag = build_global_dag(gs).unwrap().expect("global constraints");
    assert_eq!(dag.roots.len(), gs.constraints.len());
    for (r, c) in dag.roots.iter().zip(&gs.constraints) {
        assert_eq!(r.label, c.label);
    }
    let old = former_code(gs);
    let t = GlobalTables::from(gs);
    for seed in seeds {
        let inp = inputs(&t, seed);
        let new = eval_dag(&dag, &t, &inp).unwrap();
        for (k, code) in old.iter().enumerate() {
            let want = eval_code(code.as_array().unwrap(), &inp).unwrap();
            assert_eq!(new[k], want, "constraint {k}, seed {seed}");
        }
    }
}

/// Two airgroups with base and extension values, proof values of both kinds, challenges in three stages.
fn random_system(seed: u64) -> GlobalSystem {
    let mut r = Rng(seed);
    let agv = |stage| AirGroupValue { agg_type: AggType::Sum, stage };
    let mut gs = GlobalSystem {
        field: FieldCfg::goldilocks(),
        exprs: Vec::new(),
        constraints: Vec::new(),
        hints: Vec::new(),
        num_challenges: vec![1, 2, 1],
        num_publics: 3,
        airgroup_values: vec![vec![agv(1), agv(2)], vec![agv(2), agv(1), agv(2)]],
        proof_value_stages: vec![1, 2, 2],
    };
    let n = 6 + r.below(10) as usize;
    for i in 0..n {
        let leaf = |r: &mut Rng| match r.below(7) {
            0 => Expr::Const(BigUint::from([0u64, 1, 7, r.next() % GOLDILOCKS_MODULUS][r.below(4) as usize])),
            1 => Expr::Public(r.below(3) as u32),
            2 => {
                let stage = 1 + r.below(3) as u32;
                Expr::Challenge { stage, stage_id: r.below(gs.num_challenges[stage as usize - 1] as u64) as u32 }
            }
            3 => Expr::ProofValue(r.below(3) as u32),
            4 if i > 0 => Expr::Ref(r.below(i as u64) as u32),
            _ => {
                let g = r.below(2) as u32;
                Expr::AirgroupValue { airgroup: Some(g), id: r.below([2, 3][g as usize]) as u32 }
            }
        };
        let (a, b) = (Box::new(leaf(&mut r)), Box::new(leaf(&mut r)));
        gs.exprs.push(match r.below(4) {
            0 => Expr::Add(a, b),
            1 => Expr::Sub(a, b),
            2 => Expr::Mul(a, b),
            _ => Expr::Neg(a),
        });
    }
    for k in 0..1 + r.below(4) {
        gs.constraints.push(GlobalConstraint { expr: r.below(n as u64) as u32, label: format!("c{k}") });
    }
    gs
}

#[test]
fn random_systems_match_the_former_code() {
    for seed in 0..40 {
        assert_same_values(&random_system(seed), 0..4);
    }
}

#[test]
fn fibonacci_square_matches_the_former_code() {
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/../../examples/fibonacci-square/build-a4/build.pilout");
    // PilOutProxy::new panics on a missing file: CI has no fibonacci-square build
    if !std::path::Path::new(path).is_file() {
        eprintln!("skipped: no {path}");
        return;
    }
    let proxy = pil2_pilout::pilout_proxy::PilOutProxy::new(path).unwrap();
    let gs = global_from_pilout(&proxy.pilout, &FieldCfg::goldilocks()).unwrap();
    assert!(!gs.constraints.is_empty());
    assert_same_values(&gs, 0..8);
}

/// Any pilout with global constraints, e.g. ZisK's: `PIL2_GLOBAL_PILOUT=<path> cargo test ... any_pilout`.
#[test]
fn any_pilout_matches_the_former_code() {
    let Ok(path) = std::env::var("PIL2_GLOBAL_PILOUT") else {
        eprintln!("skipped: PIL2_GLOBAL_PILOUT not set");
        return;
    };
    let proxy = pil2_pilout::pilout_proxy::PilOutProxy::new(&path).unwrap();
    let gs = global_from_pilout(&proxy.pilout, &FieldCfg::goldilocks()).unwrap();
    let dag = build_global_dag(&gs).unwrap().expect("global constraints");
    eprintln!(
        "{path}: {} global constraints, {} DAG roots, {} DAG ops",
        gs.constraints.len(),
        dag.roots.len(),
        dag.op_count()
    );
    assert_same_values(&gs, 0..8);
}

#[test]
fn no_constraints_no_dag() {
    let mut gs = random_system(1);
    gs.constraints.clear();
    assert_eq!(build_global_dag(&gs).unwrap(), None);
}

#[test]
fn global_info_carries_the_validated_dag() {
    let gs = random_system(3);
    let dag = build_global_dag(&gs).unwrap().unwrap();
    let gi = serde_json::json!({ "globalConstraintDag": dag });
    assert_eq!(ConstraintDag::from_global_info(&gi, &FieldCfg::goldilocks()).unwrap(), Some(dag.clone()));
    assert_eq!(ConstraintDag::from_global_info(&serde_json::json!({}), &FieldCfg::goldilocks()).unwrap(), None);
    // an air's DAG under the global key is refused
    let mut air = dag;
    air.nodes = vec![DagNode::Public { id: 0 }];
    air.roots = vec![pil_info::core::dag::format::DagRoot {
        node: 0,
        boundary: pil_info::core::dag::format::DagBoundary::EveryRow,
        label: String::new(),
    }];
    let gi = serde_json::json!({ "globalConstraintDag": air });
    assert!(ConstraintDag::from_global_info(&gi, &FieldCfg::goldilocks()).is_err());
}
