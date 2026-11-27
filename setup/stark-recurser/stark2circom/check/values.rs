//! The inputs a checked template takes, drawn at random, and the values the constraint DAG gives on them.
//!
//! The layout is the static library's (`ConstraintRootsAt`, `VadcopFinal`, `StarkVerifierBN128`), derived from the
//! starkinfo or globalInfo alone: evaluations in evMap order, challenges in challengesMap order, values by map id.

use std::fs::File;
use std::io::Read;

use anyhow::{anyhow, bail, Context, Result};
use constraint_dag::{
    ev_col_ref, ConstraintDag, DagField, DagNode, GlobalLeaf, GlobalTables, LowerCtx, LowerTables, Operand, ResolvedDag,
};
use proofman_common::{EvMapEType, PolMap, StarkInfo};
use proofman_fields::{CubicExtensionField, Field, Goldilocks, PrimeField64};
use serde_json::Value;

pub type E3 = CubicExtensionField<Goldilocks>;

pub fn base(x: Goldilocks) -> E3 {
    E3 { value: [x, Goldilocks::ZERO, Goldilocks::ZERO] }
}

/// Goldilocks elements from the OS's randomness.
pub struct Rng(File);

impl Rng {
    pub fn new() -> Result<Self> {
        Ok(Rng(File::open("/dev/urandom").context("opening /dev/urandom")?))
    }

    pub fn element(&mut self) -> Result<u64> {
        loop {
            let mut b = [0u8; 8];
            self.0.read_exact(&mut b)?;
            let x = u64::from_le_bytes(b);
            if x < Goldilocks::ORDER_U64 {
                return Ok(x);
            }
        }
    }
}

/// One input signal of the check's main: its name, dimensions and values (flat, row-major), each below the
/// Goldilocks prime.
#[derive(Clone, Debug)]
pub struct Signal {
    pub name: &'static str,
    pub dims: Vec<usize>,
    pub values: Vec<u64>,
}

impl Signal {
    fn random(name: &'static str, dims: Vec<usize>, rng: &mut Rng) -> Result<Self> {
        let values = (0..dims.iter().product()).map(|_| rng.element()).collect::<Result<_>>()?;
        Ok(Signal { name, dims, values })
    }

    pub fn declaration(&self) -> String {
        let dims: String = self.dims.iter().map(|d| format!("[{d}]")).collect();
        format!("    signal input {}{dims};\n", self.name)
    }

    /// Each entry's name, in the order of `values`: `name[i][j]`.
    pub fn entries(&self) -> Vec<String> {
        let mut names = vec![self.name.to_string()];
        for &d in &self.dims {
            names = names.iter().flat_map(|n| (0..d).map(move |i| format!("{n}[{i}]"))).collect();
        }
        names
    }

    /// Entry `i` as a base element.
    fn at(&self, i: usize) -> Option<Goldilocks> {
        self.values.get(i).map(|&v| Goldilocks::new(v))
    }

    /// Entry `i` of a `[n][3]` signal: the extension element, or its first limb at stage 1.
    fn ext(&self, i: usize, stage: u32) -> Option<E3> {
        let limb = |j| self.at(3 * i + j);
        Some(if stage == 1 { base(limb(0)?) } else { E3 { value: [limb(0)?, limb(1)?, limb(2)?] } })
    }

    fn json(&self) -> String {
        fn nest(dims: &[usize], values: &[u64]) -> String {
            match dims.split_first() {
                None => format!("\"{}\"", values[0]),
                Some((&n, rest)) => {
                    let step = rest.iter().product::<usize>();
                    let items: Vec<String> = (0..n).map(|i| nest(rest, &values[i * step..(i + 1) * step])).collect();
                    format!("[{}]", items.join(","))
                }
            }
        }
        format!("\"{}\":{}", self.name, nest(&self.dims, &self.values))
    }
}

/// A main's input file.
pub fn input_json(inputs: &[Signal]) -> String {
    format!("{{{}}}", inputs.iter().map(Signal::json).collect::<Vec<_>>().join(","))
}

pub fn constant(v: &str) -> Option<Goldilocks> {
    let n: i128 = v.parse().ok()?;
    Some(Goldilocks::new(n.rem_euclid(Goldilocks::ORDER_U64 as i128) as u64))
}

fn small(x: u64) -> Result<u32> {
    u32::try_from(x).with_context(|| format!("starkinfo value {x} past u32"))
}

/// What an air's constraint DAG resolves against, from its starkinfo.
pub fn lower_tables(si: &StarkInfo) -> Result<LowerTables> {
    let ev_map = (si.ev_map.iter())
        .map(|e| {
            let ty = match e.type_ {
                EvMapEType::Cm => "cm",
                EvMapEType::Const => "const",
                EvMapEType::Custom => "custom",
            };
            let offset = *si.opening_points.get(e.opening_pos).context("evMap openingPos past openingPoints")?;
            ev_col_ref(ty, Some(small(e.commit_id)?), small(e.id)?, offset).context("evMap entry")
        })
        .collect::<Result<Vec<_>>>()?;
    let challenges = (si.challenges_map.iter().flatten())
        .map(|c| Ok((small(c.stage)?, small(c.stage_id)?, c.name.as_str())))
        .collect::<Result<Vec<_>>>()?;
    let stages = |m: &Option<Vec<PolMap>>| m.iter().flatten().map(|p| small(p.stage)).collect::<Result<Vec<u32>>>();
    let value_stages = [stages(&si.airvalues_map)?, stages(&si.proofvalues_map)?, stages(&si.airgroupvalues_map)?];
    LowerTables::new(ev_map, &challenges, small(si.n_publics)?, value_stages, DagField::Goldilocks)
        .map_err(|e| anyhow!("{e:?}"))
}

/// An air's DAG, checked and resolved against its starkinfo.
pub struct Air {
    pub tables: LowerTables,
    pub dag: ResolvedDag<Operand>,
    pub labels: Vec<String>,
    pub n_roots: usize,
    pub first_root_dim: usize,
}

impl Air {
    pub fn new(si: &StarkInfo, vi: &Value) -> Result<Self> {
        let dag = ConstraintDag::from_verifierinfo(vi, DagField::Goldilocks).map_err(|e| anyhow!("{e:?}"))?;
        Self::from_dag(lower_tables(si)?, &dag)
    }

    pub fn from_dag(tables: LowerTables, dag: &ConstraintDag) -> Result<Self> {
        let ctx = LowerCtx::new(tables.clone()).map_err(|e| anyhow!("{e:?}"))?;
        let first = dag.roots.first().context("the constraint DAG has no roots")?.node;
        let first_root_dim = node_dim(dag, &ctx, first)?;
        let labels = dag.roots.iter().map(|r| r.label.clone()).collect();
        let resolved = ctx.resolve_dag(dag, |_, op| Ok(op)).map_err(|e| anyhow!("{e:?}"))?;
        Ok(Air { tables, dag: resolved, labels, n_roots: dag.roots.len(), first_root_dim })
    }

    /// `ConstraintRootsAt`'s inputs: evals, challenges, publics, proof, air and airgroup values (an empty one
    /// padded to one entry).
    pub fn random_inputs(&self, rng: &mut Rng) -> Result<Vec<Signal>> {
        let t = &self.tables;
        let one = |n: usize| n.max(1);
        Ok(vec![
            Signal::random("evals", vec![t.ev_map.len(), 3], rng)?,
            Signal::random("challenges", vec![t.challenges.len(), 3], rng)?,
            Signal::random("publics", vec![one(t.n_publics as usize)], rng)?,
            Signal::random("proofvalues", vec![one(t.proof_values_stage.len()), 3], rng)?,
            Signal::random("airvalues", vec![one(t.air_values_stage.len()), 3], rng)?,
            Signal::random("airgroupvalues", vec![one(t.airgroup_values_stage.len()), 3], rng)?,
        ])
    }

    /// The value of a leaf on `inputs`, read by the input's name.
    fn leaf(inputs: &[Signal], op: &Operand) -> Option<E3> {
        let input = |name: &str| inputs.iter().find(|s| s.name == name);
        match *op {
            Operand::Number(ref v) => constant(v).map(base),
            Operand::Eval(k) => input("evals")?.ext(k as usize, 3),
            Operand::Challenge { id, .. } => input("challenges")?.ext(id as usize, 3),
            Operand::Public(k) => input("publics")?.at(k as usize).map(base),
            Operand::ProofValue { id, stage } => input("proofvalues")?.ext(id as usize, stage),
            Operand::AirValue { id, stage } => input("airvalues")?.ext(id as usize, stage),
            Operand::AirgroupValue { id, stage } => input("airgroupvalues")?.ext(id as usize, stage),
            Operand::Tmp(_) | Operand::Zi { .. } => None,
        }
    }

    pub fn roots(&self, inputs: &[Signal]) -> Result<Vec<E3>> {
        self.dag.eval_roots(|op| Self::leaf(inputs, op)).context("a DAG leaf reads past the inputs")
    }

    /// The roots folded by `std_vc`, times `Zh`: BN128's `ConstraintEval` output.
    pub fn folded_times_zh(&self, inputs: &[Signal]) -> Result<E3> {
        let zh = inputs.iter().find(|s| s.name == "Zh").and_then(|zh| zh.ext(0, 3)).context("no input Zh")?;
        let folded = self.dag.eval(|op| Self::leaf(inputs, op)).context("a DAG leaf reads past the inputs")?;
        Ok(folded * zh)
    }

    /// `ConstraintEval`'s inputs: challenges, evals, publics (possibly none) and Zh.
    pub fn random_bn128_inputs(&self, rng: &mut Rng) -> Result<Vec<Signal>> {
        let t = &self.tables;
        Ok(vec![
            Signal::random("challenges", vec![t.challenges.len(), 3], rng)?,
            Signal::random("evals", vec![t.ev_map.len(), 3], rng)?,
            Signal::random("publics", vec![t.n_publics as usize], rng)?,
            Signal::random("Zh", vec![3], rng)?,
        ])
    }
}

/// Dimension of operation node `i`: the widest of its operands'.
fn node_dim(dag: &ConstraintDag, ctx: &LowerCtx, i: u32) -> Result<usize> {
    let mut dims: Vec<usize> = Vec::with_capacity(i as usize + 1);
    for (j, n) in dag.nodes.iter().enumerate().take(i as usize + 1) {
        let d = match *n {
            DagNode::Add { a, b } | DagNode::Sub { a, b } | DagNode::Mul { a, b } => {
                dims[a as usize].max(dims[b as usize])
            }
            DagNode::Neg { a } => dims[a as usize],
            ref leaf => ctx.dim(j as u32, leaf).map_err(|e| anyhow!("{e:?}"))?,
        };
        dims.push(d);
    }
    Ok(dims[i as usize])
}

/// The global constraint DAG of a globalInfo and the shape of `GlobalConstraints`' inputs.
pub struct Global {
    pub tables: GlobalTables,
    /// `None`: no global constraints.
    pub dag: Option<(ConstraintDag, Vec<Option<GlobalLeaf>>)>,
    /// (nAg, maxAgg, nPub, nPV, nCh), as `VadcopFinal` instantiates `GlobalConstraints`.
    pub params: [usize; 5],
}

impl Global {
    pub fn new(gi: &Value) -> Result<Self> {
        let u32s = |v: &Value| v.as_u64().and_then(|n| u32::try_from(n).ok());
        let stages =
            |list: &Value| -> Option<Vec<u32>> { list.as_array()?.iter().map(|v| u32s(&v["stage"])).collect() };
        let agg_types = gi["aggTypes"].as_array().context("globalInfo has no aggTypes")?;
        let tables = GlobalTables {
            n_publics: u32s(&gi["nPublics"]).context("globalInfo has no nPublics")?,
            num_challenges: (gi["numChallenges"].as_array().into_iter().flatten())
                .map(|n| u32s(n).with_context(|| format!("numChallenges: {n}")))
                .collect::<Result<_>>()?,
            proof_values_stage: match gi.get("proofValuesMap") {
                Some(m) => stages(m).context("proofValuesMap without stages")?,
                None => Vec::new(),
            },
            airgroup_values_stage: agg_types
                .iter()
                .map(|g| stages(g).context("aggTypes without stages"))
                .collect::<Result<_>>()?,
            ext_dim: 3,
        };
        let dag = match ConstraintDag::from_global_info(gi, DagField::Goldilocks).map_err(|e| anyhow!("{e:?}"))? {
            Some(dag) => {
                let leaves = tables.resolve_all(&dag).map_err(|e| anyhow!("{e:?}"))?;
                Some((dag, leaves))
            }
            None => None,
        };
        let max_agg = tables.airgroup_values_stage.iter().map(Vec::len).max().unwrap_or(0).max(1);
        let params = [
            agg_types.len(),
            max_agg,
            tables.n_publics as usize,
            tables.proof_values_stage.len(),
            tables.num_challenges.get(1).copied().unwrap_or(0) as usize,
        ];
        Ok(Global { tables, dag, params })
    }

    pub fn random_inputs(&self, rng: &mut Rng) -> Result<Vec<Signal>> {
        let [n_ag, max_agg, n_pub, n_pv, n_ch] = self.params;
        Ok(vec![
            Signal::random("airgroupvalues", vec![n_ag, max_agg, 3], rng)?,
            Signal::random("publics", vec![n_pub], rng)?,
            Signal::random("proofValues", vec![n_pv, 3], rng)?,
            Signal::random("challenges", vec![n_ch, 3], rng)?,
        ])
    }

    /// Each root's value on `inputs`; `VadcopFinal` hands `GlobalConstraints` only the stage-2 challenges.
    pub fn roots(&self, inputs: &[Signal]) -> Result<Vec<E3>> {
        let Some((dag, leaves)) = &self.dag else { return Ok(vec![]) };
        let [airgroup_values, publics, proof_values, challenges] = inputs else {
            bail!("not GlobalConstraints' inputs")
        };
        let first_stage2 = self.tables.num_challenges.first().copied().unwrap_or(0);
        let max_agg = self.params[1];
        let mut v: Vec<E3> = Vec::with_capacity(dag.nodes.len());
        for (i, (n, leaf)) in dag.nodes.iter().zip(leaves).enumerate() {
            let x = match (leaf, n) {
                (Some(GlobalLeaf::Number(c)), _) => constant(c).map(base),
                (Some(GlobalLeaf::Public(k)), _) => publics.at(*k as usize).map(base),
                (Some(GlobalLeaf::Challenge(k)), _) => {
                    let k = k.checked_sub(first_stage2).filter(|&k| (k as usize) < self.params[4]);
                    k.and_then(|k| challenges.ext(k as usize, 3))
                }
                (Some(GlobalLeaf::ProofValue { id, stage }), _) => proof_values.ext(*id as usize, *stage),
                (Some(GlobalLeaf::AirgroupValue { airgroup, id, stage }), _) => {
                    airgroup_values.ext(*airgroup as usize * max_agg + *id as usize, *stage)
                }
                (None, &DagNode::Add { a, b }) => Some(v[a as usize] + v[b as usize]),
                (None, &DagNode::Sub { a, b }) => Some(v[a as usize] - v[b as usize]),
                (None, &DagNode::Mul { a, b }) => Some(v[a as usize] * v[b as usize]),
                (None, &DagNode::Neg { a }) => Some(-v[a as usize]),
                (None, _) => None,
            };
            v.push(x.with_context(|| format!("global node {i} reads past vadcop_final's inputs"))?);
        }
        Ok(dag.roots.iter().map(|r| v[r.node as usize]).collect())
    }
}

pub fn limbs(x: &E3) -> [u64; 3] {
    x.value.map(|l| l.as_canonical_u64())
}
