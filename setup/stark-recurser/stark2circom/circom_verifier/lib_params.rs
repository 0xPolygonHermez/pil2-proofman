//! The per-key data of the static verifier library: one `sv<sfx>_<name>()` function per [`FIELDS`] entry,
//! read by the library as `svd_<name>(k)` through the compilation's [`dispatch`], and the proof inputs.
//! Data only; FRI/STIR geometry is derived in circom (`verifier/geometry.circom`).

use std::collections::HashMap;

use anyhow::{bail, Context, Result};
use proofman_common::deep::{col_groups, fri_ev_groups};
use proofman_common::{EvMapEType, LowDegreeTest, StarkInfo};
use serde_json::Value;

use super::gl_field::{gl_exp, GL_INV_W, GL_W};

/// One L_0 tree: its proof inputs are `s0_vals{suffix}`, `s0_siblings{suffix}`, `s0_last_mt_levels{suffix}`.
struct Tree {
    suffix: String,
    width: u64,
    /// The proof input that holds its root (stage trees).
    root: String,
}

/// Where a proof input's value comes from in a circuit that verifies the proof.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum Origin {
    /// The proof's flat buffer.
    Proof,
    /// The circuit's own `publics` (or `{prefix}_publics`).
    Publics,
    /// The circuit's `proofValues`.
    ProofValues,
}

/// A verifier input: its name and shape. The single source of the verifier's declarations and of
/// the circuits' `define_stark_inputs`/`assign_stark_inputs`.
pub(crate) struct ProofInput {
    pub name: String,
    pub dims: Vec<u64>,
    pub origin: Origin,
}

/// `si` parsed, refusing what no verifier can check.
pub(crate) fn parse_stark_info(si: &Value) -> Result<StarkInfo> {
    let info: StarkInfo = serde_json::from_value(si.clone()).context("starkinfo")?;
    // An empty evMap would leave f_0 undefined (the circuit must never check against 0).
    if info.ev_map.is_empty() {
        bail!("evMap has no evaluations, so the DEEP polynomial is undefined");
    }
    Ok(info)
}

/// The L_0 trees in evaluation order (stage trees with columns and Q, fixed, custom commits) and
/// the index of the fixed one.
fn l0_trees(si: &StarkInfo) -> Result<(Vec<Tree>, usize)> {
    let q = u64::from(si.n_stages) + 1;
    let sec = |k: String| si.map_sections_n.get(&k).copied().unwrap_or(0);
    let mut trees: Vec<Tree> = (1..=q)
        .filter(|&s| s == q || sec(format!("cm{s}")) > 0)
        .map(|s| Tree { suffix: s.to_string(), width: sec(format!("cm{s}")), root: format!("root{s}") })
        .collect();
    let const_tree = trees.len();
    trees.push(Tree { suffix: "C".into(), width: si.n_constants, root: "rootC".into() });
    for cc in &si.custom_commits {
        if cc.public_values.len() != 4 {
            bail!("custom commit {}: its root needs 4 publics", cc.name);
        }
        trees.push(Tree {
            suffix: format!("_{}_0", cc.name),
            width: sec(format!("{}0", cc.name)),
            root: String::new(),
        });
    }
    Ok((trees, const_tree))
}

/// The verifier's inputs in `Proofs::proof2pointer` order (publics and proof values, which the proof
/// does not carry, where the circuits wire them). A BN128 STARK has single-element roots and Poseidon
/// paths; a Merkle path with no level below the ones sent in clear has no input.
pub(crate) fn proof_inputs(si: &StarkInfo) -> Result<Vec<ProofInput>> {
    let ss = &si.stark_struct;
    let bn128 = ss.verification_hash_type == "BN128";
    let arity = ss.merkle_tree_arity;
    if arity < 2 || !arity.is_power_of_two() {
        bail!("merkleTreeArity {arity} is not a power of two");
    }
    let ll = ss.last_level_verification;
    let (root, sib_w, ll_name) =
        if bn128 { (vec![], arity, "last_levels") } else { (vec![4], (arity - 1) * 4, "last_mt_levels") };
    let last = if bn128 { vec![arity.pow(ll as u32)] } else { vec![arity.pow(ll as u32), 4] };
    let mut v = Inputs { v: Vec::new(), arity, ll, sib_w, last };

    let len = |m: &Option<Vec<proofman_common::PolMap>>| m.as_ref().map_or(0, Vec::len) as u64;
    if si.n_publics > 0 {
        v.push("publics".into(), vec![si.n_publics], Origin::Publics);
    }
    for (name, n) in [("airgroupvalues", len(&si.airgroupvalues_map)), ("airvalues", len(&si.airvalues_map))] {
        if n > 0 {
            v.push(name.into(), vec![n, 3], Origin::Proof);
        }
    }
    if len(&si.proofvalues_map) > 0 {
        v.push("proofvalues".into(), vec![len(&si.proofvalues_map), 3], Origin::ProofValues);
    }
    for s in 1..=u64::from(si.n_stages) + 1 {
        v.push(format!("root{s}"), root.clone(), Origin::Proof);
    }
    v.push("evals".into(), vec![si.ev_map.len() as u64, 3], Origin::Proof);

    let (l0, nq) = match &ss.low_degree_test {
        LowDegreeTest::Stir(_) if bn128 => bail!("STIR is not supported for BN128 proofs"),
        LowDegreeTest::Stir(st) => (st.log_domain_sizes.first(), st.num_queries.first().copied().unwrap_or(0)),
        LowDegreeTest::Fri(f) => (f.log_domain_sizes.first(), f.num_queries),
    };
    let l0 = *l0.context("malformed low-degree test: no logDomainSizes")?;
    // proof2pointer: the fixed tree, the custom commits, then the stage trees.
    let (trees, c) = l0_trees(si)?;
    for t in std::iter::once(&trees[c]).chain(&trees[c + 1..]).chain(&trees[..c]) {
        let s = &t.suffix;
        v.tree([format!("s0_vals{s}"), format!("s0_siblings{s}"), format!("s0_{ll_name}{s}")], nq, t.width, l0);
    }
    match &ss.low_degree_test {
        LowDegreeTest::Stir(st) => {
            let m = st.num_iterations();
            let lens = [st.log_domain_sizes.len(), st.log_degrees.len(), st.num_queries.len() + 1];
            if m == 0 || lens != [m + 1; 3] {
                bail!("malformed STIR schedule: {m} folds, lengths {lens:?}");
            }
            for f in 1..=m {
                v.push(format!("s{f}_root"), root.clone(), Origin::Proof);
            }
            for (r, &k) in st.folding_factors.iter().enumerate() {
                let f = r + 1;
                let names = [format!("s{f}_vals"), format!("s{f}_siblings"), format!("s{f}_last_levels")];
                v.tree(names, st.num_queries[r], 3 << k, st.log_domain_sizes[r] - k);
            }
            if m > 1 {
                v.push("betas".into(), vec![m as u64 - 1, 3], Origin::Proof);
            }
            v.push("finalPol".into(), vec![1 << st.log_degrees[m], 3], Origin::Proof);
            v.push("nonces".into(), vec![m as u64], Origin::Proof);
            for kind in ["ansCoeffs", "shakeCoeffs"] {
                for (i, &t) in st.num_queries.iter().enumerate().take(m - 1) {
                    v.push(format!("{kind}{i}"), vec![t + 1, 3], Origin::Proof);
                }
            }
        }
        LowDegreeTest::Fri(fri) => {
            let steps = &fri.log_domain_sizes;
            for f in 1..steps.len() {
                v.push(format!("s{f}_root"), root.clone(), Origin::Proof);
            }
            for f in 1..steps.len() {
                let names = [format!("s{f}_vals"), format!("s{f}_siblings"), format!("s{f}_{ll_name}")];
                v.tree(names, nq, 3 << (steps[f - 1] - steps[f]), steps[f]);
            }
            v.push("finalPol".into(), vec![1 << steps[steps.len() - 1], 3], Origin::Proof);
            if fri.grinding_bits_queries > 0 {
                v.push("nonce".into(), vec![], Origin::Proof);
            }
        }
    }
    Ok(v.v)
}

/// [`proof_inputs`]' accumulator, with the key's Merkle shapes.
struct Inputs {
    v: Vec<ProofInput>,
    arity: u64,
    ll: u64,
    sib_w: u64,
    last: Vec<u64>,
}

impl Inputs {
    fn push(&mut self, name: String, dims: Vec<u64>, origin: Origin) {
        self.v.push(ProofInput { name, dims, origin });
    }

    /// An opened tree over 2^bits leaves: n rows of `width` values, their paths, the levels sent in clear.
    fn tree(&mut self, [vals, sibs, last]: [String; 3], n: u64, width: u64, bits: u64) {
        let levels = bits.div_ceil(u64::from(self.arity.trailing_zeros())).saturating_sub(self.ll);
        self.push(vals, vec![n, width], Origin::Proof);
        if levels > 0 {
            self.push(sibs, vec![n, levels, self.sib_w], Origin::Proof);
        }
        if self.ll > 0 {
            self.push(last, self.last.clone(), Origin::Proof);
        }
    }
}

/// A data value: a number or an array of numbers, as circom literals.
pub(crate) enum Val {
    N(String),
    A(Vec<String>),
}

fn n(v: impl ToString) -> Val {
    Val::N(v.to_string())
}

fn a<T: ToString>(v: &[T]) -> Val {
    Val::A(v.iter().map(T::to_string).collect())
}

/// The data the static library reads as `svd_<name>(k)` (key `k` of the compilation), in emission order.
/// The one declared table: `lib_params::tests` checks the library uses exactly these names.
pub(crate) const FIELDS: &[&str] = &[
    "nBits",
    "nBitsExt",
    "arity",
    "lastLevels",
    "nQueries",
    "batchSize",
    "powBits",
    "qDeg",
    "qPolEv",
    "nEvals",
    "nPublics",
    "nAirValues",
    "nAirgroupValues",
    "nProofValues",
    "nStages",
    "nChallenges",
    "hashCommits",
    "circuitType",
    "nRoots",
    "firstRootDim",
    "nSteps",
    "stir",
    "stirM",
    "stirFinalBits",
    "nTrees",
    "nGroups",
    "nPols",
    "nColGroups",
    "nSums",
    "verkeyInput",
    "enableInput",
    "proofSize",
    "offAGV",
    "offAV",
    "offEvals",
    "offFinal",
    "offNonce",
    "offBetas",
    "offNonces",
    "STEPS",
    "STIR_L",
    "STIR_K",
    "STIR_T",
    "STIR_G",
    "S0",
    "CH",
    "AV",
    "AGV",
    "OPEN_ROOT",
    "GROUP_LEN",
    "REF_COL",
    "SUMS",
    "COLS",
    "OPEN",
    "ROOT_C",
    "OFF_ROOT",
    "OFF_S0V",
    "OFF_S0S",
    "OFF_S0L",
    "TREE_ROOT",
    "TREE_PUB",
    "OFF_FROOT",
    "OFF_FV",
    "OFF_FS",
    "OFF_FL",
    "OFF_ANS",
    "OFF_SHAKE",
    "nFolds",
];

/// How a key's verifier is used by the circuit that includes it.
#[derive(Default)]
pub(crate) struct KeyOpts {
    /// Starts the transcript from the global challenge and absorbs `circuitType` (a basic air).
    pub input_challenges: bool,
    pub batch_size: Option<usize>,
    /// (nRoots, firstRootDim, vc) of `ConstraintRoots` (GL; BN128 folds them in `ConstraintEval`).
    pub roots: Option<(usize, usize, usize)>,
    pub verkey_input: bool,
    pub enable_input: bool,
    /// The verification key, when it is not an input.
    pub const_root: Option<[String; 4]>,
}

/// The key's data, one entry per [`FIELDS`] name.
pub(crate) struct KeyData {
    pub fields: Vec<(&'static str, Val)>,
}

impl KeyData {
    pub fn new(si: &StarkInfo, o: &KeyOpts) -> Result<Self> {
        let ss = &si.stark_struct;
        let bn128 = ss.verification_hash_type == "BN128";
        // The query templates see STIR as FRI with one fold tree, T_0, over [log|L_0|, log|L_0| - k_0].
        let stir = ss.low_degree_test.stir();
        let (steps, n_queries, pow_bits) = match &ss.low_degree_test {
            LowDegreeTest::Stir(stir) => {
                let m = stir.num_iterations();
                let lens = [stir.log_domain_sizes.len(), stir.log_degrees.len(), stir.num_queries.len() + 1];
                if m == 0 || lens != [m + 1; 3] || stir.grinding_bits_queries.len() != m {
                    bail!("malformed STIR schedule: {m} folds, lengths {lens:?}");
                }
                let l0 = stir.log_domain_sizes[0];
                (vec![l0, l0 - stir.folding_factors[0]], stir.num_queries[0], stir.grinding_bits_queries[0])
            }
            LowDegreeTest::Fri(fri) => (fri.log_domain_sizes.clone(), fri.num_queries, fri.grinding_bits_queries),
        };
        if steps.is_empty() {
            bail!("malformed FRI schedule: no logDomainSizes");
        }
        let n_stages = u64::from(si.n_stages);
        let q = n_stages + 1;
        let circuit_type = if o.input_challenges {
            if n_stages < 2 {
                bail!("a basic air needs stage 2 to absorb its circuitType after (nStages = {n_stages})");
            }
            si.circuit_type.filter(|&t| t > 0).context("starkInfo.circuitType missing: regenerate the setup")?
        } else {
            0
        };

        let (trees, const_tree) = l0_trees(si)?;
        // The static query templates have 8 slots of each kind (stark_verifier.circom).
        let opened = if stir.is_some() { 1 } else { steps.len() - 1 };
        if trees.len() > 8 || opened > 8 {
            bail!("{} L_0 trees and {opened} fold trees: the verifier library opens at most 8 of each", trees.len());
        }
        let mut s0 = vec![0];
        for t in &trees {
            s0.push(s0.last().unwrap() + t.width);
        }

        let cm_pols = si.cm_pols_map.as_deref().unwrap_or(&[]);
        let q_index = cm_pols
            .iter()
            .position(|p| p.stage == q && p.stage_id == 0)
            .context("cmPolsMap has no quotient polynomial")?;
        let q_pol_ev = si
            .ev_map
            .iter()
            .position(|e| e.type_ == EvMapEType::Cm && e.id == q_index as u64)
            .context("evMap has no quotient evaluation")?;

        // The transcript squeezes challengesMap in order: stages 2..nStages, then vc, xi, vf1, vf2.
        let challenges: Vec<(u64, u64)> = si.challenges_map.iter().flatten().map(|c| (c.stage, c.stage_id)).collect();
        let mut ch = vec![0u64; q as usize];
        for &(stage, _) in challenges.iter().filter(|c| (2..=n_stages).contains(&c.0)) {
            ch[stage as usize] += 1;
        }
        let drawn: Vec<(u64, u64)> = (2..=n_stages)
            .flat_map(|s| (0..ch[s as usize]).map(move |id| (s, id)))
            .chain([(q, 0), (q + 1, 0), (q + 2, 0), (q + 2, 1)])
            .collect();
        if challenges != drawn {
            bail!("challengesMap {challenges:?} is not the order the verifier transcript draws {drawn:?}");
        }
        if o.roots.is_some_and(|(_, _, vc)| vc != challenges.len() - 4) {
            bail!("std_vc must be the quotient-stage challenge {q}.0");
        }

        // DEEP: openings with evaluations by rank, columns by first appearance in evMap.
        let groups = fri_ev_groups(si);
        let cgroups = col_groups(&groups);
        let n_groups = groups.len() as u64;
        let rank: HashMap<usize, u64> =
            groups.iter().enumerate().map(|(r, g)| (g.opening as usize, r as u64)).collect();
        let mut open_root = Vec::new();
        let mut open = Vec::new();
        for g in &groups {
            let o = si.opening_points[g.opening as usize];
            open.push(o);
            let w = if o >= 0 { GL_W[ss.n_bits as usize] } else { GL_INV_W[ss.n_bits as usize] };
            open_root.push(gl_exp(w, o.unsigned_abs()));
        }
        // The k-th ref is evMap[k], which names the column's leaf the first time the column appears.
        let refs: Vec<_> = groups.iter().flat_map(|g| &g.refs).collect();
        let mut leaf: Vec<(u64, u64)> = Vec::new(); // (s0 offset, dim) per column
        for (ev, r) in si.ev_map.iter().zip(&refs) {
            if r.col as usize != leaf.len() {
                continue;
            }
            let (id, commit) = (ev.id as usize, ev.commit_id as usize);
            let (tree, pos) = match ev.type_ {
                EvMapEType::Const => (const_tree, ev.id),
                EvMapEType::Cm => {
                    let pol = cm_pols.get(id).with_context(|| format!("cmPolsMap[{id}]"))?;
                    let t = trees[..const_tree].iter().position(|t| t.suffix == pol.stage.to_string());
                    (t.with_context(|| format!("no tree for stage {}", pol.stage))?, pol.stage_pos)
                }
                EvMapEType::Custom => {
                    let pol = si.custom_commits_map.get(commit).and_then(|m| m.as_ref()?.get(id));
                    (
                        const_tree + 1 + commit,
                        pol.with_context(|| format!("customCommitsMap[{commit}][{id}]"))?.stage_pos,
                    )
                }
            };
            leaf.push((s0[tree] + pos, u64::from(r.dim)));
        }
        // A sums shared by prefix: (source, rank) with a source s < nGroups being A_s, else sum s - nGroups.
        let mut sums: Vec<u64> = Vec::new();
        let mut sum_of: HashMap<Vec<u64>, u64> = HashMap::new();
        let mut cols: Vec<u64> = Vec::new(); // per column group: its sum's source, #columns, (column, offset, dim)...
        for cg in &cgroups {
            let ranks: Vec<u64> = cg.openings.iter().map(|o| rank[o]).collect();
            let mut acc = ranks[0];
            for j in 1..ranks.len() {
                acc = match sum_of.get(&ranks[..=j]) {
                    Some(&s) => s,
                    None => {
                        let s = n_groups + sums.len() as u64 / 2;
                        sums.extend([acc, ranks[j]]);
                        sum_of.insert(ranks[..=j].to_vec(), s);
                        s
                    }
                };
            }
            cols.extend([acc, cg.cols.len() as u64]);
            for r in &cg.cols {
                let (off, dim) = leaf[r.col as usize];
                cols.extend([u64::from(r.col), off, dim]);
            }
        }

        // The flat proof (proof2pointer order): where each input starts, -1 when absent.
        let mut at: HashMap<String, i64> = HashMap::new();
        let mut size = 0i64;
        for i in proof_inputs(si)?.iter().filter(|i| i.origin == Origin::Proof) {
            at.insert(i.name.clone(), size);
            size += i.dims.iter().product::<u64>() as i64;
        }
        let off = |name: &str| at.get(name).copied().unwrap_or(-1);
        let ll = if bn128 { "last_levels" } else { "last_mt_levels" };
        let nf = match &ss.low_degree_test {
            LowDegreeTest::Stir(s) => s.num_iterations(),
            LowDegreeTest::Fri(_) => steps.len() - 1,
        };
        let (fl, fl_stir) = (if stir.is_some() { "last_levels" } else { ll }, stir.map_or(0, |s| s.num_iterations()));
        let fold = |what: &str| -> Vec<i64> {
            (1..=nf).map(|f| off(&format!("s{f}_{}", if what == "last" { fl } else { what }))).collect()
        };
        let mut tree_root = Vec::new();
        let mut tree_pub = Vec::new();
        for (t, tr) in trees.iter().enumerate() {
            let (root, pubs) = if t == const_tree {
                (-1, vec![0; 4])
            } else if t > const_tree {
                (-2, si.custom_commits[t - const_tree - 1].public_values.iter().map(|p| p.idx).collect())
            } else {
                (off(&tr.root), vec![0; 4])
            };
            tree_root.push(root);
            tree_pub.extend(pubs);
        }
        let hints =
            |kind: &str| -> Vec<i64> { (0..fl_stir.saturating_sub(1)).map(|i| off(&format!("{kind}{i}"))).collect() };

        let batch =
            o.batch_size.unwrap_or_else(|| (n_queries as usize).div_ceil(8)).clamp(1, n_queries.max(1) as usize);
        let len = |m: &Option<Vec<_>>| m.as_ref().map_or(0, Vec::len) as u64;
        let stages =
            |m: &Option<Vec<proofman_common::PolMap>>| m.iter().flatten().map(|p| p.stage).collect::<Vec<u64>>();
        let st = |v: Option<&Vec<u64>>| v.cloned().unwrap_or_default();
        let root_c = o.const_root.clone().unwrap_or_else(|| ["0".into(), "0".into(), "0".into(), "0".into()]);
        let fields = vec![
            ("nBits", n(ss.n_bits)),
            ("nBitsExt", n(ss.n_bits_ext)),
            ("arity", n(ss.merkle_tree_arity)),
            ("lastLevels", n(ss.last_level_verification)),
            ("nQueries", n(n_queries)),
            ("batchSize", n(batch)),
            ("powBits", n(pow_bits)),
            ("qDeg", n(si.q_deg)),
            ("qPolEv", n(q_pol_ev)),
            ("nEvals", n(si.ev_map.len())),
            ("nPublics", n(si.n_publics)),
            ("nAirValues", n(len(&si.airvalues_map))),
            ("nAirgroupValues", n(len(&si.airgroupvalues_map))),
            ("nProofValues", n(len(&si.proofvalues_map))),
            ("nStages", n(n_stages)),
            ("nChallenges", n(challenges.len())),
            ("hashCommits", n(u64::from(ss.hash_commits))),
            ("circuitType", n(circuit_type)),
            ("nRoots", n(o.roots.map_or(0, |r| r.0))),
            ("firstRootDim", n(o.roots.map_or(0, |r| r.1))),
            ("nSteps", n(steps.len())),
            ("stir", n(u64::from(stir.is_some()))),
            ("stirM", n(fl_stir)),
            ("stirFinalBits", n(stir.map_or(0, |s| *s.log_degrees.last().unwrap()))),
            ("nTrees", n(trees.len())),
            ("nGroups", n(n_groups)),
            ("nPols", n(leaf.len())),
            ("nColGroups", n(cgroups.len())),
            ("nSums", n(sums.len() / 2)),
            ("verkeyInput", n(u64::from(o.verkey_input))),
            ("enableInput", n(u64::from(o.enable_input))),
            ("proofSize", n(size)),
            ("offAGV", n(off("airgroupvalues"))),
            ("offAV", n(off("airvalues"))),
            ("offEvals", n(off("evals"))),
            ("offFinal", n(off("finalPol"))),
            ("offNonce", n(off("nonce"))),
            ("offBetas", n(off("betas"))),
            ("offNonces", n(off("nonces"))),
            ("STEPS", a(&steps)),
            ("STIR_L", a(&st(stir.map(|s| &s.log_domain_sizes)))),
            ("STIR_K", a(&st(stir.map(|s| &s.folding_factors)))),
            ("STIR_T", a(&st(stir.map(|s| &s.num_queries)))),
            ("STIR_G", a(&st(stir.map(|s| &s.grinding_bits_queries)))),
            ("S0", a(&s0)),
            ("CH", a(&ch)),
            ("AV", a(&stages(&si.airvalues_map))),
            ("AGV", a(&stages(&si.airgroupvalues_map))),
            ("OPEN_ROOT", a(&open_root)),
            ("GROUP_LEN", a(&groups.iter().map(|g| g.refs.len()).collect::<Vec<_>>())),
            ("REF_COL", a(&refs.iter().map(|r| r.col).collect::<Vec<_>>())),
            ("SUMS", a(&sums)),
            ("COLS", a(&cols)),
            ("OPEN", a(&open)),
            ("ROOT_C", a(&root_c)),
            ("OFF_ROOT", a(&(1..=q).map(|s| off(&format!("root{s}"))).collect::<Vec<_>>())),
            ("OFF_S0V", a(&trees.iter().map(|t| off(&format!("s0_vals{}", t.suffix))).collect::<Vec<_>>())),
            ("OFF_S0S", a(&trees.iter().map(|t| off(&format!("s0_siblings{}", t.suffix))).collect::<Vec<_>>())),
            ("OFF_S0L", a(&trees.iter().map(|t| off(&format!("s0_{ll}{}", t.suffix))).collect::<Vec<_>>())),
            ("TREE_ROOT", a(&tree_root)),
            ("TREE_PUB", a(&tree_pub)),
            ("OFF_FROOT", a(&fold("root"))),
            ("OFF_FV", a(&fold("vals"))),
            ("OFF_FS", a(&fold("siblings"))),
            ("OFF_FL", a(&fold("last"))),
            ("OFF_ANS", a(&hints("ansCoeffs"))),
            ("OFF_SHAKE", a(&hints("shakeCoeffs"))),
            ("nFolds", n(nf)),
        ];
        debug_assert!(fields.iter().map(|f| f.0).eq(FIELDS.iter().copied()));
        Ok(KeyData { fields })
    }

    /// `function sv{sfx}_<name>() { return <value>; }` per field (empty arrays padded to one entry).
    pub fn functions(&self, sfx: &str) -> String {
        let mut out = String::new();
        for (name, v) in &self.fields {
            let v = match v {
                Val::N(x) => x.clone(),
                Val::A(x) if x.is_empty() => "[0]".into(),
                Val::A(x) => format!("[{}]", x.join(", ")),
            };
            out += &format!("function sv{sfx}_{name}() {{ return {v}; }}\n");
        }
        out
    }
}

/// The compilation's `svd_<name>(k)`: field `name` of its k-th key, whose functions carry `sfxs[k]`.
pub(crate) fn dispatch(sfxs: &[String]) -> String {
    let mut out = String::new();
    for name in FIELDS {
        let mut body = String::new();
        for (k, s) in sfxs.iter().enumerate().take(sfxs.len().saturating_sub(1)) {
            body += &format!("if (k == {k}) {{ return sv{s}_{name}(); }} ");
        }
        let last = sfxs.last().map_or(String::new(), |s| format!("return sv{s}_{name}();"));
        out += &format!("function svd_{name}(k) {{ {body}{last} }}\n");
    }
    out
}
