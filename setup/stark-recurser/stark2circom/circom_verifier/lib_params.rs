//! The per-key data of the static verifier library: the circuit's main carries one `key_<name>(k)`
//! function per [`FIELDS`] entry ([`key_functions`]), the value for key `k` of the keys it verifies.
//! Data only; FRI/STIR geometry is derived in circom (`verifier/proof_shape.circom`).

use std::collections::HashMap;

use anyhow::{bail, Context, Result};
use proofman_common::deep::{col_groups, fri_ev_groups};
use proofman_common::proof_inputs::{proof_inputs, Origin};
use proofman_common::{EvMapEType, LowDegreeTest, StarkInfo};
use serde_json::Value;

use super::gl_field::{gl_exp, GL_INV_W, GL_W};

/// One L_0 tree the verifier opens: its proof inputs are `s0_vals{suffix}`, `s0_siblings{suffix}`,
/// `s0_last_levels{suffix}`.
struct Tree {
    suffix: String,
    width: u64,
    /// The proof input that holds its root (stage trees).
    root: String,
}

/// `si` parsed (keyVersion-gated), refusing what no verifier can check.
pub(crate) fn parse_stark_info(si: &Value) -> Result<StarkInfo> {
    let info = StarkInfo::try_from_value(si)?;
    // An empty evMap would leave f_0 undefined (the circuit must never check against 0).
    if info.ev_map.is_empty() {
        bail!("evMap has no evaluations, so the DEEP polynomial is undefined");
    }
    if info.stark_struct.split_linear_hash {
        bail!("splitLinearHash has no verifier circuits");
    }
    proof_inputs(&info).map_err(anyhow::Error::msg)?;
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

/// A data value: a number or an array of numbers, as circom literals.
#[derive(Debug)]
pub(crate) enum Val {
    N(String),
    A(Vec<String>),
}

impl Val {
    /// The circom literal; an empty array is `[0]` (circom has no empty arrays).
    fn literal(&self) -> String {
        match self {
            Val::N(x) => x.clone(),
            Val::A(x) if x.is_empty() => "[0]".into(),
            Val::A(x) => format!("[{}]", x.join(", ")),
        }
    }
}

fn n(v: impl ToString) -> Val {
    Val::N(v.to_string())
}

fn a<T: ToString>(v: &[T]) -> Val {
    Val::A(v.iter().map(T::to_string).collect())
}

/// The data the static library reads as `key_<name>(k)` (key `k` of the compilation), in emission order.
/// The one declared table: `lib_gl::tests` checks the library uses exactly these names.
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
    "nOpenings",
    "nPols",
    "nColGroups",
    "nSums",
    "verkeyInput",
    "enableInput",
    "proofSize",
    "offAirgroupValues",
    "offAirValues",
    "offEvals",
    "offFinal",
    "offNonce",
    "offBetas",
    "offNonces",
    "STEPS",
    "STIR_LOG_DOMAIN_SIZES",
    "STIR_FOLDING_FACTORS",
    "STIR_NUM_QUERIES",
    "STIR_GRINDING_BITS",
    "L0_TREE_START",
    "CH",
    "AIR_VALUE_STAGE",
    "AIRGROUP_VALUE_STAGE",
    "OPENING_ROOT",
    "OPENING_NUM_EVALS",
    "EVAL_COLUMN",
    "SUMS",
    "COLS",
    "OPENING_EXPONENT",
    "ROOT_C",
    "OFF_ROOT",
    "OFF_L0_VALUES",
    "OFF_L0_SIBLINGS",
    "OFF_L0_LAST_LEVELS",
    "TREE_ROOT",
    "TREE_PUB",
    "OFF_FOLD_ROOT",
    "OFF_FOLD_VALUES",
    "OFF_FOLD_SIBLINGS",
    "OFF_FOLD_LAST_LEVELS",
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

/// One key's data, one entry per [`FIELDS`] name, and the suffix of its `ConstraintRoots<suffix>`.
#[derive(Debug)]
pub struct KeyData {
    pub(crate) suffix: String,
    pub(crate) fields: Vec<(&'static str, Val)>,
}

/// The DEEP data: openings with evaluations by rank, columns by first appearance in evMap.
struct Deep {
    n_groups: u64,
    open_root: Vec<u64>,
    open: Vec<i64>,
    group_len: Vec<usize>,
    ref_col: Vec<u16>,
    /// (s0 offset, dim) per column.
    leaf: Vec<(u64, u64)>,
    /// A sums shared by prefix: (source, rank) with a source s < nGroups being A_s, else sum s - nGroups.
    sums: Vec<u64>,
    /// Per column group: its sum's source, #columns, (column, offset, dim)...
    cols: Vec<u64>,
    n_col_groups: usize,
}

fn deep_data(si: &StarkInfo, trees: &[Tree], const_tree: usize, s0: &[u64]) -> Result<Deep> {
    let n_bits = si.stark_struct.n_bits as usize;
    let cm_pols = si.cm_pols_map.as_deref().unwrap_or(&[]);
    let groups = fri_ev_groups(si);
    let cgroups = col_groups(&groups);
    let n_groups = groups.len() as u64;
    let rank: HashMap<usize, u64> = groups.iter().enumerate().map(|(r, g)| (g.opening as usize, r as u64)).collect();
    let mut open_root = Vec::new();
    let mut open = Vec::new();
    for g in &groups {
        let o = si.opening_points[g.opening as usize];
        open.push(o);
        let w = if o >= 0 { GL_W[n_bits] } else { GL_INV_W[n_bits] };
        open_root.push(gl_exp(w, o.unsigned_abs()));
    }
    // The k-th ref is evMap[k], which names the column's leaf the first time the column appears.
    let refs: Vec<_> = groups.iter().flat_map(|g| &g.refs).collect();
    let mut leaf: Vec<(u64, u64)> = Vec::new();
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
                (const_tree + 1 + commit, pol.with_context(|| format!("customCommitsMap[{commit}][{id}]"))?.stage_pos)
            }
        };
        leaf.push((s0[tree] + pos, u64::from(r.dim)));
    }
    let mut sums: Vec<u64> = Vec::new();
    let mut sum_of: HashMap<Vec<u64>, u64> = HashMap::new();
    let mut cols: Vec<u64> = Vec::new();
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
    Ok(Deep {
        n_groups,
        open_root,
        open,
        group_len: groups.iter().map(|g| g.refs.len()).collect(),
        ref_col: refs.iter().map(|r| r.col).collect(),
        leaf,
        sums,
        cols,
        n_col_groups: cgroups.len(),
    })
}

/// The flat proof (proof2pointer order): where each read input starts, and its size.
struct Offsets {
    at: HashMap<String, i64>,
    size: i64,
}

impl Offsets {
    fn new(si: &StarkInfo) -> Result<Self> {
        let (mut at, mut size) = (HashMap::new(), 0i64);
        for i in proof_inputs(si).map_err(anyhow::Error::msg)? {
            if i.origin == Origin::Proof {
                at.insert(i.name.clone(), size);
            }
            size += i.words() as i64;
        }
        Ok(Offsets { at, size })
    }

    /// Where input `name` starts, -1 when absent.
    fn of(&self, name: &str) -> i64 {
        self.at.get(name).copied().unwrap_or(-1)
    }

    /// `of` of each name.
    fn all(&self, names: impl Iterator<Item = String>) -> Vec<i64> {
        names.map(|n| self.of(&n)).collect()
    }
}

impl KeyData {
    /// `suffix` names the key's `ConstraintRoots<suffix>` in its verifier file.
    pub(crate) fn new(si: &StarkInfo, o: &KeyOpts, suffix: String) -> Result<Self> {
        let ss = &si.stark_struct;
        // Validates the low-degree schedule before it is indexed.
        let off = Offsets::new(si)?;
        // The query templates see STIR as FRI with one fold tree, T_0, over [log|L_0|, log|L_0| - k_0].
        let stir = ss.low_degree_test.stir();
        let (steps, n_queries, pow_bits, nf) = match &ss.low_degree_test {
            LowDegreeTest::Stir(st) => {
                let l0 = st.log_domain_sizes[0];
                (
                    vec![l0, l0 - st.folding_factors[0]],
                    st.num_queries[0],
                    st.grinding_bits_queries[0],
                    st.num_iterations(),
                )
            }
            LowDegreeTest::Fri(f) => {
                (f.log_domain_sizes.clone(), f.num_queries, f.grinding_bits_queries, f.log_domain_sizes.len() - 1)
            }
        };
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

        let deep = deep_data(si, &trees, const_tree, &s0)?;
        let fl_stir = stir.map_or(0, |s| s.num_iterations());
        let fold = |what: &str| off.all((1..=nf).map(|f| format!("s{f}_{what}")));
        let s0_of = |what: &str| off.all(trees.iter().map(|t| format!("s0_{what}{}", t.suffix)));
        let hints = |kind: &str| off.all((0..fl_stir.saturating_sub(1)).map(|i| format!("{kind}{i}")));
        let mut tree_root = Vec::new();
        let mut tree_pub = Vec::new();
        for (t, tr) in trees.iter().enumerate() {
            let (root, pubs) = if t == const_tree {
                (-1, vec![0; 4])
            } else if t > const_tree {
                (-2, si.custom_commits[t - const_tree - 1].public_values.iter().map(|p| p.idx).collect())
            } else {
                (off.of(&tr.root), vec![0; 4])
            };
            tree_root.push(root);
            tree_pub.extend(pubs);
        }

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
            ("nOpenings", n(deep.n_groups)),
            ("nPols", n(deep.leaf.len())),
            ("nColGroups", n(deep.n_col_groups)),
            ("nSums", n(deep.sums.len() / 2)),
            ("verkeyInput", n(u64::from(o.verkey_input))),
            ("enableInput", n(u64::from(o.enable_input))),
            ("proofSize", n(off.size)),
            ("offAirgroupValues", n(off.of("airgroupvalues"))),
            ("offAirValues", n(off.of("airvalues"))),
            ("offEvals", n(off.of("evals"))),
            ("offFinal", n(off.of("finalPol"))),
            ("offNonce", n(off.of("nonce"))),
            ("offBetas", n(off.of("betas"))),
            ("offNonces", n(off.of("nonces"))),
            ("STEPS", a(&steps)),
            ("STIR_LOG_DOMAIN_SIZES", a(&st(stir.map(|s| &s.log_domain_sizes)))),
            ("STIR_FOLDING_FACTORS", a(&st(stir.map(|s| &s.folding_factors)))),
            ("STIR_NUM_QUERIES", a(&st(stir.map(|s| &s.num_queries)))),
            ("STIR_GRINDING_BITS", a(&st(stir.map(|s| &s.grinding_bits_queries)))),
            ("L0_TREE_START", a(&s0)),
            ("CH", a(&ch)),
            ("AIR_VALUE_STAGE", a(&stages(&si.airvalues_map))),
            ("AIRGROUP_VALUE_STAGE", a(&stages(&si.airgroupvalues_map))),
            ("OPENING_ROOT", a(&deep.open_root)),
            ("OPENING_NUM_EVALS", a(&deep.group_len)),
            ("EVAL_COLUMN", a(&deep.ref_col)),
            ("SUMS", a(&deep.sums)),
            ("COLS", a(&deep.cols)),
            ("OPENING_EXPONENT", a(&deep.open)),
            ("ROOT_C", a(&root_c)),
            ("OFF_ROOT", a(&off.all((1..=q).map(|s| format!("root{s}"))))),
            ("OFF_L0_VALUES", a(&s0_of("vals"))),
            ("OFF_L0_SIBLINGS", a(&s0_of("siblings"))),
            ("OFF_L0_LAST_LEVELS", a(&s0_of("last_levels"))),
            ("TREE_ROOT", a(&tree_root)),
            ("TREE_PUB", a(&tree_pub)),
            ("OFF_FOLD_ROOT", a(&fold("root"))),
            ("OFF_FOLD_VALUES", a(&fold("vals"))),
            ("OFF_FOLD_SIBLINGS", a(&fold("siblings"))),
            ("OFF_FOLD_LAST_LEVELS", a(&fold("last_levels"))),
            ("OFF_ANS", a(&hints("ansCoeffs"))),
            ("OFF_SHAKE", a(&hints("shakeCoeffs"))),
            ("nFolds", n(nf)),
        ];
        debug_assert!(fields.iter().map(|f| f.0).eq(FIELDS.iter().copied()));
        Ok(KeyData { suffix, fields })
    }
}

/// The data of the circuit's keys: `function key_<name>(k)` per [`FIELDS`] entry, returning `keys[k]`'s
/// value (one key: the value; more: one branch per key, the last one unconditional). No keys, no functions.
pub(crate) fn key_functions(keys: &[KeyData]) -> String {
    let Some((last, rest)) = keys.split_last() else {
        return String::new();
    };
    let mut out = String::new();
    for (i, name) in FIELDS.iter().enumerate() {
        let mut body = String::new();
        for (k, key) in rest.iter().enumerate() {
            body += &format!("if (k == {k}) {{ return {}; }} ", key.fields[i].1.literal());
        }
        body += &format!("return {};", last.fields[i].1.literal());
        out += &format!("function key_{name}(k) {{ {body} }}\n");
    }
    out
}
