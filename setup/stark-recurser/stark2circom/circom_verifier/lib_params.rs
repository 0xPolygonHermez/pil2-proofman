//! The per-key data of the static verifier library: the `var` block of `StarkVerifier<suffix>`, one
//! var per library template parameter (`lib_gl::call` passes them by name), and the proof inputs.
//! Data only; FRI/STIR geometry is derived in circom (`verifier/geometry.circom`).

use std::collections::HashMap;

use anyhow::{bail, Context, Result};
use proofman_common::deep::{col_groups, fri_ev_groups};
use proofman_common::{EvMapEType, LowDegreeTest, StarkInfo};
use serde_json::Value;

use super::gl_field::{gl_exp, GL_INV_W, GL_W};

/// One L_0 tree: its proof inputs are `s0_vals{suffix}`, `s0_siblings{suffix}`, `s0_last_mt_levels{suffix}`.
pub(crate) struct Tree {
    pub suffix: String,
    pub width: u64,
    /// The signal that holds its root.
    pub root: String,
}

pub(crate) struct KeyData {
    /// `var name = value;` lines, named after the library's template parameters.
    pub vars: String,
    pub trees: Vec<Tree>,
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

impl ProofInput {
    /// `[d0][d1]...`
    pub fn shape(&self) -> String {
        self.dims.iter().map(|d| format!("[{d}]")).collect()
    }
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
        let p: Vec<String> = cc.public_values.iter().map(|p| format!("publics[{}]", p.idx)).collect();
        if p.len() != 4 {
            bail!("custom commit {}: its root needs 4 publics", cc.name);
        }
        trees.push(Tree {
            suffix: format!("_{}_0", cc.name),
            width: sec(format!("{}0", cc.name)),
            root: format!("[{}]", p.join(", ")),
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

impl KeyData {
    /// `input_challenges` starts the transcript from the global challenge and absorbs `circuitType`;
    /// `roots` = (nRoots, firstRootDim, vc) of `ConstraintRoots` (GL; BN128 folds them in `ConstraintEval`).
    pub fn new(
        si: &StarkInfo,
        input_challenges: bool,
        batch_size: Option<usize>,
        roots: Option<(usize, usize, usize)>,
    ) -> Result<Self> {
        let ss = &si.stark_struct;
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
        let circuit_type = if input_challenges {
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
        if roots.is_some_and(|(_, _, vc)| vc != challenges.len() - 4) {
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
            open.push(o.to_string());
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

        let batch = batch_size.unwrap_or_else(|| (n_queries as usize).div_ceil(8)).clamp(1, n_queries.max(1) as usize);
        let len = |m: &Option<Vec<_>>| m.as_ref().map_or(0, Vec::len) as u64;
        let stages =
            |m: &Option<Vec<proofman_common::PolMap>>| m.iter().flatten().map(|p| p.stage).collect::<Vec<u64>>();
        let scalars = [
            ("nBits", ss.n_bits),
            ("nBitsExt", ss.n_bits_ext),
            ("arity", ss.merkle_tree_arity),
            ("lastLevels", ss.last_level_verification),
            ("nQueries", n_queries),
            ("batchSize", batch as u64),
            ("powBits", pow_bits),
            ("qDeg", si.q_deg),
            ("qPolEv", q_pol_ev as u64),
            ("nEvals", si.ev_map.len() as u64),
            ("nPublics", si.n_publics),
            ("nAirValues", len(&si.airvalues_map)),
            ("nAirgroupValues", len(&si.airgroupvalues_map)),
            ("nStages", n_stages),
            ("nChallenges", challenges.len() as u64),
            ("hashCommits", u64::from(ss.hash_commits)),
            ("circuitType", circuit_type),
            ("nRoots", roots.map_or(0, |r| r.0 as u64)),
            ("firstRootDim", roots.map_or(0, |r| r.1 as u64)),
            ("nSteps", steps.len() as u64),
            ("friFold", u64::from(stir.is_none())),
            ("stir", u64::from(stir.is_some())),
            ("stirM", stir.map_or(0, |s| s.num_iterations() as u64)),
            ("stirFinalBits", stir.map_or(0, |s| *s.log_degrees.last().unwrap())),
            ("nTrees", trees.len() as u64),
            ("nGroups", n_groups),
            ("nPols", leaf.len() as u64),
            ("nColGroups", cgroups.len() as u64),
            ("nSums", sums.len() as u64 / 2),
        ];
        let st = |v: Option<&Vec<u64>>| v.cloned().unwrap_or_default();
        let arrays = [
            ("STEPS", steps),
            ("STIR_L", st(stir.map(|s| &s.log_domain_sizes))),
            ("STIR_K", st(stir.map(|s| &s.folding_factors))),
            ("STIR_T", st(stir.map(|s| &s.num_queries))),
            ("STIR_G", st(stir.map(|s| &s.grinding_bits_queries))),
            ("S0", s0),
            ("CH", ch),
            ("AV", stages(&si.airvalues_map)),
            ("AGV", stages(&si.airgroupvalues_map)),
            ("OPEN_ROOT", open_root),
            ("GROUP_LEN", groups.iter().map(|g| g.refs.len() as u64).collect()),
            ("REF_COL", refs.iter().map(|r| u64::from(r.col)).collect()),
            ("SUMS", sums),
            ("COLS", cols),
        ];
        let mut vars: String = scalars.iter().map(|(n, v)| format!("    var {n} = {v};\n")).collect();
        for (n, mut a) in arrays {
            if a.is_empty() {
                a.push(0); // circom rejects zero-length arrays
            }
            let list: Vec<String> = a.iter().map(u64::to_string).collect();
            vars += &format!("    var {n}[{}] = [{}];\n", a.len(), list.join(", "));
        }
        // The openings themselves, signed, by rank.
        vars += &format!("    var OPEN[{}] = [{}];\n", open.len(), open.join(", "));
        vars += "    var L0 = geo_levels0(arity, lastLevels, STEPS);\n";
        vars += "    var sibW = geo_sibW(arity);\n    var lastSize = geo_lastSize(arity, lastLevels);\n";
        Ok(KeyData { vars, trees })
    }
}
