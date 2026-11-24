//! Column layout: the polynomial maps, stage positions, challenges and section sizes.

use std::collections::BTreeMap;

use crate::core::im::ImChoice;
use crate::core::ir::{dim, visit, ColType, Column, ConstraintSystem, Expr, ExprId, Lengths, NamedValue};

/// One committed, fixed or custom column.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PolMap {
    pub name: String,
    pub stage: u32,
    pub dim: u32,
    pub stage_id: u32,
    pub stage_pos: u32,
    pub im_pol: bool,
    pub commit_id: Option<u32>,
    pub lengths: Lengths,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ChallengeEntry {
    pub name: String,
    pub stage: u32,
    pub dim: u32,
    pub stage_id: u32,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Layout {
    pub cm_pols_map: Vec<PolMap>,
    pub const_pols_map: Vec<PolMap>,
    /// One map per custom commit.
    pub custom_pols_map: Vec<Vec<PolMap>>,
    /// `const`, `cm<stage>` (including the Q stage) in first-insertion order, as today.
    pub map_sections_n: Vec<(String, u32)>,
    pub challenges_map: Vec<ChallengeEntry>,
    pub publics_map: Vec<NamedValue>,
    pub proof_values_map: Vec<NamedValue>,
    pub air_values_map: Vec<NamedValue>,
    pub airgroup_values_map: Vec<NamedValue>,
    /// Dimension of the Q pieces (today's cExp dimension).
    pub q_dim: u32,
    vc: usize,
}

impl Layout {
    /// Index of `std_vc` in `challenges_map`.
    pub fn vc_index(&self) -> usize {
        self.vc
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LayoutParams {
    /// im columns go to their expression's stage instead of the last one.
    pub im_pols_stages: bool,
    /// Append `std_vf1`, `std_vf2` after `std_vc`, `std_xi`.
    pub deep_challenges: bool,
    /// Number of Q pieces (stage `n_stages + 1`); 0 adds none.
    pub q_deg: u32,
    /// Prefix of the im columns' names (`<air_name>.ImPol`).
    pub air_name: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum LayoutError {
    Inconsistent { what: String },
}

/// Pilout order of witness/fixed/custom columns and challenges; `std_vc` after its stage's pilout challenges;
/// im columns appended in `ImChoice` order; then the Q pieces; `std_xi`, then `std_vf1`, `std_vf2` last.
pub fn layout(cs: &ConstraintSystem, im: &ImChoice, params: &LayoutParams) -> Result<Layout, LayoutError> {
    let q_dim = quotient_dim(cs, im);
    let cm_pols_map = cm_pols(cs, im, params, q_dim)?;
    let const_pols_map = by_id(&cs.fixed, "fixed", |c| PolMap { stage: 0, dim: 1, stage_id: c.id, ..col(c, None) })?;
    let mut custom_pols_map = Vec::with_capacity(cs.custom.len());
    for (k, cols) in cs.custom.iter().enumerate() {
        let mut m = by_id(cols, "custom", |c| col(c, Some(k as u32)))?;
        number_stages(&mut m);
        custom_pols_map.push(m);
    }
    let map_sections_n = sections(cs, &cm_pols_map, &const_pols_map, &custom_pols_map);
    let (challenges_map, vc) = challenges(cs, params);
    Ok(Layout {
        cm_pols_map,
        const_pols_map,
        custom_pols_map,
        map_sections_n,
        challenges_map,
        publics_map: cs.publics.clone(),
        proof_values_map: cs.proof_values.clone(),
        air_values_map: cs.air_values.clone(),
        airgroup_values_map: cs.airgroup_values.clone(),
        q_dim,
        vc,
    })
}

fn col(c: &Column, commit_id: Option<u32>) -> PolMap {
    PolMap {
        name: c.name.clone(),
        stage: c.stage,
        dim: c.dim,
        stage_id: 0,
        stage_pos: 0,
        im_pol: false,
        commit_id,
        lengths: c.lengths.clone(),
    }
}

/// Witness columns, then the im columns, then the Q pieces, numbered per stage.
fn cm_pols(
    cs: &ConstraintSystem,
    im: &ImChoice,
    params: &LayoutParams,
    q_dim: u32,
) -> Result<Vec<PolMap>, LayoutError> {
    let n = cs.n_stages;
    let mut m = by_id(&cs.witness, "witness", |c| col(c, None))?;
    let witness_stage: Vec<u32> = m.iter().map(|p| p.stage).collect();
    let mut stages = BTreeMap::new();
    let pol = |name: String, stage, dim, im_pol| PolMap {
        name,
        stage,
        dim,
        stage_id: 0,
        stage_pos: 0,
        im_pol,
        commit_id: None,
        lengths: None,
    };
    for (k, &e) in im.exprs.iter().enumerate() {
        let stage = if params.im_pols_stages { stage_of(cs, &witness_stage, e, &mut stages) } else { n };
        m.push(pol(format!("{}.ImPol", params.air_name), stage, dim(cs, &Expr::Ref(e)), true));
        debug_assert_eq!(m.len(), crate::core::im::im_column_id(cs, k) as usize + 1);
    }
    m.extend((0..params.q_deg).map(|i| pol(format!("Q{i}"), n + 1, q_dim, false)));
    number_stages(&mut m);
    Ok(m)
}

/// `const`, the custom sections, then `cm1..cm<n+1>`, as today's insertion order, each summing its columns' dims.
fn sections(cs: &ConstraintSystem, cm: &[PolMap], fixed: &[PolMap], custom: &[Vec<PolMap>]) -> Vec<(String, u32)> {
    let mut sections: Vec<(String, u32)> = vec![("const".into(), 0)];
    for c in &cs.custom_commits {
        let widths = c.stage_widths.iter().enumerate().filter(|(_, w)| **w > 0);
        sections.extend(widths.map(|(j, _)| (format!("{}{j}", c.name), 0)));
    }
    sections.extend((1..=cs.n_stages + 1).map(|s| (format!("cm{s}"), 0)));
    let mut add = |key: String, d: u32| match sections.iter_mut().find(|(k, _)| *k == key) {
        Some(e) => e.1 += d,
        None => sections.push((key, d)),
    };
    fixed.iter().for_each(|p| add("const".into(), p.dim));
    cm.iter().for_each(|p| add(format!("cm{}", p.stage), p.dim));
    for (k, m) in custom.iter().enumerate() {
        let name = cs.custom_commits.get(k).map_or_else(|| format!("custom{k}"), |c| c.name.clone());
        m.iter().for_each(|p| add(format!("{name}{}", p.stage), p.dim));
    }
    sections
}

/// The pilout challenges by (stage, id), `std_vc`, `std_xi`, the DEEP ones; and the index of `std_vc`.
fn challenges(cs: &ConstraintSystem, params: &LayoutParams) -> (Vec<ChallengeEntry>, usize) {
    let (n, ext) = (cs.n_stages, cs.field.ext_dim() as u32);
    let mut ch: Vec<&(u32, u32, String)> = cs.challenges.iter().collect();
    ch.sort_by_key(|c| (c.0, c.1));
    let entry = |name: &str, stage, stage_id| ChallengeEntry { name: name.into(), stage, dim: ext, stage_id };
    let mut m: Vec<ChallengeEntry> = ch.iter().map(|c| entry(&c.2, c.0, c.1)).collect();
    let vc = m.len();
    m.push(entry("std_vc", n + 1, 0));
    // today's cExp builder registers `std_xi` only when there is a constraint
    if !cs.constraints.is_empty() {
        m.push(entry("std_xi", n + 2, 0));
    }
    if params.deep_challenges {
        m.extend([entry("std_vf1", n + 3, 0), entry("std_vf2", n + 3, 1)]);
    }
    (m, vc)
}

/// `cols` placed at their ids, which must be exactly `0..cols.len()`.
fn by_id(cols: &[Column], what: &str, f: impl Fn(&Column) -> PolMap) -> Result<Vec<PolMap>, LayoutError> {
    let mut out: Vec<Option<PolMap>> = vec![None; cols.len()];
    for c in cols {
        match out.get_mut(c.id as usize) {
            Some(slot @ None) => *slot = Some(f(c)),
            _ => {
                return Err(LayoutError::Inconsistent {
                    what: format!("{what} column id {} is repeated or out of range", c.id),
                })
            }
        }
    }
    Ok(out.into_iter().flatten().collect())
}

/// `stage_id` and `stage_pos`: count and dimension of the earlier columns of the same stage.
fn number_stages(m: &mut [PolMap]) {
    let mut seen: BTreeMap<u32, (u32, u32)> = BTreeMap::new();
    for p in m {
        let (n, pos) = seen.entry(p.stage).or_default();
        (p.stage_id, p.stage_pos) = (*n, *pos);
        *n += 1;
        *pos += p.dim;
    }
}

/// Dimension of today's cExp: a fold of two or more roots carries `std_vc`, a lone root its own, none is 1.
fn quotient_dim(cs: &ConstraintSystem, im: &ImChoice) -> u32 {
    let roots = cs.constraints.iter().map(|c| &c.e);
    let ims: Vec<Expr> = im.exprs.iter().map(|&e| Expr::Ref(e)).collect();
    let all: Vec<&Expr> = roots.chain(&ims).collect();
    match all.as_slice() {
        [] => 1,
        [e] => dim(cs, e),
        _ => cs.field.ext_dim() as u32,
    }
}

/// Stage of expression `id`: the largest of its leaves' (witness by column, values by their stage, challenges theirs).
fn stage_of(cs: &ConstraintSystem, witness: &[u32], id: ExprId, memo: &mut BTreeMap<ExprId, u32>) -> u32 {
    if let Some(s) = memo.get(&id) {
        return *s;
    }
    memo.insert(id, 0); // a cycle (rejected at ingestion) reads as 0
    let mut refs = Vec::new();
    let mut s = 0;
    if let Some(e) = cs.exprs.get(id as usize) {
        visit(e, &mut |n| {
            let at = |v: &[NamedValue], i: &u32| v.get(*i as usize).map_or(0, |x| x.stage);
            s = s.max(match n {
                Expr::Col { col: ColType::Witness, id, .. } => witness.get(*id as usize).copied().unwrap_or(0),
                Expr::Challenge { stage, .. } => *stage,
                Expr::Public(_) => 1,
                Expr::ProofValue(i) => at(&cs.proof_values, i),
                Expr::AirValue(i) => at(&cs.air_values, i),
                Expr::AirgroupValue { airgroup: None, id } => at(&cs.airgroup_values, id),
                Expr::Ref(r) => {
                    refs.push(*r);
                    0
                }
                _ => 0,
            });
        });
    }
    for r in refs {
        s = s.max(stage_of(cs, witness, r, memo));
    }
    memo.insert(id, s);
    s
}
