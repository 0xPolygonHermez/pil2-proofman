//! The DEEP/FRI first-layer grouping, derived from the evaluation map.
use std::collections::{BTreeMap, HashMap};

use proofman_verifier::{FriEvalGroup, FriEvalRef};

use crate::stark_info::{EvMapEType, StarkInfo};

/// Columns that are opened at exactly the same set of openings.
#[derive(Debug, Clone)]
pub struct ColGroup {
    pub openings: Vec<usize>,
    pub cols: Vec<FriEvalRef>,
}

/// Polynomial identity of an evMap entry; `commitId` only distinguishes custom commits (C++ `deep.hpp` rule).
fn pol_key(ev: &crate::stark_info::EvMap) -> (EvMapEType, u64, u64) {
    let commit = if ev.type_ == EvMapEType::Custom { ev.commit_id } else { 0 };
    (ev.type_, commit, ev.id)
}

/// Per-opening groups of `ev_map`, with `col` the first-appearance index of each polynomial.
/// Panics if `ev_map` is not ordered by opening, references a missing opening, or is empty
/// (the runtime indexes `evals` with a running counter).
pub fn fri_ev_groups(si: &StarkInfo) -> Vec<FriEvalGroup> {
    let mut pol_index: HashMap<(EvMapEType, u64, u64), usize> = HashMap::new();
    for ev in &si.ev_map {
        let n = pol_index.len();
        pol_index.entry(pol_key(ev)).or_insert(n);
    }
    let cm_pols_map = si.cm_pols_map.as_deref().unwrap_or(&[]);
    let mut groups = Vec::new();
    let mut next_eval = 0usize;
    for o in 0..si.opening_points.len() {
        let mut refs = Vec::new();
        for (i, ev) in si.ev_map.iter().enumerate().filter(|(_, ev)| ev.opening_pos == o) {
            let (bucket, offset, dim) = match ev.type_ {
                EvMapEType::Const => (0, ev.id, 1),
                EvMapEType::Cm => {
                    let pol = &cm_pols_map[ev.id as usize];
                    (pol.stage, pol.stage_pos, pol.dim)
                }
                EvMapEType::Custom => (si.n_stages as u64 + 1 + ev.commit_id, ev.id, 1),
            };
            assert_eq!(i, next_eval, "evMap is not ordered by opening point");
            next_eval += 1;
            let col = pol_index[&pol_key(ev)];
            refs.push(FriEvalRef::new(bucket as u16, offset as u16, dim as u8, col as u16));
        }
        // Opening points from hint expressions have no evaluations.
        if !refs.is_empty() {
            groups.push(FriEvalGroup { opening: o as u16, refs });
        }
    }
    assert_eq!(next_eval, si.ev_map.len(), "evMap has entries whose openingPos is outside openingPoints");
    assert!(!groups.is_empty(), "evMap has no evaluations: DEEP needs at least one opening group");
    groups
}

/// Columns grouped by the exact set of openings they appear in, in first-appearance order
/// (same grouping as the GPU's `buildFriColGroups`).
pub fn col_groups(groups: &[FriEvalGroup]) -> Vec<ColGroup> {
    let mut cols: BTreeMap<u16, (FriEvalRef, Vec<usize>)> = BTreeMap::new();
    for g in groups {
        for r in &g.refs {
            cols.entry(r.col).or_insert_with(|| (r.clone(), Vec::new())).1.push(g.opening as usize);
        }
    }
    let mut out: Vec<ColGroup> = Vec::new();
    for (_, (r, openings)) in cols {
        match out.iter_mut().find(|cg| cg.openings == openings) {
            Some(cg) => cg.cols.push(r),
            None => out.push(ColGroup { openings, cols: vec![r] }),
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::stark_info::{EvMap, PolMap};

    fn ev(t: EvMapEType, id: u64, pos: usize, commit_id: u64) -> EvMap {
        EvMap { type_: t, id, prime: pos != 0, opening_pos: pos, commit_id }
    }

    fn si() -> StarkInfo {
        let mut si = StarkInfo { n_stages: 2, opening_points: vec![0, 1, 2], ..Default::default() };
        let pm = |stage, stage_pos, dim| PolMap { stage, stage_pos, dim, ..Default::default() };
        si.cm_pols_map = Some(vec![pm(1, 4, 1), pm(2, 0, 3), pm(3, 0, 3)]);
        // Opening 1 is empty; col indices follow first appearance: const5=0, cm0=1, cm1=2, custom=3.
        si.ev_map = vec![
            ev(EvMapEType::Const, 5, 0, 0),
            ev(EvMapEType::Cm, 0, 0, 0),
            ev(EvMapEType::Cm, 1, 0, 0),
            ev(EvMapEType::Const, 5, 2, 0),
            ev(EvMapEType::Custom, 7, 2, 1),
            ev(EvMapEType::Cm, 0, 2, 0),
        ];
        si
    }

    #[test]
    fn fri_ev_groups_matches_rules() {
        let g = fri_ev_groups(&si());
        assert_eq!(g.len(), 2);
        assert_eq!(g[0].opening, 0);
        assert_eq!(g[1].opening, 2);
        let t = |r: &FriEvalRef| (r.bucket, r.offset, r.dim, r.col);
        assert_eq!(g[0].refs.iter().map(t).collect::<Vec<_>>(), vec![(0, 5, 1, 0), (1, 4, 1, 1), (2, 0, 3, 2)]);
        assert_eq!(g[1].refs.iter().map(t).collect::<Vec<_>>(), vec![(0, 5, 1, 0), (4, 7, 1, 3), (1, 4, 1, 1)]);
    }

    #[test]
    #[should_panic(expected = "openingPos is outside openingPoints")]
    fn fri_ev_groups_rejects_out_of_range_opening() {
        let mut s = si();
        s.ev_map.push(ev(EvMapEType::Cm, 0, 3, 0));
        fri_ev_groups(&s);
    }

    #[test]
    #[should_panic(expected = "evMap has no evaluations")]
    fn fri_ev_groups_rejects_empty() {
        let mut s = si();
        s.ev_map.clear();
        fri_ev_groups(&s);
    }

    #[test]
    fn fri_ev_groups_commit_id_only_keys_custom() {
        let mut s = si();
        // cm/const with a stray commitId must alias their commitId-0 twins; custom commits stay distinct.
        s.ev_map = vec![
            ev(EvMapEType::Const, 5, 0, 0),
            ev(EvMapEType::Cm, 0, 0, 0),
            ev(EvMapEType::Custom, 7, 0, 0),
            ev(EvMapEType::Const, 5, 2, 3),
            ev(EvMapEType::Cm, 0, 2, 2),
            ev(EvMapEType::Custom, 7, 2, 1),
        ];
        let g = fri_ev_groups(&s);
        let cols = |i: usize| g[i].refs.iter().map(|r| r.col).collect::<Vec<_>>();
        assert_eq!(cols(0), vec![0, 1, 2]);
        assert_eq!(cols(1), vec![0, 1, 3]);
    }

    #[test]
    fn col_groups_by_opening_set() {
        let r = |col| FriEvalRef::new(0, col, 1, col);
        let groups = vec![
            FriEvalGroup { opening: 0, refs: vec![r(0)] },
            FriEvalGroup { opening: 1, refs: vec![r(0), r(1), r(2)] },
        ];
        let cg = col_groups(&groups);
        let got: Vec<(Vec<usize>, Vec<u16>)> =
            cg.iter().map(|g| (g.openings.clone(), g.cols.iter().map(|c| c.col).collect())).collect();
        assert_eq!(got, vec![(vec![0, 1], vec![0]), (vec![1], vec![1, 2])]);
    }
}
