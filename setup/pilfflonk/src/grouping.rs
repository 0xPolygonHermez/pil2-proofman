//! The grouping of an AIR's committed polynomials in `f_i`
//! (pilfflonk/docs/protocol.md#grouping-rules): [`group`], a pure function from the polynomials
//! ([`CommittedPol`]) to the [`Layout`].
//!
//! It is the old system's grouping, generalised to signed offsets: the classes of pil-stark's
//! `src/fflonk/helpers/fflonk_shkey.js` and the split in `f_i` of shplonkjs's
//! `src/helpers/setup.js` and `src/utils.js` (pil-stark `5e20f57`, shplonkjs `7824640`: the
//! versions of pilfflonk/docs/README.md#references, whose lines the comments below cite). For
//! offsets in `{0, 1}` the result is the old system's (rule 6), which `tests/grouping.rs` checks
//! against pil-fflonk's example `all`.
//!
//! **Input.** The polynomials, in the order the old system inserts them (`setPolDefs`,
//! `fflonk_shkey.js:34-153`); the setup's is [`crate::layout::committed_pols`]'s, by stage and index.
//! Each has a stage, its opening set `O` and its bound in coefficients *for its own `O`*: `N` for a
//! fixed column, `N + |O| + 1` for one with blinding (stages `1 … nStages`), and `Q`'s for `Q`
//! (pilfflonk/docs/protocol.md#degrees), whose pieces are the polynomials of stage
//! [`GroupingParams::q_stage`].
//!
//! **Rule 1, classes and fusion** ([`fuse`]; `fixFIndex`, `fflonk_shkey.js:244-274`). The
//! polynomials of the stages before `Q`'s are classified by `(stage, O)`. A class of fewer than
//! [`MIN_POLS`] polynomials whose `O` is not the union `U` of its stage's moves to `U`, and so joins
//! the class of `U` if there is one; a class whose `O` is `U` never moves. Class sizes are counted
//! before any move, as `fiMap` is. A polynomial with blinding that moves gains one coefficient per
//! offset it gains (`fflonk_shkey.js:256, 268`), since its blinding has `|O| + 1` coefficients for
//! the `O` of its `f` (pilfflonk/docs/protocol.md#blinding); a fixed one keeps `N`.
//!
//! **The groups and their order** (`fflonk_shkey.js:175, 244-286`; `getFCustom`,
//! `setup.js:163-186`). The old system keeps a list per offset (`polDefs = [polsXi, polsWXi]`),
//! with the polynomials opened at it in input order; a polynomial that moves is appended to the
//! list of each offset it gains, by stage, and within a stage `{0}`'s class before `{1}`'s.
//! Walking the lists in the order of their offsets, the first time a polynomial is met it is
//! inserted in the group of its class, and the first time a class is met its group is numbered.
//! Generalised: the lists go by increasing offset, and the moves by stage, then by `O`
//! (lexicographically, which puts `{0}` before `{1}`), then in input order. `Q`'s pieces, which
//! `fflonk_shkey.js:164` adds after the classes, form the last group (rule 2). Within a group
//! the polynomials go in reverse insertion order: `getFCustom` puts each new one first
//! (`setup.js:183`). That is the order of the composition, `f(X) = Σ_j p_j(X^k)·X^j` (rule 4).
//!
//! **Rule 3, the split in `f_i`** (`applyExtraScalarMuls` and `calculateMultiplePolsLength`,
//! `setup.js:47-85, 212-250`). There are `#groups + extraMuls` `f_i`: group `g` is split in
//! `c_g + 1` consecutive chunks, `Σ c_g = extraMuls`, and `extraMuls > #pols − #groups` is an error
//! (`setup.js:231-232`).
//! - A chunk has a size `k` with `k·N | r − 1` ([`is_valid_k`]): `k | r − 1`, as `getDivisors`
//!   checks (`utils.js:24-32`), and `v₂(k) + nBits ≤ 28`, which shplonkjs does not check and this
//!   grouping does. The sizes that fail are not enumerated.
//! - The splits of a group in `n` chunks are the non-decreasing sequences of `n` sizes that sum to
//!   its length, in lexicographic order (`calculateSplits`, `utils.js:39-53`). A chunk costs
//!   `max_j(deg_j·k + j)` over its polynomials `p_j`, and a split the most of its chunks'
//!   (`calculateDegree`, `setup.js:5-16`). For each `c` the group takes the first split of least
//!   cost (`calculatePolsLength`, `setup.js:19-45`); the `c` with no split are not possible.
//! - The combinations of a `c_g` per group with sum `extraMuls` are enumerated in lexicographic
//!   order of `(c_0, c_1, …)`, the groups in their order (`calculateSumCombinations`,
//!   `utils.js:55-67`). Each gives the vector of its groups' costs, which is sorted in decreasing
//!   order and compared lexicographically; a combination replaces the best only if it is strictly
//!   better (`compareSplits`, `setup.js:87-99`). No combination is an error.
//! - **The size of the search.** Both enumerations are exhaustive, as the old system's, so that the
//!   tie-breaks are its own: nothing is pruned but what cannot complete. Their size grows fast with
//!   `extraMuls` (a group of `n` polynomials has about `n^c / c!` splits in `c + 1` chunks, and the
//!   combinations are the compositions of `extraMuls`). Before it enumerates anything, [`group`]
//!   counts what it would walk (`search_steps`) and refuses more than [`MAX_SEARCH_STEPS`] with
//!   [`GroupingError::SearchTooLarge`], which says to lower `--extra-muls`, rather than prune.
//!
//! **Rule 5, the roots,** are not in the layout: whoever loads it derives them from its `k` and
//! offsets and `N`, and `powerW` is [`Layout::power_w`].
//!
//! **The order of the `f_i`.** The old system numbers them by group, and each group's chunks in
//! order (`setup.js:237-247`). The layout goes by stage (pilfflonk/docs/protocol.md#layout, which
//! `Layout` checks), so the `f_i` are sorted by stage, stably: within a stage they keep the old
//! order. The groups are split in the old order, so the tie-breaks of rule 3 are the old ones even
//! where the two orders differ, which is when a class is first met in the list of an offset other
//! than the first.

use std::collections::{BTreeMap, BTreeSet};

use proofman_pilfflonk::global_info::MAX_NBITS;
use proofman_pilfflonk::layout::is_valid_k;
use proofman_pilfflonk::{Layout, LayoutEntry, LayoutPol};

use crate::layout::CommittedPol;

/// `minPols` of `fixFIndex` (`fflonk_shkey.js:244`): a class of fewer polynomials moves to the
/// union of its stage's offsets (pilfflonk/docs/protocol.md#grouping-rules, rule 1).
pub const MIN_POLS: usize = 3;

/// The most steps the exhaustive search of rule 3 may take (`search_steps`): `2^26`, under a
/// second in a release build. Above it, [`group`] refuses rather than prune (see [the
/// module](self)).
pub const MAX_SEARCH_STEPS: u64 = 1 << 26;

/// What the grouping of an AIR depends on besides its polynomials.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GroupingParams {
    /// The AIR has `N = 2^n_bits` rows; a chunk of `k` polynomials needs `k·N | r − 1`.
    pub n_bits: u64,
    /// `--extra-muls`: how many `f_i` there are besides one per group
    /// (pilfflonk/docs/protocol.md#grouping-rules, rule 3).
    pub extra_muls: u64,
    /// `nStages + 1`: the stage of `Q`. Its polynomials are `Q`'s pieces (one if `Q` is not split),
    /// which rule 1 leaves alone and which form the last group (rule 2).
    pub q_stage: u64,
}

/// Why a set of polynomials cannot be grouped.
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub enum GroupingError {
    #[error(
        "an AIR of 2^{n_bits} rows: BN128's roots of unity allow at most 2^28 (pilfflonk/docs/protocol.md#notation)"
    )]
    NBits { n_bits: u64 },

    #[error("Q is of stage 0, the stage of the fixed columns")]
    QStage,

    #[error("{name} is of stage {stage}, after Q's, {q_stage}")]
    Stage { name: String, stage: u64, q_stage: u64 },

    /// A column the evMap does not open is not committed.
    #[error(
        "{name} is opened at no offset, and a column that is not opened is not committed \
         (pilfflonk/docs/protocol.md#layout)"
    )]
    NotOpened { name: String },

    #[error("{name} has offsets {offsets:?}, which are not increasing")]
    Offsets { name: String, offsets: Vec<i64> },

    #[error("{name} is a piece of Q, which is opened at ξ only, and has offsets {offsets:?}")]
    QOffsets { name: String, offsets: Vec<i64> },

    #[error("{name} has a bound of 0 coefficients")]
    NoCoefficients { name: String },

    #[error("{first} and {second} are both {kind} {id}")]
    DuplicateId { first: String, second: String, kind: &'static str, id: u64 },

    #[error("two polynomials are named {name}")]
    DuplicateName { name: String },

    /// Rule 2: `Q` has an `f` of its own, so there is one at least.
    #[error("no polynomial is of Q's stage, {q_stage}")]
    NoQ { q_stage: u64 },

    /// Rule 3 (`setup.js:231-232`).
    #[error(
        "{extra_muls} extra muls: {n_pols} polynomials in {n_groups} groups make at most {} f_i, so at most {} \
         extra muls (pilfflonk/docs/protocol.md#grouping-errors): lower --extra-muls",
        n_pols,
        n_pols - n_groups
    )]
    TooManyExtraMuls { extra_muls: u64, n_pols: u64, n_groups: u64 },

    /// Rule 3: no combination of splits in chunks of `k·N | r − 1` makes `#groups + extraMuls`
    /// `f_i`: a group of 5, 7, 10, 11, … polynomials cannot be one chunk. There is one with
    /// `extraMuls = #pols − #groups`, every chunk of `k = 1`.
    #[error(
        "no split of the {n_groups} groups in {n_groups} + {extra_muls} f_i has chunks of k with k·2^{n_bits} \
         dividing r - 1 (pilfflonk/docs/protocol.md#grouping-errors): a larger --extra-muls, up to \
         {max_extra_muls}, allows smaller chunks, down to k = 1"
    )]
    NoValidPartition { n_groups: u64, extra_muls: u64, n_bits: u64, max_extra_muls: u64 },

    /// Rule 3: the exhaustive search would take more than [`MAX_SEARCH_STEPS`] steps.
    #[error(
        "{extra_muls} extra muls make the exhaustive search of the grouping's rule 3 take {steps} steps or more, \
         above the {limit} the setup allows (pilfflonk/docs/protocol.md#grouping-errors): lower --extra-muls"
    )]
    SearchTooLarge { extra_muls: u64, steps: u64, limit: u64 },

    #[error("the bound of an f_i does not fit in 64 bits")]
    Overflow,
}

/// Rule 1 (see [the module](self)): `pols` as they end up, each with the `O` of its `f` and its
/// bound for that `O`, in the same order. `Q`'s pieces do not change.
///
/// The setup needs it before [`group`]: `Q`'s bound depends on the most offsets a column with
/// blinding ends up with, not on the most it had (pilfflonk/docs/protocol.md#bounds-after-fusion),
/// and `Q`'s bound does not take part in rule 1.
pub fn fuse(pols: &[CommittedPol], params: &GroupingParams) -> Result<Vec<CommittedPol>, GroupingError> {
    check(pols, params)?;
    fused(pols, params.q_stage)
}

/// The layout of `pols` (see [the module](self)): its `f_i` by stage, each with its polynomials in
/// the order of the composition, its `k`, its offsets and its bound (the cost of rule 3).
///
/// Refuses, in this order, polynomials that break the input's rules (offsets, stages, ids,
/// names), no `Q`, too many extra muls, a search of more than [`MAX_SEARCH_STEPS`] steps and no
/// valid partition: see [`GroupingError`].
pub fn group(pols: &[CommittedPol], params: &GroupingParams) -> Result<Layout, GroupingError> {
    check(pols, params)?;
    if !pols.iter().any(|p| p.stage == params.q_stage) {
        return Err(GroupingError::NoQ { q_stage: params.q_stage });
    }
    let fused = fused(pols, params.q_stage)?;
    let groups = groups(pols, &fused, params.q_stage);

    let n_pols = pols.len() as u64;
    let n_groups = groups.len() as u64;
    if params.extra_muls > n_pols - n_groups {
        return Err(GroupingError::TooManyExtraMuls { extra_muls: params.extra_muls, n_pols, n_groups });
    }

    let longest = groups.iter().map(Vec::len).max().unwrap_or(0);
    let sizes: Vec<usize> = (1..=longest).filter(|&k| is_valid_k(k as u64, params.n_bits)).collect();
    let lengths: Vec<usize> = groups.iter().map(Vec::len).collect();
    let steps = search_steps(&lengths, params.extra_muls, &sizes);
    if steps > MAX_SEARCH_STEPS {
        return Err(GroupingError::SearchTooLarge { extra_muls: params.extra_muls, steps, limit: MAX_SEARCH_STEPS });
    }
    let bounds: Vec<Vec<u64>> = groups.iter().map(|g| g.iter().map(|p| p.coefficients).collect()).collect();
    let splits = choose_splits(&bounds, params.extra_muls, &sizes)?.ok_or(GroupingError::NoValidPartition {
        n_groups,
        extra_muls: params.extra_muls,
        n_bits: params.n_bits,
        max_extra_muls: n_pols - n_groups,
    })?;

    let mut layout = Vec::with_capacity(groups.len() + params.extra_muls as usize);
    for (group, split) in groups.iter().zip(&splits) {
        let mut rest = group.as_slice();
        for &k in &split.sizes {
            let (chunk, tail) = rest.split_at(k);
            rest = tail;
            let bounds: Vec<u64> = chunk.iter().map(|p| p.coefficients).collect();
            // A group's polynomials have the same stage and offsets: those of its class.
            let (stage, offsets) = chunk.first().map(|p| (p.stage, p.offsets.clone())).unwrap_or_default();
            layout.push(LayoutEntry {
                stage,
                pols: chunk.iter().map(|p| LayoutPol { id: p.id, name: p.name.clone() }).collect(),
                k: k as u64,
                offsets,
                degree: chunk_cost(&bounds)?,
            });
        }
    }
    layout.sort_by_key(|f| f.stage);
    Ok(Layout(layout))
}

/// Refuses what the grouping cannot take: see [`GroupingError`].
fn check(pols: &[CommittedPol], params: &GroupingParams) -> Result<(), GroupingError> {
    if params.n_bits > MAX_NBITS {
        return Err(GroupingError::NBits { n_bits: params.n_bits });
    }
    if params.q_stage == 0 {
        return Err(GroupingError::QStage);
    }
    let mut ids: BTreeMap<(bool, u64), &str> = BTreeMap::new();
    let mut names = BTreeSet::new();
    for p in pols {
        let name = || p.name.clone();
        if p.stage > params.q_stage {
            return Err(GroupingError::Stage { name: name(), stage: p.stage, q_stage: params.q_stage });
        }
        if p.offsets.is_empty() {
            return Err(GroupingError::NotOpened { name: name() });
        }
        if p.offsets.windows(2).any(|w| w[0] >= w[1]) {
            return Err(GroupingError::Offsets { name: name(), offsets: p.offsets.clone() });
        }
        if p.stage == params.q_stage && p.offsets != [0] {
            return Err(GroupingError::QOffsets { name: name(), offsets: p.offsets.clone() });
        }
        if p.coefficients == 0 {
            return Err(GroupingError::NoCoefficients { name: name() });
        }
        let fixed = p.stage == 0;
        if let Some(first) = ids.insert((fixed, p.id), &p.name) {
            let kind = if fixed { "const" } else { "cm" };
            return Err(GroupingError::DuplicateId { first: first.to_string(), second: name(), kind, id: p.id });
        }
        if !names.insert(p.name.as_str()) {
            return Err(GroupingError::DuplicateName { name: name() });
        }
    }
    Ok(())
}

/// Rule 1 on `pols`, already checked.
fn fused(pols: &[CommittedPol], q_stage: u64) -> Result<Vec<CommittedPol>, GroupingError> {
    let mut sizes: BTreeMap<(u64, &[i64]), usize> = BTreeMap::new();
    let mut unions: BTreeMap<u64, BTreeSet<i64>> = BTreeMap::new();
    for p in pols.iter().filter(|p| p.stage != q_stage) {
        *sizes.entry((p.stage, p.offsets.as_slice())).or_default() += 1;
        unions.entry(p.stage).or_default().extend(p.offsets.iter().copied());
    }
    let unions: BTreeMap<u64, Vec<i64>> = unions.into_iter().map(|(s, u)| (s, u.into_iter().collect())).collect();

    pols.iter()
        .map(|p| {
            let mut fused = p.clone();
            let Some(union) = unions.get(&p.stage) else {
                return Ok(fused); // Q's stage
            };
            let small = sizes.get(&(p.stage, p.offsets.as_slice())).is_some_and(|&n| n < MIN_POLS);
            if small && p.offsets != *union {
                if p.stage != 0 {
                    let gained = (union.len() - p.offsets.len()) as u64;
                    fused.coefficients = p.coefficients.checked_add(gained).ok_or(GroupingError::Overflow)?;
                }
                fused.offsets = union.clone();
            }
            Ok(fused)
        })
        .collect()
}

/// The groups of rule 3 in the old order, each with its polynomials (of `fused`) in the order of
/// the composition: see [the module](self). `pols` are the polynomials before rule 1, whose offsets
/// say which lists they are in from the start.
fn groups<'a>(pols: &[CommittedPol], fused: &'a [CommittedPol], q_stage: u64) -> Vec<Vec<&'a CommittedPol>> {
    let columns = || pols.iter().zip(fused).enumerate().filter(|(_, (p, _))| p.stage != q_stage);

    let mut lists: BTreeMap<i64, Vec<usize>> = BTreeMap::new();
    for (i, (p, _)) in columns() {
        for &s in &p.offsets {
            lists.entry(s).or_default().push(i);
        }
    }
    let mut moved: Vec<(usize, &CommittedPol, &CommittedPol)> =
        columns().filter(|(_, (p, f))| p.offsets != f.offsets).map(|(i, (p, f))| (i, p, f)).collect();
    // Stable: in input order within a class.
    moved.sort_by(|a, b| (a.1.stage, &a.1.offsets).cmp(&(b.1.stage, &b.1.offsets)));
    for (i, p, f) in moved {
        for &s in f.offsets.iter().filter(|s| p.offsets.binary_search(s).is_err()) {
            lists.entry(s).or_default().push(i);
        }
    }

    let mut seen = BTreeSet::new();
    let mut numbers: BTreeMap<(u64, &[i64]), usize> = BTreeMap::new();
    let mut groups: Vec<Vec<&CommittedPol>> = Vec::new();
    for &i in lists.values().flatten() {
        let Some(f) = fused.get(i).filter(|_| seen.insert(i)) else {
            continue;
        };
        let number = *numbers.entry((f.stage, f.offsets.as_slice())).or_insert_with(|| {
            groups.push(Vec::new());
            groups.len() - 1
        });
        if let Some(group) = groups.get_mut(number) {
            group.push(f);
        }
    }
    groups.push(fused.iter().filter(|f| f.stage == q_stage).collect());
    for group in &mut groups {
        group.reverse();
    }
    groups
}

/// The cost of a chunk of polynomials of bounds `bounds`, `k = bounds.len()`: `max_j(deg_j·k + j)`
/// (rule 3; `setup.js:10-11, 240-241`). It is the bound of the chunk's `f`.
fn chunk_cost(bounds: &[u64]) -> Result<u64, GroupingError> {
    let k = bounds.len() as u64;
    bounds.iter().enumerate().try_fold(0, |cost, (j, &deg)| {
        let c = deg.checked_mul(k).and_then(|c| c.checked_add(j as u64)).ok_or(GroupingError::Overflow)?;
        Ok(cost.max(c))
    })
}

/// A split of a group in chunks, of these sizes, and its cost: its most costly chunk's
/// (`calculateDegree`, `setup.js:5-16`).
#[derive(Clone, Debug, PartialEq, Eq)]
struct Split {
    sizes: Vec<usize>,
    cost: u64,
}

fn split_cost(bounds: &[u64], sizes: &[usize]) -> Result<u64, GroupingError> {
    let mut rest = bounds;
    let mut cost = 0;
    for &k in sizes {
        let (chunk, tail) = rest.split_at(k);
        rest = tail;
        cost = cost.max(chunk_cost(chunk)?);
    }
    Ok(cost)
}

/// The first split of least cost of a group of bounds `bounds` in `parts` chunks of sizes
/// `sizes` (increasing), among the non-decreasing sequences of sizes in lexicographic order; `None`
/// if there is none (`calculatePolsLength`, `setup.js:19-45`, over `calculateSplits`,
/// `utils.js:39-53`).
fn best_split(bounds: &[u64], parts: usize, sizes: &[usize]) -> Result<Option<Split>, GroupingError> {
    fn walk(
        bounds: &[u64],
        parts: usize,
        sizes: &[usize],
        from: usize,
        split: &mut Vec<usize>,
        sum: usize,
        best: &mut Option<Split>,
    ) -> Result<(), GroupingError> {
        if split.len() == parts {
            if sum == bounds.len() {
                let cost = split_cost(bounds, split)?;
                if best.as_ref().is_none_or(|b| cost < b.cost) {
                    *best = Some(Split { sizes: split.clone(), cost });
                }
            }
            return Ok(());
        }
        let left = parts - split.len();
        for (i, &k) in sizes.iter().enumerate().skip(from) {
            // This chunk and the `left - 1` after it, none smaller: past the length with `k`, and
            // so with any larger size. Pruning only: the enumeration and its order are the same.
            if k.saturating_mul(left).saturating_add(sum) > bounds.len() {
                break;
            }
            split.push(k);
            walk(bounds, parts, sizes, i, split, sum + k, best)?;
            split.pop();
        }
        Ok(())
    }

    let mut best = None;
    walk(bounds, parts, sizes, 0, &mut Vec::with_capacity(parts), 0, &mut best)?;
    Ok(best)
}

/// Rule 3 between groups: the split of each group in the first combination of least cost (see
/// [the module](self)), or `None` if there is no combination (`calculateMultiplePolsLength`,
/// `setup.js:47-85`, and `compareSplits`, `setup.js:87-99`).
fn choose_splits(bounds: &[Vec<u64>], extra_muls: u64, sizes: &[usize]) -> Result<Option<Vec<Split>>, GroupingError> {
    // For each group, its best split for each c it can have, by increasing c.
    let mut options: Vec<Vec<(u64, Split)>> = Vec::with_capacity(bounds.len());
    for group in bounds {
        let mut by_c = Vec::new();
        for c in 0..=extra_muls.min(group.len().saturating_sub(1) as u64) {
            if let Some(split) = best_split(group, c as usize + 1, sizes)? {
                by_c.push((c, split));
            }
        }
        if by_c.is_empty() {
            return Ok(None);
        }
        options.push(by_c);
    }

    let mut combinations = Combinations { options: &options, chosen: Vec::with_capacity(options.len()), best: None };
    combinations.walk(0, extra_muls);
    Ok(combinations.best.map(|(_, chosen)| chosen.into_iter().cloned().collect()))
}

/// The walk over the combinations of `choose_splits`, in lexicographic order of the `c` of each
/// group (`calculateSumCombinations`, `utils.js:55-67`).
struct Combinations<'a> {
    options: &'a [Vec<(u64, Split)>],
    chosen: Vec<&'a Split>,
    /// The best combination so far, and its costs in decreasing order.
    best: Option<(Vec<u64>, Vec<&'a Split>)>,
}

impl<'a> Combinations<'a> {
    fn walk(&mut self, group: usize, left: u64) {
        let options = self.options;
        let Some(by_c) = options.get(group) else {
            if left == 0 {
                self.consider();
            }
            return;
        };
        for (c, split) in by_c {
            if *c > left {
                break;
            }
            self.chosen.push(split);
            self.walk(group + 1, left - c);
            self.chosen.pop();
        }
    }

    /// Keeps the combination chosen if it is strictly better than the best (`setup.js:74, 87-99`).
    fn consider(&mut self) {
        let mut costs: Vec<u64> = self.chosen.iter().map(|s| s.cost).collect();
        costs.sort_unstable_by(|a, b| b.cmp(a));
        if self.best.as_ref().is_none_or(|(best, _)| costs < *best) {
            self.best = Some((costs, self.chosen.clone()));
        }
    }
}

/// What the counting of `search_steps` charges for each count it holds, at least: it allocates
/// them and adds one per chunk size to each. So the counting never holds more than
/// `MAX_SEARCH_STEPS / 16` counts (32 MiB).
const STEPS_PER_COUNT: u64 = 16;

/// The number of splits of a group of `len` polynomials in `p` chunks of the sizes `sizes`
/// (increasing), for each `p` from 0 to `max_parts`: the non-decreasing sequences of `p` sizes that
/// add up to `len`, which `best_split` evaluates. Saturating.
fn split_counts(len: usize, max_parts: usize, sizes: &[usize]) -> Vec<u64> {
    // ways[p][n]: the multisets of p of the sizes added so far that add up to n. Adding the size
    // k, any number of times: ways'[p][n] = ways[p][n] + ways'[p − 1][n − k], by increasing p.
    let mut ways = vec![vec![0u64; len + 1]; max_parts + 1];
    ways[0][0] = 1;
    for &k in sizes.iter().take_while(|&&k| k <= len) {
        for p in 1..=max_parts {
            let (done, rest) = ways.split_at_mut(p);
            if let (Some(previous), Some(current)) = (done.last(), rest.first_mut()) {
                for (count, &fewer) in current[k..].iter_mut().zip(previous.iter()) {
                    *count = count.saturating_add(fewer);
                }
            }
        }
    }
    ways.iter().map(|w| w[len]).collect()
}

/// The steps of rule 3's search (see [the module](self)) for groups of `lengths` polynomials,
/// `extra_muls` and the chunk sizes `sizes`, saturating: [`group`] refuses more than
/// [`MAX_SEARCH_STEPS`] before it enumerates anything. They are, for each group of `n`
/// polynomials, which can have up to `c_max = min(extraMuls, n − 1)` extra chunks:
/// - the counting itself: `STEPS_PER_COUNT` (or `|sizes|` if more) for each of the
///   `(c_max + 2)·(n + 1)` counts of the group's splits, and 1 for each prefix sum times each `c`
///   for the combinations'; once they are too many, the rest is not counted;
/// - `n` for each split `best_split` evaluates (`split_cost` walks the group);
///
/// and then 1 for each node the walk over the combinations visits (a prefix of the groups' `c`
/// that adds up to `extraMuls` at most: `Combinations::walk` breaks at the first `c` above what is
/// left), and the number of groups for each complete combination, which `consider` sorts and
/// compares. A count of the work, not an exact tally of the loops: `best_split` also visits the
/// prefixes of splits that cannot complete, a few per split.
fn search_steps(lengths: &[usize], extra_muls: u64, sizes: &[usize]) -> u64 {
    let limit = usize::try_from(extra_muls).unwrap_or(usize::MAX);
    let mut steps = 0u64;
    // prefixes[s]: the prefixes of the groups walked so far whose c add up to s; none above
    // extraMuls, which `group` checked is at most the number of polynomials.
    let mut prefixes = vec![1u64];
    let mut visited = 1u64;
    for &n in lengths {
        let c_max = limit.min(n.saturating_sub(1));
        let counts = (c_max as u64 + 2).saturating_mul(n as u64 + 1);
        steps = steps.saturating_add(counts.saturating_mul(STEPS_PER_COUNT.max(sizes.len() as u64)));
        if steps > MAX_SEARCH_STEPS {
            return steps;
        }
        let splits = split_counts(n, c_max + 1, sizes);
        steps = steps.saturating_add((c_max as u64 + 1).saturating_mul(prefixes.len() as u64));
        if steps > MAX_SEARCH_STEPS {
            return steps;
        }
        let mut next = vec![0u64; (prefixes.len() + c_max).min(limit.saturating_add(1))];
        for (c, &count) in splits.iter().skip(1).enumerate() {
            steps = steps.saturating_add(count.saturating_mul(n as u64));
            if count == 0 {
                continue;
            }
            for (sum, &ways) in prefixes.iter().enumerate() {
                if let Some(slot) = next.get_mut(sum + c) {
                    *slot = slot.saturating_add(ways);
                }
            }
        }
        prefixes = next;
        visited = prefixes.iter().fold(visited, |v, &p| v.saturating_add(p));
    }
    let complete = prefixes.get(limit).copied().unwrap_or(0);
    steps.saturating_add(visited).saturating_add(complete.saturating_mul(lengths.len() as u64))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The sizes of a chunk for `N = 2^8`: the divisors of `r − 1` up to 9.
    const SIZES: [usize; 7] = [1, 2, 3, 4, 6, 8, 9];

    fn best(bounds: &[u64], parts: usize) -> Option<Split> {
        best_split(bounds, parts, &SIZES).unwrap()
    }

    #[test]
    fn a_chunk_costs_its_most_costly_polynomial_times_k_plus_its_index() {
        assert_eq!(chunk_cost(&[256; 6]).unwrap(), 256 * 6 + 5);
        assert_eq!(chunk_cost(&[10, 1, 1]).unwrap(), 30, "p_0 = 10: 10·3 + 0");
        assert_eq!(chunk_cost(&[1, 1, 10]).unwrap(), 32, "p_2 = 10: 10·3 + 2");
        assert_eq!(split_cost(&[10, 1, 1, 1], &[1, 3]).unwrap(), 10);
        assert_eq!(split_cost(&[10, 1, 1, 1], &[2, 2]).unwrap(), 20);
        assert_eq!(chunk_cost(&[u64::MAX, 1]), Err(GroupingError::Overflow));
    }

    #[test]
    fn a_group_takes_the_first_split_of_least_cost() {
        // [1,3] costs max(4, 2·3 + 2) = 8 and [2,2] max(4·2, 2·2 + 1) = 8: the first enumerated.
        assert_eq!(best(&[4, 2, 2, 2], 2), Some(Split { sizes: vec![1, 3], cost: 8 }));
        // [1,3] costs 5 and [2,2] 3.
        assert_eq!(best(&[1, 1, 1, 1], 2), Some(Split { sizes: vec![2, 2], cost: 3 }));
        // Nine of 258 in three: [1,2,6], [1,4,4], [2,3,4], [3,3,3], of which [3,3,3] costs least.
        assert_eq!(best(&[258; 9], 3), Some(Split { sizes: vec![3, 3, 3], cost: 776 }));
    }

    #[test]
    fn only_non_decreasing_splits_are_enumerated() {
        // [3,1] would cost max(5, 100) = 100, but it decreases: of [1,3] (302) and [2,2] (201),
        // [2,2].
        assert_eq!(best(&[1, 1, 1, 100], 2), Some(Split { sizes: vec![2, 2], cost: 201 }));
    }

    #[test]
    fn only_the_given_sizes_are_enumerated() {
        // 5 does not divide r - 1, nor 7: five polynomials are one chunk in no way, and in two
        // [1,4] or [2,3].
        assert_eq!(best(&[1; 5], 1), None);
        assert_eq!(best(&[1; 5], 2), Some(Split { sizes: vec![2, 3], cost: 5 }));
        assert_eq!(best(&[1; 7], 1), None);
        // With N = 2^27, 4·N does not divide r - 1: four in one chunk is not possible.
        assert_eq!(best_split(&[1; 4], 1, &[1, 2, 3]).unwrap(), None);
        // More chunks than polynomials.
        assert_eq!(best(&[1; 2], 3), None);
    }

    /// The non-decreasing sequences of `parts` sizes that add up to `len`, enumerated.
    fn brute_counts(len: usize, parts: usize, sizes: &[usize]) -> u64 {
        fn walk(left: usize, parts: usize, sizes: &[usize]) -> u64 {
            match parts {
                0 => u64::from(left == 0),
                _ => (0..sizes.len())
                    .map(|i| if sizes[i] <= left { walk(left - sizes[i], parts - 1, &sizes[i..]) } else { 0 })
                    .sum(),
            }
        }
        walk(len, parts, sizes)
    }

    #[test]
    fn the_splits_are_counted_as_best_split_enumerates_them() {
        // Nine in three: [1,2,6], [1,4,4], [2,3,4], [3,3,3]; in two: [1,8], [3,6].
        assert_eq!(split_counts(9, 3, &SIZES), [0, 1, 2, 4]);
        assert_eq!(split_counts(4, 4, &[1, 2, 3, 4]), [0, 1, 2, 1, 1]);
        assert_eq!(split_counts(0, 1, &SIZES), [1, 0]);
        for len in 0..30 {
            let counts = split_counts(len, 8, &SIZES);
            for (parts, &count) in counts.iter().enumerate() {
                assert_eq!(count, brute_counts(len, parts, &SIZES), "{len} in {parts}");
            }
        }
        // Sizes above the length are not counted.
        assert_eq!(split_counts(3, 2, &[1, 2, 4, 8]), [0, 0, 1]);
    }

    #[test]
    fn the_search_steps_count_the_counting_the_splits_and_the_combinations() {
        // A group of 2 (c ≤ 1) and one of 1 (c = 0), one extra mul, sizes 1 and 2:
        // - the counting: 3·3 and 2·2 counts of splits, 16 steps each, 144 + 64, and of the
        //   combinations 2 c by 1 prefix sum, then 1 c by 2: 2 + 2;
        // - the splits: [2] and [1,1] of the first, 2 each, and [1] of the second: 4 + 1;
        // - the walk: the root, (0), (1), (0,0), (1,0): 5; and the one complete, (1,0), 2.
        assert_eq!(search_steps(&[2, 1], 1, &[1, 2]), 144 + 64 + 2 + 2 + 4 + 1 + 5 + 2);
        // No extra mul: one split per group, one combination.
        assert_eq!(search_steps(&[2, 1], 0, &[1, 2]), 2 * 3 * 16 + 2 * 2 * 16 + 1 + 1 + 2 + 1 + 3 + 2);
        // Forty groups of eight and a hundred extra muls: the combinations alone are too many, and
        // the count saturates rather than overflow.
        assert!(search_steps(&[8; 40], 100, &SIZES) > MAX_SEARCH_STEPS);
        assert_eq!(search_steps(&[8; 400], 2000, &SIZES), u64::MAX);
    }
}
