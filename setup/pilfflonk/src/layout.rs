//! The polynomials the prover commits to, their bounds, and their layout in `f_i` (spec §4.2.4,
//! A.1–A.3, A.5).
//!
//! **The committed polynomials** (A.2's input) are the columns the evMap opens and `Q`. Each has
//! a stage, an opening set `O` and a bound in number of coefficients:
//!
//! - a fixed column: stage 0, `O` the offsets the evMap opens it at, and `N` coefficients: it has
//!   no blinding (A.3);
//! - a committed column, im pols included: stage `1 … nStages`, `O` as a fixed one's, and
//!   `N + |O| + 1` coefficients, of which `|O| + 1` are its blinding's (A.3);
//! - `Q`, not split: stage `nStages + 1`, `O = {0}`, and `qDeg·N + (qDeg+1)·|O|_max + 1`
//!   coefficients (A.1), `|O|_max` the largest `|O|` of the columns with blinding (the committed
//!   ones) *after the fusions* of A.2's rule 1.
//!
//! A column the evMap never opens is not committed (A.2), and [`committed_pols`] says which ones.
//!
//! **The extended domain** (A.1) is the smallest power of two that holds `Q`'s coefficients and
//! `N + |O|_max + 1`, the coefficients of the column with the most blinding, which the prover also
//! extends to it: `nBitsExt`, at most 28 (checked by `validate::check_extended_domain`).
//!
//! The bounds and the extended domain are [`proofman_pilfflonk::degrees`]'s, re-exported here: the
//! prover derives them from the pilfflonkinfo with the same functions, `|O|_max` from the layout.
//!
//! **The layout** ([`Packing`]). By default, the grouping of A.2 ([`crate::grouping::group`], plan
//! M21) with `--extra-muls`: the fusions of rule 1, the classes and the split of each in `f_i`.
//! A fusion moves a column to the offsets of its `f`, so `|O|_max`, and with it `Q`'s bound and the
//! extended domain, are computed from the fused offsets ([`crate::grouping::fuse`]) before `Q` is
//! grouped, and not from the evMap's (spec C.3.2, a defect of the old system). With
//! `--no-packing` (plan R1, for tests), [`unpacked_layout`]: one `f_i` per polynomial, `k = 1`, its
//! own `O` and its bound as `degree` (A.2's cost `max_j(deg_j·k + j)` for `k = 1`), and no fusion.
//! Either is in the order of A.5 within an AIR: by stage, the fixed `f_i` first and `Q`'s last.
//!
//! **The evMap** must be the `(column, offset)` pairs the layout opens (`Layout::check`, A.5): a
//! fused column is opened at the offsets it gains too. [`ev_map_of`] appends those pairs to the
//! evMap of the passes, after all of its entries, so that the index of every one of them, which the
//! `qVerifier`'s `eval` operands are (A.6), does not change. The proof, the transcript (A.4 step 4)
//! and the verifier list the evaluations as the evMap does, the fixed columns' first: an appended
//! pair of a fixed column comes after the other fixed ones, and one of a committed column after the
//! other committed ones, in the same place for all three.

use std::collections::{BTreeMap, BTreeSet};

pub use proofman_pilfflonk::degrees::{column_coefficients, n_bits_ext, q_coefficients, Degrees};
use proofman_pilfflonk::names::column_name;
use proofman_pilfflonk::{EvMapEntry, Layout, LayoutEntry, LayoutPol, PolMapEntry, PolType};

use crate::error::SetupError;
use crate::grouping::{fuse, group, GroupingParams};

/// How the committed polynomials go into `f_i` (see [the module](self)).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Packing {
    /// The grouping of A.2 with this `--extra-muls`: the default.
    Grouped { extra_muls: u64 },
    /// `--no-packing` (plan R1), for tests: an `f` of `k = 1` per polynomial, at its own offsets.
    Unpacked,
}

/// A polynomial the prover commits to, as the grouping of A.2 takes it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CommittedPol {
    /// 0 for a fixed column, `1 … nStages` for a committed one, `nStages + 1` for `Q`.
    pub stage: u64,
    /// Its index in `constPolsMap` if it is fixed, and in `cmPolsMap` otherwise.
    pub id: u64,
    /// `names::column_name` of its entry in the pol map.
    pub name: String,
    /// `O`: the offsets it is opened at, increasing.
    pub offsets: Vec<i64>,
    /// The bound on its number of coefficients.
    pub coefficients: u64,
}

/// What [`committed_pols`] finds.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Committed {
    /// The committed polynomials before any fusion, each with its own offsets and the bound for
    /// them, in the order of [`unpacked_layout`]'s `f_i`: the input of the grouping. `Q`'s bound is
    /// that of the layout's `|O|_max`.
    pub pols: Vec<CommittedPol>,
    /// Their layout, grouped or not.
    pub layout: Layout,
    /// The names of the columns the evMap never opens, which are not committed (A.2).
    pub unopened: Vec<String>,
    /// A.1's, with the layout's `|O|_max`.
    pub degrees: Degrees,
}

/// The committed polynomials of an AIR of `2^n_bits` rows and their layout (see [the
/// module](self)): the columns of `const_pols_map` and `cm_pols_map` that `ev_map` opens, and `Q`,
/// whose pieces (one if it is not split) are the entries of `cm_pols_map` of stage `q_stage`, laid
/// out as `packing` says. `q_deg` is A.1's; `Q` is not split.
///
/// Refuses an evaluation of a column that is not in its map, or of a piece of `Q`, which the
/// verifier computes (A.1), a `Q` of other than one piece, and what the grouping refuses
/// ([`SetupError::Grouping`]).
pub fn committed_pols(
    n_bits: u64,
    q_deg: u64,
    q_stage: u64,
    const_pols_map: &[PolMapEntry],
    cm_pols_map: &[PolMapEntry],
    ev_map: &[EvMapEntry],
    packing: Packing,
) -> Result<Committed, SetupError> {
    let mut offsets: BTreeMap<(PolType, u64), BTreeSet<i64>> = BTreeMap::new();
    for (i, e) in ev_map.iter().enumerate() {
        let map = match e.pol_type {
            PolType::Const => const_pols_map,
            PolType::Cm => cm_pols_map,
        };
        let entry = usize::try_from(e.id).ok().and_then(|id| map.get(id));
        match entry {
            None => {
                return Err(SetupError::PassesOutput(format!(
                    "evMap[{i}] is {} {}, which is not in its pol map",
                    e.pol_type.as_str(),
                    e.id
                )))
            }
            Some(p) if p.stage == q_stage => {
                return Err(SetupError::PassesOutput(format!(
                    "evMap[{i}] is {} ({}), a piece of Q, which the verifier computes (A.1)",
                    e.id, p.name
                )))
            }
            Some(_) => {}
        }
        offsets.entry((e.pol_type, e.id)).or_default().insert(e.prime);
    }

    let mut columns = Vec::new();
    let mut unopened = Vec::new();
    let mut q_pieces = Vec::new();
    let maps = [(PolType::Const, const_pols_map), (PolType::Cm, cm_pols_map)];
    for (pol_type, map) in maps {
        for (id, entry) in map.iter().enumerate() {
            let id = id as u64;
            let name = column_name(&entry.name, &entry.lengths);
            if entry.stage == q_stage {
                q_pieces.push((id, name));
                continue;
            }
            match offsets.get(&(pol_type, id)) {
                Some(o) => columns.push((entry.stage, id, name, o.iter().copied().collect::<Vec<_>>())),
                None => unopened.push(name),
            }
        }
    }

    let [(q_id, q_name)] = <[_; 1]>::try_from(q_pieces).map_err(|pieces| {
        SetupError::PassesOutput(format!("Q is not split, and cmPolsMap has {} pieces of it", pieces.len()))
    })?;

    let mut pols = Vec::with_capacity(columns.len() + 1);
    for (stage, id, name, offsets) in columns {
        let coefficients = column_coefficients(n_bits, stage, offsets.len() as u64)?;
        pols.push(CommittedPol { stage, id, name, offsets, coefficients });
    }
    // The order of the f_i (A.5): by stage, and within a stage by index. cmPolsMap has the im pols
    // after the columns of every stage, so its order is not by stage when there are several. Q,
    // of the last stage, goes last.
    pols.sort_by_key(|p| (p.stage, p.id));

    // |O|_max of the columns with blinding as the layout opens them: after the fusions of A.2's
    // rule 1 if the polynomials are grouped (spec C.3.2). Q takes no part in them.
    let params = |extra_muls| GroupingParams { n_bits, extra_muls, q_stage };
    let max_openings =
        |pols: &[CommittedPol]| pols.iter().filter(|p| p.stage != 0).map(|p| p.offsets.len() as u64).max();
    let max_openings = match packing {
        Packing::Grouped { extra_muls } => max_openings(&fuse(&pols, &params(extra_muls))?),
        Packing::Unpacked => max_openings(&pols),
    };
    let degrees = Degrees::new(n_bits, q_deg, max_openings.unwrap_or(0))?;
    pols.push(CommittedPol {
        stage: q_stage,
        id: q_id,
        name: q_name,
        offsets: vec![0],
        coefficients: degrees.q_coefficients,
    });

    let layout = match packing {
        Packing::Grouped { extra_muls } => group(&pols, &params(extra_muls))?,
        Packing::Unpacked => unpacked_layout(&pols),
    };
    Ok(Committed { pols, layout, unopened, degrees })
}

/// The evMap of a layout (see [the module](self)): `ev_map`, the passes', followed by each
/// `(column, offset)` that `layout` opens and `ev_map` does not have, the pairs a fusion adds. The
/// pairs appended are in the order the passes sort theirs (`pil-info`'s
/// `generate_constraint_polynomial_verifier_code`): by opening point, the fixed columns first, by
/// id; `openingPos` is the offset's position in `opening_points`.
///
/// Refuses an offset that is not in `opening_points`, which has every offset of the AIR's columns
/// (a fusion only moves a column to offsets of its stage's).
pub fn ev_map_of(
    ev_map: &[EvMapEntry],
    layout: &Layout,
    q_stage: u64,
    opening_points: &[i64],
) -> Result<Vec<EvMapEntry>, SetupError> {
    let present: BTreeSet<(PolType, u64, i64)> = ev_map.iter().map(|e| (e.pol_type, e.id, e.prime)).collect();
    let mut added = BTreeSet::new();
    for f in layout.0.iter().filter(|f| f.stage != q_stage) {
        let pol_type = if f.stage == 0 { PolType::Const } else { PolType::Cm };
        for pol in &f.pols {
            for &prime in &f.offsets {
                if present.contains(&(pol_type, pol.id, prime)) {
                    continue;
                }
                let Some(opening_pos) = opening_points.iter().position(|&o| o == prime) else {
                    return Err(SetupError::PassesOutput(format!(
                        "the layout opens {} {} at offset {prime}, which is not one of the opening points {:?}",
                        pol_type.as_str(),
                        pol.id,
                        opening_points
                    )));
                };
                // Const before cm: the reverse of PolType's order.
                added.insert((opening_pos as u64, pol_type != PolType::Const, pol.id, prime, pol_type));
            }
        }
    }
    let mut out = ev_map.to_vec();
    out.extend(added.into_iter().map(|(opening_pos, _, id, prime, pol_type)| EvMapEntry {
        pol_type,
        id,
        prime,
        opening_pos,
    }));
    Ok(out)
}

/// The layout of `--no-packing` (plan R1): an `f_i` of `k = 1` for each polynomial of `pols`, in
/// their order, with its offsets and its bound as `degree`.
pub fn unpacked_layout(pols: &[CommittedPol]) -> Layout {
    Layout(
        pols.iter()
            .map(|p| LayoutEntry {
                stage: p.stage,
                pols: vec![LayoutPol { id: p.id, name: p.name.clone() }],
                k: 1,
                offsets: p.offsets.clone(),
                degree: p.coefficients,
            })
            .collect(),
    )
}

/// The largest `degree` of `layout`: the powers `[τ^i]₁` the SRS must hold (M12).
pub fn max_degree(layout: &Layout) -> u64 {
    layout.0.iter().map(|f| f.degree).max().unwrap_or(0)
}
