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
//!   ones).
//!
//! A column the evMap never opens is not committed (A.2), and [`committed_pols`] says which ones.
//!
//! **The extended domain** (A.1) is the smallest power of two that holds `Q`'s coefficients and
//! `N + |O|_max + 1`, the coefficients of the column with the most blinding, which the prover also
//! extends to it: `nBitsExt`, at most 28 (checked by `validate::check_extended_domain`).
//!
//! The bounds and the extended domain are [`proofman_pilfflonk::degrees`]'s, re-exported here: the
//! prover derives them from the pilfflonkinfo with the same functions.
//!
//! **The layout.** [`unpacked_layout`] is the one of `--no-packing` (plan R1): one `f_i` per
//! polynomial, `k = 1`, its `O` and its bound as `degree` (A.2's cost `max_j(deg_j·k + j)` for
//! `k = 1`), in the order of A.5 within an AIR: by stage, the fixed `f_i` first and `Q`'s last,
//! and within a stage by their index in the pol map. The grouping of A.2 (classes, fusions and
//! `extraMuls`) comes in plan M21/M22.

use std::collections::{BTreeMap, BTreeSet};

pub use proofman_pilfflonk::degrees::{column_coefficients, n_bits_ext, q_coefficients, Degrees};
use proofman_pilfflonk::names::column_name;
use proofman_pilfflonk::{EvMapEntry, Layout, LayoutEntry, LayoutPol, PolMapEntry, PolType};

use crate::error::SetupError;

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
    /// The committed polynomials, in the order of [`unpacked_layout`]'s `f_i`.
    pub pols: Vec<CommittedPol>,
    /// The names of the columns the evMap never opens, which are not committed (A.2).
    pub unopened: Vec<String>,
    pub degrees: Degrees,
}

/// The committed polynomials of an AIR (see [the module](self)): the columns of `const_pols_map`
/// and `cm_pols_map` that `ev_map` opens, and `Q`, whose pieces (one if it is not split) are the
/// entries of `cm_pols_map` of stage `q_stage`. `q_deg` is A.1's; `Q` is not split.
///
/// Refuses an evaluation of a column that is not in its map, or of a piece of `Q`, which the
/// verifier computes (A.1), and a `Q` of other than one piece.
pub fn committed_pols(
    n_bits: u64,
    q_deg: u64,
    q_stage: u64,
    const_pols_map: &[PolMapEntry],
    cm_pols_map: &[PolMapEntry],
    ev_map: &[EvMapEntry],
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

    let max_openings = columns.iter().filter(|c| c.0 != 0).map(|c| c.3.len() as u64).max().unwrap_or(0);
    let degrees = Degrees::new(n_bits, q_deg, max_openings)?;
    let [(q_id, q_name)] = <[_; 1]>::try_from(q_pieces).map_err(|pieces| {
        SetupError::PassesOutput(format!("Q is not split, and cmPolsMap has {} pieces of it", pieces.len()))
    })?;

    let mut pols = Vec::with_capacity(columns.len() + 1);
    for (stage, id, name, offsets) in columns {
        let coefficients = column_coefficients(n_bits, stage, offsets.len() as u64)?;
        pols.push(CommittedPol { stage, id, name, offsets, coefficients });
    }
    pols.push(CommittedPol {
        stage: q_stage,
        id: q_id,
        name: q_name,
        offsets: vec![0],
        coefficients: degrees.q_coefficients,
    });
    // The order of the f_i (A.5): by stage, and within a stage by index. cmPolsMap has the im pols
    // after the columns of every stage, so its order is not by stage when there are several.
    pols.sort_by_key(|p| (p.stage, p.id));
    Ok(Committed { pols, unopened, degrees })
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
