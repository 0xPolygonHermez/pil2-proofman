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
//! **The layout.** [`unpacked_layout`] is the one of `--no-packing` (plan R1): one `f_i` per
//! polynomial, `k = 1`, its `O` and its bound as `degree` (A.2's cost `max_j(deg_j·k + j)` for
//! `k = 1`), in the order of A.5 within an AIR: by stage, the fixed `f_i` first and `Q`'s last,
//! and within a stage by their index in the pol map. The grouping of A.2 (classes, fusions and
//! `extraMuls`) comes in plan M21/M22.

use std::collections::{BTreeMap, BTreeSet};

use proofman_pilfflonk::global_info::MAX_NBITS;
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

/// The degrees of A.1 for an AIR.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Degrees {
    /// `N = 2^nBits`.
    pub n_bits: u64,
    /// `|O|_max`: the most offsets a column with blinding is opened at.
    pub max_openings: u64,
    /// The bound on `Q`'s coefficients.
    pub q_coefficients: u64,
    /// The extended domain has `2^nBitsExt` points.
    pub n_bits_ext: u64,
}

fn overflow(what: &str) -> SetupError {
    SetupError::Layout(format!("{what} does not fit in 64 bits"))
}

/// The bound on the coefficients of a column of stage `stage` opened at `n_offsets` offsets, on
/// `2^n_bits` rows: `N` for a fixed column, which has no blinding, and `N + |O| + 1` for a
/// committed one, whose blinding `(X^N − 1)·b(X)` has `|O| + 1` coefficients (A.3).
pub fn column_coefficients(n_bits: u64, stage: u64, n_offsets: u64) -> Result<u64, SetupError> {
    if n_bits > MAX_NBITS {
        return Err(SetupError::NBits { n_bits });
    }
    let n = 1u64 << n_bits;
    if stage == 0 {
        Ok(n)
    } else {
        n.checked_add(n_offsets).and_then(|c| c.checked_add(1)).ok_or_else(|| overflow("a column's bound"))
    }
}

/// The bound on the coefficients of `Q` not split (A.1): `qDeg·N + (qDeg+1)·|O|_max + 1`.
pub fn q_coefficients(n_bits: u64, q_deg: u64, max_openings: u64) -> Result<u64, SetupError> {
    if n_bits > MAX_NBITS {
        return Err(SetupError::NBits { n_bits });
    }
    let blinding = q_deg.checked_add(1).and_then(|d| d.checked_mul(max_openings));
    q_deg
        .checked_mul(1u64 << n_bits)
        .zip(blinding)
        .and_then(|(q, b)| q.checked_add(b))
        .and_then(|c| c.checked_add(1))
        .ok_or_else(|| overflow("Q's bound"))
}

/// `nBitsExt` (A.1): the smallest power of two `≥` `Q`'s coefficients and `≥ N + |O|_max + 1`,
/// the coefficients of the column with the most blinding. It is not checked against the
/// 2-adicity here: `validate::check_extended_domain` does.
pub fn n_bits_ext(n_bits: u64, q_coefficients: u64, max_openings: u64) -> Result<u64, SetupError> {
    let column = column_coefficients(n_bits, 1, max_openings)?;
    let points =
        q_coefficients.max(column).checked_next_power_of_two().ok_or_else(|| overflow("the extended domain"))?;
    Ok(u64::from(points.trailing_zeros()))
}

impl Degrees {
    /// The degrees of an AIR of `2^n_bits` rows whose constraint polynomial has degree `q_deg`
    /// (A.1), and whose committed columns (stage ≥ 1) are opened at `max_openings` offsets at
    /// most.
    pub fn new(n_bits: u64, q_deg: u64, max_openings: u64) -> Result<Self, SetupError> {
        let q_coefficients = q_coefficients(n_bits, q_deg, max_openings)?;
        let n_bits_ext = n_bits_ext(n_bits, q_coefficients, max_openings)?;
        Ok(Degrees { n_bits, max_openings, q_coefficients, n_bits_ext })
    }
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn columns_have_n_coefficients_and_their_blinding() {
        assert_eq!(column_coefficients(8, 0, 1).unwrap(), 256);
        assert_eq!(column_coefficients(8, 0, 3).unwrap(), 256, "fixed columns have no blinding");
        assert_eq!(column_coefficients(8, 1, 1).unwrap(), 258);
        assert_eq!(column_coefficients(8, 1, 4).unwrap(), 261);
        assert!(matches!(column_coefficients(29, 1, 1), Err(SetupError::NBits { n_bits: 29 })));
    }

    #[test]
    fn q_has_the_bound_of_a1() {
        // The Fibonacci: N = 256, qDeg = 1, |O|_max = 2.
        assert_eq!(q_coefficients(8, 1, 2).unwrap(), 256 + 2 * 2 + 1);
        // qDeg = 0: Q = c/Z_H of a linear c has |O|_max + 1 coefficients.
        assert_eq!(q_coefficients(3, 0, 1).unwrap(), 2);
        assert_eq!(q_coefficients(10, 3, 4).unwrap(), 3 * 1024 + 4 * 4 + 1);
        assert!(q_coefficients(28, u64::MAX / 4, 1).is_err());
    }

    #[test]
    fn the_extended_domain_holds_q_and_the_most_blinded_column() {
        // Q decides: 261 coefficients, 2^9.
        assert_eq!(n_bits_ext(8, 261, 2).unwrap(), 9);
        // Q exactly a power of two.
        assert_eq!(n_bits_ext(8, 512, 2).unwrap(), 9);
        assert_eq!(n_bits_ext(8, 513, 2).unwrap(), 10);
        // The column decides: with qDeg = 0, Q has 2 coefficients and a column N + 2 (M5).
        assert_eq!(n_bits_ext(3, 2, 1).unwrap(), 4);
        assert_eq!(
            Degrees::new(3, 0, 1).unwrap(),
            Degrees { n_bits: 3, max_openings: 1, q_coefficients: 2, n_bits_ext: 4 }
        );
        // qDeg = 1 and |O|_max = 0 (only fixed columns opened): Q has N + 1, the domain 2N.
        assert_eq!(Degrees::new(3, 1, 0).unwrap().n_bits_ext, 4);
        // At the 2-adicity: N = 2^27 and qDeg = 1 give 2^27 + 3 coefficients, 2^28 points; N =
        // 2^28 gives 2^29, which validate::check_extended_domain refuses.
        assert_eq!(Degrees::new(27, 1, 1).unwrap().n_bits_ext, 28);
        assert_eq!(Degrees::new(28, 1, 1).unwrap().n_bits_ext, 29);
    }
}
