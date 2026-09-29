//! The layout of an AIR: its list of `f_i` (spec §4.2.4, A.2), as `<air>.pilfflonkinfo.json` and
//! the vkey hold it.

use std::collections::BTreeSet;

use num_bigint::BigUint;
use serde::{Deserialize, Serialize};

use crate::error::{invalid, PilfflonkResult};
use crate::field::r;
use crate::global_info::MAX_NBITS;
use crate::names::column_name;
use crate::pilfflonk_info::{EvMapEntry, PolMapEntry, PolType};

/// The `f_i` of an AIR, in order: by ascending stage, the fixed ones (stage 0) first, and `Q`'s
/// last (stage `nStages + 1`). This is the order within the AIR of the global order of A.5, and
/// `f_i` is the entry at position `i`.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(transparent)]
pub struct Layout(pub Vec<LayoutEntry>);

/// One `f_i(X) = Σ_{j<k} p_j(X^k)·X^j` (A.2, rule 4).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LayoutEntry {
    /// 0 for the fixed columns, `1 … nStages` for the committed ones, `nStages + 1` for `Q`.
    pub stage: u64,
    /// `p_0 … p_{k-1}`, in the order of the sum: `pols[j]` is multiplied by `X^j`.
    pub pols: Vec<LayoutPol>,
    /// The number of polynomials packed, `pols.len()`. `k·N` divides `r - 1` (A.2, rule 3).
    pub k: u64,
    /// The opening set `O`: `f_i` is opened at the `k` roots of `ξ·ω^s` for each `s`, in
    /// increasing order. Signed (A.2, rule 5).
    pub offsets: Vec<i64>,
    /// The bound on the number of coefficients of `f_i`: the cost of A.2's rule 3,
    /// `max_j(deg_j·k + j)` with each `deg_j` the bound of `p_j` in coefficients (A.2). The SRS
    /// must hold at least this many powers `[τ^i]₁`.
    pub degree: u64,
}

/// A polynomial of an `f_i`.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LayoutPol {
    /// Its index in `constPolsMap` if the `f_i` is of stage 0, and in `cmPolsMap` otherwise.
    pub id: u64,
    /// Its column name in the JSON view of the proof (`names::column_name`): the vkey carries no
    /// pol maps, and the verifier finds the evaluations by these names.
    pub name: String,
}

/// What a layout is checked against.
pub(crate) struct LayoutCheck<'a> {
    pub n_bits: u64,
    /// `nStages + 1`: the stage of `Q`.
    pub q_stage: u64,
    /// The polynomials `Q` is made of: 1, or its pieces if it is split (A.1).
    pub q_pieces: u64,
    pub ev_map: &'a [EvMapEntry],
    /// `constPolsMap` and `cmPolsMap`, when the file has them (the pilfflonkinfo; the vkey does
    /// not).
    pub pol_maps: Option<(&'a [PolMapEntry], &'a [PolMapEntry])>,
}

/// The number of pieces `Q` is split into (A.1): `⌈qDeg / maxQDegree⌉` if `maxQDegree > 0` and
/// `qDeg > maxQDegree`, and otherwise 1: `Q` whole.
pub fn q_pieces(q_deg: u64, max_q_degree: u64) -> u64 {
    if max_q_degree > 0 && q_deg > max_q_degree {
        q_deg.div_ceil(max_q_degree)
    } else {
        1
    }
}

/// Whether `k·2^n_bits` divides `r - 1`: `k` is a valid factor of an `f_i` of an AIR of `2^n_bits`
/// rows (A.2, rule 3), so the roots of A.2's rule 5 exist.
pub fn is_valid_k(k: u64, n_bits: u64) -> bool {
    if k == 0 || n_bits > MAX_NBITS {
        return false;
    }
    let order = BigUint::from(k) << n_bits;
    ((r() - 1u32) % order) == BigUint::ZERO
}

fn gcd(mut a: u64, mut b: u64) -> u64 {
    while b != 0 {
        (a, b) = (b, a % b);
    }
    a
}

impl Layout {
    /// The fixed `f_i`, the first of the layout: their commitments are in the verkey and the vkey.
    pub fn n_fixed(&self) -> usize {
        self.0.iter().filter(|f| f.stage == 0).count()
    }

    /// `powerW`: the least common multiple of the `k` of every `f_i` (A.2, rule 5).
    pub fn power_w(&self) -> PilfflonkResult<u64> {
        let mut lcm = 1u64;
        for f in &self.0 {
            if f.k == 0 {
                return invalid!("an f with k = 0 has no powerW");
            }
            match (lcm / gcd(lcm, f.k)).checked_mul(f.k) {
                Some(value) => lcm = value,
                None => return invalid!("the least common multiple of the layout's k does not fit in 64 bits"),
            }
        }
        Ok(lcm)
    }

    pub(crate) fn check(&self, c: &LayoutCheck) -> PilfflonkResult<()> {
        let Some(last) = self.0.last() else {
            return invalid!("the layout has no f: Q has one at least");
        };
        if last.stage != c.q_stage {
            return invalid!("the last f of the layout is of stage {}, not of Q's, {}", last.stage, c.q_stage);
        }

        let mut previous_stage = 0;
        let mut packed = BTreeSet::new();
        let mut opened = BTreeSet::new();
        let mut n_q = 0u64;
        for (i, f) in self.0.iter().enumerate() {
            if f.stage < previous_stage || f.stage > c.q_stage {
                return invalid!(
                    "f{i} is of stage {} after one of stage {previous_stage}: the layout goes by ascending stage, \
                     up to Q's, {} (A.5)",
                    f.stage,
                    c.q_stage
                );
            }
            previous_stage = f.stage;
            if f.pols.is_empty() || f.k != f.pols.len() as u64 {
                return invalid!(
                    "f{i} has k = {} and {} polynomials: k is their number, at least 1",
                    f.k,
                    f.pols.len()
                );
            }
            if !is_valid_k(f.k, c.n_bits) {
                return invalid!("f{i} has k = {}, and k·2^{} does not divide r - 1 (A.2, rule 3)", f.k, c.n_bits);
            }
            if f.offsets.is_empty() || f.offsets.windows(2).any(|w| w[0] >= w[1]) {
                return invalid!("f{i} has offsets {:?}: they must be at least one, increasing", f.offsets);
            }
            if f.degree == 0 {
                return invalid!("f{i} has degree 0");
            }
            let is_q = f.stage == c.q_stage;
            if is_q {
                if f.offsets != [0] {
                    return invalid!("f{i} holds Q, which is opened at ξ only, and has offsets {:?}", f.offsets);
                }
                n_q += f.k;
            }
            let pol_type = if f.stage == 0 { PolType::Const } else { PolType::Cm };
            for pol in &f.pols {
                if !packed.insert((pol_type, pol.id)) {
                    return invalid!("{} {} is in two f of the layout", pol_type.as_str(), pol.id);
                }
                if let Some((const_pols, cm_pols)) = c.pol_maps {
                    let map = if f.stage == 0 { const_pols } else { cm_pols };
                    let Some(entry) = usize::try_from(pol.id).ok().and_then(|id| map.get(id)) else {
                        return invalid!("f{i} packs {} {}, which is not in the pol map", pol_type.as_str(), pol.id);
                    };
                    if entry.stage != f.stage {
                        return invalid!(
                            "f{i} is of stage {} and packs {} {}, of stage {}",
                            f.stage,
                            pol_type.as_str(),
                            pol.id,
                            entry.stage
                        );
                    }
                    let name = column_name(&entry.name, &entry.lengths);
                    if pol.name != name {
                        return invalid!(
                            "f{i} names {} {} {:?}, and its name is {name:?}",
                            pol_type.as_str(),
                            pol.id,
                            pol.name
                        );
                    }
                }
                if !is_q {
                    for &offset in &f.offsets {
                        opened.insert((pol_type, pol.id, offset));
                    }
                }
            }
        }
        if n_q != c.q_pieces {
            return invalid!("the layout packs {n_q} polynomials of Q, and Q is made of {}", c.q_pieces);
        }

        // Every evaluation of the proof is opened by SHPLONK, and SHPLONK opens every polynomial of
        // an f at every offset of the f (A.2, A.5): the evMap is the set of (column, offset) of the
        // layout, but Q's, which the verifier computes, or reads from the proof's Q_i(ξ) if it is
        // split (A.1).
        let mut evaluated = BTreeSet::new();
        for e in c.ev_map {
            if !evaluated.insert((e.pol_type, e.id, e.prime)) {
                return invalid!("the evMap has {} {} at offset {} twice", e.pol_type.as_str(), e.id, e.prime);
            }
        }
        if let Some((t, id, offset)) = evaluated.difference(&opened).next() {
            return invalid!("the evMap has {} {id} at offset {offset}, which no f of the layout opens", t.as_str());
        }
        if let Some((t, id, offset)) = opened.difference(&evaluated).next() {
            return invalid!("the layout opens {} {id} at offset {offset}, which the evMap does not have", t.as_str());
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn k_times_n_must_divide_r_minus_1() {
        // r - 1 = 2^28 · 3^2 · 13 · 29 · 983 · 11003 · 237073 · 405928799 · ...
        for k in [1, 2, 3, 4, 6, 8, 9, 12, 13, 16, 18, 26, 29] {
            assert!(is_valid_k(k, 8), "k = {k}");
        }
        for k in [0, 5, 7, 10, 11, 14, 15, 17, 27] {
            assert!(!is_valid_k(k, 8), "k = {k}");
        }
        assert!(is_valid_k(1, 28) && !is_valid_k(2, 27 + 1) && is_valid_k(2, 27) && is_valid_k(3, 28));
        assert!(!is_valid_k(1, 29));
    }

    #[test]
    fn power_w_is_the_lcm_of_the_k() {
        let f = |k| LayoutEntry {
            stage: 1,
            pols: (0..k).map(|id| LayoutPol { id, name: String::new() }).collect(),
            k,
            offsets: vec![0],
            degree: 1,
        };
        assert_eq!(Layout(vec![f(1)]).power_w().unwrap(), 1);
        assert_eq!(Layout(vec![f(4), f(6), f(3), f(1)]).power_w().unwrap(), 12);
        assert_eq!(Layout(vec![f(9), f(13)]).power_w().unwrap(), 117);
    }

    #[test]
    fn q_is_split_only_above_max_q_degree() {
        assert_eq!(q_pieces(3, 0), 1);
        assert_eq!(q_pieces(3, 3), 1);
        assert_eq!(q_pieces(3, 4), 1);
        assert_eq!(q_pieces(4, 3), 2);
        assert_eq!(q_pieces(7, 2), 4);
    }
}
