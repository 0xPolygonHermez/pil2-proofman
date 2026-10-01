//! The degrees of an AIR (pilfflonk/docs/protocol.md#degrees): the bound on the coefficients of
//! each committed polynomial and of `Q`, and the extended domain the prover evaluates `Q` on. They
//! are derived, not stored: the setup derives them before the layout exists
//! (`pilfflonk_setup::layout`), and the prover from the pilfflonkinfo ([`PilfflonkInfo::degrees`]),
//! both with these functions. The C++ prover derives them again for itself (`AirDegrees` in
//! `pil2-stark/src/pilfflonk/pilfflonk_proving_key.hpp`), and the orchestrator checks that it
//! agrees.
//!
//! - A fixed column has `N` coefficients: it has no blinding.
//! - A committed column opened at `|O|` offsets has `N + |O| + 1`, of which `|O| + 1` are its
//!   blinding's, `(X^N − 1)·b(X)` (pilfflonk/docs/protocol.md#blinding).
//! - `Q` has `qDeg·N + (qDeg+1)·|O|_max + 1`, `|O|_max` the most offsets of a committed column
//!   (with packing, of an `f` of a committed stage).
//! - Split (`maxQDegree = M > 0` and `qDeg > M`), `Q` is committed as `m = ⌈qDeg/M⌉` pieces of `M·N`
//!   coefficients, `Q(X) = Σ_i X^{i·M·N}·Q_i(X)`, and each boundary between two pieces adds two
//!   random coefficients that cancel (pilfflonk/docs/protocol.md#q-pieces): each piece but the last
//!   has `M·N + 2` coefficients, and the last one the rest of `Q`'s ([`QSplit`]).
//! - The extended domain is the smallest power of two `≥` `Q`'s coefficients and `≥ N + |O|_max +
//!   1`, the coefficients of the column with the most blinding, which the prover also extends to
//!   it: `2^nBitsExt` points, `nBitsExt ≤ 28` for BN254's roots of unity (checked by the callers).
//!   It is `Q`'s, split or not: the prover computes `Q` whole on it before it splits it.

use crate::error::{invalid, PilfflonkResult};
use crate::global_info::MAX_NBITS;
use crate::layout::q_pieces;
use crate::pilfflonk_info::PilfflonkInfo;

/// The degrees of an AIR (pilfflonk/docs/protocol.md#degrees).
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

fn check_n_bits(n_bits: u64) -> PilfflonkResult<()> {
    if n_bits > MAX_NBITS {
        return invalid!("an AIR of 2^{n_bits} rows: BN254's roots of unity allow at most 2^{MAX_NBITS}");
    }
    Ok(())
}

fn overflow<T>(what: &str) -> PilfflonkResult<T> {
    invalid!("{what} does not fit in 64 bits")
}

/// The bound on the coefficients of a column of stage `stage` opened at `n_offsets` offsets, on
/// `2^n_bits` rows: `N` for a fixed column (stage 0), which has no blinding, and `N + |O| + 1` for a
/// committed one, whose blinding `(X^N − 1)·b(X)` has `|O| + 1` coefficients
/// (pilfflonk/docs/protocol.md#blinding).
pub fn column_coefficients(n_bits: u64, stage: u64, n_offsets: u64) -> PilfflonkResult<u64> {
    check_n_bits(n_bits)?;
    let n = 1u64 << n_bits;
    if stage == 0 {
        return Ok(n);
    }
    match n.checked_add(n_offsets).and_then(|c| c.checked_add(1)) {
        Some(c) => Ok(c),
        None => overflow("a column's bound"),
    }
}

/// The bound on the coefficients of `Q` (pilfflonk/docs/protocol.md#degrees):
/// `qDeg·N + (qDeg+1)·|O|_max + 1`.
pub fn q_coefficients(n_bits: u64, q_deg: u64, max_openings: u64) -> PilfflonkResult<u64> {
    check_n_bits(n_bits)?;
    let blinding = q_deg.checked_add(1).and_then(|d| d.checked_mul(max_openings));
    let bound = q_deg
        .checked_mul(1u64 << n_bits)
        .zip(blinding)
        .and_then(|(q, b)| q.checked_add(b))
        .and_then(|c| c.checked_add(1));
    match bound {
        Some(c) => Ok(c),
        None => overflow("Q's bound"),
    }
}

/// `nBitsExt` (pilfflonk/docs/protocol.md#degrees): the smallest power of two `≥` `Q`'s
/// coefficients and `≥ N + |O|_max + 1`, the coefficients of the column with the most blinding. It
/// is not checked against the 2-adicity here: the callers do.
pub fn n_bits_ext(n_bits: u64, q_coefficients: u64, max_openings: u64) -> PilfflonkResult<u64> {
    let column = column_coefficients(n_bits, 1, max_openings)?;
    match q_coefficients.max(column).checked_next_power_of_two() {
        Some(points) => Ok(u64::from(points.trailing_zeros())),
        None => overflow("the extended domain"),
    }
}

impl Degrees {
    /// The degrees of an AIR of `2^n_bits` rows whose constraint polynomial has degree `q_deg`
    /// (pilfflonk/docs/protocol.md#degree-search), and whose committed columns (stage ≥ 1) are
    /// opened at `max_openings` offsets at most.
    pub fn new(n_bits: u64, q_deg: u64, max_openings: u64) -> PilfflonkResult<Self> {
        let q_coefficients = q_coefficients(n_bits, q_deg, max_openings)?;
        let n_bits_ext = n_bits_ext(n_bits, q_coefficients, max_openings)?;
        Ok(Degrees { n_bits, max_openings, q_coefficients, n_bits_ext })
    }
}

/// The pieces `Q_0 … Q_{m−1}` `Q` is committed as (pilfflonk/docs/protocol.md#q-pieces),
/// `m = q_pieces(qDeg, maxQDegree)`: `Q(X) = Σ_i X^{i·stride}·Q_i(X)`.
///
/// Not split (`m = 1`), the one piece is `Q`, unblinded. Split, with `S = stride = M·N` (`M =
/// maxQDegree`), piece `i` holds the coefficients `i·S … (i+1)·S − 1` of `Q`, and the last one those
/// from `(m−1)·S` to its bound; and each boundary between pieces `i` and `i + 1` has two random
/// coefficients `b_0, b_1` that cancel, as PLONK's (`pil-fflonk/src/pilfflonk_prover.cpp:697-720`):
/// `b_0·X^S + b_1·X^{S+1}` added to piece `i` and `b_0 + b_1·X` subtracted from piece `i + 1`. So
/// each piece but the last has `S + 2` coefficients, and the last `qCoefficients − (m−1)·S`, at least
/// `N + 1` (its first `qDeg − (m−1)·M ≥ 1` of `N`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct QSplit {
    /// `S = M·N`, the power of `X` piece 1 is multiplied by; 0 when `Q` is not split.
    pub stride: u64,
    /// The bound on the coefficients of each piece, `Q_0`'s first: of the pieces' `f`
    /// (pilfflonk/docs/protocol.md#layout).
    pub coefficients: Vec<u64>,
}

impl QSplit {
    /// The pieces of `Q` of degree `q_deg` on `2^n_bits` rows whose committed columns are
    /// opened at `max_openings` offsets at most, split by `max_q_degree` (0 does not split it).
    pub fn new(n_bits: u64, q_deg: u64, max_openings: u64, max_q_degree: u64) -> PilfflonkResult<Self> {
        let q_coefficients = q_coefficients(n_bits, q_deg, max_openings)?;
        let m = q_pieces(q_deg, max_q_degree);
        if m == 1 {
            return Ok(QSplit { stride: 0, coefficients: vec![q_coefficients] });
        }
        // max_q_degree < q_deg, so (m − 1)·S < qDeg·N < q_coefficients, which fits.
        let stride = max_q_degree << n_bits;
        let last = q_coefficients - (m - 1) * stride;
        let mut coefficients = vec![stride + 2; m as usize - 1];
        coefficients.push(last);
        Ok(QSplit { stride, coefficients })
    }

    /// `m`: 1 if `Q` is not split.
    pub fn n_pieces(&self) -> usize {
        self.coefficients.len()
    }
}

impl PilfflonkInfo {
    /// The degrees of the AIR, from its layout: `|O|_max` is the most offsets of an `f` of a
    /// committed stage (`1 … nStages`). What the setup derived them from, and so what its layout's
    /// degrees are made of.
    pub fn degrees(&self) -> PilfflonkResult<Degrees> {
        Degrees::new(self.n_bits, self.q_deg, self.max_openings())
    }

    /// The pieces `Q` of the AIR is committed as ([`QSplit`]), with `|O|_max` as
    /// [`PilfflonkInfo::degrees`].
    pub fn q_split(&self) -> PilfflonkResult<QSplit> {
        QSplit::new(self.n_bits, self.q_deg, self.max_openings(), self.max_q_degree)
    }

    fn max_openings(&self) -> u64 {
        let committed = self.layout.0.iter().filter(|f| f.stage >= 1 && f.stage <= self.n_stages);
        committed.map(|f| f.offsets.len() as u64).max().unwrap_or(0)
    }
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
        let err = column_coefficients(29, 1, 1).unwrap_err();
        assert!(err.to_string().contains("2^29 rows"), "{err}");
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
        // The column decides: with qDeg = 0, Q has 2 coefficients and a column N + 2.
        assert_eq!(n_bits_ext(3, 2, 1).unwrap(), 4);
        assert_eq!(
            Degrees::new(3, 0, 1).unwrap(),
            Degrees { n_bits: 3, max_openings: 1, q_coefficients: 2, n_bits_ext: 4 }
        );
        // qDeg = 1 and |O|_max = 0 (only fixed columns opened): Q has N + 1, the domain 2N.
        assert_eq!(Degrees::new(3, 1, 0).unwrap().n_bits_ext, 4);
        // At the 2-adicity: N = 2^27 and qDeg = 1 give 2^27 + 3 coefficients, 2^28 points; N =
        // 2^28 gives 2^29, which the callers refuse.
        assert_eq!(Degrees::new(27, 1, 1).unwrap().n_bits_ext, 28);
        assert_eq!(Degrees::new(28, 1, 1).unwrap().n_bits_ext, 29);
    }

    #[test]
    fn q_not_split_is_one_piece_of_its_bound() {
        let whole = QSplit { stride: 0, coefficients: vec![3 * 32 + 4 * 4 + 1] };
        for max_q_degree in [0, 3, 4, 100] {
            assert_eq!(QSplit::new(5, 3, 4, max_q_degree).unwrap(), whole, "maxQDegree {max_q_degree}");
        }
        assert_eq!(QSplit::new(3, 0, 1, 1).unwrap(), QSplit { stride: 0, coefficients: vec![2] });
        assert_eq!(whole.n_pieces(), 1);
    }

    #[test]
    fn the_pieces_of_q_have_m_n_coefficients_and_the_blinding_of_their_boundary() {
        // The fixture of the signed offsets: N = 32, qDeg = 3, |O|_max = 4, Q of 96 + 16 + 1 = 113.
        // M = 1: three pieces of 32 + 2, and the last 113 − 64 = 49, the blinding of the columns in it.
        let split = QSplit::new(5, 3, 4, 1).unwrap();
        assert_eq!(split, QSplit { stride: 32, coefficients: vec![34, 34, 49] });
        assert_eq!(split.n_pieces(), 3);
        // M = 2: two, of 64 + 2 and 113 − 64.
        assert_eq!(QSplit::new(5, 3, 4, 2).unwrap(), QSplit { stride: 64, coefficients: vec![66, 49] });
        // qDeg a multiple of M: the last piece has M·N and Q's blinding, (qDeg + 1)·|O|_max + 1.
        assert_eq!(QSplit::new(8, 4, 2, 2).unwrap(), QSplit { stride: 512, coefficients: vec![514, 512 + 5 * 2 + 1] });
        // The pieces cover Q: (m − 1)·S plus the last is Q's bound.
        for (n_bits, q_deg, max_openings, max_q_degree) in [(5, 3, 4, 1), (8, 7, 3, 2), (1, 9, 0, 4), (0, 5, 1, 1)] {
            let split = QSplit::new(n_bits, q_deg, max_openings, max_q_degree).unwrap();
            let m = split.n_pieces() as u64;
            assert_eq!(m, q_deg.div_ceil(max_q_degree));
            assert_eq!(split.stride, max_q_degree << n_bits);
            assert!(split.coefficients[..m as usize - 1].iter().all(|&c| c == split.stride + 2));
            let last = *split.coefficients.last().unwrap();
            assert_eq!((m - 1) * split.stride + last, q_coefficients(n_bits, q_deg, max_openings).unwrap());
            assert!(last > 1 << n_bits, "the last piece holds a row of N at least, and Q's blinding");
        }
        assert!(QSplit::new(29, 3, 1, 1).is_err());
    }
}
