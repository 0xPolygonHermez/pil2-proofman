//! The degrees of spec A.1 for an AIR: the bound on the coefficients of each committed polynomial
//! and of `Q`, and the extended domain the prover evaluates `Q` on. They are derived, not stored:
//! the setup derives them before the layout exists (`pilfflonk_setup::layout`), and the prover from
//! the pilfflonkinfo ([`PilfflonkInfo::degrees`]), both with these functions. The C++ prover
//! derives them again for itself (`AirDegrees` in `pil2-stark/src/pilfflonk/pilfflonk_proving_key.hpp`),
//! and the orchestrator checks that it agrees.
//!
//! - A fixed column has `N` coefficients: it has no blinding (A.3).
//! - A committed column opened at `|O|` offsets has `N + |O| + 1`, of which `|O| + 1` are its
//!   blinding's, `(X^N − 1)·b(X)` (A.3).
//! - `Q`, not split, has `qDeg·N + (qDeg+1)·|O|_max + 1` (A.1), `|O|_max` the most offsets of a
//!   committed column (with packing, of an `f` of a committed stage).
//! - The extended domain is the smallest power of two `≥` `Q`'s coefficients and `≥ N + |O|_max +
//!   1`, the coefficients of the column with the most blinding, which the prover also extends to
//!   it: `2^nBitsExt` points, `nBitsExt ≤ 28` for BN254's roots of unity (checked by the callers).

use crate::error::{invalid, PilfflonkResult};
use crate::global_info::MAX_NBITS;
use crate::pilfflonk_info::PilfflonkInfo;

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
/// committed one, whose blinding `(X^N − 1)·b(X)` has `|O| + 1` coefficients (A.3).
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

/// The bound on the coefficients of `Q` not split (A.1): `qDeg·N + (qDeg+1)·|O|_max + 1`.
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

/// `nBitsExt` (A.1): the smallest power of two `≥` `Q`'s coefficients and `≥ N + |O|_max + 1`,
/// the coefficients of the column with the most blinding. It is not checked against the
/// 2-adicity here: the callers do.
pub fn n_bits_ext(n_bits: u64, q_coefficients: u64, max_openings: u64) -> PilfflonkResult<u64> {
    let column = column_coefficients(n_bits, 1, max_openings)?;
    match q_coefficients.max(column).checked_next_power_of_two() {
        Some(points) => Ok(u64::from(points.trailing_zeros())),
        None => overflow("the extended domain"),
    }
}

impl Degrees {
    /// The degrees of an AIR of `2^n_bits` rows whose constraint polynomial has degree `q_deg`
    /// (A.1), and whose committed columns (stage ≥ 1) are opened at `max_openings` offsets at most.
    pub fn new(n_bits: u64, q_deg: u64, max_openings: u64) -> PilfflonkResult<Self> {
        let q_coefficients = q_coefficients(n_bits, q_deg, max_openings)?;
        let n_bits_ext = n_bits_ext(n_bits, q_coefficients, max_openings)?;
        Ok(Degrees { n_bits, max_openings, q_coefficients, n_bits_ext })
    }
}

impl PilfflonkInfo {
    /// The degrees of the AIR (A.1), from its layout: `|O|_max` is the most offsets of an `f` of a
    /// committed stage (`1 … nStages`). What the setup derived them from, and so what its layout's
    /// degrees are made of.
    pub fn degrees(&self) -> PilfflonkResult<Degrees> {
        let committed = self.layout.0.iter().filter(|f| f.stage >= 1 && f.stage <= self.n_stages);
        let max_openings = committed.map(|f| f.offsets.len() as u64).max().unwrap_or(0);
        Degrees::new(self.n_bits, self.q_deg, max_openings)
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
        // The column decides: with qDeg = 0, Q has 2 coefficients and a column N + 2 (M5).
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
}
