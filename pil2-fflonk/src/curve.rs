//! BN254 points, as proofs and keys carry them.
//!
//! Two fields are in play and confusing them is silent: scalars live in `Fr`
//! (see [`crate::fr`]), point coordinates live in the base field `Fq`. They are
//! different primes of the same bit length, so a coordinate reduced mod `r`
//! usually still looks like a plausible number.
//!
//! Membership is checked here rather than assumed. A commitment is an untrusted
//! input, and a point off the curve -- or on the curve but in the wrong
//! subgroup -- breaks the algebra the pairing check relies on. BN254's G1 has
//! cofactor 1, so for G1 "on the curve" and "in the group" coincide and a curve
//! equation check is sufficient. G2 does not have that property, but the only
//! G2 element here comes from the verification key rather than from a proof.

use anyhow::{Context, Result, bail};
use num_bigint::BigUint;
use num_traits::{Num, One, Zero};

/// The BN254 base field modulus, over which point coordinates are taken.
///
/// Distinct from the scalar field modulus in [`crate::transcript::FR_MODULUS`].
pub const FQ_MODULUS: &str = "21888242871839275222246405745257275088696311157297823662689037894645226208583";

/// The curve is `y^2 = x^3 + 3`.
pub const CURVE_B: u32 = 3;

pub fn fq_modulus() -> BigUint {
    BigUint::from_str_radix(FQ_MODULUS, 10).expect("modulus parses")
}

/// A point of G1, in affine coordinates, or the point at infinity.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum G1Affine {
    Infinity,
    Point { x: BigUint, y: BigUint },
}

fn fq(s: &str, what: &str) -> Result<BigUint> {
    let v = BigUint::from_str_radix(s, 10).with_context(|| format!("parsing {what} {s:?}"))?;
    if v >= fq_modulus() {
        bail!("{what} {s:?} is not a reduced base-field element");
    }
    Ok(v)
}

impl G1Affine {
    /// Parse `[x, y, z]` as proofs and keys write it.
    ///
    /// The encoding is projective but only ever normalised: `z = 1` for a real
    /// point and `z = 0` for infinity. Anything else is rejected rather than
    /// normalised, so one point has one encoding -- otherwise a prover could
    /// re-scale a commitment and change the transcript without changing the
    /// point it commits to.
    pub fn from_json(point: &[String]) -> Result<Self> {
        if point.len() != 3 {
            bail!("G1 point has {} coordinates, want 3", point.len());
        }

        let z = fq(&point[2], "z")?;
        if z.is_zero() {
            return Ok(G1Affine::Infinity);
        }
        if !z.is_one() {
            bail!("G1 point is not normalised (z = {:?}); expected 1 or 0", point[2]);
        }

        Ok(G1Affine::Point { x: fq(&point[0], "x")?, y: fq(&point[1], "y")? })
    }

    /// Whether the point satisfies `y^2 = x^3 + 3`.
    ///
    /// Infinity is a group element and passes.
    pub fn is_on_curve(&self) -> bool {
        let (x, y) = match self {
            G1Affine::Infinity => return true,
            G1Affine::Point { x, y } => (x, y),
        };

        let q = fq_modulus();
        let lhs = (y * y) % &q;
        let rhs = ((x * x % &q) * x + BigUint::from(CURVE_B)) % &q;
        lhs == rhs
    }

    /// Parse and check membership in one step, naming what failed.
    pub fn parse_checked(point: &[String], what: &str) -> Result<Self> {
        let p = Self::from_json(point).context(what.to_string())?;
        if !p.is_on_curve() {
            bail!("{what} is not a point on BN254");
        }
        Ok(p)
    }
}

/// The generator of G1, `(1, 2)`.
pub fn g1_generator() -> G1Affine {
    G1Affine::Point { x: BigUint::one(), y: BigUint::from(2u32) }
}

/// An element of G2, with coordinates in `Fq2` written as `[c0, c1]`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct G2Affine {
    pub x: [BigUint; 2],
    pub y: [BigUint; 2],
}

impl G2Affine {
    /// Parse `[[x0, x1], [y0, y1], [z0, z1]]`, as the key writes `X_2`.
    pub fn from_json(point: &[Vec<String>]) -> Result<Self> {
        if point.len() != 3 {
            bail!("G2 point has {} coordinate pairs, want 3", point.len());
        }
        for (i, c) in point.iter().enumerate() {
            if c.len() != 2 {
                bail!("G2 coordinate {i} has {} components, want 2", c.len());
            }
        }

        let z = [fq(&point[2][0], "z0")?, fq(&point[2][1], "z1")?];
        if !z[0].is_one() || !z[1].is_zero() {
            bail!("G2 point is not normalised (z = {:?})", point[2]);
        }

        Ok(G2Affine {
            x: [fq(&point[0][0], "x0")?, fq(&point[0][1], "x1")?],
            y: [fq(&point[1][0], "y0")?, fq(&point[1][1], "y1")?],
        })
    }
}

/// The generator of G2, matching the one ffiasm's `AltBn128::Engine` is built
/// with. A verifier pairing against a different generator would reject every
/// proof, so this must agree with the setup that produced the key.
pub fn g2_generator() -> G2Affine {
    let n = |s: &str| BigUint::from_str_radix(s, 10).expect("generator parses");
    G2Affine {
        x: [
            n("10857046999023057135944570762232829481370756359578518086990519993285655852781"),
            n("11559732032986387107991004021392285783925812861821192530917403151452391805634"),
        ],
        y: [
            n("8495653923123431417604973247489272438418190587263600148770280649306958101930"),
            n("4082367875863433681332203403145435568316851327593401208105741076214120093531"),
        ],
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::reference;

    fn p(x: &str, y: &str) -> Vec<String> {
        vec![x.into(), y.into(), "1".into()]
    }

    /// The two moduli are different primes. Conflating them is the easiest way
    /// to write a verifier that is subtly wrong, so it is worth stating.
    #[test]
    fn the_base_field_is_not_the_scalar_field() {
        assert_ne!(fq_modulus(), crate::transcript::fr_modulus());
        assert!(fq_modulus() > crate::transcript::fr_modulus());
    }

    #[test]
    fn the_generators_are_on_their_curves() {
        assert!(g1_generator().is_on_curve());
        // (1, 2): 4 == 1 + 3.
        assert_eq!(g1_generator(), G1Affine::from_json(&p("1", "2")).unwrap());
    }

    /// Every commitment in a real proof, and every one fixed by the key, is a
    /// genuine curve point.
    #[test]
    fn every_reference_commitment_is_on_the_curve() {
        let r = reference::load();

        for (key, point) in &r.proof.polynomials {
            let g = G1Affine::parse_checked(point, key).unwrap_or_else(|e| panic!("{key}: {e}"));
            assert!(g.is_on_curve(), "{key}");
        }
        for (key, point) in &r.setup.f_commitments {
            assert!(G1Affine::parse_checked(point, key).unwrap().is_on_curve(), "{key}");
        }
    }

    #[test]
    fn the_keys_x2_parses_as_a_g2_point() {
        let r = reference::load();
        let x2 = G2Affine::from_json(&r.setup.x2).unwrap();
        assert_ne!(x2, g2_generator(), "X_2 is [x]_2, not the generator");
    }

    /// A point off the curve must be rejected. Accepting one would let a
    /// prover work in a group where the pairing identity does not constrain it.
    #[test]
    fn rejects_a_point_off_the_curve() {
        // (1, 3): 9 != 4.
        let bad = G1Affine::from_json(&p("1", "3")).unwrap();
        assert!(!bad.is_on_curve());
        assert!(G1Affine::parse_checked(&p("1", "3"), "test point").is_err());
    }

    /// Coordinates must be reduced, so a point has one encoding.
    #[test]
    fn rejects_an_unreduced_coordinate() {
        let over = (fq_modulus() + BigUint::one()).to_str_radix(10);
        assert!(G1Affine::from_json(&p(&over, "2")).is_err());
        assert!(G1Affine::from_json(&p("1", &over)).is_err());
    }

    /// A commitment must arrive normalised. Allowing an arbitrary z would give
    /// each point many encodings, and the transcript hashes the encoding.
    #[test]
    fn rejects_an_unnormalised_point() {
        assert!(G1Affine::from_json(&["1".into(), "2".into(), "2".into()]).is_err());
        assert!(G1Affine::from_json(&["1".into(), "2".into()]).is_err());
    }

    #[test]
    fn reads_the_point_at_infinity() {
        let inf = G1Affine::from_json(&["0".into(), "0".into(), "0".into()]).unwrap();
        assert_eq!(inf, G1Affine::Infinity);
        assert!(inf.is_on_curve());
    }

    #[test]
    fn rejects_a_malformed_g2_point() {
        assert!(G2Affine::from_json(&[vec!["1".into(), "0".into()]]).is_err());
        assert!(G2Affine::from_json(&[vec!["1".into()], vec!["1".into()], vec!["1".into()]]).is_err());
    }
}
