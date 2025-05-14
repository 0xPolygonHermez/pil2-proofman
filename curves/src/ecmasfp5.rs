use std::ops::Add;

use fields::{Field, ExtensionField, PrimeField64, Goldilocks, GoldilocksQuinticExtension};

use crate::curve::EllipticCurve;

/// The EcMasFp5 curve is defined by the equation:
///         y² = x³ + Ax + B
/// over the field `GoldilocksQuinticExtension`, where:
///   - A = 3
///   - B = 8·X⁴
#[derive(Debug, Clone, PartialEq)]
pub struct EcMasFp5 {
    x: GoldilocksQuinticExtension,
    y: GoldilocksQuinticExtension,
    is_infinity: bool,
}

impl EllipticCurve<Goldilocks, GoldilocksQuinticExtension> for EcMasFp5 {
    const A: [u64; 5] = [3, 0, 0, 0, 0];
    const B: [u64; 5] = [0, 0, 0, 0, 8];
    const Z: [u64; 5] = [9, 1, 0, 0, 0];
    const C1: [u64; 5] = [0, 0, 0, 0, 12297829379609722878];
    const C2: [u64; 5] =
        [17696091661387705534, 83405823114097643, 16387838525800286325, 16625873122103441396, 8400871913885497801];

    fn new(x: GoldilocksQuinticExtension, y: GoldilocksQuinticExtension) -> Self {
        Self { x, y, is_infinity: false }
    }

    fn infinity() -> Self {
        Self { x: GoldilocksQuinticExtension::ZERO, y: GoldilocksQuinticExtension::ZERO, is_infinity: true }
    }

    fn generator() -> Self {
        let x = GoldilocksQuinticExtension::ZERO;
        let y = GoldilocksQuinticExtension::from_basis_coefficients_slice(&[
            Goldilocks::ZERO,
            Goldilocks::ZERO,
            Goldilocks::from_u64(18446741870424883713),
            Goldilocks::ZERO,
            Goldilocks::ZERO,
        ]);
        Self::new(x, y)
    }

    fn x(&self) -> GoldilocksQuinticExtension {
        self.x
    }

    fn y(&self) -> GoldilocksQuinticExtension {
        self.y
    }

    fn is_infinity(&self) -> bool {
        self.is_infinity
    }

    fn clear_cofactor(&self) -> Self {
        // Cofactor is 1
        self.clone()
    }
}

// Operator overloading
impl Add<EcMasFp5> for EcMasFp5 {
    type Output = EcMasFp5;

    fn add(self, other: EcMasFp5) -> EcMasFp5 {
        self.add_complete(&other)
    }
}

impl Add<EcMasFp5> for &EcMasFp5 {
    type Output = EcMasFp5;

    fn add(self, other: EcMasFp5) -> EcMasFp5 {
        self.add_complete(&other)
    }
}

impl Add<&EcMasFp5> for EcMasFp5 {
    type Output = EcMasFp5;

    fn add(self, other: &EcMasFp5) -> EcMasFp5 {
        self.add_complete(other)
    }
}

impl Add<&EcMasFp5> for &EcMasFp5 {
    type Output = EcMasFp5;

    fn add(self, other: &EcMasFp5) -> EcMasFp5 {
        self.add_complete(other)
    }
}

#[cfg(test)]
mod tests {
    use rand::{
        distr::{Distribution, StandardUniform},
        rng,
    };

    use super::*;

    #[test]
    fn test_is_on_curve() {
        // Test the point at infinity
        let p = EcMasFp5::infinity();
        assert!(p.is_on_curve());

        // Test subgroup generator
        let p = EcMasFp5::generator();
        assert!(p.is_on_curve());
    }

    #[test]
    fn test_addition() {
        // Test the point at infinity
        let infinity = EcMasFp5::infinity();
        assert_eq!(&infinity + &infinity, infinity);

        let p1 = EcMasFp5::generator();
        assert_eq!(&p1 + &infinity, p1);
        assert_eq!(&infinity + &p1, p1);

        let p1_neg = EcMasFp5::new(p1.x, -p1.y);
        assert!(p1_neg.is_on_curve());
        assert_eq!(&p1 + &p1_neg, infinity);
        assert_eq!(&p1_neg + &p1, infinity);

        let p2 = &p1 + &p1;
        let p2_real = EcMasFp5::new(
            GoldilocksQuinticExtension::from_basis_coefficients_slice(&[
                Goldilocks::ZERO,
                Goldilocks::from_u64(16717361812906967041),
                Goldilocks::ZERO,
                Goldilocks::ZERO,
                Goldilocks::ZERO,
            ]),
            GoldilocksQuinticExtension::from_basis_coefficients_slice(&[
                Goldilocks::ZERO,
                Goldilocks::ZERO,
                Goldilocks::from_u64(2198989700608),
                Goldilocks::ZERO,
                Goldilocks::from_u64(12884705277),
            ]),
        );
        assert!(p2.is_on_curve());
        assert_eq!(p2, p2_real);

        let p1p2 = &p1 + &p2;
        let p1p2_real = EcMasFp5::new(
            GoldilocksQuinticExtension::from_basis_coefficients_slice(&[
                Goldilocks::ZERO,
                Goldilocks::ZERO,
                Goldilocks::from_u64(14347467609544680335),
                Goldilocks::ZERO,
                Goldilocks::from_u64(12297829379609722902),
            ]),
            GoldilocksQuinticExtension::from_basis_coefficients_slice(&[
                Goldilocks::from_u64(18442240538507739137),
                Goldilocks::ZERO,
                Goldilocks::from_u64(18446737472445482497),
                Goldilocks::from_u64(17592730746524804590),
                Goldilocks::ZERO,
            ]),
        );
        assert!(p1p2.is_on_curve());
        assert_eq!(p1p2, p1p2_real);
    }

    #[test]
    fn map_to_curve() {
        // Edge cases occur at the roots of the polynomial f(x) = Z^2 · x^4 + Z · x^2 = x^2 · (Z^2 · x^2 + Z)
        // which in our field only happens when x = 0
        let p = EcMasFp5::map_to_curve(GoldilocksQuinticExtension::ZERO);
        assert!(p.is_on_curve());
    }

    #[test]
    fn test_hash_to_curve() {
        let f0 = GoldilocksQuinticExtension::ZERO;
        let f1 = GoldilocksQuinticExtension::ZERO;
        let p = EcMasFp5::hash_to_curve(f0, f1);
        assert!(p.is_on_curve());

        let f0 = GoldilocksQuinticExtension::ONE;
        let f1 = GoldilocksQuinticExtension::ONE;
        let p = EcMasFp5::hash_to_curve(f0, f1);
        assert!(p.is_on_curve());

        let f1 = GoldilocksQuinticExtension::GENERATOR;
        let p = EcMasFp5::hash_to_curve(f0, f1);
        assert!(p.is_on_curve());

        // Random tests
        let mut rng = rng();
        for _ in 0..1000 {
            let f0: GoldilocksQuinticExtension = StandardUniform.sample(&mut rng);
            let f1: GoldilocksQuinticExtension = StandardUniform.sample(&mut rng);
            let p = EcMasFp5::hash_to_curve(f0, f1);
            assert!(p.is_on_curve());
        }
    }
}
