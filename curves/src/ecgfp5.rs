use std::ops::Add;

use fields::{Field, ExtensionField, PrimeField64, Goldilocks, GoldilocksQuinticExtension};

use crate::{curve::EllipticCurve};

/// The [EcGFp5](https://eprint.iacr.org/2022/274.pdf) curve is defined by the equation:
///         y² = x³ + Ax + B
/// over the field `GoldilocksQuinticExtension`, where:
///   - A = 6148914689804861439 + 263·X
///   - B = 15713893096167979237 + 6148914689804861265·X
#[derive(Debug, Clone, PartialEq)]
pub struct EcGFp5 {
    x: GoldilocksQuinticExtension,
    y: GoldilocksQuinticExtension,
    is_infinity: bool,
}

impl EllipticCurve<Goldilocks, GoldilocksQuinticExtension> for EcGFp5 {
    const A: [u64; 5] = [6148914689804861439, 263, 0, 0, 0];
    const B: [u64; 5] = [15713893096167979237, 6148914689804861265, 0, 0, 0];
    const Z: [u64; 5] = [18446744069414584317, 18446744069414584320, 0, 0, 0];
    const C1: [u64; 5] =
        [6585749426319121644, 16990361517133133838, 3264760655763595284, 16784740989273302855, 13434657726302040770];
    const C2: [u64; 5] =
        [4795794222525505369, 3412737461722269738, 8370187669276724726, 7130825117388110979, 12052351772713910496];

    fn new(x: GoldilocksQuinticExtension, y: GoldilocksQuinticExtension) -> Self {
        Self { x, y, is_infinity: false }
    }

    fn infinity() -> Self {
        Self { x: GoldilocksQuinticExtension::ZERO, y: GoldilocksQuinticExtension::ZERO, is_infinity: true }
    }

    fn generator() -> Self {
        let x = GoldilocksQuinticExtension::ZERO;
        let y = GoldilocksQuinticExtension::from_basis_coefficients_slice(&[
            Goldilocks::from_u64(11002749681768771274),
            Goldilocks::from_u64(11642892185553879191),
            Goldilocks::from_u64(663487151061499164),
            Goldilocks::from_u64(2764891638068209098),
            Goldilocks::from_u64(2343917403129570002),
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
        // Cofactor is 2
        self.double_complete()
    }
}

// Operator overloading
impl Add<EcGFp5> for EcGFp5 {
    type Output = EcGFp5;

    fn add(self, other: EcGFp5) -> EcGFp5 {
        self.add_complete(&other)
    }
}

impl Add<EcGFp5> for &EcGFp5 {
    type Output = EcGFp5;

    fn add(self, other: EcGFp5) -> EcGFp5 {
        self.add_complete(&other)
    }
}

impl Add<&EcGFp5> for EcGFp5 {
    type Output = EcGFp5;

    fn add(self, other: &EcGFp5) -> EcGFp5 {
        self.add_complete(other)
    }
}

impl Add<&EcGFp5> for &EcGFp5 {
    type Output = EcGFp5;

    fn add(self, other: &EcGFp5) -> EcGFp5 {
        self.add_complete(other)
    }
}

#[cfg(test)]
mod tests {
    use rand::{rng, Rng};

    use super::*;

    #[test]
    fn test_is_on_curve() {
        // Test the point at infinity
        let p = EcGFp5::infinity();
        debug_assert!(p.is_on_curve());

        // Test subgroup generator
        let p = EcGFp5::generator();
        debug_assert!(p.is_on_curve());
    }

    #[test]
    fn test_addition() {
        // Test the point at infinity
        let infinity = EcGFp5::infinity();
        assert_eq!(&infinity + &infinity, infinity);

        let p1 = EcGFp5::generator();
        assert_eq!(&p1 + &infinity, p1);
        assert_eq!(&infinity + &p1, p1);

        let p1_neg = EcGFp5::new(p1.x, -p1.y);
        debug_assert!(p1_neg.is_on_curve());
        assert_eq!(&p1 + &p1_neg, infinity);
        assert_eq!(&p1_neg + &p1, infinity);

        let p2 = &p1 + &p1;
        let p2_real = EcGFp5::new(
            GoldilocksQuinticExtension::from_basis_coefficients_slice(&[
                Goldilocks::from_u64(15622315679105259),
                Goldilocks::from_u64(9233938668908914291),
                Goldilocks::from_u64(14943848313873695123),
                Goldilocks::from_u64(1210072233909776598),
                Goldilocks::from_u64(2930298871824402754),
            ]),
            GoldilocksQuinticExtension::from_basis_coefficients_slice(&[
                Goldilocks::from_u64(4471391967326616314),
                Goldilocks::from_u64(15391191233422108365),
                Goldilocks::from_u64(12545589738280459763),
                Goldilocks::from_u64(18441655962801752599),
                Goldilocks::from_u64(12893054396778703652),
            ]),
        );
        debug_assert!(p2.is_on_curve());
        assert_eq!(p2, p2_real);

        let p1p2 = &p1 + &p2;
        let p1p2_real = EcGFp5::new(
            GoldilocksQuinticExtension::from_basis_coefficients_slice(&[
                Goldilocks::from_u64(6535296575033610464),
                Goldilocks::from_u64(10296938272802226861),
                Goldilocks::from_u64(6062249350014962804),
                Goldilocks::from_u64(177124804235033586),
                Goldilocks::from_u64(7276441891717506516),
            ]),
            GoldilocksQuinticExtension::from_basis_coefficients_slice(&[
                Goldilocks::from_u64(18178031365678595731),
                Goldilocks::from_u64(11606916788478585122),
                Goldilocks::from_u64(6488177608160934983),
                Goldilocks::from_u64(12544791818053125737),
                Goldilocks::from_u64(14568464258697035512),
            ]),
        );
        debug_assert!(p1p2.is_on_curve());
        assert_eq!(p1p2, p1p2_real);
    }

    #[test]
    fn map_to_curve() {
        // Edge cases occur at the roots of the polynomial f(x) = Z^2 · x^4 + Z · x^2 = x^2 · (Z^2 · x^2 + Z)
        // which in our field only happens when x = 0
        let p = EcGFp5::map_to_curve(GoldilocksQuinticExtension::ZERO);
        debug_assert!(p.is_on_curve());
    }

    #[test]
    fn test_hash_to_curve() {
        let f0 = GoldilocksQuinticExtension::ZERO;
        let f1 = GoldilocksQuinticExtension::ZERO;
        let p = EcGFp5::hash_to_curve(f0, f1);
        debug_assert!(p.is_on_curve());

        let f0 = GoldilocksQuinticExtension::ONE;
        let f1 = GoldilocksQuinticExtension::ONE;
        let p = EcGFp5::hash_to_curve(f0, f1);
        debug_assert!(p.is_on_curve());

        let f1 = GoldilocksQuinticExtension::GENERATOR;
        let p = EcGFp5::hash_to_curve(f0, f1);
        debug_assert!(p.is_on_curve());

        // Random tests
        let mut rng = rng();
        for _ in 0..1000 {
            let f0: GoldilocksQuinticExtension = GoldilocksQuinticExtension::from_basis_coefficients_slice(
                &[Goldilocks::from_u64(rng.random_range(0..=(1 << 63) - 1)); 5],
            );
            let f1: GoldilocksQuinticExtension = GoldilocksQuinticExtension::from_basis_coefficients_slice(
                &[Goldilocks::from_u64(rng.random_range(0..=(1 << 63) - 1)); 5],
            );
            let p = EcGFp5::hash_to_curve(f0, f1);
            debug_assert!(p.is_on_curve());
        }
    }
}
