use std::ops::Add;

use p3_field::{BasedVectorSpace, Field, PrimeCharacteristicRing};
use p3_field::PrimeField64;
use p3_goldilocks::Goldilocks;

use crate::{goldilocks_quintic_extension::Squaring, GoldilocksQuinticExtension};

#[derive(Debug, Clone, PartialEq)]
struct EcGFp5 {
    x: GoldilocksQuinticExtension,
    y: GoldilocksQuinticExtension,
    is_infinity: bool,
}

impl Default for EcGFp5 {
    fn default() -> Self {
        Self { x: GoldilocksQuinticExtension::ZERO, y: GoldilocksQuinticExtension::ZERO, is_infinity: true }
    }
}

impl EcGFp5 {
    const fn new(x: GoldilocksQuinticExtension, y: GoldilocksQuinticExtension) -> Self {
        Self { x, y, is_infinity: false }
    }

    // Parameter `a` of the curve
    fn a() -> GoldilocksQuinticExtension {
        GoldilocksQuinticExtension::from_basis_coefficients_slice(&[
            Goldilocks::from_u64(6148914689804861439),
            Goldilocks::from_u64(263),
            Goldilocks::ZERO,
            Goldilocks::ZERO,
            Goldilocks::ZERO,
        ])
    }

    // Generator of the subgroup of size n, obtained by lifting zero to the curve
    #[allow(dead_code)]
    fn subgroup_gen() -> Self {
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

    fn is_infinity(&self) -> (Option<&EcGFp5>, bool) {
        if self.is_infinity { (None, true) } else { (Some(self), false) }
    }

    // Check if the point is on the curve: y^2 = x^3 + ax + b
    #[allow(dead_code)]
    fn is_on_curve(&self) -> bool {
        if self.is_infinity {
            return true;
        }

        let x = self.x;
        let y = self.y;
        y.square() == x.cube() + Self::a() * x + Self::b()
    }

    // Addition assuming points are not the point at infinity and not in the same vertical line
    fn add_incomplete(&self, other: &Self) -> Self {
        let slope = (other.y - self.y) / (other.x - self.x);
        let x = slope.square() - self.x - other.x;
        let y = slope * (self.x - x) - self.y;
        Self::new(x, y)
    }

    // Addition routine
    fn add_complete(&self, other: &Self) -> Self {
        let (p1, is_inf1) = self.is_infinity();
        let (p2, is_inf2) = other.is_infinity();

        // If one of the points is the point at infinity, return the other point.
        if is_inf1 {
            return other.clone();
        } else if is_inf2 {
            return self.clone();
        }

        let p1 = p1.unwrap();
        let p2 = p2.unwrap();

        // I ordered the following cases by probability of occurrence

        // If the points are different and not on the same vertical line
        if p1.x != p2.x {
            return p1.add_incomplete(p2);
        }

        // If the points are the same
        if p1.y == p2.y {
            return p1.double_incomplete();
        }

        // If the points are different and on the same vertical line
        Self::default()
    }

    // Doubling routine assuming the point is not the point at infinity
    fn double_incomplete(&self) -> Self {
        let slope = (self.x.square() * Goldilocks::from_u8(3) + Self::a()) / (self.y * Goldilocks::from_u8(2));
        let x = slope.square() - self.x.double();
        let y = slope * (self.x - x) - self.y;
        Self::new(x, y)
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

impl EcGFp5 {
    #[allow(dead_code)]
    const Z: [u64; 5] = [18446744069414584317, 18446744069414584320, 0, 0, 0]; // find_z_sswu(K,a,b)

    #[allow(dead_code)]
    const C1: [u64; 5] =
        [6585749426319121644, 16990361517133133838, 3264760655763595284, 16784740989273302855, 13434657726302040770]; // -b/a

    #[allow(dead_code)]
    const C2: [u64; 5] =
        [4795794222525505369, 3412737461722269738, 8370187669276724726, 7130825117388110979, 12052351772713910496]; // -1/Z

    // Parameter `b` of the curve
    fn b() -> GoldilocksQuinticExtension {
        GoldilocksQuinticExtension::from_basis_coefficients_slice(&[
            Goldilocks::from_u64(15713893096167979237),
            Goldilocks::from_u64(6148914689804861265),
            Goldilocks::ZERO,
            Goldilocks::ZERO,
            Goldilocks::ZERO,
        ])
    }

    #[allow(dead_code)]
    fn hash_to_curve(f0: GoldilocksQuinticExtension, f1: GoldilocksQuinticExtension) -> Self {
        let p0 = Self::map_to_curve(f0);
        let p1 = Self::map_to_curve(f1);
        let p = p0 + p1;
        p.clear_cofactor()
    }

    #[allow(dead_code)]
    // Implements the Simplified Shallue-van de Woestijne-Ulas method as specified in the IETF draft `draft-irtf-cfrg-hash-to-curve-10``
    fn map_to_curve(e: GoldilocksQuinticExtension) -> Self {
        let z = GoldilocksQuinticExtension::from_basis_coefficients_fn(|i| Goldilocks::from_u64(Self::Z[i]));

        let tv1 = z * e.square();
        let mut tv2 = tv1.square();
        let mut x1 = if let Some(inv) = (tv1 + tv2).try_inverse() { inv } else { GoldilocksQuinticExtension::ZERO };
        let e1 = x1 == GoldilocksQuinticExtension::ZERO;
        x1 += GoldilocksQuinticExtension::ONE;

        if e1 {
            // If (tv1 + tv2) == 0, set x1 = -1 / Z
            x1 = GoldilocksQuinticExtension::from_basis_coefficients_fn(|i| Goldilocks::from_u64(Self::C2[i]));
        }
        let c1 = GoldilocksQuinticExtension::from_basis_coefficients_fn(|i| Goldilocks::from_u64(Self::C1[i]));
        x1 *= c1; // If (tv1 + tv2) == 0, x1 = B / (Z * A), else x1 = (-B / A) * (1 + x1)

        // gx1 = x1^3 + A * x1 + B
        let mut gx1 = x1.square();
        gx1 += Self::a();
        gx1 *= x1;
        gx1 += Self::b();

        // x2 = Z * e^2 * x1
        let x2 = tv1 * x1;

        // gx2 = (Z * e^2)^3 * gx1
        tv2 *= tv1;
        let gx2 = tv2 * gx1;

        let e2 = gx1.is_square().1;
        // If gx1 is square, x = x1, y = sqrt(gx1), else x = x2 , y = sqrt(gx2)
        let (x, y) =
            if e2 { (x1, gx1.sqrt().expect("gx1 is square")) } else { (x2, gx2.sqrt().expect("gx2 is square")) };

        // Fix the sign of y
        if sign0(e) == sign0(y) {
            return Self::new(x, y);
        } else {
            return Self::new(x, -y);
        }

        fn sign0(e: GoldilocksQuinticExtension) -> bool {
            let e_coeffs: &[Goldilocks] = e.as_basis_coefficients_slice();
            let mut result = false;
            let mut zero = true;
            for coeff in e_coeffs.iter() {
                let sign_i = (coeff.as_canonical_u64() & 1) == 1;
                let zero_i = coeff.is_zero();
                result = result || (zero && sign_i);
                zero = zero && zero_i;
            }

            result
        }
    }

    fn clear_cofactor(&self) -> Self {
        self + self
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
        let p = EcGFp5::default();
        assert!(p.is_on_curve());

        // Test subgroup generator
        let p = EcGFp5::subgroup_gen();
        assert!(p.is_on_curve());
    }

    #[test]
    fn test_addition() {
        // Test the point at infinity
        let infinity = EcGFp5::default();
        assert_eq!(&infinity + &infinity, infinity);

        let p1 = EcGFp5::subgroup_gen();
        assert_eq!(&p1 + &infinity, p1);
        assert_eq!(&infinity + &p1, p1);

        let p1_neg = EcGFp5::new(p1.x, -p1.y);
        assert!(p1_neg.is_on_curve());
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
        assert!(p2.is_on_curve());
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
        assert!(p1p2.is_on_curve());
        assert_eq!(p1p2, p1p2_real);
    }

    #[test]
    fn map_to_curve() {
        // Edge cases occur at the roots of the polynomial f(x) = Z^2 · x^4 + Z · x^2 = x^2 · (Z^2 · x^2 + Z)
        // which in our field only happens when x = 0
        let p = EcGFp5::map_to_curve(GoldilocksQuinticExtension::ZERO);
        assert!(p.is_on_curve());
    }

    #[test]
    fn test_hash_to_curve() {
        let f0 = GoldilocksQuinticExtension::ZERO;
        let f1 = GoldilocksQuinticExtension::ZERO;
        let p = EcGFp5::hash_to_curve(f0, f1);
        assert!(p.is_on_curve());

        let f0 = GoldilocksQuinticExtension::ONE;
        let f1 = GoldilocksQuinticExtension::ONE;
        let p = EcGFp5::hash_to_curve(f0, f1);
        assert!(p.is_on_curve());

        let f1 = GoldilocksQuinticExtension::GENERATOR;
        let p = EcGFp5::hash_to_curve(f0, f1);
        assert!(p.is_on_curve());

        // Random tests
        let mut rng = rng();
        for _ in 0..1000 {
            let f0: GoldilocksQuinticExtension = StandardUniform.sample(&mut rng);
            let f1: GoldilocksQuinticExtension = StandardUniform.sample(&mut rng);
            let p = EcGFp5::hash_to_curve(f0, f1);
            assert!(p.is_on_curve());
        }
    }
}
