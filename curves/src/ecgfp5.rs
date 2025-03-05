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

    fn is_infinity(&self) -> (Option<&EcGFp5>, bool) {
        if self.is_infinity { (None, true) } else { (Some(self), false) }
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
            return p1.add_incomplete(&p2);
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

impl Add<&EcGFp5> for &EcGFp5 {
    type Output = EcGFp5;

    fn add(self, other: &EcGFp5) -> EcGFp5 {
        self.add_complete(other)
    }
}

impl EcGFp5 {
    const Z: [u64; 5] = [18446744069414584317, 18446744069414584320, 0, 0, 0]; // find_z_sswu(K,a,b)
    const C1: [u64; 5] =
        [6585749426319121644, 16990361517133133838, 3264760655763595284, 16784740989273302855, 13434657726302040770]; // -b/a
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

    fn hash_to_curve(f0: GoldilocksQuinticExtension, f1: GoldilocksQuinticExtension) -> Self {
        let p0 = Self::map_to_curve(f0);
        let p1 = Self::map_to_curve(f1);
        let p = p0 + p1;
        p.clear_cofactor()
    }

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
            for i in 0..5 {
                let sign_i = (e_coeffs[i].as_canonical_u64() & 1) == 1;
                let zero_i = e_coeffs[i].is_zero();
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

// TODO: Add tests
