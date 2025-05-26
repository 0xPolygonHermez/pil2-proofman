use p3_field::{Field, ExtensionField};

use crate::goldilocks_quintic_extension::SquaringFp5;

/// Trait for elliptic curves
pub trait EllipticCurve<F: Field, K: ExtensionField<F> + SquaringFp5>: Clone {
    /// Parameter `A` of the curve
    const A: [u64; 5];
    /// Parameter `B` of the curve
    const B: [u64; 5];
    /// Constant `Z` of the Simplified SWU map, it can be found using https://www.ietf.org/archive/id/draft-irtf-cfrg-hash-to-curve-10.html#sswu-z-code
    const Z: [u64; 5];
    /// -B/A
    const C1: [u64; 5];
    /// -1/Z
    const C2: [u64; 5];

    /// Create a new point on the curve
    fn new(x: K, y: K) -> Self;

    /// Return the point at infinity
    fn infinity() -> Self;

    /// Return the generator of the subgroup of the curve
    #[allow(dead_code)]
    fn generator() -> Self;

    /// Return the x coordinate of the point
    fn x(&self) -> K;

    /// Return the y coordinate of the point
    fn y(&self) -> K;

    /// Check if the point is the point at infinity
    fn is_infinity(&self) -> bool;

    /// Check if the point is on the curve
    #[allow(dead_code)]
    fn is_on_curve(&self) -> bool {
        if self.is_infinity() {
            return true;
        }

        let a = K::from_basis_coefficients_fn(|i| F::from_u64(Self::A[i]));
        let b = K::from_basis_coefficients_fn(|i| F::from_u64(Self::B[i]));
        let x = self.x();
        let y = self.y();
        y.square() == x.cube() + a * x + b
    }

    /// Addition assuming points are not the point at infinity and not in the same vertical line
    fn add_incomplete(&self, other: &Self) -> Self
    where
        Self: Sized,
    {
        let x1 = self.x();
        let y1 = self.y();
        let x2 = other.x();
        let y2 = other.y();

        let slope = (y2 - y1) / (x2 - x1);
        let x3 = slope.square() - x1 - x2;
        let y3 = slope * (x1 - x3) - y1;
        Self::new(x3, y3)
    }

    /// Doubling routine assuming the point is not the point at infinity and not of order 2
    fn double_incomplete(&self) -> Self
    where
        Self: Sized,
    {
        let x = self.x();
        let y = self.y();

        let a = K::from_basis_coefficients_fn(|i| F::from_u64(Self::A[i]));
        let slope = (x.square() * F::from_u8(3) + a) / (y * F::from_u8(2));
        let x3 = slope.square() - x.double();
        let y3 = slope * (x - x3) - y;
        Self::new(x3, y3)
    }

    /// Addition routine
    fn add_complete(&self, other: &Self) -> Self {
        // If one of the points is the point at infinity, return the other point.
        if self.is_infinity() {
            return other.clone();
        } else if other.is_infinity() {
            return self.clone();
        }

        // I ordered the following cases by probability of occurrence

        // If the points are different and not on the same vertical line
        if self.x() != other.x() {
            return self.add_incomplete(other);
        }

        // If the points are the same
        if self.y() == other.y() {
            // If the point is of order 2
            if self.y().is_zero() {
                return Self::infinity();
            }

            return self.double_incomplete();
        }

        // If the points are different and on the same vertical line
        Self::infinity()
    }

    /// Doubling routine
    fn double_complete(&self) -> Self {
        // If the point is the point at infinity or of order 2
        if self.is_infinity() || self.y().is_zero() {
            return Self::infinity();
        }

        self.double_incomplete()
    }

    /// Map a point on the curve to the working subgroup on the curve
    fn clear_cofactor(&self) -> Self;

    /// Map a field element to a point on the curve
    fn map_to_curve(f: K) -> Self {
        let z = K::from_basis_coefficients_fn(|i| F::from_u64(Self::Z[i]));

        let tv1 = z * f.square();
        let mut tv2 = tv1.square();
        let mut x1 = if let Some(inv) = (tv1 + tv2).try_inverse() { inv } else { K::ZERO };
        let e1 = x1 == K::ZERO;
        x1 += K::ONE;

        if e1 {
            // If (tv1 + tv2) == 0, set x1 = -1 / Z
            x1 = K::from_basis_coefficients_fn(|i| F::from_u64(Self::C2[i]));
        }
        let c1 = K::from_basis_coefficients_fn(|i| F::from_u64(Self::C1[i]));
        x1 *= c1; // If (tv1 + tv2) == 0, x1 = B / (Z * A), else x1 = (-B / A) * (1 + x1)

        // gx1 = x1^3 + A * x1 + B
        let a = K::from_basis_coefficients_fn(|i| F::from_u64(Self::A[i]));
        let b = K::from_basis_coefficients_fn(|i| F::from_u64(Self::B[i]));
        let mut gx1 = x1.square();
        gx1 += a;
        gx1 *= x1;
        gx1 += b;

        // x2 = Z * e^2 * x1
        let x2 = tv1 * x1;

        // gx2 = (Z * e^2)^3 * gx1 = x2^3 + A * x2 + B
        tv2 *= tv1;
        let gx2 = tv2 * gx1;

        let e2 = gx1.is_square().1;
        // If gx1 is square, x = x1, y = sqrt(gx1), else x = x2 , y = sqrt(gx2)
        let (x, y) =
            if e2 { (x1, gx1.sqrt().expect("gx1 is square")) } else { (x2, gx2.sqrt().expect("gx2 is square")) };

        // Fix the sign of y
        if f.sign0() == y.sign0() {
            Self::new(x, y)
        } else {
            Self::new(x, -y)
        }
    }

    /// Hash to the curve
    #[allow(dead_code)]
    fn hash_to_curve(f0: K, f1: K) -> Self
    where
        Self: Sized,
    {
        let p0 = Self::map_to_curve(f0);
        let p1 = Self::map_to_curve(f1);
        let p = p0.add_complete(&p1);
        p.clear_cofactor()
    }
}
