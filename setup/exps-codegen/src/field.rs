//! Exact Goldilocks (p = 2^64 - 2^32 + 1) and cubic-extension arithmetic
//! (Fp[x]/(x^3 - x - 1)), bit-identical to the device helpers in
//! `gen_common.cuh`. Used for constant folding and for the host-side
//! equivalence check of the optimized IR.

pub const P: u64 = 0xFFFF_FFFF_0000_0001;

#[inline]
pub fn add(a: u64, b: u64) -> u64 {
    let (s, c) = a.overflowing_add(b);
    let s = if c { s.wrapping_sub(P) } else { s };
    if s >= P {
        s - P
    } else {
        s
    }
}

#[inline]
pub fn sub(a: u64, b: u64) -> u64 {
    if a >= b {
        a - b
    } else {
        a.wrapping_sub(b).wrapping_add(P)
    }
}

#[inline]
pub fn mul(a: u64, b: u64) -> u64 {
    ((a as u128 * b as u128) % P as u128) as u64
}

/// One Fp3 element `a + b x + c x^2`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct F3 {
    pub a: u64,
    pub b: u64,
    pub c: u64,
}

impl F3 {
    pub const fn base(a: u64) -> F3 {
        F3 { a, b: 0, c: 0 }
    }
}

/// Same Karatsuba arrangement as `cg_mul33` (the result is the exact field
/// product, so the arrangement only matters for reading the two side by side).
pub fn mul33(x: F3, y: F3) -> F3 {
    let aa = mul(add(x.a, x.b), add(y.a, y.b));
    let bb = mul(add(x.a, x.c), add(y.a, y.c));
    let cc = mul(add(x.b, x.c), add(y.b, y.c));
    let d = mul(x.a, y.a);
    let e = mul(x.b, y.b);
    let f = mul(x.c, y.c);
    let g = sub(d, e);
    F3 { a: sub(add(cc, g), f), b: sub(sub(sub(add(aa, cc), e), e), d), c: sub(bb, g) }
}

pub fn mul31(x: F3, s: u64) -> F3 {
    F3 { a: mul(x.a, s), b: mul(x.b, s), c: mul(x.c, s) }
}

pub fn add33(x: F3, y: F3) -> F3 {
    F3 { a: add(x.a, y.a), b: add(x.b, y.b), c: add(x.c, y.c) }
}

pub fn sub33(x: F3, y: F3) -> F3 {
    F3 { a: sub(x.a, y.a), b: sub(x.b, y.b), c: sub(x.c, y.c) }
}

/// Generic op on values of any dim pair (1 or 3), mirroring the `cg_*` helpers.
pub fn apply(op: &str, x: F3, xd: u64, y: F3, yd: u64) -> F3 {
    match (op, xd, yd) {
        ("add", _, _) => add33(x, y),
        ("sub", _, _) => sub33(x, y),
        ("mul", 1, 1) => F3::base(mul(x.a, y.a)),
        ("mul", 3, 1) => mul31(x, y.a),
        ("mul", 1, 3) => mul31(y, x.a),
        ("mul", _, _) => mul33(x, y),
        _ => panic!("unexpected op {op}"),
    }
}

/// x^n by square-and-multiply.
pub fn pow3(x: F3, n: u64) -> F3 {
    let mut r = F3::base(1);
    let mut b = x;
    let mut n = n;
    while n > 0 {
        if n & 1 == 1 {
            r = mul33(r, b);
        }
        b = mul33(b, b);
        n >>= 1;
    }
    r
}

/// Tiny deterministic PRNG (splitmix64) for the equivalence check.
pub struct Rng(pub u64);
impl Rng {
    pub fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    pub fn field(&mut self) -> u64 {
        self.next() % P
    }
    pub fn f3(&mut self, dim: u64) -> F3 {
        if dim == 1 {
            F3::base(self.field())
        } else {
            F3 { a: self.field(), b: self.field(), c: self.field() }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mul33_matches_schoolbook_reduction() {
        let mut r = Rng(7);
        for _ in 0..1000 {
            let x = r.f3(3);
            let y = r.f3(3);
            // schoolbook with x^3 = x + 1, x^4 = x^2 + x
            let c0 = mul(x.a, y.a);
            let c1 = add(mul(x.a, y.b), mul(x.b, y.a));
            let c2 = add(add(mul(x.a, y.c), mul(x.b, y.b)), mul(x.c, y.a));
            let c3 = add(mul(x.b, y.c), mul(x.c, y.b));
            let c4 = mul(x.c, y.c);
            let want = F3 { a: add(c0, c3), b: add(add(c1, c3), c4), c: add(c2, c4) };
            assert_eq!(mul33(x, y), want);
        }
    }

    #[test]
    fn add_sub_roundtrip() {
        let mut r = Rng(3);
        for _ in 0..1000 {
            let a = r.field();
            let b = r.field();
            assert_eq!(sub(add(a, b), b), a);
            assert_eq!(add(sub(a, b), b), a);
        }
        assert_eq!(add(P - 1, 1), 0);
        assert_eq!(sub(0, 1), P - 1);
    }
}
