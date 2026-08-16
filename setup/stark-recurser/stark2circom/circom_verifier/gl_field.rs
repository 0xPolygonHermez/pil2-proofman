//! Goldilocks field arithmetic helpers for pil2circom code generation.
//!
//! The JS `pil2circom.js` uses an `F3g` object to compute constant integer literals
//! that get embedded verbatim into circom source. These functions replicate that
//! arithmetic using the `fields::Goldilocks` type — no Node.js needed.
//!
//! All functions operate on `u64` canonical representatives of GF(p) elements
//! and return `u64` canonical results ready to format into circom code.

use proofman_fields::{Field, PrimeField64};
use proofman_fields::Goldilocks;

/// The coset shift used by the STARK: `F.shift = 7`.
pub const GL_SHIFT: u64 = Goldilocks::SHIFT;

/// The table of 2^n-th roots of unity for n = 0..=32: `F.w[n]`.
///
/// `GL_W[n]` is a primitive 2^n-th root of unity in GF(p).
/// Matches the `W` array in `Goldilocks` and the EJS `roots()` / `F.w[n]` lookups.
pub const GL_W: [u64; 33] = Goldilocks::W;

/// The table of inverse (2^n-th roots of unity) for n = 0..=32: `invroots(n)`.
///
/// `GL_INV_W[n]` = GL_W[n]⁻¹.  Pre-computed constants matching the EJS `invroots()` table.
pub const GL_INV_W: [u64; 33] = [
    1,
    18446744069414584320,
    18446462594437873665,
    18446742969902956801,
    18442240469788262401,
    18158513693329981441,
    16140901060737761281,
    274873712576,
    9171943329124577373,
    5464760906092500108,
    4088309022520035137,
    6141391951880571024,
    386651765402340522,
    11575992183625933494,
    2841727033376697931,
    8892493137794983311,
    9071788333329385449,
    15139302138664925958,
    14996013474702747840,
    5708508531096855759,
    6451340039662992847,
    5102364342718059185,
    10420286214021487819,
    13945510089405579673,
    17538441494603169704,
    16784649996768716373,
    8974194941257008806,
    16194875529212099076,
    5506647088734794298,
    7731871677141058814,
    16558868196663692994,
    9896756522253134970,
    1644488454024429189,
];

/// GL field multiplication: `a * b mod p`.
#[inline]
pub fn gl_mul(a: u64, b: u64) -> u64 {
    (Goldilocks::new(a) * Goldilocks::new(b)).as_canonical_u64()
}

/// GL field inversion: `a⁻¹ mod p`. Panics if `a == 0`.
#[inline]
pub fn gl_inv(a: u64) -> u64 {
    Goldilocks::new(a).inverse().as_canonical_u64()
}

/// GL field exponentiation: `a^e mod p`.
pub fn gl_exp(a: u64, e: u64) -> u64 {
    let mut result = Goldilocks::new(1);
    let mut base = Goldilocks::new(a);
    let mut exp = e;
    while exp > 0 {
        if exp & 1 == 1 {
            result *= base;
        }
        base *= base;
        exp >>= 1;
    }
    result.as_canonical_u64()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_shift_is_seven() {
        assert_eq!(GL_SHIFT, 7);
    }

    #[test]
    fn test_w_roots_unity() {
        // W[1] should be a square root of W[0]=1, i.e. W[1]^2 == 1 (it's -1 mod p).
        assert_eq!(GL_W[0], 1);
        assert_eq!(gl_mul(GL_W[1], GL_W[1]), 1);
        // Each root is half the period of the previous.
        assert_eq!(gl_mul(GL_W[2], GL_W[2]), GL_W[1]);
    }

    #[test]
    fn test_inv_w_matches() {
        for i in 0..33 {
            assert_eq!(gl_mul(GL_W[i], GL_INV_W[i]), 1, "W[{i}] * INV_W[{i}] != 1");
        }
    }

    #[test]
    fn test_gl_exp() {
        // 7^0 = 1
        assert_eq!(gl_exp(7, 0), 1);
        // 7^1 = 7
        assert_eq!(gl_exp(7, 1), 7);
        // 7^2 = 49
        assert_eq!(gl_exp(7, 2), 49);
    }

    #[test]
    fn test_gl_inv() {
        let a = 12345678u64;
        assert_eq!(gl_mul(a, gl_inv(a)), 1);
    }
}
