//! Goldilocks field arithmetic helpers for pil2circom code generation.
//!
//! The JS `pil2circom.js` uses an `F3g` object to compute constant integer literals
//! that get embedded verbatim into circom source. These functions replicate that
//! arithmetic using the `fields::Goldilocks` type — no Node.js needed.
//!
//! All functions operate on `u64` canonical representatives of GF(p) elements
//! and return `u64` canonical results ready to format into circom code.

use proofman_fields::PrimeField64;
use proofman_fields::Goldilocks;

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
    fn inv_w_matches() {
        for i in 0..33 {
            assert_eq!(
                (Goldilocks::new(GL_W[i]) * Goldilocks::new(GL_INV_W[i])).as_canonical_u64(),
                1,
                "W[{i}] * INV_W[{i}] != 1"
            );
        }
        assert_eq!(gl_exp(7, 2), 49);
    }
}
