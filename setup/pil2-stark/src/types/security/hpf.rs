//! High-precision floating-point primitives shared across the security calculator.
//!
//! All soundness-error arithmetic runs at `PREC` bits to match the JS reference
//! implementation's `Decimal.js` precision (200 decimal digits). These helpers are
//! generic float plumbing — they carry no FRI/STIR domain meaning.

use std::ops::Neg;

use rug::ops::Pow;
use rug::Float;

/// Precision for all high-precision arithmetic (matches Decimal.js precision: 200 digits).
/// 200 decimal digits ~ 665 binary bits; use 700 for safety.
pub(crate) const PREC: u32 = 700;

/// Construct a high-precision Float from an integer-like value.
pub(crate) fn hpf(v: u64) -> Float {
    Float::with_val(PREC, v)
}

pub(crate) fn hpf_from_f64(v: f64) -> Float {
    Float::with_val(PREC, v)
}

/// `floor(-log2(error))` -- equivalent to JS `get_security_from_error`.
pub(crate) fn security_bits_from_error(error: &Float) -> i64 {
    let log2_val = Float::with_val(PREC, error.log2_ref());
    let neg_log2 = log2_val.neg();
    let floored = neg_log2.floor();
    floored.to_f64() as i64
}

/// Truncate a Float to `n` decimal places (round-down), matching
/// `Decimal.toDecimalPlaces(n, ROUND_DOWN)`.
pub(crate) fn truncate_decimal_places(val: &Float, n: u32) -> Float {
    // Multiply by 10^n, floor, then divide by 10^n.
    let scale = Float::with_val(PREC, Float::i_pow_u(10, n));
    let scaled = Float::with_val(PREC, val * &scale);
    let floored = scaled.floor();
    Float::with_val(PREC, floored / scale)
}

/// Goldilocks cubic-extension field size `(2^64 - 2^32 + 1)^3` (~2^191), the
/// field over which soundness errors are measured. Public: setup callers pass it
/// into the FRI/STIR parameter calculators.
pub fn goldilocks_cube_field_size() -> Float {
    // (2^64 - 2^32 + 1)^3
    let two_64 = Float::with_val(PREC, Float::i_pow_u(2, 64));
    let two_32 = Float::with_val(PREC, Float::i_pow_u(2, 32));
    let p = Float::with_val(PREC, &two_64 - &two_32) + hpf(1);
    Float::with_val(PREC, p.pow(3))
}

#[cfg(test)]
mod tests {
    use super::*;
    use rug::float::Round;

    /// Verify that the field-size helper produces ~2^191 (matches JS fieldSize).
    #[test]
    fn test_goldilocks_cube_field_size() {
        let fs = goldilocks_cube_field_size();
        let log2_fs = Float::with_val(PREC, fs.log2_ref()).to_f64_round(Round::Nearest);
        // JS reports "Field Size: 2^191"
        assert!((191.0..192.0).contains(&log2_fs), "log2(fieldSize) should be ~191, got {log2_fs}");
    }
}
