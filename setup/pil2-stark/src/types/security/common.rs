use rug::{Assign, Float, Integer, ops::Pow};

/// Precision for all high-precision arithmetic.
/// 200 decimal digits ~= 665 binary bits; use 700 for margin.
const PREC: u32 = 700;

/// Construct a high-precision Integer from an integer-like value.
pub(crate) fn api<T>(v: T) -> Integer
where
    Integer: From<T>,
{
    Integer::from(v)
}

/// Construct a high-precision Float from an integer-like value.
pub(crate) fn hpf<T>(v: T) -> Float
where
    Float: Assign<T>,
{
    Float::with_val(PREC, v)
}

/// Return `floor(-log2(error))`.
pub(crate) fn security_bits_from_error(error: &Float) -> i64 {
    let bits = -hpf(error.log2_ref());
    bits.floor().to_f64() as i64
}

/// Apply `bits` of grinding (proof-of-work): scale the error by `2^-bits`.
pub(crate) fn apply_grinding(error: &Float, bits: u32) -> Float {
    let two_pow = hpf(hpf(2).pow(bits));
    error / two_pow
}

/// Floor a `Float` to `n` decimal places.
///
/// Note: for negative numbers, this rounds toward negative infinity.
pub(crate) fn truncate_decimal_places(val: &Float, n: u32) -> Float {
    let scale = hpf(Integer::from(10).pow(n));
    let scaled = Float::with_val(PREC, val * &scale);

    scaled.floor() / scale
}

/// Return `|F|` for the Goldilocks cubic extension field.
pub fn goldilocks_safe_extension_field_size() -> Integer {
    let p: Integer = (api(1) << 64) - (api(1) << 32) + 1;
    p.pow(3)
}

/// `log2 |F|` for the Goldilocks cubic extension.
pub fn goldilocks_safe_extension_field_size_bits() -> Float {
    hpf(hpf(&goldilocks_safe_extension_field_size()).log2_ref())
}

/// Reconstruct `|F|` from its base-2 logarithm `field_size_bits`.
pub(crate) fn field_size_from_bits(field_size_bits: &Float) -> Float {
    hpf(hpf(2).pow(field_size_bits))
}
