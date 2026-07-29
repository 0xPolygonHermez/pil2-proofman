use rand::{rngs::StdRng, RngExt, SeedableRng};

/// Pseudo-random SHA2-256 input generator
pub(crate) fn random_sha2_input(seed: u64) -> ([u32; 8], [u32; 16]) {
    let mut rng = StdRng::seed_from_u64(seed);
    let state = core::array::from_fn(|_| rng.random());
    let input = core::array::from_fn(|_| rng.random());
    (state, input)
}

/// Decompose a 32-bit word into its bits, LSB first
#[inline]
pub(crate) fn bits32(w: u32) -> [bool; 32] {
    core::array::from_fn(|i| (w >> i) & 1 == 1)
}

/// Row index of the (3, 3, 2)-bit range-checker triple
#[inline]
pub(crate) fn range_row(s0_carry: u8, s1_carry: u8, w_carry: u8) -> usize {
    debug_assert!(s0_carry < 8 && s1_carry < 8 && w_carry < 4);
    (s0_carry as usize) + 8 * (s1_carry as usize) + 64 * (w_carry as usize)
}

/// σ₀(w) = (w >>> 7) ^ (w >>> 18) ^ (w >> 3)
#[inline]
pub(crate) fn small_sigma0(w: u32) -> u32 {
    w.rotate_right(7) ^ w.rotate_right(18) ^ (w >> 3)
}

/// σ₁(w) = (w >>> 17) ^ (w >>> 19) ^ (w >> 10)
#[inline]
pub(crate) fn small_sigma1(w: u32) -> u32 {
    w.rotate_right(17) ^ w.rotate_right(19) ^ (w >> 10)
}

/// Σ₀(a) = (a >>> 2) ^ (a >>> 13) ^ (a >>> 22)
#[inline]
pub(crate) fn big_sigma0(a: u32) -> u32 {
    a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22)
}

/// Σ₁(e) = (e >>> 6) ^ (e >>> 11) ^ (e >>> 25)
#[inline]
pub(crate) fn big_sigma1(e: u32) -> u32 {
    e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25)
}

/// ch(e, f, g) = (e & f) ^ (!e & g)
#[inline]
pub(crate) fn ch(e: u32, f: u32, g: u32) -> u32 {
    (e & f) ^ (!e & g)
}

/// maj(a, b, c) = (a & b) ^ (a & c) ^ (b & c)
#[inline]
pub(crate) fn maj(a: u32, b: u32, c: u32) -> u32 {
    (a & b) ^ (a & c) ^ (b & c)
}
