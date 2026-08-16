use rand::{rngs::StdRng, RngExt, SeedableRng};

/// Pseudo-random Blake2b input generator
pub(crate) fn random_blake2b_input(seed: u64) -> ([u64; 16], [u64; 16]) {
    let mut rng = StdRng::seed_from_u64(seed);
    let state = core::array::from_fn(|_| rng.random());
    let message = core::array::from_fn(|_| rng.random());
    (state, message)
}

/// Split a 64-bit word into four little-endian 16-bit limbs
#[inline]
pub(crate) fn limbs16(w: u64) -> [u16; 4] {
    core::array::from_fn(|i| ((w >> (16 * i)) & 0xffff) as u16)
}

/// Row index of a range-checker
#[inline]
pub(crate) fn range_row(v: u16) -> usize {
    v as usize
}

/// Row index of an XOR table pair (a, b)
#[inline]
pub(crate) fn table_row(a: u8, b: u8) -> usize {
    (b as usize) * 256 + a as usize
}
