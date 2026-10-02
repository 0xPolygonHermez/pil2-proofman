use rand::{rngs::StdRng, RngExt, SeedableRng};

use super::sha2_constants::{LANES, NUM_ROUNDS, RC, SLOT_BITS};

/// Pseudo-random SHA2-256 input generator
pub(crate) fn random_sha2_input(seed: u64) -> ([u32; 8], [u32; 16]) {
    let mut rng = StdRng::seed_from_u64(seed);
    let state = core::array::from_fn(|_| rng.random());
    let input = core::array::from_fn(|_| rng.random());
    (state, input)
}

/// Bit i of w
#[inline]
pub(crate) fn bit(w: u32, i: usize) -> u64 {
    ((w >> i) & 1) as u64
}

/// The same bit of every lane, packed at the slot spacing
#[inline]
pub(crate) fn slice(words: &[u32; LANES], i: usize) -> u64 {
    (0..LANES).map(|k| bit(words[k], i) << (SLOT_BITS * k)).sum()
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

/// One SHA2-256 compression: a_t, e_t and W_t of every round, and the digest
pub(crate) fn compress(
    state: &[u32; 8],
    block: &[u32; 16],
) -> ([u32; NUM_ROUNDS], [u32; NUM_ROUNDS], [u32; NUM_ROUNDS], [u32; 8]) {
    let mut w = [0u32; NUM_ROUNDS];
    w[..16].copy_from_slice(block);
    for t in 16..NUM_ROUNDS {
        w[t] =
            small_sigma1(w[t - 2]).wrapping_add(w[t - 7]).wrapping_add(small_sigma0(w[t - 15])).wrapping_add(w[t - 16]);
    }
    let [mut a, mut b, mut c, mut d, mut e, mut f, mut g, mut h] = *state;
    let (mut av, mut ev) = ([0u32; NUM_ROUNDS], [0u32; NUM_ROUNDS]);
    for t in 0..NUM_ROUNDS {
        let t1 = h.wrapping_add(big_sigma1(e)).wrapping_add(ch(e, f, g)).wrapping_add(RC[t]).wrapping_add(w[t]);
        let t2 = big_sigma0(a).wrapping_add(maj(a, b, c));
        h = g;
        g = f;
        f = e;
        e = d.wrapping_add(t1);
        d = c;
        c = b;
        b = a;
        a = t1.wrapping_add(t2);
        av[t] = a;
        ev[t] = e;
    }
    let out = [a, b, c, d, e, f, g, h];
    let digest = core::array::from_fn(|i| state[i].wrapping_add(out[i]));
    (av, ev, w, digest)
}

/// FIPS 180-4 test vector: SHA-256("abc"), one padded block from the IV
pub(crate) fn self_test() {
    const IV: [u32; 8] =
        [0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19];
    let mut block = [0u32; 16];
    block[0] = 0x61626380;
    block[15] = 24;
    let (_, _, _, digest) = compress(&IV, &block);
    assert_eq!(
        digest,
        [0xba7816bf, 0x8f01cfea, 0x414140de, 0x5dae2223, 0xb00361a3, 0x96177a9c, 0xb410ff61, 0xf20015ad],
        "SHA2-256 compression self-test"
    );
}
