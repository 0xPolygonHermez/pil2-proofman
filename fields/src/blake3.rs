//! BLAKE3-based 16-wide Goldilocks permutation for the global (contributions)
//! transcript. Self-contained Rust port of the same construction used by the
//! C++/CUDA `blake3_core` (canonical-reduced XOF output), so the global
//! challenge derivation is deterministic across the prover/verifier.

use crate::{Hash, PrimeField64};

const IV: [u32; 8] = [0x6A09E667, 0xBB67AE85, 0x3C6EF372, 0xA54FF53A, 0x510E527F, 0x9B05688C, 0x1F83D9AB, 0x5BE0CD19];
const MSG: [[usize; 16]; 7] = [
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
    [2, 6, 3, 10, 7, 0, 4, 13, 1, 11, 12, 5, 9, 14, 15, 8],
    [3, 4, 10, 12, 13, 2, 7, 14, 6, 5, 9, 0, 11, 15, 8, 1],
    [10, 7, 12, 9, 14, 3, 13, 15, 4, 0, 11, 2, 5, 8, 1, 6],
    [12, 13, 9, 11, 15, 10, 14, 8, 7, 2, 5, 3, 0, 1, 6, 4],
    [9, 14, 11, 5, 8, 12, 15, 1, 13, 3, 0, 10, 2, 6, 4, 7],
    [11, 15, 5, 0, 1, 9, 8, 6, 14, 10, 2, 12, 3, 4, 7, 13],
];
const CHUNK_START: u32 = 1;
const CHUNK_END: u32 = 2;
const ROOT: u32 = 8;
const GL_P: u64 = 0xFFFFFFFF00000001;

#[inline]
fn g(s: &mut [u32; 16], a: usize, b: usize, c: usize, d: usize, mx: u32, my: u32) {
    s[a] = s[a].wrapping_add(s[b]).wrapping_add(mx);
    s[d] = (s[d] ^ s[a]).rotate_right(16);
    s[c] = s[c].wrapping_add(s[d]);
    s[b] = (s[b] ^ s[c]).rotate_right(12);
    s[a] = s[a].wrapping_add(s[b]).wrapping_add(my);
    s[d] = (s[d] ^ s[a]).rotate_right(8);
    s[c] = s[c].wrapping_add(s[d]);
    s[b] = (s[b] ^ s[c]).rotate_right(7);
}

#[inline]
fn round(s: &mut [u32; 16], m: &[u32; 16], r: usize) {
    let x = &MSG[r];
    g(s, 0, 4, 8, 12, m[x[0]], m[x[1]]);
    g(s, 1, 5, 9, 13, m[x[2]], m[x[3]]);
    g(s, 2, 6, 10, 14, m[x[4]], m[x[5]]);
    g(s, 3, 7, 11, 15, m[x[6]], m[x[7]]);
    g(s, 0, 5, 10, 15, m[x[8]], m[x[9]]);
    g(s, 1, 6, 11, 12, m[x[10]], m[x[11]]);
    g(s, 2, 7, 8, 13, m[x[12]], m[x[13]]);
    g(s, 3, 4, 9, 14, m[x[14]], m[x[15]]);
}

fn compress_pre(cv: &[u32; 8], block: &[u32; 16], blen: u32, counter: u64, flags: u32) -> [u32; 16] {
    let mut s = [0u32; 16];
    s[..8].copy_from_slice(cv);
    s[8] = IV[0];
    s[9] = IV[1];
    s[10] = IV[2];
    s[11] = IV[3];
    s[12] = counter as u32;
    s[13] = (counter >> 32) as u32;
    s[14] = blen;
    s[15] = flags;
    for r in 0..7 {
        round(&mut s, block, r);
    }
    s
}

fn compress_xof(cv: &[u32; 8], block: &[u32; 16], blen: u32, counter: u64, flags: u32) -> [u32; 16] {
    let s = compress_pre(cv, block, blen, counter, flags);
    let mut o = [0u32; 16];
    for i in 0..8 {
        o[i] = s[i] ^ s[i + 8];
        o[i + 8] = s[i + 8] ^ cv[i];
    }
    o
}

#[inline]
fn canon(x: u64) -> u64 {
    if x >= GL_P {
        x - GL_P
    } else {
        x
    }
}

/// 8-element Goldilocks BLAKE3 permutation: the 8 little-endian u64 words are a
/// 64-byte (single-block) input; the 64-byte XOF output gives 8 canonical field
/// elements. This mirrors the C++ `TranscriptGL` arity-2 permutation
/// (`blake3core::permute_xof(in, 8, out)`), so the width-8 (arity-2) global/std
/// challenge derivation is bit-identical across prover and verifier.
pub fn blake3_permute8<F: PrimeField64>(state: &mut [F; 8]) {
    let mut w = [0u64; 8];
    for i in 0..8 {
        w[i] = state[i].as_canonical_u64();
    }
    let mut b0 = [0u32; 16];
    for k in 0..8 {
        b0[2 * k] = w[k] as u32;
        b0[2 * k + 1] = (w[k] >> 32) as u32;
    }
    // Single block: CHUNK_START|CHUNK_END|ROOT, output counter 0 (one 64-byte block).
    let o0 = compress_xof(&IV, &b0, 64, 0, CHUNK_START | CHUNK_END | ROOT);
    for k in 0..8 {
        let v0 = (o0[2 * k] as u64) | ((o0[2 * k + 1] as u64) << 32);
        state[k] = F::from_u64(canon(v0));
    }
}

/// 8-wide BLAKE3 hash marker for `Transcript<F, Blake3_8>` (arity-2 transcript).
pub struct Blake3_8;

impl<F: PrimeField64> Hash<F> for Blake3_8 {
    const WIDTH: usize = 8;
    const RATE: usize = 4;
    const CAPACITY: usize = 4;
    type State = [F; 8];
    fn hash(state: &mut [F; 8]) {
        blake3_permute8::<F>(state);
    }
}
