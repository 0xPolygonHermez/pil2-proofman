//! BLAKE3 over a swappable permutation, so the ZisK guest reaches the `blake3f` precompile.
//!
//! A drop-in for the three `blake3::Hasher` methods `Blake3Transcript` uses, so switching backends
//! is a `use`. The crate stays the implementation everywhere else and the test oracle here.

use core::mem::MaybeUninit;

const BLOCK_LEN: usize = 64;
const CHUNK_LEN: usize = 1024;

const CHUNK_START: u32 = 1 << 0;
const CHUNK_END: u32 = 1 << 1;
const PARENT: u32 = 1 << 2;
const ROOT: u32 = 1 << 3;

const IV: [u32; 8] = [0x6A09E667, 0xBB67AE85, 0x3C6EF372, 0xA54FF53A, 0x510E527F, 0x9B05688C, 0x1F83D9AB, 0x5BE0CD19];

/// Stack depth is `popcount(chunks)`, so this bounds it at 2^54 chunks (the reference's figure).
const CV_STACK_LEN: usize = 54;

/// ZisK's `SyscallBlake3fParams`. It reads both `[u64; 8]` as `[u32; 16]`, so `pack` is a
/// little-endian view rather than a conversion.
#[cfg(all(target_os = "zkvm", target_vendor = "zisk"))]
#[repr(C)]
struct SyscallBlake3fParams {
    state: *mut [u64; 8],
    input: *const [u64; 8],
}

#[cfg(all(target_os = "zkvm", target_vendor = "zisk"))]
extern "C" {
    fn syscall_blake3f(params: *mut SyscallBlake3fParams);
}

#[cfg(all(target_os = "zkvm", target_vendor = "zisk"))]
#[inline]
fn pack(words: &[u32; 16]) -> [u64; 8] {
    core::array::from_fn(|i| (words[2 * i] as u64) | ((words[2 * i + 1] as u64) << 32))
}

/// The seven rounds alone: `blake3_f` is the permutation the precompile proves, so state init and
/// feedforward stay with the caller.
#[inline]
fn permute7(v: &mut [u32; 16], m: &[u32; 16]) {
    #[cfg(all(target_os = "zkvm", target_vendor = "zisk"))]
    {
        // Packed into owned buffers, not cast in place: `[u32; 16]` is 4-aligned, `[u64; 8]` is 8.
        let mut state = pack(v);
        let input = pack(m);
        let mut params = SyscallBlake3fParams { state: &mut state, input: &input };
        unsafe { syscall_blake3f(&mut params) };
        for i in 0..8 {
            v[2 * i] = state[i] as u32;
            v[2 * i + 1] = (state[i] >> 32) as u32;
        }
    }
    #[cfg(not(all(target_os = "zkvm", target_vendor = "zisk")))]
    {
        /// Verbatim from `blake3_core.hpp` and ZisK's `round.rs`, so the two paths cannot diverge.
        const SIGMA: [[usize; 16]; 7] = [
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
            [2, 6, 3, 10, 7, 0, 4, 13, 1, 11, 12, 5, 9, 14, 15, 8],
            [3, 4, 10, 12, 13, 2, 7, 14, 6, 5, 9, 0, 11, 15, 8, 1],
            [10, 7, 12, 9, 14, 3, 13, 15, 4, 0, 11, 2, 5, 8, 1, 6],
            [12, 13, 9, 11, 15, 10, 14, 8, 7, 2, 5, 3, 0, 1, 6, 4],
            [9, 14, 11, 5, 8, 12, 15, 1, 13, 3, 0, 10, 2, 6, 4, 7],
            [11, 15, 5, 0, 1, 9, 8, 6, 14, 10, 2, 12, 3, 4, 7, 13],
        ];

        #[inline]
        #[allow(clippy::too_many_arguments)]
        fn g(v: &mut [u32; 16], a: usize, b: usize, c: usize, d: usize, x: u32, y: u32) {
            v[a] = v[a].wrapping_add(v[b]).wrapping_add(x);
            v[d] = (v[d] ^ v[a]).rotate_right(16);
            v[c] = v[c].wrapping_add(v[d]);
            v[b] = (v[b] ^ v[c]).rotate_right(12);
            v[a] = v[a].wrapping_add(v[b]).wrapping_add(y);
            v[d] = (v[d] ^ v[a]).rotate_right(8);
            v[c] = v[c].wrapping_add(v[d]);
            v[b] = (v[b] ^ v[c]).rotate_right(7);
        }

        for s in SIGMA.iter() {
            g(v, 0, 4, 8, 12, m[s[0]], m[s[1]]);
            g(v, 1, 5, 9, 13, m[s[2]], m[s[3]]);
            g(v, 2, 6, 10, 14, m[s[4]], m[s[5]]);
            g(v, 3, 7, 11, 15, m[s[6]], m[s[7]]);
            g(v, 0, 5, 10, 15, m[s[8]], m[s[9]]);
            g(v, 1, 6, 11, 12, m[s[10]], m[s[11]]);
            g(v, 2, 7, 8, 13, m[s[12]], m[s[13]]);
            g(v, 3, 4, 9, 14, m[s[14]], m[s[15]]);
        }
    }
}

/// `[0..8]` is the chaining value; all 16 are the XOF block.
#[inline]
fn compress(cv: &[u32; 8], block: &[u32; 16], counter: u64, block_len: u32, flags: u32) -> [u32; 16] {
    let mut v = [
        cv[0],
        cv[1],
        cv[2],
        cv[3],
        cv[4],
        cv[5],
        cv[6],
        cv[7],
        IV[0],
        IV[1],
        IV[2],
        IV[3],
        counter as u32,
        (counter >> 32) as u32,
        block_len,
        flags,
    ];
    permute7(&mut v, block);
    for i in 0..8 {
        v[i] ^= v[i + 8];
        v[i + 8] ^= cv[i];
    }
    v
}

#[inline]
fn first8(v: &[u32; 16]) -> [u32; 8] {
    core::array::from_fn(|i| v[i])
}

#[inline]
fn words_from_le_bytes(bytes: &[u8; BLOCK_LEN]) -> [u32; 16] {
    core::array::from_fn(|i| u32::from_le_bytes([bytes[4 * i], bytes[4 * i + 1], bytes[4 * i + 2], bytes[4 * i + 3]]))
}

/// Kept unapplied so the root can be squeezed at any output block.
#[derive(Clone, Copy)]
struct Output {
    input_cv: [u32; 8],
    block_words: [u32; 16],
    counter: u64,
    block_len: u32,
    flags: u32,
}

impl Output {
    fn chaining_value(&self) -> [u32; 8] {
        first8(&compress(&self.input_cv, &self.block_words, self.counter, self.block_len, self.flags))
    }

    /// The root compression re-run with `block` as the output-block counter.
    fn root_block(&self, block: u64) -> [u8; BLOCK_LEN] {
        let words = compress(&self.input_cv, &self.block_words, block, self.block_len, self.flags | ROOT);
        let mut out = [0u8; BLOCK_LEN];
        for (i, w) in words.iter().enumerate() {
            out[4 * i..4 * i + 4].copy_from_slice(&w.to_le_bytes());
        }
        out
    }
}

#[derive(Clone, Copy)]
struct ChunkState {
    cv: [u32; 8],
    counter: u64,
    block: [u8; BLOCK_LEN],
    block_len: u8,
    blocks_compressed: u8,
}

impl ChunkState {
    fn new(cv: [u32; 8], counter: u64) -> Self {
        ChunkState { cv, counter, block: [0; BLOCK_LEN], block_len: 0, blocks_compressed: 0 }
    }

    fn len(&self) -> usize {
        BLOCK_LEN * self.blocks_compressed as usize + self.block_len as usize
    }

    /// CHUNK_START rides only the chunk's first block.
    fn start_flag(&self) -> u32 {
        if self.blocks_compressed == 0 {
            CHUNK_START
        } else {
            0
        }
    }

    fn update(&mut self, mut input: &[u8]) {
        while !input.is_empty() {
            // Deferred: the chunk's last block carries CHUNK_END, unknowable until more arrives.
            if self.block_len as usize == BLOCK_LEN {
                let words = words_from_le_bytes(&self.block);
                self.cv = first8(&compress(&self.cv, &words, self.counter, BLOCK_LEN as u32, self.start_flag()));
                self.blocks_compressed += 1;
                self.block = [0; BLOCK_LEN];
                self.block_len = 0;
            }
            let want = BLOCK_LEN - self.block_len as usize;
            let take = want.min(input.len());
            self.block[self.block_len as usize..self.block_len as usize + take].copy_from_slice(&input[..take]);
            self.block_len += take as u8;
            input = &input[take..];
        }
    }

    fn output(&self) -> Output {
        Output {
            input_cv: self.cv,
            block_words: words_from_le_bytes(&self.block),
            counter: self.counter,
            block_len: self.block_len as u32,
            flags: self.start_flag() | CHUNK_END,
        }
    }
}

fn parent_output(left: [u32; 8], right: [u32; 8]) -> Output {
    let mut block_words = [0u32; 16];
    block_words[..8].copy_from_slice(&left);
    block_words[8..].copy_from_slice(&right);
    Output { input_cv: IV, block_words, counter: 0, block_len: BLOCK_LEN as u32, flags: PARENT }
}

/// Drop-in for `blake3::Hasher` at the surface `Blake3Transcript` uses.
#[derive(Clone)]
pub struct Hasher {
    chunk: ChunkState,
    /// Uninitialised, not zeroed: zeroing 1.7 KB costs the guest more than the node hash it
    /// precedes, and nothing at or above `cv_stack_len` is ever read.
    cv_stack: [MaybeUninit<[u32; 8]>; CV_STACK_LEN],
    cv_stack_len: u8,
}

impl Default for Hasher {
    fn default() -> Self {
        Self::new()
    }
}

impl Hasher {
    pub fn new() -> Self {
        Hasher { chunk: ChunkState::new(IV, 0), cv_stack: [MaybeUninit::uninit(); CV_STACK_LEN], cv_stack_len: 0 }
    }

    /// The entry at `cv_stack_len`, which callers have just decremented past.
    #[inline]
    fn pop(&self) -> [u32; 8] {
        // SAFETY: `push_cv` writes before incrementing, so everything below the len is initialised.
        unsafe { self.cv_stack[self.cv_stack_len as usize].assume_init() }
    }

    fn push_cv(&mut self, mut cv: [u32; 8], mut total_chunks: u64) {
        // A trailing zero bit means the subtree to the left is full, so the two are siblings.
        while total_chunks & 1 == 0 {
            self.cv_stack_len -= 1;
            cv = parent_output(self.pop(), cv).chaining_value();
            total_chunks >>= 1;
        }
        self.cv_stack[self.cv_stack_len as usize] = MaybeUninit::new(cv);
        self.cv_stack_len += 1;
    }

    pub fn update(&mut self, mut input: &[u8]) {
        while !input.is_empty() {
            if self.chunk.len() == CHUNK_LEN {
                let cv = self.chunk.output().chaining_value();
                let total = self.chunk.counter + 1;
                self.push_cv(cv, total);
                self.chunk = ChunkState::new(IV, total);
            }
            let want = CHUNK_LEN - self.chunk.len();
            let take = want.min(input.len());
            self.chunk.update(&input[..take]);
            input = &input[take..];
        }
    }

    /// `&self` like the crate's: the transcript finalizes repeatedly and absorbs afterwards.
    pub fn finalize_xof(&self) -> OutputReader {
        let mut output = self.chunk.output();
        for i in (0..self.cv_stack_len as usize).rev() {
            // SAFETY: i < cv_stack_len, so a push wrote this entry.
            output = parent_output(unsafe { self.cv_stack[i].assume_init() }, output.chaining_value());
        }
        OutputReader { output, position: 0 }
    }
}

/// Drop-in for `blake3::OutputReader`.
pub struct OutputReader {
    output: Output,
    position: u64,
}

impl OutputReader {
    pub fn set_position(&mut self, position: u64) {
        self.position = position;
    }

    pub fn fill(&mut self, buf: &mut [u8]) {
        let mut written = 0;
        while written < buf.len() {
            let block = self.output.root_block(self.position / BLOCK_LEN as u64);
            let offset = (self.position % BLOCK_LEN as u64) as usize;
            let take = (BLOCK_LEN - offset).min(buf.len() - written);
            buf[written..written + take].copy_from_slice(&block[offset..offset + take]);
            written += take;
            self.position += take as u64;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec::Vec;

    fn reference(input: &[u8], position: u64, len: usize) -> Vec<u8> {
        let mut h = ::blake3::Hasher::new();
        h.update(input);
        let mut r = h.finalize_xof();
        r.set_position(position);
        let mut out = alloc::vec![0u8; len];
        r.fill(&mut out);
        out
    }

    fn ours(input: &[u8], position: u64, len: usize) -> Vec<u8> {
        let mut h = Hasher::new();
        h.update(input);
        let mut r = h.finalize_xof();
        r.set_position(position);
        let mut out = alloc::vec![0u8; len];
        r.fill(&mut out);
        out
    }

    fn bytes(n: usize) -> Vec<u8> {
        (0..n).map(|i| (i.wrapping_mul(0x9E) ^ (i >> 5)) as u8).collect()
    }

    /// Every length that changes branch: within a block, the block and chunk boundaries, and an
    /// odd chunk count (an unbalanced merge).
    #[test]
    fn blake3_core_matches_the_reference_crate() {
        for n in
            [0, 1, 31, 32, 63, 64, 65, 127, 128, 200, 1023, 1024, 1025, 2047, 2048, 2049, 3072, 3073, 5000, 8192, 8193]
        {
            let input = bytes(n);
            assert_eq!(ours(&input, 0, 32), reference(&input, 0, 32), "digest differs at {n} bytes");
        }
    }

    /// A construction can be right at 32 bytes and wrong from output block 1 on, which is where
    /// the transcript's later challenges come from.
    #[test]
    fn the_xof_matches_the_reference_past_the_first_block() {
        let input = bytes(300);
        assert_eq!(ours(&input, 0, 512), reference(&input, 0, 512), "long XOF read differs");
        for block in [0u64, 1, 2, 7, 64] {
            assert_eq!(ours(&input, block * 64, 64), reference(&input, block * 64, 64), "XOF block {block} differs");
        }
    }

    /// `fill` is position-driven, not block-driven: `get_state` reads 32 bytes, the lattice chain 64.
    #[test]
    fn split_and_unaligned_reads_agree_with_one_long_read() {
        let input = bytes(1500);
        let whole = ours(&input, 0, 256);

        let mut h = Hasher::new();
        h.update(&input);
        let mut r = h.finalize_xof();
        let mut piecewise = Vec::new();
        for len in [7usize, 25, 32, 64, 1, 127] {
            let mut part = alloc::vec![0u8; len];
            r.fill(&mut part);
            piecewise.extend_from_slice(&part);
        }
        assert_eq!(&whole[..piecewise.len()], &piecewise[..], "split reads must match one long read");

        assert_eq!(ours(&input, 13, 100), reference(&input, 13, 100), "unaligned read differs");
    }

    /// The transcript absorbs one value per `put`, so the buffering must not care how it is cut.
    #[test]
    fn incremental_updates_match_a_single_update() {
        let input = bytes(4096);
        let one = ours(&input, 0, 64);
        for step in [1usize, 7, 63, 64, 65, 1023, 1024] {
            let mut h = Hasher::new();
            for piece in input.chunks(step) {
                h.update(piece);
            }
            let mut r = h.finalize_xof();
            let mut out = [0u8; 64];
            r.fill(&mut out);
            assert_eq!(&out[..], &one[..], "absorbing in {step}-byte pieces changed the digest");
        }
    }

    /// Pins `finalize_xof(&self)` against an "optimisation" to `&mut self`.
    #[test]
    fn finalizing_does_not_disturb_the_hasher() {
        let input = bytes(2000);
        let mut h = Hasher::new();
        h.update(&input);

        let mut first = [0u8; 32];
        h.finalize_xof().fill(&mut first);
        let mut again = [0u8; 32];
        h.finalize_xof().fill(&mut again);
        assert_eq!(first, again, "finalizing twice must give the same digest");

        h.update(&bytes(50));
        let extended: Vec<u8> = input.iter().chain(bytes(50).iter()).copied().collect();
        let mut after = [0u8; 32];
        h.finalize_xof().fill(&mut after);
        assert_eq!(&after[..], &reference(&extended, 0, 32)[..], "absorbing after finalize must still be correct");
    }
}
