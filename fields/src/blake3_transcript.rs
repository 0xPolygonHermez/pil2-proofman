//! Fiat-Shamir transcript that is a genuine BLAKE3 over the absorbed byte
//! stream, with challenges drawn from that hash's XOF.
//!
//! Wraps the reference `blake3` crate rather than reimplementing the
//! construction, so the verifier side is correct by definition and can serve as
//! the oracle for the C++ prover (`TranscriptGL` + `blake3core::Hasher`).
//!
//! Why this is not a `Transcript<F, H: Hash<F>>`: that type is a sponge, with
//! `WIDTH`/`RATE`/`CAPACITY` and a state carried *as block content*. BLAKE3
//! carries its state in the chaining value, which no sponge signature can
//! express — `Blake3_8` only fits `Hash` by pretending BLAKE3 is a width-8
//! permutation, which is exactly the construction this replaces.

use alloc::vec::Vec;

use crate::PrimeField64;

const GL_P: u64 = 0xFFFFFFFF00000001;

/// One conditional subtract, matching `blake3core::to_canonical`. A u64 is
/// always below 2p, so this is a full reduction.
#[inline]
fn canon(x: u64) -> u64 {
    if x >= GL_P {
        x - GL_P
    } else {
        x
    }
}

/// Words per 64-byte XOF output block.
const XOF_BLOCK_WORDS: usize = 8;

/// Words `get_state` returns: the 32-byte BLAKE3 digest, matching `HASH_SIZE`
/// on the C++ side (`TranscriptGL::transcriptStateSize`).
pub const BLAKE3_TRANSCRIPT_STATE_WORDS: usize = 4;

pub struct Blake3Transcript<F: PrimeField64> {
    hasher: ::blake3::Hasher,
    /// The XOF output block currently loaded.
    xof: [u64; XOF_BLOCK_WORDS],
    /// Words already consumed from `xof`.
    offset: usize,
    /// Index of the output block held in `xof`.
    block: u64,
    /// Whether `xof` reflects the current absorbed stream.
    valid: bool,
    _marker: core::marker::PhantomData<F>,
}

impl<F: PrimeField64> Default for Blake3Transcript<F> {
    fn default() -> Self {
        Self::new()
    }
}

impl<F: PrimeField64> Blake3Transcript<F> {
    pub fn new() -> Self {
        Blake3Transcript {
            hasher: ::blake3::Hasher::new(),
            xof: [0u64; XOF_BLOCK_WORDS],
            offset: 0,
            block: 0,
            valid: false,
            _marker: core::marker::PhantomData,
        }
    }

    pub fn put(&mut self, inputs: &[F]) {
        for x in inputs {
            self.hasher.update(&x.as_canonical_u64().to_le_bytes());
        }
        // The stream changed, so XOF material from the old prefix is stale.
        self.valid = false;
        self.offset = 0;
        self.block = 0;
    }

    /// Load XOF output block `self.block`.
    fn load_block(&mut self) {
        let mut reader = self.hasher.finalize_xof();
        reader.set_position(self.block * 64);
        let mut buf = [0u8; 64];
        reader.fill(&mut buf);
        for i in 0..XOF_BLOCK_WORDS {
            let raw = u64::from_le_bytes(buf[8 * i..8 * i + 8].try_into().unwrap());
            self.xof[i] = canon(raw);
        }
        self.offset = 0;
        self.valid = true;
    }

    fn get_fields1(&mut self) -> F {
        if !self.valid {
            self.block = 0;
            self.load_block();
        } else if self.offset == XOF_BLOCK_WORDS {
            self.block += 1;
            self.load_block();
        }
        let v = self.xof[self.offset];
        self.offset += 1;
        F::from_u64(v)
    }

    /// The BLAKE3 digest of the transcript so far. Does **not** consume, matching
    /// the sponge version, which reads `state` rather than draining `out`.
    pub fn get_state(&mut self) -> Vec<F> {
        let mut reader = self.hasher.finalize_xof();
        let mut buf = [0u8; 8 * BLAKE3_TRANSCRIPT_STATE_WORDS];
        reader.fill(&mut buf);
        (0..BLAKE3_TRANSCRIPT_STATE_WORDS)
            .map(|i| F::from_u64(canon(u64::from_le_bytes(buf[8 * i..8 * i + 8].try_into().unwrap()))))
            .collect()
    }

    pub fn get_field(&mut self, value: &mut [F]) {
        for v in value.iter_mut().take(3) {
            *v = self.get_fields1();
        }
    }

    /// Verbatim port of `Transcript::get_permutations` so the bit-packing stays
    /// identical across hash families -- only the source of `fields` differs.
    pub fn get_permutations(&mut self, n: u64, n_bits: u64) -> Vec<u64> {
        let total_bits = n * n_bits;
        let n_fields = ((total_bits - 1) / 63) + 1;
        let mut fields = Vec::with_capacity(n_fields as usize);
        for _ in 0..n_fields {
            fields.push(self.get_fields1());
        }

        let mut cur_field = 0usize;
        let mut cur_bit = 0u64;

        let mut permutations = alloc::vec![0u64; n as usize];
        for slot in permutations.iter_mut() {
            let mut a = 0u64;
            for j in 0..n_bits {
                let bit = (fields[cur_field].as_canonical_u64() >> cur_bit) & 1;
                if bit == 1 {
                    a += 1 << j;
                }
                cur_bit += 1;
                if cur_bit == 63 {
                    cur_bit = 0;
                    cur_field += 1;
                }
            }
            *slot = a;
        }
        permutations
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Field, Goldilocks};

    fn words(n: usize) -> Vec<Goldilocks> {
        (0..n).map(|i| Goldilocks::new((i as u64) * 7 + 3)).collect()
    }

    /// The transcript must be blake3(canonical-LE stream), so the reference
    /// crate fed the same bytes has to agree word for word.
    #[test]
    fn matches_reference_blake3() {
        for n in [0usize, 1, 7, 8, 9, 127, 128, 129, 300] {
            let xs = words(n);

            let mut t = Blake3Transcript::<Goldilocks>::new();
            t.put(&xs);
            let got = t.get_state();

            let mut h = ::blake3::Hasher::new();
            for x in &xs {
                h.update(&x.as_canonical_u64().to_le_bytes());
            }
            let mut buf = [0u8; 32];
            h.finalize_xof().fill(&mut buf);

            for i in 0..4 {
                let raw = u64::from_le_bytes(buf[8 * i..8 * i + 8].try_into().unwrap());
                assert_eq!(got[i], Goldilocks::new(canon(raw)), "n={n} word={i}");
            }
        }
    }

    /// A put after a read must restart the XOF from the new prefix.
    #[test]
    fn put_invalidates_the_xof_stream() {
        let mut t = Blake3Transcript::<Goldilocks>::new();
        t.put(&words(4));
        let mut a = [Goldilocks::ZERO; 3];
        t.get_field(&mut a);

        t.put(&words(1));
        let mut b = [Goldilocks::ZERO; 3];
        t.get_field(&mut b);
        assert_ne!(a, b);
    }

    /// get_state must not consume, so a following get_field is unaffected.
    #[test]
    fn get_state_does_not_consume() {
        let mut t1 = Blake3Transcript::<Goldilocks>::new();
        t1.put(&words(5));
        let _ = t1.get_state();
        let mut with = [Goldilocks::ZERO; 3];
        t1.get_field(&mut with);

        let mut t2 = Blake3Transcript::<Goldilocks>::new();
        t2.put(&words(5));
        let mut without = [Goldilocks::ZERO; 3];
        t2.get_field(&mut without);

        assert_eq!(with, without);
    }

    /// Reading past 8 words must advance into the next XOF block. Checked against
    /// the reference rather than merely asserting non-repetition.
    #[test]
    fn reads_past_one_block_match_the_reference() {
        let xs = words(300);
        let mut t = Blake3Transcript::<Goldilocks>::new();
        t.put(&xs);

        let mut got = Vec::new();
        for _ in 0..6 {
            let mut v = [Goldilocks::ZERO; 3];
            t.get_field(&mut v);
            got.extend_from_slice(&v);
        }

        let mut h = ::blake3::Hasher::new();
        for x in &xs {
            h.update(&x.as_canonical_u64().to_le_bytes());
        }
        let mut buf = [0u8; 192]; // 18 words spans three 64-byte XOF blocks
        h.finalize_xof().fill(&mut buf);

        for i in 0..18 {
            let raw = u64::from_le_bytes(buf[8 * i..8 * i + 8].try_into().unwrap());
            assert_eq!(got[i], Goldilocks::new(canon(raw)), "word={i}");
        }
    }

    /// The same construction the C++ side produces, pinned by golden values
    /// generated with b3sum. Guards against prover/verifier drift.
    #[test]
    fn matches_cpp_golden_vectors() {
        // n = 9, first 8 XOF words; identical to kGolden[3] in
        // pil2-stark/src/goldilocks/tests/test_blake3_transcript_cpu.cpp
        let expect: [u64; 8] = [
            2424636365142760339,
            15165381830123158802,
            9487485792073438855,
            5920058426812994410,
            16462720151111991777,
            7237086037464224556,
            14801379881922525855,
            18396241790501459263,
        ];
        let mut t = Blake3Transcript::<Goldilocks>::new();
        t.put(&words(9));
        for (i, want) in expect.iter().enumerate() {
            let mut v = [Goldilocks::ZERO; 3];
            // Read one word at a time via get_field's first slot.
            t.get_field(&mut v[..1]);
            assert_eq!(v[0], Goldilocks::new(*want), "word={i}");
        }
    }
}
