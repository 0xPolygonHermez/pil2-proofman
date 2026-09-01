//! Fiat-Shamir transcript that is a genuine SHA-256 over the absorbed byte stream.
//!
//! Wraps the reference `sha2` crate rather than reimplementing it, so this side is the oracle for
//! the C++ prover (`TranscriptGL` + `sha256core::Hasher`).
//!
//! SHA-256 has no XOF, and inventing one is the classic way to break Fiat-Shamir quietly, so
//! challenges are two nested literal FIPS hashes: `SHA256(SHA256(absorbed) || LE64(ctr))`, 4 words
//! per call. Same shape as [`crate::Blake3Transcript`], 4 words per refill instead of 8.

use alloc::vec::Vec;

use sha2::{Digest, Sha256};

use crate::blake3_transcript::canon;
use crate::PrimeField64;

/// Words a squeeze yields, and what `get_state` returns: the 32-byte digest, matching
/// `HASH_SIZE` on the C++ side (`TranscriptGL::transcriptStateSize`).
pub const SHA256_TRANSCRIPT_STATE_WORDS: usize = 4;

pub struct Sha256Transcript<F: PrimeField64> {
    hasher: Sha256,
    out: [u64; SHA256_TRANSCRIPT_STATE_WORDS],
    /// Words already consumed from `out`.
    offset: usize,
    /// Squeeze counter of the block held in `out`.
    counter: u64,
    /// Whether `out` reflects the current absorbed stream.
    valid: bool,
    _marker: core::marker::PhantomData<F>,
}

impl<F: PrimeField64> Default for Sha256Transcript<F> {
    fn default() -> Self {
        Self::new()
    }
}

impl<F: PrimeField64> Sha256Transcript<F> {
    pub fn new() -> Self {
        Sha256Transcript {
            hasher: Sha256::new(),
            out: [0u64; SHA256_TRANSCRIPT_STATE_WORDS],
            offset: 0,
            counter: 0,
            valid: false,
            _marker: core::marker::PhantomData,
        }
    }

    pub fn put(&mut self, inputs: &[F]) {
        for x in inputs {
            self.hasher.update(x.as_canonical_u64().to_le_bytes());
        }
        // The stream changed, so squeeze material from the old prefix is stale.
        self.valid = false;
        self.offset = 0;
        self.counter = 0;
    }

    /// Does not consume: the hasher is cloned, because a transcript keeps absorbing after every
    /// challenge.
    fn squeeze(&self, counter: u64) -> [u64; SHA256_TRANSCRIPT_STATE_WORDS] {
        let digest = self.hasher.clone().finalize();
        let mut msg = [0u8; 8 * SHA256_TRANSCRIPT_STATE_WORDS + 8];
        // The C++ side hashes the digest's canonical field elements, not the raw bytes.
        for i in 0..SHA256_TRANSCRIPT_STATE_WORDS {
            let w = canon(u64::from_le_bytes(digest[8 * i..8 * i + 8].try_into().unwrap()));
            msg[8 * i..8 * i + 8].copy_from_slice(&w.to_le_bytes());
        }
        msg[32..40].copy_from_slice(&counter.to_le_bytes());

        let sq = Sha256::digest(msg);
        let mut out = [0u64; SHA256_TRANSCRIPT_STATE_WORDS];
        for (i, o) in out.iter_mut().enumerate() {
            *o = canon(u64::from_le_bytes(sq[8 * i..8 * i + 8].try_into().unwrap()));
        }
        out
    }

    fn get_fields1(&mut self) -> F {
        if !self.valid {
            self.counter = 0;
            self.out = self.squeeze(self.counter);
            self.offset = 0;
            self.valid = true;
        } else if self.offset == SHA256_TRANSCRIPT_STATE_WORDS {
            self.counter += 1;
            self.out = self.squeeze(self.counter);
            self.offset = 0;
        }
        let v = self.out[self.offset];
        self.offset += 1;
        F::from_u64(v)
    }

    /// The first squeeze block, which is where `get_field` starts too, matching the C++
    /// `TranscriptGL::getState`. Does **not** consume.
    pub fn get_state(&mut self) -> Vec<F> {
        self.squeeze(0).iter().map(|&v| F::from_u64(v)).collect()
    }

    pub fn get_field(&mut self, value: &mut [F]) {
        for v in value.iter_mut().take(3) {
            *v = self.get_fields1();
        }
    }

    /// Verbatim port of `Transcript::get_permutations`: only the source of `fields` differs.
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

impl<F: PrimeField64> crate::transcript::TranscriptLike<F> for Sha256Transcript<F> {
    fn new_transcript() -> Self {
        Self::new()
    }
    fn put(&mut self, inputs: &[F]) {
        Sha256Transcript::put(self, inputs)
    }
    fn get_field(&mut self, value: &mut [F]) {
        Sha256Transcript::get_field(self, value)
    }
    fn get_state(&mut self) -> Vec<F> {
        Sha256Transcript::get_state(self)
    }
    fn get_permutations(&mut self, n: u64, n_bits: u64) -> Vec<u64> {
        Sha256Transcript::get_permutations(self, n, n_bits)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Goldilocks;

    // reference that generates the C++ vectors in sha256_core_vectors.hpp.
    const SQUEEZE_VECTORS: &[(&[u64], u64, [u64; 4])] = &[
        (&[], 0, [0xC0719FF3DC425D5C, 0xDB18D5D820A76C22, 0xE4E538F073575B61, 0xBD1B62476F3F9691]),
        (&[0x91778AED87EE5EB1], 0, [0x9163B608B0F86ABE, 0xBE2A63EE41547AA3, 0xC1A003EB4AEEED1F, 0xE6A3F6797399BFD7]),
        (&[0x91778AED87EE5EB1, 0x39B7F8A5C64CF56C, 0x69AFC5A5E88B394B, 0xFFFFFFFF00000001, 0xFFFFFFFF3B2C6C23, 0x06B6F019AA7B0DE0, 0x05921DB28E4B11AF], 0, [0x47FFBF2F2760ABAA, 0x3FB37959E4864D7D, 0x12290D3E316C8F10, 0xB71E7F6041C617F6]),
        (&[0x91778AED87EE5EB1, 0x39B7F8A5C64CF56C, 0x69AFC5A5E88B394B, 0xFFFFFFFF00000001, 0xFFFFFFFF3B2C6C23, 0x06B6F019AA7B0DE0, 0x05921DB28E4B11AF, 0x27074B2D773A6E12], 0, [0x7BBA3F6523F3D1C6, 0x7CAD66249A21FC08, 0xFA550CB7100F3588, 0x01922D948775C16F]),
        (&[0x91778AED87EE5EB1, 0x39B7F8A5C64CF56C, 0x69AFC5A5E88B394B, 0xFFFFFFFF00000001, 0xFFFFFFFF3B2C6C23, 0x06B6F019AA7B0DE0, 0x05921DB28E4B11AF, 0x27074B2D773A6E12], 1, [0xD8B76B9697DAD263, 0x5D7966D6B4819A92, 0xD40F3F7470636C31, 0x6F34506C0DD67FE8]),
        (&[0x91778AED87EE5EB1, 0x39B7F8A5C64CF56C, 0x69AFC5A5E88B394B, 0xFFFFFFFF00000001, 0xFFFFFFFF3B2C6C23, 0x06B6F019AA7B0DE0, 0x05921DB28E4B11AF, 0x27074B2D773A6E12, 0x79557F7C51C1C879], 3, [0x7C4B06FE1C911AEC, 0x45BB66146B156B3B, 0xC1CB2B58778E7A5F, 0xAF3F9D2675D73D49]),
        (&[0x91778AED87EE5EB1, 0x39B7F8A5C64CF56C, 0x69AFC5A5E88B394B, 0xFFFFFFFF00000001, 0xFFFFFFFF3B2C6C23, 0x06B6F019AA7B0DE0, 0x05921DB28E4B11AF, 0x27074B2D773A6E12, 0x79557F7C51C1C879, 0xFFFFFFFF19B20510, 0x5B541B11572AA853, 0x8765F02D0FB744E6, 0xEBA3F0C407AAB7BD, 0x5396A2E40E919088, 0xFFFFFFFF08985B92, 0x6410386B69DAE0FA, 0xC9D01B0551FF1341, 0x1133A8B66CA022BC, 0x6BF6B2038834E05B, 0xFFFFFFFF7214C260, 0x57CFA253E64D1F05, 0x6DF4CB783F427030, 0x698DEA6D97AD09BF, 0x5838782FD8D2F8E2, 0xFFFFFFFF1518B3EB, 0x865F46FA572DACE4, 0x888627BD43E80163, 0x7C105D4D327EDCB6, 0x6A5B7F784520974D, 0xFFFFFFFFD32CC033, 0xEEF06DC546A62B47, 0xB5756044166855CA, 0x69B6CF4890DFCBD1, 0xA4F6B5A02180040C, 0xFFFFFFFF4EB1AB81, 0xA1C35991D3AE381E, 0x2E1DF47A2C544095], 0, [0xEB5CB27BC75B555A, 0xC3A6AB4B722C5008, 0x0F31ECCA03A343A7, 0x72F8FB84F346F4DB]),
        (&[0x91778AED87EE5EB1, 0x39B7F8A5C64CF56C, 0x69AFC5A5E88B394B, 0xFFFFFFFF00000001, 0xFFFFFFFF3B2C6C23, 0x06B6F019AA7B0DE0, 0x05921DB28E4B11AF, 0x27074B2D773A6E12, 0x79557F7C51C1C879, 0xFFFFFFFF19B20510, 0x5B541B11572AA853, 0x8765F02D0FB744E6, 0xEBA3F0C407AAB7BD, 0x5396A2E40E919088, 0xFFFFFFFF08985B92, 0x6410386B69DAE0FA, 0xC9D01B0551FF1341, 0x1133A8B66CA022BC, 0x6BF6B2038834E05B, 0xFFFFFFFF7214C260, 0x57CFA253E64D1F05, 0x6DF4CB783F427030, 0x698DEA6D97AD09BF, 0x5838782FD8D2F8E2, 0xFFFFFFFF1518B3EB, 0x865F46FA572DACE4, 0x888627BD43E80163, 0x7C105D4D327EDCB6, 0x6A5B7F784520974D, 0xFFFFFFFFD32CC033, 0xEEF06DC546A62B47, 0xB5756044166855CA, 0x69B6CF4890DFCBD1, 0xA4F6B5A02180040C, 0xFFFFFFFF4EB1AB81, 0xA1C35991D3AE381E, 0x2E1DF47A2C544095], 7, [0x69AD0A4A23386C83, 0x99D9E57202F47B11, 0x920380B1C574AA98, 0x872D3FA6980F6E8C]),
        (&[0x91778AED87EE5EB1, 0x39B7F8A5C64CF56C, 0x69AFC5A5E88B394B, 0xFFFFFFFF00000001, 0xFFFFFFFF3B2C6C23, 0x06B6F019AA7B0DE0, 0x05921DB28E4B11AF, 0x27074B2D773A6E12, 0x79557F7C51C1C879, 0xFFFFFFFF19B20510, 0x5B541B11572AA853, 0x8765F02D0FB744E6, 0xEBA3F0C407AAB7BD, 0x5396A2E40E919088, 0xFFFFFFFF08985B92, 0x6410386B69DAE0FA, 0xC9D01B0551FF1341, 0x1133A8B66CA022BC, 0x6BF6B2038834E05B, 0xFFFFFFFF7214C260, 0x57CFA253E64D1F05, 0x6DF4CB783F427030, 0x698DEA6D97AD09BF, 0x5838782FD8D2F8E2, 0xFFFFFFFF1518B3EB, 0x865F46FA572DACE4, 0x888627BD43E80163, 0x7C105D4D327EDCB6, 0x6A5B7F784520974D, 0xFFFFFFFFD32CC033, 0xEEF06DC546A62B47, 0xB5756044166855CA, 0x69B6CF4890DFCBD1, 0xA4F6B5A02180040C, 0xFFFFFFFF4EB1AB81, 0xA1C35991D3AE381E, 0x2E1DF47A2C544095, 0x5B593F5D60F9C680, 0xEEE26ED37548E5CF, 0xFFFFFFFF9AC00E97, 0x26FDD465B5017999, 0xF6824405B609C834, 0xDD05E213C7B57E73, 0xEFD2FD6FD8F8C886, 0xFFFFFFFF6835DB8C, 0xD5E1CF0AA7857D28, 0x64C56B5B5B3D5957, 0x05B49E5744255E9A, 0x37CEC68C43898861, 0xFFFFFFFF5D1A219B, 0x8006961C7D401A7B, 0x73D6D26EB6662DEE, 0x27CECF631BACA625, 0x8291EFF62AB610D0, 0xFFFFFFFF5D538C5A, 0x6F14655FA1FC4A82, 0x5124ED7A8F611829, 0xC04AA10F92521784, 0x0C26FFC949541F83, 0xFFFFFFFF160962F1, 0x72AAC5D638C3A26D, 0x0069178EE4D02178, 0xDABBBC4B8B6FEB67, 0xC16DFE1496CEFB6A], 0, [0x8CC534ED1C222325, 0x25315353E95B33C1, 0x0E598BDDF434B363, 0xD66FBB43E9EACA6C]),
        (&[0x91778AED87EE5EB1, 0x39B7F8A5C64CF56C, 0x69AFC5A5E88B394B, 0xFFFFFFFF00000001, 0xFFFFFFFF3B2C6C23, 0x06B6F019AA7B0DE0, 0x05921DB28E4B11AF, 0x27074B2D773A6E12, 0x79557F7C51C1C879, 0xFFFFFFFF19B20510, 0x5B541B11572AA853, 0x8765F02D0FB744E6, 0xEBA3F0C407AAB7BD, 0x5396A2E40E919088, 0xFFFFFFFF08985B92, 0x6410386B69DAE0FA, 0xC9D01B0551FF1341, 0x1133A8B66CA022BC, 0x6BF6B2038834E05B, 0xFFFFFFFF7214C260, 0x57CFA253E64D1F05, 0x6DF4CB783F427030, 0x698DEA6D97AD09BF, 0x5838782FD8D2F8E2, 0xFFFFFFFF1518B3EB, 0x865F46FA572DACE4, 0x888627BD43E80163, 0x7C105D4D327EDCB6, 0x6A5B7F784520974D, 0xFFFFFFFFD32CC033, 0xEEF06DC546A62B47, 0xB5756044166855CA, 0x69B6CF4890DFCBD1, 0xA4F6B5A02180040C, 0xFFFFFFFF4EB1AB81, 0xA1C35991D3AE381E, 0x2E1DF47A2C544095, 0x5B593F5D60F9C680, 0xEEE26ED37548E5CF, 0xFFFFFFFF9AC00E97, 0x26FDD465B5017999, 0xF6824405B609C834, 0xDD05E213C7B57E73, 0xEFD2FD6FD8F8C886, 0xFFFFFFFF6835DB8C, 0xD5E1CF0AA7857D28, 0x64C56B5B5B3D5957, 0x05B49E5744255E9A, 0x37CEC68C43898861, 0xFFFFFFFF5D1A219B, 0x8006961C7D401A7B, 0x73D6D26EB6662DEE, 0x27CECF631BACA625, 0x8291EFF62AB610D0, 0xFFFFFFFF5D538C5A, 0x6F14655FA1FC4A82, 0x5124ED7A8F611829, 0xC04AA10F92521784, 0x0C26FFC949541F83, 0xFFFFFFFF160962F1, 0x72AAC5D638C3A26D, 0x0069178EE4D02178, 0xDABBBC4B8B6FEB67, 0xC16DFE1496CEFB6A, 0xFFFFFFFF5FEB179E, 0x2E882E668139E2AC, 0xFCA5ABD76423AD8B, 0xF02568C18AABF7BE, 0x42DB5159411F4FB5, 0xFFFFFFFF47C11853, 0xDD0AB1A72E3249EF, 0xDEFB420805071152, 0x05CD4B11C7F73AB9, 0xE106EEABDDAB9AD4, 0xFFFFFFFF8410D55D, 0x55B2A5E98DF79C26, 0x19315121EC82CDFD, 0xA846698C6ED739C8, 0x5D96E4BF3A4EE177, 0xFFFFFFFFE06F2408, 0x85F290C44D5C0D81, 0x7290FEA4635DDFFC, 0x98BC0F1E2EB1E49B, 0x5B36059E7F4C958E, 0xFFFFFFFF46FDCCB8, 0x98FD83933AD18170, 0x81A5959BD3A1D1FF, 0xE327433A5F71EC22, 0xB46E8B7DCA5CE149, 0xFFFFFFFF0D1342EB, 0xE69934449848CDA3, 0xA862742CEE1683F6, 0x1A5D1E07D301BD8D, 0xE711AE9F1ACFC618, 0xFFFFFFFF306A5341, 0x1E1EC705141BF10A, 0xDAE7E2F19FB6D611, 0x54F7502862A2914C, 0xBB2841A4154BBFAB, 0xFFFFFFFFD7EC0252, 0x00F79212D2B76ED5, 0xD5CB1872BF9AA7C0, 0xDF57A049018F3E0F, 0x5DF88388E019DAF2, 0xFFFFFFFF095ECE2C, 0x4E9BAA8734663D74, 0x7D7ABF661A60DAB3, 0x6BEAF04B151BBFC6, 0xA5CDEC0DD7A9711D, 0xFFFFFFFF6D5BF7A0, 0xEAB30ED99B95F997, 0x94BCEBDCD07949DA, 0x555698927C3EA2A1, 0xAB08EBEB54CCF69C, 0xFFFFFFFF0FC82F99, 0xC06389BC93124D2E, 0xF499B04179AEE465, 0x69191AB9093CC210, 0xEDD6C0646EAB8E1F, 0xFFFFFFFF420EDD4F, 0xC97DBD44B03ABA69, 0x7488ADF9BF515CC4, 0xAD3F96A823CE0BC3, 0x1D8F508236F44F96, 0xFFFFFFFFD4A79D84, 0xA45A476CADA93AB8, 0x3E10F7063F601BA7, 0xEF177489753736AA, 0x575B6A0E3F2C7331, 0xFFFFFFFF37BE2BF3], 11, [0x8A2F307CA1CD2113, 0xDEED83910BEEF46D, 0xA72335C9BD925F64, 0xF810325C41176E7B]),
    ];

    /// Checked against the SAME independent reference the C++ vectors come from, not against C++.
    #[test]
    fn squeeze_matches_the_independent_reference() {
        for (absorbed, counter, want) in SQUEEZE_VECTORS {
            let mut t = Sha256Transcript::<Goldilocks>::new();
            let elems: Vec<Goldilocks> = absorbed.iter().map(|&w| Goldilocks::from_u64(w)).collect();
            t.put(&elems);
            assert_eq!(t.squeeze(*counter), *want, "absorbed {} words, counter {counter}", absorbed.len());
        }
    }

    /// get_state must read the squeeze(0) that get_field starts from, as blake3's does.
    #[test]
    fn get_state_is_the_first_squeeze_block() {
        let mut t = Sha256Transcript::<Goldilocks>::new();
        t.put(&[Goldilocks::from_u64(1), Goldilocks::from_u64(2)]);
        let state = t.get_state();
        let mut got = [Goldilocks::from_u64(0); 3];
        t.get_field(&mut got);
        for i in 0..3 {
            assert_eq!(got[i], state[i], "challenge word {i} should come from squeeze(0)");
        }
    }

    /// Otherwise the next challenge repeats material from the shorter prefix.
    #[test]
    fn absorbing_invalidates_the_cached_squeeze() {
        let mut t = Sha256Transcript::<Goldilocks>::new();
        t.put(&[Goldilocks::from_u64(7)]);
        let mut a = [Goldilocks::from_u64(0); 3];
        t.get_field(&mut a);
        t.put(&[Goldilocks::from_u64(9)]);
        let mut b = [Goldilocks::from_u64(0); 3];
        t.get_field(&mut b);
        assert_ne!(a, b, "a challenge after a new absorb must differ");
    }

    /// A refill must advance the counter, not repeat.
    #[test]
    fn refilling_advances_the_counter() {
        let mut t = Sha256Transcript::<Goldilocks>::new();
        t.put(&[Goldilocks::from_u64(3)]);
        // 4 words per squeeze, so 6 challenges (18 words) span several counters.
        let mut seen = Vec::new();
        for _ in 0..6 {
            let mut c = [Goldilocks::from_u64(0); 3];
            t.get_field(&mut c);
            seen.extend_from_slice(&c);
        }
        let mut uniq = seen.clone();
        uniq.sort_by_key(|x| x.as_canonical_u64());
        uniq.dedup();
        assert_eq!(uniq.len(), seen.len(), "refills must not repeat words");
    }
}
