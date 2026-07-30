//! Fiat–Shamir transcript for the multilinear protocol: a thin wrapper over
//! `fields::Transcript` with extension-field challenges.

use crate::hypercube::Ext;
use crate::pcs::MlHashFamily;
use fields::{hash_state, new_transcript, Field, Goldilocks, PrimeField64, TranscriptDyn};

pub struct MlTranscript {
    inner: TranscriptDyn<Goldilocks>,
    hash: MlHashFamily,
}

impl MlTranscript {
    /// A transcript over the given hash family (Poseidon1/Poseidon2), matching
    /// the AIR's `MlParams::hash` so prover and verifier agree.
    pub fn new(hash: MlHashFamily) -> Self {
        Self { inner: new_transcript::<Goldilocks>(hash.id()), hash }
    }

    pub fn absorb(&mut self, vals: &[Goldilocks]) {
        self.inner.put(vals);
    }

    pub fn absorb_ext(&mut self, v: &Ext) {
        self.inner.put(&v.value);
    }

    pub fn absorb_exts(&mut self, vals: &[Ext]) {
        for v in vals {
            self.absorb_ext(v);
        }
    }

    pub fn absorb_root(&mut self, root: &[Goldilocks; 4]) {
        self.inner.put(root);
    }

    pub fn challenge(&mut self) -> Ext {
        let mut buf = [Goldilocks::ZERO; 3];
        self.inner.get_field(&mut buf);
        Ext::from_array(&buf)
    }

    pub fn challenges(&mut self, n: usize) -> Vec<Ext> {
        (0..n).map(|_| self.challenge()).collect()
    }

    /// Derive `n` query indices of `n_bits` bits each.
    pub fn query_indices(&mut self, n: u64, n_bits: u64) -> Vec<u64> {
        self.inner.get_permutations(n, n_bits)
    }

    /// Proof-of-work grinding. The prover searches for a `nonce`
    /// such that `H(seed ‖ nonce)` has at least `bits` trailing zero bits,
    /// where `seed` is drawn from the current transcript state.
    /// The nonce is then absorbed, so the subsequent query indices depend on it.
    /// Returns the winning nonce.
    pub fn grind(&mut self, bits: usize) -> u64 {
        if bits == 0 {
            return 0;
        }
        let seed = self.pow_seed();
        let nonce = find_pow_nonce(&seed, bits, self.hash);
        self.absorb(&[Goldilocks::from_u64(nonce)]);
        nonce
    }

    /// Verifier side of [`grind`](Self::grind): recompute the PoW check for the
    /// prover-supplied `nonce` and absorb it, keeping the transcript in lock
    /// step. Returns `false` if the nonce does not satisfy the `bits` target.
    #[must_use]
    pub fn verify_grind(&mut self, nonce: u64, bits: usize) -> bool {
        if bits == 0 {
            return true;
        }
        let seed = self.pow_seed();
        let ok = pow_ok(&seed, nonce, bits, self.hash);
        self.absorb(&[Goldilocks::from_u64(nonce)]);
        ok
    }

    /// The grinding seed: the transcript's current sponge state, binding every
    /// message absorbed so far. Drawn identically by prover and verifier.
    fn pow_seed(&mut self) -> Vec<Goldilocks> {
        self.inner.get_state()
    }
}

/// `H(seed ‖ nonce)` has `≥ bits` trailing zero bits, `H` the width-16
/// permutation of the transcript's hash family. The input is the seed state
/// (≤ 15 cells) with the nonce in the last cell.
fn pow_ok(seed: &[Goldilocks], nonce: u64, bits: usize, hash: MlHashFamily) -> bool {
    let mut input = [Goldilocks::ZERO; 16];
    let n = seed.len().min(15);
    input[..n].copy_from_slice(&seed[..n]);
    input[15] = Goldilocks::from_u64(nonce);
    hash_state(hash.id(), &mut input);
    (input[0].as_canonical_u64().trailing_zeros() as usize) >= bits
}

/// Search for the smallest `nonce` satisfying [`pow_ok`].
#[cfg(feature = "parallel")]
fn find_pow_nonce(seed: &[Goldilocks], bits: usize, hash: MlHashFamily) -> u64 {
    use rayon::prelude::*;
    const BLOCK: u64 = 1 << 16;
    let mut start = 0u64;
    loop {
        let hit = (start..start + BLOCK).into_par_iter().find_map_first(|n| pow_ok(seed, n, bits, hash).then_some(n));
        if let Some(n) = hit {
            return n;
        }
        start += BLOCK;
    }
}

#[cfg(not(feature = "parallel"))]
fn find_pow_nonce(seed: &[Goldilocks], bits: usize, hash: MlHashFamily) -> u64 {
    (0u64..).find(|&n| pow_ok(seed, n, bits, hash)).expect("grinding nonce exists")
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Prover and verifier draw the same seed and agree on the winning nonce;
    /// the found nonce clears the target and a wrong nonce is rejected.
    #[test]
    fn grind_roundtrip_and_rejection() {
        let bits = 8;
        // Same absorbed prefix on both sides.
        let seed = |t: &mut MlTranscript| t.absorb(&[Goldilocks::from_u64(42), Goldilocks::from_u64(7)]);

        let mut tp = MlTranscript::new(crate::MlHashFamily::Poseidon2);
        seed(&mut tp);
        let nonce = tp.grind(bits);

        let mut tv = MlTranscript::new(crate::MlHashFamily::Poseidon2);
        seed(&mut tv);
        assert!(tv.verify_grind(nonce, bits), "the winning nonce must verify");

        // A different nonce almost surely misses the 2^-8 target.
        let mut tw = MlTranscript::new(crate::MlHashFamily::Poseidon2);
        seed(&mut tw);
        assert!(!tw.verify_grind(nonce.wrapping_add(1), bits), "a wrong nonce must be rejected");
    }

    /// After grind/verify_grind the two transcripts stay in lock step: the next
    /// challenge matches (the nonce was absorbed identically on both sides).
    #[test]
    fn grind_keeps_transcripts_in_sync() {
        let bits = 6;
        let mut tp = MlTranscript::new(crate::MlHashFamily::Poseidon2);
        tp.absorb(&[Goldilocks::from_u64(99)]);
        let nonce = tp.grind(bits);
        let after_prover = tp.challenge();

        let mut tv = MlTranscript::new(crate::MlHashFamily::Poseidon2);
        tv.absorb(&[Goldilocks::from_u64(99)]);
        assert!(tv.verify_grind(nonce, bits));
        assert_eq!(after_prover, tv.challenge(), "post-grind transcripts diverged");
    }

    /// `bits == 0` is a pure no-op: no nonce absorbed, transcript unchanged.
    #[test]
    fn grind_zero_bits_is_noop() {
        let mut t0 = MlTranscript::new(crate::MlHashFamily::Poseidon2);
        t0.absorb(&[Goldilocks::from_u64(1)]);
        assert_eq!(t0.grind(0), 0);
        let c0 = t0.challenge();

        let mut t1 = MlTranscript::new(crate::MlHashFamily::Poseidon2);
        t1.absorb(&[Goldilocks::from_u64(1)]);
        assert_eq!(c0, t1.challenge(), "grind(0) must not touch the transcript");
    }
}
