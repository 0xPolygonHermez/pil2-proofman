//! Fiat–Shamir transcript for the multilinear protocol: a thin wrapper over
//! `fields::Transcript` (Poseidon2 sponge) with extension-field challenges.

use crate::hypercube::Ext;
use fields::{Field, Goldilocks, Poseidon2_16, Transcript};

pub struct MlTranscript {
    inner: Transcript<Goldilocks, Poseidon2_16>,
}

impl Default for MlTranscript {
    fn default() -> Self {
        Self::new()
    }
}

impl MlTranscript {
    pub fn new() -> Self {
        Self { inner: Transcript::new() }
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
}
