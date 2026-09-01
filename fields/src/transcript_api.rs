//! Runtime hash-family adapter: enum wrapper + factory.

use alloc::boxed::Box;
use alloc::vec::Vec;

use crate::{
    Blake3Transcript, Hash, Poseidon1_16, Poseidon2_16, PrimeField64, Sha256Transcript, Transcript,
    BLAKE3_TRANSCRIPT_STATE_WORDS, BLAKE3_TRANSCRIPT_XOF_WORDS, SHA256_TRANSCRIPT_STATE_WORDS,
};

pub enum TranscriptDyn<F: PrimeField64> {
    Poseidon1(Transcript<F, Poseidon1_16>),
    Poseidon2(Transcript<F, Poseidon2_16>),
    Blake3(Box<Blake3Transcript<F>>),
    Sha256(Box<Sha256Transcript<F>>),
}

impl<F: PrimeField64> TranscriptDyn<F> {
    pub fn put(&mut self, inputs: &[F]) {
        match self {
            TranscriptDyn::Poseidon1(t) => t.put(inputs),
            TranscriptDyn::Poseidon2(t) => t.put(inputs),
            TranscriptDyn::Blake3(t) => t.put(inputs),
            TranscriptDyn::Sha256(t) => t.put(inputs),
        }
    }
    pub fn get_state(&mut self) -> Vec<F> {
        match self {
            TranscriptDyn::Poseidon1(t) => t.get_state(),
            TranscriptDyn::Poseidon2(t) => t.get_state(),
            TranscriptDyn::Blake3(t) => t.get_state(),
            TranscriptDyn::Sha256(t) => t.get_state(),
        }
    }
    /// The state the lattice chain is seeded and stepped with, which is not always the
    /// digest. A sponge's chain round is its full width, and `get_state` already returns
    /// that. BLAKE3's digest is four words but one compression squeezes a whole 64-byte
    /// XOF block, so its chain runs eight words wide for the same cost -- half the rounds.
    ///
    /// `hash_state` must be called at exactly this width, and the emitted circom chain
    /// (`gen_calculate_hashes`) must use it too, or prover and verifier compute different
    /// contributions.
    pub fn get_chain_state(&mut self) -> Vec<F> {
        match self {
            TranscriptDyn::Poseidon1(t) => t.get_state(),
            TranscriptDyn::Poseidon2(t) => t.get_state(),
            TranscriptDyn::Blake3(t) => t.get_xof_block(),
            // No wider block: a squeeze IS the digest, so the chain runs 4 words wide.
            TranscriptDyn::Sha256(t) => t.get_state(),
        }
    }

    pub fn get_field(&mut self, value: &mut [F]) {
        match self {
            TranscriptDyn::Poseidon1(t) => t.get_field(value),
            TranscriptDyn::Poseidon2(t) => t.get_field(value),
            TranscriptDyn::Blake3(t) => t.get_field(value),
            TranscriptDyn::Sha256(t) => t.get_field(value),
        }
    }
}

pub fn new_transcript<F: PrimeField64>(hash_id: &str) -> TranscriptDyn<F> {
    match hash_id {
        "Poseidon1" => TranscriptDyn::Poseidon1(Transcript::<F, Poseidon1_16>::new()),
        "Poseidon2" => TranscriptDyn::Poseidon2(Transcript::<F, Poseidon2_16>::new()),
        "blake3" => TranscriptDyn::Blake3(Box::default()),
        "sha256" => TranscriptDyn::Sha256(Box::default()),
        other => panic!("Unknown hash family: {other:?}"),
    }
}

pub fn hash_state<F: PrimeField64>(hash_id: &str, state: &mut [F]) {
    match (hash_id, state.len()) {
        ("Poseidon1", 16) => Poseidon1_16::hash(state.try_into().unwrap()),
        ("Poseidon2", 16) => Poseidon2_16::hash(state.try_into().unwrap()),
        ("blake3", BLAKE3_TRANSCRIPT_STATE_WORDS) => {
            let mut transcript = Blake3Transcript::<F>::new();
            transcript.put(state);
            state.copy_from_slice(&transcript.get_state());
        }
        // Width 8 is `blake3core::permute8`: absorb the eight words, squeeze the whole
        // XOF block back. Same one compression as the width-4 form, twice the output,
        // which is what lets the lattice chain run at half the rounds.
        ("blake3", BLAKE3_TRANSCRIPT_XOF_WORDS) => {
            let mut transcript = Blake3Transcript::<F>::new();
            transcript.put(state);
            state.copy_from_slice(&transcript.get_xof_block());
        }
        ("sha256", SHA256_TRANSCRIPT_STATE_WORDS) => {
            let mut transcript = Sha256Transcript::<F>::new();
            transcript.put(state);
            state.copy_from_slice(&transcript.get_state());
        }
        (other, n) => panic!("Unknown hash family/width: {other:?}/{n}"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Goldilocks;

    /// The width `get_chain_state` returns is the contributions lattice chain's round
    /// width, and both sides of that chain must use it: the prover steps with
    /// `hash_state` at this width, and `gen_calculate_hashes` emits rounds this wide.
    /// Narrowing blake3's to the digest's four words leaves both sides self-consistent
    /// and silently disagreeing with each other, so pin it here.
    #[test]
    fn chain_state_width_is_the_families_round_width() {
        let cases = [("blake3", BLAKE3_TRANSCRIPT_XOF_WORDS), ("Poseidon1", 16), ("Poseidon2", 16)];
        for (family, want) in cases {
            let mut t = new_transcript::<Goldilocks>(family);
            t.put(&[Goldilocks::from_u64(7)]);
            assert_eq!(t.get_chain_state().len(), want, "{family} chain width");
        }
    }

    /// A chain round is `hash_state` at exactly that width, so every family's
    /// `get_chain_state` width must be one `hash_state` accepts -- it panics otherwise.
    #[test]
    fn hash_state_accepts_every_chain_width() {
        for family in ["blake3", "Poseidon1", "Poseidon2"] {
            let mut t = new_transcript::<Goldilocks>(family);
            t.put(&[Goldilocks::from_u64(1)]);
            let mut state = t.get_chain_state();
            hash_state(family, &mut state);
        }
    }

    /// blake3's chain round is one compression: the eight words `get_xof_block` returns
    /// are the same XOF block the transcript squeezes from sequentially. `get_field`
    /// draws three at a time, so its first draw is words 0..3 of that block.
    #[test]
    fn blake3_chain_state_is_the_squeezed_xof_block() {
        let input: Vec<Goldilocks> = (1..=5).map(Goldilocks::from_u64).collect();

        let mut block = Blake3Transcript::<Goldilocks>::new();
        block.put(&input);
        let whole = block.get_xof_block();
        assert_eq!(whole.len(), BLAKE3_TRANSCRIPT_XOF_WORDS);

        let mut squeezed = Blake3Transcript::<Goldilocks>::new();
        squeezed.put(&input);
        let mut first_three = [Goldilocks::from_u64(0); 3];
        squeezed.get_field(&mut first_three);

        assert_eq!(&whole[..3], &first_three[..], "same XOF block, read two ways");
    }
}
