//! Runtime hash-family adapter: enum wrapper + factory.

use alloc::vec::Vec;

use crate::{Blake3_16, Blake3_8, Hash, Poseidon1_16, Poseidon2_16, PrimeField64, Transcript};

pub enum TranscriptDyn<F: PrimeField64> {
    Poseidon1(Transcript<F, Poseidon1_16>),
    Poseidon2(Transcript<F, Poseidon2_16>),
    Blake3(Transcript<F, Blake3_16>),
    // Arity-2 (width-8) BLAKE3 transcript: mirrors the C++ TranscriptGL at
    // transcriptArity == 2. Used for the global/std challenge derivation so it
    // matches the per-Air challenges the prover/verifier compute at width-8.
    Blake3Arity2(Transcript<F, Blake3_8>),
}

impl<F: PrimeField64> TranscriptDyn<F> {
    pub fn put(&mut self, inputs: &[F]) {
        match self {
            TranscriptDyn::Poseidon1(t) => t.put(inputs),
            TranscriptDyn::Poseidon2(t) => t.put(inputs),
            TranscriptDyn::Blake3(t) => t.put(inputs),
            TranscriptDyn::Blake3Arity2(t) => t.put(inputs),
        }
    }
    pub fn get_state(&mut self) -> Vec<F> {
        match self {
            TranscriptDyn::Poseidon1(t) => t.get_state(),
            TranscriptDyn::Poseidon2(t) => t.get_state(),
            TranscriptDyn::Blake3(t) => t.get_state(),
            TranscriptDyn::Blake3Arity2(t) => t.get_state(),
        }
    }
    pub fn get_field(&mut self, value: &mut [F]) {
        match self {
            TranscriptDyn::Poseidon1(t) => t.get_field(value),
            TranscriptDyn::Poseidon2(t) => t.get_field(value),
            TranscriptDyn::Blake3(t) => t.get_field(value),
            TranscriptDyn::Blake3Arity2(t) => t.get_field(value),
        }
    }
}

pub fn new_transcript<F: PrimeField64>(hash_id: &str) -> TranscriptDyn<F> {
    match hash_id {
        "Poseidon1" => TranscriptDyn::Poseidon1(Transcript::<F, Poseidon1_16>::new()),
        "Poseidon2" => TranscriptDyn::Poseidon2(Transcript::<F, Poseidon2_16>::new()),
        "blake3" | "Blake3" => TranscriptDyn::Blake3(Transcript::<F, Blake3_16>::new()),
        other => panic!("Unknown hash family: {other:?}"),
    }
}

/// Arity-aware transcript constructor. For BLAKE3 the transcript width must
/// match the per-Air `transcriptArity` (width = 4*arity): arity-2 → width-8.
/// Poseidon transcripts are width-16 (arity-4) here as before.
pub fn new_transcript_arity<F: PrimeField64>(hash_id: &str, arity: usize) -> TranscriptDyn<F> {
    match (hash_id, arity) {
        ("blake3" | "Blake3", 2) => TranscriptDyn::Blake3Arity2(Transcript::<F, Blake3_8>::new()),
        _ => new_transcript(hash_id),
    }
}

pub fn hash_state<F: PrimeField64>(hash_id: &str, state: &mut [F; 16]) {
    match hash_id {
        "Poseidon1" => Poseidon1_16::hash(state),
        "Poseidon2" => Poseidon2_16::hash(state),
        "blake3" | "Blake3" => Blake3_16::hash(state),
        other => panic!("Unknown hash family: {other:?}"),
    }
}
