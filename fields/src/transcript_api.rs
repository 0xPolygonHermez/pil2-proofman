//! Runtime hash-family adapter: enum wrapper + factory.

use alloc::vec::Vec;

use crate::{Hash, Poseidon1_16, Poseidon2_16, PrimeField64, Transcript};

pub enum TranscriptDyn<F: PrimeField64> {
    Poseidon1(Transcript<F, Poseidon1_16>),
    Poseidon2(Transcript<F, Poseidon2_16>),
}

impl<F: PrimeField64> TranscriptDyn<F> {
    pub fn put(&mut self, inputs: &[F]) {
        match self {
            TranscriptDyn::Poseidon1(t) => t.put(inputs),
            TranscriptDyn::Poseidon2(t) => t.put(inputs),
        }
    }
    pub fn get_state(&mut self) -> Vec<F> {
        match self {
            TranscriptDyn::Poseidon1(t) => t.get_state(),
            TranscriptDyn::Poseidon2(t) => t.get_state(),
        }
    }
    pub fn get_field(&mut self, value: &mut [F]) {
        match self {
            TranscriptDyn::Poseidon1(t) => t.get_field(value),
            TranscriptDyn::Poseidon2(t) => t.get_field(value),
        }
    }
}

pub fn new_transcript<F: PrimeField64>(hash_id: &str) -> TranscriptDyn<F> {
    match hash_id {
        "Poseidon1" => TranscriptDyn::Poseidon1(Transcript::<F, Poseidon1_16>::new()),
        "Poseidon2" => TranscriptDyn::Poseidon2(Transcript::<F, Poseidon2_16>::new()),
        other => panic!("Unknown hash family: {other:?}"),
    }
}

pub fn hash_state<F: PrimeField64>(hash_id: &str, state: &mut [F; 16]) {
    match hash_id {
        "Poseidon1" => Poseidon1_16::hash(state),
        "Poseidon2" => Poseidon2_16::hash(state),
        other => panic!("Unknown hash family: {other:?}"),
    }
}
