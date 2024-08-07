use transcript::FFITranscript;

use crate::{AirInstanceCtx, ProofCtx};

#[derive(Debug, PartialEq)]
pub enum ProverStatus {
    CommitStage,
    OpeningStage,
    StagesCompleted,
}

pub trait Prover<F> {
    fn build(&mut self, air_instance_ctx: &mut AirInstanceCtx);
    fn new_transcript(&self) -> FFITranscript;
    fn num_stages(&self) -> u32;
    fn get_challenges(&self, stage_id: u32, proof_ctx: &mut ProofCtx<F>, transcript: &FFITranscript);
    fn commit_stage(&mut self, stage_id: u32, proof_ctx: &mut ProofCtx<F>) -> ProverStatus;
    fn opening_stage(&mut self, opening_id: u32, proof_ctx: &mut ProofCtx<F>) -> ProverStatus;

    // Returns a slice representing the root of a Merkle tree with a size of 256 bits.
    // This root can be inserted into a transcript and used to generate a new challenge.
    // Due to implementation reasons, we return a slice of 4 elements, each of 64 bits.
    fn get_commit_stage_root_challenge_256(&self, stage_id: u32) -> Option<[u64; 4]>;
    fn get_opening_stage_root_challenge_256(&self, opening_id: u32) -> Option<[u64; 4]>;
    fn add_root_challenge_256_to_transcript(&mut self, root_challenge: [u64; 4]);

    // fn get_subproof_values<T>(&self) -> Vec<T>;

    fn get_map_offsets(&self, stage: &str, is_extended: bool) -> u64;
    fn add_challenges_to_transcript(&self, stage: u64, proof_ctx: &mut ProofCtx<F>, transcript: &FFITranscript);
    fn add_publics_to_transcript(&self, proof_ctx: &mut ProofCtx<F>, transcript: &FFITranscript);
}
