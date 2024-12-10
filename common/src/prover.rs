use std::os::raw::c_void;
use std::sync::Arc;

use p3_field::Field;
use transcript::FFITranscript;

use crate::ProofCtx;
use crate::SetupCtx;

#[derive(Debug, PartialEq)]
pub enum ProverStatus {
    CommitStage,
    OpeningStage,
    StagesCompleted,
}
#[derive(Debug, Clone, PartialEq)]
pub enum ProofType {
    Basic,
    Compressor,
    Recursive1,
    Recursive2,
    VadcopFinal,
    RecursiveF,
}

pub struct ProverInfo {
    pub airgroup_id: usize,
    pub air_id: usize,
    pub prover_idx: usize,
    pub instance_id: usize,
}

#[derive(Debug, Clone, Copy)]
#[repr(C)]
pub struct ConstraintRowInfo {
    pub row: u64,
    pub dim: u64,
    pub value: [u64; 3usize],
}

#[derive(Debug, Clone, Copy)]
#[repr(C)]
pub struct ConstraintInfo {
    pub id: u64,
    pub stage: u64,
    pub im_pol: bool,
    pub line: *mut u8,
    pub line_size: u64,
    pub n_rows: u64,
    pub rows: [ConstraintRowInfo; 10usize],
}

#[derive(Debug, Clone, Copy)]
#[repr(C)]
pub struct ConstraintsResults {
    pub n_constraints: u64,
    pub constraints_info: *mut ConstraintInfo,
}

#[derive(Debug, Clone, Copy)]
#[repr(C)]
pub struct GlobalConstraintInfo {
    pub id: u64,
    pub dim: u64,
    pub valid: bool,
    pub value: [u64; 3usize],
    pub line: *mut u8,
    pub line_size: u64,
}

#[derive(Debug, Clone, Copy)]
#[repr(C)]
pub struct GlobalConstraintsResults {
    pub n_constraints: u64,
    pub constraints_info: *mut GlobalConstraintInfo,
}

pub trait Prover<F: Field> {
    fn build(&mut self, proof_ctx: Arc<ProofCtx<F>>);
    fn free(&mut self);
    fn new_transcript(&self) -> FFITranscript;
    fn num_stages(&self) -> u32;
    fn get_challenges(&self, stage_id: u32, proof_ctx: Arc<ProofCtx<F>>, transcript: &FFITranscript);
    fn calculate_stage(&mut self, stage_id: u32, setup_ctx: Arc<SetupCtx>, proof_ctx: Arc<ProofCtx<F>>);
    fn commit_stage(&mut self, stage_id: u32, proof_ctx: Arc<ProofCtx<F>>) -> ProverStatus;
    fn calculate_xdivxsub(&mut self, proof_ctx: Arc<ProofCtx<F>>);
    fn calculate_lev(&mut self, proof_ctx: Arc<ProofCtx<F>>);
    fn opening_stage(&mut self, opening_id: u32, setup_ctx: Arc<SetupCtx>, proof_ctx: Arc<ProofCtx<F>>)
        -> ProverStatus;

    fn get_buff_helper_size(&self) -> usize;
    fn get_proof(&self) -> *mut c_void;
    fn get_prover_info(&self) -> ProverInfo;
    fn get_zkin_proof(&self, proof_ctx: Arc<ProofCtx<F>>, output_dir: &str) -> *mut c_void;

    fn get_transcript_values(&self, stage: u64, proof_ctx: Arc<ProofCtx<F>>) -> Vec<F>;
    fn get_transcript_values_u64(&self, stage: u64, proof_ctx: Arc<ProofCtx<F>>) -> Vec<u64>;
    fn calculate_hash(&self, values: Vec<F>) -> Vec<F>;
    fn verify_constraints(&self, setup_ctx: Arc<SetupCtx>, proof_ctx: Arc<ProofCtx<F>>) -> Vec<ConstraintInfo>;

    fn get_proof_challenges(&self, global_steps: Vec<usize>, global_challenges: Vec<F>) -> Vec<F>;
}
