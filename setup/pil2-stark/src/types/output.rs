use serde::{Deserialize, Serialize};
use crate::types::stark_struct::LowDegreeTest;

/// The starkinfo.json read by proofman-common StarkInfo; the core owns every key but starkStruct and security.
pub type StarkInfoOutput = pil_info::core::project::StarkInfoOutput<StarkStructOutput, SecurityInfo>;

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(rename_all = "camelCase")]
pub struct StarkStructOutput {
    pub n_bits: usize,
    pub merkle_tree_arity: usize,
    pub transcript_arity: usize,
    pub merkle_tree_custom: bool,
    #[serde(default)]
    pub last_level_verification: usize,
    pub hash_commits: bool,
    pub n_bits_ext: usize,
    pub verification_hash_type: String,
    #[serde(flatten)]
    pub low_degree_test: LowDegreeTest,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SecurityInfo {
    pub proximity_gap: f64,
    pub proximity_parameter: f64,
    pub regime: String,
}

pub use pil_info::types::output::{CodeEntry, CodeOutput, CodeRef};
