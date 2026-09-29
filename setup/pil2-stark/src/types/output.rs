use serde::{Deserialize, Serialize};

/// Matches the starkinfo.json format read by proofman-common StarkInfo.
/// Fields use camelCase to match JS JSON.stringify output.
/// Field order matches the golden reference exactly.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(rename_all = "camelCase")]
pub struct StarkInfoOutput {
    pub name: String,
    /// cmPolsMap entries are JSON values (not PolMapEntry structs) because
    /// the golden reference requires different field orders for Q-stage
    /// entries vs regular entries.
    pub cm_pols_map: Vec<serde_json::Value>,
    pub const_pols_map: Vec<PolMapEntry>,
    pub challenges_map: Vec<ChallengeMapEntryOutput>,
    pub publics_map: Vec<PublicMapEntry>,
    pub proof_values_map: Vec<NameStageEntry>,
    pub airgroup_values_map: Vec<NameStageEntry>,
    pub air_values_map: Vec<NameStageEntry>,
    pub map_sections_n: serde_json::Map<String, serde_json::Value>,
    pub air_id: usize,
    pub airgroup_id: usize,
    pub n_constants: usize,
    pub n_publics: usize,
    pub air_group_values: Vec<serde_json::Value>,
    pub n_stages: usize,
    pub custom_commits: Vec<serde_json::Value>,
    pub custom_commits_map: Vec<serde_json::Value>,
    pub stark_struct: StarkStructOutput,
    pub boundaries: Vec<BoundaryOutput>,
    pub opening_points: Vec<i64>,
    pub c_exp_id: usize,
    pub q_dim: usize,
    pub q_deg: usize,
    pub n_constraints: usize,
    pub n_commitments_stage1: usize,
    pub ev_map: Vec<EvMapEntry>,
    pub fri_exp_id: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub security: Option<SecurityInfo>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(rename_all = "camelCase")]
pub struct StarkStructOutput {
    pub n_bits: usize,
    pub merkle_tree_arity: usize,
    pub transcript_arity: usize,
    pub merkle_tree_custom: bool,
    #[serde(default)]
    pub last_level_verification: usize,
    pub pow_bits: usize,
    pub hash_commits: bool,
    pub n_bits_ext: usize,
    pub verification_hash_type: String,
    pub steps: Vec<StepOutput>,
    pub n_queries: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct StepOutput {
    pub n_bits: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BoundaryOutput {
    pub name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub offset_min: Option<i64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub offset_max: Option<i64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct EvMapEntry {
    #[serde(rename = "type")]
    pub entry_type: String,
    pub id: usize,
    pub prime: i64,
    pub opening_pos: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub commit_id: Option<usize>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PolMapEntry {
    pub stage: usize,
    pub name: String,
    pub dim: usize,
    pub pols_map_id: usize,
    pub stage_id: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub lengths: Option<Vec<usize>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    #[serde(rename = "imPol")]
    pub im_pol: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub exp_id: Option<usize>,
    // NOTE: stagePos is added last to match JS object insertion order (setStageInfoSymbols
    // extends the object after imPol/expId have already been added in addSymbol).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stage_pos: Option<usize>,
}

/// Challenge map entry as in the golden reference.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(rename_all = "camelCase")]
pub struct ChallengeMapEntryOutput {
    pub name: String,
    pub stage: usize,
    pub dim: usize,
    pub stage_id: usize,
}

/// Public/proofvalue/airgroupvalue/airvalue map entry.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct PublicMapEntry {
    pub name: String,
    pub stage: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub lengths: Option<Vec<usize>>,
}

/// Simple name+stage entry for proofValuesMap, airgroupValuesMap, airValuesMap.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct NameStageEntry {
    pub name: String,
    pub stage: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub lengths: Option<Vec<usize>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SecurityInfo {
    pub proximity_gap: f64,
    pub proximity_parameter: f64,
    pub regime: String,
}

pub use pil_info::types::output::{CodeEntry, CodeRef};
