use serde::{Deserialize, Serialize};
use crate::types::stark_struct::LowDegreeTest;

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
    #[serde(flatten)]
    pub low_degree_test: LowDegreeTest,
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

/// Code block used in expressionsinfo.json and verifierinfo.json
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CodeOutput {
    pub tmp_used: usize,
    pub code: Vec<CodeEntry>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CodeEntry {
    pub op: String,
    pub dest: CodeRef,
    pub src: Vec<CodeRef>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct CodeRef {
    #[serde(rename = "type")]
    pub ref_type: String,
    pub id: usize,
    pub dim: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prime: Option<i64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub value: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stage: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stage_id: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub commit_id: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub opening: Option<i64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub boundary_id: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub airgroup_id: Option<usize>,
    /// Original expression id, preserved when an `exp` ref is converted to
    /// `tmp` via fixExpression (matches JS `expId` property).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub exp_id: Option<usize>,
}

impl serde::Serialize for CodeRef {
    fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        use serde::ser::SerializeMap;
        let mut map = s.serialize_map(None)?;
        map.serialize_entry("type", &self.ref_type)?;
        match self.ref_type.as_str() {
            "tmp" => {
                if let Some(eid) = self.exp_id {
                    // Pattern 3: expId before id
                    map.serialize_entry("expId", &eid)?;
                    map.serialize_entry("id", &self.id)?;
                    if let Some(prime) = self.prime {
                        map.serialize_entry("prime", &prime)?;
                    }
                } else if let Some(prime) = self.prime {
                    // Pattern 2: prime before id
                    map.serialize_entry("prime", &prime)?;
                    map.serialize_entry("id", &self.id)?;
                } else {
                    // Pattern 1: just id
                    map.serialize_entry("id", &self.id)?;
                }
                map.serialize_entry("dim", &self.dim)?;
            }
            "cm" | "const" | "custom" => {
                if let Some(eid) = self.exp_id {
                    map.serialize_entry("expId", &eid)?;
                }
                map.serialize_entry("id", &self.id)?;
                map.serialize_entry("prime", &self.prime.unwrap_or(0))?;
                map.serialize_entry("dim", &self.dim)?;
                if let Some(cid) = self.commit_id {
                    map.serialize_entry("commitId", &cid)?;
                }
            }
            "number" => {
                if let Some(ref value) = self.value {
                    map.serialize_entry("value", value)?;
                }
                map.serialize_entry("dim", &self.dim)?;
            }
            "challenge" => {
                map.serialize_entry("id", &self.id)?;
                if let Some(sid) = self.stage_id {
                    map.serialize_entry("stageId", &sid)?;
                }
                map.serialize_entry("dim", &self.dim)?;
                if let Some(stage) = self.stage {
                    map.serialize_entry("stage", &stage)?;
                }
            }
            "eval" => {
                if let Some(eid) = self.exp_id {
                    map.serialize_entry("expId", &eid)?;
                }
                map.serialize_entry("id", &self.id)?;
                map.serialize_entry("dim", &self.dim)?;
                if let Some(cid) = self.commit_id {
                    map.serialize_entry("commitId", &cid)?;
                }
            }
            "public" => {
                map.serialize_entry("id", &self.id)?;
                map.serialize_entry("dim", &self.dim)?;
            }
            "proofvalue" => {
                map.serialize_entry("id", &self.id)?;
                if let Some(stage) = self.stage {
                    map.serialize_entry("stage", &stage)?;
                }
                map.serialize_entry("dim", &self.dim)?;
            }
            "airgroupvalue" | "airvalue" => {
                map.serialize_entry("id", &self.id)?;
                if let Some(stage) = self.stage {
                    map.serialize_entry("stage", &stage)?;
                }
                map.serialize_entry("dim", &self.dim)?;
                if let Some(agid) = self.airgroup_id {
                    map.serialize_entry("airgroupId", &agid)?;
                }
            }
            "xDivXSubXi" => {
                map.serialize_entry("id", &self.id)?;
                if let Some(opening) = self.opening {
                    map.serialize_entry("opening", &opening)?;
                }
                map.serialize_entry("dim", &self.dim)?;
            }
            "Zi" => {
                if let Some(bid) = self.boundary_id {
                    map.serialize_entry("boundaryId", &bid)?;
                }
                map.serialize_entry("dim", &self.dim)?;
            }
            _ => {
                map.serialize_entry("id", &self.id)?;
                map.serialize_entry("dim", &self.dim)?;
                if let Some(prime) = self.prime {
                    map.serialize_entry("prime", &prime)?;
                }
                if let Some(ref value) = self.value {
                    map.serialize_entry("value", value)?;
                }
                if let Some(stage) = self.stage {
                    map.serialize_entry("stage", &stage)?;
                }
            }
        }
        map.end()
    }
}
