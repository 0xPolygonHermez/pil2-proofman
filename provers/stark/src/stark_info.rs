// use serde_json::Value as JsonValue;
use std::collections::HashMap;
use util::{timer_start, timer_stop_and_log};
use log::debug;
use serde::{Deserialize, Deserializer};

#[allow(dead_code)]
#[derive(Deserialize)]
pub struct Boundary {
    #[serde(rename = "name")]
    pub name: String,
    #[serde(rename = "offsetMin")]
    pub offset_min: Option<u64>,
    #[serde(rename = "offsetMax")]
    pub offset_max: Option<u64>,
}

#[allow(dead_code)]
#[derive(Deserialize)]
pub struct StepStruct {
    #[serde(rename = "nBits")]
    pub n_bits: u64,
}

#[allow(dead_code)]
#[derive(Deserialize)]
pub struct StarkStruct {
    #[serde(rename = "nBits")]
    pub n_bits: u64,
    #[serde(rename = "nBitsExt")]
    pub n_bits_ext: u64,
    #[serde(rename = "nQueries")]
    pub n_queries: u64,
    #[serde(default = "default_hash_commits", rename = "hashCommits")]
    pub hash_commits: bool,
    #[serde(rename = "verificationHashType")]
    pub verification_hash_type: String,
    #[serde(default = "default_merkle_tree_arity", rename = "merkleTreeArity")]
    pub merkle_tree_arity: u64,
    #[serde(default = "default_merkle_tree_custom", rename = "merkleTreeCustom")]
    pub merkle_tree_custom: bool,
    #[serde(rename = "steps")]
    pub steps: Vec<StepStruct>,
}

#[allow(dead_code)]
#[allow(non_camel_case_types)]
#[derive(Deserialize)]
pub enum OpType {
    #[serde(rename = "const")]
    Const = 0,
    #[serde(rename = "cm")]
    Cm = 1,
    #[serde(rename = "tmp")]
    Tmp = 2,
    #[serde(rename = "public")]
    Public = 3,
    #[serde(rename = "subproofvalue")]
    SubproofValue = 4,
    #[serde(rename = "challenge")]
    Challenge = 5,
    #[serde(rename = "number")]
    Number = 6,
}

impl OpType {
    pub fn as_integer(&self) -> u32 {
        match self {
            OpType::Const => 0,
            OpType::Cm => 1,
            OpType::Tmp => 2,
            OpType::Public => 3,
            OpType::SubproofValue => 4,
            OpType::Challenge => 5,
            OpType::Number => 6,
        }
    }
}

#[derive(Deserialize)]
pub struct PolMap {
    #[serde(default)]
    pub stage: u64,
    pub name: String,
    pub dim: u64,
    #[serde(default, rename = "imPol")]
    pub im_pol: bool,
    #[serde(default, rename = "stagePos")]
    pub stage_pos: u64,
    #[serde(rename = "stageId")]
    pub stage_id: u64,
}

#[allow(dead_code)]
#[derive(Deserialize)]
enum EvMapEType {
    #[serde(rename = "cm")]
    Cm,
    #[serde(rename = "const")]
    Const,
    #[serde(rename = "q")]
    Q,
}

fn deserialize_bool_from_int<'de, D>(deserializer: D) -> Result<bool, D::Error>
where
    D: Deserializer<'de>,
{
    let value: i32 = Deserialize::deserialize(deserializer)?;
    Ok(value != 0)
}

#[allow(dead_code)]
#[derive(Deserialize)]
pub struct EvMap {
    #[serde(rename = "type")]
    type_: EvMapEType,
    id: u64,
    #[serde(deserialize_with = "deserialize_bool_from_int")]
    prime: bool,
}

#[allow(dead_code)]
#[derive(Deserialize)]
pub struct StarkInfo {
    #[serde(rename = "starkStruct")]
    pub stark_struct: StarkStruct,

    #[serde(default, rename = "subproofId")]
    pub subproof_id: u64,
    #[serde(default, rename = "airId")]
    pub air_id: u64,

    #[serde(rename = "nPublics")]
    pub n_publics: u64,
    // Default value for nSubAirValues is 0
    #[serde(default, rename = "nSubproofValues")]
    pub n_subproof_values: u64,
    #[serde(rename = "nConstants")]
    pub n_constants: u64,
    #[serde(default, rename = "nStages")]
    pub n_stages: u32,

    #[serde(rename = "cmPolsMap")]
    pub cm_pols_map: Option<Vec<PolMap>>,
    #[serde(rename = "challengesMap")]
    pub challenges_map: Option<Vec<PolMap>>,
    #[serde(rename = "evMap")]
    pub ev_map: Vec<EvMap>,

    #[serde(default = "default_opening_points", rename = "openingPoints")]
    pub opening_points: Vec<i64>,

    #[serde(default)]
    pub boundaries: Vec<Boundary>,

    #[serde(rename = "qDeg")]
    pub q_deg: u64,
    #[serde(rename = "qDim")]
    pub q_dim: u64,

    #[serde(rename = "friExpId")]
    pub fri_exp_id: u64,
    #[serde(rename = "cExpId")]
    pub c_exp_id: u64,

    #[serde(rename = "mapSectionsN")]
    pub map_sections_n: HashMap<String, u64>,

    #[serde(default, rename = "mapOffsets")]
    pub map_offsets: HashMap<(String, bool), u64>,
    #[serde(default, rename = "mapTotalN")]
    pub map_total_n: u64,
}

fn default_opening_points() -> Vec<i64> {
    vec![0, 1]
}

fn default_merkle_tree_arity() -> u64 {
    16
}

fn default_merkle_tree_custom() -> bool {
    false
}

fn default_hash_commits() -> bool {
    false
}

impl StarkInfo {
    pub fn from_json(stark_info_json: &str) -> Self {
        timer_start!(STARK_INFO_LOAD);

        debug!("strkinfo: ··· Loading StarkInfo JSON");
        let stark_info: StarkInfo = serde_json::from_str(&stark_info_json).expect("Failed to parse JSON file");

        timer_stop_and_log!(STARK_INFO_LOAD);

        stark_info
    }
}
