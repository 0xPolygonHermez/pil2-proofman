use std::path::{Path, PathBuf};

use serde::{Serialize, Deserialize};
use serde_json::Value;
use std::fs;

use crate::ProofType;
use crate::{ProofmanResult, ProofmanError};

#[derive(Clone, Deserialize)]
pub struct ProofValueMap {
    pub name: String,
    #[serde(default)]
    pub id: u64,
    #[serde(default)]
    pub stage: u64,
}
#[derive(Clone, Deserialize)]
pub struct PublicMap {
    pub name: String,
    #[serde(default)]
    pub stage: u64,
    #[serde(default)]
    pub lengths: Vec<u64>,
}

#[derive(Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "PascalCase")]
pub enum CurveType {
    None,
    EcGFp5,
    EcMasFp5,
}

#[derive(Clone, Deserialize)]
pub struct GlobalInfo {
    pub folder_path: String,
    pub name: String,
    pub airs: Vec<Vec<GlobalInfoAir>>,
    pub air_groups: Vec<String>,
    pub curve: CurveType,

    #[serde(rename = "latticeSize")]
    pub lattice_size: Option<usize>,
    #[serde(rename = "aggTypes")]
    pub agg_types: Vec<Vec<GlobalInfoAggType>>,

    #[serde(rename = "nPublics")]
    pub n_publics: usize,
    #[serde(rename = "numChallenges")]
    pub n_challenges: Vec<usize>,

    #[serde(rename = "numProofValues", default)]
    pub n_proof_values: Vec<usize>,

    #[serde(rename = "proofValuesMap")]
    pub proof_values_map: Option<Vec<ProofValueMap>>,

    #[serde(rename = "publicsMap")]
    pub publics_map: Option<Vec<PublicMap>>,

    #[serde(rename = "merkleTreeArity")]
    pub merkle_tree_arity: usize,
}

#[derive(Debug, Clone, Deserialize)]
pub struct GlobalInfoAir {
    pub name: String,

    #[serde(rename = "hasCompressor", default)]
    pub has_compressor: Option<bool>,

    pub num_rows: usize,
}

#[derive(Clone, Deserialize, Debug)]
pub struct GlobalInfoAggType {
    #[serde(rename = "aggType")]
    pub agg_type: usize,
}

#[derive(Clone, Deserialize)]
pub struct GlobalInfoStepsFRI {
    #[serde(rename = "nBits")]
    pub n_bits: usize,
}

impl GlobalInfo {
    pub fn new(proving_key_path: &Path) -> ProofmanResult<Self> {
        tracing::debug!("··· Loading GlobalInfo JSON {}", proving_key_path.display());

        Self::from_file(&proving_key_path.display().to_string())
    }

    pub fn from_file(folder_path: &String) -> ProofmanResult<Self> {
        let file_path = Path::new(folder_path).join("pilout.globalInfo.json");

        // Read the JSON file
        let global_info_json = fs::read_to_string(&file_path)?;

        // Parse the JSON into a Value
        let mut global_info_value: Value = serde_json::from_str(&global_info_json)?;

        // Add the folder_path to the JSON object
        if let Some(obj) = global_info_value.as_object_mut() {
            obj.insert("folder_path".to_string(), Value::String(folder_path.to_string()));
        } else {
            return Err(ProofmanError::InvalidConfiguration(format!("JSON is not an object: {}", file_path.display())));
        }

        // Serialize the updated JSON object back to a string
        let updated_global_info_json = serde_json::to_string(&global_info_value)?;
        // Deserialize into GlobalInfo
        let global_info: GlobalInfo = serde_json::from_str(&updated_global_info_json)?;
        Ok(global_info)
    }

    pub fn get_proving_key_path(&self) -> PathBuf {
        PathBuf::from(self.folder_path.to_string())
    }

    pub fn get_setup_path(&self, template: &str) -> PathBuf {
        let vadcop_final_setup_folder = format!("{}/{}/{}/{}", self.folder_path, self.name, template, template);
        PathBuf::from(vadcop_final_setup_folder)
    }

    pub fn get_air_setup_path(&self, airgroup_id: usize, air_id: usize, proof_type: &ProofType) -> PathBuf {
        let type_str = match proof_type {
            ProofType::Basic => "air",
            ProofType::Compressor => "compressor",
            ProofType::Recursive1 => "recursive1",
            ProofType::Recursive2 => "recursive2",
            _ => panic!(),
        };

        let air_setup_folder = match proof_type {
            ProofType::Recursive2 => {
                format!("{}/{}/{}/recursive2/recursive2", self.folder_path, self.name, self.air_groups[airgroup_id])
            }
            ProofType::Compressor | ProofType::Recursive1 => {
                format!(
                    "{}/{}/{}/airs/{}/{}/{}",
                    self.folder_path,
                    self.name,
                    self.air_groups[airgroup_id],
                    self.airs[airgroup_id][air_id].name,
                    type_str,
                    type_str,
                )
            }
            ProofType::Basic => {
                format!(
                    "{}/{}/{}/airs/{}/{}/{}",
                    self.folder_path,
                    self.name,
                    self.air_groups[airgroup_id],
                    self.airs[airgroup_id][air_id].name,
                    type_str,
                    self.get_air_name(airgroup_id, air_id),
                )
            }
            _ => panic!(),
        };

        PathBuf::from(air_setup_folder)
    }

    pub fn get_air_group_name(&self, airgroup_id: usize) -> &str {
        &self.air_groups[airgroup_id]
    }

    pub fn get_airgroup_id(&self, air_group_name: &str) -> usize {
        self.air_groups
            .iter()
            .position(|name| name == air_group_name)
            .unwrap_or_else(|| panic!("Air group '{air_group_name}' not found"))
    }

    pub fn get_air_id(&self, air_group_name: &str, air_name: &str) -> (usize, usize) {
        let airgroup_id = self
            .air_groups
            .iter()
            .position(|name| name == air_group_name)
            .unwrap_or_else(|| panic!("Air group '{air_group_name}' not found"));

        let air_id = self.airs[airgroup_id]
            .iter()
            .position(|air| air.name == air_name)
            .unwrap_or_else(|| panic!("Air '{air_name}' not found in air group '{air_group_name}'"));

        (airgroup_id, air_id)
    }

    pub fn get_air_name(&self, airgroup_id: usize, air_id: usize) -> &str {
        &self.airs[airgroup_id][air_id].name
    }

    pub fn get_air_has_compressor(&self, airgroup_id: usize, air_id: usize) -> bool {
        self.airs[airgroup_id][air_id].has_compressor.unwrap_or(false)
    }
}
