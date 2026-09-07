use std::path::{Path, PathBuf};

use serde::{Serialize, Deserialize};
use serde_json::Value;
use std::fs;

use crate::hash_family::{self, DEFAULT_HASH_ID};
use crate::ProofType;
use crate::{ProofmanResult, ProofmanError};

fn default_hash_id() -> String {
    DEFAULT_HASH_ID.to_string()
}

/// Arities the `recursive2` circuit can be generated for. Larger values are not known to work.
pub const VALID_AGGREGATION_ARITIES: [usize; 2] = [2, 3];

pub fn is_valid_aggregation_arity(n: usize) -> bool {
    VALID_AGGREGATION_ARITIES.contains(&n)
}

/// Arity to assume when a `globalInfo.json` carries no `aggregationArity` at all.
///
/// This is Poseidon's value, kept because it is what every key written before the field existed
/// implied. It is NOT "the default arity": blake3 aggregates at 2, and anything choosing an arity
/// for a family must call [`crate::hash_family::default_aggregation_arity`] instead. Named apart
/// from that one so a glob import cannot silently resolve to the wrong one.
pub fn fallback_aggregation_arity() -> usize {
    3
}

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

    #[serde(rename = "transcriptArity")]
    pub transcript_arity: usize,

    /// Proofs each `recursive2` circuit aggregates. Fixed at setup, read back here.
    #[serde(rename = "aggregationArity", default = "fallback_aggregation_arity")]
    pub aggregation_arity: usize,

    #[serde(default = "default_hash_id")]
    pub hash: String,

    /// Whether this proving key carries the `vadcop_final_compressed` stage.
    ///
    /// Defaults to `true`, which is what every key written before the flag existed means: the stage
    /// was unconditional then. A key that skipped it says so, and the loader honours that rather
    /// than trying to read a starkinfo that was never written.
    #[serde(rename = "hasCompressedFinal", default = "default_has_compressed_final")]
    pub has_compressed_final: bool,
}

fn default_has_compressed_final() -> bool {
    true
}

#[derive(Debug, Clone, Deserialize)]
pub struct GlobalInfoAir {
    pub name: String,

    #[serde(rename = "hasCompressor", default)]
    pub has_compressor: Option<bool>,

    pub num_rows: usize,
}

impl GlobalInfoAir {
    pub fn new(name: String) -> Self {
        Self { name, has_compressor: None, num_rows: 0 }
    }
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
        if !hash_family::is_known_family(&global_info.hash) {
            return Err(ProofmanError::InvalidConfiguration(format!(
                "unknown hash family {:?}; known: {:?}",
                global_info.hash,
                hash_family::FAMILIES
            )));
        }

        // Every proving key loads through here, so this catches an arity this build has no
        // aggregation path for before any consumer chunks proofs or derives MPI tags from it.
        if !is_valid_aggregation_arity(global_info.aggregation_arity) {
            return Err(ProofmanError::InvalidConfiguration(format!(
                "proving key has aggregationArity {}, which this build does not support; valid values: {:?}",
                global_info.aggregation_arity, VALID_AGGREGATION_ARITIES
            )));
        }

        // `hash` has a serde default, `transcriptArity` does not, and the setup writes the latter as
        // exactly `hash_family::transcript_arity(hash)`. So the two disagreeing means the family is
        // wrong -- in practice a key written before `hash` existed, which is Poseidon at arity 4 but
        // takes the default. Catch it here rather than let `set_hash_family_c` below point blake3's
        // binary-tree kernels at arity-4 trees and fail somewhere unrecognisable.
        let expected_arity = hash_family::transcript_arity(&global_info.hash) as usize;
        if global_info.transcript_arity != expected_arity {
            return Err(ProofmanError::InvalidConfiguration(format!(
                "proving key has transcriptArity {} but hash family {:?} uses {}; if the key predates \
                 the `hash` field it is not {:?} -- add the right \"hash\" to {} or rebuild the key",
                global_info.transcript_arity,
                global_info.hash,
                expected_arity,
                global_info.hash,
                file_path.display()
            )));
        }

        proofman_starks_lib_c::set_hash_family_c(&global_info.hash);

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

    pub fn get_n_airs_for_airgroup(&self, airgroup_id: usize) -> usize {
        self.airs[airgroup_id].len()
    }

    pub fn get_public_starting_pos(&self, public_name: &str) -> ProofmanResult<usize> {
        if let Some(publics_map) = &self.publics_map {
            for (pos, public) in publics_map.iter().enumerate() {
                if public.name == public_name {
                    return Ok(pos);
                }
            }
        }
        Err(ProofmanError::InvalidConfiguration(format!("Public '{}' not found in publics_map", public_name)))
    }
}

#[cfg(test)]
mod aggregation_arity_tests {
    use super::*;

    #[test]
    fn only_two_and_three_are_valid_arities() {
        assert!(is_valid_aggregation_arity(2));
        assert!(is_valid_aggregation_arity(3));
        for n in [0usize, 1, 4, 5, 16] {
            assert!(!is_valid_aggregation_arity(n), "{n} must be rejected");
        }
    }

    #[test]
    fn a_key_without_the_field_is_arity_three() {
        // A key written before the field existed. Must load as 3; Default::default() gives 0.
        let json = serde_json::json!({
            "folder_path": "", "name": "t", "airs": [[]], "air_groups": [], "curve": "None",
            "aggTypes": [], "nPublics": 0, "numChallenges": [0],
            "transcriptArity": 16
        });
        let gi: GlobalInfo = serde_json::from_value(json).unwrap();
        assert_eq!(gi.aggregation_arity, 3);
    }

    #[test]
    fn from_file_rejects_an_unsupported_arity() {
        // Covers `check_setup`, `prove_air` and `soundness`, not just the full proving path.
        let dir = std::env::temp_dir().join(format!("gi_arity_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let json = serde_json::json!({
            "name": "t", "airs": [[]], "air_groups": [], "curve": "None",
            "aggTypes": [], "nPublics": 0, "numChallenges": [0],
            "transcriptArity": 16, "aggregationArity": 4
        });
        std::fs::write(dir.join("pilout.globalInfo.json"), serde_json::to_string(&json).unwrap()).unwrap();

        let result = GlobalInfo::from_file(&dir.display().to_string());
        let Err(err) = result else { panic!("arity 4 must be rejected at load") };
        assert!(err.to_string().contains("aggregationArity 4"), "unexpected error: {err}");
        std::fs::remove_dir_all(&dir).ok();
    }

    /// A key written before the `hash` field existed is Poseidon at arity 4, but `hash` has a
    /// serde default and would take blake3. The transcriptArity it DID write contradicts that, and
    /// loading must say so instead of pointing blake3's binary-tree kernels at arity-4 trees.
    #[test]
    fn from_file_rejects_a_hash_that_contradicts_the_transcript_arity() {
        let dir = std::env::temp_dir().join(format!("gi_hash_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let json = serde_json::json!({
            "name": "t", "airs": [[]], "air_groups": [], "curve": "None",
            "aggTypes": [], "nPublics": 0, "numChallenges": [0],
            "transcriptArity": 4, "aggregationArity": 3
        });
        std::fs::write(dir.join("pilout.globalInfo.json"), serde_json::to_string(&json).unwrap()).unwrap();

        let Err(err) = GlobalInfo::from_file(&dir.display().to_string()) else {
            panic!("a key with no `hash` and Poseidon's arity must not load as the default family")
        };
        let msg = err.to_string();
        assert!(msg.contains("transcriptArity 4"), "unexpected error: {msg}");

        // The same key, with the hash it was actually built with, loads.
        let mut json = json;
        json["hash"] = serde_json::json!("Poseidon1");
        std::fs::write(dir.join("pilout.globalInfo.json"), serde_json::to_string(&json).unwrap()).unwrap();
        assert!(GlobalInfo::from_file(&dir.display().to_string()).is_ok());

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn the_field_round_trips() {
        let json = serde_json::json!({
            "folder_path": "", "name": "t", "airs": [[]], "air_groups": [], "curve": "None",
            "aggTypes": [], "nPublics": 0, "numChallenges": [0],
            "transcriptArity": 16, "aggregationArity": 2
        });
        let gi: GlobalInfo = serde_json::from_value(json).unwrap();
        assert_eq!(gi.aggregation_arity, 2);
    }
}
