use serde::Deserialize;
use std::fs;
use std::path::Path;
use crate::ProofmanResult;

#[derive(Clone, Deserialize, Debug)]
pub struct PublicDefinition {
    pub name: String,
    #[serde(rename = "initialPos")]
    pub initial_pos: usize,
    #[serde(rename = "nValues")]
    pub n_values: usize,
    pub chunks: [usize; 2],
    #[serde(default, rename = "verificationKey")]
    pub verification_key: bool,
}

#[derive(Clone, Deserialize, Debug)]
pub struct PublicsInfo {
    #[serde(rename = "nPublics")]
    pub n_publics: usize,
    pub definitions: Vec<PublicDefinition>,
    #[serde(default, rename = "hasProgramVK")]
    pub has_program_vk: bool,
}

impl PublicsInfo {
    pub fn from_folder(folder_path: &Path) -> ProofmanResult<Self> {
        let file_path = folder_path.join("publics_info.json");

        let file_str = fs::read_to_string(&file_path)?;

        let publics_info: PublicsInfo = serde_json::from_str(&file_str)?;

        Ok(publics_info)
    }

    pub fn get_definition(&self, name: &str) -> Option<&PublicDefinition> {
        self.definitions.iter().find(|def| def.name == name)
    }
}
