use serde::Deserialize;
use std::any::Any;
use std::collections::HashMap;
use proofman::proof_manager_config::ProverConfiguration;

#[allow(dead_code)]
#[derive(Debug, Deserialize)]
pub struct EStarkConfig {
    pub variant: String,
    pub settings: HashMap<String, EStarkSettings>,
    pub verifier: Option<EStarkVerifier>,
}

#[allow(dead_code)]
#[derive(Debug, Deserialize)]
pub struct EStarkSettings {
    pub stark_info: String,
}

#[allow(dead_code)]
#[derive(Debug, Deserialize)]
pub struct EStarkVerifier {
    settings: HashMap<String, VerifierSettings>,
}

#[derive(Debug, Deserialize)]
pub struct VerifierSettings {}

impl ProverConfiguration for EStarkConfig {
    fn variant(&self) -> &str {
        &self.variant
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}