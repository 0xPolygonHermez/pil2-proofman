use std::str::FromStr;
use fields::PrimeField64;
use serde::{Deserialize, Serialize};
use std::path::Path;

#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
pub enum ProofType {
    #[default]
    Basic = 0,
    Compressor,
    Recursive1,
    Recursive2,
    VadcopFinal,
    VadcopFinalCompressed,
    RecursiveF,
}

impl ProofType {
    pub fn as_usize(&self) -> usize {
        match self {
            ProofType::Basic => 0,
            ProofType::Compressor => 1,
            ProofType::Recursive1 => 2,
            ProofType::Recursive2 => 3,
            ProofType::VadcopFinal => 4,
            ProofType::VadcopFinalCompressed => 5,
            ProofType::RecursiveF => 6,
        }
    }
}

impl From<ProofType> for &'static str {
    fn from(p: ProofType) -> Self {
        match p {
            ProofType::Basic => "basic",
            ProofType::Compressor => "compressor",
            ProofType::Recursive1 => "recursive1",
            ProofType::Recursive2 => "recursive2",
            ProofType::VadcopFinal => "vadcop_final",
            ProofType::VadcopFinalCompressed => "vadcop_final_compressed",
            ProofType::RecursiveF => "recursive_f",
        }
    }
}

impl FromStr for ProofType {
    type Err = ();

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "basic" => Ok(ProofType::Basic),
            "compressor" => Ok(ProofType::Compressor),
            "recursive1" => Ok(ProofType::Recursive1),
            "recursive2" => Ok(ProofType::Recursive2),
            "vadcop_final" => Ok(ProofType::VadcopFinal),
            "vadcop_final_compressed" => Ok(ProofType::VadcopFinalCompressed),
            "recursive_f" => Ok(ProofType::RecursiveF),
            _ => Err(()),
        }
    }
}

#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct Proof<F: PrimeField64> {
    pub proof_type: ProofType,
    pub airgroup_id: usize,
    pub air_id: usize,
    pub global_idx: Option<usize>,
    pub proof: Vec<u64>,
    #[serde(skip)]
    pub circom_witness: Vec<F>,
    #[serde(skip)]
    pub n_cols: usize,
}

impl<F: PrimeField64> Proof<F> {
    pub fn new(
        proof_type: ProofType,
        airgroup_id: usize,
        air_id: usize,
        global_idx: Option<usize>,
        proof: Vec<u64>,
    ) -> Self {
        Self { proof_type, global_idx, airgroup_id, air_id, proof, circom_witness: Vec::new(), n_cols: 0 }
    }

    pub fn new_witness(
        proof_type: ProofType,
        airgroup_id: usize,
        air_id: usize,
        global_idx: Option<usize>,
        circom_witness: Vec<F>,
        n_cols: usize,
    ) -> Self {
        Self { proof_type, global_idx, airgroup_id, air_id, circom_witness, proof: Vec::new(), n_cols }
    }

    /// Serialize the proof to disk (bincode). The transient `circom_witness`
    /// buffer is skipped (see the `#[serde(skip)]` fields above).
    pub fn save(&self, path: impl AsRef<Path>) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let path = path.as_ref();

        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }

        let mut file = std::fs::File::create(path).map_err(|e| {
            std::io::Error::new(e.kind(), format!("Failed to create file for saving proof: {}: {}", path.display(), e))
        })?;

        bincode::serde::encode_into_std_write(self, &mut file, bincode::config::standard())?;
        Ok(())
    }

    /// Load a proof previously written by [`Proof::save`].
    pub fn load(path: impl AsRef<Path>) -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        let mut file = std::fs::File::open(path.as_ref()).map_err(|e| {
            std::io::Error::new(
                e.kind(),
                format!("Failed to open file for loading proof: {}: {}", path.as_ref().display(), e),
            )
        })?;
        Ok(bincode::serde::decode_from_std_read(&mut file, bincode::config::standard())?)
    }
}

#[derive(Debug, Clone, Copy)]
pub struct ProverInfo {
    pub airgroup_id: usize,
    pub air_id: usize,
    pub air_instance_id: usize,
}
