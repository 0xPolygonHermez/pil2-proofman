use std::fs::File;
use std::path::Path;

use bytemuck::cast_slice;

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VadcopFinalProof {
    pub proof: Vec<u8>,
    pub public_values: Vec<u8>,
    pub compressed: bool,
}

impl VadcopFinalProof {
    pub fn new(proof: &[u64], compressed: bool) -> Self {
        let n_publics = proof[0] as usize;
        let rest = &proof[1..];

        let (publics, proof_u64) = rest.split_at(n_publics);

        Self { public_values: cast_slice(publics).to_vec(), proof: cast_slice(proof_u64).to_vec(), compressed }
    }

    pub fn save(&self, dir: impl AsRef<Path>) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let file_path = dir.as_ref().join("vadcop_final_proof.bin");

        let file = File::create(&file_path).map_err(|e| {
            std::io::Error::new(
                e.kind(),
                format!("Failed to create file for saving proof: {}: {}", file_path.display(), e),
            )
        })?;
        bincode::serialize_into(file, self)?;
        Ok(())
    }

    pub fn load(path: impl AsRef<Path>) -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        let file = File::open(path.as_ref()).map_err(|e| {
            std::io::Error::new(
                e.kind(),
                format!("Failed to open file for loading proof: {}: {}", path.as_ref().display(), e),
            )
        })?;
        let proof: VadcopFinalProof = bincode::deserialize_from(file)?;
        Ok(proof)
    }

    pub fn to_u64_vec(&self) -> Vec<u64> {
        let public_values_u64: &[u64] = cast_slice(&self.public_values);
        let proof_u64: &[u64] = cast_slice(&self.proof);

        let mut result = Vec::with_capacity(1 + public_values_u64.len() + proof_u64.len());
        result.push(public_values_u64.len() as u64);
        result.extend_from_slice(public_values_u64);
        result.extend_from_slice(proof_u64);

        result
    }
}
