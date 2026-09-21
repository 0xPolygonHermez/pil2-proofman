use alloc::format;
use alloc::string::{String, ToString};
use alloc::vec::Vec;

use proofman_fields::{Goldilocks, PrimeField64};
use serde::{Deserialize, Serialize};

#[cfg(feature = "std")]
use std::fs::File;
#[cfg(feature = "std")]
use std::path::Path;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VadcopFinalProof {
    pub proof: Vec<u64>,
    pub public_values: Vec<u64>,
    pub compressed: bool,
    pub hash: String,
}

impl VadcopFinalProof {
    pub fn new(proof: Vec<u64>, public_values: Vec<u64>, compressed: bool, hash: String) -> Self {
        Self { proof, public_values, compressed, hash }
    }

    pub fn new_from_proof(proof: &[u64], compressed: bool, hash: String) -> Result<Self, String> {
        if proof.is_empty() {
            return Err("Proof slice is empty, cannot extract public count".to_string());
        }

        let n_publics = proof[0] as usize;

        if proof.len() < n_publics + 1 {
            return Err(format!(
                "Proof slice length ({}) is insufficient for {} publics (expected at least {})",
                proof.len(),
                n_publics,
                n_publics + 1
            ));
        }

        let rest = &proof[1..];
        let (publics, proof_u64) = rest.split_at(n_publics);

        Ok(Self { public_values: publics.to_vec(), proof: proof_u64.to_vec(), compressed, hash })
    }

    #[cfg(feature = "std")]
    pub fn save(&self, path: impl AsRef<Path>) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let path = path.as_ref();

        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }

        let mut file = File::create(path).map_err(|e| {
            std::io::Error::new(
                e.kind(),
                format!("Failed to create file for saving Vadcop Final proof: {}: {}", path.display(), e),
            )
        })?;

        bincode::serde::encode_into_std_write(self, &mut file, bincode::config::standard())?;
        Ok(())
    }

    #[cfg(feature = "std")]
    pub fn load(path: impl AsRef<Path>) -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        let mut file = File::open(path.as_ref()).map_err(|e| {
            std::io::Error::new(
                e.kind(),
                format!("Failed to open file for loading proof: {}: {}", path.as_ref().display(), e),
            )
        })?;
        let proof: VadcopFinalProof = bincode::serde::decode_from_std_read(&mut file, bincode::config::standard())?;
        proof.check_canonical_publics()?;
        Ok(proof)
    }

    /// Pin the publics to one encoding.
    ///
    /// Both verifiers reduce before use -- blake3 through `Goldilocks::toU64`, poseidon through the
    /// permutation's modular add -- so `x` and `x + p` verify identically while a caller reading
    /// the raw words back as outputs sees two different values. `stark_verify` rejects this in
    /// Rust; the C++ verifier does not, so an untrusted proof is checked here.
    pub fn check_canonical_publics(&self) -> Result<(), String> {
        match self.public_values.iter().position(|&word| word >= Goldilocks::ORDER_U64) {
            Some(i) => Err(format!(
                "Public {i} is not a canonical Goldilocks element: {} >= {}",
                self.public_values[i],
                Goldilocks::ORDER_U64
            )),
            None => Ok(()),
        }
    }

    pub fn proof_with_publics(&self) -> Vec<u64> {
        let mut result = Vec::with_capacity(1 + self.public_values.len() + self.proof.len());
        result.push(self.public_values.len() as u64);
        result.extend_from_slice(&self.public_values);
        result.extend_from_slice(&self.proof);

        result
    }
}

#[cfg(all(test, feature = "std"))]
mod tests {
    use super::*;

    /// Only publics below `2^32 - 1` have an alias at all, since `x + p` has to fit in a u64.
    #[test]
    fn a_public_at_or_above_the_order_is_rejected() {
        let canonical =
            VadcopFinalProof::new(alloc::vec![7], alloc::vec![0, Goldilocks::ORDER_U64 - 1], false, "Poseidon2".into());
        assert!(canonical.check_canonical_publics().is_ok());

        for alias in [Goldilocks::ORDER_U64, Goldilocks::ORDER_U64 + 1, u64::MAX] {
            let proof = VadcopFinalProof::new(alloc::vec![7], alloc::vec![0, alias], false, "Poseidon2".into());
            let err = proof.check_canonical_publics().expect_err("{alias} must be rejected");
            assert!(err.contains("Public 1"), "{err}");
        }
    }

    /// Both encode the same field element, so every derived challenge matches -- only this check
    /// tells them apart.
    #[test]
    fn the_alias_of_a_small_public_is_the_same_field_element() {
        let x = 12345u64;
        let alias = x + Goldilocks::ORDER_U64;
        assert_eq!(Goldilocks::from_u64(alias).as_canonical_u64(), x);
        assert!(VadcopFinalProof::new(alloc::vec![7], alloc::vec![alias], false, "Poseidon2".into())
            .check_canonical_publics()
            .is_err());
    }
}
