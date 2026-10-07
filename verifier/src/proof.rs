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

    /// Written with the legacy bincode config, the same bytes that go on the wire.
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

        bincode::serde::encode_into_std_write(self, &mut file, bincode::config::legacy())?;
        Ok(())
    }

    /// Reads both the legacy encoding written by `save` and the `standard()` encoding written
    /// by earlier versions.
    #[cfg(feature = "std")]
    pub fn load(path: impl AsRef<Path>) -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        let data = std::fs::read(path.as_ref()).map_err(|e| {
            std::io::Error::new(
                e.kind(),
                format!("Failed to open file for loading proof: {}: {}", path.as_ref().display(), e),
            )
        })?;
        let proof = Self::decode(&data)?;
        proof.check_canonical_publics()?;
        Ok(proof)
    }

    #[cfg(feature = "std")]
    fn decode(data: &[u8]) -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        // Legacy has a fixed-width layout, so it is recognised exactly before decoding; a
        // standard (varint) file only passes by coincidence and then fails to decode or to
        // consume every byte, falling through to standard.
        if has_legacy_layout(data) {
            if let Ok((proof, read)) = bincode::serde::decode_from_slice::<Self, _>(data, bincode::config::legacy()) {
                if read == data.len() {
                    return Ok(proof);
                }
            }
        }
        let (proof, read) = bincode::serde::decode_from_slice::<Self, _>(data, bincode::config::standard())?;
        if read != data.len() {
            return Err(format!("{} trailing bytes after the proof", data.len() - read).into());
        }
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

/// Whether `data` is exactly `Vec<u64>, Vec<u64>, bool, String` in the legacy fixed-width
/// layout: u64 lengths, 8-byte elements, one bool byte.
#[cfg(feature = "std")]
fn has_legacy_layout(data: &[u8]) -> bool {
    fn take_len(data: &[u8], pos: usize) -> Option<usize> {
        let end = pos.checked_add(8)?;
        usize::try_from(u64::from_le_bytes(data.get(pos..end)?.try_into().ok()?)).ok()
    }
    let mut pos = 0usize;
    for _ in 0..2 {
        let Some(n) = take_len(data, pos) else { return false };
        let Some(next) = n.checked_mul(8).and_then(|b| b.checked_add(8)).and_then(|b| pos.checked_add(b)) else {
            return false;
        };
        pos = next;
    }
    if !matches!(data.get(pos), Some(0 | 1)) {
        return false;
    }
    pos += 1;
    let Some(n) = take_len(data, pos) else { return false };
    pos.checked_add(8).and_then(|p| p.checked_add(n)) == Some(data.len())
}

#[cfg(all(test, feature = "std"))]
mod tests {
    use super::*;

    #[test]
    fn save_and_load_round_trip_the_legacy_encoding() {
        let proof = VadcopFinalProof::new(alloc::vec![1, 2, 3], alloc::vec![4, 5], true, "Poseidon2".into());
        let path = std::env::temp_dir().join(format!("vadcop_final_proof_{}.bin", std::process::id()));
        proof.save(&path).unwrap();

        let expected = bincode::serde::encode_to_vec(&proof, bincode::config::legacy()).unwrap();
        assert_eq!(std::fs::read(&path).unwrap(), expected);

        let loaded = VadcopFinalProof::load(&path).unwrap();
        assert_eq!(loaded.proof, proof.proof);
        assert_eq!(loaded.public_values, proof.public_values);
        assert_eq!(loaded.compressed, proof.compressed);
        assert_eq!(loaded.hash, proof.hash);
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn load_reads_standard_encoded_files_and_rejects_corruption() {
        let proof = VadcopFinalProof::new(alloc::vec![1, 300, u64::MAX - 5], alloc::vec![4, 5], true, "Poseidon2".into());
        let dir = std::env::temp_dir().join(format!("vadcop_compat_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let load = |name: &str, bytes: &[u8]| {
            let path = dir.join(name);
            std::fs::write(&path, bytes).unwrap();
            VadcopFinalProof::load(&path)
        };

        let standard = bincode::serde::encode_to_vec(&proof, bincode::config::standard()).unwrap();
        let legacy = bincode::serde::encode_to_vec(&proof, bincode::config::legacy()).unwrap();
        assert_ne!(standard, legacy);
        for (name, bytes) in [("standard", &standard), ("legacy", &legacy)] {
            let loaded = load(name, bytes).unwrap_or_else(|e| panic!("{name}: {e}"));
            assert_eq!(loaded.proof, proof.proof, "{name}");
            assert_eq!(loaded.public_values, proof.public_values, "{name}");
            assert!(loaded.compressed, "{name}");
            assert_eq!(loaded.hash, proof.hash, "{name}");
        }

        for (name, bytes) in [("standard", &standard), ("legacy", &legacy)] {
            let mut trailing = bytes.clone();
            trailing.push(0);
            assert!(load("trailing", &trailing).is_err(), "{name} with trailing byte");
            assert!(load("truncated", &bytes[..bytes.len() - 1]).is_err(), "{name} truncated");
        }
        assert!(load("empty", &[]).is_err());
        let _ = std::fs::remove_dir_all(&dir);
    }

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
