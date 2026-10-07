use alloc::format;
use alloc::string::{String, ToString};
use alloc::vec::Vec;

use proofman_fields::{Goldilocks, PrimeField64};
use serde::{Deserialize, Serialize};

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

    /// The wire encoding: bincode `legacy()` layout. `save` writes exactly these bytes.
    ///
    /// `proof: Vec<u64>`, `public_values: Vec<u64>`, `compressed: bool`, `hash: String`, in
    /// that order; each length a little-endian u64, each element 8 little-endian bytes, the
    /// bool one byte, the string UTF-8.
    pub fn to_wire_bytes(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(8 * (3 + self.proof.len() + self.public_values.len()) + 1 + self.hash.len());
        for words in [&self.proof, &self.public_values] {
            out.extend_from_slice(&(words.len() as u64).to_le_bytes());
            out.extend(words.iter().flat_map(|w| w.to_le_bytes()));
        }
        out.push(self.compressed as u8);
        out.extend_from_slice(&(self.hash.len() as u64).to_le_bytes());
        out.extend_from_slice(self.hash.as_bytes());
        out
    }

    /// Decode [`to_wire_bytes`](Self::to_wire_bytes) output.
    ///
    /// Every length is checked against the bytes actually left before allocating, so an
    /// input never allocates more than its own size. Trailing bytes and non-canonical
    /// publics are rejected.
    pub fn from_wire_bytes(bytes: &[u8]) -> Result<Self, WireError> {
        let mut r = WireReader(bytes);
        let proof = r.words()?;
        let public_values = r.words()?;
        let compressed = match r.take(1)?[0] {
            0 => false,
            1 => true,
            b => return Err(WireError::InvalidBool(b)),
        };
        let n = r.len(1)?;
        let hash = core::str::from_utf8(r.take(n)?).map_err(|_| WireError::InvalidUtf8)?.to_string();
        if !r.0.is_empty() {
            return Err(WireError::TrailingBytes(r.0.len()));
        }
        let proof = Self { proof, public_values, compressed, hash };
        if let Some(index) = proof.public_values.iter().position(|&w| w >= Goldilocks::ORDER_U64) {
            return Err(WireError::NonCanonicalPublic { index });
        }
        Ok(proof)
    }

    /// Writes [`to_wire_bytes`](Self::to_wire_bytes).
    #[cfg(feature = "std")]
    pub fn save(&self, path: impl AsRef<Path>) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let path = path.as_ref();
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        std::fs::write(path, self.to_wire_bytes()).map_err(|e| {
            std::io::Error::new(e.kind(), format!("Failed to write Vadcop Final proof: {}: {}", path.display(), e))
        })?;
        Ok(())
    }

    /// Reads a file written by [`save`](Self::save), through [`from_wire_bytes`](Self::from_wire_bytes).
    #[cfg(feature = "std")]
    pub fn load(path: impl AsRef<Path>) -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        let path = path.as_ref();
        let data = std::fs::read(path).map_err(|e| {
            std::io::Error::new(e.kind(), format!("Failed to read proof file: {}: {}", path.display(), e))
        })?;
        Ok(Self::from_wire_bytes(&data).map_err(|e| format!("{}: {e}", path.display()))?)
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

/// Why [`VadcopFinalProof::from_wire_bytes`] rejected its input.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum WireError {
    /// The input ends before a field it declares.
    Truncated,
    /// Bytes left over after a complete proof.
    TrailingBytes(usize),
    /// The `compressed` byte is neither 0 nor 1.
    InvalidBool(u8),
    /// The hash family name is not UTF-8.
    InvalidUtf8,
    /// A public is not a canonical Goldilocks element.
    NonCanonicalPublic { index: usize },
}

impl core::fmt::Display for WireError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Truncated => write!(f, "input ends before the proof does"),
            Self::TrailingBytes(n) => write!(f, "{n} trailing bytes after the proof"),
            Self::InvalidBool(b) => write!(f, "compressed byte {b} is not 0 or 1"),
            Self::InvalidUtf8 => write!(f, "hash family is not UTF-8"),
            Self::NonCanonicalPublic { index } => write!(f, "public {index} is not a canonical Goldilocks element"),
        }
    }
}

impl core::error::Error for WireError {}

struct WireReader<'a>(&'a [u8]);

impl<'a> WireReader<'a> {
    fn take(&mut self, n: usize) -> Result<&'a [u8], WireError> {
        if n > self.0.len() {
            return Err(WireError::Truncated);
        }
        let (head, tail) = self.0.split_at(n);
        self.0 = tail;
        Ok(head)
    }

    /// A length prefix, bounded by the bytes left rather than by what the input claims.
    fn len(&mut self, elem_size: usize) -> Result<usize, WireError> {
        let n = u64::from_le_bytes(self.take(8)?.try_into().unwrap());
        match usize::try_from(n).ok().and_then(|n| n.checked_mul(elem_size).map(|b| (n, b))) {
            Some((n, bytes)) if bytes <= self.0.len() => Ok(n),
            _ => Err(WireError::Truncated),
        }
    }

    fn words(&mut self) -> Result<Vec<u64>, WireError> {
        let n = self.len(8)?;
        Ok(self.take(n * 8)?.as_chunks::<8>().0.iter().map(|c| u64::from_le_bytes(*c)).collect())
    }
}

#[cfg(all(test, feature = "std"))]
mod tests {
    use super::*;

    fn sample() -> VadcopFinalProof {
        VadcopFinalProof::new(alloc::vec![1, 300, u64::MAX - 5], alloc::vec![4, 5], true, "Poseidon2".into())
    }

    /// The hand-written encoder is bincode `legacy()` byte for byte, so serde users of that
    /// config interoperate with `from_wire_bytes`.
    #[test]
    fn the_wire_encoding_is_bincode_legacy() {
        let proof = sample();
        let bytes = proof.to_wire_bytes();
        assert_eq!(bytes, bincode::serde::encode_to_vec(&proof, bincode::config::legacy()).unwrap());

        let back = VadcopFinalProof::from_wire_bytes(&bytes).unwrap();
        assert_eq!(back.proof, proof.proof);
        assert_eq!(back.public_values, proof.public_values);
        assert_eq!(back.compressed, proof.compressed);
        assert_eq!(back.hash, proof.hash);
    }

    #[test]
    fn save_writes_the_wire_bytes_and_load_reads_them() {
        let proof = sample();
        let path = std::env::temp_dir().join(format!("vadcop_final_proof_{}.bin", std::process::id()));
        proof.save(&path).unwrap();
        assert_eq!(std::fs::read(&path).unwrap(), proof.to_wire_bytes());
        assert_eq!(VadcopFinalProof::load(&path).unwrap().proof, proof.proof);
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn from_wire_bytes_rejects_malformed_input() {
        let bytes = sample().to_wire_bytes();
        for cut in 0..bytes.len() {
            assert!(VadcopFinalProof::from_wire_bytes(&bytes[..cut]).is_err(), "truncated at {cut}");
        }
        let mut trailing = bytes.clone();
        trailing.push(0);
        assert_eq!(VadcopFinalProof::from_wire_bytes(&trailing).unwrap_err(), WireError::TrailingBytes(1));

        // The pre-wire `standard()` encoding is not accepted.
        let standard = bincode::serde::encode_to_vec(sample(), bincode::config::standard()).unwrap();
        assert!(VadcopFinalProof::from_wire_bytes(&standard).is_err());

        let mut bad_bool = VadcopFinalProof::new(alloc::vec![], alloc::vec![], false, String::new()).to_wire_bytes();
        bad_bool[16] = 2;
        assert_eq!(VadcopFinalProof::from_wire_bytes(&bad_bool).unwrap_err(), WireError::InvalidBool(2));

        let alias = VadcopFinalProof::new(alloc::vec![], alloc::vec![0, Goldilocks::ORDER_U64], false, String::new());
        assert_eq!(
            VadcopFinalProof::from_wire_bytes(&alias.to_wire_bytes()).unwrap_err(),
            WireError::NonCanonicalPublic { index: 1 }
        );
    }

    /// A declared length beyond the input fails before anything that size is allocated.
    #[test]
    fn an_oversized_length_fails_without_allocating() {
        for n in [u64::MAX, 1 << 40, 1 << 61] {
            let mut bytes = n.to_le_bytes().to_vec();
            bytes.extend_from_slice(&[0u8; 64]);
            assert_eq!(VadcopFinalProof::from_wire_bytes(&bytes).unwrap_err(), WireError::Truncated);
        }
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
