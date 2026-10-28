use serde::Deserialize;
use sha2::{Digest, Sha256};
use std::fs;
use std::path::Path;
use crate::{ProofmanError, ProofmanResult};

/// The BN254 scalar field order: the final circuit's `Bits2Num(256)` reduces the digest to it.
const BN254_ORDER: &[u8] = b"21888242871839275222246405745257275088548364400416034343698204186575808495617";

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
        Self::from_file(&folder_path.join("publics_info.json"))
    }

    pub fn from_file(file_path: &Path) -> ProofmanResult<Self> {
        let file_str = fs::read_to_string(file_path)?;
        Ok(serde_json::from_str(&file_str)?)
    }

    pub fn get_definition(&self, name: &str) -> Option<&PublicDefinition> {
        self.definitions.iter().find(|def| def.name == name)
    }

    /// The public input the final SNARK commits to: `getSha256Inputs` over `publics` and
    /// `rootc`, as a 32-byte big-endian BN254 element. Recompute it from the publics and a
    /// trusted `rootc` to bind a SNARK proof to a statement.
    ///
    /// `publics` is the vadcop_final public vector, with or without its leading
    /// `is_vadcop_final_proof` flag.
    pub fn snark_public_digest(&self, publics: &[u64], rootc: &[u64; 4]) -> ProofmanResult<[u8; 32]> {
        let mut input = self.hashed_publics(publics, true)?;
        input.extend(rootc.iter().flat_map(|w| w.to_be_bytes()));
        let digest = num_bigint::BigUint::from_bytes_be(&Sha256::digest(&input));
        let order = num_bigint::BigUint::parse_bytes(BN254_ORDER, 10).expect("valid BN254 order");
        let reduced = (digest % order).to_bytes_be();
        let mut out = [0u8; 32];
        out[32 - reduced.len()..].copy_from_slice(&reduced);
        Ok(out)
    }

    /// The non-VK publics as the final circuit hashes them: the `publicValues` bytes a
    /// Solidity verifier takes.
    pub fn solidity_public_bytes(&self, publics: &[u64]) -> ProofmanResult<Vec<u8>> {
        self.hashed_publics(publics, false)
    }

    /// The publics as `getSha256Inputs` lays them out, VK sections included or not. A
    /// single-chunk VK value is big-endian; every other chunk is little-endian, in ascending
    /// chunk order. Each value must fit its chunk, as `Num2Bits` requires.
    fn hashed_publics(&self, publics: &[u64], with_vk: bool) -> ProofmanResult<Vec<u8>> {
        let publics = if publics.len() == self.n_publics + 1 { &publics[1..] } else { publics };
        if publics.len() != self.n_publics {
            return Err(ProofmanError::InvalidConfiguration(format!(
                "{} publics, expected {}",
                publics.len(),
                self.n_publics
            )));
        }
        let mut bytes = Vec::new();
        for def in &self.definitions {
            if def.verification_key && !with_vk {
                continue;
            }
            let [n_chunks, bits] = def.chunks;
            if bits % 8 != 0 || bits == 0 || bits > 64 {
                return Err(ProofmanError::InvalidConfiguration(format!(
                    "public {} has {bits}-bit chunks; only whole bytes up to 64 bits are supported",
                    def.name
                )));
            }
            let n_bytes = bits / 8;
            let values = publics.get(def.initial_pos..def.initial_pos + def.n_values * n_chunks).ok_or_else(|| {
                ProofmanError::InvalidConfiguration(format!("public {} lies outside the publics", def.name))
            })?;
            for &value in values {
                if bits < 64 && value >> bits != 0 {
                    return Err(ProofmanError::InvalidConfiguration(format!(
                        "public {} value {value} does not fit {bits} bits",
                        def.name
                    )));
                }
                if def.verification_key && n_chunks == 1 {
                    bytes.extend_from_slice(&value.to_be_bytes()[8 - n_bytes..]);
                } else {
                    bytes.extend_from_slice(&value.to_le_bytes()[..n_bytes]);
                }
            }
        }
        Ok(bytes)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn def(name: &str, initial_pos: usize, n_values: usize, chunks: [usize; 2], vk: bool) -> PublicDefinition {
        PublicDefinition { name: name.into(), initial_pos, n_values, chunks, verification_key: vk }
    }

    /// ZisK's layout: a 4-limb program VK, then the program's outputs.
    fn vk_then_inputs(n_inputs: usize) -> PublicsInfo {
        PublicsInfo {
            n_publics: 4 + n_inputs,
            has_program_vk: true,
            definitions: vec![def("rom_root", 0, 4, [1, 64], true), def("inputs", 4, n_inputs, [1, 64], false)],
        }
    }

    /// The circuit hashes a VK limb big-endian, any other public little-endian, then rootC
    /// big-endian, and reduces the SHA-256 digest into the BN254 field.
    #[test]
    fn the_digest_hashes_vk_be_inputs_le_then_rootc_be() {
        let info = vk_then_inputs(2);
        let publics: [u64; 6] = [1, 2, 3, 4, 0x0102_0304_0506_0708, u64::MAX - 5];
        let rootc: [u64; 4] = [11, 12, 13, 14];

        let mut preimage = Vec::new();
        preimage.extend(publics[..4].iter().flat_map(|w| w.to_be_bytes()));
        preimage.extend(publics[4..].iter().flat_map(|w| w.to_le_bytes()));
        preimage.extend(rootc.iter().flat_map(|w| w.to_be_bytes()));
        let order = num_bigint::BigUint::parse_bytes(BN254_ORDER, 10).unwrap();
        let want = num_bigint::BigUint::from_bytes_be(&Sha256::digest(&preimage)) % &order;

        let got = info.snark_public_digest(&publics, &rootc).unwrap();
        assert_eq!(num_bigint::BigUint::from_bytes_be(&got), want);
        assert!(num_bigint::BigUint::from_bytes_be(&got) < order);

        // A leading is_vadcop_final_proof flag is dropped, not hashed.
        let mut flagged = vec![1];
        flagged.extend_from_slice(&publics);
        assert_eq!(info.snark_public_digest(&flagged, &rootc).unwrap(), got);
    }

    /// The Solidity bytes are the hashed publics without the VK sections.
    #[test]
    fn solidity_bytes_skip_the_vk_and_stay_little_endian() {
        let info = vk_then_inputs(2);
        let publics: [u64; 6] = [1, 2, 3, 4, 0x0102_0304_0506_0708, 0xAABB_CCDD_EEFF_0011];
        let want: Vec<u8> = publics[4..].iter().flat_map(|w| w.to_le_bytes()).collect();
        assert_eq!(info.solidity_public_bytes(&publics).unwrap(), want);
    }

    /// Sub-u64 chunks keep their low bytes in ascending chunk order, and a section is read
    /// from its `initialPos`, not from wherever the previous section ended.
    #[test]
    fn chunks_are_ascending_and_sections_start_at_their_position() {
        let info =
            PublicsInfo { n_publics: 3, has_program_vk: false, definitions: vec![def("w", 1, 1, [2, 32], false)] };
        let got = info.solidity_public_bytes(&[0xFF, 0x1122_3344, 0x5566_7788]).unwrap();
        assert_eq!(got, vec![0x44, 0x33, 0x22, 0x11, 0x88, 0x77, 0x66, 0x55]);

        // A value wider than its chunk has no circuit encoding.
        assert!(info.solidity_public_bytes(&[0, 1 << 32, 0]).is_err());
    }

    #[test]
    fn a_wrong_public_count_is_refused() {
        let info = vk_then_inputs(2);
        assert!(info.snark_public_digest(&[0; 3], &[0; 4]).is_err());
    }
}
