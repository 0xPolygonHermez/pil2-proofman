use std::collections::HashMap;
use serde::Deserialize;

pub use proofman_verifier::{LowDegreeLayout, ProofLayout, StirParams};

#[derive(Default, Deserialize, Debug, Clone)]
pub struct SecurityInfo {
    #[serde(default, rename = "proximityParameter")]
    pub proximity_parameter: f64,
    #[serde(default, rename = "proximityGap")]
    pub proximity_gap: f64,
    pub regime: String,
}

#[derive(Default, Deserialize, Debug, Clone)]
pub struct StarkStruct {
    #[serde(rename = "nBits")]
    pub n_bits: u64,
    #[serde(rename = "nBitsExt")]
    pub n_bits_ext: u64,
    #[serde(rename = "hashCommits")]
    pub hash_commits: bool,
    #[serde(rename = "verificationHashType")]
    pub verification_hash_type: String,
    #[serde(rename = "merkleTreeArity")]
    pub merkle_tree_arity: u64,
    #[serde(rename = "transcriptArity")]
    pub transcript_arity: u64,
    #[serde(rename = "merkleTreeCustom")]
    pub merkle_tree_custom: bool,
    #[serde(default, rename = "lastLevelVerification")]
    pub last_level_verification: u64,
    #[serde(default, rename = "splitLinearHash")]
    pub split_linear_hash: bool,
    #[serde(flatten)]
    pub low_degree_test: LowDegreeTest,
}

/// The low-degree test run on the batched DEEP polynomial `f₀`. Untagged: a STIR
/// object is recognised by its `lowDegreeTest: "STIR"` marker; anything else is FRI.
#[derive(Deserialize, Debug, Clone)]
#[serde(untagged)]
pub enum LowDegreeTest {
    Stir(StirStruct),
    Fri(FriStruct),
}

impl Default for LowDegreeTest {
    fn default() -> Self {
        LowDegreeTest::Fri(FriStruct::default())
    }
}

impl LowDegreeTest {
    pub fn fri(&self) -> Option<&FriStruct> {
        match self {
            LowDegreeTest::Fri(fri) => Some(fri),
            _ => None,
        }
    }

    pub fn stir(&self) -> Option<&StirStruct> {
        match self {
            LowDegreeTest::Stir(stir) => Some(stir),
            _ => None,
        }
    }

    /// For the FRI-only code paths: the FRI schedule, or a panic naming the path that still has to
    /// learn about STIR.
    pub fn expect_fri(&self, context: &str) -> &FriStruct {
        self.fri().unwrap_or_else(|| panic!("{context} supports FRI only, but the stark info selects STIR"))
    }

    /// Upper bound on the challenges the low-degree test draws after the shared part of the
    /// protocol, used to size the challenge buffer. FRI draws one folding challenge per step;
    /// STIR draws r_fold^0, then per iteration s out-of-domain points plus r_fold and r_comb, plus
    /// one query challenge per round.
    pub fn num_ldt_challenges(&self) -> usize {
        match self {
            LowDegreeTest::Fri(fri) => fri.log_domain_sizes.len(),
            LowDegreeTest::Stir(stir) => {
                let m = stir.num_iterations();
                1 + m.saturating_sub(1) * (1 + 2) + m
            }
        }
    }
}

/// FRI schedule. FRI keeps the rate
/// constant: iteration `i` folds both the degree bound and the domain by
/// `2^{kᵢ}`; the last entries are the final polynomial's, sent in clear.
#[derive(Default, Deserialize, Debug, Clone)]
pub struct FriStruct {
    /// `kᵢ`, in bits (length `M`).
    #[serde(rename = "foldingFactors")]
    pub folding_factors: Vec<u64>,
    /// `log₂ dᵢ` (length `M+1`).
    #[serde(rename = "logDegrees")]
    pub log_degrees: Vec<u64>,
    /// `log₂|Lᵢ|` (length `M+1`).
    #[serde(rename = "logDomainSizes")]
    pub log_domain_sizes: Vec<u64>,
    /// `t`: queries into `f₀`, checked pointwise through every fold.
    #[serde(default, rename = "numQueries")]
    pub num_queries: u64,
    /// Grinding (proof-of-work) budget in bits on the query-phase challenge.
    #[serde(default, rename = "grindingBitsQueries")]
    pub grinding_bits_queries: u64,
}

/// Marker that identifies a STIR stark struct (`"lowDegreeTest": "STIR"`).
#[derive(Deserialize, Debug, Clone, Copy, Default)]
pub enum StirKind {
    #[default]
    #[serde(rename = "STIR")]
    Stir,
}

/// STIR schedule, in the notation of the paper (Construction 5.2): iteration `i`
/// folds `fᵢ` by `2^{kᵢ}` and commits `g_{i+1}` on `L_{i+1}`, of half the size.
#[derive(Default, Deserialize, Debug, Clone)]
pub struct StirStruct {
    #[serde(rename = "lowDegreeTest")]
    pub kind: StirKind,
    /// `kᵢ`, in bits (length `M`).
    #[serde(rename = "foldingFactors")]
    pub folding_factors: Vec<u64>,
    /// `log₂ dᵢ` (length `M+1`; the last is the final polynomial's degree bound).
    #[serde(rename = "logDegrees")]
    pub log_degrees: Vec<u64>,
    /// `log₂|Lᵢ|` (length `M+1`).
    #[serde(rename = "logDomainSizes")]
    pub log_domain_sizes: Vec<u64>,
    /// `tᵢ`, shift queries into `fᵢ` (length `M`).
    #[serde(default, rename = "numQueries")]
    pub num_queries: Vec<u64>,
    /// Grinding bits on iteration `i`'s query message (length `M`).
    #[serde(default, rename = "grindingBitsQueries")]
    pub grinding_bits_queries: Vec<u64>,
}

impl StirStruct {
    /// `M`, the number of iterations.
    pub fn num_iterations(&self) -> usize {
        self.folding_factors.len()
    }
}

#[allow(non_camel_case_types)]
#[derive(Deserialize)]
pub enum OpType {
    #[serde(rename = "const")]
    Const = 0,
    #[serde(rename = "cm")]
    Cm = 1,
    #[serde(rename = "tmp")]
    Tmp = 2,
    #[serde(rename = "public")]
    Public = 3,
    #[serde(rename = "airgroupvalue")]
    AirgroupValue = 4,
    #[serde(rename = "challenge")]
    Challenge = 5,
    #[serde(rename = "number")]
    Number = 6,
    #[serde(rename = "string")]
    String = 7,
}

impl OpType {
    pub fn as_integer(&self) -> u32 {
        match self {
            OpType::Const => 0,
            OpType::Cm => 1,
            OpType::Tmp => 2,
            OpType::Public => 3,
            OpType::AirgroupValue => 4,
            OpType::Challenge => 5,
            OpType::Number => 6,
            OpType::String => 7,
        }
    }
}

#[derive(Default, Debug, Clone, Deserialize)]
pub struct PolMap {
    pub name: String,
    #[serde(default)]
    pub stage: u64,
    #[serde(default)]
    pub dim: u64,
    #[serde(default, rename = "imPol")]
    pub im_pol: bool,
    #[serde(default, rename = "stagePos")]
    pub stage_pos: u64,
    #[serde(default, rename = "stageId")]
    pub stage_id: u64,
    #[serde(default)]
    pub lengths: Vec<u64>,
}

#[derive(Deserialize, Debug, Clone, Copy)]
pub struct PublicValues {
    pub idx: u64,
}

#[derive(Default, Deserialize, Debug, Clone)]
pub struct CustomCommits {
    pub name: String,
    #[serde(default, rename = "stageWidths")]
    pub stage_widths: Vec<u32>,
    #[serde(rename = "publicValues")]
    pub public_values: Vec<PublicValues>,
}

#[derive(Default, Deserialize, Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum EvMapEType {
    #[serde(rename = "cm")]
    #[default]
    Cm,
    #[serde(rename = "const")]
    Const,
    #[serde(rename = "custom")]
    Custom,
}

#[derive(Default, Deserialize, Debug, Clone)]
pub struct EvMap {
    #[serde(rename = "type")]
    pub type_: EvMapEType,
    pub id: u64,
    /// Index into `openingPoints`.
    #[serde(rename = "openingPos")]
    pub opening_pos: usize,
    #[serde(default, rename = "commitId")]
    pub commit_id: u64,
}

#[derive(Default, Deserialize, Debug, Clone)]
pub struct StarkInfo {
    #[serde(rename = "starkStruct")]
    pub stark_struct: StarkStruct,

    #[serde(default, rename = "airgroupId")]
    pub airgroup_id: u64,
    #[serde(default, rename = "airId")]
    pub air_id: u64,
    /// Set on basic airs by the setup; see `circuit_type_of`.
    #[serde(default, rename = "circuitType")]
    pub circuit_type: Option<u64>,

    #[serde(rename = "nPublics")]
    pub n_publics: u64,
    #[serde(rename = "nConstants")]
    pub n_constants: u64,
    #[serde(default, rename = "nStages")]
    pub n_stages: u32,

    #[serde(rename = "constPolsMap")]
    pub const_pols_map: Option<Vec<PolMap>>,
    #[serde(rename = "cmPolsMap")]
    pub cm_pols_map: Option<Vec<PolMap>>,
    #[serde(rename = "publicsMap")]
    pub publics_map: Option<Vec<PolMap>>,
    #[serde(rename = "customCommitsMap")]
    pub custom_commits_map: Vec<Option<Vec<PolMap>>>,
    #[serde(rename = "challengesMap")]
    pub challenges_map: Option<Vec<PolMap>>,
    #[serde(rename = "airgroupValuesMap")]
    pub airgroupvalues_map: Option<Vec<PolMap>>,
    #[serde(rename = "airValuesMap")]
    pub airvalues_map: Option<Vec<PolMap>>,
    #[serde(default, rename = "proofValuesMap")]
    pub proofvalues_map: Option<Vec<PolMap>>,
    #[serde(rename = "evMap")]
    pub ev_map: Vec<EvMap>,
    /// 0 on keys older than the field.
    #[serde(default, rename = "friBatchSize")]
    pub fri_batch_size: u64,
    /// The proof travels as a Merkle multiproof; false on keys older than the field.
    #[serde(default)]
    pub multiproof: bool,

    #[serde(rename = "customCommits")]
    pub custom_commits: Vec<CustomCommits>,

    #[serde(rename = "openingPoints")]
    pub opening_points: Vec<i64>,

    #[serde(rename = "qDeg")]
    pub q_deg: u64,
    #[serde(rename = "qDim")]
    pub q_dim: u64,

    #[serde(rename = "cExpId")]
    pub c_exp_id: u64,

    #[serde(rename = "mapSectionsN")]
    pub map_sections_n: HashMap<String, u64>,

    #[serde(default, rename = "mapTotalN")]
    pub map_total_n: u64,

    #[serde(default, rename = "nConstraints")]
    pub n_constraints: u64,

    pub security: SecurityInfo,
}

/// The starkinfo `keyVersion` this prover reads.
pub use proofman_util::KEY_VERSION;

#[derive(Debug, thiserror::Error)]
pub enum StarkInfoError {
    #[error("starkinfo keyVersion {0:?}, this prover reads {KEY_VERSION}: regenerate the setup")]
    KeyVersion(Option<u64>),
    #[error("starkinfo: {0}")]
    Json(#[from] serde_json::Error),
}

/// The only `keyVersion` gate of the Rust starkinfo readers.
pub fn check_key_version(j: &serde_json::Value) -> Result<(), StarkInfoError> {
    let v = j.get("keyVersion").and_then(serde_json::Value::as_u64);
    if v == Some(u64::from(KEY_VERSION)) {
        Ok(())
    } else {
        Err(StarkInfoError::KeyVersion(v))
    }
}

impl StarkInfo {
    /// A key of another `keyVersion` (checked before the full parse) or malformed JSON is an error.
    pub fn try_from_value(j: &serde_json::Value) -> Result<Self, StarkInfoError> {
        check_key_version(j)?;
        Ok(Self::deserialize(j)?)
    }

    pub fn try_from_json(stark_info_json: &str) -> Result<Self, StarkInfoError> {
        Self::try_from_value(&serde_json::from_str(stark_info_json)?)
    }

    /// `try_from_json`, panicking on its error.
    pub fn from_json(stark_info_json: &str) -> Self {
        Self::try_from_json(stark_info_json).unwrap_or_else(|e| panic!("{e}"))
    }

    /// The flat proof's geometry, shared by the prover, the CLI and the generated verifiers.
    pub fn proof_layout(&self) -> ProofLayout {
        let ss = &self.stark_struct;
        let ldt = match &ss.low_degree_test {
            LowDegreeTest::Stir(stir) => LowDegreeLayout::Stir(StirParams {
                folding_factors: stir.folding_factors.clone(),
                log_degrees: stir.log_degrees.clone(),
                log_domain_sizes: stir.log_domain_sizes.clone(),
                num_queries: stir.num_queries.clone(),
                grinding_bits_queries: stir.grinding_bits_queries.clone(),
                arity: ss.merkle_tree_arity,
                last_level_verification: ss.last_level_verification,
                hash_commits: ss.hash_commits,
            }),
            LowDegreeTest::Fri(fri) => {
                LowDegreeLayout::Fri { steps: fri.log_domain_sizes.clone(), n_queries: fri.num_queries }
            }
        };
        ProofLayout {
            n_stages: self.n_stages,
            n_constants: self.n_constants,
            n_evals: self.ev_map.len() as u64,
            n_bits_ext: ss.n_bits_ext,
            num_vals: (0..=self.n_stages)
                .map(|i| self.map_sections_n.get(&format!("cm{}", i + 1)).copied().unwrap_or(0))
                .collect(),
            arity: ss.merkle_tree_arity,
            last_level_verification: ss.last_level_verification,
            hash_commits: ss.hash_commits,
            ldt,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const COMMON: &str = r#""nBits": 17, "merkleTreeArity": 2, "transcriptArity": 2, "merkleTreeCustom": true,
        "lastLevelVerification": 4, "hashCommits": true, "nBitsExt": 19, "verificationHashType": "GL""#;

    /// A FRI stark struct is what it always was: flat `steps` + `nQueries`, no marker.
    #[test]
    fn fri_stark_struct_deserializes_into_the_fri_variant() {
        let json = format!(
            r#"{{{COMMON}, "foldingFactors": [3, 3], "logDegrees": [17, 14, 11], "logDomainSizes": [19, 16, 13], "numQueries": 106, "grindingBitsQueries": 24}}"#
        );
        let ss: StarkStruct = serde_json::from_str(&json).unwrap();
        let fri = ss.low_degree_test.expect_fri("test");
        assert_eq!(fri.num_queries, 106);
        assert_eq!(fri.log_domain_sizes, vec![19, 16, 13]);
        assert!(ss.low_degree_test.stir().is_none());
        assert_eq!(fri.grinding_bits_queries, 24);
    }

    /// A STIR stark struct is recognised by its marker and carries the paper's schedule.
    #[test]
    fn stir_stark_struct_deserializes_into_the_stir_variant() {
        let json = format!(
            r#"{{{COMMON}, "lowDegreeTest": "STIR", "foldingFactors": [3, 3, 3, 3, 2], "logDegrees": [17, 14, 11, 8, 5, 3],
            "logDomainSizes": [19, 18, 17, 16, 15, 14], "numOodSamples": 1, "numQueries": [106, 53, 36, 27, 22],
            "grindingBitsQueries": [24, 24, 24, 24, 24]}}"#
        );
        let ss: StarkStruct = serde_json::from_str(&json).unwrap();
        let stir = ss.low_degree_test.stir().expect("STIR variant");
        assert_eq!(stir.num_iterations(), 5);
        assert_eq!(stir.folding_factors, vec![3, 3, 3, 3, 2]);
        assert_eq!(stir.log_degrees, vec![17, 14, 11, 8, 5, 3]);
        assert_eq!(stir.log_domain_sizes, vec![19, 18, 17, 16, 15, 14]);
        assert_eq!(stir.num_queries, vec![106, 53, 36, 27, 22]);
        assert!(ss.low_degree_test.fri().is_none());
    }

    /// A starkinfo without the current `keyVersion` is a stale setup.
    #[test]
    #[should_panic(expected = "regenerate the setup")]
    fn old_starkinfo_without_key_version_is_rejected() {
        StarkInfo::from_json(r#"{"nPublics": 0}"#);
    }

    #[test]
    #[should_panic(expected = "regenerate the setup")]
    fn starkinfo_with_wrong_key_version_is_rejected() {
        StarkInfo::from_json(r#"{"keyVersion": 1}"#);
    }

    /// The C++ prover's constant is the same `KEY_VERSION`.
    #[test]
    fn cpp_key_version_matches() {
        let hpp = concat!(env!("CARGO_MANIFEST_DIR"), "/../pil2-stark/src/starkpil/stark_info.hpp");
        let src = std::fs::read_to_string(hpp).unwrap();
        let line = src.lines().find(|l| l.contains("STARK_INFO_KEY_VERSION =")).unwrap();
        let n = line.rsplit('=').next().unwrap().trim().trim_end_matches(';');
        assert_eq!(n.parse::<u32>().unwrap(), KEY_VERSION, "{line}");
    }
}
