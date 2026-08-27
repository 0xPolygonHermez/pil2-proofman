// use serde_json::Value as JsonValue;
use std::collections::HashMap;
use serde::{Deserialize, Deserializer};

#[allow(dead_code)]
#[derive(Deserialize, Debug, Clone)]
pub struct Boundary {
    #[serde(rename = "name")]
    pub name: String,
    #[serde(rename = "offsetMin")]
    pub offset_min: Option<u64>,
    #[serde(rename = "offsetMax")]
    pub offset_max: Option<u64>,
}

#[allow(dead_code)]
#[derive(Deserialize, Debug, Clone, Copy)]
pub struct StepStruct {
    #[serde(rename = "nBits")]
    pub n_bits: u64,
}

#[allow(dead_code)]
#[derive(Default, Deserialize, Debug, Clone)]
pub struct SecurityInfo {
    #[serde(default, rename = "proximityParameter")]
    pub proximity_parameter: f64,
    #[serde(default, rename = "proximityGap")]
    pub proximity_gap: f64,
    pub regime: String,
}

#[allow(dead_code)]
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
    #[serde(rename = "powBits")]
    pub pow_bits: u64,
    #[serde(flatten)]
    pub low_degree_test: LowDegreeTest,
}

/// The low-degree test run on the batched DEEP polynomial `f₀`. Untagged: a STIR
/// object is recognised by its `lowDegreeTest: "STIR"` marker; anything else is FRI.
#[allow(dead_code)]
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

    /// For the FRI-only code paths (today: the whole prover and verifier): the FRI
    /// schedule, or a panic naming the path that still has to learn about STIR.
    pub fn expect_fri(&self, context: &str) -> &FriStruct {
        self.fri().unwrap_or_else(|| panic!("{context} supports FRI only, but the stark info selects STIR"))
    }
}

/// FRI schedule: `steps[i].nBits` is the log-size of round `i`'s evaluation domain.
#[allow(dead_code)]
#[derive(Default, Deserialize, Debug, Clone)]
pub struct FriStruct {
    pub steps: Vec<StepStruct>,
    #[serde(default, rename = "nQueries")]
    pub n_queries: u64,
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
#[allow(dead_code)]
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
    /// `s`, out-of-domain samples per iteration.
    #[serde(rename = "numOodSamples")]
    pub num_ood_samples: u64,
    /// `tᵢ`, shift queries into `fᵢ` (length `M`).
    #[serde(default, rename = "numQueries")]
    pub num_queries: Vec<u64>,
    /// Grinding bits on iteration `i`'s query message (length `M`).
    #[serde(default, rename = "grindingBits")]
    pub grinding_bits: Vec<u64>,
}

impl StirStruct {
    /// `M`, the number of iterations.
    pub fn num_iterations(&self) -> usize {
        self.folding_factors.len()
    }
}

#[allow(dead_code)]
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

#[allow(dead_code)]
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

#[allow(dead_code)]
#[derive(Default, Deserialize, Debug, Clone)]
enum EvMapEType {
    #[serde(rename = "cm")]
    #[default]
    Cm,
    #[serde(rename = "const")]
    Const,
    #[serde(rename = "custom")]
    Custom,
}

fn deserialize_bool_from_int<'de, D>(deserializer: D) -> Result<bool, D::Error>
where
    D: Deserializer<'de>,
{
    let value: i32 = Deserialize::deserialize(deserializer)?;
    Ok(value != 0)
}

#[allow(dead_code)]
#[derive(Default, Deserialize, Debug, Clone)]
pub struct EvMap {
    #[serde(rename = "type")]
    type_: EvMapEType,
    id: u64,
    #[serde(deserialize_with = "deserialize_bool_from_int")]
    prime: bool,
}

#[allow(dead_code)]
#[derive(Default, Deserialize, Debug, Clone)]
pub struct StarkInfo {
    #[serde(rename = "starkStruct")]
    pub stark_struct: StarkStruct,

    #[serde(default, rename = "airgroupId")]
    pub airgroup_id: u64,
    #[serde(default, rename = "airId")]
    pub air_id: u64,

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
    #[serde(rename = "evMap")]
    pub ev_map: Vec<EvMap>,

    #[serde(rename = "customCommits")]
    pub custom_commits: Vec<CustomCommits>,

    #[serde(rename = "openingPoints")]
    pub opening_points: Vec<i64>,

    #[serde(default)]
    pub boundaries: Vec<Boundary>,

    #[serde(rename = "qDeg")]
    pub q_deg: u64,
    #[serde(rename = "qDim")]
    pub q_dim: u64,

    #[serde(rename = "friExpId")]
    pub fri_exp_id: u64,
    #[serde(rename = "cExpId")]
    pub c_exp_id: u64,

    #[serde(rename = "mapSectionsN")]
    pub map_sections_n: HashMap<String, u64>,

    #[serde(default, rename = "mapOffsets")]
    pub map_offsets: HashMap<(String, bool), u64>,
    #[serde(default, rename = "mapTotalN")]
    pub map_total_n: u64,

    #[serde(default, rename = "nConstraints")]
    pub n_constraints: u64,

    pub security: SecurityInfo,
}

impl StarkInfo {
    pub fn from_json(stark_info_json: &str) -> Self {
        serde_json::from_str(stark_info_json).expect("Failed to parse JSON file")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const COMMON: &str = r#""nBits": 17, "merkleTreeArity": 2, "transcriptArity": 2, "merkleTreeCustom": true,
        "lastLevelVerification": 4, "powBits": 24, "hashCommits": true, "nBitsExt": 19, "verificationHashType": "GL""#;

    /// A FRI stark struct is what it always was: flat `steps` + `nQueries`, no marker.
    #[test]
    fn fri_stark_struct_deserializes_into_the_fri_variant() {
        let json =
            format!(r#"{{{COMMON}, "steps": [{{"nBits": 19}}, {{"nBits": 16}}, {{"nBits": 13}}], "nQueries": 106}}"#);
        let ss: StarkStruct = serde_json::from_str(&json).unwrap();
        let fri = ss.low_degree_test.expect_fri("test");
        assert_eq!(fri.n_queries, 106);
        assert_eq!(fri.steps.iter().map(|s| s.n_bits).collect::<Vec<_>>(), vec![19, 16, 13]);
        assert!(ss.low_degree_test.stir().is_none());
        assert_eq!(ss.pow_bits, 24);
    }

    /// A STIR stark struct is recognised by its marker and carries the paper's schedule.
    #[test]
    fn stir_stark_struct_deserializes_into_the_stir_variant() {
        let json = format!(
            r#"{{{COMMON}, "lowDegreeTest": "STIR", "foldingFactors": [3, 3, 3, 3, 2], "logDegrees": [17, 14, 11, 8, 5, 3],
            "logDomainSizes": [19, 18, 17, 16, 15, 14], "numOodSamples": 1, "numQueries": [106, 53, 36, 27, 22],
            "grindingBits": [24, 24, 24, 24, 24]}}"#
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
}
