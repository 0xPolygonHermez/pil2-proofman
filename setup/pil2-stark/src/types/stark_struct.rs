use proofman_starks_lib_c::GOLDILOCKS_MERKLE_TREE_ARITY;
use indexmap::IndexMap;
use serde::{Deserialize, Serialize};

const MERKLE_TREE_ARITY: usize = GOLDILOCKS_MERKLE_TREE_ARITY as usize;

/// Configuration settings provided by the user to generate a StarkStruct.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(rename_all = "camelCase")]
pub struct StarkSettings {
    #[serde(default)]
    pub verification_hash_type: Option<String>,
    #[serde(default)]
    pub hash_commits: Option<bool>,
    #[serde(default)]
    pub blowup_factor: Option<usize>,
    #[serde(default)]
    pub folding_factor: Option<usize>,
    #[serde(default)]
    pub final_degree: Option<usize>,
    #[serde(default)]
    pub merkle_tree_arity: Option<usize>,
    #[serde(default)]
    pub merkle_tree_custom: Option<bool>,
    #[serde(default)]
    pub last_level_verification: Option<usize>,
    #[serde(default)]
    pub pow_bits: Option<usize>,
    #[serde(default)]
    pub has_compressor: Option<bool>,
}

/// A single top-level entry in the starkstructs config.
///
/// The config supports two schemas, decided per top-level key:
///   * Nested  — `{ "<airgroup>": { "<air>": { ...settings... } } }`
///   * Flat    — `{ "<air>":      { ...settings... } }`  (and the special key "default")
///
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
enum ConfigEntry {
    Nested(IndexMap<String, StarkSettings>),
    Flat(StarkSettings),
}

#[derive(Debug, Clone, Default)]
pub struct StarkStructsConfig {
    entries: IndexMap<String, ConfigEntry>,
}

impl StarkStructsConfig {
    /// Parse from JSON. Accepts both nested and flat schemas (mixed is allowed).
    pub fn from_json_str(data: &str) -> serde_json::Result<Self> {
        let entries: IndexMap<String, ConfigEntry> = serde_json::from_str(data)?;
        Ok(Self { entries })
    }

    pub fn resolve(&self, airgroup_name: &str, air_name: &str) -> StarkSettings {
        let mut s = self
            .lookup_nested(airgroup_name, air_name)
            .or_else(|| self.lookup_flat(air_name))
            .or_else(|| self.lookup_flat("default"))
            .unwrap_or_default();
        if s.pow_bits.is_none() {
            s.pow_bits = Some(16);
        }
        s
    }

    fn lookup_nested(&self, airgroup_name: &str, air_name: &str) -> Option<StarkSettings> {
        match self.entries.get(airgroup_name) {
            Some(ConfigEntry::Nested(airs)) => airs.get(air_name).cloned(),
            _ => None,
        }
    }

    fn lookup_flat(&self, air_name: &str) -> Option<StarkSettings> {
        match self.entries.get(air_name) {
            Some(ConfigEntry::Flat(s)) => Some(s.clone()),
            _ => None,
        }
    }

    pub fn set_has_compressor(&mut self, air_name: &str) {
        match self.entries.get_mut(air_name) {
            Some(ConfigEntry::Flat(s)) => s.has_compressor = Some(true),
            _ => {
                self.entries.insert(
                    air_name.to_string(),
                    ConfigEntry::Flat(StarkSettings { has_compressor: Some(true), ..Default::default() }),
                );
            }
        }
    }

    pub fn has_compressor(&self, airgroup_name: &str, air_name: &str) -> bool {
        self.lookup_nested(airgroup_name, air_name)
            .or_else(|| self.lookup_flat(air_name))
            .and_then(|s| s.has_compressor)
            .unwrap_or(false)
    }
}

/// A generated stark struct describing FRI parameters for a given air.
/// Also used when loading a starkinfo.json for computation (n_queries is populated then).
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(rename_all = "camelCase")]
pub struct StarkStruct {
    pub n_bits: usize,
    pub n_bits_ext: usize,
    pub merkle_tree_arity: usize,
    pub transcript_arity: usize,
    pub merkle_tree_custom: bool,
    pub hash_commits: bool,
    pub verification_hash_type: String,
    pub last_level_verification: usize,
    pub pow_bits: usize,
    pub steps: Vec<StarkStep>,
    /// Number of FRI queries. Zero when produced by generate_stark_struct (set
    /// by pil_info via fri_security); populated when loading a starkinfo.json.
    #[serde(default)]
    pub n_queries: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct StarkStep {
    pub n_bits: usize,
}

/// Generate a StarkStruct from user settings and the air's power (nBits).
///
pub fn generate_stark_struct(settings: &StarkSettings, n_bits: usize) -> StarkStruct {
    let verification_hash_type = settings.verification_hash_type.clone().unwrap_or_else(|| "GL".to_string());

    if !["GL", "BN128"].contains(&verification_hash_type.as_str()) {
        panic!("Invalid verificationHashType: {}", verification_hash_type);
    }

    let blowup_factor = settings.blowup_factor.unwrap_or(1);
    let folding_factor = settings.folding_factor.unwrap_or(3);
    let final_degree = settings.final_degree.unwrap_or(5);

    let (merkle_tree_arity, transcript_arity, merkle_tree_custom, hash_commits, last_level_verification, pow_bits) =
        if verification_hash_type == "BN128" {
            let mta = settings.merkle_tree_arity.unwrap_or(16);
            let mtc = settings.merkle_tree_custom.unwrap_or(false);
            let pb = settings.pow_bits.unwrap_or(0);
            let llv = settings.last_level_verification.unwrap_or(0);
            (mta, mta, mtc, false, llv, pb)
        } else {
            let mta = settings.merkle_tree_arity.unwrap_or(MERKLE_TREE_ARITY);
            let pb = settings.pow_bits.unwrap_or(20);
            let llv = settings.last_level_verification.unwrap_or(2);
            (mta, MERKLE_TREE_ARITY, true, true, llv, pb)
        };

    let n_bits_ext = n_bits + blowup_factor;

    let mut steps = vec![StarkStep { n_bits: n_bits_ext }];
    let mut fri_step_bits = n_bits_ext;
    while fri_step_bits > final_degree + 1 {
        fri_step_bits =
            if fri_step_bits > folding_factor + final_degree { fri_step_bits - folding_factor } else { final_degree };
        steps.push(StarkStep { n_bits: fri_step_bits });
    }

    StarkStruct {
        n_bits,
        n_bits_ext,
        merkle_tree_arity,
        transcript_arity,
        merkle_tree_custom,
        hash_commits,
        verification_hash_type,
        last_level_verification,
        pow_bits,
        steps,
        n_queries: 0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generate_stark_struct_defaults() {
        let settings = StarkSettings::default();
        let ss = generate_stark_struct(&settings, 20);

        assert_eq!(ss.n_bits, 20);
        assert_eq!(ss.n_bits_ext, 21); // 20 + 1 (default blowup)
        assert_eq!(ss.verification_hash_type, "GL");
        assert_eq!(ss.merkle_tree_arity, MERKLE_TREE_ARITY);
        assert_eq!(ss.transcript_arity, MERKLE_TREE_ARITY);
        assert!(ss.merkle_tree_custom);
        assert!(ss.hash_commits);
        assert_eq!(ss.pow_bits, 20);
        assert_eq!(ss.last_level_verification, 2);

        // First step should be nBitsExt
        assert_eq!(ss.steps[0].n_bits, 21);
        // Last step should reach finalDegree (6): nBitsExt=21 -> 18 -> 15 -> 12 -> 9 -> 6
        assert_eq!(ss.steps.last().unwrap().n_bits, 6);
    }

    #[test]
    fn test_generate_stark_struct_bn128() {
        let settings = StarkSettings {
            verification_hash_type: Some("BN128".to_string()),
            blowup_factor: Some(2),
            folding_factor: Some(4),
            final_degree: Some(3),
            ..Default::default()
        };
        let ss = generate_stark_struct(&settings, 16);

        assert_eq!(ss.n_bits, 16);
        assert_eq!(ss.n_bits_ext, 18);
        assert_eq!(ss.verification_hash_type, "BN128");
        assert_eq!(ss.merkle_tree_arity, 16);
        assert_eq!(ss.transcript_arity, 16);
        assert!(!ss.merkle_tree_custom);
        assert!(!ss.hash_commits);
        assert_eq!(ss.pow_bits, 0);
        assert_eq!(ss.last_level_verification, 0);
        assert_eq!(ss.steps[0].n_bits, 18);
    }

    #[test]
    fn test_steps_converge_to_final_degree() {
        let settings = StarkSettings {
            blowup_factor: Some(2),
            folding_factor: Some(3),
            final_degree: Some(5),
            ..Default::default()
        };
        let ss = generate_stark_struct(&settings, 20);

        // nBitsExt = 22, folding by 3 each step: 22, 19, 16, 13, 10, 7, 5
        assert_eq!(ss.steps[0].n_bits, 22);
        let last_step = ss.steps.last().unwrap().n_bits;
        assert!(
            last_step <= settings.final_degree.unwrap() + 1,
            "Last step {} should be <= finalDegree + 1 = {}",
            last_step,
            settings.final_degree.unwrap() + 1
        );
    }

    #[test]
    #[should_panic(expected = "Invalid verificationHashType")]
    fn test_invalid_hash_type() {
        let settings = StarkSettings { verification_hash_type: Some("INVALID".to_string()), ..Default::default() };
        generate_stark_struct(&settings, 10);
    }

    #[test]
    fn test_flat_config_resolution() {
        // Flat schema: top-level keys are air names.
        let json_str = r#"{
            "Keccakf": { "powBits": 23, "lastLevelVerification": 1, "hasCompressor": true },
            "Sha256f": { "hasCompressor": true },
            "SomeAir": { "blowupFactor": 2 }
        }"#;
        let cfg = StarkStructsConfig::from_json_str(json_str).unwrap();

        // Flat lookup honors all settings regardless of the airgroup name.
        let keccak = cfg.resolve("AnyGroup", "Keccakf");
        assert_eq!(keccak.pow_bits, Some(23));
        assert_eq!(keccak.last_level_verification, Some(1));
        assert_eq!(keccak.has_compressor, Some(true));

        let some = cfg.resolve("AnyGroup", "SomeAir");
        assert_eq!(some.blowup_factor, Some(2));

        assert!(cfg.has_compressor("AnyGroup", "Keccakf"));
        assert!(cfg.has_compressor("AnyGroup", "Sha256f"));
        assert!(!cfg.has_compressor("AnyGroup", "SomeAir"));
    }

    #[test]
    fn test_nested_config_resolution() {
        // Nested schema: top-level keys are airgroup names, second level are air names.
        let json_str = r#"{
            "Zisk": {
                "Poseidon2": { "blowupFactor": 2 },
                "Keccakf": { "powBits": 23, "hasCompressor": true }
            }
        }"#;
        let cfg = StarkStructsConfig::from_json_str(json_str).unwrap();

        // Resolves only under the matching airgroup.
        let pos = cfg.resolve("Zisk", "Poseidon2");
        assert_eq!(pos.blowup_factor, Some(2));
        assert_eq!(generate_stark_struct(&pos, 20).n_bits_ext, 22); // 20 + 2

        let kec = cfg.resolve("Zisk", "Keccakf");
        assert_eq!(kec.pow_bits, Some(23));
        assert!(cfg.has_compressor("Zisk", "Keccakf"));

        // Wrong airgroup -> no match -> defaults (powBits filled to 16, blowup 1).
        let miss = cfg.resolve("OtherGroup", "Poseidon2");
        assert_eq!(miss.blowup_factor, None);
        assert_eq!(miss.pow_bits, Some(16));
        assert_eq!(generate_stark_struct(&miss, 20).n_bits_ext, 21); // 20 + 1
    }

    #[test]
    fn test_default_key_fallback() {
        let cfg = StarkStructsConfig::from_json_str(r#"{ "default": { "blowupFactor": 3 } }"#).unwrap();
        // Any unlisted air falls back to "default".
        assert_eq!(cfg.resolve("G", "Anything").blowup_factor, Some(3));
    }

    #[test]
    fn test_committed_example_configs_are_nested() {
        // Lock in backward-compat with the nested-schema config files committed in
        // pil2-components/test/special/. These are the canonical example of the
        // airgroup -> air schema. CI does not feed them to the resolver, so this is
        // their only regression coverage.
        let prods =
            concat!(env!("CARGO_MANIFEST_DIR"), "/../../pil2-components/test/special/intermediate_prods.config.json");
        if std::path::Path::new(prods).exists() {
            let data = std::fs::read_to_string(prods).unwrap();
            let cfg = StarkStructsConfig::from_json_str(&data).unwrap();
            // airgroup "Intermediates", air "ImDummyAP_24_5" -> blowupFactor 2
            assert_eq!(cfg.resolve("Intermediates", "ImDummyAP_24_5").blowup_factor, Some(2));
            assert_eq!(cfg.resolve("Intermediates", "ImDummyAP_24_9").blowup_factor, Some(3));
            // Looked up without the airgroup -> no flat entry by that name -> default.
            assert_eq!(cfg.resolve("WrongGroup", "ImDummyAP_24_5").blowup_factor, None);
        }
    }

    #[test]
    fn test_empty_object_is_harmless() {
        // An air whose settings object is empty must not panic and must resolve to defaults.
        let cfg = StarkStructsConfig::from_json_str(r#"{ "EmptyAir": {} }"#).unwrap();
        let s = cfg.resolve("G", "EmptyAir");
        assert_eq!(s.blowup_factor, None);
        assert_eq!(s.pow_bits, Some(16)); // historical default fill
        assert!(!cfg.has_compressor("G", "EmptyAir"));
    }

    #[test]
    fn test_set_has_compressor_runtime() {
        let mut cfg = StarkStructsConfig::from_json_str(r#"{ "Foo": { "blowupFactor": 2 } }"#).unwrap();
        cfg.set_has_compressor("Foo");
        assert!(cfg.has_compressor("G", "Foo"));
        // Existing settings on the flat entry are preserved.
        assert_eq!(cfg.resolve("G", "Foo").blowup_factor, Some(2));

        cfg.set_has_compressor("Bar"); // new air not previously in config
        assert!(cfg.has_compressor("G", "Bar"));
    }
}
