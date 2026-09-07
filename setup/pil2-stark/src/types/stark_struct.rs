use indexmap::IndexMap;
use serde::{Deserialize, Serialize};

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

    /// What the user configured for this air, and nothing else. Defaults belong to
    /// `generate_stark_struct`, which knows the hash family; filling one in here made
    /// `pow_bits` family-blind and shadowed the default below it.
    pub fn resolve(&self, airgroup_name: &str, air_name: &str) -> StarkSettings {
        self.lookup_nested(airgroup_name, air_name)
            .or_else(|| self.lookup_flat(air_name))
            .or_else(|| self.lookup_flat("default"))
            .unwrap_or_default()
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

/// Nodes the proof carries at the tree's bottom kept level. Fixing the node count rather than
/// the level count keeps what the proof carries per tree constant across arities.
pub const LAST_LEVEL_NODES: usize = 16;

/// BN128 has no `hash_family` entry, so it keeps the grinding `resolve` used to fill in.
pub const BN128_DEFAULT_POW_BITS: usize = 16;

/// Levels to skip so the kept level holds at most `LAST_LEVEL_NODES` nodes, capped at the tree's
/// own height: every consumer subtracts this from an unsigned level count without a floor.
pub fn default_last_level_verification(arity: usize, n_bits_ext: usize) -> usize {
    if arity < 2 {
        return 0;
    }
    // The same depth every consumer computes, not an integer approximation of it: the two differ
    // for a non-power-of-two arity, and this value is what they subtract.
    let levels = crate::verifier_hashes::merkle_path_permutations(n_bits_ext as u64, arity as u64, 0);
    LAST_LEVEL_NODES.ilog(arity).min(levels as u32) as usize
}

/// Generate a StarkStruct from user settings, the air's power (nBits) and the
/// hash family, which provides the tree/transcript arity defaults (and, for
/// families whose kernels support a single geometry, fixes them).
/// The FRI schedule that costs the in-circuit verifier the least, found exactly instead of by
/// folding at a fixed rate.
///
/// A committed step at `b` bits costs one compression per Merkle level it keeps -- `depth(b) - llv`
/// of them -- and a query opens a folding group of `2^d` extension elements, which is
/// `ceil(2^d * 3 / block)` compressions. **Path cost is linear in the bits, leaf cost exponential in
/// the fold**, so the cheapest schedule folds hard where the bits are high and gently near the end.
/// A uniform folding factor cannot express that shape.
///
/// At nBitsExt 21, arity 2, llv 4 and a terminal of 7 the optimum is `21 > 17 > 13 > 10 > 7` -- folds
/// of 4, 4, 3, 3 -- against the uniform fold-3 schedule's `21 > 18 > 15 > 12 > 9 > 6`. On the
/// fibonacci recursion that is 49 compressions a query rather than 55, and one fewer FRI tree to
/// reduce to a root.
///
/// Uniform folding is the right shape when paths are cheap: at arity 4 a fold of 3 and a fold of 4
/// tie exactly. Arity 2 doubles every path, and that is what moves the optimum off the uniform
/// schedule -- which is why this is scoped to the families that force a binary tree, and why
/// poseidon keeps the fold it was tuned with and its committed verifiers encode.
///
/// Terminating high is free: the terminal is where the walk STOPS, so folding further only buys
/// leaf cost for no path saving. The DP is free to end anywhere at or below `final_degree` and will
/// always choose `final_degree` itself.
/// One FRI schedule and what it costs: `((per-query compressions, trees), steps)`.
type Walk = ((usize, usize), Vec<usize>);

fn optimal_fri_steps(n_bits_ext: usize, final_degree: usize, llv: usize, arity: usize, hash: &str) -> Vec<usize> {
    const FE: usize = 3;
    let block = proofman_common::hash_family::compression_block_elements(hash);
    let log_arity = (arity as f64).log2().round() as usize;
    let depth = |b: usize| b.div_ceil(log_arity.max(1));
    let path = |b: usize| depth(b).saturating_sub(llv);
    let leaf = |d: u32| ((1usize << d) * FE).div_ceil(block);

    // best[b] = (cost, steps) for the walk from b down to a terminal. Cost is compared as
    // (per-query compressions, number of trees): the tree count only breaks ties, since a tree's
    // root reduction is a fixed handful of hashes against thousands per query.
    let mut best: Vec<Option<Walk>> = vec![None; n_bits_ext + 1];
    for b in 0..=n_bits_ext {
        let mut cur: Option<Walk> = if b <= final_degree { Some(((0, 0), vec![b])) } else { None };
        for d in 1..=b {
            let nb = b - d;
            let Some((sub_cost, sub_steps)) = best[nb].clone() else { continue };
            let cost = (sub_cost.0 + leaf(d as u32) + path(nb), sub_cost.1 + 1);
            if cur.as_ref().is_none_or(|(c, _)| cost < *c) {
                let mut steps = vec![b];
                steps.extend(sub_steps);
                cur = Some((cost, steps));
            }
        }
        best[b] = cur;
    }
    best[n_bits_ext].clone().expect("a FRI schedule always exists: folding by 1 reaches any terminal").1
}

pub fn generate_stark_struct(settings: &StarkSettings, n_bits: usize, hash: &str) -> StarkStruct {
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
            let pb = settings.pow_bits.unwrap_or(BN128_DEFAULT_POW_BITS);
            let llv = settings.last_level_verification.unwrap_or(0);
            (mta, mta, mtc, false, llv, pb)
        } else {
            let family_arity = proofman_common::hash_family::merkle_tree_arity(hash) as usize;
            let mta = if proofman_common::hash_family::has_forced_tree_geometry(hash) {
                if let Some(requested) = settings.merkle_tree_arity {
                    if requested != family_arity {
                        panic!("{hash} kernels only support merkle tree arity {family_arity}, settings request {requested}");
                    }
                }
                family_arity
            } else {
                settings.merkle_tree_arity.unwrap_or(family_arity)
            };
            // Grinding bits are what a family can afford to search for, not a constant: see
            // hash_family::default_grinding_bits. They come straight off the query count.
            let pb = settings.pow_bits.unwrap_or_else(|| proofman_common::hash_family::default_grinding_bits(hash));
            // Same 16-node bottom level whatever the arity, so switching hash family does not
            // silently change how deep every Merkle path is walked.
            let llv = settings
                .last_level_verification
                .unwrap_or_else(|| default_last_level_verification(mta, n_bits + blowup_factor));
            (mta, proofman_common::hash_family::transcript_arity(hash) as usize, true, true, llv, pb)
        };

    let n_bits_ext = n_bits + blowup_factor;

    let steps: Vec<StarkStep> = if proofman_common::hash_family::uses_optimal_fri_schedule(hash) {
        optimal_fri_steps(n_bits_ext, final_degree, last_level_verification, merkle_tree_arity, hash)
            .into_iter()
            .map(|n_bits| StarkStep { n_bits })
            .collect()
    } else {
        let mut steps = vec![StarkStep { n_bits: n_bits_ext }];
        let mut fri_step_bits = n_bits_ext;
        while fri_step_bits > final_degree + 1 {
            fri_step_bits = if fri_step_bits > folding_factor + final_degree {
                fri_step_bits - folding_factor
            } else {
                final_degree
            };
            steps.push(StarkStep { n_bits: fri_step_bits });
        }
        steps
    };

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
mod optimal_fri_tests {
    use super::optimal_fri_steps;

    /// The schedule the cost model picks at the recursion's own size. Folds of 4, 4, 3, 3 -- hard
    /// where the bits are high, gently near the end -- against the uniform fold-3 schedule's
    /// 21 > 18 > 15 > 12 > 9 > 6. Pinned because it is a proof format: change it and every
    /// generated verifier changes with it.
    #[test]
    fn blake3_recursion_schedule_is_the_measured_optimum() {
        assert_eq!(optimal_fri_steps(21, 7, 4, 2, "blake3"), vec![21, 17, 13, 10, 7]);
    }

    /// Terminating high is free, so the solver always ends exactly at the ceiling rather than
    /// folding past it for no path saving.
    #[test]
    fn the_walk_stops_at_the_terminal_it_is_given() {
        for ext in 12..=24 {
            for terminal in [5usize, 7] {
                let s = optimal_fri_steps(ext, terminal, 4, 2, "blake3");
                assert_eq!(s[0], ext, "the committed domain opens the schedule");
                assert_eq!(*s.last().unwrap(), terminal, "ext {ext}, terminal {terminal}");
                assert!(s.windows(2).all(|w| w[0] > w[1]), "steps must strictly decrease: {s:?}");
            }
        }
    }

    /// It has to actually beat the uniform fold it replaces, at every size the pipeline builds --
    /// otherwise the family scoping is buying nothing.
    #[test]
    fn it_never_loses_to_a_uniform_fold() {
        const FE: usize = 3;
        let cost = |steps: &[usize]| -> usize {
            steps.windows(2).map(|w| ((1usize << (w[0] - w[1])) * FE).div_ceil(8) + w[1].saturating_sub(4)).sum()
        };
        for ext in 12..=24 {
            let solved = optimal_fri_steps(ext, 7, 4, 2, "blake3");
            for fold in 2..=6usize {
                let mut uniform = vec![ext];
                let mut b = ext;
                while b > 8 {
                    b = if b > fold + 7 { b - fold } else { 7 };
                    uniform.push(b);
                }
                if *uniform.last().unwrap() != 7 {
                    continue;
                }
                assert!(
                    cost(&solved) <= cost(&uniform),
                    "ext {ext}: solved {solved:?} ({}) lost to uniform fold {fold} {uniform:?} ({})",
                    cost(&solved),
                    cost(&uniform)
                );
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The default is a fixed bottom-level *size*, so every arity carries the same 16 nodes
    /// (64 field elements) and walks its paths to the same place. Levels differ; bytes do not.
    #[test]
    fn the_default_llv_keeps_sixteen_nodes_at_every_arity() {
        for (arity, levels) in [(2, 4), (3, 2), (4, 2), (8, 1), (16, 1)] {
            assert_eq!(default_last_level_verification(arity, 22), levels, "arity {arity}");
            assert!(arity.pow(levels as u32) <= LAST_LEVEL_NODES, "arity {arity} keeps too many nodes");
        }
        assert_eq!(2usize.pow(4), LAST_LEVEL_NODES);
        assert_eq!(4usize.pow(2), LAST_LEVEL_NODES);
    }

    /// Every consumer computes `levels - llv` in unsigned arithmetic with no floor, so a tree
    /// shorter than the default must clamp here rather than underflow there.
    #[test]
    fn the_default_llv_never_exceeds_the_tree_height() {
        for arity in [2usize, 3, 4, 8, 16] {
            for n_bits_ext in 0..8usize {
                let llv = default_last_level_verification(arity, n_bits_ext);
                // Measured with the consumers' own formula, not the one under test.
                let levels =
                    crate::verifier_hashes::merkle_path_permutations(n_bits_ext as u64, arity as u64, 0) as usize;
                assert!(llv <= levels, "arity {arity}, n_bits_ext {n_bits_ext}: llv {llv} > levels {levels}");
            }
        }
        assert_eq!(default_last_level_verification(2, 3), 3);
    }

    /// blake3 hashes cheaply enough to grind 8 bits further than Poseidon, which is 8 bits the
    /// query count does not have to pay for.
    #[test]
    fn blake3_defaults_to_more_grinding_than_poseidon() {
        let settings = StarkSettings::default();
        assert_eq!(generate_stark_struct(&settings, 20, "blake3").pow_bits, 24);
        assert_eq!(generate_stark_struct(&settings, 20, "Poseidon2").pow_bits, 16);
    }

    /// A binary tree gets 4 levels rather than the 2 a quaternary one gets, which is the whole
    /// point: blake3 paths are twice as deep, so a fixed level count would charge it twice over.
    #[test]
    fn a_binary_tree_defaults_to_four_levels() {
        let settings = StarkSettings::default();
        let ss = generate_stark_struct(&settings, 20, "blake3");
        assert_eq!(ss.merkle_tree_arity, 2);
        assert_eq!(ss.last_level_verification, 4);
    }

    /// An explicit setting still wins.
    #[test]
    fn an_explicit_llv_overrides_the_default() {
        let settings = StarkSettings { last_level_verification: Some(1), ..Default::default() };
        assert_eq!(generate_stark_struct(&settings, 20, "blake3").last_level_verification, 1);
    }

    #[test]
    fn test_generate_stark_struct_defaults() {
        let settings = StarkSettings::default();
        let ss = generate_stark_struct(&settings, 20, "Poseidon1");

        assert_eq!(ss.n_bits, 20);
        assert_eq!(ss.n_bits_ext, 21); // 20 + 1 (default blowup)
        assert_eq!(ss.verification_hash_type, "GL");
        assert_eq!(ss.merkle_tree_arity, 4); // Poseidon family default
        assert_eq!(ss.transcript_arity, 4);
        assert!(ss.merkle_tree_custom);
        assert!(ss.hash_commits);
        // Grinding is per family: Poseidon spends 16 bits, blake3 24 (see hash_family).
        assert_eq!(ss.pow_bits, 16);
        // Poseidon is arity 4, so a 16-node bottom level is 2 levels.
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
        let ss = generate_stark_struct(&settings, 16, "Poseidon1");

        assert_eq!(ss.n_bits, 16);
        assert_eq!(ss.n_bits_ext, 18);
        assert_eq!(ss.verification_hash_type, "BN128");
        assert_eq!(ss.merkle_tree_arity, 16);
        assert_eq!(ss.transcript_arity, 16);
        assert!(!ss.merkle_tree_custom);
        assert!(!ss.hash_commits);
        // BN128 has no hash_family entry, so it keeps the grinding `resolve` used to fill in.
        assert_eq!(ss.pow_bits, BN128_DEFAULT_POW_BITS);
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
        let ss = generate_stark_struct(&settings, 20, "Poseidon1");

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
        generate_stark_struct(&settings, 10, "Poseidon1");
    }

    #[test]
    fn test_blake3_forces_binary_geometry() {
        let ss = generate_stark_struct(&StarkSettings::default(), 20, "blake3");
        assert_eq!(ss.merkle_tree_arity, 2);
        assert_eq!(ss.transcript_arity, 2);
        assert!(ss.merkle_tree_custom); // GL value; stored by GL trees/transcripts but only consumed on the BN128 path
    }

    #[test]
    #[should_panic(expected = "only support merkle tree arity")]
    fn test_blake3_rejects_conflicting_arity_setting() {
        let settings = StarkSettings { merkle_tree_arity: Some(4), ..Default::default() };
        generate_stark_struct(&settings, 20, "blake3");
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
        assert_eq!(generate_stark_struct(&pos, 20, "Poseidon1").n_bits_ext, 22); // 20 + 2

        let kec = cfg.resolve("Zisk", "Keccakf");
        assert_eq!(kec.pow_bits, Some(23));
        assert!(cfg.has_compressor("Zisk", "Keccakf"));

        // Wrong airgroup -> no match -> nothing configured; generate_stark_struct supplies the
        // defaults, including the family's grinding bits.
        let miss = cfg.resolve("OtherGroup", "Poseidon2");
        assert_eq!(miss.blowup_factor, None);
        assert_eq!(miss.pow_bits, None);
        assert_eq!(generate_stark_struct(&miss, 20, "Poseidon2").pow_bits, 16);
        assert_eq!(generate_stark_struct(&miss, 20, "blake3").pow_bits, 24);
        assert_eq!(generate_stark_struct(&miss, 20, "Poseidon1").n_bits_ext, 21);
        // 20 + 1
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
        assert_eq!(s.pow_bits, None, "resolve reports config, not defaults");
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
