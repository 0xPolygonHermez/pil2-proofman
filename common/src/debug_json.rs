//! `debug.json` parsing.
//!
//! Defines the serde-derived JSON schema structs, parses them into the in-memory
//! [`crate::DebugInfo`] tree, and exposes the helpers consumers use to query the result
//! at runtime ([`skip_prover_instance`], [`store_rows_info_air`]).

use std::collections::HashMap;
use std::fs;
use std::path::PathBuf;

use fields::PrimeField64;
use serde::Deserialize;

use crate::{
    AirGroupMap, AirIdMap, BucketPrefix, BucketRange, BucketRule, BusMode, Classifier, DebugInfo, GlobalInfo,
    InstanceMap, InstancesInfo, InstancesMode, OutputConfig, ProofCtx, ProofmanError, ProofmanResult,
    DEFAULT_N_PRINT_CONSTRAINTS, DEFAULT_OUTPUT_FILE_PATH, DEFAULT_PRINT_VALS,
};

// ============================================================================
// Runtime queries against `pctx.debug_info`
// ============================================================================

pub fn skip_prover_instance<F: PrimeField64>(
    pctx: &ProofCtx<F>,
    global_idx: usize,
) -> ProofmanResult<(bool, Option<InstancesInfo>)> {
    if pctx.debug_info.read().unwrap().instances_mode != InstancesMode::OnlyListed {
        return Ok((false, None));
    }

    // Tables are included by default in only-listed mode. Opt into filtering
    // them via `"instances": { "skip_tables": true }` — useful when you don't
    // want to pay the cost of processing every lookup table and accept that
    // bus debug for lookup-heavy opids will be meaningless without targets.
    if pctx.dctx_is_table(global_idx) && !pctx.debug_info.read().unwrap().skip_tables {
        return Ok((false, None));
    }

    let (airgroup_id, air_id) = pctx.dctx_get_instance_info(global_idx)?;
    let air_instance_id = pctx.dctx_find_air_instance_id(global_idx)?;

    if let Some(airgroup_id_map) = pctx.debug_info.read().unwrap().debug_instances.get(&airgroup_id) {
        if airgroup_id_map.is_empty() {
            return Ok((false, None));
        } else if let Some(air_id_map) = airgroup_id_map.get(&air_id) {
            if air_id_map.1.is_empty() {
                return Ok((false, None));
            } else if let Some(instance_id_map) = air_id_map.1.get(&air_instance_id) {
                return Ok((false, Some(instance_id_map.clone())));
            }
        }
    }

    Ok((true, None))
}

pub fn store_rows_info_air<F: PrimeField64>(
    pctx: &ProofCtx<F>,
    airgroup_id: usize,
    air_id: usize,
    instance_id: usize,
) -> bool {
    if pctx.debug_info.read().unwrap().bus_mode.store_row_info {
        return true;
    }

    if let Some(airgroup_id_map) = pctx.debug_info.read().unwrap().debug_instances.get(&airgroup_id) {
        if let Some(air_id_map) = airgroup_id_map.get(&air_id) {
            if air_id_map.0 {
                return true;
            }

            if let Some(instance_id_map) = air_id_map.1.get(&instance_id) {
                if instance_id_map.store_row_info {
                    return true;
                }
            }
        }
    }

    false
}

fn default_fast_mode() -> bool {
    true
}

fn default_to_file() -> bool {
    true
}

// ============================================================================
// JSON deserialization structs — match the schema documented in DEBUG_JSON.md.
// ============================================================================

#[derive(Debug, Deserialize)]
struct DebugJson {
    #[serde(default)]
    instances: Option<InstancesJson>,
    #[serde(default)]
    constraints: Option<ConstraintsJson>,
    #[serde(default)]
    global_constraints: Option<GlobalConstraintsJson>,
    #[serde(default)]
    bus: Option<BusJson>,
    #[serde(default)]
    output: Option<OutputJson>,
}

#[derive(Debug, Deserialize)]
struct InstancesJson {
    #[serde(default)]
    mode: Option<InstancesModeJson>,
    #[serde(default)]
    list: Option<Vec<AirGroupJson>>,
    #[serde(default)]
    skip_tables: bool,
}

#[derive(Debug, Default, Deserialize)]
#[serde(rename_all = "snake_case")]
enum InstancesModeJson {
    #[default]
    All,
    OnlyListed,
}

fn default_true() -> bool {
    true
}

#[derive(Debug, Deserialize)]
struct ConstraintsJson {
    #[serde(default = "default_true")]
    enabled: bool,
    #[serde(default)]
    max_print: Option<usize>,
}

#[derive(Debug, Deserialize)]
struct GlobalConstraintsJson {
    #[serde(default = "default_true")]
    enabled: bool,
    #[serde(default)]
    global_constraint_ids: Option<Vec<usize>>,
}

#[derive(Debug, Default, Deserialize)]
struct BusJson {
    #[serde(default)]
    opids: Option<Vec<u64>>,
    #[serde(default)]
    max_print: Option<usize>,
    #[serde(default = "default_fast_mode")]
    fast_mode: bool,
    #[serde(default)]
    store_row_info: bool,
    #[serde(default)]
    values_filter: Option<Vec<Vec<String>>>,
    #[serde(default)]
    group_by: Option<Vec<BucketRuleJson>>,
}

#[derive(Debug, Deserialize)]
struct OutputJson {
    #[serde(default = "default_to_file")]
    to_file: bool,
    #[serde(default)]
    file_path: Option<String>,
}

#[derive(Debug, Deserialize)]
struct AirGroupJson {
    #[serde(default)]
    airgroup_id: Option<usize>,
    #[serde(default)]
    airgroup: Option<String>,
    #[serde(default)]
    airs: Option<Vec<AirIdJson>>,
}

#[derive(Debug, Deserialize)]
struct AirIdJson {
    #[serde(default)]
    air_id: Option<usize>,
    #[serde(default)]
    air: Option<String>,
    #[serde(default)]
    instances: Option<Vec<InstanceJson>>,
    #[serde(default)]
    store_row_info: Option<bool>,
}

#[derive(Debug, Deserialize)]
struct InstanceJson {
    #[serde(default)]
    instance_id: Option<usize>,
    #[serde(default)]
    constraints: Option<Vec<usize>>,
    #[serde(default)]
    store_row_info: Option<bool>,
}

#[derive(Debug, Deserialize)]
struct BucketRuleJson {
    opid: u64,
    column: usize,
    #[serde(flatten)]
    classifier: BucketClassifierJson,
}

#[derive(Debug, Deserialize)]
#[serde(tag = "by", rename_all = "snake_case")]
enum BucketClassifierJson {
    Value {
        /// Optional filter list. When present, only column values in this list are
        /// tracked; rows with other values are dropped. When absent, every distinct
        /// value gets its own bucket (no filtering).
        #[serde(default)]
        values: Option<Vec<String>>,
    },
    Range {
        #[serde(default)]
        ranges: Vec<RangeJson>,
        /// When `true`, ranges may have gaps and values falling in them are dropped.
        /// When `false` (default), ranges must cover (-∞, +∞) gap-free (parse-time error).
        #[serde(default)]
        filter: bool,
    },
    Prefix {
        #[serde(default)]
        prefixes: Vec<PrefixJson>,
        /// When `true`, values matching no prefix are dropped. Otherwise they go into an
        /// implicit "no match" catch-all bucket.
        #[serde(default)]
        filter: bool,
    },
    Step {
        start: String,
        stop: String,
        step: String,
        /// When `true`, values outside `[start, stop)` are dropped. Otherwise they go into
        /// an implicit "out of range" bucket.
        #[serde(default)]
        filter: bool,
    },
}

#[derive(Debug, Deserialize)]
struct RangeJson {
    #[serde(default)]
    min: Option<String>,
    #[serde(default)]
    max: Option<String>,
}

#[derive(Debug, Deserialize)]
struct PrefixJson {
    value: String,
    bits: u8,
}

// ============================================================================
// Bucket-rule parsing helpers
// ============================================================================

/// Parse a hex (0x-prefixed) or decimal string into a u64.
fn parse_u64_str(s: &str, field: &str) -> ProofmanResult<u64> {
    let trimmed = s.trim();
    let parsed = if let Some(rest) = trimmed.strip_prefix("0x").or_else(|| trimmed.strip_prefix("0X")) {
        u64::from_str_radix(rest, 16)
    } else {
        trimmed.parse::<u64>()
    };
    parsed.map_err(|_| ProofmanError::InvalidConfiguration(format!("Failed to parse {field} as u64: {s:?}")))
}

fn build_bucket_rule(rule: BucketRuleJson) -> ProofmanResult<BucketRule> {
    let classifier = match rule.classifier {
        BucketClassifierJson::Value { values } => {
            let parsed_values = match values {
                None => None,
                Some(v) => {
                    if v.is_empty() {
                        return Err(ProofmanError::InvalidConfiguration(format!(
                            "Bucket rule for opid {}: empty `values` list — omit `values` to track all column values, or list at least one to filter",
                            rule.opid
                        )));
                    }
                    let parsed: Result<Vec<u64>, _> =
                        v.iter().map(|s| parse_u64_str(s, "value.values entry")).collect();
                    Some(parsed?)
                }
            };
            Classifier::Value { values: parsed_values }
        }
        BucketClassifierJson::Range { ranges, filter } => {
            if ranges.is_empty() {
                return Err(ProofmanError::InvalidConfiguration(format!(
                    "Bucket rule for opid {} with by=\"range\" must have at least one entry in `ranges`",
                    rule.opid
                )));
            }
            let mut parsed: Vec<BucketRange> = Vec::with_capacity(ranges.len());
            for r in ranges {
                let min = match r.min {
                    Some(s) => Some(parse_u64_str(&s, "range.min")?),
                    None => None,
                };
                let max = match r.max {
                    Some(s) => Some(parse_u64_str(&s, "range.max")?),
                    None => None,
                };
                if let (Some(lo), Some(hi)) = (min, max) {
                    if lo >= hi {
                        return Err(ProofmanError::InvalidConfiguration(format!(
                            "Bucket rule for opid {}: range min ({lo}) must be strictly less than max ({hi})",
                            rule.opid
                        )));
                    }
                }
                parsed.push(BucketRange { min, max });
            }
            // Sort by `min` (None first = -∞). The tuple key keeps `None` strictly
            // ahead of any `Some(_)` — including `Some(0)` — which `min.unwrap_or(0)`
            // would tie and let the sort reorder, breaking the gap/overlap checks below.
            parsed.sort_by_key(|r| (r.min.is_some(), r.min.unwrap_or(0)));

            if filter {
                // Filter mode: only validate non-overlap. Gaps are allowed (values in
                // gaps get dropped at runtime).
                for i in 0..parsed.len().saturating_sub(1) {
                    let prev_max = parsed[i].max.unwrap_or(u64::MAX);
                    let next_min = parsed[i + 1].min.unwrap_or(0);
                    if prev_max > next_min {
                        return Err(ProofmanError::InvalidConfiguration(format!(
                            "Bucket rule for opid {}: ranges overlap between {prev_max} and {next_min}",
                            rule.opid
                        )));
                    }
                }
            } else {
                // Default mode: ranges must cover (-∞, +∞) gap-free.
                if parsed[0].min.is_some() {
                    return Err(ProofmanError::InvalidConfiguration(format!(
                        "Bucket rule for opid {}: ranges have a gap below the first `min`; the lowest range must omit `min` to cover -∞ (or set `filter: true` to allow gaps)",
                        rule.opid
                    )));
                }
                for i in 0..parsed.len() - 1 {
                    let prev_max = parsed[i].max.ok_or_else(|| {
                        ProofmanError::InvalidConfiguration(format!(
                            "Bucket rule for opid {}: only the last range may omit `max` (or set `filter: true` to allow gaps)",
                            rule.opid
                        ))
                    })?;
                    let next_min = parsed[i + 1].min.ok_or_else(|| {
                        ProofmanError::InvalidConfiguration(format!(
                            "Bucket rule for opid {}: only the first range may omit `min` (or set `filter: true` to allow gaps)",
                            rule.opid
                        ))
                    })?;
                    if prev_max != next_min {
                        return Err(ProofmanError::InvalidConfiguration(format!(
                            "Bucket rule for opid {}: ranges have a gap or overlap between {prev_max} and {next_min} (or set `filter: true` to allow gaps)",
                            rule.opid
                        )));
                    }
                }
                if parsed.last().unwrap().max.is_some() {
                    return Err(ProofmanError::InvalidConfiguration(format!(
                        "Bucket rule for opid {}: the highest range must omit `max` to cover +∞ (or set `filter: true` to allow gaps)",
                        rule.opid
                    )));
                }
            }
            Classifier::Range { ranges: parsed, filter }
        }
        BucketClassifierJson::Prefix { prefixes, filter } => {
            if prefixes.is_empty() {
                return Err(ProofmanError::InvalidConfiguration(format!(
                    "Bucket rule for opid {} with by=\"prefix\" must have at least one entry in `prefixes`",
                    rule.opid
                )));
            }
            let mut parsed: Vec<BucketPrefix> = Vec::with_capacity(prefixes.len());
            for p in prefixes {
                if p.bits == 0 || p.bits > 64 {
                    return Err(ProofmanError::InvalidConfiguration(format!(
                        "Bucket rule for opid {}: prefix `bits` must be in 1..=64, got {}",
                        rule.opid, p.bits
                    )));
                }
                let value = parse_u64_str(&p.value, "prefix.value")?;
                if p.bits < 64 && value >= (1u64 << p.bits) {
                    return Err(ProofmanError::InvalidConfiguration(format!(
                        "Bucket rule for opid {}: prefix value {value} does not fit in {} bits",
                        rule.opid, p.bits
                    )));
                }
                parsed.push(BucketPrefix { value, bits: p.bits });
            }
            Classifier::Prefix { prefixes: parsed, filter }
        }
        BucketClassifierJson::Step { start, stop, step, filter } => {
            let start = parse_u64_str(&start, "step.start")?;
            let stop = parse_u64_str(&stop, "step.stop")?;
            let step = parse_u64_str(&step, "step.step")?;
            if step == 0 {
                return Err(ProofmanError::InvalidConfiguration(format!(
                    "Bucket rule for opid {}: `step` must be greater than 0",
                    rule.opid
                )));
            }
            if start >= stop {
                return Err(ProofmanError::InvalidConfiguration(format!(
                    "Bucket rule for opid {}: `start` ({start}) must be strictly less than `stop` ({stop})",
                    rule.opid
                )));
            }
            Classifier::Step { start, stop, step, filter }
        }
    };

    Ok(BucketRule { opid: rule.opid, column: rule.column, classifier })
}

// ============================================================================
// Top-level parser
// ============================================================================

/// Parse the `constraints` (per-air) and `global_constraints` sections into the
/// four `DebugInfo` constraint fields. `max_print` lives on `constraints` and is
/// inherited by the global pass; if `constraints` is absent it defaults to
/// `DEFAULT_N_PRINT_CONSTRAINTS`.
fn parse_constraint_sections(
    constraints: Option<ConstraintsJson>,
    global_constraints: Option<GlobalConstraintsJson>,
) -> (bool, bool, Vec<usize>, usize) {
    let (verify_constraints, n_print_constraints) = match constraints {
        Some(c) if c.enabled => (true, c.max_print.unwrap_or(DEFAULT_N_PRINT_CONSTRAINTS)),
        Some(_) => (false, DEFAULT_N_PRINT_CONSTRAINTS),
        None => (false, DEFAULT_N_PRINT_CONSTRAINTS),
    };

    let (verify_global_constraints, global_constraint_ids) = match global_constraints {
        Some(g) if g.enabled => (true, g.global_constraint_ids.unwrap_or_default()),
        _ => (false, Vec::new()),
    };

    (verify_constraints, verify_global_constraints, global_constraint_ids, n_print_constraints)
}

pub fn json_to_debug_instances_map(proving_key_path: PathBuf, json_path: String) -> ProofmanResult<DebugInfo> {
    if !proving_key_path.exists() {
        return Err(ProofmanError::InvalidConfiguration(format!(
            "Proving key folder not found at path: {proving_key_path:?}"
        )));
    }

    let global_info: GlobalInfo = GlobalInfo::new(&proving_key_path)?;

    let debug_json = fs::read_to_string(&json_path)?;
    let json: DebugJson = serde_json::from_str(&debug_json)?;

    // ---- Instances section ----
    let mut airgroup_map: AirGroupMap = HashMap::new();
    let mut instances_mode = InstancesMode::All;
    let mut skip_tables = false;

    if let Some(instances_section) = json.instances {
        instances_mode = match instances_section.mode.unwrap_or_default() {
            InstancesModeJson::All => InstancesMode::All,
            InstancesModeJson::OnlyListed => InstancesMode::OnlyListed,
        };
        skip_tables = instances_section.skip_tables;

        if let Some(list) = instances_section.list {
            for airgroup in list {
                let mut air_id_map: AirIdMap = HashMap::new();

                if airgroup.airgroup.is_none() && airgroup.airgroup_id.is_none() {
                    return Err(ProofmanError::InvalidSetup(
                        "Airgroup or airgroup_id must be defined in the JSON file".to_string(),
                    ));
                }
                if airgroup.airgroup.is_some() && airgroup.airgroup_id.is_some() {
                    return Err(ProofmanError::InvalidSetup(
                        "Only airgroup or airgroup_id can be defined in the JSON file, not both".to_string(),
                    ));
                }

                let airgroup_id = if let Some(id) = airgroup.airgroup_id {
                    id
                } else {
                    let airgroup_name = airgroup.airgroup.unwrap().to_string();
                    let airgroup_id = global_info.air_groups.iter().position(|x| x == &airgroup_name);
                    if airgroup_id.is_none() {
                        return Err(ProofmanError::InvalidSetup(format!(
                            "Airgroup name {airgroup_name} not found in global_info.airgroups"
                        )));
                    }
                    airgroup_id.unwrap()
                };

                if let Some(airs) = airgroup.airs {
                    for air in airs {
                        if air.air.is_none() && air.air_id.is_none() {
                            return Err(ProofmanError::InvalidSetup(
                                "Air or air_id must be defined in the JSON file".to_string(),
                            ));
                        }
                        if air.air.is_some() && air.air_id.is_some() {
                            return Err(ProofmanError::InvalidSetup(
                                "Only air or air_id can be defined in the JSON file, not both".to_string(),
                            ));
                        }

                        let air_id = if let Some(id) = air.air_id {
                            id
                        } else {
                            let air_name = air.air.unwrap().to_string();
                            let air_id = global_info.airs[airgroup_id].iter().position(|x| x.name == air_name);
                            if air_id.is_none() {
                                return Err(ProofmanError::InvalidSetup(format!(
                                    "Air name {air_name} not found in global_info.airgroups"
                                )));
                            }
                            air_id.unwrap()
                        };

                        let mut instance_map: InstanceMap = HashMap::new();

                        if let Some(instances) = air.instances {
                            for instance in instances {
                                let instance_constraints = instance.constraints.unwrap_or_default();
                                let store_row_info = instance.store_row_info.unwrap_or(false);
                                instance_map.insert(
                                    instance.instance_id.unwrap_or_default(),
                                    InstancesInfo { constraints: instance_constraints, store_row_info },
                                );
                            }
                        }

                        air_id_map.insert(air_id, (air.store_row_info.unwrap_or(false), instance_map));
                    }
                }

                airgroup_map.insert(airgroup_id, air_id_map);
            }
        }
    }

    // ---- Constraints + Global constraints sections ----
    let (verify_constraints, verify_global_constraints, global_constraint_ids, n_print_constraints) =
        parse_constraint_sections(json.constraints, json.global_constraints);

    // ---- Bus section ----
    // Bus debug is enabled by the mere presence of the section. Omit the section
    // entirely to disable it.
    let bus_mode = match json.bus {
        Some(b) => {
            let mut group_by: Vec<BucketRule> = Vec::new();
            if let Some(rules) = b.group_by {
                for rule in rules {
                    let opid = rule.opid;
                    let parsed = build_bucket_rule(rule)?;
                    if group_by.iter().any(|r| r.opid == opid) {
                        return Err(ProofmanError::InvalidConfiguration(format!(
                            "Duplicate bucket rule for opid {opid} in bus.group_by"
                        )));
                    }
                    group_by.push(parsed);
                }
            }
            BusMode {
                enabled: true,
                opids: b.opids.unwrap_or_default(),
                n_vals: b.max_print.unwrap_or(DEFAULT_PRINT_VALS),
                fast_mode: b.fast_mode,
                store_row_info: b.store_row_info,
                values_filter: b.values_filter.unwrap_or_default(),
                group_by,
            }
        }
        None => BusMode::new_disabled(),
    };

    // ---- Output section ----
    let output = match json.output {
        Some(o) => OutputConfig {
            to_file: o.to_file,
            file_path: PathBuf::from(o.file_path.unwrap_or_else(|| DEFAULT_OUTPUT_FILE_PATH.to_string())),
        },
        None => OutputConfig::default(),
    };

    Ok(DebugInfo {
        verify_constraints,
        verify_global_constraints,
        global_constraint_ids,
        instances_mode,
        skip_tables,
        debug_instances: airgroup_map,
        n_print_constraints,
        bus_mode,
        output,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse(json: &str) -> (bool, bool, Vec<usize>, usize) {
        let dj: DebugJson = serde_json::from_str(json).unwrap();
        parse_constraint_sections(dj.constraints, dj.global_constraints)
    }

    #[test]
    fn both_sections_present() {
        let (vc, vgc, ids, max) = parse(
            r#"{ "constraints": { "enabled": true, "max_print": 20 },
                 "global_constraints": { "enabled": true, "global_constraint_ids": [0,1,5] } }"#,
        );
        assert!(vc);
        assert!(vgc);
        assert_eq!(ids, vec![0, 1, 5]);
        assert_eq!(max, 20);
    }

    #[test]
    fn global_present_constraints_absent_uses_default_max_print() {
        let (vc, vgc, ids, max) = parse(r#"{ "global_constraints": { "enabled": true } }"#);
        assert!(!vc);
        assert!(vgc);
        assert!(ids.is_empty());
        assert_eq!(max, DEFAULT_N_PRINT_CONSTRAINTS);
    }

    #[test]
    fn global_absent_skips_global_pass() {
        let (vc, vgc, _ids, _max) = parse(r#"{ "constraints": { "enabled": true } }"#);
        assert!(vc);
        assert!(!vgc);
    }

    #[test]
    fn global_enabled_no_ids_means_all() {
        let (_vc, vgc, ids, _max) = parse(r#"{ "global_constraints": { "enabled": true } }"#);
        assert!(vgc);
        assert!(ids.is_empty());
    }

    #[test]
    fn constraints_enabled_false_disables_per_air() {
        let (vc, _vgc, _ids, _max) = parse(r#"{ "constraints": { "enabled": false } }"#);
        assert!(!vc);
    }
}
