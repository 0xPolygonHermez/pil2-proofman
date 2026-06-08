use std::collections::{HashMap, HashSet};

use colored::Colorize;
use fields::PrimeField64;
use proofman_common::{find_bucket_rule, BucketRule, Classifier, ProofCtx, ProofmanResult};

/// Per-opid storage: either a single balance (no bucketing rule) or a map of bucket keys
/// to balances (when a `group_by` rule applies).
#[derive(Clone, Debug)]
pub enum BucketStore {
    Single(i128),
    Bucketed(HashMap<u64, i128>),
}

impl Default for BucketStore {
    fn default() -> Self {
        BucketStore::Single(0)
    }
}

impl BucketStore {
    #[inline]
    fn new_for(is_bucketed: bool) -> Self {
        if is_bucketed {
            BucketStore::Bucketed(HashMap::new())
        } else {
            BucketStore::Single(0)
        }
    }

    #[inline]
    fn entry_mut(&mut self, bucket_key: u64) -> &mut i128 {
        match self {
            BucketStore::Single(b) => b,
            BucketStore::Bucketed(map) => map.entry(bucket_key).or_insert(0),
        }
    }
}

/// Small vec-based map for ~10 opids. Linear search is faster than hashing for small N.
#[derive(Clone, Debug, Default)]
pub struct DebugDataFast {
    entries: Vec<(u64, BucketStore)>,
}

impl DebugDataFast {
    #[inline]
    pub fn new() -> Self {
        Self { entries: Vec::with_capacity(16) }
    }

    /// Get or insert a `BucketStore` for the given opid. `is_bucketed` decides the storage
    /// shape for newly inserted entries; ignored if the opid already exists.
    #[inline(always)]
    pub fn get_or_insert(&mut self, opid: u64, is_bucketed: bool) -> &mut BucketStore {
        let pos = self.entries.iter().position(|(id, _)| *id == opid);
        if let Some(idx) = pos {
            &mut self.entries[idx].1
        } else {
            self.entries.push((opid, BucketStore::new_for(is_bucketed)));
            &mut self.entries.last_mut().unwrap().1
        }
    }

    #[inline(always)]
    pub fn merge_into(&self, other: &mut DebugDataFast) {
        for (opid, store) in &self.entries {
            match store {
                BucketStore::Single(b) => {
                    if *b != 0 {
                        let entry = other.get_or_insert(*opid, false).entry_mut(0);
                        *entry = entry.wrapping_add(*b);
                    }
                }
                BucketStore::Bucketed(map) => {
                    let other_store = other.get_or_insert(*opid, true);
                    if let BucketStore::Bucketed(other_map) = other_store {
                        for (bk, b) in map {
                            if *b != 0 {
                                let entry = other_map.entry(*bk).or_insert(0);
                                *entry = entry.wrapping_add(*b);
                            }
                        }
                    } else {
                        // Existing entry was Single, but we want Bucketed — should never happen
                        // because we always pass the same `is_bucketed` for a given opid.
                        unreachable!("BucketStore variant mismatch for opid {opid}: cannot merge Bucketed into Single");
                    }
                }
            }
        }
    }
}

impl IntoIterator for DebugDataFast {
    type Item = (u64, BucketStore);
    type IntoIter = std::vec::IntoIter<(u64, BucketStore)>;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        self.entries.into_iter()
    }
}

pub type DebugDataFastGlobal = HashMap<u64, HashSet<u64>>;

#[allow(clippy::too_many_arguments)]
#[inline(always)]
pub fn update_debug_data_fast(
    debug_data_fast: &mut DebugDataFast,
    debug_data_fast_global: &mut DebugDataFastGlobal,
    opid: u64,
    hash: u64,
    bucket_key: u64,
    is_bucketed: bool,
    is_proves: bool,
    times: u64,
    is_global: bool,
) -> ProofmanResult<()> {
    // Skip duplicate global values
    if is_global && !debug_data_fast_global.entry(opid).or_default().insert(hash) {
        return Ok(());
    }

    let contribution = (hash as u128).wrapping_mul(times as u128) as i128;
    let store = debug_data_fast.get_or_insert(opid, is_bucketed);
    let bus = store.entry_mut(bucket_key);

    if is_proves {
        *bus = bus.wrapping_add(contribution);
    } else {
        *bus = bus.wrapping_sub(contribution);
    }

    Ok(())
}

/// Format a single bucket of an opid for human-readable reporting.
fn format_bucket(rule: &BucketRule, bucket_key: u64) -> String {
    match &rule.classifier {
        Classifier::Value { .. } => format!("col[{}] = 0x{bucket_key:x}", rule.column),
        Classifier::Range { ranges, .. } => {
            let idx = bucket_key as usize;
            let r = &ranges[idx];
            let lo = r.min.map(|v| format!("0x{v:x}")).unwrap_or_else(|| "-∞".to_string());
            let hi = r.max.map(|v| format!("0x{v:x}")).unwrap_or_else(|| "+∞".to_string());
            format!("col[{}] in [{lo}, {hi})", rule.column)
        }
        Classifier::Prefix { prefixes, .. } => {
            let idx = bucket_key as usize;
            if idx < prefixes.len() {
                let p = &prefixes[idx];
                format!("col[{}] starts with 0x{:x} ({} bits)", rule.column, p.value, p.bits)
            } else {
                format!("col[{}] matches no prefix", rule.column)
            }
        }
        Classifier::Step { start, stop, step, .. } => {
            let oor = crate::step_oor_index(*start, *stop, *step);
            if bucket_key == oor {
                format!("col[{}] outside [0x{start:x}, 0x{stop:x})", rule.column)
            } else {
                let lo = start + bucket_key * step;
                let hi = lo.saturating_add(*step).min(*stop);
                format!("col[{}] in [0x{lo:x}, 0x{hi:x})", rule.column)
            }
        }
    }
}

pub fn check_invalid_opids<F: PrimeField64>(
    pctx: &ProofCtx<F>,
    debug_data_fast: DebugDataFast,
    is_prod: bool,
) -> Vec<u64> {
    let mut invalid_opids = Vec::new();
    let mut report_lines: Vec<String> = Vec::new();

    let group_by = pctx.debug_info.read().unwrap().bus_mode.group_by.clone();
    let label = if is_prod { "std_prod" } else { "std_sum" };

    for (opid, store) in debug_data_fast.into_iter() {
        match store {
            BucketStore::Single(b) => {
                if b != 0 {
                    invalid_opids.push(opid);
                }
            }
            BucketStore::Bucketed(map) => {
                let rule = find_bucket_rule(&group_by, opid);
                let mut opid_is_invalid = false;
                for (bk, b) in &map {
                    if *b != 0 {
                        opid_is_invalid = true;
                        if let Some(r) = rule {
                            report_lines
                                .push(format!("[{label}] opid {opid}, {}: balance = {b}", format_bucket(r, *bk)));
                        }
                    }
                }
                if opid_is_invalid {
                    invalid_opids.push(opid);
                }
            }
        }
    }

    if !invalid_opids.is_empty() {
        tracing::error!(
            "··· {}",
            format!("\u{2717} [{label}] The following opids do not match {invalid_opids:?}").bright_red().bold()
        );
        for line in &report_lines {
            tracing::error!("··· {}", line.bright_red());
        }
    } else {
        tracing::info!("··· {}", format!("\u{2713} [{label}] All bus values match.").bright_green().bold());
    }

    invalid_opids
}
