//! In-memory debug report accumulated by the std-lib bus checker.
//!
//! Populated alongside the file/stdout output that [`crate::OutputConfig`] already
//! drives, so callers of `verify-constraints` can consume the same information
//! programmatically and render it however they like.
//!
//! See `DEBUG_REPORT.md` at the repo root for the full JSON shape, field-by-field
//! documentation, and examples.

use serde::Serialize;

#[derive(Debug, Clone, Default, Serialize)]
pub struct DebugReport {
    /// `true` if no mismatched bus values were found during the run.
    pub all_ok: bool,
    pub bus_sections: Vec<BusSection>,
}

#[derive(Debug, Clone, Serialize)]
pub struct BusSection {
    pub opid: u64,
    pub mismatched: bool,
    pub num_overassumed: usize,
    pub num_overproven: usize,
    pub buckets: Vec<BusBucket>,
}

#[derive(Debug, Clone, Serialize)]
pub struct BusBucket {
    pub bucket_key: u64,
    /// Human-readable bucket description, derived from the per-opid bucketing rule.
    /// `None` when the opid has no bucketing rule (single implicit bucket with key 0).
    pub bucket_label: Option<String>,
    pub overassumed: Vec<BusValueMismatch>,
    pub overproven: Vec<BusValueMismatch>,
}

#[derive(Debug, Clone, Serialize)]
pub struct BusValueMismatch {
    /// Bus value tuple as canonical u64 field elements. Extended-field components
    /// are flattened in order alongside base-field components.
    pub vals: Vec<u64>,
    pub hash: u64,
    pub num_assumes: u64,
    pub num_proves: u64,
    /// Aggregated global-level occurrence info (one entry, when present).
    pub global_origin: Option<BusValueGlobalOrigin>,
    /// Per-AIR-instance row occurrences. Each entry collects the row indices where
    /// this bus value appeared in a specific (airgroup, air, instance, hint) site.
    pub local_origins: Vec<BusValueLocalOrigin>,
}

#[derive(Debug, Clone, Serialize)]
pub struct BusValueGlobalOrigin {
    pub airgroup_id: usize,
    pub airgroup_name: String,
    pub piop_name: String,
    pub expression_names: Vec<String>,
    pub is_prod: bool,
}

#[derive(Debug, Clone, Serialize)]
pub struct BusValueLocalOrigin {
    pub airgroup_id: usize,
    pub airgroup_name: String,
    pub air_id: usize,
    pub air_name: String,
    pub instance_id: usize,
    pub hint_id: usize,
    pub piop_name: String,
    pub expression_names: Vec<String>,
    pub is_prod: bool,
    /// Row indices (sorted ascending). Not truncated — the caller decides what to render.
    pub rows: Vec<usize>,
}

impl DebugReport {
    pub fn new() -> Self {
        Self { all_ok: true, bus_sections: Vec::new() }
    }

    pub fn reset(&mut self) {
        self.all_ok = true;
        self.bus_sections.clear();
    }

    pub fn push(&mut self, section: BusSection) {
        if section.mismatched {
            self.all_ok = false;
        }
        self.bus_sections.push(section);
    }
}
