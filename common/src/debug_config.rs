//! In-memory representation of `debug.json` configuration.
//!
//! Everything in this module is a *runtime* config struct populated by the JSON
//! parser in [`crate::debug_json`]. The four top-level sections of `debug.json`
//! map to fields on [`DebugInfo`]:
//!
//! | `debug.json` section | [`DebugInfo`] field             |
//! |----------------------|---------------------------------|
//! | `instances`          | `instances_mode`, `debug_instances` |
//! | `constraints`        | `verify_constraints`, `debug_global_instances`, `n_print_constraints` |
//! | `bus`                | `bus_mode`                      |
//! | `output`             | `output`                        |

use std::collections::HashMap;
use std::path::PathBuf;

pub const DEFAULT_PRINT_VALS: usize = 10;
pub const DEFAULT_N_PRINT_CONSTRAINTS: usize = 10;
pub const DEFAULT_OUTPUT_FILE_PATH: &str = "tmp/debug.log";

// ============================================================================
// Top-level config
// ============================================================================

/// Snapshot of `debug.json`, populated by the parser and read by all the debug
/// subsystems through `pctx.debug_info`.
#[derive(Debug, Clone)]
pub struct DebugInfo {
    /// Whether to run the constraint verification pass. Mirrors presence of the
    /// `constraints` section in `debug.json`.
    pub verify_constraints: bool,
    /// Instance filtering mode. `OnlyListed` skips all instances not listed in
    /// `debug_instances`; `All` processes everything (with `debug_instances` only used
    /// for per-instance debug config).
    pub instances_mode: InstancesMode,
    /// When `instances_mode == OnlyListed`, controls whether table instances
    /// are subject to the filter too. Default is `false` — tables are always
    /// included regardless of the `list`, because skipping a lookup table
    /// would cause spurious bus mismatches across every air that consumes it.
    /// Set to `true` to make tables follow the same filter as everything else
    /// (e.g. to skip large tables you don't care about).
    pub skip_tables: bool,
    /// Hierarchical per-instance debug config: airgroup → air → (air_store_row_info, instances).
    pub debug_instances: AirGroupMap,
    /// Global constraint indices to verify. Empty means all globals.
    pub debug_global_instances: Vec<usize>,
    /// Maximum number of mismatched constraints to print per failure.
    pub n_print_constraints: usize,
    /// Bus / std-lookup debug config.
    pub bus_mode: BusMode,
    /// Output destination for debug reporting.
    pub output: OutputConfig,
}

impl Default for DebugInfo {
    fn default() -> Self {
        Self {
            verify_constraints: false,
            instances_mode: InstancesMode::All,
            skip_tables: false,
            debug_instances: Default::default(),
            debug_global_instances: Default::default(),
            n_print_constraints: DEFAULT_N_PRINT_CONSTRAINTS,
            bus_mode: BusMode::new_disabled(),
            output: OutputConfig::default(),
        }
    }
}

impl DebugInfo {
    /// Default "debug everything" config used when the user passes `--debug` with no JSON path.
    pub fn new_debug() -> Self {
        Self {
            verify_constraints: true,
            instances_mode: InstancesMode::All,
            skip_tables: false,
            debug_instances: HashMap::new(),
            debug_global_instances: Vec::new(),
            n_print_constraints: DEFAULT_N_PRINT_CONSTRAINTS,
            bus_mode: BusMode::new_default_debug(),
            output: OutputConfig::default(),
        }
    }
}

// ============================================================================
// Instances
// ============================================================================

/// Instance-processing filter mode (replaces the old `skip_prover_instances: bool`).
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub enum InstancesMode {
    #[default]
    All,
    OnlyListed,
}

/// Per-instance debug config carried as the leaf of the
/// airgroup → air → instance hierarchy.
#[derive(Debug, Clone)]
pub struct InstancesInfo {
    pub constraints: Vec<usize>,
    pub store_row_info: bool,
}

pub type AirGroupMap = HashMap<usize, AirIdMap>;
pub type AirIdMap = HashMap<usize, (bool, InstanceMap)>;
pub type InstanceMap = HashMap<usize, InstancesInfo>;

// ============================================================================
// Bus / std-lookup debugging
// ============================================================================

/// Bus / std-lookup debugging configuration.
///
/// `enabled` mirrors the presence of the `bus` section in `debug.json`. Other fields are
/// only consulted when `enabled` is true.
#[derive(Debug, Clone)]
pub struct BusMode {
    pub enabled: bool,
    pub opids: Vec<u64>,
    pub n_vals: usize,
    pub fast_mode: bool,
    pub store_row_info: bool,
    pub values_filter: Vec<Vec<String>>,
    /// Per-opid bucketing rules. Stored as a `Vec` (not a `HashMap`) because in practice
    /// there are ~10 rules with scattered u64 opid keys — a linear scan over `rule.opid`
    /// beats hashing + probing for that N. See [`find_bucket_rule`].
    pub group_by: Vec<BucketRule>,
}

impl BusMode {
    pub fn new_disabled() -> Self {
        Self {
            enabled: false,
            opids: Vec::new(),
            n_vals: DEFAULT_PRINT_VALS,
            fast_mode: false,
            store_row_info: false,
            values_filter: Vec::new(),
            group_by: Vec::new(),
        }
    }

    pub fn new_default_debug() -> Self {
        Self {
            enabled: true,
            opids: Vec::new(),
            n_vals: DEFAULT_PRINT_VALS,
            fast_mode: true,
            store_row_info: false,
            values_filter: Vec::new(),
            group_by: Vec::new(),
        }
    }
}

/// Linear scan over `group_by` looking for the rule matching `opid`. O(N) but with N≈10
/// scattered u64 keys, this is faster than a HashMap lookup (no hashing, no probing,
/// cache-friendly contiguous access).
#[inline]
pub fn find_bucket_rule(group_by: &[BucketRule], opid: u64) -> Option<&BucketRule> {
    group_by.iter().find(|r| r.opid == opid)
}

impl Default for BusMode {
    fn default() -> Self {
        Self::new_disabled()
    }
}

// ============================================================================
// Per-opid bucketing (group_by)
// ============================================================================

/// Per-opid bucketing rule.
///
/// Subdivides per-opid bus-debug storage by extracting a value from `column` of the bus
/// value tuple and classifying it. The bucket key is a single `u64` — the raw value for
/// `Classifier::Value`, or a bucket index for `Classifier::Range`/`Classifier::Prefix`.
#[derive(Debug, Clone)]
pub struct BucketRule {
    pub opid: u64,
    pub column: usize,
    pub classifier: Classifier,
}

/// Classifier variants. All support an opt-in **filter mode** that drops unmatched rows
/// entirely (useful for narrowing a debug run after fast-mode bucketing has identified
/// the interesting buckets).
///
/// Filter activation differs by variant because the underlying notion of "unmatched"
/// differs:
/// - `Value` filters when `values: Some([...])` is set (only listed values are tracked).
/// - `Range` filters when `filter: true` (gap-free coverage no longer required; values in
///   gaps are dropped).
/// - `Prefix` filters when `filter: true` (the implicit "no match" catch-all is dropped).
/// - `Step` filters when `filter: true` (the implicit "out of range" bucket is dropped).
#[derive(Debug, Clone)]
pub enum Classifier {
    Value {
        values: Option<Vec<u64>>,
    },
    Range {
        ranges: Vec<BucketRange>,
        filter: bool,
    },
    Prefix {
        prefixes: Vec<BucketPrefix>,
        filter: bool,
    },
    /// Uniform-step buckets over `[start, stop)` of width `step`.
    /// Bucket index = `(col - start) / step`. Values outside `[start, stop)` land in an
    /// implicit "out of range" bucket whose index is `ceil((stop - start) / step)` — or
    /// are dropped entirely if `filter: true`.
    Step {
        start: u64,
        stop: u64,
        step: u64,
        filter: bool,
    },
}

#[derive(Debug, Clone, Copy)]
pub struct BucketRange {
    pub min: Option<u64>,
    pub max: Option<u64>,
}

#[derive(Debug, Clone, Copy)]
pub struct BucketPrefix {
    pub value: u64,
    pub bits: u8,
}

// ============================================================================
// Output destination
// ============================================================================

/// Output destination for debug reporting.
#[derive(Debug, Clone)]
pub struct OutputConfig {
    pub to_file: bool,
    pub file_path: PathBuf,
}

impl Default for OutputConfig {
    fn default() -> Self {
        // Debug output is typically voluminous (especially with `max_print` in the
        // millions), so writing to a file is the right default. Use `to_file: false`
        // explicitly to send it to stdout.
        Self { to_file: true, file_path: PathBuf::from(DEFAULT_OUTPUT_FILE_PATH) }
    }
}
