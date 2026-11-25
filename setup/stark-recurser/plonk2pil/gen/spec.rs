//! The declarative layout of a plonk2pil AIR: which cells each custom gate's signals take, which
//! plonk gates each row offers (the PIL selectors), and how the plonk constraints are packed. Pure
//! data; [`super::place`] is the one code path that applies it.

use std::ops::Range;

use proofman_common::hash_family::{GateRole, HashFamily};

use crate::plonk2pil::r1cs::types::GateBandKind;

/// What plonk2pil sets up.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SetupKind {
    Compressor,
    Aggregation,
    Wrap,
}

impl std::fmt::Display for SetupKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            SetupKind::Compressor => "compressor",
            SetupKind::Aggregation => "aggregation",
            SetupKind::Wrap => "wrap",
        })
    }
}

/// The plonk gates of one coefficient set on a row: gate `g` reads `a[3g..3g+3]`, the set is
/// `C[coef..coef+5]`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GateGroup {
    pub coef: usize,
    pub gates: Range<usize>,
}

/// A kind of row the plonk constraints may take gates on: its selectors switch on the gates of
/// `mask` (bit `g` = gate `g`); `groups[0]` is opened by the constraint
/// that takes the row, the others queue for the next new coefficient key.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RowKind {
    pub mask: u16,
    pub groups: Vec<GateGroup>,
}

/// Signals `signal..signal + len` of a use go to `a[col..col + len]` at row offset `row`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Run {
    pub signal: usize,
    pub len: usize,
    pub col: usize,
    pub row: usize,
}

/// `n` signals laid `width` to a row from `a[0]`.
pub fn rows_of(n: usize, width: usize) -> Vec<Run> {
    (0..n.div_ceil(width)).map(|r| Run { signal: r * width, len: width.min(n - r * width), col: 0, row: r }).collect()
}

/// The uses a section places.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Gates {
    Role(GateRole),
    /// `Num2Bytes` uses of this many 16-bit chunks.
    RangeChunks(usize),
}

/// Fixed columns a section's rows set besides the signal cells.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConstFill {
    None,
    /// `constFFT[i]` at `C[i % c_cols]` of row offset `i / c_cols`.
    Fft4,
    /// The `PoseidonT(5)` round constants `K`, `POSEIDON` and the round kind.
    PoseidonT,
    /// `RANGE_CHECK` and the chunk weights `2^(16k)` in `weights[k]`; the payload is the chunks.
    RangeCheck {
        weights: &'static str,
    },
}

/// One custom gate, as consecutive row bands of `rows` rows, `per_row` uses side by side `stride`
/// columns apart (a short last row repeats its last use if `pad`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Section {
    pub gates: Gates,
    pub n_signals: Option<usize>,
    pub rows: usize,
    pub per_row: usize,
    pub stride: usize,
    pub pad: bool,
    pub cells: Vec<Run>,
    pub band: Option<GateBandKind>,
    pub fill: ConstFill,
    /// `(row offset, index into AirLayout::row_kinds)`: the rows the plonk constraints may share.
    pub row_kinds: Vec<(usize, usize)>,
}

impl Section {
    /// A one-use-a-row section of `cells` with nothing else.
    pub fn plain(gates: Gates, n_signals: usize, rows: usize, cells: Vec<Run>) -> Self {
        Self {
            gates,
            n_signals: Some(n_signals),
            rows,
            per_row: 1,
            stride: 0,
            pad: false,
            cells,
            band: None,
            fill: ConstFill::None,
            row_kinds: Vec::new(),
        }
    }
}

/// How the plonk constraints are packed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PlonkPolicy {
    /// In constraint order: a constraint joins an open slot of its key, else opens the next queued
    /// coefficient group, else the first row of the first [`RowKind`] with one left, else a new
    /// `pure` row.
    Stream { pure: RowKind },
    /// Grouped by key (`sorted` by it, else in first-seen order), `gates_per_row` to a dedicated
    /// row of one coefficient set; a key's leftovers go to the row kinds' single gates first.
    /// `selector`, if any, is a fixed column set on the dedicated rows.
    GroupByKey { gates_per_row: usize, sorted: bool, selector: Option<&'static str> },
}

/// Where rows come from, past the hash blocks.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Blocks {
    None,
    /// blake3's 56-row blocks of `lanes` lanes; every other gate takes whole runs of block interiors.
    Blake3 {
        lanes: usize,
    },
    /// The blake3 BN128 wrap's one-use 64-row blocks; rows 8..56 of each, then the rows past them.
    Blake3Wrap,
}

/// A value the PIL airtemplate takes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Param {
    PlonkRows,
    Uses(GateRole),
    /// Rows of the role's section (its uses over `per_row`).
    Rows(GateRole),
    PublicsStart,
    NodeBlocks,
    ChunkBlocks,
    ParentBlocks,
    Lanes,
}

/// How a named fixed column starts before any row sets it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Init {
    Zero,
    /// `i mod 2^16`.
    RangeTable,
}

/// A layout's whole AIR.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AirLayout {
    pub kind: SetupKind,
    pub family: HashFamily,
    /// Columns of `a[]` the map spans.
    pub committed: usize,
    /// Connection (`S`) columns.
    pub conn_cols: usize,
    /// `C` columns.
    pub c_cols: usize,
    pub pil_file: &'static str,
    pub pil_template: &'static str,
    pub pil_params: Vec<(&'static str, Param)>,
    pub max_degree: usize,
    /// The caller cannot change `max_degree`.
    pub degree_fixed: bool,
    pub default_name: &'static str,
    /// The default name gets a time stamp.
    pub stamped: bool,
    pub blocks: Blocks,
    pub sections: Vec<Section>,
    /// Piggyback row kinds, in the order the plonk constraints take them.
    pub row_kinds: Vec<RowKind>,
    pub plonk: PlonkPolicy,
    /// The fixed columns in output order, `C` and `S` included.
    pub fixed: Vec<(&'static str, usize, Init)>,
    /// A `PUBLICS_ROW[k]` column per public row, after the others.
    pub publics_row_fixed: bool,
    /// The least `nBits` the AIR's own tables need.
    pub min_bits: usize,
    pub band_aux: u64,
}
