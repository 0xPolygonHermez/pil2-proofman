//! blake3-family Plonk-to-PIL setups.
//!
//! Geometry differs from the poseidon families in one structural way: BLAKE3 is a **56-row block**
//! gate hosting `LANES` permutations in parallel column groups, not a 1-to-10-row gate. See
//! docs/superpowers/specs/2026-08-25-blake3-recursion-air-design.md.

pub mod aggregation;

/// Rows in one BLAKE3 block: 7 rounds x 8 G evaluations.
pub const BLAKE3_CLOCKS: usize = 56;

/// The `a[]` plonk/connection band, also the `S[]` width: the width of Blake3Compress's input row,
/// which is also exactly six three-wire plonk gates.
pub const BAND_COLS: usize = 18;

/// Coefficient columns `C[..]`.
pub const C_COLS: usize = 5;

/// CMul gates (9 signals each) sharing one band row.
pub const CMUL_PER_ROW: usize = 2;

/// Rows an EvPol4 (27 signals) or FFT4 (24) gate spans on the 18-column band: the outputs land on
/// the next row, where the PIL reads them through primes.
pub const TWO_ROW_GATE_ROWS: usize = 2;

/// Permutation columns `blake3Lanes` declares per lane.
pub const PERM_COLS_PER_LANE: usize = 51;

/// Boundary columns the aggregator declares per lane: 1 canonicity witness (`dinv`) and 2
/// feedforward result bytes.
///
/// `outBytes` is ONE 2-byte group per lane: the feedforward result at clocks 24..55, two bytes a row,
/// with `cvBytes` aliased onto it at clocks 0..7 and `vb''`'s three top bits at 16..23 -- fixed
/// clocks that never share a row. Counting the alias twice does not break the exec file, since `write_exec_file` trims
/// all-zero columns, but it makes `stage1_cols` disagree with the air and any capacity check built on
/// it wrong.
pub const BOUNDARY_COLS_PER_LANE: usize = 3;

/// `mul_table` and `mul_range`, shared by every lane.
pub const TABLE_MUL_COLS: usize = 2;

/// Signal indices of `Blake3Compress(flags, isParent)`: `in[16], blockLen, counterLo, out[16]`.
/// 34 signals, every one a trace cell -- flags and isParent are template parameters, so the setup
/// reads their values off the gate id rather than placing their signals.
pub mod compress_signal {
    pub const COUNT: usize = 34;
    /// The input row is exactly the band's width: in[16] then blockLen then counterLo.
    pub const IN_CELLS: usize = 18;
    /// The output row: out[0..16] as u32, leaving a[16..18] free.
    pub const OUT_CELLS: usize = 16;
}

/// Committed stage-1 columns of the aggregator air at `lanes` lanes.
///
/// Pinned by a test rather than trusted: this is the number the packer must agree with, and the
/// per-lane figure changes with every column the air adds or folds away.
pub fn stage1_cols(lanes: usize) -> usize {
    BAND_COLS + (PERM_COLS_PER_LANE + BOUNDARY_COLS_PER_LANE) * lanes + TABLE_MUL_COLS
}

/// Permutations an air of `n` rows with `lanes` lanes can hold.
///
/// The trace cannot be filled to the brim, and the reason is the backward primes rather than the
/// block geometry. `air.CLK[i]` reads row `r - i`; for `r < i` that **wraps** to the end of the
/// trace, so every clock selector would fire spuriously on the first rows unless the wrapped window
/// is padding. The window is the DEEPEST prime, and the air anchors the clocks twice -- `CLK_0` for
/// deepest prime, `CLOCKS - 1`. Hence `56*blocks + 55 <= n`, and NOT `(n / 56) - 1`, which is the
/// same only when 56 divides `n`.
pub fn blake3_capacity(n: usize, lanes: usize) -> usize {
    blake3_max_blocks(n) * lanes
}

/// Lanes the aggregator is built at unless overridden.
pub const DEFAULT_LANES: usize = 4;

/// Deepest backward prime any clock selector uses; the wrap window that many rows must stay padding.
/// Mirrors the single `CLK_0` anchor in blake3/aggregator.pil.
pub const CLOCK_WRAP_ROWS: usize = BLAKE3_CLOCKS - 1;

/// Blocks of 56 rows that fit in `n` rows, leaving the clock selectors' wrap window as padding.
pub fn blake3_max_blocks(n: usize) -> usize {
    n.saturating_sub(CLOCK_WRAP_ROWS) / BLAKE3_CLOCKS
}

pub struct PilTemplateParams<'a> {
    pub namespace_name: &'a str,
    pub n_bits: usize,
    pub n_publics: u32,
    pub n_publics_start: usize,
    pub max_constraint_degree: usize,
    pub n_plonk_rows: usize,
    pub n_cmul_rows: usize,
    pub n_ev_pol4: usize,
    pub n_fft4: usize,
    pub n_tree_selector4: usize,
    /// SelectValArity2 rows, one gate a row; the PIL sizes the band's block run with it.
    pub n_sel_val_rows: usize,
    pub n_node_blocks: usize,
    pub n_chunk_blocks: usize,
    pub n_parent_blocks: usize,
    pub lanes: usize,
}

pub fn gen_pil_str(p: &PilTemplateParams<'_>) -> String {
    format!(
        "require \"blake3/aggregator.pil\";\n\n\
         set_std_mode(STD_MODE_ONE_INSTANCE);\n\n\
         set_max_constraint_degree({md});\n\n\
         public publics[{np}];\n\n\
         airgroup {ns}  {{\n    \
         Aggregator (N: 2**{nb}, nPublics: {np}, nPublicsStart: {nps}, nPlonkRows: {npl}, nCMulRows: {ncm}, nEvPol4: {nev}, nFFT4: {nf4}, \
         nTreeSelector4: {nts}, nSelValRows: {nsv}, nNodeBlocks: {nnb}, \
         nChunkBlocks: {ncb}, nParentBlocks: {npb}, LANES: {nl}) alias {ns};\n\
         }}",
        ns = p.namespace_name,
        nb = p.n_bits,
        np = p.n_publics,
        nps = p.n_publics_start,
        md = p.max_constraint_degree,
        npl = p.n_plonk_rows,
        ncm = p.n_cmul_rows,
        nev = p.n_ev_pol4,
        nf4 = p.n_fft4,
        nts = p.n_tree_selector4,
        nsv = p.n_sel_val_rows,
        nnb = p.n_node_blocks,
        ncb = p.n_chunk_blocks,
        npb = p.n_parent_blocks,
        nl = p.lanes,
    )
}
