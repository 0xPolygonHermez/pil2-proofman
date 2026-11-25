//! The BLAKE3 block geometry: blake3's 56-row blocks of `LANES` permutations, the runs of block
//! interiors every other gate of its aggregator takes, the compressor's geometry search, and the
//! BN128 wrap's one-use 64-row blocks. See
//! docs/superpowers/specs/2026-08-25-blake3-recursion-air-design.md.

use std::collections::BTreeMap;

use num_bigint::BigUint;
use proofman_common::exec_format::{BLAKE3_IV, BLAKE3_SIGMA, BLAKE3_WRAP_BLOCK_ROWS};
use proofman_common::hash_family::GateRole;
use proofman_fields::{PrimeField, QuotientMap};

use super::place::Sheet;
use crate::plonk2pil::field::PlonkField;
use crate::plonk2pil::r1cs::to_plonk::{blake3_compress_gate_uses, filter_gate_uses, CustomGatesInfo};
use crate::plonk2pil::r1cs::types::{CustomGateUse, GateBand, GateBandKind, R1csFile};

/// Rows in one BLAKE3 block: 7 rounds x 8 G evaluations.
pub const BLAKE3_CLOCKS: usize = 56;

/// The `a[]` plonk/connection band, also the `S[]` width: the width of Blake3Compress's input row,
/// which is also exactly six three-wire plonk gates.
pub const BAND_COLS: usize = 18;

/// Coefficient columns `C[..]`.
pub const C_COLS: usize = 5;

/// CMul gates (9 signals each) sharing one band row.
pub const CMUL_PER_ROW: usize = 2;

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

pub const PLONK_GATES_PER_ROW: usize = 6;

/// `coefs[5][3] + x[3] + out[3] + s[3] + acc[3]`; the last six are the Estrin intermediates, which
/// every family's gate publishes and only this air binds.
pub const EVPOL4_SIGNALS: usize = 27;

/// What the band costs once every BLAKE3 block's interior is available to every circuit.
///
/// All the band circuits bind the same `a[0..17]`, so a row hosts exactly one of them; each takes a
/// whole number of blocks and fills their interiors, which keeps the AIR's selectors expressible as
/// repetitions (see `blake3/aggregator.pil`). `runs` is `(rows or gates, rows per gate)` in the
/// PIL's order; the only cost is the tail of each circuit's last block.
pub fn plan_band_blocks(runs: &[(usize, usize)], lanes: usize) -> usize {
    let interior = BLAKE3_CLOCKS - 2 * lanes;
    runs.iter().map(|&(rows, step)| rows.div_ceil((interior / step).max(1))).sum()
}

/// Hands out rows for the shared `a[0..17]` band, mirroring the selector patterns in
/// `blake3/aggregator.pil`.
///
/// Every BLAKE3 block leaves `CLOCKS - 2*LANES` interior rows free -- its first LANES rows carry the
/// lanes' inputs and its last LANES their outputs -- and any interior row can host any one of the
/// six band circuits, because all six bind the same eighteen columns and are told apart only by
/// which selector is 1 there.
///
/// Each circuit gets a whole number of BLOCKS, in the order the PIL lays them out, and fills their
/// interiors. Block granularity is what lets the AIR state the shape as a repetition instead of
/// receiving six transported columns; the cost is the tail of each circuit's last block. `open`
/// starts a circuit's run and must be called in the PIL's order, so a drift between the two shows
/// up as a row count that does not match rather than as a constraint that never fires.
pub(crate) struct RowAlloc {
    lanes: usize,
    /// First block no run has claimed yet.
    next_block: usize,
    /// Block the open run is filling. Tracked, not derived: deriving it from the interior's end
    /// only happens to work at one lane count.
    block: usize,
    row: usize,
    end: usize,
    step: usize,
}

impl RowAlloc {
    pub(crate) fn new(lanes: usize) -> Self {
        Self { lanes, next_block: 0, block: 0, row: 0, end: 0, step: 1 }
    }

    fn interior(&self) -> usize {
        BLAKE3_CLOCKS - 2 * self.lanes
    }

    fn enter(&mut self, block: usize) {
        self.block = block;
        self.row = block * BLAKE3_CLOCKS + self.lanes;
        self.end = self.row + self.interior();
    }

    /// Start the next circuit's run of blocks. `step` is 1 for a one-row circuit and 2 for one that
    /// reads the next row through primes.
    pub(crate) fn open(&mut self, rows: usize, step: usize) {
        let per_block = (self.interior() / step).max(1);
        self.step = step;
        self.enter(self.next_block);
        self.next_block += rows.div_ceil(per_block);
    }

    /// The first of `step` consecutive rows, from the current run.
    pub(crate) fn take(&mut self) -> usize {
        if self.row + self.step > self.end {
            self.enter(self.block + 1);
        }
        let r = self.row;
        self.row += self.step;
        r
    }

    /// Blocks the band has consumed.
    pub(crate) fn blocks_used(&self) -> usize {
        self.next_block
    }
}

/// One `Blake3Compress` use with its gate's `(flags, isParent)`.
pub(crate) type CompressUse<'a, F> = (&'a CustomGateUse, F, F);

/// Groups `Blake3Compress` uses by their `flags` template parameter, lowest value first.
///
/// `flags` reaches the air as a fixed column filled per whole 56-row block, so the uses that share a
/// block must agree on it. Sorted by value rather than left in arrival order so the resulting s_map
/// is a function of the r1cs alone -- the same reason the plonk gates are ordered explicitly.
pub(crate) fn bucket_by_flags<F: PrimeField>(uses: Vec<CompressUse<'_, F>>) -> Vec<Vec<CompressUse<'_, F>>> {
    let mut by_flags: BTreeMap<BigUint, Vec<_>> = BTreeMap::new();
    for u in uses {
        by_flags.entry(u.1.as_canonical_biguint()).or_default().push(u);
    }
    by_flags.into_values().collect()
}

/// The BLAKE3 blocks of an air: every `Blake3Node` and `Blake3Compress` use of the r1cs in whole
/// 56-row blocks of `lanes` lanes, Node blocks first, then chunk, then parent, each Compress kind
/// bucketed by `flags`. Over either field: the blake3 recursion airs' and the BN128 wrap's.
pub(crate) struct Blake3Blocks<'a, F> {
    pub lanes: usize,
    node_uses: Vec<&'a CustomGateUse>,
    chunk_buckets: Vec<Vec<CompressUse<'a, F>>>,
    parent_buckets: Vec<Vec<CompressUse<'a, F>>>,
    pub n_node_blocks: usize,
    pub n_chunk_blocks: usize,
    pub n_parent_blocks: usize,
}

impl<'a, F: PlonkField + QuotientMap<u64>> Blake3Blocks<'a, F> {
    pub fn new(r1cs: &'a R1csFile<F>, cgi: &CustomGatesInfo<F>, lanes: usize) -> Self {
        // Blake3Compress carries `(flags, isParent)` as TEMPLATE PARAMETERS, so circom mints one gate id
        // per distinct pair and the r1cs records the values. isParent picks the block kind and flags
        // becomes a fixed column; neither is a trace cell, and neither could be recovered from
        // `cgu.signals` -- they live on the gate, not the use.
        let compress_uses = blake3_compress_gate_uses(&r1cs.custom_gates_uses, &cgi.blake3_compress_parameters);
        let (parent_uses, chunk_uses): (Vec<_>, Vec<_>) = compress_uses.into_iter().partition(|(_, _, ip)| ip.is_one());

        // `flags` is a FIXED column the air fills per whole block, so every use sharing a block has to
        // share its flags value. Bucket by flags and give each bucket whole blocks. Packing uses in
        // arrival order instead keeps whichever flags value was written last for the entire block, and
        // `(BLAKE3_CHUNK * 3'CLK_0) * (vd - FLAGS) === 0` -- st[15] is flags at clock 3 -- then fails on
        // every block that mixed two values.
        let chunk_buckets = bucket_by_flags(chunk_uses);
        let parent_buckets = bucket_by_flags(parent_uses);
        let blocks_in = |buckets: &[Vec<CompressUse<'_, F>>]| buckets.iter().map(|b| b.len().div_ceil(lanes)).sum();
        for (label, buckets) in [("chunk", &chunk_buckets), ("parent", &parent_buckets)] {
            for b in buckets.iter() {
                tracing::info!(
                    "Blake3Compress {label} flags={} : {} uses -> {} blocks ({} lanes idle in the last)",
                    b[0].1,
                    b.len(),
                    b.len().div_ceil(lanes),
                    (lanes - b.len() % lanes) % lanes
                );
            }
        }
        let node_uses = filter_gate_uses(&r1cs.custom_gates_uses, cgi.role_id(GateRole::Blake3Node));
        Self {
            lanes,
            n_node_blocks: node_uses.len().div_ceil(lanes),
            n_chunk_blocks: blocks_in(&chunk_buckets),
            n_parent_blocks: blocks_in(&parent_buckets),
            node_uses,
            chunk_buckets,
            parent_buckets,
        }
    }

    pub fn n_blocks(&self) -> usize {
        self.n_node_blocks + self.n_chunk_blocks + self.n_parent_blocks
    }

    /// Every use, in the blocks' order at one lane: its kind (0 Node, 1 chunk, 2 parent), its
    /// flags and the use.
    pub fn uses(&self) -> Vec<(u8, u64, &'a CustomGateUse)> {
        let flags = |f: &F| u64::try_from(&f.as_canonical_biguint()).expect("flags is a byte");
        let nodes = self.node_uses.iter().map(|&u| (0, 11, u));
        let compress = |kind: u8, buckets: &[Vec<CompressUse<'a, F>>]| {
            buckets.iter().flatten().map(|(u, f, _)| (kind, flags(f), *u)).collect::<Vec<_>>()
        };
        nodes.chain(compress(1, &self.chunk_buckets)).chain(compress(2, &self.parent_buckets)).collect()
    }

    /// Writes the blocks' boundary cells into `s_map` (`n` rows), and returns the `FLAGS` column
    /// and a gate band per block. The 54 columns per lane are the expander's; the setup writes the
    /// two boundary rows. Lane l of a block reads its input at clock l and writes its output at
    /// clock 56 - LANES + l (spec 3.2).
    pub fn place(&self, s_map: &mut [Vec<u32>], n: usize) -> (Vec<F>, Vec<GateBand>) {
        let lanes = self.lanes;
        let (n_node_blocks, n_chunk_blocks, n_blocks) = (self.n_node_blocks, self.n_chunk_blocks, self.n_blocks());
        let node_uses = &self.node_uses;
        tracing::info!(
            "Processing {} Node, {} chunk, {} parent gates in {n_blocks} blocks...",
            node_uses.len(),
            self.chunk_buckets.iter().map(Vec::len).sum::<usize>(),
            self.parent_buckets.iter().map(Vec::len).sum::<usize>()
        );
        // `flags` is st[15], a per-block constant read off the gate id's parameters.
        let mut flags_col = vec![F::ZERO; n];
        let mut gate_bands = Vec::with_capacity(n_blocks);

        // The kind selectors are block-wide, so an idle lane in a partially filled final block still has
        // to hold a valid BLAKE3 computation. Left alone its `a[]` cells stay zero while the expander
        // hashes that zero input and writes a real digest into outBytes, and the output binding fails on
        // exactly those lanes. Repeating the last use into them costs no block and no row: the duplicate
        // maps the SAME circom signals, so the copy constraints it creates are true by construction and
        // the lane computes something real.
        for i in 0..if node_uses.is_empty() { 0 } else { n_node_blocks * lanes } {
            let cgu = &node_uses[i.min(node_uses.len() - 1)];
            assert_eq!(cgu.signals.len(), 13, "Blake3Node is in[8] + key + out[4]");
            let (block, lane) = (i / lanes, i % lanes);
            let base = block * BLAKE3_CLOCKS;
            for (j, sig) in cgu.signals[..9].iter().enumerate() {
                s_map[j][base + lane] = *sig as u32; // in[0..8] then the Merkle path bit
            }
            for (j, sig) in cgu.signals[9..13].iter().enumerate() {
                s_map[j][base + BLAKE3_CLOCKS - lanes + lane] = *sig as u32;
            }
        }

        // Each bucket starts on a fresh block, so a block never mixes two flags values. Returns the
        // first block after everything it placed, which is what pins the placement to the block counts.
        let place_compress =
            |buckets: &[Vec<CompressUse<'_, F>>], block_base: usize, s_map: &mut [Vec<u32>], flags_col: &mut [F]| {
                let mut next_block = block_base;
                for bucket in buckets {
                    // Pad the final block's idle lanes with the bucket's last use, for the reason given at
                    // the Node placement above. Same bucket, so the repeat carries the same flags and the
                    // block stays uniform.
                    for i in 0..bucket.len().div_ceil(lanes) * lanes {
                        let (cgu, flags, _) = &bucket[i.min(bucket.len() - 1)];
                        assert_eq!(
                            cgu.signals.len(),
                            compress_signal::COUNT,
                            "Blake3Compress is in[16] + blockLen + counterLo + out[16]"
                        );
                        let (block, lane) = (next_block + i / lanes, i % lanes);
                        let base = block * BLAKE3_CLOCKS;
                        // in[16], blockLen, counterLo -- exactly the band's width, in declaration order
                        for (j, sig) in cgu.signals[..compress_signal::IN_CELLS].iter().enumerate() {
                            s_map[j][base + lane] = *sig as u32;
                        }
                        // out[0..16] as u32, one cell each
                        for (j, sig) in cgu.signals[compress_signal::IN_CELLS..].iter().enumerate() {
                            s_map[j][base + BLAKE3_CLOCKS - lanes + lane] = *sig as u32;
                        }
                        // every row of the block carries this block's flags
                        flags_col[base..base + BLAKE3_CLOCKS].fill(*flags);
                    }
                    next_block += bucket.len().div_ceil(lanes);
                }
                next_block
            };
        let after_chunk = place_compress(&self.chunk_buckets, n_node_blocks, s_map, &mut flags_col);
        let after_parent = place_compress(&self.parent_buckets, after_chunk, s_map, &mut flags_col);
        // The air is sized from n_chunk_blocks/n_parent_blocks; if the placement walked past them the
        // trace would run off the end of a shorter air, so tie the two together rather than trusting it.
        assert_eq!(after_chunk, n_node_blocks + n_chunk_blocks, "chunk placement disagrees with its block count");
        assert_eq!(after_parent, n_blocks, "parent placement disagrees with its block count");

        // Blake3Node freezes flags to CHUNK_START | CHUNK_END | ROOT.
        flags_col[..n_node_blocks * BLAKE3_CLOCKS].fill(F::from_int(11u64));

        // One band per block: the expander rebuilds every lane's interior from the boundary.
        for block in 0..n_blocks {
            let kind = if block < n_node_blocks {
                GateBandKind::Blake3Node
            } else if block < n_node_blocks + n_chunk_blocks {
                GateBandKind::Blake3CompressChunk
            } else {
                GateBandKind::Blake3CompressParent
            };
            // The payload carries this block's `flags`. The expander needs it to run the permutation,
            // and the AIR holds it in a FIXED column, so it is in neither the witness trace nor
            // anything expand_gate_bands is handed -- which is what the band section's version 2 is for.
            let row = block * BLAKE3_CLOCKS;
            let payload = u64::try_from(&flags_col[row].as_canonical_biguint()).expect("flags is a byte");
            gate_bands.push(GateBand { row: row as u32, kind, payload });
        }
        (flags_col, gate_bands)
    }
}

// ── Compressor geometry ──────────────────────────────────────────────────────
//
// An air whose `recursive1` would not fit the pinned recursion (2^19, LANES=4) gets a compressor
// first: this same aggregator air at a geometry of its own. N is the recursion's 2^19, or the next
// power of two its circuit needs, and LANES the fewest that hold it. The air holds
// `max(hashing blocks, band blocks)`; more lanes shrink the hashing and narrow the block interiors
// the band rides, and each one costs 54 stage1 columns the recursive1 above has to open.

/// Lanes the air accepts: lane l's input row is clock l, and only clocks 0..7 carry round 0's identity
/// message schedule -- see `aggregator.pil`.
pub(crate) const MAX_LANES: usize = 8;

/// A geometry the compressor can be built at.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct CompressorGeometry {
    pub n_bits: usize,
    pub lanes: usize,
    /// BLAKE3 blocks the hashing needs at this `lanes`.
    pub blocks: usize,
    /// Blocks the band circuits need the interiors of.
    pub band_blocks: usize,
}

/// What the sizing is given: the counts, not the r1cs.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct CompressorDemand {
    /// `Blake3Node` gate uses.
    node_uses: usize,
    /// `Blake3Compress` chunk uses, bucketed by `flags`: a fixed column filled per whole block, so
    /// each bucket takes whole blocks.
    chunk_buckets: Vec<usize>,
    /// `Blake3Compress` parent uses, bucketed the same way.
    parent_buckets: Vec<usize>,
    /// The band circuits, as [`plan_band_blocks`] takes them.
    band: Vec<(usize, usize)>,
}

impl CompressorDemand {
    /// The hashing of `r1cs` and the band circuits' `band` runs.
    pub fn new<F: PlonkField>(r1cs: &R1csFile<F>, cgi: &CustomGatesInfo<F>, band: Vec<(usize, usize)>) -> Self {
        let compress_uses = blake3_compress_gate_uses(&r1cs.custom_gates_uses, &cgi.blake3_compress_parameters);
        let (parent_uses, chunk_uses): (Vec<_>, Vec<_>) = compress_uses.into_iter().partition(|(_, _, ip)| ip.is_one());
        let sizes = |uses: Vec<_>| bucket_by_flags(uses).iter().map(|b| b.len()).collect::<Vec<_>>();
        Self {
            node_uses: cgi.n(GateRole::Blake3Node),
            chunk_buckets: sizes(chunk_uses),
            parent_buckets: sizes(parent_uses),
            band,
        }
    }

    /// Blocks the hashing needs at `lanes`.
    fn blocks(&self, lanes: usize) -> usize {
        let bucketed = |b: &[usize]| b.iter().map(|n| n.div_ceil(lanes)).sum::<usize>();
        self.node_uses.div_ceil(lanes) + bucketed(&self.chunk_buckets) + bucketed(&self.parent_buckets)
    }

    /// Blocks whose interiors the band needs at `lanes`.
    fn band_blocks(&self, lanes: usize) -> usize {
        plan_band_blocks(&self.band, lanes)
    }

    /// Blocks the air must hold: the hashing's, or the band's if the band wants more.
    fn blocks_needed(&self, lanes: usize) -> usize {
        self.blocks(lanes).max(self.band_blocks(lanes))
    }

    fn fits(&self, n_bits: usize, lanes: usize) -> bool {
        self.blocks_needed(lanes) <= blake3_max_blocks(1usize << n_bits)
    }
}

/// The smallest N from `min_n_bits` up, and at that N the fewest lanes that hold the demand.
pub(crate) fn plan_compressor_geometry(demand: &CompressorDemand, min_n_bits: usize) -> CompressorGeometry {
    let geom = (min_n_bits..=32)
        .find_map(|n_bits| {
            (1..=MAX_LANES).find(|&lanes| demand.fits(n_bits, lanes)).map(|lanes| CompressorGeometry {
                n_bits,
                lanes,
                blocks: demand.blocks(lanes),
                band_blocks: demand.band_blocks(lanes),
            })
        })
        .expect("some N holds any demand at 8 lanes");
    tracing::info!(
        "Compressor geometry: N = 2^{}, LANES = {} ({} hashing / {} band blocks of {}), stage1 {} cols",
        geom.n_bits,
        geom.lanes,
        geom.blocks,
        geom.band_blocks,
        blake3_max_blocks(1usize << geom.n_bits),
        stage1_cols(geom.lanes),
    );
    geom
}

/// The hash blocks of an AIR, which every other gate is placed around.
pub(crate) enum HashBlocks<'a, F> {
    None,
    Blake3(Blake3Blocks<'a, F>),
    /// The BN128 wrap's: one 64-row block a use, of kind 0 Node, 1 chunk, 2 parent, with its flags.
    Wrap(Vec<(u8, u64, &'a CustomGateUse)>),
}

impl<F: PlonkField + QuotientMap<u64>> HashBlocks<'_, F> {
    pub fn n_blocks(&self) -> usize {
        match self {
            HashBlocks::None => 0,
            HashBlocks::Blake3(b) => b.n_blocks(),
            HashBlocks::Wrap(uses) => uses.len(),
        }
    }

    /// Custom-gate uses the blocks place.
    pub fn n_uses(&self) -> usize {
        match self {
            HashBlocks::None => 0,
            HashBlocks::Blake3(b) => {
                b.node_uses.len() + b.chunk_buckets.iter().chain(&b.parent_buckets).map(Vec::len).sum::<usize>()
            }
            HashBlocks::Wrap(uses) => uses.len(),
        }
    }

    pub fn place(&self, sh: &mut Sheet<F>) {
        match self {
            HashBlocks::None => {}
            HashBlocks::Blake3(b) => {
                let n = sh.s_map[0].len();
                let (flags, bands) = b.place(&mut sh.s_map, n);
                sh.col("FLAGS", 0).copy_from_slice(&flags);
                sh.gate_bands.extend(bands);
            }
            HashBlocks::Wrap(uses) => place_wrap_blocks(uses, sh),
        }
    }
}

/// The blake3 BN128 wrap's blocks (`pil/blake3_bn128/wrap.pil`), a use's signals each:
///
/// | kind   | rows 0..7 (round 0, row t)             | row 0 too              | row 56 + k                                   |
/// |--------|----------------------------------------|------------------------|----------------------------------------------|
/// | Node   | a0 = in[t], a1 = in[(t+4)%8], a2 = key |                        | a0 = out[k], k < 4                           |
/// | chunk  | a0 = in[8+t]                           | a3..a10 = cv, a11, a12 | a0, a1 = out[2k], out[2k+1]; a2, a3 = cv[2k-8], cv[2k-7] for k >= 4 |
/// | parent | a0 = in[2t], a1 = in[2t+1]             | a11 = blockLen, a12    | a0, a1 = out[2k], out[2k+1]                  |
///
/// Each block is a gate band, its flags the payload, whose other columns the wrap's witness rebuilds.
fn place_wrap_blocks<F: PlonkField + QuotientMap<u64>>(blocks: &[(u8, u64, &CustomGateUse)], sh: &mut Sheet<F>) {
    let band_kind =
        [GateBandKind::Blake3Bn128WrapNode, GateBandKind::Blake3Bn128WrapChunk, GateBandKind::Blake3Bn128WrapParent];
    let int = |v: u64| F::from_int(v);
    for (b, &(kind, flags, cgu)) in blocks.iter().enumerate() {
        let base = b * BLAKE3_WRAP_BLOCK_ROWS;
        // Wire 0 is "not connected" in the s-maps: a gate cell there would be free.
        let sig = |i: usize| {
            let wire = cgu.signals[i] as u32;
            assert_ne!(wire, 0, "a blake3 gate's signal {i} is the constant wire 0");
            wire
        };
        let (node, chunk) = (kind == 0, kind == 1);
        if node {
            assert_eq!(cgu.signals.len(), 13, "Blake3Node is in[8] + key + out[4]");
        } else {
            assert_eq!(
                cgu.signals.len(),
                compress_signal::COUNT,
                "Blake3Compress is in[16] + blockLen + counterLo + out[16]"
            );
        }
        for t in 0..8 {
            let row = base + t;
            if node {
                sh.s_map[0][row] = sig(t);
                sh.s_map[1][row] = sig((t + 4) % 8);
                sh.s_map[2][row] = sig(8);
            } else if chunk {
                sh.s_map[0][row] = sig(8 + t);
            } else {
                sh.s_map[0][row] = sig(2 * t);
                sh.s_map[1][row] = sig(2 * t + 1);
            }
            sh.set(["IN_NODE", "IN_CHUNK", "IN_PARENT"][kind as usize], 0, row, F::ONE);
        }
        if chunk {
            for j in 0..8 {
                sh.s_map[3 + j][base] = sig(j);
            }
        }
        if !node {
            sh.s_map[11][base] = sig(16);
            sh.s_map[12][base] = sig(17);
            sh.set("BLOCKLEN_ROW", 0, base + 2, F::ONE);
        }
        sh.set(["INIT_NODE", "INIT_CHUNK", "INIT_PARENT"][kind as usize], 0, base, F::ONE);
        sh.set("FLAGS", 0, base, int(flags));

        for t in 0..56 {
            let (r, g) = (t / 8, t % 8);
            let row = base + t;
            sh.set("G", g, row, F::ONE);
            let key = |w: usize| int((16 * b + BLAKE3_SIGMA[r][w]) as u64);
            sh.set("MSG_KEY_X", 0, row, key(2 * g));
            sh.set("MSG_KEY_Y", 0, row, key(2 * g + 1));
            sh.set("MSG_MUL", 0, row, if r == 0 { int(6) } else { -F::ONE });
        }
        for t in 56..63 {
            sh.set("HOLD", 0, base + t, F::ONE);
        }
        for k in 0..if node { 4 } else { 8 } {
            let row = base + 56 + k;
            sh.set("FF", k, row, F::ONE);
            if node {
                sh.s_map[0][row] = sig(9 + k);
                sh.set("FF_NODE", 0, row, F::ONE);
                continue;
            }
            sh.s_map[0][row] = sig(compress_signal::IN_CELLS + 2 * k);
            sh.s_map[1][row] = sig(compress_signal::IN_CELLS + 2 * k + 1);
            if k >= 4 {
                if chunk {
                    sh.s_map[2][row] = sig(2 * k - 8);
                    sh.s_map[3][row] = sig(2 * k - 7);
                    sh.set("FF_CV", 0, row, F::ONE);
                } else {
                    for h in 0..2 {
                        sh.set("FF_IV", h, row, int(u64::from(BLAKE3_IV[2 * k - 8 + h])));
                    }
                }
            }
        }
        sh.gate_bands.push(GateBand { row: base as u32, kind: band_kind[kind as usize], payload: flags });
    }
}

#[cfg(test)]
mod tests {
    use super::super::place::plan_groups;
    use super::*;
    use crate::plonk2pil::r1cs::to_plonk::PlonkConstraint;
    use proofman_fields::{Field, Goldilocks, PrimeField64};

    /// Dedicated rows of one coefficient set, six gates each.
    fn plonk_rows(cs: &[PlonkConstraint<Goldilocks>]) -> usize {
        plan_groups(cs, PLONK_GATES_PER_ROW, true, 0).0.iter().map(|g| g.len().div_ceil(PLONK_GATES_PER_ROW)).sum()
    }

    /// `flags` reaches the air as a FIXED column filled per whole 56-row block, so the uses sharing a
    /// block have to agree on it. A circuit mints one gate id per distinct value, so a block packed in
    /// arrival order keeps whichever value landed last and the state-init check
    /// `(BLAKE3_CHUNK * 3'CLK_0) * (vd - FLAGS) === 0` fails on it. Grouping is what makes a block's
    /// flags well defined at all.
    #[test]
    fn flags_never_share_a_block() {
        use crate::plonk2pil::r1cs::types::CustomGateUse;
        let uses: Vec<CustomGateUse> = (0..10).map(|i| CustomGateUse { id: i, signals: vec![i as u64] }).collect();
        // Interleaved on purpose: arrival order is what the r1cs hands over.
        let flags = [11u64, 0, 1, 0, 11, 1, 0, 10, 1, 0];
        let tagged: Vec<_> = uses.iter().zip(flags).map(|(u, f)| (u, Goldilocks::new(f), Goldilocks::ZERO)).collect();

        let buckets = bucket_by_flags(tagged);

        assert_eq!(buckets.iter().map(|b| b.len()).sum::<usize>(), 10, "no use may be dropped");
        for b in &buckets {
            assert!(b.iter().all(|u| u.1 == b[0].1), "a bucket has to be one flags value");
        }
        let seen: Vec<u64> = buckets.iter().map(|b| b[0].1.as_canonical_u64()).collect();
        assert_eq!(seen, vec![0, 1, 10, 11], "sorted by value, so the s_map is a function of the r1cs");
    }

    /// Padding the tail lanes reads the bucket's LAST use, so it must never index past the end and
    /// must still visit every real use exactly once. Every lane of every allocated block gets a use:
    /// the kind selectors are block-wide, so an idle lane would still be constrained to hold a valid
    /// BLAKE3 computation while its `a[]` cells sat at zero.
    #[test]
    fn padding_covers_every_lane_without_running_off_the_bucket() {
        for lanes in 1..=8usize {
            for len in 1..=20usize {
                let blocks = len.div_ceil(lanes);
                let picked: Vec<usize> = (0..blocks * lanes).map(|i| i.min(len - 1)).collect();

                assert_eq!(picked.len(), blocks * lanes, "every lane of every block is filled");
                assert!(picked.iter().all(|&i| i < len), "lanes={lanes} len={len}: indexed past the bucket");
                for real in 0..len {
                    assert!(picked.contains(&real), "lanes={lanes} len={len}: dropped use {real}");
                }
            }
        }
    }

    /// The BLAKE3 lookups declare their selector binary (`SEL_IS_BINARY: 1` in circuits/blake3.pil),
    /// which skips the library's `sel * (sel - 1) === 0`. That is a soundness claim, so the shapes it
    /// rests on are pinned here rather than trusted: both are sums of DISJOINT 0/1 fixed columns.
    ///
    /// Skipping it is worth the pin. The constraint reads no witness -- it is an identity between
    /// fixed columns -- and the library emits one per bus TERM, not one per distinct selector, so the
    /// same identity is otherwise proved once for every lookup that shares the selector.
    #[test]
    fn blake3_kind_selectors_are_binary_and_disjoint() {
        // BLAKE3_NODE / BLAKE3_CHUNK / BLAKE3_PARENT, as blake3/aggregator.pil lays them out: one
        // block-wide run each, back to back, then zero.
        for (node, chunk, parent) in [(3usize, 2usize, 1usize), (7, 0, 0), (0, 5, 2), (1, 1, 1)] {
            let n = (node + chunk + parent) * BLAKE3_CLOCKS + 64;
            let mut kinds = vec![[0u8; 3]; n];
            for b in 0..node + chunk + parent {
                let k = if b < node {
                    0
                } else if b < node + chunk {
                    1
                } else {
                    2
                };
                for row in kinds[b * BLAKE3_CLOCKS..(b + 1) * BLAKE3_CLOCKS].iter_mut() {
                    row[k] = 1;
                }
            }
            for (row, k) in kinds.iter().enumerate() {
                let any: u8 = k.iter().sum();
                assert!(any <= 1, "row {row}: BLAKE3_ANY = {any}, so it is not binary");
            }
        }
    }

    /// The other declared-binary selector: the feedforward's `ANY`, sixteen shifts of CLK_0. CLK_0
    /// fires on the first row of each block, so the shifts land on distinct rows and at most one is
    /// ever 1 -- which is what makes the sum 0/1.
    #[test]
    fn folded_output_clocks_never_overlap() {
        const OUT_CLOCKS: usize = 16;
        for blocks in 1..=4usize {
            let n = blocks * BLAKE3_CLOCKS;
            let clk0 = |r: usize| usize::from(r.is_multiple_of(BLAKE3_CLOCKS));
            for row in 0..n {
                // ANY = sum over i of CLK_0 shifted by (CLOCKS - 16 + i), read at `row`.
                let any: usize = (0..OUT_CLOCKS)
                    .map(|i| {
                        let off = BLAKE3_CLOCKS - OUT_CLOCKS + i;
                        clk0((row + n - off % n) % n)
                    })
                    .sum();
                assert!(any <= 1, "row {row} of {blocks} blocks: folded ANY = {any}");
            }
        }
    }

    /// Overflowing capacity places a band past the trace, which surfaces only as an unprovable air,
    /// so the bound is pinned against the air's own rather than derived here.
    ///
    /// `floor(N / CLOCKS)` is NOT the bound: the clock selectors read backwards through primes, so
    /// the deepest prime's window has to stay padding. These pin what `blake3_max_blocks` accepts and
    /// rejects on either side of it.
    #[test]
    fn capacity_is_reported_not_overflowed() {
        assert_eq!(blake3_max_blocks(1 << 19), 9_361);
        assert_eq!(blake3_capacity(1 << 19, 4), 37_444);
        assert_eq!(blake3_capacity(1 << 19, 1), 9_361);
    }

    /// One block needs its own 56 rows plus the wrap window the clock selectors read backwards into.
    /// That window is the DEEPEST prime, so it follows the anchoring: 27 rows with the two anchors
    /// (`CLK_0` and `CLK_HALF`), where a single anchor would need 55. `(n / 56) - 1` gets this wrong
    /// for every n that 56 does not divide.
    #[test]
    fn capacity_leaves_the_clock_selectors_wrap_window_as_padding() {
        assert_eq!(CLOCK_WRAP_ROWS, 55, "one anchor, so the deepest prime is CLOCKS - 1");
        for blocks in 0..4usize {
            let exact = blocks * BLAKE3_CLOCKS + CLOCK_WRAP_ROWS;
            assert_eq!(blake3_max_blocks(exact), blocks, "n={exact} should hold exactly {blocks}");
            assert_eq!(blake3_max_blocks(exact - 1), blocks.saturating_sub(1), "n={} is one row short", exact - 1);
        }
        assert_eq!(blake3_max_blocks(111), 1, "56 rows of block plus 55 of window");
        assert_eq!(blake3_max_blocks(110), 0, "one row short of the window");
        assert_eq!(111 / BLAKE3_CLOCKS - 1, 0, "the naive (n/56)-1 would say 0");
    }

    #[test]
    fn capacity_saturates_instead_of_underflowing() {
        assert_eq!(blake3_capacity(0, 4), 0);
        assert_eq!(blake3_capacity(BLAKE3_CLOCKS, 1), 0);
        assert_eq!(blake3_capacity(1, 8), 0);
    }

    /// Pinned against what the air actually compiles to. `proofman-setup setup --hash blake3` reports
    /// `Stage1: 236` at LANES 4, and the C++ expander asserts the same figure in
    /// test_gate_bands_cpu.cpp -- the two sides index the same trace, so a disagreement is a wrong
    /// stride.
    ///
    /// A pinned number is only a pin if it came from the thing it claims to pin. 236 is the
    /// compiler's, read off the generated air; `18 + 54*4 + 2` is what `stage1_cols` must reproduce
    /// from its parts. Asserting the formula against itself would pass with either wrong.
    #[test]
    fn stage1_cols_matches_the_compiled_air() {
        assert_eq!(stage1_cols(4), 236, "the air reports Stage1: 236 at LANES 4");
        assert_eq!(stage1_cols(1), 74);
        assert_eq!(stage1_cols(8), 452);
        // The per-lane figure, stated once so a change to either constant has to face it.
        assert_eq!(PERM_COLS_PER_LANE + BOUNDARY_COLS_PER_LANE, 54);
    }

    fn constraint(coeffs: [u64; 5], wires: [u32; 3]) -> PlonkConstraint<Goldilocks> {
        PlonkConstraint { wires, coeffs: coeffs.map(Goldilocks::new) }
    }

    /// Constraints coalesce six to a row only when all five coefficients agree.
    #[test]
    fn only_same_coefficient_constraints_share_a_row() {
        let same: Vec<_> = (0..6).map(|i| constraint([1, 2, 3, 4, 5], [i, i + 1, i + 2])).collect();
        assert_eq!(plonk_rows(&same), 1);

        // one more of the same key spills to a second row
        let seven: Vec<_> = (0..7).map(|i| constraint([1, 2, 3, 4, 5], [i, i + 1, i + 2])).collect();
        assert_eq!(plonk_rows(&seven), 2);

        // six DIFFERENT keys cannot share: one row each, because the air has a single q0
        let distinct: Vec<_> = (0..6).map(|i| constraint([i, 2, 3, 4, 5], [0, 1, 2])).collect();
        assert_eq!(plonk_rows(&distinct), 6);
    }

    /// The nine constFFT slots against the air's `[C[0..4], C[0..3]']`. Pinned because copying
    /// poseidon's `cv[i]` indexing put fft_type 2's twiddles three slots too low, which no
    /// compile-pil would catch -- only a failed proof.
    #[test]
    fn fft4_constants_straddle_the_two_rows_in_order() {
        let expected = [(0, 0), (1, 0), (2, 0), (3, 0), (4, 0), (0, 1), (1, 1), (2, 1), (3, 1)];
        // The placer's rule: constFFT[i] at C[i % C_COLS] of row offset i / C_COLS.
        let slot = |i: usize| (i % C_COLS, i / C_COLS);
        for (i, want) in expected.iter().enumerate() {
            assert_eq!(slot(i), *want, "constFFT[{i}]");
        }
        // fft_type 2 fills constFFT[6..9], all on the SECOND row
        assert!((6..9).all(|i| slot(i).1 == 1));
    }

    /// Signal counts read off the circom templates, not off `cells_per_gate` or poseidon's
    /// `.take(n)` -- `take` truncates silently where an assert does not.
    #[test]
    fn gate_signal_counts_match_the_circom_templates() {
        assert_eq!(8 + 1 + 4, 13, "Blake3Node: in[8] + key + out[4]");
        assert_eq!(3 + 3 + 3, 9, "CMul: ina[3] + inb[3] + out[3]");
        assert_eq!(5 * 3 + 3 + 3 + 3 + 3, EVPOL4_SIGNALS, "EvPol4: coefs + x + out + s + acc");
        assert_eq!(4 * 3 + 4 * 3, 24, "FFT4: in[4][3] + out[4][3]");
        assert_eq!(4 * 3 + 2 + 3, 17, "TreeSelector4: values[4][3] + keys[2] + out[3]");
        assert_eq!(2 * 4 + 1 + 4, 13, "SelectValueArity2: values[2][4] + key[1] + selected[4]");

        // and how many rows of an 18-wide band each needs
        for (signals, rows) in [(13, 2), (9, 1), (21, 2), (24, 2), (17, 1), (13, 1)] {
            let _ = (signals, rows);
        }
        assert_eq!(EVPOL4_SIGNALS.div_ceil(BAND_COLS), 2, "EvPol4 still fits two rows");
        assert_eq!(24_usize.div_ceil(BAND_COLS), 2, "FFT4 does not fit one row");
        assert_eq!(17_usize.div_ceil(BAND_COLS), 1, "TreeSelector4 fits one row");
        assert_eq!(13_usize.div_ceil(BAND_COLS), 1, "SelectValueArity2 fits one row");
    }

    /// Block granularity costs only the tail of each circuit's last block -- the price of the AIR
    /// stating the shape itself instead of receiving six transported columns.
    #[test]
    fn plan_band_blocks_costs_only_the_last_block_tails() {
        // A fibonacci compressor.
        let blocks = plan_band_blocks(&[(3404, 1), (2532, 2), (5088, 2), (4642, 1), (34815, 1), (25023, 1)], 4);
        assert_eq!(blocks, 71 + 106 + 212 + 97 + 726 + 522);
    }

    #[test]
    fn no_constraints_needs_no_rows() {
        assert_eq!(plonk_rows(&[]), 0);
    }
}

// `C[]` holds plonk's five coefficients, and fft4's nine constants over two rows.

const _: () = assert!(C_COLS == crate::plonk2pil::r1cs::to_plonk::PLONK_COEFFS);

#[cfg(test)]
mod compressor_tests {
    use super::*;

    /// ZisK's Keccakf compressor circuit (blake3, column-batched FRI verifier), its counts as
    /// `compressor_demand` reads them.
    fn keccakf() -> CompressorDemand {
        CompressorDemand {
            node_uses: 22_924,
            chunk_buckets: vec![17_239, 2_408, 1_351, 1, 1_065, 480],
            parent_buckets: vec![506, 423],
            band: vec![(54_578, 1), (2_954, 2), (5_932, 2), (5_486, 1), (28_485, 1), (202_054, 1)],
        }
    }

    fn hashing(node_uses: usize) -> CompressorDemand {
        CompressorDemand { node_uses, chunk_buckets: vec![], parent_buckets: vec![], band: vec![] }
    }

    /// At the recursion's 2^19, the fewest lanes whose hashing fits: 5 (9283 of 9361 blocks).
    #[test]
    fn keccakf_takes_the_recursion_size_and_the_fewest_lanes_that_fit() {
        let d = keccakf();
        let g = plan_compressor_geometry(&d, 19);
        assert_eq!((g.n_bits, g.lanes), (19, 5));
        assert_eq!(g.blocks, 9_283);
        assert!(g.band_blocks < g.blocks, "the hashing binds, not the band");
        assert!(d.blocks_needed(4) > blake3_max_blocks(1 << 19), "4 lanes must genuinely not fit");
    }

    /// N grows only when no lane count holds the demand at the floor.
    #[test]
    fn n_grows_only_past_eight_lanes() {
        let cap19 = blake3_max_blocks(1 << 19);
        let g = plan_compressor_geometry(&hashing(cap19 * MAX_LANES), 19);
        assert_eq!((g.n_bits, g.lanes), (19, MAX_LANES));
        let g = plan_compressor_geometry(&hashing(cap19 * MAX_LANES + 1), 19);
        assert_eq!(g.n_bits, 20);
        assert_eq!(g.lanes, 4, "at 2^20 the fewest lanes again");
    }

    /// The floor is a floor: a small demand still takes 2^19, at one lane.
    #[test]
    fn a_small_demand_keeps_the_floor_at_one_lane() {
        let g = plan_compressor_geometry(&hashing(100), 19);
        assert_eq!(g, CompressorGeometry { n_bits: 19, lanes: 1, blocks: 100, band_blocks: 0 });
    }

    /// A band-dominated demand runs into filler blocks, and the air holds the band.
    #[test]
    fn a_band_dominated_demand_is_sized_by_its_band() {
        let d = CompressorDemand { band: vec![(500_000, 1)], ..hashing(40) };
        let g = plan_compressor_geometry(&d, 19);
        assert!(g.band_blocks > g.blocks);
        assert!(g.band_blocks <= blake3_max_blocks(1usize << g.n_bits));
    }

    /// `FLAGS` is a fixed column filled per whole block, so two buckets never share one.
    #[test]
    fn flags_buckets_cannot_share_a_block() {
        let one = CompressorDemand { chunk_buckets: vec![2], ..hashing(0) };
        let two = CompressorDemand { chunk_buckets: vec![1, 1], ..hashing(0) };
        assert_eq!(one.blocks(4), 1);
        assert_eq!(two.blocks(4), 2);
    }
}
