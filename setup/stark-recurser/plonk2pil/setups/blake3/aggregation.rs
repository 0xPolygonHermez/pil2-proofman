//! Aggregation setup for the blake3 family.
//!
//! Geometry, in trace order: `nNodeBlocks` BLAKE3 blocks of 56 rows, then the one- and two-row
//! custom gates, then a dedicated plonk band. See `blake3/aggregator.pil`.
//!
//! **Plonk placement is deliberately simple.** The air declares a single coefficient set (`C[5]`,
//! `q0`) shared by all six `plonk` calls, and it adds no piggyback to the custom-gate rows. So the
//! six gates of a row are interchangeable but must share their coefficients, and planning reduces
//! to: group the constraints by coefficient key, and let each group of at most six fill one row.
//! Poseidon needs a tier system because it has two coefficient sets per row and piggybacks on
//! poseidon/cmul/evpol rows; none of that applies here.
//!
//! One coefficient set rather than two, because of how recursion r1cs files are shaped:
//! Real recursion r1cs files have a wildly skewed key distribution -- one key is 64-75% of all
//! constraints and there are only ~1-2k distinct keys against 240k-590k constraints -- so grouping
//! wastes almost nothing:
//!
//! | r1cs | constraints | keys | rows, one q | rows, two q | ideal |
//! |---|---|---|---|---|---|
//! | FibonacciSquare_recursive1 | 428,214 | 1,938 | 72,387 | 71,680 | 71,369 |
//! | FibonacciSquare_compressor | 588,717 | 1,059 | 98,686 | 98,333 | 98,120 |
//! | vadcop_final | 242,342 | 1,921 | 41,546 | 40,700 | 40,391 |
//!
//! One q costs 0.6-2.9% more rows than a perfect packing, and two q would recover only about half
//! of that -- for five more fixed columns and the whole tier machinery. See the `measure` module.
//!
//! The rows the interior of a BLAKE3 block leaves free are plonk rows like any other -- `PLONK` is
//! 1 there -- so they are spent before any dedicated row is added.

use super::{
    blake3_max_blocks, compress_signal, gen_pil_str, stage1_cols, BandLayout, PilTemplateParams, AGGREGATOR_LAYOUT,
    BLAKE3_CLOCKS, CLOCK_WRAP_ROWS,
};
use crate::plonk2pil::merge_copies::{apply_remap_to_s_map, r1cs2plonk_merged, verify_merge_soundness};
use crate::plonk2pil::r1cs::to_plonk::{
    blake3_compress_gate_uses, ckey, filter_fft4_gate_uses, filter_gate_uses, get_custom_gates_info, PlonkConstraint,
};
use crate::plonk2pil::r1cs::types::{FixedPol, GateBand, GateBandKind, PlonkOptions, R1csFile, SetupResult};
use crate::plonk2pil::utils::{build_fixed_pols, build_s_polynomials, mulp};
use proofman_common::hash_family::GateRole;
use std::collections::HashMap;

fn rand_hex() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    format!("{:x}", SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_nanos() as u64)
}

/// Plonk gates on one row of the `a[0..17]` band: six, at three wires each.
pub const PLONK_GATES_PER_ROW: usize = 6;

/// `coefs[5][3] + x[3] + out[3] + s[3] + acc[3]`; the last six are the Estrin intermediates, which
/// every family's gate publishes and only this air binds.
pub const EVPOL4_SIGNALS: usize = 27;

/// How many rows a set of plonk constraints needs, and how they split between the free interior of
/// the BLAKE3 blocks and a dedicated band.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PlonkPlan {
    /// Rows the constraints need in total.
    pub rows_needed: usize,
    /// Of those, rows taken from the interior of BLAKE3 blocks (free -- `PLONK` is 1 there).
    pub rows_in_blocks: usize,
    /// Rows that need a dedicated band after the custom gates. This is `nPlonkRows`.
    pub rows_dedicated: usize,
}

/// Rows the interior of `blocks` BLAKE3 blocks leaves free for plonk.
///
/// A block's first `lanes` rows carry the lanes' inputs and its last `lanes` rows their outputs, so
/// `PLONK` is 0 there and 1 on the `56 - 2*lanes` rows between. At LANES=4 that is 48 rows per
/// block, or 288 plonk gates.
pub fn plonk_rows_inside_blocks(blocks: usize, lanes: usize, clocks: usize) -> usize {
    blocks * clocks.saturating_sub(2 * lanes)
}

/// Group the constraints by coefficient key and lay them out six to a row.
///
/// Grouping is what the single coefficient set forces: two constraints may share a row only if all
/// five of their coefficients agree, which is exactly `ckey`. A key with 13 constraints takes three
/// rows, the last one two-thirds empty -- that waste is the price of one `q` instead of two, and it
/// is visible in `rows_needed` rather than hidden.
pub fn plan_plonk_rows(
    constraints: &[PlonkConstraint],
    blocks: usize,
    lanes: usize,
    clocks: usize,
    gates_per_row: usize,
) -> PlonkPlan {
    let mut per_key: HashMap<String, usize> = HashMap::new();
    for c in constraints {
        *per_key.entry(ckey(c)).or_insert(0) += 1;
    }
    let rows_needed: usize = per_key.values().map(|n| n.div_ceil(gates_per_row)).sum();

    let available = plonk_rows_inside_blocks(blocks, lanes, clocks);
    let rows_in_blocks = rows_needed.min(available);
    PlonkPlan { rows_needed, rows_in_blocks, rows_dedicated: rows_needed - rows_in_blocks }
}

/// Where `constFFT[i]` lives in the blake3 band.
///
/// `fft4` reads nine constants, but this air has only `C[5]`, so they straddle the gate's two rows:
/// `[C0..C4]` on the gate row and `[C0'..C3']` on the next. Poseidon has `C[10]` and needs no
/// straddle, which is why its packer writes `cv[i]` directly -- copying that indexing here silently
/// put fft_type-2's three twiddles into `constFFT[0..3]` instead of `constFFT[6..9]`.
fn fft4_const_slot(i: usize, c_cols: usize) -> (usize, usize) {
    if i < c_cols {
        (i, 0)
    } else {
        (i - c_cols, 1)
    }
}

/// How many of the air's blocks the band's six circuits consume.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BandPlan {
    /// Blocks whose interiors the band occupies.
    pub blocks: usize,
    /// Interior rows lost to the tail of each circuit's last block.
    pub tail_waste: usize,
}

/// What the band costs once every BLAKE3 block's interior is available to every circuit.
///
/// All six circuits bind the same `a[0..17]`, so a row hosts exactly one of them; each takes a whole
/// number of blocks and fills their interiors. Block granularity is what keeps the AIR's selectors
/// expressible as repetitions -- see the comment on them in `blake3/aggregator.pil` -- and its only
/// cost is the tail of each circuit's last block.
/// `rows` is `[cmul, evPol4, fft4, treeSelector4, selectValArity2, plonk]` -- the order the PIL lays
/// the bands out in, which is also `CompressorDemand::band_rows_by_circuit`. One array rather than
/// six positional arguments: the order is load-bearing and six same-typed parameters hide a swap.
pub fn plan_band_blocks(rows: [usize; 6], lanes: usize, layout: &BandLayout) -> BandPlan {
    let [cmul_rows, ev_pol4, fft4, tree, sel_val, plonk_rows] = rows;
    let interior = BLAKE3_CLOCKS - 2 * lanes;
    // A gate that spans two rows fits `interior / 2` per block, and each one it does not fill wastes
    // both of its rows -- which is what `step` counts.
    let per = |rows_per_gate: usize| interior / rows_per_gate;
    let mut blocks = 0;
    let mut tail_waste = 0;
    for (rows, per_block, step) in [
        (cmul_rows, interior, 1),
        (ev_pol4, per(layout.evpol4_rows), layout.evpol4_rows),
        (fft4, per(layout.fft4_rows), layout.fft4_rows),
        (tree, interior, 1),
        (sel_val, interior, 1),
        (plonk_rows, interior, 1),
    ] {
        let b = rows.div_ceil(per_block.max(1));
        blocks += b;
        tail_waste += (b * per_block - rows) * step;
    }
    BandPlan { blocks, tail_waste }
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
struct RowAlloc {
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
    fn new(lanes: usize) -> Self {
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
    fn open(&mut self, rows: usize, step: usize) {
        let per_block = (self.interior() / step).max(1);
        self.step = step;
        self.enter(self.next_block);
        self.next_block += rows.div_ceil(per_block);
    }

    /// The first of `step` consecutive rows, from the current run.
    fn take(&mut self) -> usize {
        if self.row + self.step > self.end {
            self.enter(self.block + 1);
        }
        let r = self.row;
        self.row += self.step;
        r
    }

    /// Blocks the band has consumed.
    fn blocks_used(&self) -> usize {
        self.next_block
    }
}

/// Coefficients a plonk constraint carries: the array is 3 signals then the `q` values.
pub const PLONK_COEFFS: usize = std::mem::size_of::<PlonkConstraint>() / std::mem::size_of::<u64>() - 3;

/// Writes one plonk row's coefficients into the `C[..]` columns.
///
/// Bounded by the CONSTRAINT, never by `cv.len()`: the two are not the same number. `C[]` is sized
/// for the widest circuit that uses it -- on the 27-column band that is fft4, with nine constants on
/// one row -- while plonk always has five. Iterating over `cv` read four `u64` past the end of a
/// `[u64; 8]` constraint, which is an out-of-bounds panic on the compressor's very first plonk row.
/// The columns above `PLONK_COEFFS` stay zero, and the PIL's plonk gate reads only `C[0..5]`.
fn write_plonk_coeffs(cv: &mut [Vec<u64>], row: usize, c: &PlonkConstraint) {
    for j in 0..PLONK_COEFFS {
        cv[j][row] = c[3 + j];
    }
}

/// Groups `Blake3Compress` uses by their `flags` template parameter, lowest value first.
///
/// `flags` reaches the air as a fixed column filled per whole 56-row block, so the uses that share a
/// block must agree on it. Sorted by value rather than left in arrival order so the resulting s_map
/// is a function of the r1cs alone -- the same reason the plonk gates are ordered explicitly.
pub(super) fn bucket_by_flags(
    uses: Vec<(&crate::plonk2pil::r1cs::types::CustomGateUse, u64, u64)>,
) -> Vec<Vec<(&crate::plonk2pil::r1cs::types::CustomGateUse, u64, u64)>> {
    let mut by_flags: std::collections::BTreeMap<u64, Vec<_>> = std::collections::BTreeMap::new();
    for u in uses {
        by_flags.entry(u.1).or_default().push(u);
    }
    by_flags.into_values().collect()
}

pub fn aggregation_blake3(r1cs: &R1csFile, options: &PlonkOptions) -> SetupResult {
    build_blake3_air(r1cs, options, &AGGREGATOR_LAYOUT)
}

/// The one placement routine both blake3 recursion airs share; `layout` is all they differ by.
///
/// The BLAKE3 block half -- boundary cells, flags, the gate-band list -- is identical between them,
/// which is why this is one function and not two. The band half reads its packing off the layout.
pub fn build_blake3_air(r1cs: &R1csFile, options: &PlonkOptions, layout: &BandLayout) -> SetupResult {
    let (plonk_constraints, plonk_additions, copy_merge) = r1cs2plonk_merged(r1cs, options.merge_copies);

    let mut cgi = get_custom_gates_info(r1cs);
    let lanes = options.blake3_lanes.unwrap_or(4);
    assert!((1..=8).contains(&lanes), "LANES must be in 1..8 (the air's boundary depth caps it), got {lanes}");

    // Blake3Compress carries `(flags, isParent)` as TEMPLATE PARAMETERS, so circom mints one gate id
    // per distinct pair and the r1cs records the values. isParent picks the block kind and flags
    // becomes a fixed column; neither is a trace cell, and neither could be recovered from
    // `cgu.signals` -- they live on the gate, not the use.
    let compress_uses = blake3_compress_gate_uses(&r1cs.custom_gates_uses, &cgi.blake3_compress_parameters);
    let (parent_uses, chunk_uses): (Vec<_>, Vec<_>) = compress_uses.into_iter().partition(|(_, _, ip)| *ip == 1);
    let (n_chunk_uses, n_parent_uses) = (chunk_uses.len(), parent_uses.len());

    // `flags` is a FIXED column the air fills per whole block, so every use sharing a block has to
    // share its flags value. Bucket by flags and give each bucket whole blocks. Packing uses in
    // arrival order instead keeps whichever flags value was written last for the entire block, and
    // `(BLAKE3_CHUNK * 3'CLK_0) * (vd - FLAGS) === 0` -- st[15] is flags at clock 3 -- then fails on
    // every block that mixed two values.
    let chunk_buckets = bucket_by_flags(chunk_uses);
    let parent_buckets = bucket_by_flags(parent_uses);
    let blocks_in = |buckets: &[Vec<(&crate::plonk2pil::r1cs::types::CustomGateUse, u64, u64)>]| {
        buckets.iter().map(|b| b.len().div_ceil(lanes)).sum::<usize>()
    };

    let n_node = cgi.n(GateRole::Blake3Node);
    let n_node_blocks = n_node.div_ceil(lanes);
    let n_chunk_blocks = blocks_in(&chunk_buckets);
    let n_parent_blocks = blocks_in(&parent_buckets);
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
    let n_blocks = n_node_blocks + n_chunk_blocks + n_parent_blocks;
    let n_blake3_rows = n_blocks * BLAKE3_CLOCKS;

    let n_cmul_rows = cgi.n(GateRole::CMul).div_ceil(layout.cmul_per_row);
    let n_ev_pol4 = cgi.n(GateRole::EvPol4);
    let n_fft4 = cgi.n(GateRole::Fft4);
    let n_tree_selector4 = cgi.n(GateRole::TreeSelector);
    let n_select_val_arity2 = cgi.n(GateRole::SelectValArity2);
    // Packed like cmul: several gates share a row where the band is wide enough for them, so the
    // ROW count is what the plan and the allocator want, not the gate count.
    let n_sel_val_rows = n_select_val_arity2.div_ceil(layout.selval_per_row);

    // How many rows each gate takes is the layout's; on the 18-column band EvPol4 (27 signals) and
    // FFT4 (24) need two, on the 27-column band everything fits one.
    let n_gate_rows =
        n_cmul_rows + layout.evpol4_rows * n_ev_pol4 + layout.fft4_rows * n_fft4 + n_tree_selector4 + n_sel_val_rows;

    // Every block's interior is offered to every one of the six band circuits, not just to plonk and
    // not just the Node blocks. Appending the gates in dedicated bands after the BLAKE3 rows instead
    // costs rows the interiors already have free, and enough of them to push the air to the next
    // power of two.
    let n_plonk_rows =
        plan_plonk_rows(&plonk_constraints, 0, lanes, BLAKE3_CLOCKS, layout.plonk_gates_per_row).rows_needed;
    let plan = plan_band_blocks(
        [n_cmul_rows, n_ev_pol4, n_fft4, n_tree_selector4, n_sel_val_rows, n_plonk_rows],
        lanes,
        layout,
    );
    cgi.n_plonk_rows = n_plonk_rows;

    // Reported in one place, with the arithmetic visible: "N gate rows" says nothing about which
    // gate, and a constraint count says nothing about rows until the packing is spelled out.
    let ideal_plonk_rows = plonk_constraints.len().div_ceil(layout.plonk_gates_per_row);
    tracing::info!(
        "Plonk: {} constraints -> {} rows ({} per row, grouped by coefficient key; {} rows part-filled)",
        plonk_constraints.len(),
        n_plonk_rows,
        layout.plonk_gates_per_row,
        n_plonk_rows.saturating_sub(ideal_plonk_rows)
    );
    tracing::info!(
        "Gate rows: cmul {} ({} gates, {} per row) + fft4 {} + evPol4 {} + treeSelector4 {} + selectValArity2 {} = {}",
        n_cmul_rows,
        cgi.n(GateRole::CMul),
        layout.cmul_per_row,
        layout.fft4_rows * n_fft4,
        layout.evpol4_rows * n_ev_pol4,
        n_tree_selector4,
        n_sel_val_rows,
        n_gate_rows
    );
    tracing::info!(
        "Band: {} gate + {} plonk = {} rows in {} of {} block interiors ({} available, {} lost to block tails)",
        n_gate_rows,
        n_plonk_rows,
        n_gate_rows + n_plonk_rows,
        plan.blocks,
        n_blocks,
        plonk_rows_inside_blocks(n_blocks, lanes, BLAKE3_CLOCKS),
        plan.tail_waste
    );
    tracing::info!(
        "BLAKE3: {} blocks x {} rows = {} rows, which is what sizes the air",
        n_blocks,
        BLAKE3_CLOCKS,
        n_blake3_rows
    );

    // The air is as tall as the LARGER of the two, not as tall as the hashing. A band that wants
    // more block interiors than the hashing pays for does not make the air unbuildable -- it makes
    // it taller -- and reporting only the hashing understated `n_bits_natural`, which is what
    // decides whether this air needs a compressor. Panicking here pre-empted that decision
    // entirely: `recursive.rs` catches NeedsCompressorError and retries with a compressor, and a
    // panic cannot be caught. Measured on ZisK's Keccakf recursive1, where the band needs 22024
    // interiors against the hashing's 11283.
    if plan.blocks > n_blocks {
        tracing::warn!(
            "plonk-dominated: the band needs {} block interiors against the hashing's {n_blocks}, so \
             the band sizes this air. A compressor is what fixes this if the result is too tall.",
            plan.blocks
        );
    }
    let n_used = plan.blocks.max(n_blocks) * BLAKE3_CLOCKS;
    // The wrap window is part of what the air must hold, not slack on top of it: the clock selectors
    // read backwards, so the last CLOCKS-1 rows cannot carry a block. Sizing from `n_used` alone
    // leaves the air one block short whenever `blocks * 56` lands just under a power of two -- 16
    // block counts below 40,000 do, each missing by exactly one, and each of them tripped the
    // capacity assert below.
    let sized = n_used + CLOCK_WRAP_ROWS;
    // ceil(log2(sized)) in usize. The u32 `log2` helper would truncate a demand past 2^32 rows into a
    // small n_bits, and the compressor's planner is allowed to hand us n_bits above its preferred cap.
    let n_bits = sized.next_power_of_two().trailing_zeros().max(1) as usize;
    // Never below the floor: an air reusing another air's starkSetup has to match its rows. The
    // pre-floor size is kept -- it is what decides whether the circuit itself is too big.
    let n_bits_natural = n_bits;
    let n_bits = n_bits.max(options.min_n_bits.unwrap_or(0));
    let n = 1usize << n_bits;

    // The clock selectors read backwards and wrap, so the last CLOCKS-1 rows must stay padding.
    let capacity = blake3_max_blocks(n);
    // Against what the air actually has to hold, not just the hashing: when the band is the dominant
    // side, `n_blocks` alone fits while the band's interiors do not, and the placement would then run
    // off the end of the trace with a far less legible index panic.
    let needed = plan.blocks.max(n_blocks);
    assert!(
        needed <= capacity,
        "{n_node} Blake3Node gates need {n_blocks} blocks of {BLAKE3_CLOCKS} rows and the band needs \
         {} interiors, so {needed} blocks in total, but an air of {n} rows holds only {capacity} (the \
         last {} rows are the clock selectors' wrap window). Raise LANES or N.",
        plan.blocks,
        BLAKE3_CLOCKS - 1
    );

    let n_publics = r1cs.header.n_outputs + r1cs.header.n_pub_inputs;
    let max_degree = options.max_constraint_degree.unwrap_or(5);
    let airgroup_name = options.airgroup_name.clone().unwrap_or_else(|| format!("Blake3Agg{}", rand_hex()));

    let pil_str = gen_pil_str(&PilTemplateParams {
        template_file: layout.template,
        template_name: layout.template_name,
        namespace_name: &airgroup_name,
        n_bits,
        n_publics,
        max_constraint_degree: max_degree,
        n_plonk_rows,
        n_cmul_rows,
        n_ev_pol4,
        n_fft4,
        n_tree_selector4,
        n_sel_val_rows,
        n_node_blocks,
        n_chunk_blocks,
        n_parent_blocks,
        lanes,
    });

    tracing::info!("NUsed: {n_used}, nBits: {n_bits}, N: {n}, blocks: {n_blocks}, LANES: {lanes}");

    let committed = stage1_cols(lanes, layout.band);
    let mut s_map: Vec<Vec<u32>> = (0..committed).map(|_| vec![0u32; n]).collect();
    let mut cv: Vec<Vec<u64>> = (0..layout.c_cols).map(|_| vec![0u64; n]).collect();
    let mut gate_bands: Vec<GateBand> = Vec::new();

    let node_uses = filter_gate_uses(&r1cs.custom_gates_uses, cgi.role_id(GateRole::Blake3Node));
    let cmul_uses = filter_gate_uses(&r1cs.custom_gates_uses, cgi.role_id(GateRole::CMul));
    let fft4_uses = filter_fft4_gate_uses(&r1cs.custom_gates_uses, &cgi.fft4_parameters);
    let ev_pol4_uses = filter_gate_uses(&r1cs.custom_gates_uses, cgi.role_id(GateRole::EvPol4));
    let tree_uses = filter_gate_uses(&r1cs.custom_gates_uses, cgi.role_id(GateRole::TreeSelector));
    let sel_val_uses = filter_gate_uses(&r1cs.custom_gates_uses, cgi.role_id(GateRole::SelectValArity2));

    // ── BLAKE3 blocks: boundary cells only ────────────────────────────────────
    // The 59 columns per lane are the expander's; the setup writes the two boundary rows. Lane l of
    // a block reads its input at clock l and writes its output at clock 56 - LANES + l (spec 3.2).
    // Kinds are grouped so the air's block-wide selectors stay compact patterns: Node, then chunk,
    // then parent.
    tracing::info!(
        "Processing {} Node, {} chunk, {} parent gates in {n_blocks} blocks...",
        node_uses.len(),
        n_chunk_uses,
        n_parent_uses
    );
    // `flags` is st[15], a per-block constant read off the gate id's parameters.
    let mut flags_col = vec![0u64; n];

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
    let place_compress = |buckets: &[Vec<(&crate::plonk2pil::r1cs::types::CustomGateUse, u64, u64)>],
                          block_base: usize,
                          s_map: &mut Vec<Vec<u32>>,
                          flags_col: &mut Vec<u64>|
     -> usize {
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
    let after_chunk = place_compress(&chunk_buckets, n_node_blocks, &mut s_map, &mut flags_col);
    let after_parent = place_compress(&parent_buckets, after_chunk, &mut s_map, &mut flags_col);
    // The air is sized from n_chunk_blocks/n_parent_blocks; if the placement walked past them the
    // trace would run off the end of a shorter air, so tie the two together rather than trusting it.
    assert_eq!(after_chunk, n_node_blocks + n_chunk_blocks, "chunk placement disagrees with its block count");
    assert_eq!(after_parent, n_blocks, "parent placement disagrees with its block count");

    // Blake3Node freezes flags to CHUNK_START | CHUNK_END | ROOT.
    flags_col[..n_node_blocks * BLAKE3_CLOCKS].fill(11);

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
        gate_bands.push(GateBand { row: row as u32, kind, payload: flags_col[row] });
    }

    // Runs are opened in the order blake3/aggregator.pil lays the selectors out. The AIR states
    // where each circuit lives; this only has to agree with it.
    let mut alloc = RowAlloc::new(lanes);

    // ── CMul: 9 signals each, `cmul_per_row` to a row ─────────────────────────
    tracing::info!("Processing {} cmul gates...", cmul_uses.len());
    alloc.open(n_cmul_rows, 1);
    let mut r = 0usize;
    for (i, cgu) in cmul_uses.iter().enumerate() {
        assert_eq!(cgu.signals.len(), 9);
        let half = i % layout.cmul_per_row;
        if half == 0 {
            r = alloc.take();
        }
        for (j, sig) in cgu.signals.iter().enumerate() {
            s_map[half * 9 + j][r] = *sig as u32;
        }
    }

    // ── EvPol4: 27 signals, over `evpol4_rows` rows of the band ───────────────
    // The poseidon airs place only the first 21 and bind nothing to the rest.
    tracing::info!("Processing {} evPol4 gates...", ev_pol4_uses.len());
    alloc.open(n_ev_pol4, layout.evpol4_rows);
    for cgu in &ev_pol4_uses {
        assert_eq!(cgu.signals.len(), EVPOL4_SIGNALS);
        let r = alloc.take();
        // 27 signals: one row where the band is at least that wide, otherwise the first `band` on the
        // gate row and the rest on the next -- which is what the PIL reads through primes.
        for (j, sig) in cgu.signals.iter().enumerate() {
            s_map[j % layout.band][r + j / layout.band] = *sig as u32;
        }
    }

    // ── FFT4: 24 signals, in at a[0..12] and out over `fft4_rows` rows ───────
    tracing::info!("Processing {} fft4 gates...", fft4_uses.len());
    alloc.open(n_fft4, layout.fft4_rows);
    for cgu in &fft4_uses {
        assert_eq!(cgu.signals.len(), 24);
        let r = alloc.take();
        // Two rows: in on the gate row, out on the next, both at a[0..12]. One row: in at a[0..12]
        // and out at a[12..24], which needs a band of 24.
        for (j, pair) in cgu.signals[..12].iter().zip(&cgu.signals[12..24]).enumerate() {
            s_map[j][r] = *pair.0 as u32;
            if layout.fft4_rows == 2 {
                s_map[j][r + 1] = *pair.1 as u32;
            } else {
                s_map[12 + j][r] = *pair.1 as u32;
            }
        }
        // Build constFFT by ITS OWN index, exactly as poseidon does, then scatter through
        // fft4_const_slot. Writing cv[..] directly would silently misplace fft_type 2.
        let p = cgi.fft4_parameters.get(&cgu.id).expect("FFT4 params");
        let (fft_type, scale, first_w, inc_w) = (p[3], p[2], p[0], p[1]);
        let fw2 = mulp(first_w, first_w);
        let mut cfft = [0u64; 9];
        if fft_type == 4 {
            cfft[0] = scale;
            cfft[1] = mulp(scale, fw2);
            cfft[2] = mulp(scale, first_w);
            cfft[3] = mulp(mulp(scale, first_w), fw2);
            cfft[4] = mulp(mulp(scale, first_w), inc_w);
            cfft[5] = mulp(mulp(mulp(scale, first_w), fw2), inc_w);
        } else if fft_type == 2 {
            cfft[6] = scale;
            cfft[7] = mulp(scale, first_w);
            cfft[8] = mulp(mulp(scale, first_w), inc_w);
        } else {
            panic!("Invalid FFT4 type: {fft_type}");
        }
        for (i, v) in cfft.iter().enumerate() {
            let (col, off) = fft4_const_slot(i, layout.c_cols);
            cv[col][r + off] = *v;
        }
    }

    // ── TreeSelector4: 17 signals in one row ─────────────────────────────────
    tracing::info!("Processing {} treeSelector4 gates...", tree_uses.len());
    alloc.open(n_tree_selector4, 1);
    for cgu in &tree_uses {
        assert_eq!(cgu.signals.len(), 17, "blake3 uses radix 4 (17 signals), not radix 8 (30)");
        let r = alloc.take();
        for (j, sig) in cgu.signals.iter().enumerate() {
            s_map[j][r] = *sig as u32;
        }
    }

    // ── SelectValueArity2: 13 signals each, `selval_per_row` to a row ────────
    tracing::info!("Processing {} selectValueArity2 gates...", sel_val_uses.len());
    alloc.open(n_sel_val_rows, 1);
    let mut sv_row = 0usize;
    for (i, cgu) in sel_val_uses.iter().enumerate() {
        assert_eq!(cgu.signals.len(), 13, "blake3 is an arity-2 family (13 signals), not arity 4 (22)");
        let slot = i % layout.selval_per_row;
        if slot == 0 {
            sv_row = alloc.take();
        }
        for (j, sig) in cgu.signals.iter().enumerate() {
            s_map[slot * 13 + j][sv_row] = *sig as u32;
        }
    }

    // ── Plonk constraints ─────────────────────────────────────────────────────
    // One coefficient set per row, so a row holds up to six constraints that share `ckey`. A row
    // that ends partly filled has its LAST constraint duplicated into the free gates: every gate's
    // constraint fires wherever PLONK is 1, so an untouched gate would read signal 0 in all three
    // wires and fail. Duplication is what poseidon's `partial`/`half` slots do too.
    tracing::info!("Placing {} plonk constraints...", plonk_constraints.len());
    let mut by_key: HashMap<String, Vec<&PlonkConstraint>> = HashMap::new();
    for c in &plonk_constraints {
        by_key.entry(ckey(c)).or_default().push(c);
    }
    // Deterministic order: the s_map must not depend on HashMap iteration order.
    let mut keys: Vec<String> = by_key.keys().cloned().collect();
    keys.sort_unstable();

    alloc.open(n_plonk_rows, 1);
    let mut next_row = 0usize;
    for k in &keys {
        for group in by_key[k].chunks(layout.plonk_gates_per_row) {
            let row = alloc.take();
            next_row += 1;
            write_plonk_coeffs(&mut cv, row, group[0]);
            for gate in 0..layout.plonk_gates_per_row {
                let c = group[gate.min(group.len() - 1)];
                s_map[3 * gate][row] = c[0] as u32;
                s_map[3 * gate + 1][row] = c[1] as u32;
                s_map[3 * gate + 2][row] = c[2] as u32;
            }
        }
    }
    assert_eq!(next_row, n_plonk_rows, "plonk row plan disagreed with the placement");
    assert_eq!(
        alloc.blocks_used(),
        plan.blocks,
        "the placement consumed a different number of blocks than the plan, so the AIR's selector \
         patterns describe different rows than the ones just written"
    );

    // ── S polynomials ─────────────────────────────────────────────────────────
    apply_remap_to_s_map(&mut s_map, &copy_merge.remap);
    verify_merge_soundness(&s_map, &copy_merge.merged_reps, layout.band);
    let sv = build_s_polynomials(layout.band, n, n_bits, n_used, &s_map);
    let mut fixed_pols = build_fixed_pols(&airgroup_name, &cv, &sv);
    fixed_pols.push(FixedPol { name: format!("{airgroup_name}.FLAGS"), index: 0, values: flags_col });

    SetupResult {
        gate_bands,
        fixed_pols,
        pil_str,
        n_bits,
        n_bits_natural,
        n_used,
        s_map,
        plonk_additions,
        airgroup_name: airgroup_name.clone(),
        air_name: airgroup_name,
        // LANES and the band width, packed: LANES low, band high. Both are setup parameters the
        // expander cannot derive -- the column count alone cannot tell (band 18, LANES 8) from
        // (band 27, LANES 8) apart -- and the band decides where every lane column starts. Taking
        // it from a constant put the compressor's lanes 9 columns too low.
        band_aux: (lanes as u64) | ((layout.band as u64) << 32),
    }
}

#[cfg(test)]
mod tests {
    use super::super::{
        blake3_capacity, blake3_max_blocks, stage1_cols, BAND_COLS, BLAKE3_CLOCKS, BOUNDARY_COLS_PER_LANE,
        CLOCK_WRAP_ROWS, COMPRESSOR_BAND_COLS, PERM_COLS_PER_LANE,
    };
    use super::*;

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
        let tagged: Vec<_> = uses.iter().zip(flags).map(|(u, f)| (u, f, 0u64)).collect();

        let buckets = bucket_by_flags(tagged);

        assert_eq!(buckets.iter().map(|b| b.len()).sum::<usize>(), 10, "no use may be dropped");
        for b in &buckets {
            assert!(b.iter().all(|u| u.1 == b[0].1), "a bucket has to be one flags value");
        }
        let seen: Vec<u64> = buckets.iter().map(|b| b[0].1).collect();
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
    /// `Stage1: 256` at LANES 4, and the C++ expander asserts the same figure in
    /// test_gate_bands_cpu.cpp -- the two sides index the same trace, so a disagreement is a wrong
    /// stride.
    ///
    /// A pinned number is only a pin if it came from the thing it claims to pin. 256 is the
    /// compiler's, read off the generated air; `18 + 59*4 + 2` is what `stage1_cols` must reproduce
    /// from its parts. Asserting the formula against itself would pass with either wrong.
    #[test]
    fn stage1_cols_matches_the_compiled_air() {
        assert_eq!(stage1_cols(4, BAND_COLS), 256, "the air reports Stage1: 256 at LANES 4");
        assert_eq!(stage1_cols(1, BAND_COLS), 79);
        assert_eq!(stage1_cols(8, BAND_COLS), 492);
        // The compressor's wider band, for the same lanes: nine more columns.
        assert_eq!(stage1_cols(4, COMPRESSOR_BAND_COLS), 265);
        assert_eq!(stage1_cols(3, COMPRESSOR_BAND_COLS), 206);
        // The per-lane figure, stated once so a change to either constant has to face it.
        assert_eq!(PERM_COLS_PER_LANE + BOUNDARY_COLS_PER_LANE, 59);
    }

    /// `PLONK` in the air is `[[0:LANES, 1:(56-2*LANES), 0:LANES]:nNodeBlocks, ...]`.
    #[test]
    fn interior_rows_match_the_airs_plonk_pattern() {
        assert_eq!(plonk_rows_inside_blocks(1, 4, BLAKE3_CLOCKS), 48);
        assert_eq!(plonk_rows_inside_blocks(10, 4, BLAKE3_CLOCKS), 480);
        assert_eq!(plonk_rows_inside_blocks(1, 1, BLAKE3_CLOCKS), 54);
        assert_eq!(plonk_rows_inside_blocks(1, 8, BLAKE3_CLOCKS), 40);
        // LANES cannot exceed 8, but saturate rather than underflow if it ever did
        assert_eq!(plonk_rows_inside_blocks(1, 28, BLAKE3_CLOCKS), 0);
        assert_eq!(plonk_rows_inside_blocks(1, 40, BLAKE3_CLOCKS), 0);
    }

    fn constraint(coeffs: [u64; 5], wires: [u64; 3]) -> PlonkConstraint {
        [wires[0], wires[1], wires[2], coeffs[0], coeffs[1], coeffs[2], coeffs[3], coeffs[4]]
    }

    /// Constraints coalesce six to a row only when all five coefficients agree.
    #[test]
    fn only_same_coefficient_constraints_share_a_row() {
        let same: Vec<_> = (0..6).map(|i| constraint([1, 2, 3, 4, 5], [i, i + 1, i + 2])).collect();
        assert_eq!(plan_plonk_rows(&same, 0, 4, BLAKE3_CLOCKS, AGGREGATOR_LAYOUT.plonk_gates_per_row).rows_needed, 1);

        // one more of the same key spills to a second row
        let seven: Vec<_> = (0..7).map(|i| constraint([1, 2, 3, 4, 5], [i, i + 1, i + 2])).collect();
        assert_eq!(plan_plonk_rows(&seven, 0, 4, BLAKE3_CLOCKS, AGGREGATOR_LAYOUT.plonk_gates_per_row).rows_needed, 2);

        // six DIFFERENT keys cannot share: one row each, because the air has a single q0
        let distinct: Vec<_> = (0..6).map(|i| constraint([i, 2, 3, 4, 5], [0, 1, 2])).collect();
        assert_eq!(
            plan_plonk_rows(&distinct, 0, 4, BLAKE3_CLOCKS, AGGREGATOR_LAYOUT.plonk_gates_per_row).rows_needed,
            6
        );
    }

    /// Interior rows are spent before any dedicated row is added.
    #[test]
    fn block_interiors_are_spent_before_dedicated_rows() {
        let many: Vec<_> = (0..300u64).map(|i| constraint([1, 2, 3, 4, 5], [i, i, i])).collect();
        assert_eq!(many.len().div_ceil(AGGREGATOR_LAYOUT.plonk_gates_per_row), 50);

        // one block at LANES=4 offers 48 interior rows
        let p = plan_plonk_rows(&many, 1, 4, BLAKE3_CLOCKS, AGGREGATOR_LAYOUT.plonk_gates_per_row);
        assert_eq!(p, PlonkPlan { rows_needed: 50, rows_in_blocks: 48, rows_dedicated: 2 });

        // two blocks swallow all of it
        let p = plan_plonk_rows(&many, 2, 4, BLAKE3_CLOCKS, AGGREGATOR_LAYOUT.plonk_gates_per_row);
        assert_eq!(p, PlonkPlan { rows_needed: 50, rows_in_blocks: 50, rows_dedicated: 0 });

        // and with no blocks every row is dedicated
        let p = plan_plonk_rows(&many, 0, 4, BLAKE3_CLOCKS, AGGREGATOR_LAYOUT.plonk_gates_per_row);
        assert_eq!(p, PlonkPlan { rows_needed: 50, rows_in_blocks: 0, rows_dedicated: 50 });
    }

    /// The nine constFFT slots against the air's `[C[0..4], C[0..3]']`. Pinned because copying
    /// poseidon's `cv[i]` indexing put fft_type 2's twiddles three slots too low, which no
    /// compile-pil would catch -- only a failed proof.
    #[test]
    fn fft4_constants_straddle_the_two_rows_in_order() {
        let expected = [(0, 0), (1, 0), (2, 0), (3, 0), (4, 0), (0, 1), (1, 1), (2, 1), (3, 1)];
        for (i, want) in expected.iter().enumerate() {
            assert_eq!(fft4_const_slot(i, 5), *want, "constFFT[{i}]");
        }
        // fft_type 4 fills constFFT[0..6]: five on the gate row, one on the next
        assert!((0..6).all(|i| fft4_const_slot(i, 5).1 == 0 || fft4_const_slot(i, 5) == (0, 1)));
        // fft_type 2 fills constFFT[6..9], all on the SECOND row
        assert!((6..9).all(|i| fft4_const_slot(i, 5).1 == 1));
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

    /// The rows the packer writes must be exactly the rows the AIR's selector is 1 on. Both sides
    /// are stated independently here -- the pattern is re-derived from the PIL's own formula -- so a
    /// drift shows up as a differing row set rather than as a constraint that silently never fires.
    fn pil_pattern_rows(at_block: usize, rows: usize, step: usize, lanes: usize) -> Vec<usize> {
        let interior = BLAKE3_CLOCKS - 2 * lanes;
        let per_block = interior / step;
        let blocks = rows.div_ceil(per_block);
        let mut out = Vec::new();
        for b in 0..blocks {
            let base = (at_block + b) * BLAKE3_CLOCKS + lanes;
            for k in 0..per_block {
                out.push(base + k * step);
            }
        }
        out
    }

    #[test]
    fn row_alloc_matches_the_pil_selector_pattern() {
        for lanes in [1usize, 2, 4, 8] {
            // The six runs in the order blake3/aggregator.pil lays them out.
            let runs = [(70usize, 1usize), (13, 2), (25, 2), (9, 1), (100, 1), (61, 1)];
            let mut alloc = RowAlloc::new(lanes);
            let mut at = 0usize;
            for (rows, step) in runs {
                let want = pil_pattern_rows(at, rows, step, lanes);
                alloc.open(rows, step);
                let got: Vec<usize> = (0..rows).map(|_| alloc.take()).collect();
                assert_eq!(got, want[..rows], "LANES={lanes}, run of {rows} at step {step}");
                let per_block = (BLAKE3_CLOCKS - 2 * lanes) / step;
                at += rows.div_ceil(per_block);
            }
            assert_eq!(alloc.blocks_used(), at, "blocks consumed must match the PIL's run starts");
        }
    }

    /// A pair's second row carries its outputs, so it can never be a block's boundary row.
    #[test]
    fn pairs_stay_inside_one_block() {
        for lanes in [1usize, 2, 4, 8] {
            let interior = BLAKE3_CLOCKS - 2 * lanes;
            assert_eq!(interior % 2, 0, "an odd interior would leave a pair straddling the seam");
            let mut alloc = RowAlloc::new(lanes);
            alloc.open(200, 2);
            for _ in 0..200 {
                let r = alloc.take();
                assert_eq!(r / BLAKE3_CLOCKS, (r + 1) / BLAKE3_CLOCKS, "LANES={lanes}: pair crosses a block");
                let clk = r % BLAKE3_CLOCKS;
                assert!(clk >= lanes && clk + 1 < BLAKE3_CLOCKS - lanes, "LANES={lanes}: pair touches a boundary row");
            }
        }
    }

    /// Block granularity costs only the tail of each circuit's last block -- the price of the AIR
    /// stating the shape itself instead of receiving six transported columns.
    #[test]
    fn plan_band_blocks_costs_only_the_last_block_tails() {
        // A fibonacci compressor.
        let p = plan_band_blocks([3404, 2532, 5088, 4642, 34815, 25023], 4, &AGGREGATOR_LAYOUT);
        assert_eq!(p.blocks, 71 + 106 + 212 + 97 + 726 + 522);
        // fft4 wastes nothing; the other five leave a tail. 108 rows of 433k.
        assert_eq!(p.tail_waste, 4 + 24 + 14 + 33 + 33);
        assert!(p.blocks < 9033, "the band fits well inside the blocks the hashing already needs");
    }

    #[test]
    fn no_constraints_needs_no_rows() {
        assert_eq!(
            plan_plonk_rows(&[], 9361, 4, BLAKE3_CLOCKS, AGGREGATOR_LAYOUT.plonk_gates_per_row),
            PlonkPlan { rows_needed: 0, rows_in_blocks: 0, rows_dedicated: 0 }
        );
    }
}

#[cfg(test)]
mod measure {
    use super::*;
    use crate::plonk2pil::r1cs::to_plonk::r1cs2plonk;
    use crate::plonk2pil::r1cs::types::read_r1cs_from_bytes;

    /// The harness behind the module-level table: how much does ONE coefficient set cost against
    /// two? Kept so the decision can be rechecked against a new circuit rather than trusted.
    ///
    /// `BLAKE3_MEASURE_R1CS=/path/to.r1cs cargo test --release -- --ignored --nocapture measure`
    #[test]
    #[ignore]
    fn coefficient_key_distribution() {
        let path = std::env::var("BLAKE3_MEASURE_R1CS").expect("set BLAKE3_MEASURE_R1CS");
        let data = std::fs::read(&path).unwrap();
        let r1cs = read_r1cs_from_bytes(&data).unwrap();
        let (cs, _adds) = r1cs2plonk(&r1cs);

        let mut per_key: HashMap<String, usize> = HashMap::new();
        for c in &cs {
            *per_key.entry(ckey(c)).or_insert(0) += 1;
        }
        let mut counts: Vec<usize> = per_key.values().copied().collect();
        counts.sort_unstable_by(|a, b| b.cmp(a));

        let one_q: usize = counts.iter().map(|n| n.div_ceil(6)).sum();
        // two q sets of three gates each: a row holds two distinct keys, three constraints apiece
        let halves: usize = counts.iter().map(|n| n.div_ceil(3)).sum();
        let two_q = halves.div_ceil(2);

        println!("{path}");
        println!("  constraints        : {}", cs.len());
        println!("  distinct keys      : {}", counts.len());
        println!("  top key counts     : {:?}", &counts[..counts.len().min(8)]);
        println!("  singleton keys     : {}", counts.iter().filter(|&&n| n == 1).count());
        println!("  rows, ONE q  (6/row): {one_q}");
        println!("  rows, TWO q  (3+3)  : {two_q}");
        println!("  ideal (no key split): {}", cs.len().div_ceil(6));
    }
}

#[cfg(test)]
mod plonk_coeff_tests {
    use super::super::{AGGREGATOR_LAYOUT, COMPRESSOR_LAYOUT};
    use super::{write_plonk_coeffs, PlonkConstraint, PLONK_COEFFS};

    /// `C[]` is sized for the widest circuit sharing it, which on the 27-column band is fft4's nine
    /// constants -- NOT for plonk's five. A write bounded by the column count instead of by the
    /// constraint read past the end of a `[u64; 8]`, panicking on the compressor's first plonk row.
    /// The trap, as a compile-time fact: the compressor's `C[]` really is wider than plonk needs, so
    /// a write bounded by the column count really would run off the end of the constraint.
    const _: () = assert!(COMPRESSOR_LAYOUT.c_cols > PLONK_COEFFS);
    const _: () = assert!(AGGREGATOR_LAYOUT.c_cols == PLONK_COEFFS);

    #[test]
    fn a_wider_c_band_does_not_make_plonk_read_past_its_constraint() {
        assert_eq!(PLONK_COEFFS, 5, "3 signals + 5 q values");

        let c: PlonkConstraint = [101, 102, 103, 11, 22, 33, 44, 55];
        for layout in [AGGREGATOR_LAYOUT, COMPRESSOR_LAYOUT] {
            let mut cv: Vec<Vec<u64>> = (0..layout.c_cols).map(|_| vec![0u64; 4]).collect();
            write_plonk_coeffs(&mut cv, 2, &c);
            for (j, col) in cv.iter().enumerate() {
                // Plonk's five carry the coefficients; everything above them is fft4's and a plonk
                // row must leave it alone. No row other than the target is touched either.
                let want = if j < PLONK_COEFFS { c[3 + j] } else { 0 };
                assert_eq!(col[2], want, "c_cols {}: column {j}", layout.c_cols);
                assert_eq!(col[0], 0, "c_cols {}: column {j} bled into row 0", layout.c_cols);
                assert_eq!(col[3], 0, "c_cols {}: column {j} bled into row 3", layout.c_cols);
            }
        }
    }
}

#[cfg(test)]
mod audit_geometry {
    use super::super::{BLAKE3_CLOCKS as CLOCKS, AGGREGATOR_LAYOUT, COMPRESSOR_LAYOUT};
    use super::*;

    #[derive(Clone, Copy, PartialEq, Eq, Debug)]
    enum Owner {
        InLane(usize),
        OutLane(usize),
        Band,
        Free,
    }

    /// What the AIR says a clock inside a block is for, read straight off the selector shapes.
    fn owner(clk: usize, lanes: usize, band_on: bool) -> Owner {
        if clk < lanes {
            return Owner::InLane(clk);
        }
        if clk >= CLOCKS - lanes {
            return Owner::OutLane(clk - (CLOCKS - lanes));
        }
        if band_on {
            Owner::Band
        } else {
            Owner::Free
        }
    }

    /// The PIL's fixed selector column for one circuit:
    /// `[[0:CLOCKS]:at, [0:LANES, <p>:INTERIOR, 0:LANES]:nblocks, 0...]`
    /// where <p> is `1` at step 1 and `[1,0]` at step 2.
    fn pil_selector(at: usize, nblocks: usize, step: usize, lanes: usize, air_blocks: usize) -> Vec<bool> {
        let interior = CLOCKS - 2 * lanes;
        let mut s = vec![false; air_blocks * CLOCKS];
        for b in at..at + nblocks {
            for k in 0..interior {
                if k % step == 0 {
                    s[b * CLOCKS + lanes + k] = true;
                }
            }
        }
        s
    }

    /// Every row of every block has exactly one owner, and a band row is never a boundary row.
    #[test]
    fn boundary_rows_and_band_rows_partition_the_block_at_every_lane_count() {
        for lanes in 1..=8usize {
            let interior = CLOCKS - 2 * lanes;
            assert_eq!(interior % 2, 0, "LANES={lanes}: odd interior would straddle a pair");
            let mut seen_in = vec![0usize; lanes];
            let mut seen_out = vec![0usize; lanes];
            let mut band = 0usize;
            for clk in 0..CLOCKS {
                match owner(clk, lanes, true) {
                    Owner::InLane(l) => seen_in[l] += 1,
                    Owner::OutLane(l) => seen_out[l] += 1,
                    Owner::Band => band += 1,
                    Owner::Free => unreachable!(),
                }
            }
            assert!(seen_in.iter().all(|&n| n == 1), "LANES={lanes}: an input row is shared or missing");
            assert!(seen_out.iter().all(|&n| n == 1), "LANES={lanes}: an output row is shared or missing");
            assert_eq!(band, interior, "LANES={lanes}: band rows != interior");
            assert_eq!(band + 2 * lanes, CLOCKS);
        }
    }

    /// Every `a[]` read the BLAKE3 half performs lands on that lane's own boundary row, never a
    /// band row. Shifts transcribed from circuits/blake3.pil.
    #[test]
    fn every_blake3_boundary_read_lands_on_a_boundary_row() {
        for lanes in 1..=8usize {
            for lane in 0..lanes {
                // blake3NodeBoundary / blake3CompressBoundaryKind: fires at clock i (0..8),
                // reads through blake3Shift(a[j], i - lane).
                for i in 0..8usize {
                    let fire = i as isize;
                    let read = fire - (i as isize - lane as isize);
                    assert_eq!(read, lane as isize, "LANES={lanes} lane={lane} i={i}: message read drifted");
                    assert!(matches!(owner(read as usize, lanes, true), Owner::InLane(l) if l == lane));
                }
                // blake3CompressBoundaryShared: vd cells at clocks 0..3, blake3Shift(a[..], c - lane).
                for (c, _cell) in [(0usize, 17usize), (2, 16)] {
                    let read = c as isize - (c as isize - lane as isize);
                    assert!(matches!(owner(read as usize, lanes, true), Owner::InLane(l) if l == lane));
                }
                // Output rows: R = CLOCKS - LANES + lane, read in place.
                let r = CLOCKS - lanes + lane;
                assert!(
                    matches!(owner(r, lanes, true), Owner::OutLane(l) if l == lane),
                    "LANES={lanes} lane={lane}: output row {r} is not this lane's"
                );
                // OUT_CLK[R - OUT_FIRST] must index inside the 16-wide clock set.
                let idx = r as isize - (CLOCKS as isize - 16);
                assert!((0..16).contains(&idx), "LANES={lanes} lane={lane}: OUT_CLK index {idx} out of range");
            }
        }
    }

    /// The packer's rows are exactly the rows the AIR's selectors switch on. A gate placed where
    /// its selector is 0 is a gate the AIR never evaluates.
    #[test]
    fn every_placed_gate_row_has_its_selector_on_and_no_row_carries_two() {
        for layout in [&AGGREGATOR_LAYOUT, &COMPRESSOR_LAYOUT] {
            for lanes in 1..=8usize {
                // (rows-or-gates, step) in the order both the PIL and the packer lay them out.
                let runs = [
                    (137usize, 1usize),       // cmul rows
                    (61, layout.evpol4_rows), // evpol4 gates
                    (49, layout.fft4_rows),   // fft4 gates
                    (23, 1),                  // treeselector rows
                    (91, 1),                  // selval rows
                    (211, 1),                 // plonk rows
                ];
                let interior = CLOCKS - 2 * lanes;
                let total_blocks: usize = runs.iter().map(|(n, s)| n.div_ceil(interior / s)).sum();
                let air_blocks = total_blocks + 3; // a few filler blocks past the band

                // Build the six PIL selector columns exactly as the air declares them.
                let mut at = 0usize;
                let mut sels: Vec<Vec<bool>> = Vec::new();
                for (n, step) in runs {
                    let nb = n.div_ceil(interior / step);
                    sels.push(pil_selector(at, nb, step, lanes, air_blocks));
                    at += nb;
                }

                // Run the real allocator over the same runs.
                let mut alloc = RowAlloc::new(lanes);
                let mut placed: Vec<Vec<usize>> = Vec::new();
                for (n, step) in runs {
                    alloc.open(n, step);
                    placed.push((0..n).map(|_| alloc.take()).collect());
                }
                assert_eq!(alloc.blocks_used(), at, "layout={} LANES={lanes}: block runs disagree", layout.template);

                // 1. Every placed row has its own selector on.
                for (c, rows) in placed.iter().enumerate() {
                    for &r in rows {
                        assert!(
                            r < sels[c].len(),
                            "layout={} LANES={lanes} circuit {c}: row {r} past the air",
                            layout.template
                        );
                        assert!(
                            sels[c][r],
                            "layout={} LANES={lanes} circuit {c}: row {r} placed where the \
                                selector is OFF -- that gate would be unconstrained",
                            layout.template
                        );
                    }
                }
                // 2. No row carries two selectors, and no selector touches a boundary row.
                for r in 0..air_blocks * CLOCKS {
                    let on: Vec<usize> = (0..6).filter(|&c| sels[c][r]).collect();
                    assert!(
                        on.len() <= 1,
                        "layout={} LANES={lanes}: row {r} carries selectors {on:?}",
                        layout.template
                    );
                    if let Some(&c) = on.first() {
                        let clk = r % CLOCKS;
                        assert!(
                            matches!(owner(clk, lanes, true), Owner::Band),
                            "layout={} LANES={lanes}: circuit {c} selector is on at clock {clk}, a \
                                 BLAKE3 boundary row",
                            layout.template
                        );
                    }
                }
                // 3. A two-row gate's second row is inside the same block and carries no selector.
                for (c, rows) in placed.iter().enumerate() {
                    let step = runs[c].1;
                    if step == 2 {
                        for &r in rows {
                            assert_eq!(
                                r / CLOCKS,
                                (r + 1) / CLOCKS,
                                "layout={} LANES={lanes}: pair crosses a block",
                                layout.template
                            );
                            assert!(
                                !sels[c][r + 1],
                                "layout={} LANES={lanes}: pair's second row is also a gate row",
                                layout.template
                            );
                            assert!(
                                matches!(owner((r + 1) % CLOCKS, lanes, true), Owner::Band),
                                "layout={} LANES={lanes}: pair's second row is a boundary row",
                                layout.template
                            );
                        }
                    }
                }
                // 4. Selector-on rows a gate does not fill (each circuit's last-block tail) are
                //    safe: an all-zero row satisfies every band circuit, plonk's q being 0 there.
                for (c, rows) in placed.iter().enumerate() {
                    let on: usize = sels[c].iter().filter(|&&b| b).count();
                    assert!(
                        on >= rows.len(),
                        "layout={} LANES={lanes} circuit {c}: fewer selector rows than gates",
                        layout.template
                    );
                }
            }
        }
    }

    /// The band's a[] cells a gate claims must fit the band, at both widths.
    #[test]
    fn no_gate_reaches_past_the_band_it_is_placed_in() {
        for layout in [&AGGREGATOR_LAYOUT, &COMPRESSOR_LAYOUT] {
            let b = layout.band;
            assert!(layout.cmul_per_row * 9 <= b, "cmul overruns {}", layout.template);
            assert!(layout.plonk_gates_per_row * 3 <= b, "plonk overruns {}", layout.template);
            assert!(layout.selval_per_row * 13 <= b, "selval overruns {}", layout.template);
            assert!(17 <= b, "treeselector overruns {}", layout.template);
            assert!(24usize.div_ceil(layout.fft4_rows) <= b, "fft4 overruns {}", layout.template);
            assert!(EVPOL4_SIGNALS.div_ceil(layout.evpol4_rows) <= b, "evpol4 overruns {}", layout.template);
            // Blake3Compress input row is exactly 18 cells; the band must hold them.
            assert!(compress_signal::IN_CELLS <= b, "compress input overruns {}", layout.template);
            assert!(compress_signal::OUT_CELLS <= b, "compress output overruns {}", layout.template);
        }
    }
}
