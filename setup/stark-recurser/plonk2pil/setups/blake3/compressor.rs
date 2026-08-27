//! Compressor setup for the blake3 family.
//!
//! `recursive1` and `recursive2` must be the SAME air, so the recursion runs at one pinned geometry
//! (2^19 rows, LANES=4). An air whose `recursive1` would not fit gets a compressor first --
//! `hash_family::recursive_bits_threshold("blake3")` is the trigger. A compressor matches nothing,
//! so unlike the recursion pair its geometry is free, and this module sizes it per air.
//!
//! # Three floors
//!
//! 1. **Table.** `blake3Tables` needs all 2^17 `(a, b, rot)` triples, so `N >= 2^17` always. The
//!    period 2^17 divides every larger power of two, so any N above it tiles exactly.
//! 2. **Blocks.** `56 * blocks + 55` rows, `blocks = ceil(uses / lanes)` per kind; the 55 are the
//!    clock selectors' wrap window (`blake3_max_blocks`).
//! 3. **Band.** The six band circuits ride the block interiors, `56 - 2*lanes` rows each, so
//!    `band_blocks <= blocks`.
//!
//! # The search: N answers to the band, LANES to the hashing
//!
//! The knobs are not symmetric, which is what makes this a decomposition rather than a sweep.
//! `lanes` touches only the hashing: more of them means fewer blocks (floor 2 down), a narrower
//! interior (floor 3 up), and 59 more `stage1_cols`. `N` is the only knob the band's rows can move.
//!
//! So the decision reads off which floor is unsatisfied. Start N at the floors, then at that N: if
//! some `lanes <= 8` holds the hashing and leaves the band inside the interiors, take the FEWEST
//! such -- more only buys columns. If none does, the binding floor is N: double it and ask again.
//! Doubling rather than solving because floor 2 depends on `lanes` through a ceiling.
//!
//! Sizing from the band alone stops at an N where floor 2 is unsatisfiable at every lanes: measured,
//! a fibonacci compressor's band wants 2^17 while its hashing needs 4331 blocks against a capacity
//! of 2339. Hence a starting floor, not the answer.
//!
//! [`SizingPolicy::FewestLanes`] is the other end of the trade, kept because the two genuinely
//! disagree: 2^18/LANES=8 against 2^21/LANES=1 on that compressor. Fewer lanes means fewer stage1
//! columns, and those are openings the pinned `recursive1` verifies -- a cost the whole tree shares,
//! where rows are this air's alone. Not the default because a compressor exists to be small.

use super::aggregation::{aggregation_blake3, bucket_by_flags, plan_band_blocks, plan_plonk_rows};
use super::{blake3_max_blocks, stage1_cols, BLAKE3_CLOCKS};
use crate::plonk2pil::merge_copies::r1cs2plonk_merged;
use crate::plonk2pil::r1cs::to_plonk::{blake3_compress_gate_uses, get_custom_gates_info};
use crate::plonk2pil::r1cs::types::{PlonkOptions, R1csFile, SetupResult};
use crate::plonk2pil::utils::log2;
use proofman_common::hash_family::GateRole;

/// Complex multiplications per row: `a[0..9]` and `a[9..18]`. Mirrors `aggregation`.
const CMUL_PER_ROW: usize = 2;

/// Lanes the air accepts. Above 8 the boundary opening depth exceeds 7 -- see `aggregator.pil`.
const MAX_LANES: usize = 8;

/// The table's floor, in bits. `blake3Tables` errors below it.
const TABLE_N_BITS: usize = 17;

/// How large the compressor's own air may grow while chasing fewer lanes.
///
/// Not a correctness bound -- the compressor is proved once and matches nothing. It bounds what its
/// PROOF costs the pinned `recursive1` that verifies it: a taller air means deeper Merkle trees and
/// so more compressions per query there. 2^21 is four times the pinned recursion size, past which
/// the thing doing the shrinking is bigger than what it shrank.
pub const DEFAULT_MAX_N_BITS: usize = 21;

/// Which of two geometries to prefer when both fit.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SizingPolicy {
    /// Smallest N the band's own rows allow, then the fewest lanes that hold the hashing there;
    /// double N when no lanes does. The default -- see the module's search section.
    ///
    /// "Band" is all SIX circuits, not just plonk: cmul, evpol4, fft4, treeselector4,
    /// selectValueArity2 and plonk all bind `a[0..17]` and all ride the block interiors, so the row
    /// count N has to cover is their sum. See `CompressorDemand::band_rows_by_circuit`.
    #[default]
    BandFirst,
    /// Fewest lanes, letting N grow to the cap. Trades this air's rows for stage1 columns, which are
    /// the openings the pinned `recursive1` verifies.
    FewestLanes,
}

/// A geometry the compressor can be built at.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CompressorGeometry {
    pub n_bits: usize,
    pub lanes: usize,
    /// BLAKE3 blocks the hashing needs at this `lanes`.
    pub blocks: usize,
    /// Blocks the band's six circuits need the interiors of.
    pub band_blocks: usize,
}

impl CompressorGeometry {
    /// Committed stage1 columns, which is what the next stage opens.
    pub fn stage1_cols(&self) -> usize {
        stage1_cols(self.lanes)
    }

    /// Trace cells, as a stand-in for this air's own prover cost.
    pub fn cells(&self) -> usize {
        (1usize << self.n_bits) * self.stage1_cols()
    }
}

/// What the sizing search is given: the counts, not the r1cs.
///
/// Separate from [`plan_compressor_geometry`] so the policy is testable without an r1cs.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CompressorDemand {
    /// `Blake3Node` gate uses.
    pub node_uses: usize,
    /// `Blake3Compress` chunk uses, already bucketed by `flags`.
    ///
    /// Bucketed rather than summed because `FLAGS` is a fixed column the air fills per whole block,
    /// so uses that disagree on it cannot share one. Each bucket takes whole blocks.
    pub chunk_buckets: Vec<usize>,
    /// `Blake3Compress` parent uses, bucketed the same way.
    pub parent_buckets: Vec<usize>,
    /// Rows the six band circuits need, in `plan_band_blocks` order, with the two-row gates counted
    /// twice: `[cmul, 2*evpol4, 2*fft4, tree, selval, plonk]`.
    pub band_rows_by_circuit: [usize; 6],
}

impl CompressorDemand {
    /// Blocks the hashing needs at `lanes`.
    pub fn blocks(&self, lanes: usize) -> usize {
        let bucketed = |b: &[usize]| b.iter().map(|n| n.div_ceil(lanes)).sum::<usize>();
        self.node_uses.div_ceil(lanes) + bucketed(&self.chunk_buckets) + bucketed(&self.parent_buckets)
    }

    /// Rows the six band circuits need if they had to have their own -- an 18-wide band and nothing
    /// to hide in.
    ///
    /// The search's starting floor. A row count and not a block count, because at this point `lanes`
    /// -- and so the interior width -- is not chosen yet.
    pub fn band_rows(&self) -> usize {
        self.band_rows_by_circuit.iter().sum()
    }

    /// Blocks whose interiors the band needs at `lanes`.
    pub fn band_blocks(&self, lanes: usize) -> usize {
        let [cmul, ev_pol4, fft4, tree, sel_val, plonk] = self.band_rows_by_circuit;
        // ev_pol4 and fft4 arrive here already doubled; plan_band_blocks wants the gate count.
        plan_band_blocks(cmul, ev_pol4 / 2, fft4 / 2, tree, sel_val, plonk, lanes).blocks
    }

    /// Whether `lanes` fits an air of `n_bits`, satisfying floors 2 and 3.
    fn fits(&self, n_bits: usize, lanes: usize) -> bool {
        let blocks = self.blocks(lanes);
        blocks <= blake3_max_blocks(1usize << n_bits) && self.band_blocks(lanes) <= blocks
    }

    /// Smallest N at which `lanes` fits, or `None` past `max_n_bits`.
    fn smallest_n_bits_for(&self, lanes: usize, start: usize, max_n_bits: usize) -> Option<usize> {
        (start..=max_n_bits).find(|&n_bits| self.fits(n_bits, lanes))
    }

    /// The floor every candidate starts from: the band's own rows, the table, and any caller pin.
    fn start_n_bits(&self, min_n_bits: usize) -> usize {
        let rows = self.band_rows();
        let band_n_bits = if rows <= 1 { 1 } else { log2((rows - 1) as u32) as usize + 1 };
        band_n_bits.max(TABLE_N_BITS).max(min_n_bits)
    }
}

/// Every geometry that fits, one per lanes count, ascending by lanes.
///
/// Exposed so a caller can report the trade rather than only the choice.
pub fn compressor_candidates(
    demand: &CompressorDemand,
    min_n_bits: usize,
    max_n_bits: usize,
) -> Vec<CompressorGeometry> {
    let start = demand.start_n_bits(min_n_bits);
    (1..=MAX_LANES)
        .filter_map(|lanes| {
            let n_bits = demand.smallest_n_bits_for(lanes, start, max_n_bits)?;
            Some(CompressorGeometry {
                n_bits,
                lanes,
                blocks: demand.blocks(lanes),
                band_blocks: demand.band_blocks(lanes),
            })
        })
        .collect()
}

/// The geometry `policy` picks, or `None` when nothing fits within `max_n_bits`.
///
/// `min_n_bits` is a floor the caller can raise -- an air reusing another's starkSetup has to match
/// its row count. It never lowers the table's or the blocks' floors.
pub fn plan_compressor_geometry(
    demand: &CompressorDemand,
    min_n_bits: usize,
    max_n_bits: usize,
    policy: SizingPolicy,
) -> Option<CompressorGeometry> {
    let candidates = compressor_candidates(demand, min_n_bits, max_n_bits);
    match policy {
        // Smallest N, and among ties the fewest lanes. This IS the decomposition: every candidate
        // already holds the band, so the minimum over N is the first N where any lanes fits, and the
        // lanes tie-break is "the fewest that fits there".
        SizingPolicy::BandFirst => candidates.into_iter().min_by_key(|g| (g.n_bits, g.lanes)),
        // Ascending by lanes already, so the first is the fewest.
        SizingPolicy::FewestLanes => candidates.into_iter().next(),
    }
}

/// Reads the demand off an r1cs, mirroring what `aggregation_blake3` counts.
pub fn compressor_demand(r1cs: &R1csFile, options: &PlonkOptions) -> CompressorDemand {
    let (plonk_constraints, _, _) = r1cs2plonk_merged(r1cs, options.merge_copies);
    let cgi = get_custom_gates_info(r1cs);

    let compress_uses = blake3_compress_gate_uses(&r1cs.custom_gates_uses, &cgi.blake3_compress_parameters);
    let (parent_uses, chunk_uses): (Vec<_>, Vec<_>) = compress_uses.into_iter().partition(|(_, _, ip)| *ip == 1);
    let sizes = |uses: Vec<_>| bucket_by_flags(uses).iter().map(|b| b.len()).collect::<Vec<_>>();

    let cmul = cgi.n(GateRole::CMul).div_ceil(CMUL_PER_ROW);
    let ev_pol4 = cgi.n(GateRole::EvPol4);
    let fft4 = cgi.n(GateRole::Fft4);
    let tree = cgi.n(GateRole::TreeSelector);
    let sel_val = cgi.n(GateRole::SelectValArity2);
    // `plan_plonk_rows` with 0 blocks reports the rows the constraints need outright, before any
    // block interior is offered to them -- which is what the band's own floor means. `lanes` is
    // irrelevant at 0 blocks, so 1 stands in for the not-yet-chosen value.
    let plonk = plan_plonk_rows(&plonk_constraints, 0, 1, BLAKE3_CLOCKS).rows_needed;

    CompressorDemand {
        node_uses: cgi.n(GateRole::Blake3Node),
        chunk_buckets: sizes(chunk_uses),
        parent_buckets: sizes(parent_uses),
        // EvPol4 (21 signals) and FFT4 (24) do not fit an 18-wide band, so they take two rows.
        band_rows_by_circuit: [cmul, 2 * ev_pol4, 2 * fft4, tree, sel_val, plonk],
    }
}

/// Compressor setup: plan the geometry, then build the aggregator air at it.
///
/// The air itself is `blake3/aggregator.pil` unchanged. The compressor and the aggregator differ in
/// what their CIRCOM circuit proves -- verifying a basic proof versus aggregating two recursive ones
/// -- but a recursion air is only a carrier for plonk rows plus the custom gates, and the two draw on
/// the same gate set. What differs is the SIZING: the aggregator is pinned to the shared recursion
/// shape, the compressor picks its own.
pub fn compressor_blake3(r1cs: &R1csFile, options: &PlonkOptions) -> SetupResult {
    // An explicit --blake3-lanes outranks the search: the caller is stating the geometry.
    if options.blake3_lanes.is_some() {
        tracing::info!("Compressor: LANES pinned by the caller, skipping the geometry search");
        return aggregation_blake3(r1cs, options);
    }

    let demand = compressor_demand(r1cs, options);
    let min_n_bits = options.min_n_bits.unwrap_or(0);
    let policy = SizingPolicy::default();
    let candidates = compressor_candidates(&demand, min_n_bits, DEFAULT_MAX_N_BITS);

    // The whole trade, not just the pick, so the choice is auditable.
    tracing::info!(
        "Compressor demand: {} Blake3Node + {} chunk + {} parent uses, band floor {} rows",
        demand.node_uses,
        demand.chunk_buckets.iter().sum::<usize>(),
        demand.parent_buckets.iter().sum::<usize>(),
        demand.band_rows(),
    );
    for lanes in 1..=MAX_LANES {
        match candidates.iter().find(|g| g.lanes == lanes) {
            Some(g) => tracing::info!(
                "  LANES={lanes}: N = 2^{}, {} blocks of {} capacity, band {} -> stage1 {} cols, {:.1} M cells",
                g.n_bits,
                g.blocks,
                blake3_max_blocks(1usize << g.n_bits),
                g.band_blocks,
                g.stage1_cols(),
                g.cells() as f64 / 1e6,
            ),
            None => tracing::info!(
                "  LANES={lanes}: no N <= 2^{DEFAULT_MAX_N_BITS} fits ({} blocks, band {})",
                demand.blocks(lanes),
                demand.band_blocks(lanes),
            ),
        }
    }

    let geom = plan_compressor_geometry(&demand, min_n_bits, DEFAULT_MAX_N_BITS, policy).unwrap_or_else(|| {
        panic!(
            "no (N <= 2^{DEFAULT_MAX_N_BITS}, LANES <= {MAX_LANES}) holds {} Blake3Node + {} chunk + \
             {} parent uses and a band of {} rows",
            demand.node_uses,
            demand.chunk_buckets.iter().sum::<usize>(),
            demand.parent_buckets.iter().sum::<usize>(),
            demand.band_rows(),
        )
    });
    tracing::info!("Compressor geometry ({policy:?}): N = 2^{}, LANES = {}", geom.n_bits, geom.lanes);

    let opts = PlonkOptions { blake3_lanes: Some(geom.lanes), min_n_bits: Some(geom.n_bits), ..options.clone() };
    aggregation_blake3(r1cs, &opts)
}

#[cfg(test)]
mod tests {
    use super::super::aggregation::PLONK_GATES_PER_ROW;
    use super::*;

    const CAP: usize = DEFAULT_MAX_N_BITS;

    /// The measured fibonacci compressor: `Compressor.pil` reports nNodeBlocks 7054, nChunkBlocks
    /// 1607, nParentBlocks 0 at LANES=4, with the band's six counts as generated.
    fn fibonacci() -> CompressorDemand {
        CompressorDemand {
            node_uses: 7054 * 4,
            chunk_buckets: vec![1607 * 4],
            parent_buckets: vec![],
            band_rows_by_circuit: [3404, 2 * 3376, 2 * 6776, 6119, 31650, 22927],
        }
    }

    /// A band with no hashing at all still clears the table's floor, and takes one lane because
    /// nothing forces more.
    #[test]
    fn the_table_floor_holds_when_there_is_no_hashing() {
        let d = CompressorDemand {
            node_uses: 0,
            chunk_buckets: vec![],
            parent_buckets: vec![],
            band_rows_by_circuit: [0; 6],
        };
        let g = plan_compressor_geometry(&d, 0, CAP, SizingPolicy::FewestLanes).unwrap();
        assert_eq!(g.n_bits, TABLE_N_BITS, "N is never below the table's 2^17");
        assert_eq!(g.lanes, 1, "nothing forces a second lane");
    }

    /// The band alone says 2^17, and at 2^17 the hashing fits at NO lanes. This is the case a
    /// band-only sizing rule gets wrong, and the reason the search walks N.
    #[test]
    fn the_band_floor_alone_is_not_a_geometry() {
        let d = fibonacci();
        assert_eq!(d.band_rows(), 84_404);
        assert_eq!(d.start_n_bits(0), TABLE_N_BITS, "the band's own floor lands on the table's");
        let cap17 = blake3_max_blocks(1 << 17);
        for lanes in 1..=MAX_LANES {
            assert!(d.blocks(lanes) > cap17, "LANES={lanes} unexpectedly fits 2^17");
        }
    }

    /// The default is the decomposition: N no larger than the floors force, and at that N the fewest
    /// lanes that holds the hashing. Lanes past that would only buy columns.
    #[test]
    fn band_first_is_the_default_and_takes_the_smallest_air() {
        let d = fibonacci();
        let g = plan_compressor_geometry(&d, 0, CAP, SizingPolicy::default()).unwrap();
        let all = compressor_candidates(&d, 0, CAP);
        assert!(all.iter().all(|c| c.n_bits >= g.n_bits), "no candidate has a smaller N: {g:?}");
        // Fewest lanes AMONG those at the chosen N -- not fewest overall.
        let at_n = all.iter().filter(|c| c.n_bits == g.n_bits).map(|c| c.lanes).min().unwrap();
        assert_eq!(g.lanes, at_n, "the tie-break must take the fewest lanes at that N");
    }

    /// When no lanes fits, the binding floor is N and the search doubles it rather than adding lanes.
    #[test]
    fn no_lanes_fitting_means_n_grows() {
        let d = fibonacci();
        let start = d.start_n_bits(0);
        for lanes in 1..=MAX_LANES {
            assert!(!d.fits(start, lanes), "LANES={lanes} unexpectedly fits the starting N");
        }
        let g = plan_compressor_geometry(&d, 0, CAP, SizingPolicy::BandFirst).unwrap();
        assert!(g.n_bits > start, "N had to grow past the band's floor: {g:?}");
    }

    /// The two policies are a genuinely different answer, not a tie -- which is why the choice is
    /// explicit rather than implied by the search order.
    #[test]
    fn the_two_policies_disagree_on_the_measured_compressor() {
        let d = fibonacci();
        let few = plan_compressor_geometry(&d, 0, CAP, SizingPolicy::FewestLanes).unwrap();
        let plonk = plan_compressor_geometry(&d, 0, CAP, SizingPolicy::BandFirst).unwrap();
        assert_ne!(few.lanes, plonk.lanes, "the policies must not silently coincide");
        assert!(plonk.n_bits <= few.n_bits, "BandFirst cannot pick a taller air");
        assert!(few.stage1_cols() <= plonk.stage1_cols(), "FewestLanes cannot pick more columns");
    }

    /// The cap is what makes FewestLanes safe: squeeze it and the search degrades to more lanes
    /// instead of failing.
    #[test]
    fn a_tight_cap_degrades_to_more_lanes_rather_than_failing() {
        let d = fibonacci();
        let roomy = plan_compressor_geometry(&d, 0, CAP, SizingPolicy::FewestLanes).unwrap();
        let tight = plan_compressor_geometry(&d, 0, 18, SizingPolicy::FewestLanes).unwrap();
        assert!(tight.lanes > roomy.lanes, "a tighter cap must buy lanes, not fail");
        assert!(tight.n_bits <= 18);
    }

    /// `min_n_bits` raises the floor and never lowers the others.
    #[test]
    fn a_pinned_floor_outranks_the_search_but_not_the_other_floors() {
        let d = CompressorDemand {
            node_uses: 100,
            chunk_buckets: vec![],
            parent_buckets: vec![],
            band_rows_by_circuit: [0; 6],
        };
        assert_eq!(plan_compressor_geometry(&d, 20, CAP, SizingPolicy::FewestLanes).unwrap().n_bits, 20);
        assert_eq!(
            plan_compressor_geometry(&d, 10, CAP, SizingPolicy::FewestLanes).unwrap().n_bits,
            TABLE_N_BITS,
            "10 is below the table's floor"
        );
    }

    /// Flags bucketing costs blocks: two buckets of one use each take two blocks, not one, because
    /// `FLAGS` is a fixed column the air fills per whole block.
    #[test]
    fn flags_buckets_cannot_share_a_block() {
        let one = CompressorDemand {
            node_uses: 0,
            chunk_buckets: vec![2],
            parent_buckets: vec![],
            band_rows_by_circuit: [0; 6],
        };
        let two = CompressorDemand { chunk_buckets: vec![1, 1], ..one.clone() };
        assert_eq!(one.blocks(4), 1);
        assert_eq!(two.blocks(4), 2, "one block per bucket, however few uses it holds");
    }

    /// A plonk-dominated demand is never accepted with a band that overflows the interiors it was
    /// given. Refusing is a valid outcome; an overflowing geometry is not.
    #[test]
    fn a_plonk_dominated_demand_never_returns_an_overflowing_band() {
        let d = CompressorDemand {
            node_uses: 40,
            chunk_buckets: vec![],
            parent_buckets: vec![],
            band_rows_by_circuit: [0, 0, 0, 0, 0, 500_000],
        };
        if let Some(g) = plan_compressor_geometry(&d, 0, CAP, SizingPolicy::FewestLanes) {
            assert!(g.band_blocks <= g.blocks, "returned a geometry whose band does not fit: {g:?}");
        }
    }

    /// Every geometry either policy returns satisfies all three floors and is minimal for its
    /// policy, over a sweep of demands.
    #[test]
    fn every_returned_geometry_satisfies_all_three_floors_and_is_minimal() {
        for node in [0usize, 100, 5_000, 28_216, 120_000] {
            for chunk in [0usize, 1, 6_428, 40_000] {
                for band in [0usize, 84_404, 300_000] {
                    let d = CompressorDemand {
                        node_uses: node,
                        chunk_buckets: if chunk == 0 { vec![] } else { vec![chunk] },
                        parent_buckets: vec![],
                        band_rows_by_circuit: [0, 0, 0, 0, 0, band],
                    };
                    for policy in [SizingPolicy::FewestLanes, SizingPolicy::BandFirst] {
                        let Some(g) = plan_compressor_geometry(&d, 0, CAP, policy) else { continue };
                        assert!(g.n_bits >= TABLE_N_BITS, "table floor: {g:?}");
                        assert!(g.n_bits <= CAP, "cap: {g:?}");
                        assert!(g.blocks <= blake3_max_blocks(1usize << g.n_bits), "block floor: {g:?}");
                        assert!(g.band_blocks <= g.blocks, "band floor: {g:?}");
                        assert!((1..=MAX_LANES).contains(&g.lanes), "lanes out of range: {g:?}");
                        match policy {
                            // No fewer lanes fits anywhere within the cap.
                            SizingPolicy::FewestLanes => {
                                for lanes in 1..g.lanes {
                                    assert!(
                                        d.smallest_n_bits_for(lanes, d.start_n_bits(0), CAP).is_none(),
                                        "LANES={lanes} would have fitted: {g:?}"
                                    );
                                }
                            }
                            // No smaller N fits at any lanes.
                            SizingPolicy::BandFirst => {
                                if g.n_bits > d.start_n_bits(0) {
                                    for lanes in 1..=MAX_LANES {
                                        assert!(
                                            !d.fits(g.n_bits - 1, lanes),
                                            "2^{} LANES={lanes} would have fitted: {g:?}",
                                            g.n_bits - 1
                                        );
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    /// The band's row count is what the six circuits need with nothing to hide in, and the two-row
    /// gates are counted twice. Pins the convention `compressor_demand` fills the array with.
    #[test]
    fn band_rows_counts_the_two_row_gates_twice() {
        let d = CompressorDemand {
            node_uses: 0,
            chunk_buckets: vec![],
            parent_buckets: vec![],
            band_rows_by_circuit: [10, 2 * 5, 2 * 7, 3, 4, 100],
        };
        assert_eq!(d.band_rows(), 10 + 10 + 14 + 3 + 4 + 100);
    }

    /// Plonk rows come from gate slots, not constraints: six gates a row.
    #[test]
    fn the_band_floor_uses_six_plonk_gates_a_row() {
        assert_eq!(PLONK_GATES_PER_ROW, 6);
    }

    /// The band floor is ALL SIX circuits of the connection band, not plonk alone. Every one of them
    /// binds `a[0..17]` and rides the block interiors, so each contributes rows N has to cover.
    #[test]
    fn the_band_floor_counts_every_circuit_not_just_plonk() {
        let plonk_only = CompressorDemand {
            node_uses: 0,
            chunk_buckets: vec![],
            parent_buckets: vec![],
            band_rows_by_circuit: [0, 0, 0, 0, 0, 22_927],
        };
        let all_six = fibonacci();
        assert_eq!(plonk_only.band_rows(), 22_927);
        assert_eq!(all_six.band_rows(), 84_404, "the other five circuits are 61k of the 84k");
        assert!(all_six.band_rows() > 3 * plonk_only.band_rows(), "plonk alone would understate N");
        // And each of the six moves the floor on its own.
        for i in 0..6 {
            let mut one = [0usize; 6];
            one[i] = 1000;
            let d = CompressorDemand { band_rows_by_circuit: one, ..plonk_only.clone() };
            assert_eq!(d.band_rows(), 1000, "circuit {i} must count toward the band floor");
        }
    }
}
