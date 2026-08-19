//! Predictive linear-hash cell estimation for the recursive verifier.
//!
//! This is an *analytical* model: given a handful of high-level parameters
//! (dimension exponent, blowup, stage1 column count, cells-per-Blake3) it
//! estimates how many witness *cells* the recursive verifier spends on
//! **linear-hash reconstruction** — without compiling any circuit.
//!
//! It is complementary to `plonk2pil::estimate`, which counts cells off an
//! already-compiled R1CS. Here we predict from parameters, before compilation.
//!
//! ## What we count
//!
//! At query time the recursive verifier reconstructs a Blake3 *linear hash* for
//! each committed oracle. Blake3 is used as a sponge with **rate 4** Goldilocks
//! field elements (capacity 4), so reconstructing the hash of a leaf holding
//! `c` field elements costs `ceil(c / 4)` permutations.
//!
//! Two kinds of oracle contribute:
//!
//! * **Trace / Q stages** — fixed (28 cols), stageQ (18), stage2 (0) and stage1
//!   (free). Column counts are base-field and used directly: `ceil(cols / 4)`.
//! * **FRI folding layers** — every folding step *except the last* opens a leaf
//!   of `2^drop` sibling values, each in the **degree-3 extension** field, so
//!   `ceil(2^drop * 3 / 4)` permutations per step. The final layer is sent in
//!   the clear (final degree test) and is not opened.
//!
//! Everything is multiplied by `n_queries` (from [`super::security`]) and by
//! `n_proofs` (recursion verifies two proofs at once; a basic AIR verifier
//! opens one), then by the free `cells_per_blake3` to get total cells.

use super::security::goldilocks_safe_extension_field_size;
use super::security::pcs::{Batching, Fri, FriConfig};
use super::security::regimes::DecodingRegime;

// ---------------------------------------------------------------------------
// Fixed constants (same for any STARK we model here)
// ---------------------------------------------------------------------------

const TARGET_SECURITY_BITS: u64 = 128;
/// Proof-of-work grinding bits (both ZisK Main and recursion pin this to 20),
/// so the query phase only needs `TARGET_SECURITY_BITS - GRINDING_BITS` bits.
const GRINDING_BITS: u64 = 20;

/// Recursion / FRI oracles live in the cubic extension of Goldilocks.
const EXTENSION_DEGREE: u64 = 3;

/// Instance geometry per hash family, from `examples/hashes` (README throughput
/// table + the generated trace rows). One permutation spans
/// `cells_per_perm / instance_cols` rows of a single instance.
///
/// Blake3: 56 clocks x 108 columns (`CLOCKS = NUM_G_PER_ROUND * NUM_ROUNDS` in
/// `blake3.pil`). Blake2b: 96 clocks x 190 columns — full 12-round blake2b.
/// NB: `blake2b.pil` currently pins `NUM_ROUNDS = 8` (64 clocks, 12160 cells)
/// citing "Too Much Crypto"; this model tracks the 12-round cost instead.
///
/// The stage widths are per instance and scale with the packed count; the fixed
/// columns encode the shared round schedule and do NOT scale. Both families'
/// stage widths fall exactly six short of their instance width (54+48 vs 108,
/// 100+84 vs 190) — the fit check is driven by rows/permutation, so it does not
/// depend on that split.
pub const BLAKE3_CELLS_PER_PERM: u64 = 6048;
pub const BLAKE3_INSTANCE_COLS: u64 = 108;
pub const BLAKE3_FIXED_COLS: u64 = 8;
/// Compression-block width in Goldilocks elements: blake3's 64-byte block holds
/// 8, blake2b's 128-byte block holds 16. A Merkle node is a fixed-width
/// compression, NOT a sponge absorption, so it uses the full block and pays no
/// capacity — which is why blake2b folds an arity-4 node (4 x 4 = 16 elements =
/// 128 bytes) in a single permutation.
pub const BLAKE3_COMPRESSION_ELEMS: u64 = 8;
pub const BLAKE3_STAGE1_COLS: u64 = 54;
pub const BLAKE3_STAGE2_COLS: u64 = 48;

pub const BLAKE2_CELLS_PER_PERM: u64 = 18240;
pub const BLAKE2_INSTANCE_COLS: u64 = 190;
pub const BLAKE2_FIXED_COLS: u64 = 7;
pub const BLAKE2_COMPRESSION_ELEMS: u64 = 16;
pub const BLAKE2_STAGE1_COLS: u64 = 100;
pub const BLAKE2_STAGE2_COLS: u64 = 84;

/// Largest extended domain the model supports, in bits. `pcs::Fri` indexes the
/// evaluation domain with a `u32`, so `n + blowup_bits` must stay below 32.
/// A 2^32-row domain is far beyond anything provable in practice; the bound only
/// clips the extreme corner of the parameter sweep.
pub const MAX_EXT_BITS: u32 = 31;

/// Merkle digest width in Goldilocks field elements (`HASH_SIZE` in pil2-stark's
/// `merkleTreeGL`). An internal node absorbs `arity` of these.
const DIGEST_ELEMS: u64 = 4;

/// The hash used for linear hashing. Each variant fixes the sponge rate (field
/// elements absorbed per permutation) and carries a free per-permutation cell
/// cost — both Blake2 and Blake3 are under investigation.
#[derive(Debug, Clone, Copy)]
pub enum HashFamily {
    /// Blake3-style sponge: rate 4, free cell cost.
    Blake3 { cells_per_perm: u64 },
    /// Blake2-style sponge: rate 12, free cell cost.
    Blake2 { cells_per_perm: u64 },
}

impl HashFamily {
    /// Field elements absorbed per permutation (the sponge rate).
    pub fn sponge_rate(&self) -> u64 {
        match self {
            HashFamily::Blake3 { .. } => 4,
            HashFamily::Blake2 { .. } => 12,
        }
    }

    /// Witness cells one permutation occupies in the verifier circuit.
    pub fn cells_per_perm(&self) -> u64 {
        match self {
            HashFamily::Blake3 { cells_per_perm } | HashFamily::Blake2 { cells_per_perm } => *cells_per_perm,
        }
    }

    /// Merkle arity implied by the compression block: as many `DIGEST_ELEMS`
    /// children as fit in one block. Blake3 (8 elements) takes two children — a
    /// binary tree; blake2b (16) takes four — a quaternary tree.
    pub fn tree_arity(&self) -> u64 {
        (self.compression_elems() / DIGEST_ELEMS).max(2)
    }

    /// Elements one compression block holds, used for Merkle nodes. Wider than
    /// `sponge_rate` because fixed-width compression reserves no capacity.
    pub fn compression_elems(&self) -> u64 {
        match self {
            HashFamily::Blake3 { .. } => BLAKE3_COMPRESSION_ELEMS,
            HashFamily::Blake2 { .. } => BLAKE2_COMPRESSION_ELEMS,
        }
    }

    /// Full committed width of one instance (all stages).
    pub fn instance_cols(&self) -> u64 {
        match self {
            HashFamily::Blake3 { .. } => BLAKE3_INSTANCE_COLS,
            HashFamily::Blake2 { .. } => BLAKE2_INSTANCE_COLS,
        }
    }

    /// Fixed (constant) columns. Shared across packed instances, so not scaled.
    pub fn fixed_cols(&self) -> u64 {
        match self {
            HashFamily::Blake3 { .. } => BLAKE3_FIXED_COLS,
            HashFamily::Blake2 { .. } => BLAKE2_FIXED_COLS,
        }
    }

    /// Stage1 witness columns for ONE instance.
    pub fn stage1_cols(&self) -> u64 {
        match self {
            HashFamily::Blake3 { .. } => BLAKE3_STAGE1_COLS,
            HashFamily::Blake2 { .. } => BLAKE2_STAGE1_COLS,
        }
    }

    /// Stage2 columns for ONE instance.
    pub fn stage2_cols(&self) -> u64 {
        match self {
            HashFamily::Blake3 { .. } => BLAKE3_STAGE2_COLS,
            HashFamily::Blake2 { .. } => BLAKE2_STAGE2_COLS,
        }
    }
}

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/// Free inputs to the model. One parameter set fully describes the STARK being
/// verified, so the same model serves both recursion and a basic AIR (e.g. ZisK
/// Main) — they differ only in these values.
#[derive(Debug, Clone, Copy)]
pub struct CellModelParams {
    /// Dimension exponent: the trace has `2^n` rows.
    pub n: u32,
    /// Blowup exponent: FRI rate is `1 / 2^blowup_bits`.
    pub blowup_bits: u32,
    /// FRI folding step size, in bits. Each step folds `2^fold_bits` siblings
    /// (e.g. 4 -> `[4,4,4,3]`, 3 -> `[3,3,3,3,3]`).
    pub fold_bits: u32,
    /// Column counts for the four opened oracles: fixed, stageQ, stage2, stage1.
    pub fixed_cols: u64,
    pub stageq_cols: u64,
    pub stage2_cols: u64,
    pub stage1_cols: u64,
    /// Opening points and evaluated functions — feed the FRI security calc that
    /// determines `n_queries`.
    pub n_opening_points: u64,
    pub n_functions: u64,
    /// How many proofs this verifier opens (recursion verifies 2; a basic AIR
    /// verifier opens 1).
    pub n_proofs: u64,
    /// The hash used for linear hashing (sets sponge rate + cells/permutation).
    pub hash: HashFamily,
    /// Proof-of-work grinding bits, fed to the FRI security calculator.
    /// Defaults to `GRINDING_BITS` (20).
    pub grinding_bits: u64,
    /// Blake instances packed side by side, when the verifier is blake-packed.
    /// `0` means "not blake-packed" (e.g. ZisK Main), which keeps the older
    /// stage1 cell-packing fit check.
    pub blakes: u64,
}

impl CellModelParams {
    /// The recursion-defaults preset: fixed=28, stageQ=18, 8 opening points,
    /// 200 functions, verifies 2 proofs. `stage2_cols` defaults to `stage1/3`
    /// (use [`recursion_with_stage2`](Self::recursion_with_stage2) to override).
    pub fn recursion(n: u32, blowup_bits: u32, fold_bits: u32, stage1_cols: u64, hash: HashFamily) -> Self {
        Self::recursion_with_stage2(n, blowup_bits, fold_bits, stage1_cols, stage1_cols / 3, hash)
    }

    /// Recursion preset with an explicit `stage2_cols`.
    pub fn recursion_with_stage2(
        n: u32,
        blowup_bits: u32,
        fold_bits: u32,
        stage1_cols: u64,
        stage2_cols: u64,
        hash: HashFamily,
    ) -> Self {
        CellModelParams {
            n,
            blowup_bits,
            fold_bits,
            fixed_cols: 28,
            stageq_cols: 18,
            stage2_cols,
            stage1_cols,
            n_opening_points: 8,
            n_functions: 200,
            n_proofs: 2,
            hash,
            grinding_bits: GRINDING_BITS,
            blakes: 0,
        }
    }

    /// Recursion preset whose stage widths come from packing `blakes` instances
    /// of `hash` side by side. Stage widths scale with the count; the fixed
    /// columns are the shared round schedule and do not.
    pub fn recursion_blakes(n: u32, blowup_bits: u32, fold_bits: u32, blakes: u64, hash: HashFamily) -> Self {
        let mut p = Self::recursion_with_stage2(
            n,
            blowup_bits,
            fold_bits,
            hash.stage1_cols() * blakes,
            hash.stage2_cols() * blakes,
            hash,
        );
        p.fixed_cols = hash.fixed_cols();
        p.blakes = blakes;
        p
    }

    /// The hardcoded ZisK Main basic-AIR verifier:
    /// nBits=22, blowup=1, fold=3, single proof, stages fixed=3 / stageQ=6 /
    /// stage2=24 / stage1=38, 3 opening points, 61 evaluated functions.
    /// `hash` is shared with the recursion run it is compared against.
    pub fn zisk_main(fold_bits: u32, hash: HashFamily) -> Self {
        CellModelParams {
            n: 22,
            blowup_bits: 1,
            fold_bits,
            fixed_cols: 3,
            stageq_cols: 6,
            stage2_cols: 24,
            stage1_cols: 38,
            n_opening_points: 3,
            n_functions: 61,
            n_proofs: 1,
            hash,
            grinding_bits: GRINDING_BITS,
            blakes: 0,
        }
    }
}

/// Per-stage hash breakdown.
#[derive(Debug, Clone)]
pub struct StageHashes {
    pub name: &'static str,
    pub cols: u64,
    /// `ceil(cols / SPONGE_RATE)`.
    pub hashes: u64,
}

/// Per-FRI-step hash breakdown (one entry per *opened* folding layer).
#[derive(Debug, Clone)]
pub struct FriStepHashes {
    /// The bit-drop for this folding step.
    pub bit_drop: u64,
    /// Field elements opened: `2^bit_drop * EXTENSION_DEGREE`.
    pub elements: u64,
    /// `ceil(elements / SPONGE_RATE)`.
    pub hashes: u64,
}

/// Full linear-hash cell estimate.
#[derive(Debug, Clone)]
pub struct CellEstimate {
    /// Number of queries opening stage-0 commitments.
    pub n_queries: u64,
    /// Folding factors as bit-drops, e.g. `[4, 4, 4, 3]`.
    pub folding_factors: Vec<u64>,
    pub trace_stages: Vec<StageHashes>,
    pub fri_steps: Vec<FriStepHashes>,
    /// Trace/Q stage hashes for one proof, one query.
    pub trace_hashes_per_query: u64,
    /// Folding hashes for one proof, one query.
    pub fri_hashes_per_query: u64,
    /// `trace_hashes_per_query + fri_hashes_per_query`. Leaf (linear-hash) work
    /// only — see `merkle_hashes_per_query` for the authentication paths.
    pub hashes_per_query: u64,
    /// Merkle authentication-path permutations for one proof, one query: one path
    /// per committed stage tree plus one per opened folding layer.
    pub merkle_hashes_per_query: u64,
    /// Number of proofs opened (echoed from params, for the report).
    pub n_proofs: u64,
    /// Leaf-hash permutations: `hashes_per_query * n_queries * n_proofs`.
    pub total_linear_hashes: u64,
    /// Authentication-path permutations across every query and proof.
    pub total_merkle_hashes: u64,
    /// Every permutation the verifier performs: `total_linear_hashes +
    /// total_merkle_hashes`.
    pub total_hashes: u64,
    /// `total_hashes * cells_per_perm`.
    pub total_cells: u64,
    /// Stage1 columns the verifier would actually need to pack `total_cells`
    /// into `2^n` rows: `ceil(total_cells / 2^n)`.
    pub needed_stage1_cols: u64,
    /// The `stage1_cols` assumed in the input (for the fit comparison).
    pub assumed_stage1_cols: u64,
    /// Blake instances the verifier needs packed side by side. A permutation
    /// occupies `rows_per_perm` rows of ONE instance, so `k` instances clear
    /// `k * 2^n / rows_per_perm` permutations.
    pub blakes_needed: u64,
    /// Blake instances assumed in the input (0 when not blake-packed).
    pub assumed_blakes: u64,
    /// Rows one permutation occupies in a blake instance (56 at 6048 cells).
    pub rows_per_perm: u64,
    /// Whether the configuration fits. Blake-packed presets ask
    /// `blakes_needed <= assumed_blakes`; otherwise `needed_stage1_cols <=
    /// assumed_stage1_cols`.
    pub fits: bool,
    /// Prover GPU memory estimate, in field elements (ported from
    /// `pil::info::get_prover_memory`).
    pub prover_memory_field_elems: u64,
    /// Prover GPU memory estimate, in GB (`field_elems * 8 / 1024^3`).
    pub prover_memory_gb: f64,
    /// Grinding bits this estimate was computed with (echoed from the params, so
    /// downstream fit-checks can re-cost a sibling AIR under the same grinding).
    pub grinding_bits: u64,
}

// ---------------------------------------------------------------------------
// Core helpers
// ---------------------------------------------------------------------------

/// Permutations to absorb `n_elements` field elements at the given sponge rate.
fn hashes_for_elements(n_elements: u64, sponge_rate: u64) -> u64 {
    n_elements.div_ceil(sponge_rate)
}

/// Rows one permutation occupies in an instance: its cell cost spread across the
/// instance's full committed width. Blake3: 6048 / 108 = 56. Blake2b: 12160 / 190 = 64.
fn rows_per_perm(cells_per_perm: u64, instance_cols: u64) -> u64 {
    cells_per_perm.div_ceil(instance_cols).max(1)
}

/// Levels in the authentication path of a tree over `2^height_bits` leaves.
/// Each level consumes `log2(arity)` bits of height; a partial top level still
/// costs a whole compression.
///
/// Uses the same `log2(n_leafs) / log2(arity)` form as
/// `query_params::calculate_mtp_hashes`, so a non-power-of-two arity (pil2-stark
/// supports 3 via `PoseidonGoldilocks<12>`) is handled rather than dividing by
/// zero. Exact for the powers of two we actually use.
fn merkle_path_depth(height_bits: u64, arity: u64) -> u64 {
    debug_assert!(arity >= 2, "Merkle arity must be at least 2");
    ((height_bits as f64) / (arity as f64).log2()).ceil() as u64
}

/// Permutations to recompute ONE internal node. The verifier re-derives the
/// parent from all `arity` child digests — `arity * DIGEST_ELEMS` elements — fed
/// through the compression function's full block. At arity 4 that is 16 elements
/// (128 bytes): one blake2b block, two blake3 blocks.
fn merkle_node_perms(arity: u64, compression_elems: u64) -> u64 {
    hashes_for_elements(arity * DIGEST_ELEMS, compression_elems)
}

/// Permutations to verify ONE authentication path from leaf to root.
///
/// This assumes the full path is walked in-circuit. pil2-stark's
/// `verifyMerkleRoot` honours `lastLevelVerification`, which would shave the top
/// level or two; the model does not, so it is a slight over-estimate.
fn merkle_path_perms(height_bits: u64, arity: u64, compression_elems: u64) -> u64 {
    merkle_path_depth(height_bits, arity) * merkle_node_perms(arity, compression_elems)
}

/// Derive FRI folding factors, as bit-drops of the polynomial DIMENSION.
///
/// FRI folds the polynomial, not the evaluation domain: each step halves the
/// degree `fold_bits` times while the rate stays fixed, so the domain shrinks in
/// lockstep and the drops are the same either way. What differs is where the
/// schedule STOPS. Folding must stop while the polynomial still has coefficients,
/// so the schedule runs `n -> 6` and the final layer holds `2^6` coefficients
/// over a `2^(6 + blowup_bits)` domain.
///
/// This is deliberately NOT what `generate_stark_struct` does: it folds the
/// extended domain down to `final_degree` regardless of blowup, which puts the
/// final dimension at `final_degree - blowup_bits` — degenerate at blowup 2^5-2^6
/// and negative beyond. `pcs::Fri` rejects such a schedule outright. Mirroring
/// that here would invent FRI layers the prover cannot produce, and every
/// phantom layer costs a leaf hash and a Merkle path on every query.
///
/// The returned list is the *drops only* (summing to `n - 6`); the leftover
/// degree-6 final layer is implicit and not listed.
///
/// Per the opening convention (mirroring `security::query_params::calculate_query_num_hashes`,
/// which iterates `folding_factors[..len-1]`), the **last entry is the final
/// fold and is not opened** — only the earlier folds produce query openings.
///
/// Examples (n = 15): `fold_bits=4` -> `[4,5]`; `fold_bits=3` -> `[3,3,3]`.
fn derive_folding_factors(n: u32, _blowup_bits: u32, fold_bits: u32) -> Vec<u64> {
    let step = fold_bits as i64;
    let mut remaining = n as i64;
    let mut factors = Vec::new();
    while remaining > 6 {
        // Full `fold_bits` drop while it leaves more than 6 bits; otherwise size
        // the final drop to land exactly on dimension 6.
        let drop = if remaining - step > 6 { step } else { remaining - 6 };
        factors.push(drop as u64);
        remaining -= drop;
    }
    factors
}

/// Number of Merkle-tree nodes (field elements) for `height` leaves at the given
/// arity. Mirrors `pil::info::get_num_nodes_mt` exactly (incl. the `* 4`).
fn get_num_nodes_mt(height: u64, arity: u64) -> u64 {
    let mut num_nodes = height;
    let mut nodes_level = height;
    while nodes_level > 1 {
        let extra_zeros = (arity - (nodes_level % arity)) % arity;
        num_nodes += extra_zeros;
        let next_n = nodes_level.div_ceil(arity);
        num_nodes += next_n;
        nodes_level = next_n;
    }
    num_nodes * 4
}

/// Estimate prover GPU memory (in field elements) for the STARK described by `p`.
///
/// Ported from `pil::info::get_prover_memory`. The stage column counts map onto
/// `map_sections_n`: cm1 = stage1, cm2 = stage2, cm{n_stages+1} = Q (stageQ);
/// `fixed_cols` is `n_constants`. Custom commits are not modeled (none in the
/// recursion / basic-AIR verifiers here). `boundaries` is taken as
/// `n_opening_points` (one boundary buffer per opening point).
fn prover_memory_field_elems(p: &CellModelParams, folding_factors: &[u64]) -> u64 {
    let n = 1u64 << p.n;
    let n_extended = 1u64 << (p.n + p.blowup_bits);
    let arity = p.hash.tree_arity();
    let num_nodes = get_num_nodes_mt(n_extended, arity);

    let mut mem: u64 = 0;

    // Constants (no custom commits in these verifiers).
    let n_constants = p.fixed_cols;
    mem += 2 + n_extended * n_constants + num_nodes;
    if (n_constants * n * 8) / (1024 * 1024) < 512 {
        mem += n * n_constants;
    }

    // Trace sections cm1..cm{n_stages+1}: stage1, stage2, then Q (= stageQ).
    let sections = [p.stage1_cols, p.stage2_cols, p.stageq_cols];
    let n_stages = 2u64; // stage1, stage2
    let mut offset_traces: u64 = 0;
    for (i, &section_n) in sections.iter().enumerate() {
        if i as u64 == 1 {
            offset_traces = mem;
        }
        mem += section_n * n_extended + num_nodes;
    }
    // offset_traces accumulates the committed (non-extended) trace for stages 1..n_stages.
    for &section_n in sections.iter().take(n_stages as usize) {
        offset_traces += section_n * n;
    }
    if offset_traces > mem {
        mem = offset_traces;
    }

    // Evals + boundary buffers over the extended domain.
    let boundaries = p.n_opening_points;
    mem += (EXTENSION_DEGREE + EXTENSION_DEGREE + boundaries) * n_extended;

    // FRI step folding buffers. `folding_factors` are the per-step bit-drops over
    // the extended domain; the implicit final layer lands on degree 6.
    let mut step_bits: Vec<u64> = Vec::with_capacity(folding_factors.len() + 1);
    step_bits.push((p.n + p.blowup_bits) as u64);
    let mut cur = (p.n + p.blowup_bits) as u64;
    for &drop in folding_factors {
        cur -= drop;
        step_bits.push(cur);
    }
    for w in step_bits.windows(2) {
        let (sa, sb) = (w[0], w[1]);
        let height = 1u64 << sb;
        let width = ((1u64 << sa) / height) * EXTENSION_DEGREE;
        mem += height * width + get_num_nodes_mt(height, arity);
    }

    mem
}

// ---------------------------------------------------------------------------
// Model
// ---------------------------------------------------------------------------

/// Estimate the verifier's linear-hash cell footprint for the STARK described
/// by `p` (recursion or a basic AIR alike).
pub fn estimate_linear_hash_cells(p: &CellModelParams) -> CellEstimate {
    let folding_factors = derive_folding_factors(p.n, p.blowup_bits, p.fold_bits);
    let rate = p.hash.sponge_rate();

    // Trace / Q stages: column count used directly, no extension factor.
    let trace_stages: Vec<StageHashes> =
        [("fixed", p.fixed_cols), ("stageQ", p.stageq_cols), ("stage2", p.stage2_cols), ("stage1", p.stage1_cols)]
            .into_iter()
            .map(|(name, cols)| StageHashes { name, cols, hashes: hashes_for_elements(cols, rate) })
            .collect();
    let trace_hashes_per_query: u64 = trace_stages.iter().map(|s| s.hashes).sum();

    // Every committed stage is its own Merkle tree over the extended domain, and
    // each query walks one authentication path into each. Stages with no columns
    // are never committed, so they have no tree.
    let arity = p.hash.tree_arity();
    let compression = p.hash.compression_elems();
    let ext_bits = (p.n + p.blowup_bits) as u64;
    let n_trace_trees = trace_stages.iter().filter(|s| s.cols > 0).count() as u64;
    let trace_merkle_per_query = n_trace_trees * merkle_path_perms(ext_bits, arity, compression);

    // Query count comes from the same calculator that sizes real proving keys
    // (`security::pcs::Fri`), so the model reflects what the pipeline actually
    // produces. It splits soundness into batching / per-fold / query components,
    // each with its own grinding budget, rather than buying every bit with extra
    // queries.
    assert!(
        p.n + p.blowup_bits <= MAX_EXT_BITS,
        "extended domain 2^{} exceeds the {}-bit limit of pcs::Fri (n={}, blowup={})",
        p.n + p.blowup_bits,
        MAX_EXT_BITS,
        p.n,
        p.blowup_bits
    );
    let fri = Fri::new(FriConfig {
        field_size: goldilocks_safe_extension_field_size(),
        regime: DecodingRegime::Jbr,
        trace_length: 1u32 << p.n,
        rate: 1.0 / (1u64 << p.blowup_bits) as f64,
        batching: Batching::Powers,
        batch_size: p.n_functions,
        log_folding_factors: folding_factors.iter().map(|&d| d as u32).collect(),
        max_grinding_bits_query: p.grinding_bits,
        use_max_grinding_bits_query: true,
        tree_arity: arity,
        hash_size_bits: 256,
        target_security_bits: TARGET_SECURITY_BITS,
    });
    let n_queries = fri.security_params().n_queries;

    // The last fold is not opened: its layer is sent in the clear for the final
    // degree test. Every earlier fold contributes one opened layer.
    let n_open = folding_factors.len().saturating_sub(1);
    let fri_steps: Vec<FriStepHashes> = folding_factors[..n_open]
        .iter()
        .map(|&bit_drop| {
            let elements = (1u64 << bit_drop) * EXTENSION_DEGREE;
            FriStepHashes { bit_drop, elements, hashes: hashes_for_elements(elements, rate) }
        })
        .collect();
    let fri_hashes_per_query: u64 = fri_steps.iter().map(|s| s.hashes).sum();
    let total_linear_hashes = (trace_hashes_per_query + fri_hashes_per_query) * n_queries * p.n_proofs;

    // One authentication path per opened folding layer. Layer `i`'s tree holds
    // `2^h` leaves after its bit-drop is applied — the same tree list
    // `prover_memory_field_elems` allocates for.
    let mut h = ext_bits;
    let fri_merkle_per_query: u64 = folding_factors[..n_open]
        .iter()
        .map(|&bit_drop| {
            h -= bit_drop;
            merkle_path_perms(h, arity, compression)
        })
        .sum();
    let merkle_hashes_per_query = trace_merkle_per_query + fri_merkle_per_query;
    let total_merkle_hashes = merkle_hashes_per_query * n_queries * p.n_proofs;
    let folding_factors_out = folding_factors.clone();

    let hashes_per_query = trace_hashes_per_query + fri_hashes_per_query;
    let total_hashes = total_linear_hashes + total_merkle_hashes;
    let total_cells = total_hashes * p.hash.cells_per_perm();

    // Self-verification fit: pack total_cells into 2^n rows -> columns needed.
    let rows = 1u64 << p.n;
    let needed_stage1_cols = total_cells.div_ceil(rows);

    // Blake packing: a permutation occupies `rows_per_perm` rows of one instance,
    // so the verifier needs `total_hashes * rows_per_perm / 2^n` instances. When
    // the params are not blake-packed, fall back to the stage1 cell check.
    let rows_per_perm = rows_per_perm(p.hash.cells_per_perm(), p.hash.instance_cols());
    let blakes_needed = (total_hashes * rows_per_perm).div_ceil(rows);
    let fits = if p.blakes > 0 { blakes_needed <= p.blakes } else { needed_stage1_cols <= p.stage1_cols };

    let prover_memory_field_elems = prover_memory_field_elems(p, &folding_factors);
    let prover_memory_gb = (prover_memory_field_elems as f64 * 8.0) / (1024.0 * 1024.0 * 1024.0);

    CellEstimate {
        n_queries,
        folding_factors: folding_factors_out,
        trace_stages,
        fri_steps,
        trace_hashes_per_query,
        fri_hashes_per_query,
        hashes_per_query,
        merkle_hashes_per_query,
        n_proofs: p.n_proofs,
        total_linear_hashes,
        total_merkle_hashes,
        total_hashes,
        total_cells,
        needed_stage1_cols,
        assumed_stage1_cols: p.stage1_cols,
        blakes_needed,
        assumed_blakes: p.blakes,
        rows_per_perm,
        fits,
        prover_memory_field_elems,
        prover_memory_gb,
        grinding_bits: p.grinding_bits,
    }
}

/// Smallest number of blake instances that verifies itself: the least `k` with
/// `estimate(k).blakes_needed <= k`.
///
/// This is a fixed point, not a division. Packing another instance widens stage1
/// and stage2, and the verifier must hash its own widened trace — so the
/// requirement grows with `k` too. Where each added instance costs more
/// permutations than the `2^n` rows it contributes, no `k` ever catches up and
/// this returns `None` (raise `n`, the blowup, or the folding factor instead).
pub fn min_blakes_for_self_fit(
    n: u32,
    blowup_bits: u32,
    fold_bits: u32,
    hash: HashFamily,
    max_blakes: u64,
) -> Option<u64> {
    (1..=max_blakes).find(|&k| {
        estimate_linear_hash_cells(&CellModelParams::recursion_blakes(n, blowup_bits, fold_bits, k, hash)).fits
    })
}

/// Result of checking whether the hardcoded ZisK Main verifier's linear-hash
/// cells fit inside a recursion estimate's total-cell budget.
#[derive(Debug, Clone)]
pub struct MainFitCheck {
    /// The recursion run's total linear-hash cells (the budget).
    pub recursion_total_cells: u64,
    /// The full ZisK Main estimate.
    pub main: CellEstimate,
    /// ZisK Main's total linear-hash cells.
    pub main_total_cells: u64,
    /// Whether Main fits: `main_total_cells <= recursion_total_cells`.
    pub main_fits: bool,
}

/// Given a recursion estimate, compute the hardcoded ZisK Main verifier estimate
/// (sharing `fold_bits`, `hash` and `grinding_bits`) and check whether Main's
/// total cells fit within the recursion's total-cell budget.
pub fn main_fits_in_recursion(recursion: &CellEstimate, fold_bits: u32, hash: HashFamily) -> MainFitCheck {
    let mut main_params = CellModelParams::zisk_main(fold_bits, hash);
    main_params.grinding_bits = recursion.grinding_bits;
    let main = estimate_linear_hash_cells(&main_params);
    let main_total_cells = main.total_cells;
    MainFitCheck {
        recursion_total_cells: recursion.total_cells,
        main_total_cells,
        main_fits: main_total_cells <= recursion.total_cells,
        main,
    }
}

impl CellEstimate {
    /// Human-readable per-stage + FRI breakdown.
    pub fn report(&self) -> String {
        let mut s = String::new();
        s.push_str(&format!("n_queries        = {}\n", self.n_queries));
        s.push_str(&format!("folding_factors  = {:?}\n", self.folding_factors));
        s.push_str("--- trace / Q stages (no extension) ---\n");
        for st in &self.trace_stages {
            s.push_str(&format!("  {:8} cols={:>6}  hashes={:>4}\n", st.name, st.cols, st.hashes));
        }
        s.push_str("--- FRI folding layers (x3 extension, last layer excluded) ---\n");
        for (i, st) in self.fri_steps.iter().enumerate() {
            s.push_str(&format!(
                "  step {:<2} drop={}  elements={:>4}  hashes={:>4}\n",
                i, st.bit_drop, st.elements, st.hashes
            ));
        }
        s.push_str(&format!("trace_hashes/query = {}\n", self.trace_hashes_per_query));
        s.push_str(&format!("fri_hashes/query   = {}\n", self.fri_hashes_per_query));
        s.push_str(&format!("leaf_hashes/query  = {}\n", self.hashes_per_query));
        s.push_str(&format!("merkle_hashes/query= {}\n", self.merkle_hashes_per_query));
        s.push_str(&format!(
            "linear_hashes      = {} (= {} leaf hashes/query x {} queries x {} proofs)\n",
            self.total_linear_hashes, self.hashes_per_query, self.n_queries, self.n_proofs
        ));
        let merkle_pct = if self.total_hashes > 0 {
            100.0 * self.total_merkle_hashes as f64 / self.total_hashes as f64
        } else {
            0.0
        };
        s.push_str(&format!("merkle_hashes      = {} ({:.0}% of total)\n", self.total_merkle_hashes, merkle_pct));
        s.push_str(&format!("total_hashes       = {}\n", self.total_hashes));
        s.push_str(&format!("total_cells        = {}\n", self.total_cells));
        s.push_str(&format!(
            "prover_memory      = {:.2} GB ({} field elems)\n",
            self.prover_memory_gb, self.prover_memory_field_elems
        ));
        s
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ceil_hashes_50_cols_is_13() {
        // The worked example: 50 columns at sponge rate 4 -> ceil(50/4) = 13.
        assert_eq!(hashes_for_elements(50, 4), 13);
    }

    #[test]
    fn fri_step_hashes_extension_3() {
        // rate 4: bit-drop 4 -> 16 siblings x 3 = 48 elements -> ceil(48/4) = 12.
        assert_eq!(hashes_for_elements((1u64 << 4) * EXTENSION_DEGREE, 4), 12);
        // rate 4: bit-drop 3 -> 8 siblings x 3 = 24 elements -> ceil(24/4) = 6.
        assert_eq!(hashes_for_elements((1u64 << 3) * EXTENSION_DEGREE, 4), 6);
        // rate 12 (Poseidon2/Blake2): 24 elements -> ceil(24/12) = 2.
        assert_eq!(hashes_for_elements((1u64 << 3) * EXTENSION_DEGREE, 12), 2);
    }

    /// Convenience: a Blake3 hash with the given cell cost for tests.
    fn b3(cells: u64) -> HashFamily {
        HashFamily::Blake3 { cells_per_perm: cells }
    }

    // -----------------------------------------------------------------------
    // Per-family tree arity
    // -----------------------------------------------------------------------

    /// The compression block fixes the natural arity: blake3's 64-byte block
    /// holds two digests (binary tree), blake2b's 128-byte block holds four
    /// (quaternary). Either way a node is exactly one permutation.
    #[test]
    fn compression_block_fixes_the_tree_arity() {
        let b3 = HashFamily::Blake3 { cells_per_perm: BLAKE3_CELLS_PER_PERM };
        let b2 = HashFamily::Blake2 { cells_per_perm: BLAKE2_CELLS_PER_PERM };
        assert_eq!((b3.tree_arity(), b2.tree_arity()), (2, 4));
        assert_eq!(merkle_node_perms(b3.tree_arity(), b3.compression_elems()), 1);
        assert_eq!(merkle_node_perms(b2.tree_arity(), b2.compression_elems()), 1);
    }

    /// Blake2b's quaternary tree genuinely HALVES the permutations per path:
    /// blake3 walks 22 levels at one block each, blake2b walks 11.
    #[test]
    fn blake2_halves_the_path_permutations_of_blake3() {
        let b3 = HashFamily::Blake3 { cells_per_perm: BLAKE3_CELLS_PER_PERM };
        let b2 = HashFamily::Blake2 { cells_per_perm: BLAKE2_CELLS_PER_PERM };
        let path = |h: HashFamily, bits| merkle_path_perms(bits, h.tree_arity(), h.compression_elems());
        assert_eq!(path(b3, 22), 22, "binary tree: one block per level");
        assert_eq!(path(b2, 22), 11, "quaternary tree: half the levels");
        assert_eq!(path(b3, 22), 2 * path(b2, 22));
    }

    /// Path cost scales as `arity / log2(arity)` x `DIGEST_ELEMS / block`, so the
    /// arity that exactly fills the compression block is optimal:
    ///   - below it the node underfills the block and wastes it (blake2b laid out
    ///     binary doubles: 24 levels x a half-used block instead of 12 x a full one)
    ///   - above it the extra blocks per node outrun the levels saved (arity 8)
    ///   - arity 2 and 4 happen to TIE for blake3, since 2/log2(2) = 4/log2(4) = 2
    ///
    /// This is why each family's natural arity is block / DIGEST_ELEMS.
    #[test]
    fn natural_arity_fills_the_block_and_is_optimal() {
        // blake3, 8-element block: binary and quaternary tie, arity 8 is worse.
        assert_eq!(merkle_path_perms(24, 2, BLAKE3_COMPRESSION_ELEMS), 24);
        assert_eq!(merkle_path_perms(24, 4, BLAKE3_COMPRESSION_ELEMS), 24);
        assert_eq!(merkle_path_perms(24, 8, BLAKE3_COMPRESSION_ELEMS), 32);
        // blake2b, 16-element block: quaternary is optimal, binary WASTES half of
        // every block and costs double.
        assert_eq!(merkle_path_perms(24, 4, BLAKE2_COMPRESSION_ELEMS), 12);
        assert_eq!(merkle_path_perms(24, 2, BLAKE2_COMPRESSION_ELEMS), 24);
        assert_eq!(merkle_path_perms(24, 8, BLAKE2_COMPRESSION_ELEMS), 16);
    }

    // -----------------------------------------------------------------------
    // Tree-node compression (distinct from sponge absorption)
    // -----------------------------------------------------------------------

    /// A Merkle node is a fixed-width compression, not a sponge absorption, so it
    /// uses the full block: blake3 64B = 8 elements, blake2b 128B = 16.
    #[test]
    fn compression_block_is_wider_than_the_sponge_rate() {
        let b3 = HashFamily::Blake3 { cells_per_perm: BLAKE3_CELLS_PER_PERM };
        let b2 = HashFamily::Blake2 { cells_per_perm: BLAKE2_CELLS_PER_PERM };
        assert_eq!((b3.sponge_rate(), b3.compression_elems()), (4, 8));
        assert_eq!((b2.sponge_rate(), b2.compression_elems()), (12, 16));
    }

    /// An arity-4 node absorbs 4 digests x 4 elements = 16 elements = 128 bytes.
    /// That is exactly ONE blake2b block (4-to-1) and two blake3 blocks.
    #[test]
    fn blake2_compresses_four_children_in_one_permutation() {
        assert_eq!(merkle_node_perms(4, BLAKE2_COMPRESSION_ELEMS), 1);
        assert_eq!(merkle_node_perms(4, BLAKE3_COMPRESSION_ELEMS), 2);
    }

    // -----------------------------------------------------------------------
    // Per-family instance geometry
    // -----------------------------------------------------------------------

    /// Each family has its own committed width and permutation shape, taken from
    /// `examples/hashes` (README throughput table + generated trace rows).
    #[test]
    fn each_hash_family_carries_its_own_instance_geometry() {
        let b3 = HashFamily::Blake3 { cells_per_perm: BLAKE3_CELLS_PER_PERM };
        assert_eq!((b3.fixed_cols(), b3.stage1_cols(), b3.stage2_cols()), (8, 54, 48));
        assert_eq!(b3.instance_cols(), 108);

        let b2 = HashFamily::Blake2 { cells_per_perm: BLAKE2_CELLS_PER_PERM };
        assert_eq!((b2.fixed_cols(), b2.stage1_cols(), b2.stage2_cols()), (7, 100, 84));
        assert_eq!(b2.instance_cols(), 190);
    }

    /// Blake2b is 96 clocks x 190 columns = 18240 cells (12 rounds x 8 G), so its
    /// permutation spans well over Blake3's 56 rows.
    #[test]
    fn blake2_permutation_is_96_rows_of_190_columns() {
        assert_eq!(BLAKE2_CELLS_PER_PERM, 18240);
        assert_eq!(rows_per_perm(BLAKE2_CELLS_PER_PERM, BLAKE2_INSTANCE_COLS), 96);
    }

    /// Stage widths scale with the instance count; fixed columns are the shared
    /// round schedule, so they do NOT scale.
    #[test]
    fn recursion_blakes_scales_stages_but_not_fixed() {
        let hash = HashFamily::Blake2 { cells_per_perm: BLAKE2_CELLS_PER_PERM };
        let p = CellModelParams::recursion_blakes(20, 2, 3, 3, hash);
        assert_eq!((p.stage1_cols, p.stage2_cols), (300, 252));
        assert_eq!(p.fixed_cols, 7, "fixed columns are shared across packed instances");
    }

    // -----------------------------------------------------------------------
    // Blake-packing: how many blake instances the verifier needs side by side
    // -----------------------------------------------------------------------

    /// examples/hashes: one blake3 permutation is 56 clocks x 108 columns = 6048
    /// cells, so the per-permutation cost decomposes exactly into rows.
    #[test]
    fn blake3_permutation_is_56_rows_of_108_columns() {
        assert_eq!(BLAKE3_CELLS_PER_PERM, 6048);
        assert_eq!(BLAKE3_INSTANCE_COLS, 108);
        assert_eq!(rows_per_perm(BLAKE3_CELLS_PER_PERM, BLAKE3_INSTANCE_COLS), 56);
    }

    /// The stage split is per instance and scales linearly with the count.
    #[test]
    fn recursion_blakes_scales_stage_columns_linearly() {
        let one = CellModelParams::recursion_blakes(17, 2, 3, 1, b3(BLAKE3_CELLS_PER_PERM));
        assert_eq!((one.stage1_cols, one.stage2_cols), (54, 48));
        let two = CellModelParams::recursion_blakes(17, 2, 3, 2, b3(BLAKE3_CELLS_PER_PERM));
        assert_eq!((two.stage1_cols, two.stage2_cols), (108, 96));
    }

    /// A permutation occupies 56 rows of ONE instance, so `k` instances clear
    /// `k * 2^n / 56` permutations. This is the count the map has to report.
    #[test]
    fn blakes_needed_is_total_hashes_scaled_by_rows_per_perm() {
        let p = CellModelParams::recursion_blakes(17, 2, 3, 4, b3(BLAKE3_CELLS_PER_PERM));
        let e = estimate_linear_hash_cells(&p);
        let expected = (e.total_hashes * 56).div_ceil(1u64 << 17);
        assert_eq!(e.blakes_needed, expected);
    }

    /// With the blake preset the fit check asks whether the assumed instances are
    /// enough — not whether cells squeeze into stage1 alone.
    #[test]
    fn blake_preset_fit_compares_against_assumed_instances() {
        let hash = b3(BLAKE3_CELLS_PER_PERM);
        let tight = estimate_linear_hash_cells(&CellModelParams::recursion_blakes(17, 2, 3, 1, hash));
        assert!(!tight.fits, "one instance cannot absorb the whole verifier");
        assert_eq!(tight.assumed_blakes, 1);
        assert!(tight.blakes_needed > 1);
    }

    /// Packing another blake widens stage1/stage2, and the verifier has to hash
    /// its own widened trace — so the requirement is self-referential, not a
    /// single division. This is why the answer needs a fixed point.
    #[test]
    fn adding_a_blake_raises_the_requirement_too() {
        let hash = b3(BLAKE3_CELLS_PER_PERM);
        let one = estimate_linear_hash_cells(&CellModelParams::recursion_blakes(17, 2, 3, 1, hash));
        let many = estimate_linear_hash_cells(&CellModelParams::recursion_blakes(17, 2, 3, 8, hash));
        assert!(many.blakes_needed > one.blakes_needed, "wider stages cost more hashes");
    }

    /// The solver returns the smallest self-consistent instance count, and that
    /// count really does fit.
    #[test]
    fn min_blakes_for_self_fit_returns_a_fixed_point() {
        let hash = b3(BLAKE3_CELLS_PER_PERM);
        let k = min_blakes_for_self_fit(21, 2, 3, hash, 512).expect("expected a fixed point at n=21");
        let at_k = estimate_linear_hash_cells(&CellModelParams::recursion_blakes(21, 2, 3, k, hash));
        assert!(at_k.fits, "solver's answer must fit");
        if k > 1 {
            let below = estimate_linear_hash_cells(&CellModelParams::recursion_blakes(21, 2, 3, k - 1, hash));
            assert!(!below.fits, "k must be the SMALLEST fitting count");
        }
    }

    /// When every added blake costs more hashes than it clears, there is no fixed
    /// point and the solver must say so rather than loop.
    #[test]
    fn min_blakes_for_self_fit_reports_divergence() {
        let hash = b3(BLAKE3_CELLS_PER_PERM);
        assert_eq!(min_blakes_for_self_fit(17, 2, 3, hash, 512), None);
    }

    /// Presets that are not blake-packed (ZisK Main) keep the stage1 cell check.
    #[test]
    fn non_blake_preset_keeps_stage1_fit() {
        let e = estimate_linear_hash_cells(&CellModelParams::zisk_main(3, b3(700)));
        assert_eq!(e.fits, e.needed_stage1_cols <= e.assumed_stage1_cols);
    }

    // -----------------------------------------------------------------------
    // Merkle authentication-path accounting
    // -----------------------------------------------------------------------

    /// AUDIT: every FRI layer must be accounted for exactly once. The schedule
    /// folds the extended domain down to degree 2^6; each fold commits the folded
    /// codeword in its own tree, and the verifier opens ALL of them except the
    /// last, whose polynomial is sent in the clear for the final degree test.
    ///
    /// Layer i's tree has `2^h` leaves after drop i is applied, and holds
    /// `2^drop_i` extension values per leaf — so the leaf sizes and the tree
    /// heights must line up index for index.
    #[test]
    fn every_fri_layer_is_opened_except_the_final_one() {
        for n in [17u32, 20, 22, 24] {
            for blowup in 1u32..=8 {
                if n + blowup > MAX_EXT_BITS {
                    continue;
                }
                for fold_bits in 1u32..=4 {
                    let ext = (n + blowup) as u64;
                    let ff = derive_folding_factors(n, blowup, fold_bits);
                    let total: u64 = ff.iter().sum();

                    // the schedule folds the DIMENSION to 2^6, so it is independent
                    // of the blowup and always leaves a representable polynomial
                    assert_eq!(
                        total,
                        (n - 6) as u64,
                        "schedule must land on dimension 2^6 (n={n} blowup={blowup} fold={fold_bits})"
                    );

                    let p = CellModelParams::recursion_blakes(n, blowup, fold_bits, 4, b3(BLAKE3_CELLS_PER_PERM));
                    let e = estimate_linear_hash_cells(&p);

                    // one opened step per fold, minus the final unopened layer
                    assert_eq!(e.fri_steps.len(), ff.len() - 1, "opened layers must be folds - 1");

                    // heights walk down with the drops, and the FIRST opened tree
                    // is the one AFTER the first fold (the 2^ext oracles are the
                    // stage trees, counted separately)
                    let mut h = ext;
                    for (i, step) in e.fri_steps.iter().enumerate() {
                        assert_eq!(step.bit_drop, ff[i], "step {i} drop mismatch");
                        assert_eq!(step.elements, (1u64 << ff[i]) * EXTENSION_DEGREE, "step {i} leaf size");
                        h -= ff[i];
                    }
                    // after the opened steps, exactly one fold remains; the final
                    // layer's DOMAIN is 2^(6 + blowup) holding 2^6 coefficients
                    let remaining = ff[ff.len() - 1];
                    assert_eq!(h - remaining, 6 + blowup as u64, "final layer domain must be 2^(6+blowup)");
                }
            }
        }
    }

    /// AUDIT: pil2-stark's `merkleTreeGL` supports arity 3 (PoseidonGoldilocks<12>),
    /// so a future family with a 12-element block would produce arity 3. The depth
    /// helper must handle a non-power-of-two arity rather than dividing by zero.
    #[test]
    fn merkle_path_depth_handles_non_power_of_two_arity() {
        // 2^20 leaves at arity 3: ceil(20 / log2 3) = ceil(12.62) = 13 levels.
        assert_eq!(merkle_path_depth(20, 3), 13);
        assert_eq!(merkle_path_depth(9, 3), 6); // ceil(9 / 1.585) = ceil(5.68)
    }

    /// Arity 4 groups two bits of height per level, so a 2^20-leaf tree is 10
    /// levels deep.
    #[test]
    fn merkle_path_depth_groups_log2_arity_bits_per_level() {
        assert_eq!(merkle_path_depth(20, 4), 10);
        assert_eq!(merkle_path_depth(2, 4), 1);
        assert_eq!(merkle_path_depth(0, 4), 0);
    }

    /// An odd height still needs a whole final level.
    #[test]
    fn merkle_path_depth_rounds_partial_level_up() {
        assert_eq!(merkle_path_depth(21, 4), 11);
    }

    /// One arity-4 node feeds 4 child digests of 4 field elements = 16 elements
    /// through the compression block: 2 blocks for blake3 (8), 1 for blake2b (16).
    #[test]
    fn merkle_node_perms_fills_the_compression_block() {
        assert_eq!(merkle_node_perms(4, BLAKE3_COMPRESSION_ELEMS), 2);
        assert_eq!(merkle_node_perms(4, BLAKE2_COMPRESSION_ELEMS), 1);
    }

    /// A full path is depth x per-node cost.
    #[test]
    fn merkle_path_perms_is_depth_times_node_cost() {
        assert_eq!(merkle_path_perms(20, 4, BLAKE3_COMPRESSION_ELEMS), 20);
        assert_eq!(merkle_path_perms(20, 4, BLAKE2_COMPRESSION_ELEMS), 10);
    }

    /// The headline total is exactly the leaf (linear-hash) work plus the
    /// authentication-path work — no double counting, nothing dropped.
    #[test]
    fn total_hashes_splits_into_linear_and_merkle() {
        let e = estimate_linear_hash_cells(&CellModelParams::recursion(17, 2, 3, 100, b3(1)));
        assert!(e.total_merkle_hashes > 0, "merkle paths must be counted");
        assert_eq!(e.total_hashes, e.total_linear_hashes + e.total_merkle_hashes);
        assert_eq!(e.total_cells, e.total_hashes * 1);
    }

    /// Raising the blowup buys fewer queries but taller trees, so the Merkle
    /// share of the work rises with it. This is the trade-off the map exists to
    /// show, so pin it.
    #[test]
    fn merkle_share_grows_with_blowup() {
        let share = |blowup: u32| {
            let e = estimate_linear_hash_cells(&CellModelParams::recursion(17, blowup, 3, 100, b3(1)));
            e.total_merkle_hashes as f64 / e.total_hashes as f64
        };
        assert!(share(6) > share(1), "merkle share should grow with blowup: {} vs {}", share(6), share(1));
    }

    /// Build a HashFamily from the `HASH` env var; `blake_cells` is the
    /// per-permutation cost for the (investigated) Blake variants.
    fn hash_from_env(blake_cells: u64) -> HashFamily {
        match std::env::var("HASH").unwrap_or_default().to_lowercase().as_str() {
            "blake2" => HashFamily::Blake2 { cells_per_perm: blake_cells },
            _ => HashFamily::Blake3 { cells_per_perm: blake_cells },
        }
    }

    #[test]
    fn folding_factors_n15_blowup6_fold4() {
        // FRI folds the POLYNOMIAL, so the schedule runs on the dimension:
        // 15 -> 11 -> 7 -> [6]; the tail drop is sized to land on 6, never above
        // fold_bits.
        assert_eq!(derive_folding_factors(15, 6, 4), vec![4, 4, 1]);
    }

    #[test]
    fn folding_factors_n15_blowup6_fold3() {
        // Dimension 15, 3-bit steps: 15->12->9->[6], drops sum to 9.
        assert_eq!(derive_folding_factors(15, 6, 3), vec![3, 3, 3]);
    }

    #[test]
    fn folding_factors_land_on_dimension_6() {
        // The drops fold the DIMENSION down to 2^6, so they sum to `n - 6` and
        // the schedule does not depend on the blowup. Each drop is 1..=fold_bits.
        for fold_bits in 2u32..=5 {
            for n in 8u32..24 {
                for blowup in 1u32..=8 {
                    let f = derive_folding_factors(n, blowup, fold_bits);
                    let sum: u64 = f.iter().sum();
                    assert_eq!(sum, (n - 6) as u64, "fold={fold_bits} n={n} blowup={blowup} -> {f:?} sum wrong");
                    assert!(
                        f.iter().all(|&d| (1..=fold_bits as u64).contains(&d)),
                        "fold={fold_bits} n={n} blowup={blowup} -> {f:?} bad drop"
                    );
                }
            }
        }
    }

    /// The whole point of the fix: folding the polynomial cannot depend on how
    /// much the domain is blown up, and the final dimension must stay positive.
    #[test]
    fn fold_schedule_is_independent_of_blowup() {
        for n in [17u32, 20, 24] {
            let base = derive_folding_factors(n, 1, 3);
            for blowup in 2u32..=8 {
                assert_eq!(derive_folding_factors(n, blowup, 3), base, "schedule moved with blowup at n={n}");
            }
            // dimension after every drop stays above the stopping degree
            let mut dim = n as i64;
            for d in &base {
                dim -= *d as i64;
            }
            assert_eq!(dim, 6, "must land exactly on dimension 2^6 at n={n}");
        }
    }

    #[test]
    fn trace_hashes_fixed_stageq_stage2() {
        // fixed=ceil(28/4)=7, stageQ=ceil(18/4)=5, stage2=ceil(0/4)=0.
        let est = estimate_linear_hash_cells(&CellModelParams::recursion(15, 6, 4, 0, b3(1)));
        let by = |name| est.trace_stages.iter().find(|s| s.name == name).unwrap().hashes;
        assert_eq!(by("fixed"), 7);
        assert_eq!(by("stageQ"), 5);
        assert_eq!(by("stage2"), 0);
        assert_eq!(by("stage1"), 0);
        assert_eq!(est.trace_hashes_per_query, 12);
    }

    #[test]
    fn fri_excludes_last_layer() {
        // dimension 15 folds [4,4,1] -> open the first 2 layers only: 12+12 = 24.
        let est = estimate_linear_hash_cells(&CellModelParams::recursion(15, 6, 4, 0, b3(1)));
        assert_eq!(est.folding_factors, vec![4, 4, 1]);
        assert_eq!(est.fri_steps.len(), 2);
        assert_eq!(est.fri_hashes_per_query, 24);
    }

    /// Report harness: estimate a *recursion* verifier from the chosen params,
    /// then check whether the hardcoded ZisK Main verifier's cells fit inside the
    /// recursion's total-cell budget. `BLOWUP_FACTOR` is the blowup *exponent*
    /// (rate = 1/2^BLOWUP_FACTOR); `N` is the dimension exponent; `FOLD_BITS` is
    /// the FRI folding step size (shared by both estimates).
    ///   N=18 BLOWUP_FACTOR=6 FOLD_BITS=3 COLS_STAGE1=100 CELLS_PER_BLAKE3=5000 \
    ///     cargo test -p pil2-stark-setup --lib types::cells::tests::cells_report -- --ignored --nocapture
    #[test]
    #[ignore]
    fn cells_report() {
        let env = |k: &str, d: u64| std::env::var(k).ok().and_then(|v| v.parse().ok()).unwrap_or(d);
        let n = env("N", 18) as u32;
        let blowup_bits = env("BLOWUP_FACTOR", 6) as u32;
        let fold_bits = env("FOLD_BITS", 3) as u32;
        let stage1_cols = env("COLS_STAGE1", 100);
        // HASH = blake3 | blake2 | poseidon2 (default blake3); CELLS_PER_BLAKE3 is
        // the per-permutation cell cost for the Blake variants.
        let hash = hash_from_env(env("CELLS_PER_BLAKE3", 5000));
        // Recursion stage2 defaults to stage1/3; override with COLS_STAGE2.
        let stage2_cols = env("COLS_STAGE2", stage1_cols / 3);

        let rec_params =
            CellModelParams::recursion_with_stage2(n, blowup_bits, fold_bits, stage1_cols, stage2_cols, hash);
        let rec = estimate_linear_hash_cells(&rec_params);
        eprintln!("\n========== RECURSION verifier ==========");
        eprintln!("params: {rec_params:?}");
        eprint!("{}", rec.report());

        let chk = main_fits_in_recursion(&rec, fold_bits, hash);
        eprintln!("\n========== ZisK MAIN verifier (hardcoded) ==========");
        eprintln!("total_hashes  = {}", chk.main.total_hashes);
        eprintln!("total_cells   = {}", chk.main_total_cells);
        eprintln!("prover_memory = {:.2} GB", chk.main.prover_memory_gb);

        eprintln!("\n========== SUMMARY ==========");
        // Self-verification fit: recursion's cells must pack into its own 2^n rows.
        eprintln!(
            "self-verification: needed_stage1_cols = {} vs assumed {} -> fits = {} ({} {} {})",
            rec.needed_stage1_cols,
            rec.assumed_stage1_cols,
            rec.fits,
            rec.needed_stage1_cols,
            if rec.fits { "<=" } else { ">" },
            rec.assumed_stage1_cols
        );
        // Main-fits-in-recursion: Main's cells must fit the recursion cell budget.
        eprintln!(
            "main-fits-in-recursion: main_total_cells = {} vs recursion {} -> fits = {} ({} {} {})",
            chk.main_total_cells,
            chk.recursion_total_cells,
            chk.main_fits,
            chk.main_total_cells,
            if chk.main_fits { "<=" } else { ">" },
            chk.recursion_total_cells
        );
    }

    /// The browser model derives query counts in closed form as
    /// `ceil((target - grinding) / -log2(1 - pp))` with `pp = 1 - sqrt(rate) - 1/300`
    /// (JBR at alpha = 0). That must reproduce `pcs::Fri` exactly, or the page and
    /// the proving-key pipeline disagree.
    #[test]
    fn closed_form_query_count_matches_pcs() {
        for n in [17u32, 20, 24] {
            for blowup in 1u32..=8 {
                if n + blowup > MAX_EXT_BITS {
                    continue;
                }
                for grinding in [0u64, 20, 32] {
                    let mut p = CellModelParams::recursion(n, blowup, 3, 100, b3(6048));
                    p.grinding_bits = grinding;
                    let e = estimate_linear_hash_cells(&p);
                    let rate = 1.0 / (1u64 << blowup) as f64;
                    let pp = 1.0 - rate.sqrt() - 1.0 / 300.0;
                    let predicted = (((TARGET_SECURITY_BITS - grinding) as f64) / -(1.0 - pp).log2()).ceil() as u64;
                    assert_eq!(
                        predicted, e.n_queries,
                        "closed form diverged at n={n} blowup=2^{blowup} grinding={grinding}"
                    );
                }
            }
        }
    }

    /// Every soundness component must still clear the target; the query count is
    /// only sound in combination with the grinding pcs allocates.
    #[test]
    fn pcs_reaches_the_security_target() {
        use crate::types::security::pcs::{Batching, Fri, FriConfig};
        use crate::types::security::regimes::DecodingRegime;
        for n in [17u32, 20, 24] {
            for blowup in 1u32..=8 {
                if n + blowup > MAX_EXT_BITS {
                    continue;
                }
                let p = CellModelParams::recursion(n, blowup, 3, 100, b3(6048));
                let ff = derive_folding_factors(n, blowup, 3);
                let fri = Fri::new(FriConfig {
                    field_size: crate::types::security::goldilocks_safe_extension_field_size(),
                    regime: DecodingRegime::Jbr,
                    trace_length: 1u32 << n,
                    rate: 1.0 / (1u64 << blowup) as f64,
                    batching: Batching::Powers,
                    batch_size: p.n_functions,
                    log_folding_factors: ff.iter().map(|&d| d as u32).collect(),
                    max_grinding_bits_query: p.grinding_bits,
                    use_max_grinding_bits_query: true,
                    tree_arity: p.hash.tree_arity(),
                    hash_size_bits: 256,
                    target_security_bits: TARGET_SECURITY_BITS,
                });
                let min = fri.security_levels().iter().map(|(_, b)| *b).min().unwrap();
                assert!(min >= 128, "n={n} blowup=2^{blowup} only reaches {min} bits");
            }
        }
    }

    /// AUDIT: full per-cell dump for cross-checking the HTML port field by field.
    ///   HASH=blake3|blake2 BLAKES=k cargo test -p pil2-stark-setup --lib \
    ///     types::cells::tests::audit_dump -- --ignored --nocapture
    #[test]
    #[ignore]
    fn audit_dump() {
        let hash = match std::env::var("HASH").unwrap_or_default().to_lowercase().as_str() {
            "blake2" => HashFamily::Blake2 { cells_per_perm: BLAKE2_CELLS_PER_PERM },
            _ => HashFamily::Blake3 { cells_per_perm: BLAKE3_CELLS_PER_PERM },
        };
        let k: u64 = std::env::var("BLAKES").ok().and_then(|v| v.parse().ok()).unwrap_or(4);
        println!("n,blowup,queries,leaf_pq,merkle_pq,linear,merkle,total_hashes,total_cells,blakes_needed,rows_per_perm,fits,mem_elems,main_hashes,main_cells,main_fits");
        for n in 17u32..=24 {
            for blowup in 1u32..=8 {
                if n + blowup > MAX_EXT_BITS {
                    continue;
                }
                let p = CellModelParams::recursion_blakes(n, blowup, 3, k, hash);
                let e = estimate_linear_hash_cells(&p);
                let chk = main_fits_in_recursion(&e, 3, hash);
                println!(
                    "{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}",
                    n,
                    blowup,
                    e.n_queries,
                    e.hashes_per_query,
                    e.merkle_hashes_per_query,
                    e.total_linear_hashes,
                    e.total_merkle_hashes,
                    e.total_hashes,
                    e.total_cells,
                    e.blakes_needed,
                    e.rows_per_perm,
                    e.fits,
                    e.prover_memory_field_elems,
                    chk.main.total_hashes,
                    chk.main_total_cells,
                    chk.main_fits,
                );
            }
        }
    }

    /// How many blake3 instances the recursive verifier needs packed side by
    /// side to verify itself, across (N, blowup) at folding factor 3. `-` means
    /// no fixed point: every added instance costs more permutations than the
    /// 2^N rows it contributes.
    ///   cargo test -p pil2-stark-setup --lib types::cells::tests::blake_packing_table -- --ignored --nocapture
    #[test]
    #[ignore]
    fn blake_packing_table() {
        let hash = match std::env::var("HASH").unwrap_or_default().to_lowercase().as_str() {
            "blake2" => HashFamily::Blake2 { cells_per_perm: BLAKE2_CELLS_PER_PERM },
            _ => HashFamily::Blake3 { cells_per_perm: BLAKE3_CELLS_PER_PERM },
        };
        let fold_bits = 3;
        print!("{:>4}", "N\\b");
        for blowup in 1u32..=8 {
            print!("{:>6}", format!("2^{blowup}"));
        }
        println!();
        for n in 17u32..=24 {
            print!("{n:>4}");
            for blowup in 1u32..=8 {
                match min_blakes_for_self_fit(n, blowup, fold_bits, hash, 4096) {
                    Some(k) => print!("{k:>6}"),
                    None => print!("{:>6}", "-"),
                }
            }
            println!();
        }
    }

    /// CSV sweep over (N, blowup) for the recursion verifier, with Main-fit. Lets
    /// us see the data shape before picking a graphic. Fixed: fold=3, stage1=100,
    /// cells_per_blake3=5000 (override via env COLS_STAGE1 / FOLD_BITS / CELLS_PER_BLAKE3).
    /// Hash totals are split into leaf (linear) and Merkle-path work.
    ///   cargo test -p pil2-stark-setup --lib types::cells::tests::cells_sweep_csv -- --ignored --nocapture
    #[test]
    #[ignore]
    fn cells_sweep_csv() {
        let env = |k: &str, d: u64| std::env::var(k).ok().and_then(|v| v.parse().ok()).unwrap_or(d);
        let fold_bits = env("FOLD_BITS", 3) as u32;
        let stage1 = env("COLS_STAGE1", 100);
        let hash = hash_from_env(env("CELLS_PER_BLAKE3", 5000));

        println!("N,blowup,fold,stage1,n_queries,hashes_per_query,linear_hashes,merkle_hashes,total_hashes,total_cells,needed_stage1,self_fits,prover_mem_gb,main_total_cells,main_fits");
        for n in 14u32..=20 {
            for blowup in 1u32..=8 {
                let rec = estimate_linear_hash_cells(&CellModelParams::recursion(n, blowup, fold_bits, stage1, hash));
                let chk = main_fits_in_recursion(&rec, fold_bits, hash);
                println!(
                    "{},{},{},{},{},{},{},{},{},{},{},{},{:.3},{},{}",
                    n,
                    blowup,
                    fold_bits,
                    stage1,
                    rec.n_queries,
                    rec.hashes_per_query,
                    rec.total_linear_hashes,
                    rec.total_merkle_hashes,
                    rec.total_hashes,
                    rec.total_cells,
                    rec.needed_stage1_cols,
                    rec.fits,
                    rec.prover_memory_gb,
                    chk.main_total_cells,
                    chk.main_fits,
                );
            }
        }
    }

    #[test]
    fn main_fits_in_recursion_total_cells() {
        // Run a recursion estimate, then check the hardcoded ZisK Main verifier's
        // total_cells fits within the recursion's total_cells budget.
        let rec = estimate_linear_hash_cells(&CellModelParams::recursion(18, 6, 3, 100, b3(5000)));
        let chk = main_fits_in_recursion(&rec, 3, b3(5000));
        assert_eq!(chk.recursion_total_cells, rec.total_cells);
        assert_eq!(chk.main_total_cells, chk.main.total_cells);
        assert_eq!(chk.main_fits, chk.main.total_cells <= rec.total_cells);
    }

    /// ZisK Main basic-AIR verifier: nBits=22, blowup=1, fold=3, single proof,
    /// stages fixed=3 / stageQ=6 / stage2=24 / stage1=38, 3 opening points,
    /// 61 evaluated functions. Checks the fit independently of recursion.
    #[test]
    fn zisk_main_fit_check() {
        let p = CellModelParams::zisk_main(3, b3(700));
        let est = estimate_linear_hash_cells(&p);
        // Single proof, so the leaf-hash total is hashes_per_query * n_queries * 1.
        assert_eq!(est.n_proofs, 1);
        assert_eq!(est.total_linear_hashes, est.hashes_per_query * est.n_queries);
        // Trace hashes use the Main stage columns directly:
        // ceil(3/4)=1 + ceil(6/4)=2 + ceil(24/4)=6 + ceil(38/4)=10 = 19.
        assert_eq!(est.trace_hashes_per_query, 1 + 2 + 6 + 10);
        // Fit check is internally consistent.
        assert_eq!(est.needed_stage1_cols, est.total_cells.div_ceil(1u64 << 22));
        assert_eq!(est.fits, est.needed_stage1_cols <= 38);
    }

    #[test]
    fn self_verification_fit_check() {
        // The verifier autoverifies to itself: its total_cells must pack into
        // `2^n` rows. needed_stage1_cols = ceil(total_cells / 2^n), compared to
        // the stage1_cols we assumed.
        let p = CellModelParams::recursion(15, 6, 4, 100, b3(700));
        let est = estimate_linear_hash_cells(&p);
        let rows = 1u64 << 15;
        assert_eq!(est.needed_stage1_cols, est.total_cells.div_ceil(rows));
        // fits == (needed <= assumed).
        assert_eq!(est.fits, est.needed_stage1_cols <= 100);
    }

    #[test]
    fn end_to_end_consistency() {
        // n=15, blowup=6, stage1=100, cells_per_blake3=700.
        // stage2 defaults to stage1/3 = 33.
        let est = estimate_linear_hash_cells(&CellModelParams::recursion(15, 6, 4, 100, b3(700)));
        // trace = ceil(28/4)=7 + ceil(18/4)=5 + ceil(33/4)=9 + ceil(100/4)=25 = 46.
        // fri: dimension 15 folds [4,4,1]; the last is not opened, so 2 layers of
        // ceil(2^4*3/4)=12 => 24. Under the old domain-folding schedule this was
        // 36, i.e. one phantom layer.
        assert_eq!(est.trace_hashes_per_query, 46);
        assert_eq!(est.fri_hashes_per_query, 24);
        assert_eq!(est.hashes_per_query, 70);
        // Totals are internally consistent (recursion verifies 2 proofs).
        assert_eq!(est.total_linear_hashes, est.hashes_per_query * est.n_queries * 2);
        assert_eq!(est.total_cells, est.total_hashes * 700);
        assert!(est.n_queries > 0);
    }

    #[test]
    fn hash_family_rate_and_cells() {
        assert_eq!(HashFamily::Blake3 { cells_per_perm: 5000 }.sponge_rate(), 4);
        assert_eq!(HashFamily::Blake2 { cells_per_perm: 5000 }.sponge_rate(), 12);
        assert_eq!(HashFamily::Blake3 { cells_per_perm: 5000 }.cells_per_perm(), 5000);
        assert_eq!(HashFamily::Blake2 { cells_per_perm: 3200 }.cells_per_perm(), 3200);
    }

    #[test]
    fn switching_hash_changes_rate_everywhere() {
        // Same config, different hash: rate 12 (Blake2) halves/thirds the per-leaf
        // permutation counts vs rate 4 (Blake3), in BOTH trace stages and FRI.
        let b3 = estimate_linear_hash_cells(&CellModelParams::recursion(17, 2, 3, 144, b3(5000)));
        let b2 = estimate_linear_hash_cells(&CellModelParams::recursion(
            17,
            2,
            3,
            144,
            HashFamily::Blake2 { cells_per_perm: 5000 },
        ));
        // trace: rate4 = ceil(28/4)+ceil(18/4)+ceil(48/4)+ceil(144/4) = 7+5+12+36 = 60
        //        rate12= ceil(28/12)+ceil(18/12)+ceil(48/12)+ceil(144/12)=3+2+4+12 = 21
        assert_eq!(b3.trace_hashes_per_query, 60);
        assert_eq!(b2.trace_hashes_per_query, 21);
        // FRI per opened step: rate4 ceil(24/4)=6 vs rate12 ceil(24/12)=2.
        assert!(b3.fri_hashes_per_query > b2.fri_hashes_per_query);
        // Same n_queries (rate doesn't affect the security calc).
        assert_eq!(b3.n_queries, b2.n_queries);
    }

    #[test]
    fn mt_nodes_matches_reference() {
        // arity 4, height 16: 16 -> 4 -> 1 nodes = 16 + 4 + 1 = 21, times 4 = 84.
        assert_eq!(get_num_nodes_mt(16, 4), 84);
        // height 1 -> just the single node: 1 * 4 = 4.
        assert_eq!(get_num_nodes_mt(1, 4), 4);
    }

    #[test]
    fn prover_memory_positive_and_scales_with_n() {
        let small = estimate_linear_hash_cells(&CellModelParams::recursion(15, 6, 3, 100, b3(5000)));
        let big = estimate_linear_hash_cells(&CellModelParams::recursion(18, 6, 3, 100, b3(5000)));
        assert!(small.prover_memory_field_elems > 0);
        assert!(small.prover_memory_gb > 0.0);
        // Larger trace -> strictly more prover memory.
        assert!(big.prover_memory_field_elems > small.prover_memory_field_elems);
        // GB is consistent with the field-elem count.
        let expect_gb = (big.prover_memory_field_elems as f64 * 8.0) / (1024.0 * 1024.0 * 1024.0);
        assert!((big.prover_memory_gb - expect_gb).abs() < 1e-9);
    }

    #[test]
    fn recursion_stage2_defaults_to_third_of_stage1() {
        let est = estimate_linear_hash_cells(&CellModelParams::recursion(15, 6, 4, 99, b3(1)));
        let stage2 = est.trace_stages.iter().find(|s| s.name == "stage2").unwrap();
        assert_eq!(stage2.cols, 33, "stage2 should default to stage1/3");
    }

    #[test]
    fn grinding_reduces_queries() {
        // With 20 grinding bits the query phase only needs 128-20=108 bits, so
        // n_queries is strictly lower than the no-grinding count for the same params.
        let est = estimate_linear_hash_cells(&CellModelParams::recursion(15, 6, 4, 100, b3(700)));
        // Sanity: positive and well below the ~64 a 128-bit no-grind JBR run needs.
        assert!(est.n_queries > 0 && est.n_queries < 64, "got {}", est.n_queries);
    }

    #[test]
    fn grinding_bits_default_20_and_configurable() {
        let p = CellModelParams::recursion(15, 6, 4, 100, b3(1));
        assert_eq!(p.grinding_bits, 20, "default grinding must be 20");
        let p2 = CellModelParams { grinding_bits: 16, ..p };
        assert_eq!(p2.grinding_bits, 16);
    }
}
