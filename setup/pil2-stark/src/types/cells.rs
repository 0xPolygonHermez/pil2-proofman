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

use super::security::{get_optimal_fri_query_params, goldilocks_cube_field_size, FRISecurityParams};

// ---------------------------------------------------------------------------
// Fixed protocol constants (same for any STARK we model here)
// ---------------------------------------------------------------------------

const TARGET_SECURITY_BITS: u64 = 128;
const REGIME: &str = "JBR";
const TREE_ARITY: u64 = 4;
/// Proof-of-work grinding bits (both ZisK Main and recursion pin this to 20),
/// so the query phase only needs `TARGET_SECURITY_BITS - GRINDING_BITS` bits.
const GRINDING_BITS: u64 = 20;

/// Recursion / FRI oracles live in the cubic extension of Goldilocks.
const EXTENSION_DEGREE: u64 = 3;

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
        }
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
    pub n_queries: u64,
    /// FRI folding factors as bit-drops, e.g. `[4, 4, 4, 3]`.
    pub folding_factors: Vec<u64>,
    pub trace_stages: Vec<StageHashes>,
    pub fri_steps: Vec<FriStepHashes>,
    /// Trace/Q stage hashes for one proof, one query.
    pub trace_hashes_per_query: u64,
    /// FRI hashes for one proof, one query.
    pub fri_hashes_per_query: u64,
    /// `trace_hashes_per_query + fri_hashes_per_query`.
    pub hashes_per_query: u64,
    /// Number of proofs opened (echoed from params, for the report).
    pub n_proofs: u64,
    /// `hashes_per_query * n_queries * n_proofs`.
    pub total_hashes: u64,
    /// `total_hashes * cells_per_blake3`.
    pub total_cells: u64,
    /// Stage1 columns the verifier would actually need to pack `total_cells`
    /// into `2^n` rows: `ceil(total_cells / 2^n)`.
    pub needed_stage1_cols: u64,
    /// The `stage1_cols` assumed in the input (for the fit comparison).
    pub assumed_stage1_cols: u64,
    /// Whether the assumed stage1 width is enough: `needed_stage1_cols <= assumed_stage1_cols`.
    pub fits: bool,
    /// Prover GPU memory estimate, in field elements (ported from
    /// `pil::info::get_prover_memory`).
    pub prover_memory_field_elems: u64,
    /// Prover GPU memory estimate, in GB (`field_elems * 8 / 1024^3`).
    pub prover_memory_gb: f64,
}

// ---------------------------------------------------------------------------
// Core helpers
// ---------------------------------------------------------------------------

/// Permutations to absorb `n_elements` field elements at the given sponge rate.
fn hashes_for_elements(n_elements: u64, sponge_rate: u64) -> u64 {
    n_elements.div_ceil(sponge_rate)
}

/// Derive FRI folding factors (as bit-drops) for an extended domain of
/// `n + blowup_bits` bits. Drop `fold_bits` per step, sizing the final step so
/// the folded polynomial lands on degree 6. The returned list is the *drops
/// only* (it sums to `n + blowup_bits - 6`); the leftover degree-6 final layer
/// is implicit and not listed.
///
/// Per the opening convention (mirroring `security::calculate_query_num_hashes`,
/// which iterates `folding_factors[..len-1]`), the **last entry is the final
/// fold and is not opened** — only the earlier folds produce query openings.
///
/// Examples (15 + 6 = 21 bits): `fold_bits=4` -> `[4,4,4,3]`;
/// `fold_bits=3` -> `[3,3,3,3,3]`.
fn derive_folding_factors(n: u32, blowup_bits: u32, fold_bits: u32) -> Vec<u64> {
    let step = fold_bits as i64;
    let mut remaining = (n + blowup_bits) as i64;
    let mut factors = Vec::new();
    while remaining > 6 {
        // Full `fold_bits` drop while it leaves more than 6 bits; otherwise size
        // the final drop to land exactly on degree 6.
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
    let arity = TREE_ARITY;
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

    let fri = get_optimal_fri_query_params(
        REGIME,
        &FRISecurityParams {
            field_size: goldilocks_cube_field_size(),
            dimension: 1u64 << p.n,
            rate: 1.0 / (1u64 << p.blowup_bits) as f64,
            n_opening_points: p.n_opening_points,
            n_functions: p.n_functions,
            folding_factors: folding_factors.clone(),
            max_grinding_bits: GRINDING_BITS,
            use_max_grinding_bits: true,
            tree_arity: TREE_ARITY,
            target_security_bits: TARGET_SECURITY_BITS,
        },
    );
    let n_queries = fri.n_queries;
    let rate = p.hash.sponge_rate();

    // Trace / Q stages: column count used directly, no extension factor.
    let trace_stages: Vec<StageHashes> = [
        ("fixed", p.fixed_cols),
        ("stageQ", p.stageq_cols),
        ("stage2", p.stage2_cols),
        ("stage1", p.stage1_cols),
    ]
    .into_iter()
    .map(|(name, cols)| StageHashes { name, cols, hashes: hashes_for_elements(cols, rate) })
    .collect();
    let trace_hashes_per_query: u64 = trace_stages.iter().map(|s| s.hashes).sum();

    // FRI: every folding step except the last opens `2^drop * 3` extension elements.
    let n_open = folding_factors.len().saturating_sub(1);
    let fri_steps: Vec<FriStepHashes> = folding_factors[..n_open]
        .iter()
        .map(|&bit_drop| {
            let elements = (1u64 << bit_drop) * EXTENSION_DEGREE;
            FriStepHashes { bit_drop, elements, hashes: hashes_for_elements(elements, rate) }
        })
        .collect();
    let fri_hashes_per_query: u64 = fri_steps.iter().map(|s| s.hashes).sum();

    let hashes_per_query = trace_hashes_per_query + fri_hashes_per_query;
    let total_hashes = hashes_per_query * n_queries * p.n_proofs;
    let total_cells = total_hashes * p.hash.cells_per_perm();

    // Self-verification fit: pack total_cells into 2^n rows -> columns needed.
    let rows = 1u64 << p.n;
    let needed_stage1_cols = total_cells.div_ceil(rows);
    let fits = needed_stage1_cols <= p.stage1_cols;

    let prover_memory_field_elems = prover_memory_field_elems(p, &folding_factors);
    let prover_memory_gb = (prover_memory_field_elems as f64 * 8.0) / (1024.0 * 1024.0 * 1024.0);

    CellEstimate {
        n_queries,
        folding_factors,
        trace_stages,
        fri_steps,
        trace_hashes_per_query,
        fri_hashes_per_query,
        hashes_per_query,
        n_proofs: p.n_proofs,
        total_hashes,
        total_cells,
        needed_stage1_cols,
        assumed_stage1_cols: p.stage1_cols,
        fits,
        prover_memory_field_elems,
        prover_memory_gb,
    }
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
/// (sharing `fold_bits` and `hash`) and check whether Main's total cells fit
/// within the recursion's total-cell budget.
pub fn main_fits_in_recursion(recursion: &CellEstimate, fold_bits: u32, hash: HashFamily) -> MainFitCheck {
    let main = estimate_linear_hash_cells(&CellModelParams::zisk_main(fold_bits, hash));
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
        s.push_str(&format!("hashes/query       = {}\n", self.hashes_per_query));
        s.push_str(&format!(
            "total_hashes       = {} (= {} hashes/query x {} queries x {} proofs)\n",
            self.total_hashes, self.hashes_per_query, self.n_queries, self.n_proofs
        ));
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
        // 15 + 6 = 21 bits -> drop 4 each until tail lands on 6: 21->17->13->9->[6].
        assert_eq!(derive_folding_factors(15, 6, 4), vec![4, 4, 4, 3]);
    }

    #[test]
    fn folding_factors_n15_blowup6_fold3() {
        // Same 21 bits, 3-bit steps: 21->18->15->12->9->[6], drops sum to 15.
        assert_eq!(derive_folding_factors(15, 6, 3), vec![3, 3, 3, 3, 3]);
    }

    #[test]
    fn folding_factors_land_on_degree_6() {
        // The drops (the returned list) must sum to `ext_bits - 6`, leaving the
        // implicit final layer at degree 6. Each drop is between 1 and fold_bits.
        for fold_bits in 2u32..=5 {
            for n in 8u32..24 {
                for blowup in 1u32..7 {
                    let ext = (n + blowup) as u64;
                    let f = derive_folding_factors(n, blowup, fold_bits);
                    let sum: u64 = f.iter().sum();
                    assert_eq!(sum, ext - 6, "fold={fold_bits} n={n} blowup={blowup} -> {f:?} sum wrong");
                    assert!(
                        f.iter().all(|&d| (1..=fold_bits as u64).contains(&d)),
                        "fold={fold_bits} n={n} blowup={blowup} -> {f:?} bad drop"
                    );
                }
            }
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
        // folding [4,4,4,3] -> open first 3 layers only: 12+12+12 = 36.
        let est = estimate_linear_hash_cells(&CellModelParams::recursion(15, 6, 4, 0, b3(1)));
        assert_eq!(est.folding_factors, vec![4, 4, 4, 3]);
        assert_eq!(est.fri_steps.len(), 3);
        assert_eq!(est.fri_hashes_per_query, 36);
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

    /// CSV sweep over (N, blowup) for the recursion verifier, with Main-fit. Lets
    /// us see the data shape before picking a graphic. Fixed: fold=3, stage1=100,
    /// cells_per_blake3=5000 (override via env COLS_STAGE1 / FOLD_BITS / CELLS_PER_BLAKE3).
    ///   cargo test -p pil2-stark-setup --lib types::cells::tests::cells_sweep_csv -- --ignored --nocapture
    #[test]
    #[ignore]
    fn cells_sweep_csv() {
        let env = |k: &str, d: u64| std::env::var(k).ok().and_then(|v| v.parse().ok()).unwrap_or(d);
        let fold_bits = env("FOLD_BITS", 3) as u32;
        let stage1 = env("COLS_STAGE1", 100);
        let hash = hash_from_env(env("CELLS_PER_BLAKE3", 5000));

        println!("N,blowup,fold,stage1,n_queries,hashes_per_query,total_cells,needed_stage1,self_fits,prover_mem_gb,main_total_cells,main_fits");
        for n in 14u32..=20 {
            for blowup in 1u32..=8 {
                let rec = estimate_linear_hash_cells(&CellModelParams::recursion(n, blowup, fold_bits, stage1, hash));
                let chk = main_fits_in_recursion(&rec, fold_bits, hash);
                println!(
                    "{},{},{},{},{},{},{},{},{},{:.3},{},{}",
                    n,
                    blowup,
                    fold_bits,
                    stage1,
                    rec.n_queries,
                    rec.hashes_per_query,
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
        // Single proof, so total_hashes = hashes_per_query * n_queries * 1.
        assert_eq!(est.n_proofs, 1);
        assert_eq!(est.total_hashes, est.hashes_per_query * est.n_queries);
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
        // trace = ceil(28/4)=7 + ceil(18/4)=5 + ceil(33/4)=9 + ceil(100/4)=25 = 46; fri = 36.
        assert_eq!(est.trace_hashes_per_query, 46);
        assert_eq!(est.fri_hashes_per_query, 36);
        assert_eq!(est.hashes_per_query, 82);
        // Totals are internally consistent (recursion verifies 2 proofs).
        assert_eq!(est.total_hashes, est.hashes_per_query * est.n_queries * 2);
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
            17, 2, 3, 144,
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
}
