//! Verifier-hash comparison: FRI vs STIR over the ZisK AIR set.
//!
//! This is an ANALYSIS TOOL, not part of the prover/setup pipeline. It estimates
//! the number of Poseidon permutations a verifier performs for each AIR under FRI
//! vs STIR low-degree testing, using the crate's security-parameter calculator
//! (`pil2_stark_setup::types::security`) to pick query counts at 128-bit security.
//!
//! Run it:
//!   cargo run -p pil2-stark-setup --example verifier_hash_comparison
//!
//! The hash-cost model lives here (not in the library) because nothing in the
//! real prove/setup path consumes it — its only consumer is this report.
//!
//! Counts the Poseidon permutations a verifier performs. Two hash modes:
//!   * Merkle-tree COMPRESSION (arity 4): one fixed-width permutation per
//!     internal node on an authentication path.
//!   * Sponge LINEAR hash (rate 12, capacity 4): hashing a leaf of `w` field
//!     elements costs `ceil(w / 12)` permutations (absorb-only).
//!
//! The verifier's work has three parts:
//!   1. First-phase commitment openings: per query the verifier opens one leaf in
//!      EACH committed-stage Merkle tree (Fixed, Stage1, Stage2, Q), verifies its
//!      path (compression) and sponge-hashes the opened row.
//!   2. Low-degree-test layer openings: FRI re-opens its query set at EVERY fold
//!      layer; STIR opens `t_i` leaves at each (shrinking) round.
//!   3. Grinding is proof-of-work; it costs the verifier a single hash, omitted.

use pil2_stark_setup::types::security::{
    get_optimal_fri_query_params, goldilocks_cube_field_size, try_get_optimal_stir_query_params, FRISecurityParams,
    StirQueryResult, StirSecurityParams,
};

/// Poseidon sponge mode used for the linear leaf hash.
const SPONGE_RATE: u64 = 12;

/// Per-stage commitment description for one AIR: the number of committed columns
/// in each Merkle tree the verifier opens during the first phase.
#[derive(Debug, Clone)]
pub struct AirCommitments {
    /// Columns per committed-stage tree (e.g. [Fixed, Stage1, Stage2, Q]).
    pub stage_widths: Vec<u64>,
}

/// Verifier Poseidon-permutation count split into the two distinct hash kinds:
/// the linear (leaf) hash that absorbs committed columns / opened values, and
/// the internal-node hashes walked along Merkle authentication paths.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct VerifierHashBreakdown {
    /// Sponge permutations for the linear/leaf hash (column & folded-value absorption).
    pub linear_hash_perms: f64,
    /// Permutations for internal Merkle-path nodes (one per level walked).
    pub internal_node_perms: f64,
}

impl VerifierHashBreakdown {
    /// Total verifier permutations (linear + internal).
    pub fn total(&self) -> f64 {
        self.linear_hash_perms + self.internal_node_perms
    }
}

impl std::ops::AddAssign for VerifierHashBreakdown {
    fn add_assign(&mut self, o: Self) {
        self.linear_hash_perms += o.linear_hash_perms;
        self.internal_node_perms += o.internal_node_perms;
    }
}

impl std::ops::Mul<f64> for VerifierHashBreakdown {
    type Output = Self;
    fn mul(self, s: f64) -> Self {
        Self { linear_hash_perms: self.linear_hash_perms * s, internal_node_perms: self.internal_node_perms * s }
    }
}

/// Verifier permutations split by PHASE as well as kind:
///   - `pre_fri`: first-phase commitment opening (open every committed-stage tree
///     once per query) — happens BEFORE any FRI folding.
///   - `folding`: the FRI/STIR low-degree-test layers/rounds (the folding phase).
///   - `transcript`: the once-per-proof Fiat-Shamir sponge (phase-agnostic; all
///     linear, charged separately so it doesn't distort the pre/fold split).
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct PhasedHashBreakdown {
    pub pre_fri: VerifierHashBreakdown,
    pub folding: VerifierHashBreakdown,
    pub transcript: VerifierHashBreakdown,
}

impl PhasedHashBreakdown {
    /// Total verifier permutations across all phases and kinds.
    pub fn total(&self) -> f64 {
        self.pre_fri.total() + self.folding.total() + self.transcript.total()
    }
}

impl std::ops::AddAssign for PhasedHashBreakdown {
    fn add_assign(&mut self, o: Self) {
        self.pre_fri += o.pre_fri;
        self.folding += o.folding;
        self.transcript += o.transcript;
    }
}

/// Sponge permutations to linear-hash `width` field elements at rate 12.
fn sponge_perms(width: u64) -> f64 {
    if width == 0 {
        0.0
    } else {
        (width as f64 / SPONGE_RATE as f64).ceil()
    }
}

/// Fiat-Shamir transcript permutations the verifier runs ONCE per proof (not per
/// query). Absorbs every Merkle root (4 base elements each — the `n_stages` stage
/// roots + const tree root + Q-stage root + one per FRI step), the opened
/// evaluation vector (`3 * n_evals` base elements), and the final FRI polynomial
/// (`2^last_step_bits` degree-3 coefficients). IDENTICAL for FRI and STIR.
fn transcript_sponge_perms(n_stages: u64, n_fri_step_roots: u64, n_evals: u64, final_pol_size: u64) -> f64 {
    let n_roots = n_stages + 2 + n_fri_step_roots;
    let root_elems = n_roots * 4;
    let eval_elems = n_evals * 3; // degree-3 extension
    let final_pol_elems = final_pol_size * 3;
    let absorbed = (root_elems + eval_elems + final_pol_elems) as f64;
    (absorbed / SPONGE_RATE as f64).ceil()
}

/// Poseidon permutations to walk ONE Merkle authentication path over `domain_size`
/// leaves at the given arity, minus the `last_level_verification` top levels.
/// ONE permutation per level — `ceil(log_arity(domain_size)) - llv` — matching
/// `MerkleTreeGL::getMerkleProofLength` / `calculateRootFromProof` (one
/// `permuteTrunc` per level). NOT multiplied by `(arity - 1)`.
fn verifier_path_perms(tree_arity: u64, domain_size: f64, last_level_verification: u64) -> f64 {
    let levels = (domain_size.log2() / (tree_arity as f64).log2()).ceil();
    (levels - last_level_verification as f64).max(0.0)
}

/// First-phase verifier hashes PER QUERY: open one leaf in EACH committed-stage
/// tree (stage1, stage2, …, Q, and the constant/fixed tree). Per tree it
/// sponge-hashes the opened row (rate 12) and walks one authentication path.
fn first_phase_hashes_per_query(
    commit: &AirCommitments,
    tree_arity: u64,
    ext_domain_size: f64,
    last_level_verification: u64,
) -> VerifierHashBreakdown {
    let path = verifier_path_perms(tree_arity, ext_domain_size, last_level_verification);
    let mut b = VerifierHashBreakdown::default();
    for &w in commit.stage_widths.iter().filter(|&&w| w > 0) {
        b.linear_hash_perms += sponge_perms(w);
        b.internal_node_perms += path;
    }
    b
}

/// One low-degree-test layer/round opening: walk one authentication path over a
/// committed FRI-step tree of `step_domain_bits` leaves and sponge-hash the
/// opened folded values (`fold` codeword values, each a degree-3 extension
/// element => `3 * fold` base elements).
fn ldt_open_hashes(
    tree_arity: u64,
    step_domain_bits: u64,
    fold: u64,
    last_level_verification: u64,
) -> VerifierHashBreakdown {
    let domain = 2f64.powi(step_domain_bits as i32);
    VerifierHashBreakdown {
        internal_node_perms: verifier_path_perms(tree_arity, domain, last_level_verification),
        linear_hash_perms: sponge_perms(3 * fold),
    }
}

/// Per-step fold factor from consecutive FRI-step domain sizes:
/// `fold_i = 2^(bits_i - bits_{i+1})`. The last step has no successor (fold 1).
fn step_fold(step_bits: &[u64], i: usize) -> u64 {
    if i + 1 < step_bits.len() {
        1u64 << step_bits[i].saturating_sub(step_bits[i + 1])
    } else {
        1
    }
}

/// Total verifier Poseidon permutations for a FRI proof of one AIR.
///
/// `fri_step_bits` is the per-step committed-tree domain in bits, exactly as
/// `starkInfo.starkStruct.steps` reports: `fri_step_bits[0]` is the extended
/// (first-phase) domain `nBitsExt`; each later entry is that FRI layer's own
/// (folded) tree size — NOT `ext / arity^i`.
fn fri_verifier_hashes(
    commit: &AirCommitments,
    fri_step_bits: &[u64],
    n_queries: u64,
    tree_arity: u64,
    last_level_verification: u64,
) -> PhasedHashBreakdown {
    let q = n_queries as f64;
    let ext_domain = 2f64.powi(*fri_step_bits.first().unwrap_or(&0) as i32);

    let mut out = PhasedHashBreakdown::default();
    out.pre_fri = first_phase_hashes_per_query(commit, tree_arity, ext_domain, last_level_verification) * q;
    for i in 1..fri_step_bits.len() {
        let fold = step_fold(fri_step_bits, i - 1);
        out.folding += ldt_open_hashes(tree_arity, fri_step_bits[i], fold, last_level_verification) * q;
    }
    out
}

/// Total verifier Poseidon permutations for a STIR proof of one AIR.
///
/// STIR opens `t_i` leaves at each round; the round domain (path length) shrinks
/// each round, and `t_i` shrinks as the rate improves. The first-phase commitment
/// is opened `t_0` times (round 0's query count).
fn stir_verifier_hashes(
    commit: &AirCommitments,
    n_bits: u64,
    blowup_bits: u64,
    schedule: &StirQueryResult,
    tree_arity: u64,
    last_level_verification: u64,
) -> PhasedHashBreakdown {
    let ext_domain = 2f64.powi((n_bits + blowup_bits) as i32);
    let t0 = schedule.rounds.first().map(|r| r.repetitions).unwrap_or(0) as f64;

    let mut out = PhasedHashBreakdown::default();
    out.pre_fri = first_phase_hashes_per_query(commit, tree_arity, ext_domain, last_level_verification) * t0;
    for r in &schedule.rounds {
        let layer_domain = r.dimension as f64 / r.rate;
        let step_bits = layer_domain.log2().ceil() as u64;
        out.folding += ldt_open_hashes(tree_arity, step_bits, r.fold, last_level_verification) * r.repetitions as f64;
    }
    out
}

/// Print one AIR row: per scheme, the PRE-FRI (commitment-open) and FOLDING phase
/// subtotals and the grand total.
#[allow(clippy::too_many_arguments)]
fn print_hash_row(
    name: &str,
    n_bits: u64,
    fri: &PhasedHashBreakdown,
    has_stir: bool,
    stir: &PhasedHashBreakdown,
    ratio: f64,
    winner: &str,
) {
    if has_stir {
        println!(
            "{:<20} {:>4} | {:>8.0} {:>8.0} {:>8.0} | {:>8.0} {:>8.0} {:>8.0} | {:>5.2} {}",
            name,
            n_bits,
            fri.pre_fri.total(),
            fri.folding.total(),
            fri.total(),
            stir.pre_fri.total(),
            stir.folding.total(),
            stir.total(),
            ratio,
            winner,
        );
    } else {
        println!(
            "{:<20} {:>4} | {:>8.0} {:>8.0} {:>8.0} | {:>26} | {:>5} {}",
            name,
            n_bits,
            fri.pre_fri.total(),
            fri.folding.total(),
            fri.total(),
            "(no STIR schedule)",
            "-",
            winner,
        );
    }
}

fn main() {
    // (name, n_bits, blowup_bits, [fixed, stage1, stage2, q], n_evals, opening_points)
    let airs: &[(&str, u64, u64, [u64; 4], u64, u64)] = &[
        ("Main", 22, 0, [3, 38, 24, 6], 61, 3),
        ("Rom", 22, 0, [1, 1, 9, 6], 21, 3),
        ("Mem", 22, 0, [2, 13, 9, 6], 29, 3),
        ("InputData", 21, 0, [2, 9, 14, 6], 27, 3),
        ("RomData", 21, 0, [1, 5, 3, 6], 14, 3),
        ("MemAlign", 21, 0, [2, 29, 18, 6], 63, 3),
        ("MemAlignByte", 22, 0, [1, 16, 12, 6], 25, 3),
        ("MemAlignReadByte", 22, 0, [1, 10, 9, 6], 18, 3),
        ("MemAlignWriteByte", 22, 0, [1, 14, 12, 6], 23, 3),
        ("Arith", 21, 0, [1, 44, 46, 6], 65, 3),
        ("Binary", 22, 0, [1, 39, 15, 6], 49, 3),
        ("BinaryAdd", 22, 0, [1, 10, 9, 6], 18, 3),
        ("BinaryExtension", 22, 0, [1, 29, 18, 6], 40, 3),
        ("Add256", 20, 0, [1, 47, 51, 6], 69, 3),
        ("ArithEq", 20, 0, [2, 45, 39, 6], 473, 36),
        ("ArithEq384", 20, 0, [2, 35, 39, 6], 539, 54),
        ("Keccakf", 17, 0, [2, 2137, 880, 6], 4066, 26),
        ("Sha256f", 18, 0, [2, 102, 14, 6], 1266, 87),
        ("Poseidon", 17, 1, [2, 84, 143, 12], 377, 17),
        ("Blake2br", 18, 0, [3, 159, 44, 6], 789, 29),
        ("Dma", 21, 0, [1, 34, 21, 6], 46, 3),
        ("DmaMemCpy", 21, 0, [1, 22, 18, 6], 33, 3),
        ("DmaInputCpy", 21, 0, [1, 16, 18, 6], 27, 3),
        ("Dma64Aligned", 21, 0, [1, 35, 36, 6], 62, 3),
        ("Dma64AlignedInputCpy", 21, 0, [1, 25, 27, 6], 44, 3),
        ("Dma64AlignedMemSet", 21, 0, [1, 14, 15, 6], 30, 3),
        ("Dma64AlignedMem", 21, 0, [1, 26, 18, 6], 46, 3),
        ("Dma64AlignedMemCpy", 21, 0, [1, 31, 30, 6], 52, 3),
        ("DmaUnaligned", 21, 0, [1, 24, 12, 6], 52, 3),
        ("DmaPrePost", 21, 0, [1, 66, 32, 6], 83, 3),
        ("DmaPrePostMemCpy", 21, 0, [1, 55, 30, 6], 70, 3),
        ("DmaPrePostInputCpy", 21, 0, [1, 32, 21, 6], 44, 3),
        ("VirtualTableZisk0", 21, 0, [88, 23, 36, 6], 127, 3),
        ("VirtualTableZisk1", 21, 0, [73, 8, 15, 6], 90, 3),
        // Recursive circuit: N=17, blowup 3 (rate 2^-3), stages
        // Fixed=49, Stage1=62, Stage2=18, Stage3=StageQ=21.
        ("Recursive", 17, 3, [49, 62, 18, 21], 150, 3),
    ];

    let tree_arity = 4u64;
    let max_grinding_bits = 22u64;
    let target = 128u64;
    let stopping_bits = 6u64; // fold down to 2^6, per STIR paper §6.2
    let llv = 2u64; // verifier truncates top `llv` Merkle levels (matches starkStruct.lastLevelVerification)

    println!();
    println!("ZisK verifier Poseidon hashes @ {target} bits, MT arity {tree_arity}, sponge rate {SPONGE_RATE}");
    println!(
        "{:<20} {:>4} | {:>8} {:>8} {:>8} | {:>8} {:>8} {:>8} | {:>5} {}",
        "AIR", "bits", "FRI:pre", "fold", "total", "STIR:pre", "fold", "total", "ratio", "win",
    );
    println!("{}", "-".repeat(92));

    let (mut fri_sum, mut stir_sum) = (PhasedHashBreakdown::default(), PhasedHashBreakdown::default());
    for &(name, n_bits, blowup_bits, widths, n_evals, opening_points) in airs {
        let commit = AirCommitments { stage_widths: widths.to_vec() };
        let n_functions = n_evals.max(1);
        // `blowup_bits` is the LDT rate exponent (rate = 2^-blowup). blowupFactor 1
        // (blowup_bits=0) => extended domain 2x trace => rate 1/2 => 1 blowup bit.
        let blow = blowup_bits.max(1);
        let rate = 1.0 / (1u64 << blow) as f64;
        let ext_bits = n_bits + blow;

        const K: u64 = 4;
        let k_bits = K.trailing_zeros() as u64; // log2(K) = 2
        let n_layers = ((n_bits.saturating_sub(stopping_bits)).div_ceil(k_bits)).max(1);
        let fri_folds: Vec<u64> = vec![K; n_layers as usize];
        // Step-tree domains in bits, mirroring starkStruct.steps: step 0 is the
        // extended domain (nBitsExt); each fold-by-K step drops k_bits bits.
        let fri_step_bits: Vec<u64> =
            std::iter::once(ext_bits).chain((0..n_layers).map(|i| ext_bits - k_bits * (i + 1))).collect();
        let fri = FRISecurityParams {
            field_size: goldilocks_cube_field_size(),
            dimension: 1u64 << n_bits,
            rate,
            n_opening_points: opening_points,
            n_functions,
            folding_factors: fri_folds.clone(),
            max_grinding_bits,
            use_max_grinding_bits: true,
            tree_arity,
            target_security_bits: target,
        };
        let fri_q = get_optimal_fri_query_params("JBR", &fri).n_queries;
        let mut fri_hash = fri_verifier_hashes(&commit, &fri_step_bits, fri_q, tree_arity, llv);
        let n_stages = 2u64; // base ZisK AIRs: stage1 + stage2
        let final_pol = 1u64 << stopping_bits;
        fri_hash.transcript.linear_hash_perms = transcript_sponge_perms(n_stages, n_layers, n_evals, final_pol);

        let stir = StirSecurityParams {
            field_size: goldilocks_cube_field_size(),
            dimension: 1u64 << n_bits,
            rate,
            n_opening_points: opening_points,
            n_functions,
            max_grinding_bits,
            use_max_grinding_bits: true,
            tree_arity,
            target_security_bits: target,
            fold_candidates: vec![K], // pin to the SAME fold factor as FRI
            stopping_degree: 1u64 << stopping_bits,
            max_rounds: 24,
        };

        match try_get_optimal_stir_query_params("JBR", &stir) {
            Some(s) => {
                let mut stir_hash = stir_verifier_hashes(&commit, n_bits, blow, &s, tree_arity, llv);
                stir_hash.transcript.linear_hash_perms =
                    transcript_sponge_perms(n_stages, s.n_rounds as u64, n_evals, final_pol);
                let ratio = fri_hash.total() / stir_hash.total();
                let winner = if stir_hash.total() < fri_hash.total() { "STIR" } else { "FRI" };
                fri_sum += fri_hash;
                stir_sum += stir_hash;
                print_hash_row(name, n_bits, &fri_hash, true, &stir_hash, ratio, winner);
            }
            None => {
                fri_sum += fri_hash;
                print_hash_row(name, n_bits, &fri_hash, false, &PhasedHashBreakdown::default(), 0.0, "FRI");
            }
        }
    }
    println!("{}", "-".repeat(92));
    println!(
        "TOTAL  FRI: preFRI(lin {:.0} int {:.0})  fold(lin {:.0} int {:.0})  ts {:.0} = {:.0}",
        fri_sum.pre_fri.linear_hash_perms,
        fri_sum.pre_fri.internal_node_perms,
        fri_sum.folding.linear_hash_perms,
        fri_sum.folding.internal_node_perms,
        fri_sum.transcript.total(),
        fri_sum.total(),
    );
    println!(
        "       STIR: preFRI(lin {:.0} int {:.0})  fold(lin {:.0} int {:.0})  ts {:.0} = {:.0}  | overall_ratio={:.2}x",
        stir_sum.pre_fri.linear_hash_perms,
        stir_sum.pre_fri.internal_node_perms,
        stir_sum.folding.linear_hash_perms,
        stir_sum.folding.internal_node_perms,
        stir_sum.transcript.total(),
        stir_sum.total(),
        fri_sum.total() / stir_sum.total(),
    );
}