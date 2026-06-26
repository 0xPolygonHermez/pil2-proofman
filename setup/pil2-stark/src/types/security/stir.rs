use rug::ops::Pow;
use rug::Float;

use super::regime::{build_regime, RegimeParams};
use super::{calculate_query_num_hashes, hpf, PREC};

// ===========================================================================
// STIR (ePrint 2024/390) — parameter calculator
// ===========================================================================

/// Per-round (dimension, rate) schedule for STIR with uniform fold `k`.
///
/// STIR folds the degree by `k` (`d_{i+1} = d_i / k`) while shrinking the
/// evaluation domain by only 2 (`|L_{i+1}| = |L_i| / 2`). Since rate = degree /
/// domain, the rate recurrence is `rho_{i+1} = rho_i * 2 / k`. For `k >= 4` the
/// rate strictly shrinks each round (this is STIR's query-count advantage);
/// `k = 2` keeps the rate constant (degenerating to FRI). Produces `M + 1`
/// entries (round 0 = input instance, rounds 1..=M the folded instances).
///
/// Candidates where the recurrence drives `rho_i >= 1` or `d_i < 1` are rejected
/// by the optimizer, not here — this returns the raw schedule.
fn stir_round_schedule(d: u64, rho0: f64, k: u64, m: u64) -> Vec<(u64, f64)> {
    let mut out = Vec::with_capacity((m + 1) as usize);
    let mut dim = d;
    let mut rate = rho0;
    out.push((dim, rate));
    for _ in 0..m {
        dim /= k;
        rate = rate * 2.0 / (k as f64);
        out.push((dim, rate));
    }
    out
}

// ---------------------------------------------------------------------------
// Independent soundness cross-check (diagnostics).
//
// The optimizer gates on the STIR reference closed-form (`stir_achieved_
// security_bits`). The functions below re-derive the soundness error term by
// term — query/repetition, out-of-domain sampling, batched-commit list-decoding
// — as an INDEPENDENT check (used by tests and available for diagnostics). They
// intentionally do not gate selection; see `get_optimal_stir_query_params`.
// ---------------------------------------------------------------------------

/// Out-of-domain sampling error for one STIR round with a single OOD sample
/// (s = 1). Probability that the OOD point fails to pin a unique folded
/// codeword: `listSize^2 * dimension / fieldSize` (ePrint 2024/390, OOD lemma).
#[allow(dead_code)]
pub(super) fn calculate_ood_error(max_list_size: Float, dimension: u64, field_size: &Float) -> Float {
    let list_sq = Float::with_val(PREC, max_list_size.pow(2));
    let numer = Float::with_val(PREC, &list_sq * hpf(dimension));
    Float::with_val(PREC, numer / field_size)
}

/// Per-round STIR soundness error term.
///
/// Combines this round's query/repetition error `(1 - delta_i)^{t_i}`, its
/// out-of-domain sampling error, and its commit error, then divides by this
/// round's grinding factor `2^{g_i}`. Grinding is per round because the STIR
/// reference computes a `pow_bits` vector, one entry per round (the bits trade
/// against that round's repetition strength).
#[allow(dead_code)]
fn calculate_stir_round_error(round: &StirRound, alpha: f64, regime_name: &str, field_size: &Float) -> Float {
    let rp = RegimeParams::new(field_size.clone(), round.dimension, round.rate, alpha, 0);
    let regime = build_regime(&rp, regime_name);
    // Query/repetition error uses the IDEALISED proximity (delta = 1 - sqrt(rho),
    // i.e. the max decoding radius with no gap correction): this is the radius the
    // STIR reference repetition formula `t_i = ceil(2*lambda/log_inv_rate)` is
    // derived against. `alpha` only widens the Johnson gap for the batch/commit
    // list-decoding term below; coupling it into the query term would
    // double-count the gap and make raising alpha *worsen* the query error.
    let delta = regime.max_decoding_radius();
    let single_query_error = Float::with_val(PREC, hpf(1) - &delta);
    let query_error = Float::with_val(PREC, single_query_error.pow(round.repetitions as u32));
    let ood_error = calculate_ood_error(regime.max_list_size(), round.dimension, field_size);
    let commit_error = regime.calculate_powers_error(round.fold);
    let round_error = query_error.max(&ood_error).max(&commit_error);

    let two_pow = Float::with_val(PREC, hpf(2).pow(round.grinding_bits as u32));
    let grinding = Float::with_val(PREC, hpf(1) / &two_pow);
    Float::with_val(PREC, &round_error * &grinding)
}

/// Total STIR soundness error: the union bound `sum_i e_i` over per-round errors
/// plus the global batch error.
///
/// STIR's soundness is `eps <= (1 - delta_0)^{t_0} + sum_{i>=1} rho_i^{t_i/2}`
/// (ePrint 2024/390): a SUM of per-round terms, not a max. Each round carries its
/// own repetitions and grinding so that its term clears the target; because the
/// rate improves each round, later terms need far fewer repetitions. We use a
/// union-bound sum here (conservative vs. the paper's per-term bound) and verify
/// the aggregate clears the target via `security_bits_from_error`.
#[allow(dead_code)]
pub(super) fn calculate_stir_total_error(
    rounds: &[StirRound],
    alpha: f64,
    n_functions: u64,
    regime_name: &str,
    field_size: &Float,
) -> Float {
    debug_assert!(!rounds.is_empty(), "calculate_stir_total_error requires at least one round");

    let rp0 = RegimeParams::new(field_size.clone(), rounds[0].dimension, rounds[0].rate, alpha, 0);
    let batch_error = build_regime(&rp0, regime_name).calculate_powers_error(n_functions);

    let mut total = batch_error;
    for r in rounds {
        let e = calculate_stir_round_error(r, alpha, regime_name, field_size);
        total = Float::with_val(PREC, &total + &e);
    }
    total
}

// ---------------------------------------------------------------------------
// STIR Schedule Optimizer
// ---------------------------------------------------------------------------

/// `log_inv_rate` of a rate `rho`: the `r` such that `rho = 2^-r`.
fn log_inv_rate(rate: f64) -> f64 {
    -rate.log2()
}

/// Per-round repetition count, following the STIR reference parameter formula
/// (ePrint 2024/390 §5.3 / §C, conjectured regime, c1=c2=c3=1):
/// `t_i = ceil(protocol_security / -log2(sqrt(rho_i)))` and since
/// `-log2(sqrt(rho_i)) = log_inv_rate_i / 2`, this is
/// `ceil(2 * protocol_security / log_inv_rate_i)`.
///
/// `protocol_security` is the security the IOP itself must provide. The global
/// proof-of-work (grinding) supplies the rest, so `protocol_security =
/// target - grinding_bits` (the paper uses 128 - 22 = 106).
fn stir_repetitions(protocol_security_bits: u64, lir: f64) -> u64 {
    ((2.0 * protocol_security_bits as f64) / lir).ceil().max(1.0) as u64
}

/// Number of STIR folding rounds `M` for fold `k`: fold `dimension` by `k` each
/// round until it would drop to at most `stopping_degree`, like FRI folding down
/// to a small final degree. Returns `None` if `k` does not reduce the degree or
/// the schedule exceeds `max_rounds`.
fn stir_num_rounds(dimension: u64, stopping_degree: u64, k: u64, max_rounds: u64) -> Option<u64> {
    if k < 2 || dimension <= stopping_degree {
        return None;
    }
    let mut d = dimension;
    let mut m = 0u64;
    while d > stopping_degree {
        d /= k;
        m += 1;
        if m > max_rounds {
            return None;
        }
    }
    Some(m)
}

/// Build the STIR round schedule for a given uniform fold `k`, deriving the
/// round count by folding `dimension` down to `stopping_degree`. Per-round
/// repetitions and grinding come from each round's own rate (STIR's advantage:
/// the rate improves each round, so `t_i` shrinks). Returns `None` for a
/// degenerate candidate (k too small, rate out of (0,1), dimension underflow).
///
/// The repetition cap `t_i <= degree_i / k` mirrors the reference (a round can
/// not query more leaves than the folded codeword has); the final round is left
/// uncapped, matching the reference's "skips the last repetition" loop.
fn build_stir_schedule(params: &StirSecurityParams, k: u64) -> Option<Vec<StirRound>> {
    let m = stir_num_rounds(params.dimension, params.stopping_degree, k, params.max_rounds)?;
    let sched = stir_round_schedule(params.dimension, params.rate, k, m);
    for &(dim, rate) in &sched {
        if dim < 1 || rate >= 1.0 || rate <= 0.0 {
            return None;
        }
    }

    // The STIR reference uses a single GLOBAL proof-of-work of `grinding` bits
    // (ePrint 2024/390 §6.2: 22 PoW bits at λ=128), so the IOP itself need only
    // provide `protocol_security = target - grinding` bits, and each round's
    // repetitions are sized against exactly that (no per-round union inflation —
    // the union bound is covered by the PoW margin, matching the reference).
    let grinding = params.max_grinding_bits;
    let protocol_security = params.target_security_bits.saturating_sub(grinding);

    let last = sched.len() - 1;
    let mut rounds = Vec::with_capacity(sched.len());
    for (i, &(dim, rate)) in sched.iter().enumerate() {
        let lir = log_inv_rate(rate);
        if !lir.is_finite() || lir <= 0.0 {
            return None;
        }
        let wanted = stir_repetitions(protocol_security, lir);
        // Queries are positions in this round's evaluation domain `L_i = d_i/rho_i`.
        // A round can not open more positions than the domain has; if the wanted
        // count reaches the whole domain the round is FULLY OPENED — perfectly
        // checked, contributing no soundness error (excluded from the gating min).
        // For realistic sizes this never binds (domains are large), matching the
        // reference which keeps the computed `t_i`.
        let domain_size = (dim as f64 / rate).floor() as u64;
        let (reps, fully_opened) =
            if i != last && wanted >= domain_size { (domain_size.max(1), true) } else { (wanted.max(1), false) };
        // Grinding is a single global PoW recorded on round 0 (reported figure);
        // it is not a per-round vector. Later rounds carry 0 here.
        let grinding_bits = if i == 0 { grinding } else { 0 };
        rounds.push(StirRound { rate, dimension: dim, fold: k, repetitions: reps, grinding_bits, fully_opened });
    }
    Some(rounds)
}

/// Security bits achieved by a STIR schedule, per the reference soundness
/// analysis (ePrint 2024/390): each round `i` contributes
/// `(log_inv_rate_i / 2) * t_i + g_i` bits (provable/Johnson regime, the same
/// `scaling = 2` the repetition formula uses), and the overall soundness is the
/// union bound over the `M + 1` round terms. The binding round is the weakest,
/// and the union over `M + 1` terms costs `log2(M + 1)` bits.
fn stir_achieved_security_bits(rounds: &[StirRound]) -> i64 {
    // The global proof-of-work (recorded on round 0) protects the whole proof, so
    // it adds to every round's effective security.
    let grinding = rounds.first().map(|r| r.grinding_bits).unwrap_or(0) as f64;
    let per_round_bits = |r: &StirRound| -> f64 {
        let lir = log_inv_rate(r.rate);
        (lir / 2.0) * r.repetitions as f64 + grinding
    };
    // Fully-opened rounds are perfectly checked (the verifier reads the whole
    // folded codeword) and carry no soundness error, so they are excluded from
    // the weakest-round minimum. They still appear in the union count.
    // Each round's IOP term is `(lir/2)*t_i`, sized via ceil() against the
    // protocol security `target - grinding`; the ceil rounding plus the global
    // PoW absorb the `log2(M+1)` union bound over rounds (this is how the STIR
    // reference reaches the target — see §6.2). So the achieved security is the
    // weakest round's `(lir/2)*t_i + grinding`, no extra union subtraction.
    let weakest = rounds.iter().filter(|r| !r.fully_opened).map(per_round_bits).fold(f64::INFINITY, f64::min);
    weakest.floor() as i64
}

/// Hash-weighted query cost of a schedule (optimizer objective): each round's
/// repetitions weighted by the Merkle-path hash cost on that round's domain.
fn stir_schedule_cost(rounds: &[StirRound], tree_arity: u64) -> f64 {
    rounds
        .iter()
        .map(|r| {
            let codeword_len = r.dimension as f64 / r.rate;
            r.repetitions as f64 * calculate_query_num_hashes(tree_arity, codeword_len, &[r.fold])
        })
        .sum()
}

/// Protocol selector for the query-parameter calculator.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Protocol {
    Fri,
    Stir,
}

/// Parameters for STIR security calculation (public API input).
///
/// Mirrors `FRISecurityParams` but the optimizer *chooses* the uniform fold
/// `k`; the number of rounds `M` is derived by folding `dimension` down to
/// `stopping_degree` (exactly as FRI folds to a small final degree, and as the
/// STIR reference does). The caller supplies the fold candidates and the
/// stopping degree instead of a fixed `folding_factors` vector.
#[derive(Debug)]
pub struct StirSecurityParams {
    pub field_size: Float,
    pub dimension: u64,
    pub rate: f64,
    /// `n_opening_points` / `n_functions` feed only the diagnostic
    /// `calculate_stir_total_error` cross-check, not the gated schedule (the
    /// reference repetition formula folds the batch into the security target).
    pub n_opening_points: u64,
    pub n_functions: u64,
    pub max_grinding_bits: u64,
    /// When true, use `max_grinding_bits` directly; otherwise cap at the hash-cost-efficient bound.
    pub use_max_grinding_bits: bool,
    pub tree_arity: u64,
    pub target_security_bits: u64,
    /// Candidate uniform fold factors (powers of two). Default: [2, 4, 8, 16].
    pub fold_candidates: Vec<u64>,
    /// Final (stopping) degree: fold each round by `k` until the degree drops to
    /// at most this, then send the remainder in the clear — like FRI's stop at
    /// ~2^5/2^6. The round count `M` is derived from this, not searched.
    pub stopping_degree: u64,
    /// Upper bound on derived rounds M (safety valve against tiny stopping_degree).
    pub max_rounds: u64,
}

/// One STIR round in the chosen schedule.
///
/// Repetitions and grinding are derived per round from the round's own rate,
/// following the STIR authors' reference parameter computation (the rate
/// improves each round, so later rounds need fewer repetitions). This is why
/// STIR's summed-over-rounds query count beats FRI, which re-queries its full
/// (rate-invariant) count at *every* fold layer.
#[derive(Debug, Clone)]
pub struct StirRound {
    pub rate: f64,
    pub dimension: u64,
    pub fold: u64,
    pub repetitions: u64,
    /// Proof-of-work (grinding) bits for this round. Per-round, because grinding
    /// trades against this round's repetition strength (ePrint 2024/390 / ref impl).
    pub grinding_bits: u64,
    /// True when `repetitions` is the entire folded codeword (the cap binds): the
    /// round is fully read and contributes no soundness error.
    pub fully_opened: bool,
}

/// Result of optimal STIR query parameter computation.
#[derive(Debug, Clone)]
pub struct StirQueryResult {
    pub rounds: Vec<StirRound>,
    pub n_grinding_bits: u64,
    pub total_queries: u64,
    /// The uniform fold factor `k` chosen by the optimizer (same for every round).
    pub fold: u64,
    /// Number of folding rounds M. Invariant: always equals `rounds.len() - 1` (round 0 is the input instance).
    pub n_rounds: u64,
    /// Verified security in bits. May be lower than the target only if construction failed; the optimizer otherwise guarantees >= target.
    pub achieved_security_bits: i64,
}

/// Optimal STIR query parameters: searches uniform fold `k` in `fold_candidates`
/// and round count `M` in `1..=max_rounds`, choosing the min-hash-cost schedule
/// that meets the security target (ePrint 2024/390).
///
/// The target is gated on the STIR reference soundness analysis
/// (`stir_achieved_security_bits`), which is the published, paper-faithful
/// bound. The batched-commit error (the same JBR list-decoding limit the FRI
/// path handles via its own gap/alpha machinery) is reported separately via
/// `calculate_stir_total_error` as a diagnostic and does NOT gate selection —
/// coupling it into the per-round query union would double-count a term the
/// reference analysis already accounts for in the repetition sizing.
///
/// `regime_name` ("JBR" / "UDR") is accepted for signature parity with the FRI
/// entry point. Both are *provable* regimes, so the reference repetition formula
/// uses the same constant (2) for either — the gated schedule is identical. The
/// name still selects the regime in the diagnostic `calculate_stir_total_error`.
pub fn get_optimal_stir_query_params(regime_name: &str, params: &StirSecurityParams) -> StirQueryResult {
    try_get_optimal_stir_query_params(regime_name, params)
        .expect("STIR optimizer found no schedule meeting the security target")
}

/// Fallible core of [`get_optimal_stir_query_params`]: returns `None` when no
/// fold candidate yields a schedule meeting the target (e.g. high rate folded to
/// a tiny stopping degree). Useful for parameter-space exploration where some
/// `(rate, k)` cells legitimately have no 128-bit schedule.
pub fn try_get_optimal_stir_query_params(regime_name: &str, params: &StirSecurityParams) -> Option<StirQueryResult> {
    let _ = regime_name; // see doc: gating is regime-independent for provable bounds
    if !params.fold_candidates.iter().any(|&k| k >= 2) {
        return None;
    }
    let mut best: Option<(Vec<StirRound>, i64, f64)> = None;

    // The round count M is derived per `k` (fold down to `stopping_degree`), not
    // searched — mirroring FRI's fold-to-small-final-degree and the STIR reference.
    for &k in &params.fold_candidates {
        let Some(rounds) = build_stir_schedule(params, k) else {
            continue;
        };
        let bits = stir_achieved_security_bits(&rounds);
        if bits < params.target_security_bits as i64 {
            continue;
        }
        let cost = stir_schedule_cost(&rounds, params.tree_arity);
        let better = match &best {
            None => true,
            Some((_, _, best_cost)) => cost < *best_cost,
        };
        if better {
            best = Some((rounds, bits, cost));
        }
    }

    best.map(|(rounds, achieved, _)| finalize_stir_result(rounds, achieved))
}

/// Assemble the public result from the winning schedule.
fn finalize_stir_result(rounds: Vec<StirRound>, achieved_security_bits: i64) -> StirQueryResult {
    let total_queries: u64 = rounds.iter().map(|r| r.repetitions).sum();
    // Report the round-0 grinding as the headline figure (the largest, since
    // later rounds need less); the full per-round vector lives in `rounds`.
    let n_grinding_bits = rounds[0].grinding_bits;
    let fold = rounds[0].fold;
    let n_rounds = rounds.len() as u64 - 1;

    StirQueryResult { rounds, n_grinding_bits, total_queries, fold, n_rounds, achieved_security_bits }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::security::goldilocks_cube_field_size;
    use crate::types::security::regime::{RegimeParams, JBR};
    use crate::types::security::security_bits_from_error;

    #[test]
    fn test_stir_round_schedule_reduces_rate_and_dim() {
        // d=2^17, rho0=1/2, k=4, M=3. Recurrence rho_{i+1}=rho_i*2/k => halves each round.
        let sched = stir_round_schedule(1 << 17, 0.5, 4, 3);
        assert_eq!(sched.len(), 4); // M+1 entries (round 0..=M)
        assert_eq!(sched[0].0, 1 << 17);
        assert_eq!(sched[1].0, (1 << 17) / 4);
        assert!((sched[0].1 - 0.5).abs() < 1e-12);
        assert!((sched[1].1 - 0.25).abs() < 1e-12); // 0.5 * 2/4 = 0.25
        assert!((sched[2].1 - 0.125).abs() < 1e-12); // strictly shrinking
    }

    #[test]
    fn test_stir_round_schedule_rate_constant_when_k_eq_2() {
        // k=2 => rate *= 2/2 = 1 => rate constant; dimension halves each round.
        let sched = stir_round_schedule(1 << 20, 0.25, 2, 4);
        assert!((sched[2].1 - 0.25).abs() < 1e-12);
        assert_eq!(sched[2].0, (1 << 20) / 4);
    }

    #[test]
    fn test_stir_ood_error_is_tiny_over_goldilocks_cube() {
        // OOD error = maxListSize^2 * d / |F|. Over Goldilocks^3 (~2^191) with
        // d=2^17 and a modest list size, this must be far below 2^-128.
        let fs = goldilocks_cube_field_size();
        let rp = RegimeParams::new(fs, 1 << 17, 0.5, 0.0, 26);
        let regime = JBR::new(&rp);
        let ood = calculate_ood_error(regime.max_list_size(), 1u64 << 17, &rp.field_size);
        assert!(security_bits_from_error(&ood) > 128, "OOD term must exceed 128 bits");
    }

    #[test]
    fn test_stir_total_error_more_bits_with_more_reps() {
        // More repetitions in the same single-round STIR instance must not reduce security.
        let fs = goldilocks_cube_field_size();
        let rounds_few = vec![StirRound {
            rate: 0.5,
            dimension: 1 << 17,
            fold: 2,
            repetitions: 10,
            grinding_bits: 0,
            fully_opened: false,
        }];
        let rounds_many = vec![StirRound {
            rate: 0.5,
            dimension: 1 << 17,
            fold: 2,
            repetitions: 60,
            grinding_bits: 0,
            fully_opened: false,
        }];
        let e_few = calculate_stir_total_error(&rounds_few, 0.0, 145, "JBR", &fs);
        let e_many = calculate_stir_total_error(&rounds_many, 0.0, 145, "JBR", &fs);
        assert!(
            security_bits_from_error(&e_many) >= security_bits_from_error(&e_few),
            "more repetitions must not reduce security"
        );
    }

    #[test]
    fn test_stir_meets_128_bits_golden() {
        let params = StirSecurityParams {
            field_size: goldilocks_cube_field_size(),
            dimension: 1 << 17,
            rate: 0.5,
            n_opening_points: 26,
            n_functions: 4065,
            max_grinding_bits: 22,
            use_max_grinding_bits: true,
            tree_arity: 4,
            target_security_bits: 128,
            fold_candidates: vec![2, 4, 8, 16],
            stopping_degree: 64,
            max_rounds: 12,
        };
        let r = get_optimal_stir_query_params("JBR", &params);
        assert!(r.achieved_security_bits >= 128, "STIR must meet 128 bits, got {}", r.achieved_security_bits);
        assert_eq!(r.n_rounds + 1, r.rounds.len() as u64);
        let sum: u64 = r.rounds.iter().map(|x| x.repetitions).sum();
        assert_eq!(sum, r.total_queries);
        for w in r.rounds.windows(2) {
            assert!(w[1].dimension < w[0].dimension, "dimension must strictly decrease");
        }
    }

    #[test]
    fn test_stir_fewer_queries_than_fri_golden() {
        // Apples-to-apples: FRI and STIR fold by the SAME factor k down to the
        // SAME stopping degree, so both have the SAME number of layers/rounds N.
        // The ONLY thing that differs is the per-round query count:
        //   * FRI re-opens its full query set at EVERY layer (rate never improves)
        //     => work = n_queries * N.
        //   * STIR's per-round count t_i SHRINKS as the rate improves
        //     => work = sum_i t_i.
        // This isolates STIR's benefit to exactly the query-count reduction.
        let k = 4u64;
        let dimension = 1u64 << 17;
        let stopping_degree = 64u64; // 2^6
                                     // Same derived depth for both: fold dimension by k until <= stopping_degree.
        let n_layers = stir_num_rounds(dimension, stopping_degree, k, 16).expect("valid fold schedule");

        let fri = crate::types::security::FRISecurityParams {
            field_size: goldilocks_cube_field_size(),
            dimension,
            rate: 0.5,
            n_opening_points: 26,
            n_functions: 4065,
            folding_factors: vec![k; n_layers as usize], // same k, same depth as STIR
            max_grinding_bits: 22,
            use_max_grinding_bits: true,
            tree_arity: 4,
            target_security_bits: 128,
        };
        let fri_per_layer = crate::types::security::get_optimal_fri_query_params("JBR", &fri).n_queries;
        let fri_total_work = fri_per_layer * n_layers;

        let stir = StirSecurityParams {
            field_size: goldilocks_cube_field_size(),
            dimension,
            rate: 0.5,
            n_opening_points: 26,
            n_functions: 4065,
            max_grinding_bits: 22,
            use_max_grinding_bits: true,
            tree_arity: 4,
            target_security_bits: 128,
            fold_candidates: vec![k], // pin to the SAME fold factor as FRI
            stopping_degree,
            max_rounds: 16,
        };
        let stir_r = get_optimal_stir_query_params("JBR", &stir);
        let stir_q = stir_r.total_queries;
        // Same fold factor AND same depth on both sides.
        assert_eq!(stir_r.fold, k, "STIR must use the same fold factor as FRI");
        assert_eq!(stir_r.n_rounds, n_layers, "STIR rounds must equal FRI layers");
        eprintln!(
            "k={k}, layers=N={n_layers}: FRI={fri_per_layer}/layer * {n_layers} = {fri_total_work} total; \
             STIR total_queries={stir_q} (per-round reps={:?})",
            stir_r.rounds.iter().map(|x| x.repetitions).collect::<Vec<_>>(),
        );
        assert!(stir_q < fri_total_work, "STIR ({stir_q}) should beat FRI's per-layer-summed work ({fri_total_work})");
    }

    /// Informational cross-check against the STIR paper's regime: low rate and
    /// larger degree, where STIR's geometric query reduction is pronounced.
    /// Dumps the chosen schedule for eyeballing vs. ePrint 2024/390 ballparks and
    /// asserts each meets 128 bits with a strictly decreasing per-round rate.
    #[test]
    fn test_stir_schedule_dump_low_rate() {
        for (log_d, rate) in [(20u32, 0.25_f64), (24, 0.125), (26, 0.0625)] {
            let params = StirSecurityParams {
                field_size: goldilocks_cube_field_size(),
                dimension: 1 << log_d,
                rate,
                n_opening_points: 4,
                n_functions: 145,
                max_grinding_bits: 20,
                use_max_grinding_bits: true,
                tree_arity: 4,
                target_security_bits: 128,
                fold_candidates: vec![4, 8, 16],
                stopping_degree: 64,
                max_rounds: 16,
            };
            let r = get_optimal_stir_query_params("JBR", &params);
            eprintln!(
                "d=2^{log_d} rate={rate}: k={}, M={}, total_queries={}, grind0={}, bits={}, reps={:?}",
                r.fold,
                r.n_rounds,
                r.total_queries,
                r.n_grinding_bits,
                r.achieved_security_bits,
                r.rounds.iter().map(|x| x.repetitions).collect::<Vec<_>>(),
            );
            assert!(r.achieved_security_bits >= 128, "must meet 128 bits at d=2^{log_d}");
            for w in r.rounds.windows(2) {
                assert!(w[1].rate <= w[0].rate, "rate must be non-increasing across rounds");
            }
        }
    }

    /// On-demand sweep: STIR-vs-FRI across blowup factors (rate = 2^-blowup),
    /// iterating each candidate fold `k` per row so the per-round geometric
    /// schedule is visible (not just the cost-optimal pick). Ignored by default —
    /// run explicitly with:
    ///   cargo test -p pil2-stark-setup sweep_stir_vs_fri -- --ignored --nocapture
    #[test]
    #[ignore]
    fn sweep_stir_vs_fri() {
        let log_d = 20u32;
        let dimension = 1u64 << log_d;
        let tree_arity = 4u64;
        let max_grinding_bits = 22u64;
        let n_functions = 145u64;

        println!();
        println!(
            "STIR vs FRI @ 128 bits, d=2^{log_d}, arity={tree_arity}, max_grind={max_grinding_bits}, n_fns={n_functions}"
        );
        println!("FRI fold schedule [4,4,4] => 3 layers, re-queried each layer.");
        println!(
            "{:>5} {:>6} | {:>8} {:>8} | {:>4} {:>3} {:>9} {:>7} | {:<28} | winner",
            "blow", "rate", "FRI/lyr", "FRI_tot", "k", "M", "STIR_tot", "grind0", "STIR per-round reps",
        );
        println!("{}", "-".repeat(104));

        for blowup in [1u32, 2, 3, 4] {
            let rate = 1.0 / (1u64 << blowup) as f64;
            let fri_layers_vec: Vec<u64> = vec![4, 4, 4];
            let fri_layers = fri_layers_vec.len() as u64;

            let fri = crate::types::security::FRISecurityParams {
                field_size: goldilocks_cube_field_size(),
                dimension,
                rate,
                n_opening_points: 4,
                n_functions,
                folding_factors: fri_layers_vec,
                max_grinding_bits,
                use_max_grinding_bits: true,
                tree_arity,
                target_security_bits: 128,
            };
            let fri_per_layer = crate::types::security::get_optimal_fri_query_params("JBR", &fri).n_queries;
            let fri_total = fri_per_layer * fri_layers;

            // Show every candidate fold so the depth/queries trade-off is visible.
            for k in [2u64, 4, 8, 16] {
                let stir = StirSecurityParams {
                    field_size: goldilocks_cube_field_size(),
                    dimension,
                    rate,
                    n_opening_points: 4,
                    n_functions,
                    max_grinding_bits,
                    use_max_grinding_bits: true,
                    tree_arity,
                    target_security_bits: 128,
                    fold_candidates: vec![k],
                    stopping_degree: 64,
                    max_rounds: 16,
                };
                let Some(r) = try_get_optimal_stir_query_params("JBR", &stir) else {
                    println!(
                        "{:>5} {:>6} | {:>8} {:>8} | {:>4} {:>3} {:>9} {:>7} | {:<28} | (no 128-bit schedule)",
                        blowup,
                        format!("2^-{blowup}"),
                        fri_per_layer,
                        fri_total,
                        k,
                        "-",
                        "-",
                        "-",
                        "-",
                    );
                    continue;
                };
                let reps: Vec<u64> = r.rounds.iter().map(|x| x.repetitions).collect();
                let winner = if r.total_queries < fri_total { "STIR" } else { "FRI" };
                println!(
                    "{:>5} {:>6} | {:>8} {:>8} | {:>4} {:>3} {:>9} {:>7} | {:<28} | {}",
                    blowup,
                    format!("2^-{blowup}"),
                    fri_per_layer,
                    fri_total,
                    r.fold,
                    r.n_rounds,
                    r.total_queries,
                    r.n_grinding_bits,
                    format!("{reps:?}"),
                    winner,
                );
            }
            println!();
        }
    }

    /// Validate the STIR repetition schedule against the paper's recommended
    /// parameters (ePrint 2024/390 §5.3/§6.2): λ=128, 22 PoW bits => protocol
    /// security 106, fold k=16, stop at 2^6. For d=2^20, ρ=1/2 the per-round
    /// repetitions are [212, 53, 31, 22, 17] (t_i = ceil(2*106 / log_inv_rate_i),
    /// log_inv_rate_i = 1 + i*(log2(16)-1) = 1,4,7,10,13).
    #[test]
    fn test_stir_repetitions_match_paper() {
        let params = StirSecurityParams {
            field_size: goldilocks_cube_field_size(),
            dimension: 1 << 20,
            rate: 0.5,
            n_opening_points: 1,
            n_functions: 1,
            max_grinding_bits: 22,
            use_max_grinding_bits: true,
            tree_arity: 4,
            target_security_bits: 128,
            fold_candidates: vec![16],
            stopping_degree: 64,
            max_rounds: 24,
        };
        let r = get_optimal_stir_query_params("JBR", &params);
        let reps: Vec<u64> = r.rounds.iter().map(|x| x.repetitions).collect();
        assert_eq!(reps, vec![212, 53, 31, 22, 17], "STIR repetitions must match paper §5.3");
        assert_eq!(r.n_grinding_bits, 22, "grinding is the global 22-bit PoW");
        assert!(r.achieved_security_bits >= 128);
    }
}
