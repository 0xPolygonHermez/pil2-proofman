# STIR/FRI Parameterized Security Calculator — Design

**Date:** 2026-06-15
**File:** `setup/pil2-stark/src/types/security.rs`
**Reference:** STIR — *Reed–Solomon Proximity Testing with Fewer Queries*, ePrint 2024/390 (§5–7)
**Scope:** Parameter **calculator only**. No prover, verifier, or circom changes.

## Goal

Extend `security.rs` so the query-parameter calculator supports **both FRI and STIR**,
selectable by a `protocol` parameter, while **keeping the target security at 128 bits**.

For STIR the calculator must, given the field size, initial dimension/rate, and a target of
128 bits, **choose an optimal uniform folding factor `k` and round count `M`**, then derive the
per-round repetition counts `tᵢ`, grinding bits, and report the total query cost. The 128-bit
guarantee must be a **paper-faithful transcription** of STIR's soundness analysis (Theorem 5.1
and the round-by-round error terms of §6), not a heuristic approximation.

## Non-Goals (YAGNI)

- No prover/verifier/circom implementation of STIR (this is step one only).
- No capacity/conjectured-rate-1 proximity bound. We support the **provable Johnson (JBR)** and
  **Unique-Decoding (UDR)** regimes only, selectable exactly like the existing FRI path.
- No tunable OOD-sample count.

## IMPLEMENTATION UPDATE (2026-06-15, verified against the authors' reference impl)

The IACR PDF blocks automated fetch, so the soundness model was verified against the STIR
authors' own Rust reference implementation (`github.com/WizardOfMenlo/stir`, by co-author Fenzi):

- **Rate recurrence (corrected):** `ρ_{i+1} = ρ_i · 2 / k` (domain halves, degree folds by `k`).
  The original draft `ρ·k/2` was inverted; with it STIR could never beat FRI.
- **Per-round repetitions:** `t_i = ceil(2 · λ / log_inv_rate_i)` (provable/Johnson; constant 2).
- **Per-round grinding:** `g_i = max(0, ceil(λ' − (log_inv_rate_i/2)·t_i))`, capped at
  `max_grinding_bits`, where `λ'` is the target raised by the `log2(#rounds)` union penalty. This
  is the user-requested per-round grinding — a vector, one entry per round.
- **Gating on the reference closed-form.** `achieved = min_i((log_inv_rate_i/2)·t_i + g_i) −
  log2(#rounds)`. The independent term-by-term error sum (`calculate_stir_total_error`, including
  the OOD term) is retained as a **diagnostic cross-check only**, not a gate — coupling the
  batched-commit list-decoding term into the per-round query union double-counts a term the
  reference already folds into the repetition sizing (the FRI path handles that same term via its
  own gap/alpha machinery, separately).
- **KEY COMPARISON INSIGHT:** STIR's win shows only against FRI's *true* verifier work
  `n_queries × n_fold_layers` (FRI re-queries its full count at every layer because its rate never
  improves). Comparing STIR's round-sum to FRI's single-layer count is the apples-to-oranges error
  that makes STIR look worse than it is. STIR beats FRI per-layer-summed work in every swept config.

## Background: how STIR differs from FRI (and why it needs fewer queries)

FRI keeps the rate `ρ` (roughly) constant across folding rounds, so each round's proximity
parameter `δ ≈ 1 − √ρ` (JBR) barely improves, and the single-query distinguishing probability
`1 − δ` stays close to `√ρ`. To reach 128 bits, FRI needs **many** queries (current production:
~219 for the golden config).

STIR **reduces the rate every round**: it folds by `k` while shrinking the evaluation domain by
2, so the next round's rate improves roughly as

```
ρ_{i+1} ≈ ρ_i · k / 2
```

(degree drops by `k`, domain `|L|` halves). Because `ρᵢ` shrinks geometrically, the per-round
proximity parameter `δᵢ = 1 − √ρᵢ` grows toward 1, and the per-round repetition count `tᵢ` needed
to reach a security target **decreases geometrically**. Net query complexity is
`O(log d + λ · log(log d / log(1/ρ)))` vs FRI's `O(λ · log d)`.

## STIR soundness terms (what the total-error function sums/maxes)

Per round `i ∈ {0..M}`, with round rate `ρᵢ` and round dimension `dᵢ`:

1. **Fold proximity error** — from the proximity generator at `(dᵢ, ρᵢ)`. Reuses the existing
   `JBR`/`UDR` `calculate_powers_error(n_functions)` machinery, constructed on a **fresh
   `RegimeParams` for that round's reduced rate and dimension**. This is where STIR's advantage
   shows up numerically.
2. **Out-of-domain (OOD) sampling error** — `s = 1` sample. Error term of the form
   `(dᵢ · L_listsize²) / |F|` (STIR §6 OOD lemma), i.e. the probability the OOD point fails to
   pin a unique folded codeword. New term; absent from FRI.
3. **Proximity-gen / batching error** — analogous to FRI's `calculate_batch_commit_error`; the
   batched-function and per-round-commit errors, taken as a max.
4. **Query (repetition) error** — `(1 − δᵢ)^{tᵢ}` at the round's **improved** rate, combined with
   grinding `2^{-g}`.

`calculate_stir_total_error` returns the dominant (max over rounds of the per-round combined
error, combined with the global batch/OOD contributions). `security_bits_from_error` then maps
that to bits, exactly as the FRI path does today.

## API design

### Protocol selector

```rust
pub enum Protocol { Fri, Stir }
```

### Public entry points

- Keep `get_optimal_fri_query_params(regime_name, &FRISecurityParams) -> FRIQueryResult`
  **unchanged** — all six current callers compile untouched. Internally it now delegates to the
  shared FRI path.
- Add `get_optimal_stir_query_params(regime_name, &StirSecurityParams) -> StirQueryResult`.
- (Optional thin unifier `get_optimal_query_params(protocol, regime_name, …)` only if it reads
  cleanly; not required.)

`regime_name` accepts `"JBR"` or `"UDR"` for **both** protocols (per decision: selectable).

### Inputs

`FRISecurityParams` stays exactly as-is (back-compat). Add:

```rust
pub struct StirSecurityParams {
    pub field_size: Float,
    pub dimension: u64,          // initial degree bound d
    pub rate: f64,               // initial rate ρ₀
    pub n_opening_points: u64,
    pub n_functions: u64,
    pub max_grinding_bits: u64,
    pub use_max_grinding_bits: bool,
    pub tree_arity: u64,
    pub target_security_bits: u64,
    // STIR-specific search bounds (optimizer chooses within these):
    pub fold_candidates: Vec<u64>,   // default {2,4,8,16}; powers of two
    pub max_rounds: u64,             // upper bound on M (safety valve)
    // OOD fixed at s = 1 internally; not exposed.
}
```

### Outputs

```rust
pub struct StirRound {
    pub rate: f64,
    pub dimension: u64,
    pub fold: u64,          // k for this round (uniform across rounds in this design)
    pub repetitions: u64,   // tᵢ
}

pub struct StirQueryResult {
    pub rounds: Vec<StirRound>,
    pub n_grinding_bits: u64,
    pub total_queries: u64,  // Σ tᵢ
    pub fold: u64,           // chosen uniform k
    pub n_rounds: u64,       // M
    pub achieved_security_bits: i64,
}
```

`FRIQueryResult` unchanged.

## Internal structure

Reused untouched: `RegimeParams`, `JBR`, `UDR`, `DecodingRegime` trait,
`calculate_powers_error`, `calculate_mtp_hashes`, `calculate_query_num_hashes`,
`security_bits_from_error`, `goldilocks_cube_field_size`, all helpers.

New (added in a clearly delimited STIR section):

- `fn stir_round_schedule(d, ρ₀, k, M) -> Vec<(dᵢ, ρᵢ)>` — derives per-round reduced
  dimension/rate.
- `fn calculate_ood_error(regime/listsize, dᵢ, field_size) -> Float`.
- `fn calculate_stir_round_error(regime_i, tᵢ, grinding, n_functions) -> Float`.
- `fn calculate_stir_total_error(rounds, grinding, n_functions, regime_name, field_size) -> Float`.
- `fn solve_round_repetitions(...)` — per round, `ceil` the `tᵢ` that meets the residual bit
  budget at `δᵢ`.
- `fn optimize_stir_schedule(params, regime_name) -> StirQueryResult` — outer search over
  `k ∈ fold_candidates` and `M ∈ 1..=max_rounds`; for each, build the regime per round, solve
  `tᵢ`, verify total ≥ 128 bits, compute total hash-weighted query cost via
  `calculate_query_num_hashes`, keep the **min-cost** schedule that meets the target.

## Error handling

- `optimize_stir_schedule`: if **no** `(k, M)` in the search space meets 128 bits, return an
  `Err`/panic with a clear message (mirrors FRI's `alpha` safety valve).
- Per-round `RegimeParams` must keep `δᵢ > 0` and JBR's gap assertion satisfied; if a round's
  reduced rate degenerates (e.g. `ρᵢ → 1` because `k/2 ≥ 1` pushed it up, or `dᵢ` underflows), that
  `(k, M)` candidate is **skipped**, not fatal.
- Guard against `dᵢ < 1` (too many rounds for the starting degree).

## Testing

1. **FRI regression:** all existing tests in this file must pass unchanged
   (`test_golden_jbr_optimal_params` = 219 queries, `test_udr_optimal_params` = 289,
   `test_compressor_params` = 110, `test_specified_ranges_params` = 228, etc.).
2. **STIR meets target:** for the golden config (d=2¹⁷, ρ=1/2, Goldilocks³), assert
   `achieved_security_bits ≥ 128`.
3. **STIR fewer queries than FRI:** assert STIR `total_queries` is **substantially lower** than
   the FRI count for the same config (the paper's headline; sanity bound e.g. STIR < FRI/2).
4. **STIR schedule sanity:** per-round `rate` is monotonically non-increasing; `dimension`
   strictly decreasing; `Σ repetitions == total_queries`.
5. **Paper cross-check (best-effort):** add an `eprintln!`-style informational test dumping the
   chosen `(k, M, tᵢ)` for d∈{2²⁰, 2²⁶} at ρ=1/2 to compare against STIR §7/Table 2 ballparks.
   Asserted loosely (order of magnitude), since the paper's field and exact term constants differ.

## Risks / notes

- The exact constants in STIR's OOD and proximity-gen error terms (§6) must be transcribed
  carefully; an off-by-constant changes bit counts. The spec fixes the *shape*; the implementation
  PR will cite the precise equation numbers inline in comments next to each term.
- We deliberately mirror the existing file's high-precision `rug::Float` style and the JBR/UDR
  Decimal.js-compat quirks so STIR results are reproducible the same way FRI's are.
