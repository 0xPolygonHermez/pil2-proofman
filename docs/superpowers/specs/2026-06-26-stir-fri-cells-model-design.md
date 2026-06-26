# STIR-vs-FRI in the Cells Model — Design

**Date:** 2026-06-26
**Status:** Approved, pending spec review

## Goal

Extend the linear-hash **cells model** so each recursive-verifier configuration can
be costed under **FRI or STIR** low-degree testing, selectable by a protocol knob.
Two artifacts change in lockstep:

- `setup/pil2-stark/src/types/cells.rs` — the analytical Rust model.
- `docs/cells-model/recursion-cell-memory-model.html` — the interactive heatmap that
  reimplements that model in inline JS.

The STIR query-parameter calculator (`setup/pil2-stark/src/types/security/stir.rs`,
`get_optimal_stir_query_params` / `try_get_optimal_stir_query_params`) already exists
and is the source of truth for STIR schedules. This work *consumes* it from the cells
model; it does not modify the calculator.

## Key invariant

> **Stage 0 (trace/Q commitment) hashing is structurally unchanged.** Only the
> folding term and its query-count multiplier differ between FRI and STIR.

Concretely, the per-leaf hash counts for `fixed / stageQ / stage2 / stage1`
(`ceil(cols / sponge_rate)`) are identical for both protocols. What differs:

| | FRI | STIR |
|---|---|---|
| Stage-0 open multiplier | `n_queries` | `t₀` (round-0 repetitions) |
| Folding term | `Σ_layer ceil(2^drop·3 / rate)` × `n_queries` | `Σ_round t_i · ceil(k·3 / rate)` |
| Fold factor | `2^fold_bits` per layer | `k = 2^fold_bits` (pinned, same as FRI) |
| Stopping degree | implicit final degree 6 | `stopping_degree = 2^6` |

`k = 2^fold_bits` and `stopping_degree = 2^6` are pinned so FRI and STIR fold by the
same factor down to the same degree — the apples-to-apples comparison the calculator's
`test_stir_fewer_queries_than_fri_golden` uses.

## Fixed protocol constants (shared by both paths)

Unchanged from the current FRI model: `TARGET_SECURITY_BITS = 128`, `REGIME = "JBR"`,
`TREE_ARITY = 4`, `EXTENSION_DEGREE = 3`.

## Configurable grinding bits (amendment 2026-06-26)

Grinding bits are **configurable**, not a fixed constant. `CellModelParams` carries a
`grinding_bits: u64` field (default **20**, the former `GRINDING_BITS` value) that feeds
the `max_grinding_bits` of BOTH the FRI and STIR security calls — one shared knob, so
FRI and STIR are compared at the same grinding level. For STIR this sets
`protocol_security = 128 − grinding_bits`. The HTML exposes it as a live slider next to
the protocol toggle; `n_queries` / `t_i` / total cells recompute on change. The old
`GRINDING_BITS` constant remains only as the default seed for the field.

## Part 1 — Rust (`cells.rs`)

### Input

Add a protocol selector to `CellModelParams`:

```rust
pub use super::security::Protocol; // { Fri, Stir }

pub struct CellModelParams {
    // ... existing fields ...
    pub protocol: Protocol,
}
```

- Presets (`recursion`, `recursion_with_stage2`, `zisk_main`) default `protocol: Fri`
  so all existing call sites and tests are byte-identical.
- Add a `.with_protocol(Protocol) -> Self` builder (or a `*_stir` preset variant) for
  the STIR runs and the sweep.

### Folding computation — branch on protocol

**FRI path — unchanged.** `derive_folding_factors`, per-opened-layer
`ceil(2^drop·3 / rate)`, `fri_hashes_per_query`, stage-0 `× n_queries`,
`total_hashes = hashes_per_query · n_queries · n_proofs`. No behavioral change.

**STIR path — new.** Build the calculator input from the model params:

```rust
StirSecurityParams {
    field_size: goldilocks_cube_field_size(),
    dimension: 1 << n,
    rate: 1.0 / (1 << blowup_bits) as f64,
    n_opening_points, n_functions,        // diagnostic only in the calculator
    max_grinding_bits: GRINDING_BITS,
    use_max_grinding_bits: true,
    tree_arity: TREE_ARITY,
    target_security_bits: TARGET_SECURITY_BITS,
    fold_candidates: vec![1 << fold_bits], // k = 2^fold_bits, pinned
    stopping_degree: 1 << 6,               // 2^6, matches FRI implicit final degree
    max_rounds: 24,
}
```

Call `try_get_optimal_stir_query_params("JBR", &params)`:

- **`Some(schedule)`** →
  - `t₀ = schedule.rounds[0].repetitions`.
  - Stage-0 trace/Q hashes: same per-leaf counts as FRI, opened **t₀×**.
  - Folding cells: `Σ_round ( t_i · ceil(k·3 / sponge_rate) )` where `k = round.fold`
    extension siblings → `3·k` base elements per opened round at the model's sponge
    rate.
  - `total_hashes = (stage0_hashes·t₀ + folding_hashes_summed) · n_proofs`.
  - `total_cells = total_hashes · cells_per_perm`.
- **`None`** (no 128-bit STIR schedule for this cell) → estimate is marked infeasible.

### `CellEstimate` changes

- Add `pub protocol: Protocol`.
- `n_queries`: for STIR carries `t₀` (the stage-0 open count); keep the field name but
  document the dual meaning, OR add `pub stir_total_queries: Option<u64>` for the
  Σ t_i figure. (Implementation detail — decided during writing-plans.)
- Keep `folding_factors` / `fri_steps` populated on the FRI path. Add a parallel
  `pub stir_rounds: Vec<StirRoundHashes>` (`{ rate, dimension, repetitions, hashes }`)
  for the STIR report; empty on the FRI path.
- Add a feasibility flag for "no STIR schedule": either return `Option<CellEstimate>`
  from a fallible variant, or a `pub schedule_found: bool`. The fit checks
  (`needed_stage1_cols`, `fits`, `main_fits_in_recursion`) stay **protocol-agnostic** —
  they operate on `total_cells`.
- `report()` prints the FRI layer breakdown or the STIR round breakdown depending on
  `protocol`.

### Tests

- **FRI regression:** every existing FRI assertion still passes unchanged
  (default `protocol: Fri`).
- **STIR golden:** for a pinned `(n, blowup, k)` the per-round reps match the
  calculator's output (cross-referenced against `stir.rs`
  `test_stir_repetitions_match_paper` / `test_stir_meets_128_bits_golden`).
- **STIR stage-0 opened t₀×:** `total_hashes` for STIR equals
  `stage0_hashes·t₀ + Σ folding` (× n_proofs), NOT `× n_queries`.
- **STIR folding beats FRI folding at same k:** at a config where the calculator's
  golden test shows STIR < FRI, the STIR folding-cell sum is below the FRI one.
- **Infeasible cell:** a `(rate, k)` with no 128-bit schedule yields the
  no-schedule outcome (None / `schedule_found == false`), not a panic.
- **Fit checks protocol-agnostic:** `main_fits_in_recursion` works for a STIR
  recursion estimate.

### Sweep

Extend `cells_sweep_csv` (or add `cells_sweep_csv_stir`) to emit STIR columns
(`stir_t0, stir_total_queries, stir_total_cells, stir_schedule_found`) alongside the
FRI columns, so the Rust↔JS cross-check has ground truth.

## Part 2 — Interactive HTML (`recursion-cell-memory-model.html`)

### New control

A **Protocol** selector (FRI / STIR) in the `.controls` bar, a live knob like the
hash/metric selectors — flipping it recomputes the whole heatmap.

### JS port of the STIR path

Port the minimal STIR schedule math (currently absent from the page):

- `stir_round_schedule(d, ρ₀, k, m)` — `ρ_{i+1} = ρ_i·2/k`, `d_{i+1} = d_i/k`.
- `stir_num_rounds(d, stopping_degree=2^6, k, max_rounds=24)` — fold `d` down to the
  stopping degree.
- `stir_repetitions(protocol_security, lir)` — `ceil(2·protocol_security / lir)`,
  `protocol_security = 128 − 20 = 108`. **NB:** the cells model pins
  `GRINDING_BITS = 20`, so protocol security is **108**, not the **106** the STIR
  calculator's paper-golden test uses (22 grinding bits). The `max_grinding_bits` fed
  to the calculator is `GRINDING_BITS = 20` for both protocols here — do not "correct"
  it to 22. (FRI in this model already uses 20.)
- Security gate — weakest non-`fully_opened` round: `(lir/2)·t_i + grinding ≥ 128`;
  `fully_opened` when `t_i ≥ domain_size = floor(d_i/ρ_i)` (excluded from the gate).
- For the pinned `k = 2^fold_bits`, this reproduces
  `get_optimal_stir_query_params` for that single candidate.

Cells: folding `= Σ t_i·ceil(k·3 / rate_hash)`; stage-0 opened t₀×; `× n_proofs ×
cells_per_perm` — identical formula to the Rust side.

A cell with no 128-bit STIR schedule reuses the existing greyed/`over` styling (or a
distinct "no schedule" marker) so the heatmap stays readable.

### Verification of the port

The README already promises the in-browser `n_queries` reproduces the Rust
`security.rs` output exactly across the swept range. Extend that contract to STIR:
spot-check the JS STIR totals (`t₀`, `Σ t_i`, `total_cells`, schedule-found) against
the Rust sweep CSV for several `(N, blowup, fold)` cells, and document the check in the
README. The Rust↔JS cross-check is the guard against the risk of porting the optimizer
(float precision, the `fully_opened` edge, the security gate).

### README update

Note the new Protocol toggle, what STIR changes (folding term + t₀ multiplier, stage-0
layout unchanged), and the `k = 2^fold_bits` / `stopping_degree = 2^6` convention.

## Out of scope

- Modifying the STIR/FRI query-parameter calculator in `security/`.
- The separate `verifier_hash_comparison.rs` example (a Poseidon-permutation model,
  not the cells model) — left as-is; it is the reference for the STIR folding shape.
- Merkle-path hashing, custom gates, transcript costs in the cells model (the model
  deliberately ignores these today; unchanged).
- Letting the optimizer search `k` freely — pinned to `2^fold_bits` for apples-to-apples.

## Risks

- **JS optimizer port fidelity** — highest risk. Mitigated by the Rust↔JS cross-check
  against the sweep CSV.
- **`n_queries` field overload** — `n_queries` meaning `t₀` for STIR could confuse
  downstream readers of `CellEstimate`. Resolve by clear doc or a dedicated field
  during writing-plans.
