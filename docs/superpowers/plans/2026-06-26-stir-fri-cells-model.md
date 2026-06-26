# STIR-vs-FRI in the Cells Model — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a STIR low-degree-test path to the linear-hash cells model (`cells.rs`) and its interactive HTML mirror, so each recursive-verifier config can be costed under FRI or STIR with a protocol toggle.

**Architecture:** `cells.rs` gains a `Protocol` field on `CellModelParams` (defaults `Fri`, leaving every existing path byte-identical). The STIR path consumes the existing `try_get_optimal_stir_query_params` calculator with `k = 2^fold_bits` pinned and `stopping_degree = 2^6`, then counts folding cells as `Σ_round tᵢ·ceil(k·3/rate)` with stage-0 trace/Q hashes opened `t₀×`. The HTML page reimplements the same STIR schedule + security gate in inline JS, cross-checked against a Rust sweep CSV.

**Tech Stack:** Rust (`pil2-stark-setup` crate), inline JS/HTML (single self-contained file).

## Global Constraints

- Target security: **128 bits**. Regime: **JBR**. Tree arity: **4**. Extension degree: **3**.
- Grinding: **configurable** via `CellModelParams.grinding_bits` (default **20**, the `GRINDING_BITS` const), feeding `max_grinding_bits` of BOTH the FRI and STIR security calls — one shared knob (Task 2b). For STIR, `protocol_security = 128 − grinding_bits`. Do NOT hardcode 22 anywhere. (Other constants stay fixed: `TARGET_SECURITY_BITS`, `REGIME`, `TREE_ARITY`, `EXTENSION_DEGREE`.)
- STIR fold is pinned to `k = 2^fold_bits` (apples-to-apples with FRI); `stopping_degree = 2^6`; `max_rounds = 24`.
- The FRI path must remain behaviorally identical: presets default `protocol: Fri`, and all current FRI tests pass unchanged.
- Crate name for all `cargo` commands: **`pil2-stark-setup`**.
- Test invocation: `cargo test -p pil2-stark-setup --lib <path>`. (Feature flags are NOT needed for `types::cells` / `types::security` unit tests.)
- Commit after each task. End commit messages with the Co-Authored-By trailer:
  `Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>`

---

## File Structure

- `setup/pil2-stark/src/types/cells.rs` — MODIFY. Add `Protocol` field, STIR folding branch, `CellEstimate` STIR fields, STIR report, tests.
- `setup/pil2-stark/src/types/security/mod.rs` — already re-exports `Protocol`, `StirSecurityParams`, `try_get_optimal_stir_query_params`. No change needed (verify in Task 1).
- `docs/cells-model/recursion-cell-memory-model.html` — MODIFY. Protocol selector + JS STIR port.
- `docs/cells-model/README.md` — MODIFY. Document the toggle + cross-check.

---

## Task 1: Add `Protocol` to `CellModelParams` (FRI path unchanged)

**Files:**
- Modify: `setup/pil2-stark/src/types/cells.rs`
- Test: same file (`mod tests`)

**Interfaces:**
- Consumes: `super::security::Protocol` (enum `{ Fri, Stir }`, already re-exported from `security/mod.rs`).
- Produces: `CellModelParams { ..., protocol: Protocol }`; presets default `Fri`; new builder `CellModelParams::with_protocol(self, p: Protocol) -> Self`.

- [ ] **Step 1: Verify `Protocol` is importable**

Run: `cargo build -p pil2-stark-setup 2>&1 | head -5` (baseline, should already compile).
Then confirm the re-export exists:
Run: `grep -n 'pub use stir::' setup/pil2-stark/src/types/security/mod.rs`
Expected: a line listing `Protocol` (already present per mod.rs lines 24-27).

- [ ] **Step 2: Write the failing test**

Add to `mod tests` in `cells.rs`:

```rust
#[test]
fn protocol_defaults_to_fri_and_builder_overrides() {
    let p = CellModelParams::recursion(15, 6, 4, 0, b3(1));
    assert!(matches!(p.protocol, Protocol::Fri));
    let p2 = p.with_protocol(Protocol::Stir);
    assert!(matches!(p2.protocol, Protocol::Stir));
}
```

- [ ] **Step 3: Run test to verify it fails**

Run: `cargo test -p pil2-stark-setup --lib types::cells::tests::protocol_defaults_to_fri_and_builder_overrides 2>&1 | tail -20`
Expected: FAIL to compile — `no field protocol` / `no method with_protocol` / `Protocol` not in scope.

- [ ] **Step 4: Implement**

At the top of `cells.rs`, extend the security import:

```rust
use super::security::{get_optimal_fri_query_params, goldilocks_cube_field_size, FRISecurityParams, Protocol};
```

Add the field to `CellModelParams` (after `hash`):

```rust
    /// The hash used for linear hashing (sets sponge rate + cells/permutation).
    pub hash: HashFamily,
    /// Low-degree-test protocol: FRI (default) or STIR.
    pub protocol: Protocol,
```

In BOTH `recursion_with_stage2` and `zisk_main`, add `protocol: Protocol::Fri,` to the returned struct literal. (The `recursion` preset delegates to `recursion_with_stage2`, so it inherits the default.)

Add the builder in `impl CellModelParams`:

```rust
    /// Override the low-degree-test protocol (defaults to FRI in every preset).
    pub fn with_protocol(mut self, protocol: Protocol) -> Self {
        self.protocol = protocol;
        self
    }
```

- [ ] **Step 5: Run test to verify it passes (and FRI tests still pass)**

Run: `cargo test -p pil2-stark-setup --lib types::cells 2>&1 | tail -20`
Expected: PASS — the new test passes and ALL existing `types::cells` tests still pass.

- [ ] **Step 6: Commit**

```bash
git add setup/pil2-stark/src/types/cells.rs
git commit -m "feat(cells): add Protocol field to CellModelParams (defaults FRI)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 2: STIR folding helper — schedule + per-round folding hashes

**Files:**
- Modify: `setup/pil2-stark/src/types/cells.rs`
- Test: same file (`mod tests`)

**Interfaces:**
- Consumes: `super::security::{StirSecurityParams, try_get_optimal_stir_query_params, StirRound}` (re-exported), `hashes_for_elements`, `EXTENSION_DEGREE`, `GRINDING_BITS`, `TREE_ARITY`, `TARGET_SECURITY_BITS`.
- Produces:
  - `pub struct StirRoundHashes { pub rate: f64, pub dimension: u64, pub repetitions: u64, pub hashes: u64 }`
  - `fn build_stir_for_model(p: &CellModelParams, folding_factors_len_hint: ()) -> Option<StirQueryResult>` — internal; builds `StirSecurityParams` from model params and calls the calculator. (Signature finalized below — no hint arg actually needed.)
  - `fn stir_fold_hashes(rounds: &[StirRound], rate: u64) -> (Vec<StirRoundHashes>, u64)` returns per-round breakdown and the summed folding hashes `Σ tᵢ·ceil(k·3/rate)`.

The exact internal signatures:

```rust
fn build_stir_for_model(p: &CellModelParams) -> Option<super::security::StirQueryResult>;
fn stir_fold_hashes(rounds: &[super::security::StirRound], sponge_rate: u64) -> (Vec<StirRoundHashes>, u64);
```

- [ ] **Step 1: Write the failing tests**

Add to `mod tests`:

```rust
#[test]
fn stir_for_model_returns_schedule_for_standard_config() {
    // n=17, blowup=1 (rate 1/2), fold_bits=2 => k=4, stopping 2^6.
    let p = CellModelParams::recursion(17, 1, 2, 100, b3(1)).with_protocol(Protocol::Stir);
    let sched = build_stir_for_model(&p).expect("a 128-bit STIR schedule should exist here");
    assert_eq!(sched.fold, 4, "k must be 2^fold_bits = 4");
    assert!(sched.achieved_security_bits >= 128);
    // round dimension strictly decreases
    for w in sched.rounds.windows(2) {
        assert!(w[1].dimension < w[0].dimension);
    }
}

#[test]
fn stir_fold_hashes_sums_per_round_reps_times_leaf() {
    // Two synthetic rounds, k=4 => 4*3 = 12 elements/leaf.
    // rate 4 (Blake3): ceil(12/4) = 3 hashes per opened leaf.
    let rounds = vec![
        super::security::StirRound { rate: 0.5, dimension: 1 << 17, fold: 4, repetitions: 10, grinding_bits: 20, fully_opened: false },
        super::security::StirRound { rate: 0.25, dimension: 1 << 15, fold: 4, repetitions: 6, grinding_bits: 0, fully_opened: false },
    ];
    let (breakdown, total) = stir_fold_hashes(&rounds, 4);
    assert_eq!(breakdown.len(), 2);
    assert_eq!(breakdown[0].hashes, 3); // ceil(4*3/4)
    assert_eq!(breakdown[0].repetitions, 10);
    // total = 10*3 + 6*3 = 48
    assert_eq!(total, 48);
}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cargo test -p pil2-stark-setup --lib types::cells::tests::stir_for_model_returns_schedule_for_standard_config types::cells::tests::stir_fold_hashes_sums_per_round_reps_times_leaf 2>&1 | tail -20`
Expected: FAIL to compile — `build_stir_for_model` / `stir_fold_hashes` / `StirRoundHashes` undefined.

- [ ] **Step 3: Implement**

Extend the security import to pull the STIR types:

```rust
use super::security::{
    get_optimal_fri_query_params, goldilocks_cube_field_size, try_get_optimal_stir_query_params,
    FRISecurityParams, Protocol, StirQueryResult, StirRound, StirSecurityParams,
};
```

Add the public breakdown struct near `FriStepHashes`:

```rust
/// Per-STIR-round hash breakdown (one entry per folding round).
#[derive(Debug, Clone)]
pub struct StirRoundHashes {
    /// This round's rate (= dimension / domain).
    pub rate: f64,
    /// This round's folded dimension.
    pub dimension: u64,
    /// Repetitions (queries) opened this round: STIR's per-round `t_i`.
    pub repetitions: u64,
    /// `ceil(fold * EXTENSION_DEGREE / sponge_rate)` — hashes per opened leaf.
    pub hashes: u64,
}
```

Add the helpers in the "Core helpers" section:

```rust
/// Build a STIR schedule for the model params: fold `k = 2^fold_bits` pinned,
/// stopping degree `2^6`, grinding `GRINDING_BITS`, JBR, target 128. Returns
/// `None` when no 128-bit schedule exists for this `(rate, k)` cell.
fn build_stir_for_model(p: &CellModelParams) -> Option<StirQueryResult> {
    let k = 1u64 << p.fold_bits;
    let params = StirSecurityParams {
        field_size: goldilocks_cube_field_size(),
        dimension: 1u64 << p.n,
        rate: 1.0 / (1u64 << p.blowup_bits) as f64,
        n_opening_points: p.n_opening_points,
        n_functions: p.n_functions,
        max_grinding_bits: GRINDING_BITS,
        use_max_grinding_bits: true,
        tree_arity: TREE_ARITY,
        target_security_bits: TARGET_SECURITY_BITS,
        fold_candidates: vec![k],
        stopping_degree: 1u64 << 6,
        max_rounds: 24,
    };
    try_get_optimal_stir_query_params(REGIME, &params)
}

/// Per-round STIR folding hashes: each round opens `t_i` leaves, each leaf holding
/// `fold` degree-3 extension values (`fold * EXTENSION_DEGREE` base elements).
/// Returns the per-round breakdown and the summed folding hashes `Σ t_i * hashes_i`.
fn stir_fold_hashes(rounds: &[StirRound], sponge_rate: u64) -> (Vec<StirRoundHashes>, u64) {
    let mut breakdown = Vec::with_capacity(rounds.len());
    let mut total: u64 = 0;
    for r in rounds {
        let elements = r.fold * EXTENSION_DEGREE;
        let hashes = hashes_for_elements(elements, sponge_rate);
        total += hashes * r.repetitions;
        breakdown.push(StirRoundHashes { rate: r.rate, dimension: r.dimension, repetitions: r.repetitions, hashes });
    }
    (breakdown, total)
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cargo test -p pil2-stark-setup --lib types::cells::tests::stir_for_model_returns_schedule_for_standard_config types::cells::tests::stir_fold_hashes_sums_per_round_reps_times_leaf 2>&1 | tail -20`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add setup/pil2-stark/src/types/cells.rs
git commit -m "feat(cells): STIR schedule builder + per-round folding hash helper

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 2b: Make `grinding_bits` a configurable `CellModelParams` field

**Why:** User requirement (2026-06-26) — grinding bits must be configurable (and shown
in the HTML, Tasks 6/7). One shared knob feeds BOTH the FRI and STIR security calcs so
the two are compared at the same grinding level.

**Files:**
- Modify: `setup/pil2-stark/src/types/cells.rs`
- Test: same file (`mod tests`)

**Interfaces:**
- Consumes: existing `GRINDING_BITS` const (becomes the default seed), Task 2's `build_stir_for_model`.
- Produces: `CellModelParams.grinding_bits: u64` (default 20 in every preset); `build_stir_for_model` uses `p.grinding_bits` instead of the const. The FRI security call (Task 3) will likewise use `p.grinding_bits`.

- [ ] **Step 1: Write the failing test**

```rust
#[test]
fn grinding_bits_default_20_and_configurable() {
    let p = CellModelParams::recursion(15, 6, 4, 100, b3(1));
    assert_eq!(p.grinding_bits, 20, "default grinding must be 20");
    let p2 = CellModelParams { grinding_bits: 16, ..p };
    assert_eq!(p2.grinding_bits, 16);
    // More grinding -> STIR needs fewer protocol-security bits -> schedule still found,
    // and fewer grinding -> harder. Sanity: both build a schedule at a feasible config.
    let s_hi = build_stir_for_model(&CellModelParams::recursion(20, 1, 2, 100, b3(1))
        .with_protocol(Protocol::Stir));
    assert!(s_hi.is_some());
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p pil2-stark-setup --lib types::cells::tests::grinding_bits_default_20_and_configurable 2>&1 | tail -20`
Expected: FAIL to compile — `CellModelParams` has no field `grinding_bits`.

- [ ] **Step 3: Implement — add the field**

Add to `CellModelParams` (after `protocol`):

```rust
    /// Low-degree-test protocol: FRI (default) or STIR.
    pub protocol: Protocol,
    /// Proof-of-work grinding bits, fed to BOTH the FRI and STIR security calcs.
    /// Defaults to `GRINDING_BITS` (20). For STIR, `protocol_security = 128 - grinding_bits`.
    pub grinding_bits: u64,
```

In `recursion_with_stage2` and `zisk_main`, add `grinding_bits: GRINDING_BITS,` to the struct literal.

- [ ] **Step 4: Implement — use the field in `build_stir_for_model`**

In `build_stir_for_model`, change the `max_grinding_bits` line from the const to the field:

```rust
        max_grinding_bits: p.grinding_bits,
```

(Leave the rest of the function as Task 2 wrote it. The `GRINDING_BITS` const stays — it is now the default seed used by the presets.)

- [ ] **Step 5: Run test to verify it passes (and no regression)**

Run: `cargo test -p pil2-stark-setup --lib types::cells 2>&1 | tail -20`
Expected: PASS — the new test and all prior `types::cells` tests.

- [ ] **Step 6: Commit** — SKIP (no commits this session; see Global Constraints). Leave changes in the working tree.

---

## Task 3: STIR fields on `CellEstimate` + protocol branch in `estimate_linear_hash_cells`

**Files:**
- Modify: `setup/pil2-stark/src/types/cells.rs`
- Test: same file (`mod tests`)

**Interfaces:**
- Consumes: Task 1 (`protocol`), Task 2 (`build_stir_for_model`, `stir_fold_hashes`, `StirRoundHashes`).
- Produces: `CellEstimate` gains:
  - `pub protocol: Protocol`
  - `pub stir_rounds: Vec<StirRoundHashes>` (empty on the FRI path)
  - `pub schedule_found: bool` (always `true` for FRI; `false` for a STIR cell with no 128-bit schedule)
  - `n_queries` semantics: FRI = the FRI query count; STIR = `t₀` (round-0 reps). Documented on the field.

  The FRI branch is byte-identical to today. The STIR branch sets:
  - `n_queries = t₀`, `folding_factors = [k; n_rounds]` (for the report only), `fri_steps = []`,
  - `trace_hashes_per_query` = same per-leaf counts as FRI,
  - `fri_hashes_per_query` = the summed STIR folding hashes for ONE proof (i.e. `stir_fold_total`). **NB:** this field means per-query for FRI but the already-summed-over-rounds total for STIR — because STIR has no flat per-query folding count. Document this dual meaning on the field; Task 4's `stir_folding_beats_fri_folding_same_k` depends on it. Do not "normalize" it.
  - `hashes_per_query` = `trace_hashes_per_query` (kept meaning "stage-0 per open"); the STIR total is computed directly (see below) rather than via the FRI `× n_queries` formula.
  - `total_hashes = (trace_hashes_per_query * t0 + stir_fold_total) * n_proofs`.

- [ ] **Step 1: Write the failing tests**

```rust
#[test]
fn fri_estimate_is_unchanged_with_explicit_protocol() {
    // Same config, Fri explicitly: identical to the default-preset FRI estimate.
    let def = estimate_linear_hash_cells(&CellModelParams::recursion(15, 6, 4, 100, b3(700)));
    let exp = estimate_linear_hash_cells(
        &CellModelParams::recursion(15, 6, 4, 100, b3(700)).with_protocol(Protocol::Fri),
    );
    assert_eq!(def.total_cells, exp.total_cells);
    assert_eq!(def.total_hashes, exp.total_hashes);
    assert!(exp.schedule_found);
    assert!(exp.stir_rounds.is_empty());
    assert!(matches!(exp.protocol, Protocol::Fri));
}

#[test]
fn stir_estimate_opens_stage0_t0_times_and_sums_folding() {
    // n=17, blowup=1, fold_bits=2 (k=4), stage1=100, Blake3 cells=1.
    let p = CellModelParams::recursion(17, 1, 2, 100, b3(1)).with_protocol(Protocol::Stir);
    let est = estimate_linear_hash_cells(&p);
    assert!(est.schedule_found, "expected a 128-bit STIR schedule");
    assert!(matches!(est.protocol, Protocol::Stir));
    assert!(!est.stir_rounds.is_empty());
    // n_queries carries t0 for STIR.
    let t0 = est.n_queries;
    // Recompute the STIR total independently and compare.
    let sched = build_stir_for_model(&p).unwrap();
    assert_eq!(t0, sched.rounds[0].repetitions);
    let (_, fold_total) = stir_fold_hashes(&sched.rounds, p.hash.sponge_rate());
    let expected_total = (est.trace_hashes_per_query * t0 + fold_total) * p.n_proofs;
    assert_eq!(est.total_hashes, expected_total);
    assert_eq!(est.total_cells, est.total_hashes * p.hash.cells_per_perm());
}

#[test]
fn stir_trace_hashes_match_fri_trace_hashes() {
    // Stage-0 per-leaf counts identical between protocols (the invariant).
    let fri = estimate_linear_hash_cells(&CellModelParams::recursion(17, 1, 2, 100, b3(1)));
    let stir = estimate_linear_hash_cells(
        &CellModelParams::recursion(17, 1, 2, 100, b3(1)).with_protocol(Protocol::Stir),
    );
    assert_eq!(fri.trace_hashes_per_query, stir.trace_hashes_per_query);
}

#[test]
fn stir_no_schedule_sets_flag_not_panic() {
    // A high rate folded to a tiny stopping degree may have no 128-bit schedule.
    // blowup=8 (rate 1/256) is comfortably feasible; use a deliberately hard cell:
    // small n with large fold leaves too few rounds. n=8, fold_bits=4 (k=16), blowup=1.
    let p = CellModelParams::recursion(8, 1, 4, 50, b3(1)).with_protocol(Protocol::Stir);
    let est = estimate_linear_hash_cells(&p);
    // Either it found a schedule or it set the flag false — never panics.
    if !est.schedule_found {
        assert_eq!(est.total_cells, 0);
        assert!(est.stir_rounds.is_empty());
    }
}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cargo test -p pil2-stark-setup --lib types::cells::tests::fri_estimate_is_unchanged_with_explicit_protocol types::cells::tests::stir_estimate_opens_stage0_t0_times_and_sums_folding types::cells::tests::stir_trace_hashes_match_fri_trace_hashes types::cells::tests::stir_no_schedule_sets_flag_not_panic 2>&1 | tail -25`
Expected: FAIL to compile — `CellEstimate` has no `protocol` / `stir_rounds` / `schedule_found`.

- [ ] **Step 3: Implement — add the fields**

In `struct CellEstimate`, add after `n_queries` doc:

```rust
    /// Number of queries opening stage-0 commitments. FRI: the FRI query count.
    /// STIR: `t₀`, round-0 repetitions (stage-0 is opened t₀ times, not n_queries).
    pub n_queries: u64,
```

Update the existing `fri_hashes_per_query` doc-comment to note the dual meaning:

```rust
    /// FRI: folding hashes for one proof, one query. STIR: folding hashes for one
    /// proof, ALREADY SUMMED over rounds (STIR has no flat per-query count).
    pub fri_hashes_per_query: u64,
```

and add new fields (anywhere in the struct, e.g. after `fri_steps`):

```rust
    /// Low-degree-test protocol this estimate was computed for.
    pub protocol: Protocol,
    /// Per-round STIR folding breakdown (empty on the FRI path).
    pub stir_rounds: Vec<StirRoundHashes>,
    /// Whether a security-meeting schedule was found. Always true for FRI; false
    /// for a STIR cell with no 128-bit schedule (in which case totals are 0).
    pub schedule_found: bool,
```

- [ ] **Step 4: Implement — branch `estimate_linear_hash_cells`**

Replace the body of `estimate_linear_hash_cells` so the trace/stage computation is shared and the folding/total computation branches on `p.protocol`. The trace stages block is unchanged. After computing `trace_stages` and `trace_hashes_per_query`, replace everything from `// FRI: every folding step...` down to the `CellEstimate { ... }` return with:

```rust
    // Folding + totals branch on protocol. Stage-0 (trace/Q) per-leaf counts above
    // are identical for both; only the multiplier and folding term differ.
    let (
        n_queries,
        folding_factors_out,
        fri_steps,
        fri_hashes_per_query,
        stir_rounds,
        total_hashes,
        schedule_found,
    ) = match p.protocol {
        Protocol::Fri => {
            let fri = get_optimal_fri_query_params(
                REGIME,
                &FRISecurityParams {
                    field_size: goldilocks_cube_field_size(),
                    dimension: 1u64 << p.n,
                    rate: 1.0 / (1u64 << p.blowup_bits) as f64,
                    n_opening_points: p.n_opening_points,
                    n_functions: p.n_functions,
                    folding_factors: folding_factors.clone(),
                    max_grinding_bits: p.grinding_bits,
                    use_max_grinding_bits: true,
                    tree_arity: TREE_ARITY,
                    target_security_bits: TARGET_SECURITY_BITS,
                },
            );
            let n_queries = fri.n_queries;
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
            (n_queries, folding_factors.clone(), fri_steps, fri_hashes_per_query, Vec::new(), total_hashes, true)
        }
        Protocol::Stir => match build_stir_for_model(p) {
            Some(sched) => {
                let t0 = sched.rounds.first().map(|r| r.repetitions).unwrap_or(0);
                let (stir_rounds, fold_total) = stir_fold_hashes(&sched.rounds, rate);
                // Report-only folding_factors: k repeated per round.
                let k = 1u64 << p.fold_bits;
                let ff_out = vec![k; sched.rounds.len()];
                let total_hashes = (trace_hashes_per_query * t0 + fold_total) * p.n_proofs;
                (t0, ff_out, Vec::new(), fold_total, stir_rounds, total_hashes, true)
            }
            None => (0, Vec::new(), Vec::new(), 0, Vec::new(), 0, false),
        },
    };

    let hashes_per_query = trace_hashes_per_query + fri_hashes_per_query;
    let total_cells = total_hashes * p.hash.cells_per_perm();

    // Self-verification fit: pack total_cells into 2^n rows -> columns needed.
    let rows = 1u64 << p.n;
    let needed_stage1_cols = total_cells.div_ceil(rows);
    let fits = needed_stage1_cols <= p.stage1_cols;

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
        n_proofs: p.n_proofs,
        total_hashes,
        total_cells,
        needed_stage1_cols,
        assumed_stage1_cols: p.stage1_cols,
        fits,
        prover_memory_field_elems,
        prover_memory_gb,
        protocol: p.protocol,
        stir_rounds,
        schedule_found,
    }
}
```

Note: the `let rate = p.hash.sponge_rate();` line stays where it is (before the match). The `let folding_factors = derive_folding_factors(...)` line at the top of the function stays — it is still used for `prover_memory_field_elems` (which models FRI-style step buffers) and as the FRI folding source. Keep it.

- [ ] **Step 5: Run all cells tests**

Run: `cargo test -p pil2-stark-setup --lib types::cells 2>&1 | tail -25`
Expected: PASS — the 4 new tests pass and every prior `types::cells` test still passes.

- [ ] **Step 6: Commit**

```bash
git add setup/pil2-stark/src/types/cells.rs
git commit -m "feat(cells): STIR branch in estimate_linear_hash_cells (t0 stage-0, summed folding)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 4: STIR report + STIR-beats-FRI folding assertion

**Files:**
- Modify: `setup/pil2-stark/src/types/cells.rs`
- Test: same file (`mod tests`)

**Interfaces:**
- Consumes: Task 3 (`protocol`, `stir_rounds`, `schedule_found`).
- Produces: `CellEstimate::report()` prints the STIR round breakdown when `protocol == Stir`. No new public signature.

- [ ] **Step 1: Write the failing tests**

```rust
#[test]
fn stir_report_lists_rounds() {
    let p = CellModelParams::recursion(17, 1, 2, 100, b3(1)).with_protocol(Protocol::Stir);
    let est = estimate_linear_hash_cells(&p);
    let r = est.report();
    assert!(r.contains("STIR rounds"), "report should have a STIR section, got:\n{r}");
}

#[test]
fn stir_folding_beats_fri_folding_same_k() {
    // Same config, fold_bits=2 (k=4), low rate where STIR's geometric reduction shows.
    // Compare folding-only cells: STIR's summed folding < FRI's per-layer*n_queries folding.
    let base = CellModelParams::recursion(20, 1, 2, 200, b3(1));
    let fri = estimate_linear_hash_cells(&base);
    let stir = estimate_linear_hash_cells(&base.with_protocol(Protocol::Stir));
    assert!(stir.schedule_found);
    // FRI folding cells for one proof = fri_hashes_per_query * n_queries.
    let fri_fold = fri.fri_hashes_per_query * fri.n_queries;
    // STIR folding cells for one proof = sum over rounds of reps*hashes = fri_hashes_per_query (STIR meaning).
    let stir_fold = stir.fri_hashes_per_query; // already the summed total for one proof
    assert!(stir_fold < fri_fold, "STIR folding {stir_fold} should beat FRI folding {fri_fold}");
}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cargo test -p pil2-stark-setup --lib types::cells::tests::stir_report_lists_rounds types::cells::tests::stir_folding_beats_fri_folding_same_k 2>&1 | tail -20`
Expected: `stir_report_lists_rounds` FAILS (no "STIR rounds" in report). `stir_folding_beats_fri_folding_same_k` may already pass — if so, leave it as a regression guard.

- [ ] **Step 3: Implement — branch `report()`**

In `impl CellEstimate { pub fn report(&self) -> String { ... } }`, replace the FRI folding-layer print block with a protocol branch. Replace:

```rust
        s.push_str("--- FRI folding layers (x3 extension, last layer excluded) ---\n");
        for (i, st) in self.fri_steps.iter().enumerate() {
            s.push_str(&format!(
                "  step {:<2} drop={}  elements={:>4}  hashes={:>4}\n",
                i, st.bit_drop, st.elements, st.hashes
            ));
        }
```

with:

```rust
        match self.protocol {
            Protocol::Fri => {
                s.push_str("--- FRI folding layers (x3 extension, last layer excluded) ---\n");
                for (i, st) in self.fri_steps.iter().enumerate() {
                    s.push_str(&format!(
                        "  step {:<2} drop={}  elements={:>4}  hashes={:>4}\n",
                        i, st.bit_drop, st.elements, st.hashes
                    ));
                }
            }
            Protocol::Stir => {
                s.push_str("--- STIR rounds (per-round t_i, x3 extension) ---\n");
                for (i, r) in self.stir_rounds.iter().enumerate() {
                    s.push_str(&format!(
                        "  round {:<2} dim=2^{:<2} rate={:.4} reps={:>4} hashes/leaf={:>3}\n",
                        i,
                        (r.dimension as f64).log2().round() as u64,
                        r.rate,
                        r.repetitions,
                        r.hashes,
                    ));
                }
            }
        }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cargo test -p pil2-stark-setup --lib types::cells 2>&1 | tail -20`
Expected: PASS — all `types::cells` tests, including the two new ones.

- [ ] **Step 5: Commit**

```bash
git add setup/pil2-stark/src/types/cells.rs
git commit -m "feat(cells): STIR round report + STIR-beats-FRI folding test

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 5: Extend `cells_sweep_csv` with STIR columns (ground truth for the JS port)

**Files:**
- Modify: `setup/pil2-stark/src/types/cells.rs` (the `cells_sweep_csv` ignored test)
- Test: the ignored sweep itself (manual run)

**Interfaces:**
- Consumes: Task 3 (`protocol`, `schedule_found`, `stir_rounds`, `n_queries`).
- Produces: additional CSV columns `stir_t0,stir_total_queries,stir_total_cells,stir_schedule_found` so the HTML JS port can be diffed against Rust.

- [ ] **Step 1: Modify the sweep**

In `cells_sweep_csv`, after computing the FRI `rec` and `chk`, also compute the STIR estimate and append columns. Replace the header `println!` to add the STIR columns:

```rust
        println!("N,blowup,fold,stage1,n_queries,hashes_per_query,total_cells,needed_stage1,self_fits,prover_mem_gb,main_total_cells,main_fits,stir_t0,stir_total_queries,stir_total_cells,stir_schedule_found");
```

and inside the loop, after the FRI `rec`:

```rust
                let stir = estimate_linear_hash_cells(
                    &CellModelParams::recursion(n, blowup, fold_bits, stage1, hash).with_protocol(Protocol::Stir),
                );
                let stir_total_queries: u64 = stir.stir_rounds.iter().map(|r| r.repetitions).sum();
```

and extend the row `println!` format string + args with:

```rust
                    ,stir.n_queries, stir_total_queries, stir.total_cells, stir.schedule_found
```

(append these to the existing `println!("{},{},...", ...)` — add `,{},{},{},{}` to the format and the four values to the args).

- [ ] **Step 2: Run the sweep and capture ground truth**

Run: `cargo test -p pil2-stark-setup --lib types::cells::tests::cells_sweep_csv -- --ignored --nocapture 2>&1 | tee /tmp/claude-1003/-home-roger-pil2-proofman/6af33f04-56eb-4837-9d62-1c40ae55d8d7/scratchpad/cells_sweep.csv | tail -20`
Expected: a CSV with the new STIR columns populated; `stir_schedule_found` mostly `true`, possibly `false` for the hardest high-rate/small-N cells. Note 3-4 representative rows for the JS cross-check in Task 8.

- [ ] **Step 3: Commit**

```bash
git add setup/pil2-stark/src/types/cells.rs
git commit -m "test(cells): STIR columns in cells_sweep_csv (JS-port ground truth)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 6: HTML — Protocol selector control

**Files:**
- Modify: `docs/cells-model/recursion-cell-memory-model.html`

**Interfaces:**
- Produces: a `<select id="protocol">` in the `.controls` bar with options `fri` / `stir`, wired to the existing recompute path. A JS global `protocol` read from it.

- [ ] **Step 1: Inspect the existing controls + recompute wiring**

Run: `grep -n 'id="hash"\|id="metric"\|addEventListener\|function render\|function recompute\|select' docs/cells-model/recursion-cell-memory-model.html | head -40`
Identify (a) the markup pattern for an existing `<select>` (e.g. the hash selector), and (b) the function that recomputes/redraws the heatmap and the listeners that call it.

- [ ] **Step 2: Add the selector markup**

Next to the existing hash `<select>` in the `.controls` block, add (match the surrounding `.ctrl` wrapper markup exactly):

```html
<div class="ctrl">
  <label>Protocol</label>
  <select id="protocol">
    <option value="fri" selected>FRI</option>
    <option value="stir">STIR</option>
  </select>
</div>
```

Also add a **grinding-bits** slider (user requirement — configurable + shown). Mirror
the existing range-slider `.ctrl` markup in the file (the `<label>` shows the live value
in a `<b>` like the other sliders). Default 20, range 0–32:

```html
<div class="ctrl">
  <label>Grinding bits <b id="grindingVal">20</b></label>
  <input type="range" id="grinding" min="0" max="32" step="1" value="20">
</div>
```

- [ ] **Step 3: Wire both controls to recompute**

In the JS, where the other controls read their values and attach listeners, add the
protocol read and the grinding read + live label, both mirroring the existing
selectors/sliders. Adapt names to the file's actual render function:

```js
const protocolSel = document.getElementById('protocol');
function getProtocol() { return protocolSel.value; } // 'fri' | 'stir'
protocolSel.addEventListener('change', render); // same handler the hash selector uses

const grindingInput = document.getElementById('grinding');
const grindingVal = document.getElementById('grindingVal');
function getGrinding() { return parseInt(grindingInput.value, 10); } // 0..32
grindingInput.addEventListener('input', () => { grindingVal.textContent = grindingInput.value; render(); });
```

`getGrinding()` feeds BOTH the FRI `n_queries` calc and the STIR schedule in Task 7
(`protocol_security = 128 - getGrinding()`). If the page already has a FRI `n_queries`
JS that hardcodes 20 grinding bits, switch it to read `getGrinding()` too, so the FRI
heatmap also responds to the slider (shared knob, matching the Rust `grinding_bits` field).

- [ ] **Step 4: Verify it loads (no behavior change yet)**

Open `docs/cells-model/recursion-cell-memory-model.html` in a browser (or run a headless check). The Protocol dropdown appears; selecting STIR does not yet change the heatmap (wired in Task 7). No JS console errors.

Run (smoke check the file is still valid HTML / no obvious syntax break):
`node --check <(sed -n '/<script>/,/<\/script>/p' docs/cells-model/recursion-cell-memory-model.html | sed '1d;$d') 2>&1 | head -5 || echo "manual browser check"`
Expected: no syntax error (or fall back to manual browser check).

- [ ] **Step 5: Commit**

```bash
git add docs/cells-model/recursion-cell-memory-model.html
git commit -m "feat(cells-html): add FRI/STIR protocol selector

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 7: HTML — JS port of the STIR schedule, security gate, and cells

**Files:**
- Modify: `docs/cells-model/recursion-cell-memory-model.html`

**Interfaces:**
- Consumes: Task 6 (`getProtocol()`), the existing JS that computes a cell's `total_cells` / `n_queries` / folding for FRI.
- Produces: JS functions `stirRoundSchedule`, `stirNumRounds`, `stirRepetitions`, `stirAchievedBits`, `buildStirSchedule(n, blowupBits, foldBits)`, `stirCellTotals(...)`, and a branch in the per-cell compute that uses STIR when `getProtocol()==='stir'`.

- [ ] **Step 1: Add the STIR JS (port of `stir.rs`, pinned k)**

Add near the existing FRI/security JS:

```js
// --- STIR port (mirrors setup/pil2-stark/src/types/security/stir.rs) ---
const STIR_DEFAULT_GRINDING_BITS = 20;  // default seed; the live value comes from getGrinding()
const STIR_TARGET_BITS  = 128;
const STIR_STOPPING_DEG = 64;           // 2^6
const STIR_MAX_ROUNDS   = 24;
const EXT_DEG = 3;

function logInvRate(rate) { return -Math.log2(rate); }

function stirNumRounds(dimension, k) {
  if (k < 2 || dimension <= STIR_STOPPING_DEG) return null;
  let d = dimension, m = 0;
  while (d > STIR_STOPPING_DEG) { d = Math.floor(d / k); m++; if (m > STIR_MAX_ROUNDS) return null; }
  return m;
}

function stirRoundSchedule(d, rho0, k, m) {
  const out = [[d, rho0]];
  let dim = d, rate = rho0;
  for (let i = 0; i < m; i++) { dim = Math.floor(dim / k); rate = rate * 2 / k; out.push([dim, rate]); }
  return out;
}

function stirRepetitions(protocolSecurity, lir) {
  return Math.max(1, Math.ceil((2 * protocolSecurity) / lir));
}

// Returns { rounds:[{rate,dimension,fold,repetitions,fullyOpened}], achievedBits } or null.
// `grindingBits` is the live slider value (getGrinding()); defaults to 20.
function buildStirSchedule(n, blowupBits, foldBits, grindingBits = STIR_DEFAULT_GRINDING_BITS) {
  const k = 1 << foldBits;
  const dimension = 2 ** n;
  const rho0 = 1 / (2 ** blowupBits);
  const m = stirNumRounds(dimension, k);
  if (m === null) return null;
  const sched = stirRoundSchedule(dimension, rho0, k, m);
  for (const [dim, rate] of sched) if (dim < 1 || rate >= 1 || rate <= 0) return null;

  const protocolSecurity = STIR_TARGET_BITS - grindingBits; // e.g. 128-20 = 108
  const last = sched.length - 1;
  const rounds = [];
  for (let i = 0; i < sched.length; i++) {
    const [dim, rate] = sched[i];
    const lir = logInvRate(rate);
    if (!isFinite(lir) || lir <= 0) return null;
    const wanted = stirRepetitions(protocolSecurity, lir);
    const domainSize = Math.floor(dim / rate);
    let reps, fullyOpened;
    if (i !== last && wanted >= domainSize) { reps = Math.max(1, domainSize); fullyOpened = true; }
    else { reps = Math.max(1, wanted); fullyOpened = false; }
    rounds.push({ rate, dimension: dim, fold: k, repetitions: reps, fullyOpened });
  }
  // Achieved bits: weakest non-fully-opened round's (lir/2)*t + grinding.
  let weakest = Infinity;
  for (const r of rounds) {
    if (r.fullyOpened) continue;
    const bits = (logInvRate(r.rate) / 2) * r.repetitions + grindingBits;
    if (bits < weakest) weakest = bits;
  }
  const achievedBits = Math.floor(weakest);
  if (achievedBits < STIR_TARGET_BITS) return null;   // no 128-bit schedule for this cell
  return { rounds, achievedBits };
}
```

- [ ] **Step 2: Add the STIR cell-totals function**

```js
// STIR cells for one cell. `spongeRate` = 4 (Blake3) | 12 (Blake2).
// Returns { scheduleFound, t0, totalQueries, foldTotal, totalCells } using the
// SAME trace-hash count the FRI path computes (stage-0 invariant).
function stirCellTotals(n, blowupBits, foldBits, traceHashesPerQuery, nProofs, spongeRate, cellsPerPerm, grindingBits) {
  const sched = buildStirSchedule(n, blowupBits, foldBits, grindingBits);
  if (!sched) return { scheduleFound: false, t0: 0, totalQueries: 0, foldTotal: 0, totalCells: 0, rounds: [] };
  const t0 = sched.rounds[0].repetitions;
  let foldTotal = 0, totalQueries = 0;
  const leafHashes = Math.ceil((sched.rounds[0].fold * EXT_DEG) / spongeRate); // k same every round
  for (const r of sched.rounds) { foldTotal += r.repetitions * leafHashes; totalQueries += r.repetitions; }
  const totalHashes = (traceHashesPerQuery * t0 + foldTotal) * nProofs;
  return { scheduleFound: true, t0, totalQueries, foldTotal, totalCells: totalHashes * cellsPerPerm, rounds: sched.rounds };
}
```

- [ ] **Step 3: Branch the per-cell compute on protocol**

Find the JS that computes each heatmap cell's FRI `total_cells` (the function the heatmap loop calls per (N, blowup)). Wrap the folding/total part so that when `getProtocol()==='stir'` it calls `stirCellTotals(...)` with the already-computed `traceHashesPerQuery`, `nProofs`, `spongeRate` (from the hash selector), and `cellsPerPerm`. Use the STIR `totalCells` for the color/feasibility, and when `scheduleFound===false` mark the cell with the existing greyed/`over` class (reuse the over-budget styling) so it reads as "no schedule".

Adapt to the file's variable names; the shape is:

```js
let totalCells, scheduleFound = true, nQueriesShown, stirRounds = null;
if (getProtocol() === 'stir') {
  const s = stirCellTotals(N, blowup, foldBits, traceHashesPerQuery, nProofs, spongeRate, cellsPerPerm, getGrinding());
  totalCells = s.totalCells; scheduleFound = s.scheduleFound;
  nQueriesShown = s.t0; stirRounds = s.rounds;
} else {
  // existing FRI computation unchanged, but the FRI n_queries calc must read getGrinding()
  // (shared knob) instead of a hardcoded 20 — see Task 6 Step 3.
  totalCells = friTotalCells; nQueriesShown = friNQueries;
}
```

Ensure the cell-detail panel (click handler) shows the STIR round breakdown when `stirRounds` is set, and the "no schedule" state when `scheduleFound===false`.

- [ ] **Step 4: Smoke-test in browser**

Open the file, toggle Protocol → STIR. Expected: the heatmap recomputes; most cells show a STIR cell-cost; the hardest high-rate cells (if any) appear greyed as "no schedule"; clicking a cell shows the per-round breakdown. No console errors. FRI mode is unchanged from before.

- [ ] **Step 5: Commit**

```bash
git add docs/cells-model/recursion-cell-memory-model.html
git commit -m "feat(cells-html): JS STIR schedule + cells, protocol-branched compute

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 8: Cross-check JS STIR vs Rust sweep + README update

**Files:**
- Modify: `docs/cells-model/README.md`
- Reference: `/tmp/claude-1003/-home-roger-pil2-proofman/6af33f04-56eb-4837-9d62-1c40ae55d8d7/scratchpad/cells_sweep.csv` (from Task 5)

**Interfaces:**
- Consumes: Task 5 (Rust ground-truth CSV), Task 7 (JS STIR).
- Produces: a documented Rust↔JS agreement for STIR; README notes the toggle.

- [ ] **Step 1: Cross-check 3-4 cells**

Pick 3-4 `(N, blowup, fold=3, stage1=100, cells=5000)` rows from the Task-5 CSV (the sweep's default params: fold_bits=3, stage1=100, cells_per_blake3=5000, Blake3). In the browser console (with those exact knob values set: stage1=100, cells/perm=5000, Blake3, STIR), evaluate `stirCellTotals(N, blowup, 3, traceHashesPerQuery, 2, 4, 5000)` for each and compare `t0`, `totalQueries`, `totalCells`, `scheduleFound` against the CSV's `stir_t0,stir_total_queries,stir_total_cells,stir_schedule_found`.

Expected: exact match on `t0`, `totalQueries`, `scheduleFound`; `totalCells` exact (all integer math). If any differ, the JS port has a bug — fix in `recursion-cell-memory-model.html` (likely the `Math.floor` in `stirRoundSchedule`/`domainSize`, or the security gate `<` vs `<=`) and re-check before proceeding. Do NOT adjust the Rust side to match the JS.

- [ ] **Step 2: Update the README**

In `docs/cells-model/README.md`, under "What it shows" add the Protocol toggle, and under "Where the numbers come from" add the STIR note. Insert after the "Live knobs" paragraph:

```markdown
- **Protocol** toggle (**FRI** vs **STIR**, ePrint 2024/390). STIR keeps the same
  stage-0 (trace/Q) commitment hashing as FRI, but folds with per-round query
  counts `t_i` that shrink as the rate improves: folding cells are
  `Σ_round t_i · ceil(k·3 / rate)` and stage-0 is opened `t_0` times (round-0
  reps) rather than the flat FRI `n_queries`. STIR folds by the same factor
  `k = 2^fold` down to degree `2^6` as FRI (apples-to-apples). Cells with no
  128-bit STIR schedule are greyed as "no schedule".
```

and after the `cells_sweep_csv` code block add:

```markdown
The STIR columns (`stir_t0, stir_total_queries, stir_total_cells,
stir_schedule_found`) in that CSV are the ground truth the in-browser STIR port is
checked against; the JS reproduces them exactly across the swept range. Grinding is
pinned to 20 bits for both protocols (protocol security 108), matching the FRI path.
```

- [ ] **Step 3: Commit**

```bash
git add docs/cells-model/README.md
git commit -m "docs(cells-html): document FRI/STIR toggle + Rust-JS STIR cross-check

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Final verification

- [ ] **All Rust cells tests pass:**

Run: `cargo test -p pil2-stark-setup --lib types::cells 2>&1 | tail -15`
Expected: PASS, no failures.

- [ ] **Security calculator tests still pass (untouched, sanity):**

Run: `cargo test -p pil2-stark-setup --lib types::security 2>&1 | tail -15`
Expected: PASS.

- [ ] **HTML cross-check recorded** (Task 8 agreement noted in README).

- [ ] **No clippy regressions on the changed file:**

Run: `cargo clippy -p pil2-stark-setup --lib 2>&1 | grep -A3 'cells.rs' | head -20`
Expected: no new warnings attributable to the cells changes.
