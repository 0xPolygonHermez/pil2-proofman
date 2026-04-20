# pil2-proofman — GPU Optimization Plan

Companion to [ARCHITECTURE.md](ARCHITECTURE.md). This is an initial,
revisable plan — expect the measurement step (§2) to reshape the work
streams in §3.

---

## 1. Context

- **Target build:** `cargo build --release --features gpu` — the GPU
  path exclusively.
- **Target deployment:** multi-GPU (typical ZisK worker is 2-8 GPUs).
- **Available hardware for this work:** a single GPU. Implications:
  - Everything intra-worker is directly measurable: stream pool
    behavior, VRAM pressure, kernel overlap, witness-regeneration
    cost, AIR-constant reload cost.
  - Everything inter-GPU is *not* directly measurable and must be
    reasoned about from the architecture plus single-GPU data;
    validation happens when the multi-GPU box is available.
- **Target workload:** ZisK ethereum-block proving. Fibonacci remains
  a smoke test; perf numbers must come from a zisk input (small input
  for iteration, one full block for truth).
- **Reference point for kernel-level work already done:** the PLONK
  GPU pipeline tuned over Phases 1-16 in `memory/MEMORY.md` — Group C
  (basic + recursive STARKs) and Group B (commit) kernels have **not**
  received the same treatment, so there is plausibly headroom at both
  the orchestration layer and inside `gen_proof_c` / `commit_witness_c`.

### 1.1 Thesis

From ARCHITECTURE §3.8 (temporal structure): Group C is ~58 % of wall
time and Group B ~17 %. The current scheduler is **reactive/greedy**:

- Basic proofs dispatch in whatever order the witness pipeline emits
  instances.
- Recursive stages fire from completion callbacks (`recursive_tx`)
  the moment a predecessor finishes.
- Witness buffers are discarded at the end of Group B and
  **recomputed on demand** when Phase 3 picks each instance up.
- Streams are a fungible pool: `n_streams_non_recursive` for basic,
  the remainder for recursive; no affinity per AIR type.

That design is robust and memory-frugal but leaves three concrete
things on the table:

1. **Unnecessary critical-path work.** Witness recomputation is
   hidden behind GPU proving, but the CPU cycles it consumes could
   be feeding recursive-witness generation (Circom) instead. And any
   instance whose recompute blocks its dependent aggregation lands
   directly on the critical path.
2. **No global view of dependencies.** The full DAG is known the
   moment Phase 0 finishes — every instance, every compressor/rec1
   stage, the Recursive2 tree per airgroup. A greedy scheduler that
   ignores this cannot minimize makespan; a DAG-aware one can.
3. **Stream-level thrash.** Streams process instances of mixed AIR
   types, which means repeatedly swapping AIR-specific constants
   (trees, stark-info, setup data) in and out of the address they
   are bound to. Pinning AIRs to streams (or to GPUs, in the
   multi-GPU case) eliminates those reloads.

The proposed direction: **move from reactive scheduling to a
DAG-driven scheduler** that plans the full Group B + C tree up
front, minimizes the critical path, persists witnesses instead of
recomputing them, and assigns work to streams/GPUs with AIR-type
affinity and a VRAM budget.

---

## 2. Step 0 — Measurement baseline

Before writing any scheduler, establish what the current pipeline
actually does on a realistic workload. This is the step whose output
will ratify or redirect §3.

### 2.1 Captures to collect

| # | Workload | Config | Tool | Purpose |
|---|----------|--------|------|---------|
| M1 | Small zisk input (minimal-ish ethereum block) | default, `-vv` | `proofman-cli` logs | Per-phase wall clock; tail-instance identification; Group A/B/C/D proportions at v0.17.0 |
| M2 | Same as M1 | default | Nsight Systems | Stream occupancy during Group C; identify whether basic streams or recursive streams are the idle side |
| M3 | Same as M1 | vary `n_streams_non_recursive` ±2 from default | `-vv` + Nsight | Sensitivity: is the current split right for this workload on 1 GPU? |
| M4 | Same as M1 | `--minimal-memory` on vs off | `-vv` | Cost of witness regeneration vs memory-kept witnesses — the core §3.2 question |
| M5 | Full ethereum block | default + `--minimal-memory` (required for VRAM) | `-vv` | Ground-truth Group proportions; calibrates all projections |

### 2.2 Metrics to derive from the captures

- **Per-group wall time** — does v0.17.0 still show A ≈ 18 %, B ≈ 17 %,
  C ≈ 58 %, D ≈ 7 % shape?
- **Per-AIR distribution** in Group C — how much time per AIR type,
  how many instances each, Recursive2 tree depth per airgroup.
- **Stream occupancy histogram** — fraction of Group C where
  (basic streams busy, recursive streams busy) is (1,1), (1,0), (0,1),
  (0,0). The (1,0) / (0,1) fractions are pure orchestration wins.
- **VRAM high-water mark** — `MemoryHandler` and
  `MemoryHandlerRecursive` peak occupancy; any dry-out stalls (waits
  on `to_be_released_buffer`).
- **Constant-reload events** — how often `gen_proof_c` swaps AIR
  constants on a given stream. If unmeasurable from `-vv`, add a
  counter.
- **Witness regeneration time** per instance — already implicitly in
  the Phase 3 window of `-vv`; isolate it by comparing M4's on/off.

### 2.3 Exit criteria for Step 0

- Current-state Group proportions, stream idle rate, VRAM high-water,
  constant-reload count, and witness-regen share are **all numbers**.
- We know which §3 work stream the biggest lever belongs to before
  committing code.

---

## 3. Proposed work streams

Four parallel axes. They interact (3.1 needs 3.2; 3.3 constrains 3.1
and 3.4), so §4 sequences them.

### 3.1 DAG-driven scheduler

**Idea.** After Phase 0, the full dependency graph is known:

```
   [per-instance]         [per-instance]         [per-instance]        [per-airgroup tree]
   Witness_i  ─▶ Basic_i ─▶ (Compressor_i?) ─▶ Recursive1_i ─▶ Rec2_level0_k (arity 3)
                                                                   │
                                                                   ▼
                                                            Rec2_level1_k
                                                                   │ …
                                                                   ▼
                                                            Airgroup_root
                                                                              ─▶ VadcopFinal
```

Build this DAG explicitly, annotate each node with:

- Estimated duration (from Step 0 per-AIR measurements).
- Required witness (which instance / which Circom input).
- VRAM footprint while running.
- Whether it is a GPU task (basic / recursive STARK) or CPU task
  (witness gen / Circom witness).

Then schedule to **minimize critical path** subject to:

- VRAM budget per GPU.
- Stream count per GPU.
- AIR-constant affinity (§3.3).
- Witness availability (§3.2).

**Concretely.** Replace the callback-driven dispatch in
[proofman.rs:2052-2198](proofman/src/proofman.rs#L2052-L2198) and
[proofman.rs:2280-2448](proofman/src/proofman.rs#L2280-L2448) with a
scheduler that:

1. Precomputes a topological order weighted by longest-path-to-sink
   (critical-path length).
2. Maintains a ready queue ordered by that weight.
3. When a stream frees up, pops the highest-priority task whose
   predecessors are done and whose resources fit.
4. Eagerly schedules recursive aggregations the moment their 3
   inputs exist (same as today) but **chooses which 3** to aggregate
   first to shorten the critical path (today: FIFO within an
   airgroup).

**Open design questions.**

- Granularity: keep the Recursive2 arity of 3, or let the scheduler
  choose arity per level based on slack?
- Runtime vs static: fully static schedule (computed once) or
  priority-driven dynamic (recomputed as actuals replace estimates)?
  Dynamic is more robust to estimate error.
- How to feed the scheduler duration estimates without measuring
  every run — per-AIR lookup table refreshed opportunistically?

**Files most affected.** [proofman/src/proofman.rs](proofman/src/proofman.rs)
(dispatch + callback loops), [proofman/src/recursion.rs](proofman/src/recursion.rs)
(aggregation entry points), likely a new `scheduler.rs` alongside.

### 3.2 Witness storage hierarchy (avoid recompute)

**Idea.** Recomputing witnesses in Phase 3 costs CPU cycles that
could feed the recursive pipeline. Replace recompute with a tiered
store:

| Tier | Medium | When to use | Cost |
|------|--------|-------------|------|
| T1 | Keep in system RAM | when RAM budget allows | zero recompute, zero I/O |
| T2 | Persist to fast local disk (NVMe); async prefetch before Phase 3 schedules the instance | when RAM is tight but disk bandwidth ≥ consumption rate | one serialized write + one async read per witness |
| T3 | Recompute (current behavior) | fallback when neither fits | witness CPU cost paid again |

**Why it matters on multi-GPU.** T3 cost scales with *GPU
throughput* — faster GPUs starve on CPU witness regeneration first.
T2 turns the bottleneck into disk bandwidth (3-7 GB/s on NVMe,
parallelizable across drives) and leaves CPU cores for Circom
recursive-witness generation.

**Concretely.**

- Change the `MemoryHandler` release path
  ([proofman.rs memory handler code paths](proofman/src/proofman.rs)
  plumbs through `to_be_released_buffer` /
  `release_buffer`) to have three dispositions: *keep resident*,
  *spill to disk*, *free and recompute later*. The policy is
  per-instance based on (witness size, remaining uses).
- Add an async prefetch queue wired into the DAG scheduler: when a
  task is within N positions of ready and its witness is on disk,
  kick off the read.
- Disk format: just the raw `Vec<F>` backing the instance buffer
  with a small header (AIR id, domain size, checksum). Keep it flat
  so mmap + `cudaMemcpy` works end-to-end without copies.

**Open design questions.**

- Where to stage on the way from disk → GPU: pinned host buffer?
  GPUDirect Storage if available? For the common-case single-NVMe
  host, pinned host + `cudaMemcpyAsync` is probably enough.
- What percentage of witnesses to spill vs keep? Needs M4 cost data.
- Interplay with `minimal_memory`: this supersedes it; the new knob
  is a target RAM budget, not an on/off flag.

**Files most affected.** `MemoryHandler` / `MemoryHandlerRecursive`
in [common/](common/), witness dispatch in
[proofman.rs:3585-3710](proofman/src/proofman.rs#L3585-L3710), and
the storage backend itself (new module).

### 3.3 Stream / GPU affinity by AIR type

**Idea.** Group instances of the same AIR on the same stream (or
small cluster of streams) so AIR constants load once and stay put.
On multi-GPU: pin each AIR type to a subset of GPUs so constants
live on the right device.

**Concretely.**

- Today, const data per AIR is loaded via `load_device_const_pols`
  / `load_device_setups` at startup, kept resident on device, and
  indexed by `(airgroup_id, air_id)`. Constants aren't *reloaded*
  per proof on a single GPU, so the win on 1 GPU is smaller than
  the name suggests — the actual cost being paid is **cache
  thrash** across a stream's kernel sequence (L2 / shared-memory
  residency of setup data).
- On multi-GPU the story changes: each GPU has its own copy of
  the constants for all AIRs; an AIR routed to a cold GPU pays a
  full warmup, and the scheduler may route the next instance of
  the same AIR to yet another cold GPU. AIR-to-GPU affinity fixes
  that and is a structural win, not a cache one.
- Scheduler change: tag each ready GPU task with its
  `(airgroup_id, air_id)`; prefer streams on the GPU that has most
  recently run the same tag. On a tie, prefer the less-loaded GPU
  (§3.4).

**Open design questions.**

- Full partition (each AIR pinned to exactly one GPU) vs. soft
  affinity (preferred, but allowed to spill)? Full partition
  creates load imbalance when one AIR dominates; soft affinity
  needs tie-breaking.
- Does Recursive1 / Recursive2 constant data dominate basic-AIR
  constant data? If yes, affinity matters more for recursive
  streams than for basic.

**Files most affected.** Setup load paths
([common/](common/) + `proofman_starks_lib_c`), scheduler (§3.1),
and potentially the stream allocation in
`ProofMan::new` ([proofman.rs:1595-1601](proofman/src/proofman.rs#L1595-L1601)).

### 3.4 Multi-GPU load balancing & occupancy

**Idea.** Maximize the number of concurrent GPU tasks subject to
VRAM and then balance them across GPUs.

**Concretely.**

- **VRAM budget as first-class.** Each GPU has a budget; the
  scheduler tracks in-flight-task footprint and blocks admission
  of new tasks when it would exceed budget. This lets us raise
  stream counts without OOM risk.
- **Work-stealing between GPUs.** If GPU-A's queue empties before
  GPU-B's, A steals from B's ready list (subject to AIR affinity
  penalty). Simple rule first: steal only when idle.
- **Occupancy tuning.** Inside `gen_proof_c`, kernel occupancy is
  a kernel-level concern (MEMORY.md territory). From the
  orchestrator side the lever is *how many concurrent proofs* —
  more streams = more per-kernel contention, fewer = underutilized
  SMs. Find the right count per GPU from M3 and from multi-GPU
  validation.
- **Aggregator placement.** In the ZisK distributed layer, Phase 5
  + 6 collapse onto one GPU (ARCHITECTURE §3.8 Group D).
  Optimization here is pure scheduling — overlap the tail of Group
  C with the first aggregation steps of Group D on the same
  worker, so Group D's single-GPU time partly hides behind the
  still-running Group C on other GPUs.

**Open design questions.**

- VRAM accounting: static per-task estimate (upfront) or dynamic
  (read CUDA free-memory each admission)? Dynamic is safer but
  racy; static plus a safety margin is probably enough.
- Where work-stealing lives: in the scheduler (§3.1) or as a
  separate balancer? Likely merged into §3.1.

**Files most affected.** Scheduler (§3.1), GPU setup in
`ProofMan::new`, any FFI that binds a proof to a device.

---

## 4. Sequencing

The four work streams are not independent. Proposed order:

1. **Step 0 — Measurement** (§2). Hard prerequisite for everything.
2. **3.2 — Witness storage hierarchy.** Can land behind the current
   scheduler with a flag (`--witness-policy={recompute,spill,keep}`).
   Gives immediate wins if M4 shows recompute is non-trivial, and
   eliminates a confounding variable before we start tuning the
   scheduler.
3. **3.1 — DAG scheduler.** The core change. Depends on 3.2 because
   scheduling benefits shrink if recompute is still in the critical
   path.
4. **3.3 — AIR affinity** and **3.4 — VRAM/occupancy tuning.** Layer
   on top of 3.1 as policies inside the scheduler. 3.3 is probably
   low-risk and can land early; 3.4 needs multi-GPU hardware to
   validate meaningfully.
5. **Cross-cutting: scheduler v2 with multi-GPU validation.** Only
   once a multi-GPU box is available.

Each step ends with the same measurement set as §2 so we can report
Group-proportion changes against a known baseline.

---

## 5. Validation strategy

- **Correctness.** Every change must keep `verify_proof` green on
  fibonacci + zisk small input + a held-out ethereum block. Proof
  bytes should be bit-identical unless the change deliberately
  reorders aggregation (in which case: re-verify each time).
- **Regression harness.** A scripted run of M1 + M5 with
  before/after per-group timings, stream occupancy, and VRAM
  high-water. Land it as a CI-adjacent script, not necessarily in
  CI itself.
- **No-regression on small inputs.** Fibonacci and zisk small input
  should not get slower by more than a small epsilon — the new
  scheduler's overhead has to be demonstrably amortized.

---

## 6. Open questions & risks

- **Estimate error in the DAG scheduler.** If per-task duration
  estimates are wrong by a lot, static scheduling can be worse
  than greedy. Dynamic re-prioritization mitigates this but adds
  complexity. Decide after Step 0.
- **Witness size on disk for a full block.** Unknown until M5 —
  if the total is tens of TBs, T2 (disk spill) stops being
  reasonable and the story becomes RAM-only + recompute for the
  tail.
- **Multi-GPU validation gap.** Key decisions (3.4, partially 3.3)
  cannot be closed on a 1-GPU box. Mitigation: land them behind
  feature flags, validate on the multi-GPU machine when available,
  roll forward.
- **Interaction with the ZisK coordinator.** The zisk distributed
  layer assumes per-worker `AggProofs` arrive in a certain shape.
  Any scheduler change that reorders aggregations must preserve
  that interface (ARCHITECTURE §9).
- **Don't compete with MEMORY.md kernel work.** If Group C stalls
  turn out to be kernel-bound (a single kernel serialising a
  stream), orchestration won't help and the work should redirect
  to kernel tuning in [pil2-stark/](pil2-stark/). Step 0 tells us
  which one.

---

## 7. Short-list of the next concrete actions

1. Pick a representative small zisk input (sub-minute on 1 GPU).
2. Run M1, M2, M4 — collect the five metrics in §2.2.
3. Decide whether the measurement confirms the orchestration-first
   thesis, and iterate this document.
4. If confirmed, start 3.2 (witness storage) behind a flag so the
   baseline remains runnable.
