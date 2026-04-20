# Session handoff — pil2-proofman GPU optimization

Brief to load into a new Claude Code session on the new server so it
can pick up where this one left off. Read [ARCHITECTURE.md](ARCHITECTURE.md)
and [OPTIMIZATIONS.md](OPTIMIZATIONS.md) as the primary references —
this file is the *index + decisions log + open questions*, not a
re-statement of those documents.

---

## 1. What we're doing

Optimizing the GPU proving pipeline of `pil2-proofman`. Two artifacts
were produced this session:

- **[ARCHITECTURE.md](ARCHITECTURE.md)** — end-to-end map of the
  proof workflow (Phases 0-6), the `ProofMan<F>` orchestrator, data
  flow, threading model, GPU integration with `pil2-stark`, the
  ZisK distributed mode, and a temporal/concurrency-group view of
  the pipeline (Groups A/B/C/D) anchored on a 4-worker × 2-GPU
  v0.14.0 trace (1.2 / 1.1 / 3.8 / 0.5 s out of 6.6 s).
- **[OPTIMIZATIONS.md](OPTIMIZATIONS.md)** — initial plan: a
  measurement Step 0 followed by four work streams
  (DAG scheduler, witness storage hierarchy, AIR/GPU affinity,
  multi-GPU load balancing & occupancy), plus sequencing,
  validation strategy, and open risks.

Both documents are the source of truth — the new session should read
them rather than rely on this summary.

---

## 2. Scope decisions made this session

| Decision | Detail |
|----------|--------|
| **Build target** | GPU only (`cargo build --release --features gpu`). CPU info kept where it already exists for context. |
| **Phases in scope** | 0-6. Phase 7 (SNARK wrapper / Plonk/Fflonk / RecursiveF) is **out of scope** and explicitly marked so in ARCHITECTURE.md. |
| **Workload for perf** | ZisK ethereum block (full block for ground truth, small zisk input for iteration). Fibonacci is a 10-second smoke test only — *not* a perf decision target. |
| **Hardware available now** | Single GPU. The new server may differ — confirm at the start of the next session. |
| **Hardware target** | Multi-GPU (typical ZisK worker is 2-8 GPUs). |
| **Branch** | `pre-develop-0.17.0` (verify on new server with `git status`). |

---

## 3. Working thesis

From ARCHITECTURE §3.8: Group C (Phases 3 + 4 — basic STARKs +
recursive tree) is ~58 % of wall time. Current scheduler is
**reactive/greedy**: callback-driven dispatch, witnesses recomputed
on demand at the start of Phase 3, no global view of dependencies, no
AIR-to-stream affinity.

Proposed direction: **move from reactive to DAG-driven scheduling**.
The full dependency graph is known after Phase 0; a critical-path
weighted scheduler with persistent witnesses and AIR/GPU affinity
should outperform the current greedy approach on multi-GPU.

---

## 4. Honest expectation calibration

Discussed and aligned this session:

- **Single-GPU expected gain: small to modest** (single-digit %, up to
  ~15 % if measurement shows witness recompute is consuming CPU
  cycles that could feed recursive Circom witness generation). Most
  proposed levers (work stealing, AIR-to-GPU pinning, aggregator
  placement) are multi-GPU multipliers that collapse on one device.
- **Multi-GPU expected gain: substantially larger** but unmeasurable
  here.
- **Risk:** if Group C turns out to be **kernel-bound** (a single
  hot kernel serialising a stream), orchestration won't help and
  effort should redirect to kernel work in [pil2-stark/](pil2-stark/)
  — same playbook as the PLONK Phases 1-16 work tracked in
  `memory/MEMORY.md` (took PLONK from ~6 s to ~1.12 s).
- **Higher-EV path for single-GPU wall-clock specifically:**
  kernel-level optimisation inside `gen_proof_c` /
  `commit_witness_c`, *not* orchestration. Group C basic and
  recursive STARK kernels haven't received the same kernel-level
  treatment that PLONK has.

---

## 5. Step 0 — what to do first (per OPTIMIZATIONS.md §2)

Before any code change, collect:

| # | Workload | Config | Tool |
|---|----------|--------|------|
| M1 | Small zisk input | default + `-vv` | `proofman-cli` logs |
| M2 | Same as M1 | default | Nsight Systems |
| M3 | Same as M1 | vary `n_streams_non_recursive` ±2 | `-vv` + Nsight |
| M4 | Same as M1 | `--minimal-memory` on vs off | `-vv` |
| M5 | Full ethereum block | + `--minimal-memory` | `-vv` |

Five metrics from these:

1. Per-group wall-time (does v0.17.0 still match A 18 % / B 17 % /
   C 58 % / D 7 %?).
2. Per-AIR distribution in Group C, Recursive2 tree depth per airgroup.
3. Stream-occupancy histogram during Group C — fraction (basic busy,
   recursive busy) is (1,1)/(1,0)/(0,1)/(0,0). The (1,0) and (0,1)
   slices are pure orchestration wins.
4. VRAM high-water on `MemoryHandler` and `MemoryHandlerRecursive`;
   any dry-out stalls.
5. Witness regeneration share (M4 on/off comparison).

**Hard exit criterion for Step 0:** all five metrics are *numbers*
before any §3 work stream begins.

---

## 6. Open questions to drive the discussion next

1. On a current GPU run (v0.17.0), does the Group breakdown still
   match the v0.14.0 trace?
2. During Group C, are both stream pools saturated, or is one side
   waiting? → dictates whether to rebalance
   `n_streams_non_recursive` or target the slow side.
3. Does the recursive witness memory pool ever run dry? → if yes,
   recursive proving stalls invisibly and the fix is sizing, not
   kernel tuning.
4. Is Recursive2 arity = 3 still right at current GPU capacity? →
   would 4 or 5 better amortize per-proof overhead?
5. In Group D, is the limiter aggregation compute or idle time
   waiting on peer workers' `AggProofs`?
6. **Precondition for any §3 work:** is Group C orchestration-bound
   or kernel-bound? Decided by the stream-occupancy histogram (M2).

---

## 7. The four optimization work streams (summary, see OPTIMIZATIONS §3)

| # | Name | Single-GPU value | Multi-GPU value | Depends on |
|---|------|------------------|-----------------|-----------|
| 3.1 | DAG-driven scheduler (replaces reactive callback dispatch) | small (critical-path tail only) | large | needs 3.2 to be meaningful |
| 3.2 | Witness storage hierarchy (T1 RAM / T2 NVMe spill / T3 recompute) | only if M4 shows recompute is non-trivial CPU contention | universally helpful | independent — land first behind a flag |
| 3.3 | AIR-type stream/GPU affinity | small (cache effect only) | large (avoids cold-GPU constant warmup) | rides inside 3.1 |
| 3.4 | Multi-GPU load balancing, VRAM admission, work stealing | mostly inert | the multi-GPU multiplier | rides inside 3.1; needs multi-GPU box to validate |

Proposed sequence: **Step 0 → 3.2 → 3.1 → 3.3 → 3.4**.

---

## 8. Repo layout & key file anchors

Working trees:

- `/home/rick/pil2-proofman/` — primary repo, branch
  `pre-develop-0.17.0`. Contains `pil2-stark` as a submodule.
- `/home/rick/zisk/` — peer repo. Distributed orchestration in
  `/home/rick/zisk/distributed/` (covered in ARCHITECTURE §9).

Key files (paths and the section of ARCHITECTURE.md they're explained
in):

- [proofman/src/proofman.rs](proofman/src/proofman.rs) — central
  orchestrator (4137 lines). Major regions: `_generate_proof`
  (1747-2705), reactive callback dispatch (2052-2198, 2280-2448),
  `gen_proof` (3740-3814), `outer_aggregations` (2997-3070).
  Architecture §2 + §3.
- [proofman/src/recursion.rs](proofman/src/recursion.rs) — Compressor
  / Recursive1 / Recursive2 driver, `generate_vadcop_final_proof`.
  `N_RECURSIVE_PROOFS_PER_AGGREGATION = 3` is at line 29.
  Architecture §3 (Phases 4-6).
- [common/](common/) — `ProofCtx`, `SetupCtx`, `SetupsVadcop`,
  `MemoryHandler`, `MemoryHandlerRecursive`, `MpiCtx`. Architecture §4.
- [witness/](witness/) — `WitnessLibrary`, `WitnessManager`.
- [provers/starks-lib-c/](provers/starks-lib-c/) — FFI to
  `pil2-stark`: `gen_proof_c`, `commit_witness_c`, GPU stream calls.
- [pil2-stark/src/starkpil/](pil2-stark/src/starkpil/) — STARK GPU
  kernels (where any kernel-level optimisation work would land).
- ZisK distributed: `/home/rick/zisk/distributed/crates/{coordinator,worker,grpc-api,common}/src/`.

---

## 9. Build & run commands

From `memory/MEMORY.md` (replicate to new server's auto-memory if
possible — see §11):

```bash
# Build the GPU library + CLI
cd pil2-stark && make -j starks_lib_gpu
cd ..
touch provers/starks-lib-c/build.rs
cargo build --release --features gpu --bin proofman-cli

# Fibonacci smoke test (NOT for perf)
cargo run --release --features gpu --bin proofman-cli -- \
  prove-snark \
  -k examples/fibonacci-square/build/provingKeySnark \
  -p examples/fibonacci-square/build/proofs/vadcop_final_proof.bin \
  -vv -o tmp

# Verify
cargo run --release --features gpu --bin proofman-cli -- \
  verify-snark -p tmp/snark_proof.bin \
  -k examples/fibonacci-square/build/provingKeySnark/final/final.verkey.json
```

For ZisK ethereum-block runs, use the `zisk-coordinator` /
`zisk-worker` binaries from `/home/rick/zisk/distributed/`. See
[that repo's README](../zisk/distributed/README.md) and ARCHITECTURE
§9.2 for the gRPC orchestration details.

---

## 10. Recent git activity (context for any drift since session start)

Branch `pre-develop-0.17.0`, `git status` at session start:

```
?? .claude/
?? ARCHITECTURE.md          (created this session)
?? OPTIMIZATIONS.md          (created this session)
?? conversation.md           (this file)
```

Recent commits at session start:

```
0b02a210 Feature/clean goldilocks cpu (#465)
691b3781 Reducing memory required plonk/fflonk setup (#464)
fdce210b Fix plonk multigpu (#462)
6600fb09 GPU kernel optimizations: cudaGraph, expression shared memory temps, fold/NTT fixes (#452)
131fbe58 Fix build.rs bug (#461)
```

If the new server's checkout has moved past these, re-run the
ARCHITECTURE-mapping check on any large diffs to `proofman.rs`
before relying on the line numbers in ARCHITECTURE.md.

---

## 11. Auto-memory note

This session ran with auto-memory at
`/home/rick/.claude/projects/-home-rick-pil2-proofman/memory/`. The
**MEMORY.md** there contains:

- Build commands (replicated above for portability).
- Critical pitfalls accumulated over the PLONK GPU work (Phases 1-16):
  Polynomial constructor zeroing, GPU OOM patterns, CUDA stream sync
  with NTT, `blindCoefficients` dual modification, fused gather kernel
  bounds, GPU `Fr::toMontgomery` calling convention, GPU poly eval
  blinding correction, `d_scanWork` overflow, the `PLONK_GPU_TIMING`
  flag.
- PLONK GPU timing breakdown (~1.12 s, broken down by round).
- CPU memory cleanup notes (288 MB saved).
- References to two related project files:
  `project_goldilocks_gpu.md` and `project_poseidon_v1_port.md`.

If the new server doesn't share this directory, copy it across
verbatim before starting — most of those pitfalls are not in the
codebase comments and re-discovering them costs hours.

---

## 12. Suggested first message on the new session

Something like:

> "Pick up the pil2-proofman GPU optimization work. Read
> conversation.md, ARCHITECTURE.md, and OPTIMIZATIONS.md in
> /home/rick/pil2-proofman/. We're at Step 0 of OPTIMIZATIONS §2 —
> need to pick a small zisk input and run M1 / M2 / M4. Confirm:
> (a) GPU available on this box, (b) zisk repo present at
> ../zisk, (c) which zisk input to use as the small-iteration
> workload."

That re-loads the context, confirms the immediate next action, and
flags the three things that depend on the new server's state.
