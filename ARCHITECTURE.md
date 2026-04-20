# pil2-proofman — Architecture Overview

This document provides an end-to-end map of the proof-generation workflow
orchestrated by `proofman`. It is intended as the reference map before
planning optimizations: every phase is described with its purpose,
orchestrator function, file/line anchors, inputs, outputs, and relevant
data structures.

All line references target the current `pre-develop-0.17.0` branch.

> **Scope note.** From this revision onward the document focuses on the
> **GPU execution path** (`cargo build --release --features gpu`).
> CPU information is retained where it already exists for context, but
> concurrency analysis, performance discussion, and optimization plans
> all assume the GPU build: device buffers are the primary memory,
> `n_streams` partitions real hardware streams across GPUs, and
> `gen_proof_c` is asynchronous with callback-driven completion.

---

## 1. Workspace layout

Workspace members ([Cargo.toml](Cargo.toml)):

| Crate | Role |
|-------|------|
| [cli/](cli/) | `proofman-cli` binary — parses args, loads keys, dispatches to `ProofMan`. |
| [proofman/](proofman/) | Central orchestrator. Contains `ProofMan<F>`, the full proving pipeline, and the recursion driver. |
| [common/](common/) | Shared context types: `ProofCtx`, `SetupCtx`, `SetupsVadcop`, `Setup`, `MemoryHandler`, `MpiCtx`, `Proof`, `ProofType`, `ProofOptions`. |
| [witness/](witness/) | `WitnessLibrary` trait + `WitnessManager`. Per-project witness libs are loaded as dynamic libraries and plugged in through this crate. |
| [provers/starks-lib-c/](provers/starks-lib-c/) | FFI bindings to the C++/CUDA STARK prover in [pil2-stark/](pil2-stark/) (the submodule). Entry points: `gen_proof_c`, `commit_witness_c`, `calculate_witness_expressions_c`, GPU stream management. |
| [hints/](hints/) | Hint expression evaluator (used during witness / constraint evaluation). |
| [verifier/](verifier/) | Rust verifiers: `verify_recursive2`, `verify_vadcop_final`, `verify_vadcop_final_compressed`. |
| [curves/](curves/), [fields/](fields/) | Finite-field / elliptic-curve primitives (Goldilocks, Quintic extension, Poseidon). |
| [util/](util/) | `VadcopFinalProof` type, timers, buffer helpers. |
| [pil2-components/](pil2-components/) | Standard PIL2 components (std lib + example components). |
| [pil2-stark/](pil2-stark/) | C++/CUDA STARK and Circom witness calculator. Built via `make -j starks_lib_gpu` and linked by `starks-lib-c`. (Also hosts a PLONK/FFLONK prover used by the SNARK-wrapping path, which is out of scope here.) |
| [soundness/](soundness/), [pilout/](pilout/), [macros/](macros/) | Supporting tooling. |
| [examples/](examples/) | Reference programs (`fibonacci-square`, `test-recursive`). |

The CLI lives in [cli/src/main.rs](cli/src/main.rs) with commands under
[cli/src/commands/](cli/src/commands/). The relevant ones for the
proving flow are [prove.rs](cli/src/commands/prove.rs),
[verify_constraints.rs](cli/src/commands/verify_constraints.rs),
and [setup.rs](cli/src/commands/setup.rs).
(`prove_snark.rs` exists for the optional SNARK-wrapping pass but is
out of scope in this document.)

---

## 2. Top-level orchestration

### 2.1 Entry point

The CLI `prove` command (see [cli/src/commands/prove.rs](cli/src/commands/prove.rs))
builds a `ProofMan<Goldilocks>` and calls either
`verify_proof_constraints()` (debug mode) or `generate_proof()`
(production). The returned `ProvePhaseResult::Full` contains the
`VadcopFinalProof`, which is the final output considered in this
document. (An optional SNARK-wrapping pass via `SnarkWrapper` exists
but is out of scope here.)

### 2.2 `ProofMan<F>` — the orchestrator

Defined in [proofman/src/proofman.rs:209-267](proofman/src/proofman.rs#L209-L267).
Key state:

- **Contexts.** `pctx: Arc<ProofCtx<F>>`, `sctx: Arc<SetupCtx<F>>`,
  `setups: Arc<SetupsVadcop<F>>` (compressor / rec1 / rec2 / vadcop-final
  / recursive-f), `mpi_ctx: Arc<MpiCtx>`, `wcm: Arc<WitnessManager<F>>`.
- **GPU config.** `gpu_params: ParamsGPU`, `n_streams`,
  `n_streams_non_recursive`, `n_gpus`. Streams are partitioned between
  the basic-proof pool (`n_streams_non_recursive`) and the recursive
  pool (remaining).
- **Buffers.** `aux_trace`, `const_pols`, `const_tree`,
  `prover_buffer_recursive`, plus GPU-side device buffers addressed by
  `pctx.get_device_buffers_ptr()`. Host allocations are elided under
  `feature = "gpu"` since the data lives on the device.
- **Proof storage.** `proofs` (basic), `compressor_proofs`,
  `recursive1_proofs`, `recursive2_proofs` (per-airgroup vec),
  `recursive2_proofs_ongoing`.
- **Contribution storage.** `roots_contributions: Vec<[F; 4]>`,
  `values_contributions: Vec<Mutex<Vec<F>>>`.
- **Channels (crossbeam).** `witness_tx/rx` + priority variant,
  `contributions_tx/rx`, `proofs_tx/rx`, `compressor_witness_tx/rx`,
  `rec1_witness_tx/rx`, `rec2_witness_tx/rx`, `recursive_tx/rx`
  (proof-completion callbacks), plus `tx_threads/rx_threads` for a
  global thread permit pool.
- **Synchronization.** `Counter` (`proofs_pending`,
  `total_outer_agg_proofs`), `outer_aggregation_state`, an
  `AtomicBool outer_agg_proofs_finished`, and a
  `CancellationInfo`/`CancellationToken` pair that propagates errors
  across worker threads and MPI ranks.

`ProofMan::new()` ([proofman/src/proofman.rs:1516-1731](proofman/src/proofman.rs#L1516-L1731))
sets up all of the above: it loads setups, pre-allocates host/device
buffers (see `load_device_setups`, `load_device_const_pols`,
`initialize_proofman`), sizes the stream pools, and spins up shared
channels.

### 2.3 Phase enums

[proofman/src/proofman.rs:269-296](proofman/src/proofman.rs#L269-L296):

```rust
enum ProvePhase { Contributions, Internal, Full }
enum ProvePhaseInputs { Contributions(), Internal(Vec<ContributionsInfo>), Full() }
enum ProvePhaseResult {
    Contributions(Vec<ContributionsInfo>),
    Internal(Vec<AggProofs>),
    Full(Option<String>, Option<VadcopFinalProof>),
}
```

This tri-phase split exists for distributed proving: each worker can
first compute `Contributions`, exchange them, then resume with
`Internal` / `Full`.

### 2.4 Top-level driver

- `ProofMan::generate_proof` ([proofman.rs:1397](proofman/src/proofman.rs#L1397))
  is the public entry; it forwards to
- `ProofMan::_generate_proof` ([proofman.rs:1747-2705](proofman/src/proofman.rs#L1747-L2705)),
  which implements the whole pipeline in one monolithic function. The
  phases below correspond to consecutive regions of this function.

---

## 3. Pipeline phases

Numbering matches the user's mental model: *execute → commit & challenge
→ prove → aggregate → final*. (An optional SNARK-wrapping pass after
the final proof is out of scope for this document.)

### Phase 0 — Planning / execution plan

- `ProofMan::execute` ([proofman.rs:608](proofman/src/proofman.rs#L608))
  and `execute_` ([proofman.rs:632](proofman/src/proofman.rs#L632)):
  load the witness library, let it declare *which* instances of each AIR
  it intends to commit, and build a `PlanningInfo`.
- Called by CLI `execute` (dry-run) and implicitly at the start of
  `_generate_proof` through `self.exec()`
  ([proofman.rs:1841](proofman/src/proofman.rs#L1841)).
- Output: per-instance plan (airgroup/air IDs, instance counts), stored
  in `pctx`'s distributed context (`dctx_*` methods).

### Phase 1 — Witness generation

Purpose: produce the committed witness traces for every instance the
current MPI worker owns.

- Location: [proofman.rs:1833-1906](proofman/src/proofman.rs#L1833-L1906).
- Helpers:
  - `ProofMan::calc_witness_handler`
    ([proofman.rs:3466](proofman/src/proofman.rs#L3466)) — spawns a
    handler thread that drains `witness_rx` and kicks downstream work
    (commitments or proofs) as soon as witnesses are ready.
  - `ProofMan::calculate_witness`
    ([proofman.rs:3585-3710](proofman/src/proofman.rs#L3585-L3710)) —
    iterates over `my_instances_sorted_no_tables`, calls
    `WitnessManager::pre_calculate_witness` and `calculate_witness`,
    respecting `num_threads_per_witness` and the `tx_threads` permit
    pool. Under `feature = "gpu"` it batches pre-calculation and
    streams traces straight into device buffers.
  - Witness for "table" AIRs (state read-only tables computed from the
    aggregated witness) is produced afterwards at
    [proofman.rs:1893-1904](proofman/src/proofman.rs#L1893-L1904).
- Inputs: loaded witness library, public inputs, `pctx` planning.
- Outputs: populated instance buffers (host or device), ready for
  commitment; notifications on `witness_tx`.
- GPU split: if `feature = "gpu"` and not `minimal_memory`, `pctx` holds
  a `witness_tx` sender and witness-ready signals push commitment work
  into the contribution threads. Otherwise CPU path commits inline.

### Phase 2 — Commitments and global challenge

Purpose: commit each instance witness via a Merkle tree (Poseidon over
Goldilocks), Poseidon-hash the resulting roots + public values into
per-instance contributions, then aggregate across workers into a single
global challenge that seeds every downstream transcript.

- Region: [proofman.rs:1764-2012](proofman/src/proofman.rs#L1764-L2012).
- Worker threads (`n_streams` of them) spawned at
  [proofman.rs:1785-1831](proofman/src/proofman.rs#L1785-L1831):
  each reads `contributions_rx` and invokes
  `ProofMan::get_contribution_air`
  ([proofman.rs:4055](proofman/src/proofman.rs#L4055)). That function
  calls `commit_witness_c` (Merkle commit, via
  [provers/starks-lib-c/](provers/starks-lib-c/)), stores the root into
  `roots_contributions[instance_id]`, and records the packed challenge
  input into `values_contributions[instance_id]`.
- Per-worker reduction:
  `calculate_internal_contributions`
  ([challenge_accumulation.rs](proofman/src/challenge_accumulation.rs),
  invoked at [proofman.rs:1927](proofman/src/proofman.rs#L1927)) folds
  all local (root, values) pairs into a single worker contribution
  (Poseidon + EC / lattice accumulation depending on
  `pctx.global_info.curve`).
- MPI gather: `mpi_ctx.distribute_roots(...)`
  ([proofman.rs:1937](proofman/src/proofman.rs#L1937)) all-gathers
  worker contributions.
- Global reduction: `aggregate_contributions`
  ([proofman.rs:1943](proofman/src/proofman.rs#L1943)) produces the
  final shared challenge.
- Exit path for `ProvePhase::Contributions`:
  [proofman.rs:1948-1971](proofman/src/proofman.rs#L1948-L1971) returns
  `ProvePhaseResult::Contributions(...)` here without proving.
- Outputs used downstream: `pctx.global_challenge`, which seeds every
  instance's STARK transcript in Phase 3.

### Phase 3 — Basic STARK proofs (per instance)

Purpose: one STARK proof per instance.

- Region: [proofman.rs:2014-2525](proofman/src/proofman.rs#L2014-L2525).
- The driver consumes two sources of instances:
  1. Instances whose witnesses were kept resident (initial batch,
     [proofman.rs:2203-2232](proofman/src/proofman.rs#L2203-L2232)).
  2. Instances waiting for witness completion (stream processing,
     [proofman.rs:2280-2448](proofman/src/proofman.rs#L2280-L2448)).
- Core call: `ProofMan::gen_proof`
  ([proofman.rs:3740-3814](proofman/src/proofman.rs#L3740-L3814)). It:
  - Calls `initialize_instance_c` if needed.
  - Invokes `gen_proof_c` (C FFI) to run the STARK prover. Under GPU,
    the call is asynchronous on a stream; under CPU it is synchronous
    and immediately triggers the completion callback via
    `launch_callback_c`.
  - Stores the resulting `Proof<F>` in `proofs[instance_id]`.
- Stream management: `get_stream_proofs_non_blocking_c`
  ([proofman.rs:2508](proofman/src/proofman.rs#L2508)) and
  `get_stream_proofs_c` ([proofman.rs:2511](proofman/src/proofman.rs#L2511))
  drain completed proofs from the GPU.
- Completion signal: each finished proof posts `(instance_id, "Basic")`
  to `recursive_tx`. Recursive worker threads
  ([proofman.rs:2052-2198](proofman/src/proofman.rs#L2052-L2198)) pick
  them up and feed the next stage.
- `proofs_pending: Counter` tracks outstanding basic proofs. Main
  thread waits with
  `proofs_pending.wait_until_zero_and_check_streams(...)`
  ([proofman.rs:2507](proofman/src/proofman.rs#L2507)).
- GPU split: buffer pointers passed to `gen_proof_c` are null on GPU
  (data is resident on device). CPU path passes `aux_trace`,
  `const_pols`, `const_tree`.

### Phase 4 — Recursive witness + intermediate proofs

Purpose: fold basic proofs into progressively smaller recursive proofs.
Active only when `aggregation` is enabled. Stages, per AIR:

- **Compressor** (optional, only if the AIR has a compressor circuit).
- **Recursive1** (always, after compressor or directly after Basic).
- **Recursive2** (pairwise tree, aggregating 3 inputs at a time).

Files: [proofman/src/recursion.rs](proofman/src/recursion.rs) +
recursive callback handlers in proofman.rs.

- Callback handler threads:
  [proofman.rs:2052-2198](proofman/src/proofman.rs#L2052-L2198) and
  [proofman.rs:3017-3070](proofman/src/proofman.rs#L3017-L3070). Each
  receives `(id, proof_type)` on `recursive_rx` and dispatches based on
  `proof_type`.
- **Witness generation:** [recursion.rs:68-244](proofman/src/recursion.rs#L68-L244)
  - `gen_witness_recursive` ([recursion.rs:68](proofman/src/recursion.rs#L68))
    takes a `Basic` or `Compressor` proof, augments it with circom
    publics / verification keys, and calls `generate_witness`
    ([recursion.rs:946](proofman/src/recursion.rs#L946)) which runs the
    Circom witness calculator (C FFI). The emitted witness is routed on
    `compressor_witness_tx` or `rec1_witness_tx`.
  - `gen_witness_aggregation`
    ([recursion.rs:180](proofman/src/recursion.rs#L180)) merges 3
    Recursive1 (or Recursive2) proofs with their verifier keys and
    produces a Recursive2 witness, routed on `rec2_witness_tx`. The
    arity is `N_RECURSIVE_PROOFS_PER_AGGREGATION = 3`
    ([recursion.rs:29](proofman/src/recursion.rs#L29)).
- **Proof generation:** `generate_recursive_proof`
  ([recursion.rs:291](proofman/src/recursion.rs#L291)) runs the STARK
  prover on a recursive witness via `gen_proof_c`, producing the next
  `Proof<F>`. Completion posts back through `recursive_tx` with the
  proper `proof_type` string ("Compressor" / "Recursive1" /
  "Recursive2").
- Storage: `compressor_proofs[id]`, `recursive1_proofs[id]`,
  and `recursive2_proofs[airgroup_id]: Vec<Proof<F>>`. Recursive2
  proofs pool per airgroup; when a pool reaches size 3, the three
  oldest are popped and fed into another aggregation.
- GPU split: recursive proofs use the remaining
  `n_streams - n_streams_non_recursive` streams. `pctx` exposes a
  separate `memory_handler_recursive_witness` so witness memory for
  recursive circuits is pooled independently of basic witnesses.

### Phase 5 — Outer aggregation + per-worker final proof

Purpose: finish the Recursive2 tree per airgroup within each worker,
yielding one aggregated proof per airgroup that the worker owns.

- Function: `ProofMan::outer_aggregations`
  ([proofman.rs:2997](proofman/src/proofman.rs#L2997)). Triggered by
  `ensure_outer_aggregations_started`
  ([proofman.rs:313](proofman/src/proofman.rs#L313)) as soon as the
  first Recursive2 proof is ready.
- It keeps pulling Recursive2 proofs for each airgroup, feeds them
  through `gen_witness_aggregation` + `generate_recursive_proof`, and
  reduces until one proof remains per airgroup.
- Helper counting: `total_recursive_proofs`
  ([recursion.rs:1040](proofman/src/recursion.rs#L1040)) predicts how
  many Recursive2 rounds are needed; `total_outer_agg_proofs` tracks
  completion.
- Per-worker result: `aggregate_worker_proofs`
  ([recursion.rs:432](proofman/src/recursion.rs#L432)) packages the
  surviving per-airgroup proofs into `AggProofs`. Under
  `ProvePhase::Internal`, these are returned directly
  ([proofman.rs:2651](proofman/src/proofman.rs#L2651)) so a higher-level
  orchestrator can aggregate across workers.

### Phase 6 — Cross-worker reception and VadcopFinal proof

Purpose: on the coordinator (MPI rank 0 or the single-process case),
aggregate all workers' per-airgroup proofs into the final VadcopFinal
proof.

- Region: [proofman.rs:2655-2705](proofman/src/proofman.rs#L2655-L2705)
  and [proofman.rs:2962-2989](proofman/src/proofman.rs#L2962-L2989).
- `ProofMan::receive_aggregated_proofs`
  ([proofman.rs:2731](proofman/src/proofman.rs#L2731)) collects
  `AggProofs` from all ranks via MPI (or trivially from local state),
  then drives the final aggregation tree to a single Recursive2 root
  per airgroup and finally across airgroups.
- `generate_vadcop_final_proof`
  ([recursion.rs:606](proofman/src/recursion.rs#L606)) emits the
  `VadcopFinal` proof.
- Optional compression:
  `generate_vadcop_final_compressed_proof`
  ([recursion.rs:710](proofman/src/recursion.rs#L710)) — activated by
  `options.compressed`.
- Verification (optional): `verify_vadcop_final` /
  `verify_vadcop_final_compressed` from
  [verifier/](verifier/), invoked at
  [proofman.rs:2673-2698](proofman/src/proofman.rs#L2673-L2698) when
  `options.verify_proofs` is set.
- Output: `ProvePhaseResult::Full(proof_id, Some(VadcopFinalProof))`.

### Temporal structure & concurrency groups

The phases above are a *logical* decomposition, not a *temporal*
one. At runtime the pipeline compresses into four wall-clock groups
separated by hard synchronization points. The channel-and-thread
architecture from §5 exists precisely to make these groups overlap
internally.

**Group A — Phase 0 (sequential, low parallelism).**
`exec()` / `execute_()` runs alone. It loads the witness library,
walks the instance plan, and populates `pctx` distributed state. Almost
all work is on a single logical thread; CPU utilization is low because
the work is bookkeeping, not arithmetic.

**Group B — Phases 1 + 2 (witness + commit, overlapped).**
Witness generation runs on the CPU cores while Merkle commitment +
Poseidon contribution hashing runs on the GPU. `calc_witness_handler`
pushes each ready witness onto `contributions_rx`; the `n_streams`
contribution workers enqueue `commit_witness_c` on GPU streams
(typically via D2H/H2D transfers plus Merkle kernels) while the next
CPU witnesses are still being computed. This group is the "CPU and
GPU used at the same time" portion of the trace. It ends with a
**hard barrier**: `get_stream_proofs_c(...)` drains any in-flight
commitments from the GPU, then `mpi_ctx.distribute_roots(...)` +
`aggregate_contributions(...)` produce the global challenge, which
every Phase 3 transcript is seeded from.

**Group C — Phases 3 + 4 (basic + recursive, overlapped).**
`gen_proof_c` runs on GPU streams; as each basic STARK proof
finishes, its completion callback fires `(instance_id, "Basic")` on
`recursive_rx`, and the recursive-callback workers
([proofman.rs:2052-2198](proofman/src/proofman.rs#L2052-L2198))
immediately start generating the next-stage witness (Compressor /
Recursive1) on CPU and enqueue the recursive proof on one of the
`n_streams - n_streams_non_recursive` recursive streams. Recursive2
pools aggregate 3 inputs at a time
(`N_RECURSIVE_PROOFS_PER_AGGREGATION`) and feed the next tree level
as soon as each triple is complete. Witness buffers freed at the end
of Group B are **regenerated on demand** when Phase 3 schedules each
instance; because this happens on CPU while basic proofs are on GPU
it is effectively hidden. This is the dominant group in the trace
(≈58 %), where GPU utilization is high and the pipeline's job is to
keep both the basic-proof streams and the recursive streams saturated.

**Group D — Phase 5 (alone, single GPU).**
The per-worker outer-aggregation tree finishes the Recursive2 cascade
down to one proof per airgroup. In the distributed topology this runs
on whichever worker has outstanding aggregations; in the "final
aggregator" case it also absorbs remote `AggProofs` arriving from
peer workers. The trace shows it collapsing onto a single GPU —
aggregation is sequential per airgroup, so multi-GPU parallelism
largely evaporates here.

**Group E — Phase 6 (single machine).**
VadcopFinal proof generation on the aggregator. Single GPU,
single-instance proof; effectively a fixed tail cost. Out of current
optimization scope. (An optional SNARK-wrapping pass would follow on
the same machine; also out of scope here.)

#### Illustrative trace (4 workers × 2 GPUs, ~v0.14.0)

| Group | Phases | Wall time | Share | Notes |
|-------|--------|-----------|-------|-------|
| A | 0 | 1.2 s | 18 % | Sequential, low CPU/GPU utilization |
| B | 1 + 2 | 1.1 s | 17 % | CPU witness + GPU commit co-utilized; hard barrier at end (global challenge) |
| C | 3 + 4 | 3.8 s | 58 % | GPU-dominated; basic-proof streams overlapped with recursive witness/proof; witnesses regenerated on demand |
| D | 5 | 0.5 s |  7 % | Single GPU, largely sequential |
| — | total | 6.6 s | 100 % | Phase 6 (and the out-of-scope SNARK pass) excluded from the trace |

The trace corresponds to 4 workers × 2 GPUs on a ~v0.14.0 build; the
absolute numbers have moved since, but the shape
(A sequential → B overlapped CPU/GPU with a barrier → C GPU-dominated,
dominant → D single-GPU tail) is intrinsic to the workflow and
matches the current architecture.

#### Implications for the GPU path

- **Group B is latency-bound at the barrier.** CPU witness and GPU
  commitment are already pipelined through `n_streams` contribution
  workers; the wall-clock floor is
  `max(per-worker witness+commit) + all-gather RTT`. Within a single
  worker, closing this group faster means either feeding the GPU
  faster (CPU-side witness throughput) or reducing the per-instance
  commitment kernels.
- **Group C dominates and is the right target.** 58 % of wall time.
  Two GPU pipelines (basic-proof streams and recursive streams) share
  device memory and the stream pool. Contention points: stream
  partitioning (`n_streams_non_recursive` vs. recursive streams),
  GPU memory pressure between basic witnesses, recursive witnesses,
  and constants, the Recursive2 tree arity, and kernel occupancy
  inside `gen_proof_c`.
- **Group D is a single-GPU serial tail.** Multi-GPU scale-up stops
  helping here; once Group C finishes, the remaining aggregation
  time on one GPU sets a hard floor.

---

## 4. Data flow between phases

Primary types (defined in [common/](common/) unless noted):

- **`Proof<F>`** — `{ proof_type: ProofType, airgroup_id, air_id,
  global_idx, proof: Vec<u64> }`. The `ProofType` values relevant to
  the in-scope flow are `{ Basic, Compressor, Recursive1, Recursive2,
  VadcopFinal }`. Proofs travel through channels; each phase
  "upgrades" the type.
- **`ProofCtx<F>`** — holds:
  - Global info: AIR groups, AIR metadata, curve type,
    proof/public values.
  - Distributed context (`dctx_*`): instance ownership, rank/process
    info, current plan.
  - Runtime state: `global_challenge`, per-instance device buffer
    pointers, witness/proof channel senders (`set_witness_tx`,
    `set_proof_tx`).
- **`SetupCtx<F>` / `Setup<F>`** — per-AIR setup: stark-info,
  constant polynomials, verification keys, Merkle trees of constants.
  Initialized once in `ProofMan::new`, loaded to device in
  `load_device_setups`.
- **`SetupsVadcop<F>`** — aggregation-stage setups: Compressor,
  Recursive1, Recursive2, VadcopFinal. Each stage gets its own Circom
  circuit and STARK setup. (The struct also carries a `RecursiveF`
  setup used by the out-of-scope SNARK-wrapping pass.)
- **`MemoryHandler<F>` / `MemoryHandlerRecursive<F>`** — pooled
  buffers for instance witnesses and recursive witnesses respectively.
  `to_be_released_buffer(id, ...)` queues a buffer for reuse once the
  consumer signals it's done.
- **`ContributionsInfo`** — `{ challenge: Vec<u64>, airgroup_id,
  worker_index, aggregated }`. Unit of exchange in Phase 2.
- **`AggProofs`** — per-worker aggregated proof bundle. Unit of
  exchange between Phase 5 and Phase 6.
- **`VadcopFinalProof`** (from [util/](util/)) — final STARK proof
  output.

---

## 5. Concurrency / threading model

`ProofMan` coordinates several long-lived thread groups, all fed by
crossbeam channels.

| Pool | Spawn site | Channel in | Work |
|------|-----------|-----------|------|
| Thread permit pool | `ProofMan::new` ([proofman.rs:1655-1668](proofman/src/proofman.rs#L1655-L1668)) | `tx_threads/rx_threads` | Global concurrency cap for witness / commitment calls. |
| Contribution workers (`n_streams`) | [proofman.rs:1785-1831](proofman/src/proofman.rs#L1785-L1831) | `contributions_rx` | Commit witness + Poseidon over roots & publics. |
| Witness handler (1) | `calc_witness_handler` ([proofman.rs:3466](proofman/src/proofman.rs#L3466)) | `witness_rx` | Dispatch witness-ready instances into commitment / proof flows. |
| Recursive callback workers (`n_streams`) | [proofman.rs:2052-2198](proofman/src/proofman.rs#L2052-L2198) | `recursive_rx` | Convert completed proofs into next-stage witnesses, schedule recursive proofs, aggregate Recursive2 pools. |
| Stream pump (`n_streams`) | [proofman.rs:3005-3014](proofman/src/proofman.rs#L3005-L3014) | (polls GPU) | `get_stream_proofs_non_blocking_c` drain loop for recursive streams during outer aggregation. |

Cancellation is signalled by writing to `cancellation_info`
([proofman.rs:147-206](proofman/src/proofman.rs#L147-L206)); every long
loop checks `check_cancel(notify_mpi)` so errors propagate promptly and
MPI peers are notified.

---

## 6. GPU / CPU split and the STARK C library

GPU integration is compile-gated on `cfg(feature = "gpu")` and
runtime-controlled through `ParamsGPU`. The main differences relative
to the CPU path:

1. **Initialization.** `set_gpu_mode_c`, `init_gpu_setup_c`,
   `get_num_gpus_c` ([proofman.rs:11](proofman/src/proofman.rs#L11))
   are called during setup. Device buffers are allocated once and
   addressed through `pctx.get_device_buffers_ptr()`; CPU vectors
   (`aux_trace`, `const_pols`, `const_tree`) remain empty under GPU.
2. **Witness staging.** `pctx.set_witness_tx(...)` wires the witness
   producer directly into the GPU pipeline so traces stream onto the
   device.
3. **Proof dispatch.** `gen_proof_c` is asynchronous; completion is
   polled with `get_stream_proofs_non_blocking_c` (fast path) and
   finalized with `get_stream_proofs_c` (drain). Callbacks are
   registered once with `register_proof_done_callback_c` and cleared on
   shutdown via `clear_proof_done_callback_c`.
4. **Device lifetime.** `free_device_buffers_c` is invoked from
   `ProofMan::Drop` ([proofman.rs:303-310](proofman/src/proofman.rs#L303-L310)).

All kernels — STARK, Poseidon/Merkle, MSM, FFT — live in the
`pil2-stark` submodule. The most relevant GPU source files for the
current optimization work are:

- STARK/NTT/Merkle: `pil2-stark/src/starkpil/*.cu`,
  `pil2-stark/src/goldilocks/*`.

Build flow: `cd pil2-stark && make -j starks_lib_gpu`, then
`touch provers/starks-lib-c/build.rs && cargo build --release --features gpu --bin proofman-cli`.

---

## 7. Summary table

| # | Phase | Orchestrator | File / lines | Key FFI / helper | Output |
|---|-------|-------------|--------------|------------------|--------|
| 0 | Planning | `exec` / `execute_` | [proofman.rs:608-700](proofman/src/proofman.rs#L608-L700) | — | `PlanningInfo` in `pctx` |
| 1 | Witness generation | `calc_witness_handler`, `calculate_witness` | [proofman.rs:3466-3710](proofman/src/proofman.rs#L3466-L3710) | `WitnessManager::calculate_witness`, `calculate_trace_instance_c` | Instance witnesses (host or device) |
| 2 | Commitments + global challenge | worker loop + `get_contribution_air` + `calculate_internal_contributions` + `aggregate_contributions` | [proofman.rs:1764-2012](proofman/src/proofman.rs#L1764-L2012), [proofman.rs:4055+](proofman/src/proofman.rs#L4055), [challenge_accumulation.rs](proofman/src/challenge_accumulation.rs) | `commit_witness_c`, `mpi_ctx.distribute_roots` | `pctx.global_challenge`, `ContributionsInfo` |
| 3 | Basic STARK proofs | `gen_proof` | [proofman.rs:2014-2525](proofman/src/proofman.rs#L2014-L2525), [proofman.rs:3740-3814](proofman/src/proofman.rs#L3740-L3814) | `initialize_instance_c`, `gen_proof_c` | `proofs[id]: Proof<Basic>` |
| 4 | Recursive witnesses + proofs (Compressor / Rec1 / Rec2) | recursive callback workers | [proofman.rs:2052-2198](proofman/src/proofman.rs#L2052-L2198), [recursion.rs:68-430](proofman/src/recursion.rs#L68-L430) | `generate_witness`, `generate_recursive_proof`, `gen_proof_c` | Per-stage proofs |
| 5 | Outer aggregation | `outer_aggregations`, `aggregate_worker_proofs` | [proofman.rs:2997-3070](proofman/src/proofman.rs#L2997-L3070), [recursion.rs:432](proofman/src/recursion.rs#L432) | `gen_witness_aggregation`, `generate_recursive_proof` | `AggProofs` per worker |
| 6 | Final VadcopFinal proof | `receive_aggregated_proofs`, `generate_vadcop_final_proof` | [proofman.rs:2655-2705](proofman/src/proofman.rs#L2655-L2705), [recursion.rs:606-782](proofman/src/recursion.rs#L606-L782) | `gen_proof_c`, `verify_vadcop_final` | `VadcopFinalProof` |

(An optional SNARK-wrapping pass — `SnarkWrapper::generate_final_snark_proof`
in [snark_wrapper.rs](proofman/src/snark_wrapper.rs) — runs after
Phase 6 to produce a Plonk/FFlonk on-chain proof, but is out of scope
in this document.)

---

## 8. Starting points for optimization (GPU)

Using the temporal groups defined in §3.8, the optimization surface
collapses to three real targets:

### Group B — witness + commit (≈17 % of wall time)

CPU witness generation and GPU commitment already overlap; the
remaining cost is:

- **CPU witness throughput.** If the GPU contribution streams go
  idle, the limiter is CPU-side witness generation.
  `calculate_witness` spawns one thread per instance
  ([proofman.rs:3585-3710](proofman/src/proofman.rs#L3585-L3710))
  up to `max_num_threads / num_threads_per_witness`. Uneven instance
  sizes produce tail latency at the barrier.
- **Commit kernel cost on GPU.** `commit_witness_c` runs the
  Poseidon-over-Goldilocks Merkle tree on-device. If the CPU side is
  producing witnesses faster than the GPU can consume them, the
  optimisation lives in the Merkle kernels (and the H2D transfer
  that precedes them).
- **Stream occupancy.** Whether all `n_streams` contribution workers
  stay busy during Group B tells us which side dominates — idle
  streams = CPU-bound; backed-up witness queue = GPU-bound.
- **Barrier overhead.** `get_stream_proofs_c` at
  [proofman.rs:1922](proofman/src/proofman.rs#L1922) forces a drain
  before the global challenge is computed; any straggler stream
  extends Group B's tail.

### Group C — basic + recursive (≈58 % of wall time, the real target)

Two GPU pipelines — basic proofs and the recursive tree — share
device memory and the stream pool.

- **Stream partitioning.** `n_streams_non_recursive` (basic pool)
  and `n_streams - n_streams_non_recursive` (recursive pool) are
  chosen in `ProofMan::new`
  ([proofman.rs:1595-1601](proofman/src/proofman.rs#L1595-L1601)).
  Under-provisioning either side leaves GPU cycles on the table;
  over-provisioning increases VRAM pressure and contention for
  shared kernels (NTT/MSM working buffers).
- **GPU memory budget.** Basic witnesses, recursive witnesses, and
  pre-loaded constants all share device memory addressed via
  `pctx.get_device_buffers_ptr()`. `MemoryHandler` and
  `MemoryHandlerRecursive` gate how many in-flight proofs coexist.
  When either runs dry upstream work stalls waiting for
  `to_be_released_buffer` → `release_buffer` cycles — stalls that do
  not show up in per-phase timers but are visible in stream/thread
  occupancy.
- **Recursive-tree arity.**
  `N_RECURSIVE_PROOFS_PER_AGGREGATION = 3`
  ([recursion.rs:29](proofman/src/recursion.rs#L29)) trades tree
  depth for per-aggregation size. Higher arity = fewer aggregation
  proofs but each is a bigger STARK (more VRAM, longer stream
  residency); lower arity = more, cheaper proofs.
- **Witness regeneration.** Buffers freed at the end of Group B are
  recomputed on CPU when Phase 3 schedules each instance. Because
  this runs concurrently with GPU proving it is mostly hidden, but
  it holds CPU cores that could otherwise feed recursive-witness
  generation (Circom) — a knob worth measuring rather than assuming.
- **Inside `gen_proof_c`.** Kernel-level optimisations (NTT, MSM,
  Merkle, constraint evaluation) live in
  [pil2-stark/](pil2-stark/) — once the orchestration keeps streams
  saturated, the remaining wins are per-kernel.
- **Callback latency.** `register_proof_done_callback_c` +
  `recursive_tx` wakes the recursive worker, which then kicks off a
  CPU Circom witness and enqueues the next GPU proof. If that
  hand-off adds jitter, recursive streams miss their slot. Measuring
  callback-to-enqueue latency is the diagnostic.

### Group D — outer aggregation tail (≈7 % of wall time)

- Single GPU, sequential per airgroup
  ([proofman.rs:2997-3070](proofman/src/proofman.rs#L2997-L3070)).
- Multi-GPU scale-up stops helping here; if Group C shrinks enough
  this becomes the new bottleneck.
- Per-airgroup parallelism exists in principle (each tree is
  independent) but is serialized through the same recursive stream
  pool on one GPU.

### Where to look for data

All per-phase wall-clock accounting is emitted by
`timer_start_info!` / `timer_start_debug!` from [util/](util/). Run
with `-vv` and the log groups line up 1-to-1 with Phases 0-6. For
Group C internals watch `CALCULATING_WITNESS`, the implicit
`gen_proof_c` intervals, and
`ensure_outer_aggregations_started` / `outer_aggregations`. For GPU
kernel-level detail, Nsight Systems traces around `gen_proof_c` and
`commit_witness_c` are the right tool — they show stream overlap
directly and make the distinction between "all streams busy" and
"streams busy but one kernel serialising everything" obvious.

### Open questions to drive the discussion

1. On a current GPU run (v0.17.0) what is the per-group breakdown,
   and has Group C shifted relative to the v0.14.0 trace?
2. During Group C, are both the basic-proof streams and the recursive
   streams saturated, or is one side waiting? That dictates whether
   to rebalance `n_streams_non_recursive` or to target the slow side.
3. Does the recursive witness memory pool ever run dry? If yes,
   recursive proving stalls invisibly and the fix is memory sizing,
   not kernel tuning.
4. Is the Recursive2 arity of 3 still the right trade-off at current
   GPU capacity, or would arity-4/arity-5 better amortize per-proof
   overhead?
5. In Group D, is the limiter the aggregation STARK itself or idle
   time waiting on peer workers' `AggProofs`?

The answers determine whether the first move is stream-pool
rebalancing, arity tuning, memory-pool resizing, kernel-level
optimisation inside `gen_proof_c`, or something more structural
(e.g. pipelining Group D with a still-running Group C tail).

---

## 9. Distributed mode

`pil2-proofman` supports two layers of distribution, which are composed
when the prover is run as part of ZisK:

- **Intra-worker MPI** — built directly into `ProofMan<F>` via
  `MpiCtx` ([common/](common/)). Multiple MPI ranks cooperate within a
  single "worker" (typically one host or a tightly-coupled cluster),
  exchanging contributions and aggregated proofs through
  `mpi_ctx.distribute_roots(...)` / `mpi_ctx.broadcast(...)` and
  the tri-phase API (`ProvePhase::{Contributions, Internal, Full}`).
- **Inter-machine coordinator/worker** — a gRPC-based orchestration
  layer in the ZisK repo at [`../zisk/distributed/`](../zisk/distributed/).
  A single coordinator process fans a proof job out across many
  independent workers, each of which is a self-contained MPI-capable
  `ProofMan` instance.

This section documents both.

### 9.1 Intra-worker MPI (inside `pil2-proofman`)

When built with `--cfg distributed` the `mpi` crate is linked in and
`ProofMan::new` wires `MpiCtx` into `pctx`. The protocol is
implicit in `_generate_proof` ([proofman.rs:1747-2705](proofman/src/proofman.rs#L1747-L2705)):

1. Every rank runs Phases 0-2 locally on its slice of instances.
2. `mpi_ctx.distribute_roots(internal_contribution)`
   ([proofman.rs:1937](proofman/src/proofman.rs#L1937)) all-gathers
   partial contributions across ranks, producing a consistent global
   challenge on all ranks.
3. Every rank runs Phase 3 (basic proofs) and Phase 4 (compressor /
   Recursive1) on its instances, then Phase 5 (outer Recursive2
   aggregation) locally, yielding per-airgroup `AggProofs`.
4. Rank 0 calls `ProofMan::receive_aggregated_proofs`
   ([proofman.rs:2731](proofman/src/proofman.rs#L2731)) which gathers
   remote `AggProofs` from the other ranks over MPI and finishes the
   VadcopFinal proof (Phase 6).
5. Rank 0 returns the `VadcopFinalProof`; the other ranks return an
   empty `ProvePhaseResult::Internal(Vec::new())`.

`ProvePhase` controls which portion of this flow a single call
performs:

- `ProvePhase::Contributions` — runs Phase 1 + Phase 2 and returns
  `ContributionsInfo` only (suitable for external aggregation).
- `ProvePhase::Internal` — requires `ProvePhaseInputs::Internal(...)`
  with *externally supplied* contributions; skips the internal
  contribution step and runs Phases 3-5, returning per-airgroup
  `AggProofs`.
- `ProvePhase::Full` — runs everything end-to-end.

The tri-phase API is what makes the outer gRPC layer possible: the
ZisK coordinator can invoke `Contributions` across all workers, gather
and redistribute the result, then invoke `Internal` on provers and
aggregate externally.

Cancellation is propagated across ranks through
`mpi_ctx.notify_cancellation()` ([proofman.rs:514](proofman/src/proofman.rs#L514)),
consumed by the `CancellationThread`
([proofman.rs:147-183](proofman/src/proofman.rs#L147-L183)).

### 9.2 Inter-machine coordinator / worker (ZisK)

Code location: [`../zisk/distributed/`](../zisk/distributed/). Crates:

| Crate | Role |
|-------|------|
| `coordinator` | `zisk-coordinator` binary, gRPC service, job scheduler, worker pool, aggregator selection. |
| `worker` | `zisk-worker` binary, gRPC client, runs a `ZiskProver` backed by `pil2-proofman`'s `ProofMan`. |
| `grpc-api` | Protobuf-generated types + bidirectional `WorkerStream` RPC. |
| `common` | `JobPhase`, `WorkerState`, DTOs shared between coordinator and worker. |

Transport is purely gRPC (TCP, default port 50051); MPI is **never**
used between machines. If a worker spans multiple ranks it uses MPI
internally (see §9.3).

#### Job lifecycle

1. **Launch.** A client calls the coordinator's admin RPC
   `LaunchProof` ([coordinator_grpc.rs:367-380](../zisk/distributed/crates/coordinator/src/coordinator_grpc.rs#L367-L380))
   with `inputs_uri`, optional `hints_uri`, and a
   `compute_capacity`.
2. **Worker selection and partitioning.** `create_job` /
   `partition_and_allocate_by_capacity`
   ([workers_pool.rs:427-535](../zisk/distributed/crates/coordinator/src/workers_pool.rs#L427-L535))
   picks idle workers round-robin until the requested capacity is
   covered, then splits the compute units into per-worker
   `allocation: Vec<u32>` slices. Each worker also gets a `rank_id`
   and `total_workers` count.
3. **Phase 1 dispatch — Partial Contributions.** The coordinator
   sends `ExecuteTaskRequest(ContributionParams{...})` to every
   selected worker over the bidirectional `WorkerStream`
   ([coordinator.rs dispatch_contributions_messages, ~680-723](../zisk/distributed/crates/coordinator/src/coordinator.rs)).
4. **Phase 1 execution.** Each worker's rank-0 calls
   `prover.prove_phase(ProvePhaseInputs::Contributions(), options, ProvePhase::Contributions)`
   ([worker.rs:726-741](../zisk/distributed/crates/worker/src/worker.rs#L726-L741))
   and returns `ContributionsInfo` via `ExecuteTaskResponse`.
5. **Phase 1 gather.** Coordinator collects all `ContributionsInfo`,
   validates them in `handle_contributions_completion`
   ([coordinator.rs:1174-1223](../zisk/distributed/crates/coordinator/src/coordinator.rs#L1174-L1223)),
   and advances the job to `Running(Prove)`.
6. **Phase 2 dispatch — Prove.** Coordinator sends
   `ExecuteTaskRequest(ProveParams{ challenges: Vec<Challenges>, ... })`
   — the *full* challenge set — back to every worker.
7. **Phase 2 execution.** Each worker runs
   `prover.prove_phase(ProvePhaseInputs::Internal(challenges), options, ProvePhase::Internal)`
   ([worker.rs:888-904](../zisk/distributed/crates/worker/src/worker.rs#L888-L904))
   and returns `Vec<AggProofs>` (its share of per-airgroup aggregated
   Recursive2 proofs).
8. **Aggregator selection.** The first worker to return its
   `AggProofs` is promoted to the Aggregator in
   `resolve_aggregator_assignment`
   ([coordinator.rs:2106-2141](../zisk/distributed/crates/coordinator/src/coordinator.rs#L2106-L2141));
   the other workers are returned to the `Idle` pool immediately so
   they can pick up new jobs while aggregation runs.
9. **Phase 3 dispatch — Aggregation.** Coordinator sends
   `ExecuteTaskRequest(AggParams{ agg_proofs, last_proof, final_proof, compressed })`
   to the chosen aggregator only
   ([coordinator.rs:2267-2298](../zisk/distributed/crates/coordinator/src/coordinator.rs#L2267-L2298)).
10. **Phase 3 execution.** The aggregator registers the external
    `AggProofs` via `prover.register_aggregated_proofs(...)` and calls
    `prover.aggregate_proofs(agg_proofs, last_proof, final_proof, &options)`
    ([worker.rs:962-994](../zisk/distributed/crates/worker/src/worker.rs#L962-L994)),
    which drives `ProofMan::receive_aggregated_proofs` to produce the
    final `VadcopFinalProof` (optionally compressed).
11. **Finalization.** Coordinator receives the final proof,
    `post_launch_proof`
    ([coordinator.rs:2321-2648](../zisk/distributed/crates/coordinator/src/coordinator.rs#L2321-L2648))
    persists it, triggers any configured webhook, and marks the job
    `Completed`.

Note how the three ZisK phases are a direct 1-to-1 with
`ProvePhase`:

| ZisK phase | pil2-proofman API | Returned payload |
|-----------|-------------------|------------------|
| Partial Contributions | `ProvePhase::Contributions` + `ProvePhaseInputs::Contributions()` | `ProvePhaseResult::Contributions(Vec<ContributionsInfo>)` |
| Prove | `ProvePhase::Internal` + `ProvePhaseInputs::Internal(Vec<ContributionsInfo>)` | `ProvePhaseResult::Internal(Vec<AggProofs>)` |
| Aggregation | `aggregate_proofs(...)` driven by `register_aggregated_proofs(...)` | `VadcopFinalProof` |

#### gRPC transport

- Single bidirectional RPC: `WorkerStream(stream WorkerMessage) returns (stream CoordinatorMessage)`
  ([zisk_distributed_api.proto:13](../zisk/distributed/crates/grpc-api/proto/zisk_distributed_api.proto)).
- Worker → coordinator messages: `Register` / `Reconnect`,
  `HeartbeatAck`, `ExecuteTaskResponse` (per phase), `WorkerError`.
- Coordinator → worker messages: `RegisterResponse`, `Heartbeat`,
  `ExecuteTaskRequest` (carrying `ContributionParams` /
  `ProveParams` / `AggParams` / `ExecutionParams`), optional
  `StreamData` for the hints-stream mode.
- Admin RPCs (localhost, client-facing): `LaunchProof`, `JobsList`,
  `WorkersList`, `JobStatus`, `SystemStatus`, `HealthCheck`.

#### Worker state machine

States from [`common/src/types.rs`](../zisk/distributed/crates/common/src/types.rs):
`Disconnected → Connecting → Idle → Computing((JobId, JobPhase)) → Idle`.

Transitions driven by the coordinator:

```
Idle
  ↓ (ExecuteTaskRequest: Contributions)
Computing((job, Contributions))
  ↓ (ExecuteTaskResponse + dispatch_prove)
Computing((job, Prove))
  ↓ (ExecuteTaskResponse)
  ├── not first to finish → Idle            (freed, ready for next job)
  └── first to finish     → Computing((job, Aggregate))
                            ↓
                          Idle
```

Phase-1 / phase-2 timeouts are configurable
(`coordinator.phase1_timeout_seconds`,
`coordinator.phase2_timeout_seconds`).

### 9.3 Two-layer composition

When a ZisK worker runs with multiple MPI ranks (i.e. a small cluster
behind a single gRPC endpoint), the system uses **both** layers:

- **Rank 0 of the worker** owns the gRPC stream and drives
  `ProofMan` directly. `WorkerNodeGrpc`
  ([worker_node.rs:118-148](../zisk/distributed/crates/worker/src/worker_node.rs#L118-L148))
  is the main loop for this rank.
- **Ranks ≥ 1** run `WorkerNodeMpi`
  ([worker_node.rs:94-101](../zisk/distributed/crates/worker/src/worker_node.rs#L94-L101))
  which blocks on `handle_mpi_broadcast_request()` and mirrors the
  work dispatched by rank 0.
- Before each `ProofMan` call, rank 0 serialises the task and calls
  `ProofMan::mpi_broadcast(...)` (see
  [worker.rs partial_contribution_mpi_broadcast / prove_mpi_broadcast](../zisk/distributed/crates/worker/src/worker.rs))
  so all ranks enter `_generate_proof` in lock-step. Phases 1-5 then
  run under intra-worker MPI as described in §9.1.
- Aggregation (ZisK Phase 3) is **not** MPI-broadcast — the aggregator
  runs `receive_aggregated_proofs` single-process on rank 0.

This means optimization questions land in different places:

- **Proof-generation throughput** (Phases 1-5): dominated by intra-worker
  pil2-proofman + GPU kernels. MPI is used to stitch local state.
- **Fleet utilization** (worker idle time, scheduling latency): lives in
  the ZisK coordinator's worker pool and aggregator-selection policy.
- **Wire cost** (contribution / `AggProofs` serialization, gRPC frame
  size): defined by the protobuf schema in
  [`grpc-api/proto/`](../zisk/distributed/crates/grpc-api/proto/) and
  the DTO conversions in
  [`grpc-api/src/conversions.rs`](../zisk/distributed/crates/grpc-api/src/conversions.rs).

### 9.4 Key file anchors (ZisK distributed)

| Concern | File | Lines |
|---|---|---|
| Job launch / lifecycle | [coordinator.rs](../zisk/distributed/crates/coordinator/src/coordinator.rs) | 339-389, 1096-1223, 1891-2648 |
| Worker pool / partitioning | [workers_pool.rs](../zisk/distributed/crates/coordinator/src/workers_pool.rs) | 427-535 |
| gRPC service impl | [coordinator_grpc.rs](../zisk/distributed/crates/coordinator/src/coordinator_grpc.rs) | 367-488 |
| Worker task dispatch | [worker.rs](../zisk/distributed/crates/worker/src/worker.rs) | 437-463, 512-530, 681-994, 1014-1119 |
| Worker node roles (rank-0 vs rank-N) | [worker_node.rs](../zisk/distributed/crates/worker/src/worker_node.rs) | 39-71, 94-148, 563-935 |
| Protobuf definitions | [grpc-api/proto/zisk_distributed_api.proto](../zisk/distributed/crates/grpc-api/proto/zisk_distributed_api.proto) | whole file |
