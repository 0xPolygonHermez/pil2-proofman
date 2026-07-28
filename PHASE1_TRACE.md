# Phase-1 (contribution) tracing

Opt-in instrumentation of `CALCULATING_CONTRIBUTIONS`, added to answer one question: when a worker
becomes the phase-1 straggler, is the extra time **CPU-side work**, **waiting on a GPU stream**, or
**contended atomics in the shared multiplicity tables**?

Off unless `ZISK_TRACE_PHASE1=1`, and the part that touches the per-row multiplicity loop needs a
cargo feature on top of that. See [Two switches, and why](#two-switches-and-why) for the measured
reason.

```ini
# zisk-worker.service
Environment=ZISK_TRACE_PHASE1=1
Environment=ZISK_TRACE_DIR=/var/lib/zisk-worker/traces   # optional, this is the default
```

Then, after a run:

```sh
./tools/phase1_report.py /var/lib/zisk-worker/traces
```

## The zisk side

Three changes live in the zisk tree, on the same branch pair:

- **`distributed/crates/worker/src/worker.rs`**, in `execute_contribution_task`: one call to
  `phase1_trace::set_job_tag(&job_id.to_string())` before `prove_phase`. Without it proofman never
  sees the cluster's `JobId` — with it, every trace file is named after its job and every record
  carries `job_id`, so they join straight to the coordinator's `[Phase1] WorkerId(N) … JobId(…)`
  lines. No-op unless the env var is set.
- **`distributed/crates/worker/src/worker_node.rs`**, in `run()`: one call to
  `phase1_trace::set_worker_label(worker_id)` at startup. **This one is required for any
  cross-worker analysis** — see the identity note below.
- **`distributed/crates/worker/Cargo.toml` and `cli/Cargo.toml`**: a `phase1-trace` feature that
  forwards to `proofman/phase1-trace`, which in turn reaches `pil-std-lib`. Without this the atomic
  accounting cannot be switched on at all, since its gate is a compile-time `const`.

```sh
cargo build --release --features phase1-trace      # only needed for the mul_tables line
```

Note zisk consumes proofman from git (`branch = "feature/colMajor-ntt-scheduler"`), so the proofman
side of this has to be pushed to that branch before a zisk build picks it up — or the
`# Proofman Local development` path block in zisk's root `Cargo.toml` uncommented for a local build.

## Worker identity: `wk` is not the partition index

What `dctx` hands proofman is the *partition* index the coordinator allocated, and that is a
lexicographic rank of hostnames, not a machine identity:

```
worker-10 -> 0   worker-11 -> 1   worker-13 -> 2   worker-2 -> 3   worker-3 -> 4
worker-5  -> 5   worker-6  -> 6   worker-8  -> 7   worker-9 -> 8
```

`worker-5` and `worker-6` map to 5 and 6, so a spot-check of two workers looks correct while the
other seven are misattributed. Every line and every record therefore carries **both**, named for
what it is: `wk<WorkerId>` from `set_worker_label`, and `pidx<n>` from `dctx`. If the worker never
called `set_worker_label` you get `wk?`, and `phase1_report.py` prints a warning rather than
grouping on something that is not a machine.

## Two corrections to the brief

**`CALCULATING_CONTRIBUTIONS` was not uninstrumented.** It already had four sub-spans —
`PREPARING_CONTRIBUTIONS`, `CALCULATING_WITNESS`, `CALCULATING_TABLES`,
`CALCULATING_INNER_CONTRIBUTIONS` — emitted through `timer_start_debug!`, i.e. at `debug` level,
which the worker's `info`-level journal drops. The brief's "exactly one marker inside 1470ms" is a
log-level artefact, not missing code. The genuinely unbracketed sections were: the witness-thread
join, the contribution-worker drain, `get_stream_proofs_c`, and `calculate_internal_contributions`.
Those four are what the new spans cover, so the parent now accounts for itself.

**`ZISK_TRACE_PHASE1=1` deliberately does not just turn on debug.** Debug level is what
`PROOFMAN_SUMCHECK=1` effectively did — hundreds of journal lines per job, itself a suspect for
added jitter. The new spans emit at `info` but only under the new variable, so the journal gains
~20 lines per job and the bulk data goes to a file.

## What lands in the journal

Span pairs, in the existing convention (`>>> NAME` / `<<< NAME (Nms)`):

```
>>> CONTRIB_WITNESS        / <<< CONTRIB_WITNESS (Nms)         witness computation for my instances
>>> CONTRIB_JOIN_WITNESS   / <<< CONTRIB_JOIN_WITNESS (Nms)    joining witness threads still in flight
>>> CONTRIB_TABLES         / <<< CONTRIB_TABLES (Nms)          table instances (incl. the mul tables)
>>> CONTRIB_DRAIN          / <<< CONTRIB_DRAIN (Nms)           draining the contribution workers
>>> CONTRIB_STREAM_PROOFS  / <<< CONTRIB_STREAM_PROOFS (Nms)   get_stream_proofs_c
>>> CONTRIB_CHALLENGE      / <<< CONTRIB_CHALLENGE (Nms)       calculate_internal_contributions
```

Then aggregated `···` lines, one per step plus three summaries:

```
··· [phase1 wk9 pidx8 job7f3a] queue_wait      n=20 thread_total=  812.3ms mean_conc=0.55 median= 40.10ms p90= 51.02ms max= 88.40ms slowest=inst12@cpu7(P)
··· [phase1 wk9 pidx8 job7f3a] commit_witness  n=20 thread_total= 2511.4ms mean_conc=1.71 median=125.20ms p90=199.80ms max=331.10ms slowest=inst12@cpu7(P)
··· [phase1 wk9 pidx8 job7f3a] gpu_stream_wait n=20 thread_total=  190.2ms mean_conc=0.13 median=  8.90ms p90= 14.30ms max= 22.70ms slowest=inst5@cpu21(E)
··· [phase1 wk9 pidx8 job7f3a] cpu_placement topology=thread_siblings (16P+16E) thread_work=2701.6ms on_ecore=120.3ms (4.5%) migrations=3 cpus=[3P 7P 21E]
··· [phase1 wk9 pidx8 job7f3a] mul_tables calls=8412000 rows=105155834 updates=105155834 | big: n=1200 2.67ns/u over 900000 updates max_call=3.114ms | window: n=16000 11.67ns/u over 12000000 updates (incl. caller loop) | measured 12.3% of updates, clock_floor=18ns | zeroing=12.4ms over 4194304 slots
··· [phase1 wk9 pidx8 job7f3a] mul_tables_core P=11.11ns/u over 9000000 updates | E=10.87ns/u over 3900000 updates | P+E=12900000 of 12900000 measured (labels are only as good as the topology line)
··· [phase1 wk9 pidx8 job7f3a] total=1471.2ms rayon_threads=32 samples=29 work_cpu_freq min=1100MHz median=5400MHz (n=118) psi_delta_us cpu=48120 mem=0 io=311
```

(illustrative shape, not measured — see "Status" below)

## What lands in the trace file

`<ZISK_TRACE_DIR>/phase1-wk<worker>-<job_id>-<unix_ms>.jsonl`, one file per job (a
per-process sequence number replaces `<job_id>` when proofman runs outside the cluster):

- `{"t":"header",...}` — worker id, `job_id`, sequence, wall-clock start, total µs, rayon threads,
  P/E counts
- `{"t":"step",...}` — **per instance**: step, µs, offset within the phase, cpu in/out + P/E class, tid
- `{"t":"sample",...}` — every 50ms: every cpu's `scaling_cur_freq` and the three PSI counters
- `{"t":"mul_tables",...}` — the atomic accounting for the job

Volume is ~200 lines (~30 KB) per job per worker, so ~10 MB/hour at 330 jobs/hour. The files are
never pruned — rotate or delete them yourself between runs.

## Reading it

**Is the tail CPU or GPU?** Compare the `commit_witness` and `gpu_stream_wait` rows in the
slow-decile column of `phase1_report.py`. `commit_witness` is `get_contribution_air` — unpack,
LDE/Merkle, and enqueueing the GPU work; `gpu_stream_wait` is `wait_stream_commit_done_c`, pure
blocking on the stream the commit ran on. A tail in `queue_wait` means neither: the contribution
worker was starved because the witness side had nothing ready.

**Is it core placement?** `cpu_placement` gives the share of contribution work that ran on E-cores
and how often a thread was migrated mid-step. `mul_tables` gives ns/update split by P and E — the
direct test of "the same atomic loop costs more on an E-core". If a straggler job shows a jump in
E-core share or in E ns/update, the P→E migration hypothesis is confirmed for that job; if E share
is 0% and ns/update is flat, it is refuted for that job.

**Is it the atomics?** Needs the `phase1-trace` feature. `mul_tables` counts every `fetch_add` into
the shared multiplicity vectors across all four tables ([`std_virtual_table.rs`](pil2-components/lib/std/rs/src/std_virtual_table.rs),
`range_check/{u8air,u16air,specified_ranges}.rs`). `ns/update` is the contention signal: a few ns
uncontended, inflating as threads pile onto the same cache lines. `zeroing` is the per-job
`par_iter` that resets every table — a parallel store over millions of atomics that nothing was
measuring before. Counts come from per-thread accumulators flushed every 64k rows and timing from a
1-in-512 sample, so treat ns/update as good enough to compare P against E, not to the last decimal.

**Is it frequency?** `min_cpu_freq` is the floor over the whole phase from the 50ms sampler, not a
begin/end snapshot, so a dip that lasts one job is visible. `psi_delta_us cpu=` is runqueue
pressure — the thing that shows contention with no kernel log entry.

## Two switches, and why

| build | `ZISK_TRACE_PHASE1` | what you get | hot-path cost |
|---|---|---|---|
| default | unset | nothing | none measurable |
| default | `=1` | spans, per-instance CPU/GPU-wait split, placement, freq, PSI | none measurable |
| `--features phase1-trace` | `=1` | the above **plus** the multiplicity-table (atomic) accounting | the traced path, see below |
| `--features phase1-trace` | unset | nothing | **a runtime branch stays in the per-row loop — avoid this combination** |

The atomic accounting needs a cargo feature because its gate sits in the per-row multiplicity loop,
and `inc_virtual_row` reaches that loop **once per table row** (`keccakf.rs:226`,
`arith_full.rs:116/120`, `binary_basic.rs:903`, …) — tens of millions of calls per job. Measured on
an EPYC 7773X, 8–16 threads, 65536-entry table, 6 alternating rounds per variant:

```
baseline (pre-change)     1.9 - 2.2 ns/update
runtime bool gate, false  +0.25 .. +1.84 ns/update   =  +10 .. +74 ms/job at 40M updates
const-false gate          -0.04 .. +0.18 ns/update   =  inside the run-to-run noise band
```

The box is shared, so the spread is wide and the absolute numbers will not transfer to an
i9-14900K — but the ordering held in all four runs: a runtime bool in that loop is measurable, and
`const false &&` folding it away is not. Hence `phase1_trace::COMPILED`, a `const false` without
the feature, which also dead-codes `update_traced`.

## Cost when disabled

Every touched path, and what is left of it in a default build with `ZISK_TRACE_PHASE1` unset:

| path | frequency | left when off |
|---|---|---|
| `update` in the 4 multiplicity tables | **per row** | nothing: `COMPILED` is a `const false`, so the gate and the call to `update_traced` are both folded out |
| the `fetch_add` itself | per row | unchanged code. It lives in one `#[inline(always)] bump` used by both loops, so the untraced loop inlines to what it was before |
| `update_traced` | never | `#[cold]`, never called, off the hot icache path |
| contribution worker recv loop | per instance + once per 1ms poll | one `Option::is_some` branch. The `Instant::now()` and `sched_getcpu()` are inside the `map`, so neither runs |
| `get_contribution_air`, `wait_stream_commit_done_c`, `to_be_released_buffer` | per instance (177/job) | `phase1_trace::timed(&None, …)` matches `None` and calls the closure directly: one branch on an `Option` discriminant |
| `Phase1Trace::new` | once per job | one `OnceLock` read, returns `None`. No thread, no allocation |
| table zeroing in `execute` | once per job per table | one `OnceLock` read |
| `timer_*_phase1!` spans | 6 per job | one `OnceLock` read each; no `tracing` call |

The per-row row of that table is the one that was measured (above); the rest are per-job or
per-instance and argued from the code shape, not benchmarked — at 177 events per job there is
nothing there to measure. If you want the whole thing verified end to end, the cheap check is a
same-binary A/B: 30 min with the variable unset against the pre-instrumentation build, comparing
phase-1 medians (not means — the tail is what this exercise is about).

## Where the code is

| what | file:line |
|---|---|
| gate, topology, P/E class, collector, aggregation | `common/src/phase1_trace.rs` |
| `timer_start_phase1!` / `timer_stop_and_log_phase1!` | `common/src/phase1_trace.rs` |
| per-instance steps in the contribution workers | `proofman/src/proofman.rs`, `compute_contributions` loop |
| main-thread spans | `proofman/src/proofman.rs`, the `CALCULATING_CONTRIBUTIONS` block |
| atomic accounting | the four `update`/`update_traced` pairs in `pil2-components/lib/std/rs/src` |

Each traced table keeps its original loop untouched and gets a `#[cold] update_traced` twin; the
`fetch_add` itself lives in a single `#[inline(always)] bump` shared by both, so the two paths
cannot drift and the untraced build is byte-identical in that loop.

## Status

The instrumentation compiles and is off by default. **No measurements are included**: this was
written without access to the cluster (no `zisk-cluster-ansible`, no worker hosts reachable from
the machine it was developed on), so `PHASE1_FINDINGS.md` — the span table, the attribution, and
the §6.5 experiments — has to come from a run on the real workers.

Known gaps to close when you run it:

- **`queue_wait` is measured per `recv_timeout` call**, so it is a lower bound on starvation: the
  loop polls with a 1ms timeout, and only the call that returned an instance is recorded.
- **The 50ms sampler is one extra thread** while tracing is on. It reads sysfs, so it perturbs
  scheduling slightly — enough to matter if you are chasing a 1ms effect, not a 200ms one.
- **Targets #2 (the 107ms `EXECUTE` gap) and #4 (ASM chunk consumption) are not done.** They live
  in the zisk tree (`executor/src/executor.rs`, `prover-backend/src/prover/asm.rs`), not here.
