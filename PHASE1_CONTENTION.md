# Phase-1 contention: instrument for §6, no findings yet

**Status: the measurements in this document have not been taken.** I have no access to the cluster
(no reachable worker hosts, no `zisk-cluster-ansible`), so what follows is the instrument for §6 and
the analysis that turns its output into the §8 answers — not the answers. Section 6 below states
exactly what has to be run and what each outcome would mean; §7 is the "could not determine" list
the brief asks for, written for the instrument rather than for the findings, because that is all
that can honestly be filled in from here.

The two broken instruments named in §7 of the brief were fixed first, in the previous change:
`min_cpu_freq` now samples only the CPUs the work ran on (§4 of the brief), and the `wkN` label is
now the cluster `WorkerId` with the partition index reported separately as `pidx`. Details in
[PHASE1_TRACE.md](PHASE1_TRACE.md).

---

## 1. The decisive measurement (§6.1)

`Phase1Trace::witness_span_begin/end` snapshot **every thread in the process** at both ends of
`CALCULATING_WITNESS`, from `/proc/self/task/<tid>/stat`: `utime+stime`, run state, and thread name.
No thread body is instrumented, which is deliberate — the competitor for the cores may be tokio,
cuda, or a pool this crate has never heard of, and all of them appear in this scan.

Journal line, once per job:

```
··· [phase1 wk9 pidx8 job5b713c89] witness_cpu wall=1420.3ms cpu=6002.1ms cores=4.23
    psi_cpu_some=334785us full=12010us runnable=8->13 loops=32 max_conc=21 chunks=1024
    scan_cost=5.1ms | top: zisk-worker=5640ms cuda00=300ms tokio-rt-worker=60ms
```

`tools/phase1_report.py` splits the jobs at the terciles **of the witness span itself** and computes
the one number that decides it:

```
extra wall time that appears as extra cpu time: <growth>
    growth = (cpu_slow/cpu_fast − 1) / (wall_slow/wall_fast − 1)
```

The work is byte-identical every job, so:

| growth | meaning | next step |
|---|---|---|
| ≤ 25% | threads were **off-cpu** — descheduled, waiting for a core | the competitor table on the same output, then §6.2 |
| ≥ 75% | threads were **on-cpu the whole time and needed more of it** — running slower | frequency (now measurable), IPC, cache/memory bandwidth |
| 25–75% | mixed | do not pick one; report both |

I verified the discriminator against synthetic traces built for all three cases — constant CPU with
growing wall, CPU growing proportionally, and a 50/50 split — and it returns `0%` / `100%` / `50%`
with the matching verdict. That tests the *analysis*, not the kernel accounting behind it.

**A verdict of "running slower" would be the first hypothesis in §3 that survives**, because §3
killed the waiting-side candidates (starvation, GPU wait, migration) and left the running-slower
side unmeasured. A verdict of "waiting" points at §6.2 with 0.76-of-24-cores average consumption
still to explain, which means the competitor is bursty and short-lived — look at `procs_running` in
the dense samples before believing any thread-name attribution.

## 2. Time resolution (§6.2)

The sampler now runs at **10ms while the witness span is open** and 50ms outside it
(`ZISK_TRACE_DENSE_MS` / `ZISK_TRACE_IDLE_MS`). Every dense tick records:

- `/proc/pressure/cpu` **some and full** — so PSI can be attributed to the sub-interval, and so a
  stall where *every* task was blocked is distinguishable from one where any was
- `/proc/stat procs_running` — machine-wide runnable count, catching competitors outside the process
- the `worker_loop` concurrency gauge (an atomic read, free)
- `in_witness`, so dense ticks can be separated from idle ones in analysis

Per-CPU frequency is only read on every 5th dense tick: 32 sysfs reads at 10ms would make the
sampler a competitor for the cores it is measuring. Span-boundary PSI is exact regardless of tick
alignment because it is read directly in `witness_span_begin/end`.

The per-thread state scan (`R` counts by thread name) is **not** on the 10ms path — it is ~166 file
reads, ~2.5ms, which at 10ms would be 25% of a core. It happens at the two span boundaries only,
and `scan_cost` in the output reports what it cost so the observer stays visible in the data.

## 3. Collect parallelism (§6.3)

`worker_loop_enter()` returns a guard that counts, in
[`executor/src/witness/collector.rs:362`](../zisk/executor/src/witness/collector.rs):

- `loops_entered` — how many of the 32 spawned loops actually ran
- `max_active_loops` — how many were ever simultaneously inside the loop (the high-water mark, not
  the pool size, which is what `rayon_threads=32` reports today)
- `chunks_done` — chunks drained, for the imbalance question

The report flags `max_active_loops < 0.75 × loops_entered` as an unsaturated pool. §1 of the brief
predicts uniformly slower loops rather than a tail; this measures it independently.

**Not measured:** the per-loop chunk *distribution*. The guard counts chunks per loop but only the
sum is aggregated, so "one loop finished long after the others" is still invisible. Fixing that
needs a per-loop histogram — see §7.

## 4. The per-call pool (§6.4)

[`mem_counters_cursor.rs:60`](../zisk/state-machines/mem/src/mem_counters_cursor.rs) builds a fresh
16-thread rayon pool on every `MemCountersCursor::prepare`. Now timed:

```
··· [phase1 wk9 pidx8 job5b713c89] pool_builds n=48 total=51.00ms max=4.20ms in_witness_span=48
```

`in_witness_span` is the interesting field: 16 thread spawns landing inside the witness window is a
plausible burst source that every average in §3 would have hidden. **This is a hypothesis to test,
not a fix to apply** — and note that even a large `total` here is a *correlate*, not a cause, until
§6.1 says whether the span was waiting or running slower.

## 5. How to run it

```sh
# proofman side must be pushed to the branch zisk pins first
cargo build --release --features phase1-trace      # feature only needed for the mul_tables line
```

```ini
Environment=ZISK_TRACE_PHASE1=1
```

Then, per §6.5 and §7 of the brief:

1. Capture `PROOF INSTANCES SUMMARY` from all nine workers at the start of the window. A restart
   re-derives the mapping; a changed mapping is a changed experiment.
2. Run ≥30 min (~330 jobs) with no other variable changed.
3. `./tools/phase1_report.py /var/lib/zisk-worker/traces` per worker, and again over worker-9 and
   worker-3 together — the 93/244 against 6/244 pair.
4. **Null run for the observer effect:** same build, `ZISK_TRACE_PHASE1` unset, ≥30 min, compare
   phase-1 **medians**. The trace adds two `/proc` scans (~5ms) and a 10ms sampler per job; that
   should be invisible against a 1.4s span, but it has not been demonstrated on the real host and
   the brief is right to insist.

## 6. What each outcome means for the next step

| §6.1 verdict | corroborating evidence to check on the same output | next measurement |
|---|---|---|
| waiting | `cores` falls; `procs_running` spikes in dense ticks; `psi_cpu_full` non-zero | who is runnable: per-thread `R` states at 10ms, which needs a cheaper scan than `/proc` (see §7) |
| running slower | `cores` holds; `work_cpu_freq min` drops on slow jobs; E-core share flat | perf counters — cycles, instructions, IPC per span; nothing short of that separates frequency from bandwidth |
| mixed | both partially | split the span: the collect scope against the rest of `calculate_witness` |

## 7. What I could not determine, and what it would take

- **Everything in §8.1–8.3 of the brief.** No cluster access from here; the instrument is untested
  against a real straggler. Every number in this document's examples is illustrative.
- **Whether the observer perturbs the result.** Argued from cost (~5ms of scans against a 1.4s
  span), not demonstrated. The null run in §5.4 settles it.
- **Per-loop chunk distribution** (§6.3). Needs a fixed-size per-loop histogram written at scope
  exit; the current guard aggregates to a sum. ~20 lines, in zisk.
- **Frequency vs cache/bandwidth**, if the verdict is "running slower". `scaling_cur_freq` sampling
  cannot separate a 20% frequency drop from a 20% IPC drop. That needs `perf_event_open` on cycles
  and instructions per span — a real addition, and the first thing here that would need care about
  permissions (`perf_event_paranoid`) and per-thread event inheritance.
- **Which thread group holds the cores at 10ms resolution.** The boundary scans give CPU *totals* by
  thread name across the whole span, so a competitor that burns 40ms in one burst and idles
  otherwise looks identical to one that trickles. Resolving that needs either a `taskstats`
  netlink reader or a `perf sched` capture, both heavier than anything here.
- **Whether the 166 threads matter at all.** §3 measured them idle over 10s. The boundary scan now
  measures them over the span instead, which is the right window — but if they turn out to be
  bursty within it, this instrument cannot see the burst either.

## 8. Cost when the trace is off

Unchanged from [PHASE1_TRACE.md](PHASE1_TRACE.md): the per-row multiplicity gate is a compile-time
`const false` without the cargo feature, and everything added here is behind
`Option`/`enabled()` checks at per-job or per-span frequency. The two new zisk call sites
(`worker_loop_enter`, `note_pool_build`) return immediately when tracing is off — `worker_loop_enter`
once per spawned loop (32 per collect), `note_pool_build` once per pool construction (~48 per job).
Neither is on a per-row path.
