#!/usr/bin/env python3
"""Aggregate the phase-1 contribution traces written by ZISK_TRACE_PHASE1=1.

    ./tools/phase1_report.py /var/lib/zisk-worker/traces            # one worker
    ./tools/phase1_report.py 'wk*/traces'                           # several, collected by worker

Per step, over the per-job totals: median, p99, max and the slowest-decile minus fastest-decile
delta -- the attribution method that localises variance. Per-step totals are summed across the
concurrent contribution workers, so they exceed the phase duration; mean_conc reports how many
workers that implies were busy on average. Also splits work by P-core / E-core (only as good as the
topology line) and reports the multiplicity-table (atomic) accounting.
"""
import glob
import json
import math
import os
import sys
from collections import defaultdict

STEPS = ["queue_wait", "commit_witness", "gpu_stream_wait", "release_buffer"]


def pct(sorted_values, p):
    if not sorted_values:
        return 0.0
    k = min(len(sorted_values) - 1, int(round(p * (len(sorted_values) - 1))))
    return sorted_values[k]


def sd(values):
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    return math.sqrt(sum((v - mean) ** 2 for v in values) / (len(values) - 1))


def load(paths):
    """One dict per job (trace file)."""
    jobs = []
    for path in paths:
        job = {"path": path, "header": None, "steps": defaultdict(list), "samples": [], "mul": None,
               "witness": None, "pools": None}
        try:
            with open(path) as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    rec = json.loads(line)
                    kind = rec.get("t")
                    if kind == "header":
                        job["header"] = rec
                    elif kind == "step":
                        job["steps"][rec["step"]].append(rec)
                    elif kind == "sample":
                        job["samples"].append(rec)
                    elif kind == "mul_tables":
                        job["mul"] = rec
                    elif kind == "witness_cpu":
                        job["witness"] = rec
                    elif kind == "pool_builds":
                        job["pools"] = rec
        except (OSError, json.JSONDecodeError) as e:
            print(f"skipping {path}: {e}", file=sys.stderr)
            continue
        if job["header"]:
            jobs.append(job)
    return jobs


def step_total_ms(job, step):
    return sum(r["dur_us"] for r in job["steps"].get(step, [])) / 1000.0


def worker_of(job):
    """Cluster WorkerId when the worker supplied one, else the partition index -- which is a
    lexicographic rank of hostnames and NOT a machine identity."""
    h = job["header"]
    label = h.get("worker_label") or ""
    return f"wk{label}" if label else f"pidx{h.get('partition_idx', h.get('worker_id', '?'))}"


def report(jobs):
    print(f"{len(jobs)} jobs")
    workers = sorted({worker_of(j) for j in jobs})
    print(f"workers: {workers}")
    if any(not j["header"].get("worker_label") for j in jobs):
        print("  WARNING: some traces carry no WorkerId, only a partition index -- do not join")
        print("           those on worker identity (worker-10 is partition 0, worker-2 is 3)")
    total_ms = sorted(j["header"]["total_us"] / 1000.0 for j in jobs)
    print(
        f"CALCULATING_CONTRIBUTIONS  median={pct(total_ms,0.5):.1f}ms  p90={pct(total_ms,0.9):.1f}ms  "
        f"p99={pct(total_ms,0.99):.1f}ms  max={total_ms[-1]:.1f}ms  sd={sd(total_ms):.1f}ms"
    )

    # slow decile vs fast decile, by the job's own total: this is what says which step carries
    # the straggler penalty
    ranked = sorted(jobs, key=lambda j: j["header"]["total_us"])
    decile = max(1, len(ranked) // 10)
    fast, slow = ranked[:decile], ranked[-decile:]

    print("\nper-step totals are summed across concurrent contribution workers, so they exceed the")
    print("phase duration; mean_conc = that sum / phase, i.e. how many workers were busy on average")
    print(f"\n{'step':<18}{'n/job':>7}{'median':>10}{'mean_conc':>10}{'p99':>9}{'max':>9}{'slow-fast':>11}")
    for step in STEPS:
        per_job = sorted(step_total_ms(j, step) for j in jobs)
        if not any(per_job):
            continue
        counts = [len(j["steps"].get(step, [])) for j in jobs]
        delta = (sum(step_total_ms(j, step) for j in slow) / len(slow)) - (
            sum(step_total_ms(j, step) for j in fast) / len(fast)
        )
        phase = [j["header"]["total_us"] / 1000.0 for j in jobs]
        conc = sum(per_job) / max(1e-9, sum(phase))
        print(
            f"{step:<18}{sum(counts)/len(counts):>7.0f}{pct(per_job,0.5):>9.1f}m{conc:>10.2f}"
            f"{pct(per_job,0.99):>8.1f}m{per_job[-1]:>8.1f}m{delta:>+10.1f}m"
        )
    unaccounted = [
        j["header"]["total_us"] / 1000.0 - sum(step_total_ms(j, s) for s in STEPS if s != "queue_wait") for j in jobs
    ]
    print(f"{'(main thread)':<18}{'':>7}{sorted(unaccounted)[len(unaccounted)//2]:>9.1f}m")

    # slowest instances, by summed non-wait work
    per_instance = defaultdict(float)
    for job in jobs:
        for step in STEPS:
            if step == "queue_wait":
                continue
            for rec in job["steps"].get(step, []):
                per_instance[rec["instance_id"]] += rec["dur_us"] / 1000.0
    worst = sorted(per_instance.items(), key=lambda kv: -kv[1])[:8]
    print("\nheaviest instances (total ms over all jobs): " + ", ".join(f"inst{i}={v:.0f}m" for i, v in worst))

    # P/E placement
    p_us = e_us = 0
    migrations = 0
    for job in jobs:
        for step in STEPS:
            if step == "queue_wait":
                continue
            for rec in job["steps"].get(step, []):
                if rec["cpu_in_class"] == "E":
                    e_us += rec["dur_us"]
                else:
                    p_us += rec["dur_us"]
                migrations += rec["cpu_in"] != rec["cpu_out"]
    work = p_us + e_us
    hdr = jobs[0]["header"]
    print(
        f"\nplacement: {hdr['n_cpus_perf']}P+{hdr['n_cpus_eff']}E ({hdr['topology_source']})  "
        f"on_P={p_us/1000:.0f}ms on_E={e_us/1000:.0f}ms ({100.0*e_us/work if work else 0:.1f}% on E)  "
        f"migrations={migrations}"
    )

    # frequency of the cpus that ran the work, while they ran it. A min over all 32 cpus just
    # reports the package floor of the idle E-cores (~790MHz) in every job, fast or slow.
    floors = []
    for job in jobs:
        seen = []
        for step in STEPS:
            if step == "queue_wait":
                continue
            for rec in job["steps"].get(step, []):
                cpu, lo, hi = rec["cpu_in"], rec["offset_us"], rec["offset_us"] + rec["dur_us"]
                for s in job["samples"]:
                    if lo <= s["offset_us"] <= hi and 0 <= cpu < len(s["freq_khz"]) and s["freq_khz"][cpu] > 0:
                        seen.append(s["freq_khz"][cpu] / 1000.0)
        if seen:
            floors.append(min(seen))
    if floors:
        floors.sort()
        print(f"working-cpu freq per job: median floor={pct(floors,0.5):.0f}MHz worst={floors[0]:.0f}MHz")

    # PSI: stalls that leave nothing in journalctl -k
    for res in ("psi_cpu_us", "psi_mem_us", "psi_io_us"):
        deltas = []
        for job in jobs:
            vals = [s[res] for s in job["samples"]]
            if len(vals) >= 2:
                deltas.append(vals[-1] - vals[0])
        if deltas and any(deltas):
            deltas.sort()
            print(f"{res} per job: median={pct(deltas,0.5):.0f} max={deltas[-1]:.0f}")

    # 6.1: waiting, or running slower? cpu_us is what the witness threads actually got on a CPU;
    # wall_us is how long the span took. Same work either way, so:
    #   cpu/wall constant across fast and slow jobs  -> executing slower (freq, IPC, bandwidth)
    #   cpu/wall falls on slow jobs                  -> descheduled, waiting for a core
    # Reported as a split between the slowest and fastest thirds *of the witness span itself*.
    wits = [j for j in jobs if j.get("witness") and j["witness"]["wall_us"]]
    if wits:
        wits.sort(key=lambda j: j["witness"]["wall_us"])
        third = max(1, len(wits) // 3)
        fast, slow = wits[:third], wits[-third:]

        def summarize(group):
            wall = sum(j["witness"]["wall_us"] for j in group) / len(group) / 1000.0
            cpu = sum(j["witness"]["cpu_us"] for j in group) / len(group) / 1000.0
            psi_some = sum(j["witness"]["psi_cpu_some_us"] for j in group) / len(group)
            psi_full = sum(j["witness"]["psi_cpu_full_us"] for j in group) / len(group)
            conc = sum(j["witness"]["max_active_loops"] for j in group) / len(group)
            return wall, cpu, cpu / wall if wall else 0.0, psi_some, psi_full, conc

        fw, fc, fr, fps, fpf, fcc = summarize(fast)
        sw, sc, sr, sps, spf, scc = summarize(slow)
        print(f"\n=== 6.1 waiting vs running slower ({len(wits)} jobs with a witness span) ===")
        print(f"{'':<10}{'wall':>10}{'cpu':>10}{'cores':>8}{'psi_some':>11}{'psi_full':>11}{'max_loops':>11}")
        print(f"{'fastest⅓':<10}{fw:>9.0f}m{fc:>9.0f}m{fr:>8.2f}{fps:>11.0f}{fpf:>11.0f}{fcc:>11.1f}")
        print(f"{'slowest⅓':<10}{sw:>9.0f}m{sc:>9.0f}m{sr:>8.2f}{sps:>11.0f}{spf:>11.0f}{scc:>11.1f}")
        wall_ratio = sw / fw if fw else 0.0
        cpu_ratio = sc / fc if fc else 0.0
        # What fraction of the extra wall time shows up as extra CPU time? The work is identical
        # every job, so 0 means the threads were off-cpu for the difference (waiting) and 1 means
        # they were on-cpu for all of it and simply needed more of it (running slower).
        growth = (cpu_ratio - 1.0) / (wall_ratio - 1.0) if wall_ratio > 1.0001 else float("nan")
        print(f"  slow/fast: wall x{wall_ratio:.2f}  cpu x{cpu_ratio:.2f}  cores {fr:.2f} -> {sr:.2f}")
        print(f"  extra wall time that appears as extra cpu time: {growth:.0%}")
        if wall_ratio < 1.05:
            print("  VERDICT: no meaningful spread in this set -- collect more, or this is not the span")
        elif growth <= 0.25:
            print("  VERDICT: cpu time flat while wall grew -> threads were WAITING for a core")
            print("           (next: who was runnable -- see the competitor table below)")
        elif growth >= 0.75:
            print("  VERDICT: cpu time grew with wall time -> threads were RUNNING SLOWER")
            print("           (same instructions, more cycles: frequency, IPC, cache/bandwidth)")
        else:
            print("  VERDICT: mixed -- part waiting, part slower; do not pick one yet")

        # who consumed the cores during the span, slow jobs vs fast
        def by_name(group):
            acc = defaultdict(float)
            for j in group:
                for t in j["witness"].get("by_thread_name", []):
                    acc[t["name"]] += t["cpu_us"] / len(group) / 1000.0
            return acc

        fast_names, slow_names = by_name(fast), by_name(slow)
        names = sorted(set(fast_names) | set(slow_names), key=lambda n: -slow_names.get(n, 0))
        print(f"\n  cpu-ms by thread name during the span (mean per job)")
        print(f"    {'thread':<24}{'fastest⅓':>12}{'slowest⅓':>12}{'delta':>10}")
        for n in names[:8]:
            f_, s_ = fast_names.get(n, 0.0), slow_names.get(n, 0.0)
            print(f"    {n:<24}{f_:>11.0f}m{s_:>11.0f}m{s_-f_:>+9.0f}m")
        scan = sum(j["witness"]["scan_us"] for j in wits) / len(wits) / 1000.0
        print(f"  observer cost: {scan:.1f}ms per job in /proc scans (2 per span)")

        # 6.3: achieved parallelism and the chunk tail
        loops = sorted(j["witness"]["max_active_loops"] for j in wits)
        chunks = sorted(j["witness"]["chunks_done"] for j in wits)
        entered = sorted(j["witness"]["loops_entered"] for j in wits)
        print(f"\n=== 6.3 collect parallelism ===")
        print(f"  worker_loops spawned median={pct(entered,0.5):.0f}  max simultaneously active median={pct(loops,0.5):.0f}")
        print(f"  chunks processed median={pct(chunks,0.5):.0f}")
        if pct(loops, 0.5) and pct(entered, 0.5) and pct(loops, 0.5) < pct(entered, 0.5) * 0.75:
            print("  NOTE: fewer loops were ever active than were spawned -- the pool is not saturated")

    # 6.4 the per-call pools
    pools = [j["pools"] for j in jobs if j.get("pools") and j["pools"]["n"]]
    if pools:
        n = sorted(p["n"] for p in pools)
        tot = sorted(p["total_ns"] / 1e6 for p in pools)
        mx = sorted(p["max_ns"] / 1e6 for p in pools)
        inside = sorted(p["in_witness_span"] for p in pools)
        print(f"\n=== 6.4 per-call rayon pools (MemCountersCursor::prepare) ===")
        print(f"  builds/job median={pct(n,0.5):.0f}  construction total median={pct(tot,0.5):.2f}ms max={mx[-1]:.2f}ms")
        print(f"  inside the witness span: median={pct(inside,0.5):.0f} of {pct(n,0.5):.0f}")

    # the atomics. Each class is divided by its own updates: mixing sampled nanoseconds with the
    # full update count (what the first version printed) gives a sub-ns figure that is meaningless.
    muls = [j["mul"] for j in jobs if j["mul"] and j["mul"].get("updates")]
    if muls:
        def per_update(items, ns_key, upd_key):
            vals = sorted(m[ns_key] / m[upd_key] for m in items if m.get(upd_key))
            return vals

        big = per_update(muls, "big_ns", "big_updates")
        win = per_update(muls, "win_ns", "win_updates")
        p = per_update(muls, "perf_ns", "perf_updates")
        e = per_update(muls, "eff_ns", "eff_updates")
        cov = sorted(
            100.0 * (m.get("big_updates", 0) + m.get("win_updates", 0)) / m["updates"] for m in muls
        )
        zero_ms = sorted(m["zero_ns"] / 1e6 for m in muls)
        clock = muls[0].get("clock_floor_ns", 0)
        print(f"\nmultiplicity tables: updates/job median={sorted(m['updates'] for m in muls)[len(muls)//2]:,}")
        print(f"  measured coverage: median={pct(cov,0.5):.1f}% of updates   clock floor={clock}ns")
        if big:
            print(f"  big calls (>=64 rows, timed individually): median={pct(big,0.5):.2f} p99={pct(big,0.99):.2f} ns/update")
        if win:
            print(f"  small calls (windowed, incl. caller loop):  median={pct(win,0.5):.2f} p99={pct(win,0.99):.2f} ns/update")
        if p and e:
            print(f"  on P-cores {pct(p,0.5):.2f} ns/u vs E-cores {pct(e,0.5):.2f} ns/u (measured subset;")
            print("    only as trustworthy as the topology line above)")
        elif p or e:
            print("  P/E split unavailable: all measurements landed on one core class")
        print(f"  per-job zeroing (par_iter store): median={pct(zero_ms,0.5):.1f}ms max={zero_ms[-1]:.1f}ms")

    # per-worker, so a rotating straggler is visible
    by_worker = defaultdict(list)
    for job in jobs:
        by_worker[worker_of(job)].append(job["header"]["total_us"] / 1000.0)
    if len(by_worker) > 1:
        print("\nper worker (CALCULATING_CONTRIBUTIONS ms):")
        for wk in sorted(by_worker):
            v = sorted(by_worker[wk])
            print(f"  {wk:<8} n={len(v):<5} median={pct(v,0.5):>7.1f} p99={pct(v,0.99):>7.1f} max={v[-1]:>7.1f}")


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return 1
    paths = []
    for arg in sys.argv[1:]:
        if os.path.isdir(arg):
            paths += glob.glob(os.path.join(arg, "phase1-*.jsonl"))
        else:
            paths += glob.glob(arg) + glob.glob(os.path.join(arg, "phase1-*.jsonl"))
    jobs = load(sorted(set(paths)))
    if not jobs:
        print("no trace files found", file=sys.stderr)
        return 1
    report(jobs)
    return 0


if __name__ == "__main__":
    sys.exit(main())
