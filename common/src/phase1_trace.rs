//! Opt-in tracing of the contributions phase (phase 1).
//!
//! Everything here is gated behind `ZISK_TRACE_PHASE1=1` and is a no-op when unset: the only cost
//! left in the build is one relaxed load of a `OnceLock<bool>` per span boundary.
//!
//! What it answers, per job: how the ~1.5s of `CALCULATING_CONTRIBUTIONS` splits between CPU work
//! and waiting on a GPU stream, which instances were slowest, and which CPU (P-core or E-core, at
//! what frequency) each piece of work ran on. The per-instance records go to a file under
//! `ZISK_TRACE_DIR` (default `/var/lib/zisk-worker/traces`); the journal only gets an aggregated
//! summary, in the existing `···` style.

use std::fmt::Write as _;
use std::fs::File;
use std::io::{BufWriter, Write as _};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

/// Same output as `timer_start_info!`, but only when phase-1 tracing is enabled. Lets the
/// contributions breakdown reach the journal at info level without turning on every other
/// debug span (the previous debug mode emitted hundreds of lines per job).
#[macro_export]
macro_rules! timer_start_phase1 {
    ($name:ident) => {
        #[allow(non_snake_case)]
        let $name = std::time::Instant::now();
        if $crate::phase1_trace::enabled() {
            tracing::info!(">>> {}", stringify!($name));
        }
    };
}

/// Counterpart of [`timer_start_phase1!`]
#[macro_export]
macro_rules! timer_stop_and_log_phase1 {
    ($name:ident) => {
        #[allow(non_snake_case)]
        let $name = std::time::Instant::now() - $name;
        if $crate::phase1_trace::enabled() {
            tracing::info!("<<< {} ({}ms)", stringify!($name), $name.as_millis());
        }
    };
}

/// Steps of the per-instance contribution work, as seen by a contribution worker thread
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ContribStep {
    /// Blocked in `contributions_rx.recv_timeout`, i.e. no witness ready to commit yet
    QueueWait,
    /// `get_contribution_air`: unpack + commit witness (CPU, plus enqueueing GPU work)
    CommitWitness,
    /// `wait_stream_commit_done_c`: blocked on the GPU stream that ran the commit
    GpuStreamWait,
    /// Returning the shared witness buffer to the memory handler
    ReleaseBuffer,
}

impl ContribStep {
    pub fn as_str(&self) -> &'static str {
        match self {
            ContribStep::QueueWait => "queue_wait",
            ContribStep::CommitWitness => "commit_witness",
            ContribStep::GpuStreamWait => "gpu_stream_wait",
            ContribStep::ReleaseBuffer => "release_buffer",
        }
    }
}

/// Whether the per-row multiplicity accounting is compiled in (feature `phase1-trace`).
///
/// `const false` in a normal build, so `if COMPILED && self.traced` in the multiplicity loops folds
/// away entirely and `update_traced` is dead code. That matters: measured on an EPYC 7773X, leaving
/// a runtime bool branch in that loop costs +0.45 ns/update, which is ~18ms per job at 40M updates.
/// Everything else in this module is per-instance or per-job and stays runtime-gated.
#[cfg(feature = "phase1-trace")]
pub const COMPILED: bool = true;
#[cfg(not(feature = "phase1-trace"))]
pub const COMPILED: bool = false;

/// Whether phase-1 tracing was requested. Read once; cheap enough to call per span boundary.
#[inline]
pub fn enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| matches!(std::env::var("ZISK_TRACE_PHASE1").as_deref(), Ok("1") | Ok("true")))
}

fn env_u64(name: &str, default: u64) -> u64 {
    std::env::var(name).ok().and_then(|v| v.parse().ok()).unwrap_or(default)
}

fn trace_dir() -> PathBuf {
    PathBuf::from(std::env::var("ZISK_TRACE_DIR").unwrap_or_else(|_| "/var/lib/zisk-worker/traces".to_string()))
}

/// Label of the job about to be traced, so its records can be joined to the coordinator's
/// `[Phase1] WorkerId(N) ... for JobId(...)` lines. Set by the cluster worker before it enters the
/// contribution phase; a proofman used standalone just leaves it empty.
///
/// A global rather than a `prove_phase` argument on purpose: this is diagnostics, and threading a
/// parameter through the proving API for it would be a worse trade.
pub fn set_job_tag(tag: &str) {
    if !enabled() {
        return;
    }
    if let Ok(mut current) = job_tag_slot().lock() {
        *current = sanitize(tag);
    }
}

/// The cluster's own `WorkerId` for this process, set once at worker startup.
///
/// Required for any cross-worker join: what `dctx` calls the worker index is the *partition* index
/// the coordinator handed out, which is a lexicographic rank of hostnames — `worker-10` sorts to
/// index 0, `worker-2` to index 3. Two of the nine happen to coincide with their hostname number,
/// so a spot-check of the wrong label looks correct.
pub fn set_worker_label(label: &str) {
    if !enabled() {
        return;
    }
    if let Ok(mut current) = worker_label_slot().lock() {
        *current = sanitize(label);
    }
}

/// Keep a label usable as a filename component and as a whitespace-delimited log field
fn sanitize(raw: &str) -> String {
    raw.chars().map(|c| if c.is_ascii_alphanumeric() || c == '-' || c == '_' { c } else { '_' }).collect()
}

fn job_tag_slot() -> &'static Mutex<String> {
    static TAG: OnceLock<Mutex<String>> = OnceLock::new();
    TAG.get_or_init(|| Mutex::new(String::new()))
}

fn worker_label_slot() -> &'static Mutex<String> {
    static LABEL: OnceLock<Mutex<String>> = OnceLock::new();
    LABEL.get_or_init(|| Mutex::new(String::new()))
}

fn job_tag() -> String {
    job_tag_slot().lock().map(|t| t.clone()).unwrap_or_default()
}

fn worker_label() -> String {
    worker_label_slot().lock().map(|t| t.clone()).unwrap_or_default()
}

/// Cost of one `Instant::now()` pair, measured once. Printed with the multiplicity numbers so a
/// per-update figure can be sanity-checked against the clock's own floor.
fn clock_overhead_ns() -> u64 {
    static OVERHEAD: OnceLock<u64> = OnceLock::new();
    *OVERHEAD.get_or_init(|| {
        let mut best = u64::MAX;
        for _ in 0..1000 {
            let start = Instant::now();
            let elapsed = start.elapsed();
            best = best.min(elapsed.as_nanos() as u64);
        }
        best
    })
}

/// CPU id this thread is running on right now, -1 if unavailable
#[inline]
pub fn current_cpu() -> i32 {
    // SAFETY: sched_getcpu takes no arguments and only reads the caller's CPU id
    unsafe { libc::sched_getcpu() }
}

/// P-core / E-core map, derived once from sysfs (never hardcoded: core counts and the id ranges
/// differ between hybrid parts, and the ranges are not contiguous on every kernel).
struct CpuTopology {
    /// index = cpu id, value = true when this cpu has the highest capacity class present
    is_perf: Vec<bool>,
    n_perf: usize,
    n_eff: usize,
    /// how the classification was obtained, for the summary line
    source: &'static str,
}

impl CpuTopology {
    fn detect() -> Self {
        let n_cpus = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1);

        // SMT first, because on Alder/Raptor Lake it is exact: P-cores are hyperthreaded (2 thread
        // siblings), E-cores are not (1). An i9-14900K gives 8 P-cores -> 16 P threads and 16
        // E-cores -> 16 E threads.
        let siblings: Vec<usize> = (0..n_cpus).map(Self::sibling_count).collect();
        if siblings.iter().any(|s| *s > 1) && siblings.contains(&1) {
            let is_perf: Vec<bool> = siblings.iter().map(|s| *s > 1).collect();
            let n_perf = is_perf.iter().filter(|p| **p).count();
            return Self { n_perf, n_eff: n_cpus - n_perf, is_perf, source: "thread_siblings" };
        }

        // Otherwise split a numeric attribute at its widest gap. Never `== max`: Turbo Boost Max
        // 3.0 gives the two favoured P-cores a higher cpuinfo_max_freq than their siblings, so
        // `== max` labelled 4 of 32 threads as P on a 16P+16E part.
        for (source, path) in
            [("cpu_capacity", "cpu_capacity"), ("cpuinfo_max_freq", "cpufreq/cpuinfo_max_freq")].into_iter()
        {
            let values: Vec<Option<u64>> = (0..n_cpus)
                .map(|cpu| {
                    std::fs::read_to_string(format!("/sys/devices/system/cpu/cpu{cpu}/{path}"))
                        .ok()
                        .and_then(|s| s.trim().parse::<u64>().ok())
                })
                .collect();
            let mut distinct: Vec<u64> = values.iter().flatten().copied().collect();
            distinct.sort_unstable();
            distinct.dedup();
            if distinct.is_empty() {
                continue;
            }
            if distinct.len() == 1 {
                return Self { is_perf: vec![true; n_cpus], n_perf: n_cpus, n_eff: 0, source: "uniform" };
            }

            let is_perf = split_by_widest_gap(&values, &distinct);
            let n_perf = is_perf.iter().filter(|p| **p).count();
            return Self { n_perf, n_eff: n_cpus - n_perf, is_perf, source };
        }

        Self { is_perf: vec![true; n_cpus], n_perf: n_cpus, n_eff: 0, source: "unknown" }
    }

    /// How many logical CPUs share this CPU's physical core
    fn sibling_count(cpu: usize) -> usize {
        for attr in ["topology/thread_siblings_list", "topology/core_cpus_list"] {
            if let Ok(text) = std::fs::read_to_string(format!("/sys/devices/system/cpu/cpu{cpu}/{attr}")) {
                let count: usize = text
                    .trim()
                    .split(',')
                    .map(|part| match part.split_once('-') {
                        // ranges are inclusive: "0-1" is two siblings
                        Some((lo, hi)) => match (lo.trim().parse::<usize>(), hi.trim().parse::<usize>()) {
                            (Ok(lo), Ok(hi)) if hi >= lo => hi - lo + 1,
                            _ => 0,
                        },
                        None => usize::from(!part.trim().is_empty()),
                    })
                    .sum();
                if count > 0 {
                    return count;
                }
            }
        }
        0
    }

    fn get() -> &'static CpuTopology {
        static TOPOLOGY: OnceLock<CpuTopology> = OnceLock::new();
        TOPOLOGY.get_or_init(CpuTopology::detect)
    }

    fn class(&self, cpu: i32) -> char {
        match usize::try_from(cpu).ok().and_then(|c| self.is_perf.get(c)) {
            Some(true) => 'P',
            Some(false) => 'E',
            None => '?',
        }
    }
}

/// Classify by splitting at the widest *relative* gap between distinct values: on an i9-14900K the
/// 4.4GHz E-core step against 5.7GHz P-cores dwarfs the 5.7 -> 6.0 step that Turbo Boost Max 3.0
/// gives the two favoured P-cores. `distinct` must be sorted ascending.
fn split_by_widest_gap(values: &[Option<u64>], distinct: &[u64]) -> Vec<bool> {
    let (mut split_at, mut widest) = (distinct[distinct.len() - 1], 0.0_f64);
    for pair in distinct.windows(2) {
        let ratio = pair[1] as f64 / pair[0] as f64;
        if ratio > widest {
            widest = ratio;
            split_at = pair[1];
        }
    }
    values.iter().map(|v| v.map(|v| v >= split_at).unwrap_or(true)).collect()
}

/// P/E class of a cpu id: 'P', 'E', or '?' when the topology could not be derived
#[inline]
pub fn cpu_class(cpu: i32) -> char {
    CpuTopology::get().class(cpu)
}

/// Contention accounting for the shared multiplicity tables (`VirtualTableAir`, `U8Air`, `U16Air`,
/// `SpecifiedRanges`): each of them does one `fetch_add(Relaxed)` per row into a `Vec<AtomicU64>`
/// that every witness thread writes to concurrently, so the cost per update depends on how badly
/// the cache line is contended — and on a hybrid CPU, on which core cluster the writer sits.
///
/// Timing is split by call size, because the two cases cannot be measured the same way:
///
/// * **big** calls (`size_hint >= MIN_TIMED_ROWS`) are timed individually — two clock reads spread
///   over at least 64 atomics is negligible, and the figure is clean.
/// * **small** calls, overwhelmingly the single-row `inc_virtual_row` in the state machines, cannot
///   be timed one by one: a `~5ns` operation measured with a `~20ns` clock is mostly clock. Instead
///   a *window* of `WINDOW_CALLS` consecutive small calls is timed with one clock pair, which is
///   unbiased across rows but includes whatever the caller does between calls.
///
/// Reporting each class against its own row count is the point: dividing sampled nanoseconds by
/// *all* updates (what the first version did) produced a sub-ns figure that was simply wrong.
///
/// Process-wide because the tables are process-wide; [`MulStats::reset`] is called when a trace
/// starts so the numbers belong to one job.
#[derive(Default)]
pub struct MulStats {
    // exact totals (to within one flush threshold per thread)
    calls: AtomicU64,
    rows: AtomicU64,
    updates: AtomicU64,
    // individually timed calls
    big_calls: AtomicU64,
    big_ns: AtomicU64,
    big_updates: AtomicU64,
    max_call_ns: AtomicU64,
    // window-timed small calls
    windows: AtomicU64,
    win_ns: AtomicU64,
    win_updates: AtomicU64,
    // measured time split by the core class it was measured on
    perf_ns: AtomicU64,
    perf_updates: AtomicU64,
    eff_ns: AtomicU64,
    eff_updates: AtomicU64,
    /// the `par_iter` that zeroes every table at the start of each job
    zero_ns: AtomicU64,
    zero_slots: AtomicU64,
}

/// Process-wide multiplicity-table accounting
pub fn mul_stats() -> &'static MulStats {
    static STATS: OnceLock<MulStats> = OnceLock::new();
    STATS.get_or_init(MulStats::default)
}

/// Per-thread accumulator. `RefCell` rather than `Cell` on purpose: this is mutated in place on the
/// per-row path, and copying the whole struct in and out per call would cost more than the atomic
/// being measured.
#[derive(Default)]
struct MulAcc {
    calls: u64,
    rows: u64,
    updates: u64,
    big_calls: u64,
    big_ns: u64,
    big_updates: u64,
    max_call_ns: u64,
    windows: u64,
    win_ns: u64,
    win_updates: u64,
    perf_ns: u64,
    perf_updates: u64,
    eff_ns: u64,
    eff_updates: u64,
    /// open window, if any
    win_start: Option<Instant>,
    win_calls: u64,
    win_pending_updates: u64,
}

/// Rows a thread accumulates before touching the shared counters
const FLUSH_ROWS: u64 = 1 << 16;
/// A call this size or larger is timed on its own
const MIN_TIMED_ROWS: usize = 64;
/// Small calls timed together under one clock pair
const WINDOW_CALLS: u64 = 512;

thread_local! {
    static MUL_ACC: std::cell::RefCell<MulAcc> = std::cell::RefCell::new(MulAcc::default());
}

/// What [`mul_end`] must do for a call, decided before its loop runs
#[derive(Clone, Copy, Default)]
pub struct MulTiming {
    /// set for a call being timed on its own
    big_start: Option<Instant>,
}

/// Call at the top of a traced `update`, with `iter.size_hint().0` as the row estimate.
#[inline]
pub fn mul_begin(rows_hint: usize) -> MulTiming {
    if rows_hint >= MIN_TIMED_ROWS {
        return MulTiming { big_start: Some(Instant::now()) };
    }
    // opening a window costs one clock read per WINDOW_CALLS calls
    MUL_ACC.with(|cell| {
        let mut acc = cell.borrow_mut();
        if acc.win_start.is_none() {
            acc.win_start = Some(Instant::now());
            acc.win_calls = 0;
            acc.win_pending_updates = 0;
        }
    });
    MulTiming::default()
}

/// Call at the bottom of a traced `update` with what the loop actually did
#[inline]
pub fn mul_end(timing: MulTiming, rows: u64, updates: u64) {
    MUL_ACC.with(|cell| {
        let mut acc = cell.borrow_mut();
        acc.calls += 1;
        acc.rows += rows;
        acc.updates += updates;

        match timing.big_start {
            Some(start) => {
                let ns = start.elapsed().as_nanos() as u64;
                acc.big_calls += 1;
                acc.big_ns += ns;
                acc.big_updates += updates;
                acc.max_call_ns = acc.max_call_ns.max(ns);
                attribute_core(&mut acc, ns, updates);
            }
            None => {
                acc.win_calls += 1;
                acc.win_pending_updates += updates;
                if acc.win_calls >= WINDOW_CALLS {
                    let ns = acc.win_start.map(|s| s.elapsed().as_nanos() as u64).unwrap_or(0);
                    let window_updates = acc.win_pending_updates;
                    acc.windows += 1;
                    acc.win_ns += ns;
                    acc.win_updates += window_updates;
                    attribute_core(&mut acc, ns, window_updates);
                    acc.win_start = None;
                    acc.win_calls = 0;
                    acc.win_pending_updates = 0;
                }
            }
        }

        if acc.rows >= FLUSH_ROWS {
            mul_stats().flush(&acc);
            acc.reset_flushed();
        }
    });
}

/// Attribute one measurement to the core class it was taken on. One `sched_getcpu` per big call or
/// per window, never per row.
#[inline]
fn attribute_core(acc: &mut MulAcc, ns: u64, updates: u64) {
    if cpu_class(current_cpu()) == 'E' {
        acc.eff_ns += ns;
        acc.eff_updates += updates;
    } else {
        acc.perf_ns += ns;
        acc.perf_updates += updates;
    }
}

impl MulAcc {
    /// Zero everything that was just flushed, keeping any window still open
    fn reset_flushed(&mut self) {
        let (win_start, win_calls, win_pending_updates) = (self.win_start, self.win_calls, self.win_pending_updates);
        *self = MulAcc { win_start, win_calls, win_pending_updates, ..Default::default() };
    }
}

/// Push whatever this thread has accumulated below the flush threshold. Called on the threads that
/// finish the phase; residue on other threads is lost, so counts are exact only to `FLUSH_ROWS`.
pub fn mul_flush_thread() {
    MUL_ACC.with(|cell| {
        let mut acc = cell.borrow_mut();
        if acc.calls > 0 {
            mul_stats().flush(&acc);
            acc.reset_flushed();
        }
    });
}

impl MulStats {
    pub fn reset(&self) {
        for counter in [
            &self.calls,
            &self.rows,
            &self.updates,
            &self.big_calls,
            &self.big_ns,
            &self.big_updates,
            &self.max_call_ns,
            &self.windows,
            &self.win_ns,
            &self.win_updates,
            &self.perf_ns,
            &self.perf_updates,
            &self.eff_ns,
            &self.eff_updates,
            &self.zero_ns,
            &self.zero_slots,
        ] {
            counter.store(0, Ordering::Relaxed);
        }
    }

    /// Flush one thread's accumulated counts into the shared totals. Called once per `FLUSH_ROWS`,
    /// not per call: `inc_virtual_row` reaches `update` once per table row, so touching these
    /// shared lines per call would add hundreds of ns to every row.
    fn flush(&self, acc: &MulAcc) {
        for (counter, value) in [
            (&self.calls, acc.calls),
            (&self.rows, acc.rows),
            (&self.updates, acc.updates),
            (&self.big_calls, acc.big_calls),
            (&self.big_ns, acc.big_ns),
            (&self.big_updates, acc.big_updates),
            (&self.windows, acc.windows),
            (&self.win_ns, acc.win_ns),
            (&self.win_updates, acc.win_updates),
            (&self.perf_ns, acc.perf_ns),
            (&self.perf_updates, acc.perf_updates),
            (&self.eff_ns, acc.eff_ns),
            (&self.eff_updates, acc.eff_updates),
        ] {
            if value > 0 {
                counter.fetch_add(value, Ordering::Relaxed);
            }
        }
        if acc.max_call_ns > 0 {
            self.max_call_ns.fetch_max(acc.max_call_ns, Ordering::Relaxed);
        }
    }

    /// The per-job zeroing of a table's multiplicities
    pub fn record_zeroing(&self, slots: u64, elapsed: Duration) {
        self.zero_slots.fetch_add(slots, Ordering::Relaxed);
        self.zero_ns.fetch_add(elapsed.as_nanos() as u64, Ordering::Relaxed);
    }

    /// ns per update, or `None` when nothing in that class was measured. Numerator and denominator
    /// always come from the same class — that is the whole point of the split.
    fn ns_per_update(ns: u64, updates: u64) -> Option<f64> {
        (updates > 0).then(|| ns as f64 / updates as f64)
    }

    fn fmt_ns_per_update(ns: u64, updates: u64) -> String {
        match Self::ns_per_update(ns, updates) {
            Some(v) => format!("{v:.2}"),
            None => "n/a".to_string(),
        }
    }

    fn report(&self, label: &str) {
        let (calls, rows) = (self.calls.load(Ordering::Relaxed), self.rows.load(Ordering::Relaxed));
        if calls == 0 {
            return;
        }
        let updates = self.updates.load(Ordering::Relaxed);
        let (big_calls, big_ns, big_updates) = (
            self.big_calls.load(Ordering::Relaxed),
            self.big_ns.load(Ordering::Relaxed),
            self.big_updates.load(Ordering::Relaxed),
        );
        let (windows, win_ns, win_updates) = (
            self.windows.load(Ordering::Relaxed),
            self.win_ns.load(Ordering::Relaxed),
            self.win_updates.load(Ordering::Relaxed),
        );
        let measured = big_updates + win_updates;
        let coverage = if updates > 0 { 100.0 * measured as f64 / updates as f64 } else { 0.0 };

        tracing::info!(
            "··· [phase1 {}] mul_tables calls={} rows={} updates={} | big: n={} {}ns/u over {} updates max_call={:.3}ms | window: n={} {}ns/u over {} updates (incl. caller loop) | measured {:.1}% of updates, clock_floor={}ns | zeroing={:.1}ms over {} slots",
            label,
            calls,
            rows,
            updates,
            big_calls,
            Self::fmt_ns_per_update(big_ns, big_updates),
            big_updates,
            self.max_call_ns.load(Ordering::Relaxed) as f64 / 1e6,
            windows,
            Self::fmt_ns_per_update(win_ns, win_updates),
            win_updates,
            coverage,
            clock_overhead_ns(),
            self.zero_ns.load(Ordering::Relaxed) as f64 / 1e6,
            self.zero_slots.load(Ordering::Relaxed),
        );

        // P against E over the measured subset only. Both sides come from the same mix of big and
        // window measurements, so the comparison is meaningful even though the absolute figure is
        // not the whole phase.
        let (perf_ns, perf_updates) = (self.perf_ns.load(Ordering::Relaxed), self.perf_updates.load(Ordering::Relaxed));
        let (eff_ns, eff_updates) = (self.eff_ns.load(Ordering::Relaxed), self.eff_updates.load(Ordering::Relaxed));
        tracing::info!(
            "··· [phase1 {}] mul_tables_core P={}ns/u over {} updates | E={}ns/u over {} updates | P+E={} of {} measured (labels are only as good as the topology line)",
            label,
            Self::fmt_ns_per_update(perf_ns, perf_updates),
            perf_updates,
            Self::fmt_ns_per_update(eff_ns, eff_updates),
            eff_updates,
            perf_updates + eff_updates,
            measured,
        );
    }

    fn json(&self) -> String {
        format!(
            r#"{{"t":"mul_tables","calls":{},"rows":{},"updates":{},"big_calls":{},"big_ns":{},"big_updates":{},"max_call_ns":{},"windows":{},"win_ns":{},"win_updates":{},"perf_ns":{},"perf_updates":{},"eff_ns":{},"eff_updates":{},"zero_ns":{},"zero_slots":{},"clock_floor_ns":{}}}"#,
            self.calls.load(Ordering::Relaxed),
            self.rows.load(Ordering::Relaxed),
            self.updates.load(Ordering::Relaxed),
            self.big_calls.load(Ordering::Relaxed),
            self.big_ns.load(Ordering::Relaxed),
            self.big_updates.load(Ordering::Relaxed),
            self.max_call_ns.load(Ordering::Relaxed),
            self.windows.load(Ordering::Relaxed),
            self.win_ns.load(Ordering::Relaxed),
            self.win_updates.load(Ordering::Relaxed),
            self.perf_ns.load(Ordering::Relaxed),
            self.perf_updates.load(Ordering::Relaxed),
            self.eff_ns.load(Ordering::Relaxed),
            self.eff_updates.load(Ordering::Relaxed),
            self.zero_ns.load(Ordering::Relaxed),
            self.zero_slots.load(Ordering::Relaxed),
            clock_overhead_ns(),
        )
    }
}

/// One measured step of one instance's contribution
struct Record {
    instance_id: usize,
    step: ContribStep,
    /// µs since the trace started
    offset_us: u64,
    dur_us: u64,
    cpu_in: i32,
    cpu_out: i32,
    thread: u64,
}

/// `/proc/pressure/<res>` "some" total, in µs. PSI is what shows runqueue/memory/io stalls that
/// leave no trace in `journalctl -k`.
fn psi_some_total(resource: &str) -> Option<u64> {
    let text = std::fs::read_to_string(format!("/proc/pressure/{resource}")).ok()?;
    for line in text.lines() {
        if let Some(rest) = line.strip_prefix("some ") {
            for field in rest.split_whitespace() {
                if let Some(v) = field.strip_prefix("total=") {
                    return v.parse().ok();
                }
            }
        }
    }
    None
}

fn freq_khz(cpu: usize) -> Option<u64> {
    std::fs::read_to_string(format!("/sys/devices/system/cpu/cpu{cpu}/cpufreq/scaling_cur_freq"))
        .ok()?
        .trim()
        .parse()
        .ok()
}

/// One 50ms tick of the sampler
struct Sample {
    offset_us: u64,
    psi: PsiSnapshot,
    /// machine-wide runnable count, to catch competitors outside this process
    procs_running: u64,
    /// threads inside `Collector::worker_loop` at this instant
    active_loops: u64,
    /// whether `CALCULATING_WITNESS` was open — dense ticks only happen while it is
    in_witness: bool,
    /// index = cpu id, kHz, 0 when unreadable; only filled on the slow ticks
    freq_khz: Vec<u32>,
}

#[derive(Default, Clone, Copy)]
struct PsiSnapshot {
    cpu: u64,
    memory: u64,
    io: u64,
}

impl PsiSnapshot {
    fn take() -> Self {
        Self {
            cpu: psi_some_total("cpu").unwrap_or(0),
            memory: psi_some_total("memory").unwrap_or(0),
            io: psi_some_total("io").unwrap_or(0),
        }
    }
}

/// PSI including the `full` line: `some` counts any task stalled, `full` counts intervals where
/// *every* runnable task was stalled. Distinguishing them matters for attributing a stall.
fn psi_totals(resource: &str) -> (u64, u64) {
    let mut some = 0;
    let mut full = 0;
    if let Ok(text) = std::fs::read_to_string(format!("/proc/pressure/{resource}")) {
        for line in text.lines() {
            let target = if line.starts_with("some ") {
                &mut some
            } else if line.starts_with("full ") {
                &mut full
            } else {
                continue;
            };
            for field in line.split_whitespace() {
                if let Some(v) = field.strip_prefix("total=") {
                    *target = v.parse().unwrap_or(0);
                }
            }
        }
    }
    (some, full)
}

/// Clock ticks per second, for turning `/proc` utime/stime into time
fn clock_ticks_per_sec() -> u64 {
    static HZ: OnceLock<u64> = OnceLock::new();
    // SAFETY: sysconf with a constant name, no arguments touched
    *HZ.get_or_init(|| unsafe { libc::sysconf(libc::_SC_CLK_TCK) }.max(1) as u64)
}

/// Per-thread CPU time and run state for the whole process, read from `/proc/self/task`.
///
/// This is the §6.1 instrument: taken at both ends of a span it says whether the threads doing the
/// work consumed *more CPU* (running slower — lower frequency, worse IPC, cache/bandwidth
/// contention) or the *same CPU over a longer wall time* (descheduled — waiting for a core). No
/// thread body is touched, and it covers threads this crate knows nothing about, which is the point:
/// the competitor for the cores may be tokio, cuda or the ad-hoc pools.
pub struct ThreadCpu {
    /// tid -> (comm, cpu ticks)
    threads: std::collections::HashMap<u32, (String, u64)>,
    /// how many threads were in state R at the instant of the scan
    runnable: usize,
    /// what the scan itself cost, so the observer is visible in the data
    scan_us: u64,
}

impl ThreadCpu {
    pub fn snapshot() -> Self {
        let started = Instant::now();
        let mut threads = std::collections::HashMap::with_capacity(256);
        let mut runnable = 0;
        if let Ok(entries) = std::fs::read_dir("/proc/self/task") {
            for entry in entries.flatten() {
                let tid: u32 = match entry.file_name().to_str().and_then(|n| n.parse().ok()) {
                    Some(tid) => tid,
                    None => continue,
                };
                let text = match std::fs::read_to_string(entry.path().join("stat")) {
                    Ok(text) => text,
                    // threads come and go mid-scan; a vanished one is not an error
                    Err(_) => continue,
                };
                // comm can contain spaces and parens, so split after the last ')'
                let (comm, rest) = match text.rfind(')') {
                    Some(close) => {
                        let open = text.find('(').map(|o| o + 1).unwrap_or(0);
                        (text[open..close].to_string(), &text[close + 1..])
                    }
                    None => continue,
                };
                let fields: Vec<&str> = rest.split_whitespace().collect();
                // after the comm: state, ppid, pgrp, session, tty, tpgid, flags, minflt, cminflt,
                // majflt, cmajflt, utime, stime -> utime is index 11
                if fields.first() == Some(&"R") {
                    runnable += 1;
                }
                let ticks = match (fields.get(11), fields.get(12)) {
                    (Some(u), Some(s)) => u.parse::<u64>().unwrap_or(0).saturating_add(s.parse::<u64>().unwrap_or(0)),
                    _ => 0,
                };
                threads.insert(tid, (comm, ticks));
            }
        }
        Self { threads, runnable, scan_us: started.elapsed().as_micros() as u64 }
    }

    /// CPU time consumed between two snapshots, in µs, grouped by thread name.
    ///
    /// Thread names, not tids: rayon's pool threads are unnamed (they inherit the process name) and
    /// tokio's are all `tokio-rt-worker`, so the group is what identifies a competitor.
    fn delta_by_name(&self, later: &ThreadCpu) -> Vec<(String, u64)> {
        let us_per_tick = 1_000_000 / clock_ticks_per_sec();
        let mut by_name: std::collections::HashMap<&str, u64> = std::collections::HashMap::new();
        for (tid, (name, then)) in &self.threads {
            if let Some((_, now)) = later.threads.get(tid) {
                let ticks = now.saturating_sub(*then);
                if ticks > 0 {
                    *by_name.entry(name.as_str()).or_insert(0) += ticks * us_per_tick;
                }
            }
        }
        // threads that appeared during the span: all of their CPU time belongs to it
        for (tid, (name, now)) in &later.threads {
            if !self.threads.contains_key(tid) && *now > 0 {
                *by_name.entry(name.as_str()).or_insert(0) += now * us_per_tick;
            }
        }
        let mut out: Vec<(String, u64)> = by_name.into_iter().map(|(k, v)| (k.to_string(), v)).collect();
        out.sort_by_key(|(_, us)| std::cmp::Reverse(*us));
        out
    }
}

/// Machine-wide runnable count, one cheap file read
fn procs_running() -> u64 {
    std::fs::read_to_string("/proc/stat")
        .ok()
        .and_then(|text| {
            text.lines().find_map(|l| l.strip_prefix("procs_running ")).and_then(|v| v.trim().parse::<u64>().ok())
        })
        .unwrap_or(0)
}

/// Gauges the witness collector and the ad-hoc pools feed, so the sampler can watch them without
/// knowing anything about zisk. All no-ops in a build without the feature.
#[derive(Default)]
pub struct CollectGauges {
    /// threads currently inside `Collector::worker_loop`
    active_loops: AtomicU64,
    /// high-water mark of the above
    max_active_loops: AtomicU64,
    /// chunks processed, and how they were spread over the loops
    chunks_done: AtomicU64,
    loops_entered: AtomicU64,
    /// per-call rayon pool construction (`MemCountersCursor::prepare`)
    pool_builds: AtomicU64,
    pool_build_ns: AtomicU64,
    pool_build_max_ns: AtomicU64,
    /// how many of those landed while the witness span was open
    pool_builds_in_span: AtomicU64,
}

pub fn gauges() -> &'static CollectGauges {
    static GAUGES: OnceLock<CollectGauges> = OnceLock::new();
    GAUGES.get_or_init(CollectGauges::default)
}

/// Set while `CALCULATING_WITNESS` is open: makes the sampler dense and attributes pool builds
static WITNESS_SPAN_OPEN: AtomicBool = AtomicBool::new(false);

impl CollectGauges {
    fn reset(&self) {
        for counter in [
            &self.active_loops,
            &self.max_active_loops,
            &self.chunks_done,
            &self.loops_entered,
            &self.pool_builds,
            &self.pool_build_ns,
            &self.pool_build_max_ns,
            &self.pool_builds_in_span,
        ] {
            counter.store(0, Ordering::Relaxed);
        }
    }
}

/// Entering `Collector::worker_loop`. Returns a guard so the count cannot leak on an early return.
pub fn worker_loop_enter() -> Option<WorkerLoopGuard> {
    if !enabled() {
        return None;
    }
    let g = gauges();
    let active = g.active_loops.fetch_add(1, Ordering::Relaxed) + 1;
    g.max_active_loops.fetch_max(active, Ordering::Relaxed);
    g.loops_entered.fetch_add(1, Ordering::Relaxed);
    Some(WorkerLoopGuard { chunks: 0 })
}

/// Decrements the active-loop gauge on drop and reports what the loop got through
pub struct WorkerLoopGuard {
    chunks: u64,
}

impl WorkerLoopGuard {
    /// One chunk taken by this loop
    #[inline]
    pub fn chunk_done(&mut self) {
        self.chunks += 1;
    }
}

impl Drop for WorkerLoopGuard {
    fn drop(&mut self) {
        let g = gauges();
        g.active_loops.fetch_sub(1, Ordering::Relaxed);
        g.chunks_done.fetch_add(self.chunks, Ordering::Relaxed);
    }
}

/// One construction of a per-call rayon pool
pub fn note_pool_build(elapsed: Duration) {
    if !enabled() {
        return;
    }
    let g = gauges();
    let ns = elapsed.as_nanos() as u64;
    g.pool_builds.fetch_add(1, Ordering::Relaxed);
    g.pool_build_ns.fetch_add(ns, Ordering::Relaxed);
    g.pool_build_max_ns.fetch_max(ns, Ordering::Relaxed);
    if WITNESS_SPAN_OPEN.load(Ordering::Relaxed) {
        g.pool_builds_in_span.fetch_add(1, Ordering::Relaxed);
    }
}

/// Captured at `CALCULATING_WITNESS` entry
struct WitnessSpanStart {
    at: Instant,
    cpu: ThreadCpu,
    psi_some: u64,
    psi_full: u64,
}

/// What a span cost in wall time against what its threads actually got on a CPU
struct SpanCpu {
    wall_us: u64,
    by_name: Vec<(String, u64)>,
    cpu_us_total: u64,
    psi_cpu_some_us: u64,
    psi_cpu_full_us: u64,
    runnable_at_start: usize,
    runnable_at_end: usize,
    scan_us: u64,
    max_active_loops: u64,
    loops_entered: u64,
    chunks_done: u64,
}

/// Collector for one job's contributions phase. Created at `CALCULATING_CONTRIBUTIONS` entry,
/// shared with the contribution worker threads, consumed by [`Phase1Trace::finish`].
pub struct Phase1Trace {
    /// the partition index from `dctx`, i.e. a lexicographic rank of hostnames — NOT the WorkerId
    partition_idx: i32,
    /// the cluster's WorkerId, when the worker set one
    worker_label: String,
    seq: u64,
    /// the cluster's JobId if the worker set one, else empty
    job_tag: String,
    start: Instant,
    start_unix_ms: u128,
    records: Mutex<Vec<Record>>,
    psi_start: PsiSnapshot,
    /// §6.1: state captured while `CALCULATING_WITNESS` is open
    witness_span: Mutex<Option<WitnessSpanStart>>,
    witness_cpu: Mutex<Option<SpanCpu>>,
    /// set by the frequency/PSI sampler thread when it should stop
    sampler_stop: Arc<AtomicBool>,
    sampler: Mutex<Option<std::thread::JoinHandle<Vec<Sample>>>>,
}

impl Phase1Trace {
    /// Log prefix. `wk` is the cluster's own WorkerId, `pidx` the partition index proofman was
    /// given — they are different numbering schemes and conflating them misattributes data to the
    /// wrong machine, so both are printed and neither is called just "worker".
    fn label(&self) -> String {
        let worker = if self.worker_label.is_empty() { "?".to_string() } else { self.worker_label.clone() };
        let job = if self.job_tag.is_empty() { format!("seq{}", self.seq) } else { format!("job{}", self.job_tag) };
        format!("wk{} pidx{} {}", worker, self.partition_idx, job)
    }

    /// `None` when tracing is off, so every call site collapses to an `Option` check
    pub fn new(partition_idx: i32) -> Option<Arc<Self>> {
        if !enabled() {
            return None;
        }
        static SEQ: AtomicU64 = AtomicU64::new(0);
        mul_stats().reset();
        gauges().reset();
        let sampler_stop = Arc::new(AtomicBool::new(false));
        let trace = Arc::new(Self {
            partition_idx,
            worker_label: worker_label(),
            seq: SEQ.fetch_add(1, Ordering::Relaxed),
            job_tag: job_tag(),
            start: Instant::now(),
            start_unix_ms: SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_millis()).unwrap_or(0),
            records: Mutex::new(Vec::with_capacity(512)),
            psi_start: PsiSnapshot::take(),
            witness_span: Mutex::new(None),
            witness_cpu: Mutex::new(None),
            sampler_stop: sampler_stop.clone(),
            sampler: Mutex::new(None),
        });
        *trace.sampler.lock().unwrap() = Some(Self::spawn_sampler(trace.start, sampler_stop));
        Some(trace)
    }

    /// Samples every cpu's current frequency and the PSI counters every 50ms for the duration of
    /// the phase: a p-state dip or a runqueue stall that lasts one job is invisible in a
    /// begin/end snapshot.
    fn spawn_sampler(start: Instant, stop: Arc<AtomicBool>) -> std::thread::JoinHandle<Vec<Sample>> {
        let n_cpus = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1);
        // 10ms while the witness span is open, 50ms outside it. The signal is bursty below 100ms,
        // so the dense window has to be dense — but only where it is needed, because a 32-file
        // frequency scan every 10ms would itself be a competitor for the cores.
        let dense_ms: u64 = env_u64("ZISK_TRACE_DENSE_MS", 10);
        let idle_ms: u64 = env_u64("ZISK_TRACE_IDLE_MS", 50);
        std::thread::spawn(move || {
            let mut out = Vec::with_capacity(256);
            let mut tick: u64 = 0;
            while !stop.load(Ordering::Relaxed) {
                let in_witness = WITNESS_SPAN_OPEN.load(Ordering::Relaxed);
                // the per-cpu frequency scan is the expensive part: every 5th dense tick
                let want_freq = !in_witness || tick.is_multiple_of(5);
                out.push(Sample {
                    offset_us: start.elapsed().as_micros() as u64,
                    psi: PsiSnapshot::take(),
                    procs_running: procs_running(),
                    active_loops: gauges().active_loops.load(Ordering::Relaxed),
                    in_witness,
                    freq_khz: if want_freq {
                        (0..n_cpus).map(|c| freq_khz(c).unwrap_or(0) as u32).collect()
                    } else {
                        Vec::new()
                    },
                });
                tick += 1;
                std::thread::sleep(Duration::from_millis(if in_witness { dense_ms } else { idle_ms }));
            }
            out
        })
    }

    /// Frequency of the CPUs that actually ran contribution work, while they were running it.
    ///
    /// A min over all CPUs is useless here: the idle E-cores sit at the package floor (~790MHz on
    /// these hosts) in every job, fast or slow, which is exactly what the first version reported.
    fn working_freqs_mhz(records: &[Record], samples: &[Sample]) -> Vec<u64> {
        let mut out = Vec::new();
        for rec in records.iter().filter(|r| r.step != ContribStep::QueueWait) {
            let cpu = match usize::try_from(rec.cpu_in) {
                Ok(cpu) => cpu,
                Err(_) => continue,
            };
            let (from, to) = (rec.offset_us, rec.offset_us + rec.dur_us);
            let mut overlapping = samples
                .iter()
                .filter(|s| !s.freq_khz.is_empty() && s.offset_us >= from && s.offset_us <= to)
                .peekable();
            if overlapping.peek().is_some() {
                for sample in overlapping {
                    if let Some(khz) = sample.freq_khz.get(cpu).copied().filter(|k| *k > 0) {
                        out.push(khz as u64 / 1000);
                    }
                }
            } else if let Some(nearest) = samples
                .iter()
                .filter(|s| !s.freq_khz.is_empty())
                .min_by_key(|s| s.offset_us.abs_diff(rec.offset_us + rec.dur_us / 2))
            {
                // steps shorter than the 50ms tick get the closest sample instead of nothing
                if let Some(khz) = nearest.freq_khz.get(cpu).copied().filter(|k| *k > 0) {
                    out.push(khz as u64 / 1000);
                }
            }
        }
        out.sort_unstable();
        out
    }

    /// Open the witness span: snapshot every thread's CPU time and PSI, and make the sampler dense.
    ///
    /// The two snapshots are what answer "waiting or running slower". Each costs one pass over
    /// `/proc/self/task` (~166 threads on the cluster workers); `scan_us` in the output says what
    /// that cost, so the observer stays visible.
    pub fn witness_span_begin(&self) {
        let (psi_some, psi_full) = psi_totals("cpu");
        let start = WitnessSpanStart { at: Instant::now(), cpu: ThreadCpu::snapshot(), psi_some, psi_full };
        *self.witness_span.lock().unwrap() = Some(start);
        WITNESS_SPAN_OPEN.store(true, Ordering::Relaxed);
    }

    /// Close the witness span and keep the deltas for the summary
    pub fn witness_span_end(&self) {
        WITNESS_SPAN_OPEN.store(false, Ordering::Relaxed);
        let start = match self.witness_span.lock().unwrap().take() {
            Some(start) => start,
            None => return,
        };
        let end_cpu = ThreadCpu::snapshot();
        let (psi_some, psi_full) = psi_totals("cpu");
        let by_name = start.cpu.delta_by_name(&end_cpu);
        let g = gauges();
        let span = SpanCpu {
            wall_us: start.at.elapsed().as_micros() as u64,
            cpu_us_total: by_name.iter().map(|(_, us)| *us).sum(),
            by_name,
            psi_cpu_some_us: psi_some.saturating_sub(start.psi_some),
            psi_cpu_full_us: psi_full.saturating_sub(start.psi_full),
            runnable_at_start: start.cpu.runnable,
            runnable_at_end: end_cpu.runnable,
            scan_us: start.cpu.scan_us + end_cpu.scan_us,
            max_active_loops: g.max_active_loops.load(Ordering::Relaxed),
            loops_entered: g.loops_entered.load(Ordering::Relaxed),
            chunks_done: g.chunks_done.load(Ordering::Relaxed),
        };
        *self.witness_cpu.lock().unwrap() = Some(span);
    }

    /// Record one step. `cpu_in`/`cpu_out` come from [`current_cpu`] around the measured section.
    pub fn record(&self, instance_id: usize, step: ContribStep, started: Instant, cpu_in: i32, cpu_out: i32) {
        let now = Instant::now();
        let rec = Record {
            instance_id,
            step,
            offset_us: started.saturating_duration_since(self.start).as_micros() as u64,
            dur_us: now.saturating_duration_since(started).as_micros() as u64,
            cpu_in,
            cpu_out,
            thread: thread_id(),
        };
        self.records.lock().unwrap().push(rec);
    }

    /// Emit the aggregated summary to the journal and the per-instance records to
    /// `<ZISK_TRACE_DIR>/phase1-wk<worker>-<seq>-<unix_ms>.jsonl`.
    pub fn finish(&self, total: Duration) {
        self.sampler_stop.store(true, Ordering::Relaxed);
        let samples = self.sampler.lock().unwrap().take().and_then(|h| h.join().ok()).unwrap_or_default();

        let records = self.records.lock().unwrap();
        let psi_end = PsiSnapshot::take();
        let psi_delta = PsiSnapshot {
            cpu: psi_end.cpu.saturating_sub(self.psi_start.cpu),
            memory: psi_end.memory.saturating_sub(self.psi_start.memory),
            io: psi_end.io.saturating_sub(self.psi_start.io),
        };

        for step in
            [ContribStep::QueueWait, ContribStep::CommitWitness, ContribStep::GpuStreamWait, ContribStep::ReleaseBuffer]
        {
            let mut durs: Vec<u64> = records.iter().filter(|r| r.step == step).map(|r| r.dur_us).collect();
            if durs.is_empty() {
                continue;
            }
            durs.sort_unstable();
            let n = durs.len();
            let total_us: u64 = durs.iter().sum();
            let slowest = records
                .iter()
                .filter(|r| r.step == step)
                .max_by_key(|r| r.dur_us)
                .map(|r| (r.instance_id, r.cpu_in, cpu_class(r.cpu_in)));
            let (slow_inst, slow_cpu, slow_class) = slowest.unwrap_or((usize::MAX, -1, '?'));
            // thread_total sums concurrent contribution workers, so it can and does exceed the
            // phase duration; mean_conc is that sum over the phase, i.e. how many workers were
            // busy on average. Neither is a share of the critical path on its own.
            let phase_us = total.as_micros().max(1) as f64;
            tracing::info!(
                "··· [phase1 {}] {:<16} n={:<4} thread_total={:>7.1}ms mean_conc={:>4.2} median={:>6.2}ms p90={:>6.2}ms max={:>7.2}ms slowest=inst{}@cpu{}({})",
                self.label(),
                step.as_str(),
                n,
                total_us as f64 / 1000.0,
                total_us as f64 / phase_us,
                durs[n / 2] as f64 / 1000.0,
                durs[n * 9 / 10] as f64 / 1000.0,
                durs[n - 1] as f64 / 1000.0,
                slow_inst,
                slow_cpu,
                slow_class,
            );
        }

        // CPU placement: what fraction of the measured work ran on E-cores, and how often a
        // thread was migrated mid-step. Both are invisible in the current logs.
        let topology = CpuTopology::get();
        let worked: Vec<&Record> = records.iter().filter(|r| r.step != ContribStep::QueueWait).collect();
        let work_us: u64 = worked.iter().map(|r| r.dur_us).sum();
        let ecore_us: u64 = worked.iter().filter(|r| cpu_class(r.cpu_in) == 'E').map(|r| r.dur_us).sum();
        let migrations = worked.iter().filter(|r| r.cpu_in != r.cpu_out).count();
        let cpus_used: std::collections::BTreeSet<i32> = worked.iter().map(|r| r.cpu_in).collect();
        let mut placement = String::new();
        for cpu in &cpus_used {
            let _ = write!(placement, "{}{} ", cpu, cpu_class(*cpu));
        }
        tracing::info!(
            "··· [phase1 {}] cpu_placement topology={} ({}P+{}E) thread_work={:.1}ms on_ecore={:.1}ms ({:.1}%) migrations={} cpus=[{}]",
            self.label(),
            topology.source,
            topology.n_perf,
            topology.n_eff,
            work_us as f64 / 1000.0,
            ecore_us as f64 / 1000.0,
            if work_us > 0 { 100.0 * ecore_us as f64 / work_us as f64 } else { 0.0 },
            migrations,
            placement.trim_end(),
        );

        // p-state / thermald evidence, restricted to the cpus that ran the work while they ran it
        let working = Self::working_freqs_mhz(&records, &samples);
        let (min_freq, median_freq, mean_freq) = match working.len() {
            0 => ("n/a".to_string(), "n/a".to_string(), "n/a".to_string()),
            n => (
                working[0].to_string(),
                working[n / 2].to_string(),
                // samples are at a fixed interval, so an unweighted mean over per-step
                // observations is already time-weighted across the span
                format!("{}", working.iter().sum::<u64>() / n as u64),
            ),
        };
        tracing::info!(
            "··· [phase1 {}] total={:.1}ms rayon_threads={} samples={} work_cpu_freq min={}MHz median={}MHz mean={}MHz (n={}) psi_delta_us cpu={} mem={} io={}",
            self.label(),
            total.as_secs_f64() * 1000.0,
            rayon::current_num_threads(),
            samples.len(),
            min_freq,
            median_freq,
            mean_freq,
            working.len(),
            psi_delta.cpu,
            psi_delta.memory,
            psi_delta.io,
        );

        // §6.1: the decisive line. cpu_us is what the threads actually got on a CPU during the
        // witness span; wall_us is how long it took. If cpu/wall holds between fast and slow jobs
        // the threads are running slower; if it drops, they were descheduled and waiting.
        if let Some(span) = self.witness_cpu.lock().unwrap().as_ref() {
            let wall_ms = span.wall_us as f64 / 1000.0;
            let cpu_ms = span.cpu_us_total as f64 / 1000.0;
            let mut top = String::new();
            for (name, us) in span.by_name.iter().take(4) {
                let _ = write!(top, "{}={:.0}ms ", name, *us as f64 / 1000.0);
            }
            tracing::info!(
                "··· [phase1 {}] witness_cpu wall={:.1}ms cpu={:.1}ms cores={:.2} psi_cpu_some={}us full={}us runnable={}->{} loops={} max_conc={} chunks={} scan_cost={:.1}ms | top: {}",
                self.label(),
                wall_ms,
                cpu_ms,
                if wall_ms > 0.0 { cpu_ms / wall_ms } else { 0.0 },
                span.psi_cpu_some_us,
                span.psi_cpu_full_us,
                span.runnable_at_start,
                span.runnable_at_end,
                span.loops_entered,
                span.max_active_loops,
                span.chunks_done,
                span.scan_us as f64 / 1000.0,
                top.trim_end(),
            );
        }

        // §6.4: the per-call pool. Reported whether or not it looks guilty — a zero here is a
        // result too.
        let g = gauges();
        let pool_builds = g.pool_builds.load(Ordering::Relaxed);
        if pool_builds > 0 {
            tracing::info!(
                "··· [phase1 {}] pool_builds n={} total={:.2}ms max={:.2}ms in_witness_span={}",
                self.label(),
                pool_builds,
                g.pool_build_ns.load(Ordering::Relaxed) as f64 / 1e6,
                g.pool_build_max_ns.load(Ordering::Relaxed) as f64 / 1e6,
                g.pool_builds_in_span.load(Ordering::Relaxed),
            );
        }

        mul_flush_thread();
        mul_stats().report(&self.label());

        if let Err(e) = self.write_jsonl(&records, &samples, total) {
            tracing::warn!("··· [phase1 {}] could not write trace file: {e}", self.label());
        }
    }

    fn write_jsonl(&self, records: &[Record], samples: &[Sample], total: Duration) -> std::io::Result<()> {
        let dir = trace_dir();
        std::fs::create_dir_all(&dir)?;
        let worker = if self.worker_label.is_empty() {
            format!("pidx{}", self.partition_idx)
        } else {
            format!("wk{}", self.worker_label)
        };
        let job = if self.job_tag.is_empty() { format!("seq{}", self.seq) } else { self.job_tag.clone() };
        let path = dir.join(format!("phase1-{}-{}-{}.jsonl", worker, job, self.start_unix_ms));
        let mut out = BufWriter::new(File::create(&path)?);
        writeln!(
            out,
            r#"{{"t":"header","worker_label":"{}","partition_idx":{},"seq":{},"job_id":"{}","start_unix_ms":{},"total_us":{},"rayon_threads":{},"n_cpus_perf":{},"n_cpus_eff":{},"topology_source":"{}"}}"#,
            self.worker_label,
            self.partition_idx,
            self.seq,
            self.job_tag,
            self.start_unix_ms,
            total.as_micros(),
            rayon::current_num_threads(),
            CpuTopology::get().n_perf,
            CpuTopology::get().n_eff,
            CpuTopology::get().source,
        )?;
        for r in records {
            writeln!(
                out,
                r#"{{"t":"step","step":"{}","instance_id":{},"offset_us":{},"dur_us":{},"cpu_in":{},"cpu_in_class":"{}","cpu_out":{},"cpu_out_class":"{}","thread":{}}}"#,
                r.step.as_str(),
                r.instance_id,
                r.offset_us,
                r.dur_us,
                r.cpu_in,
                cpu_class(r.cpu_in),
                r.cpu_out,
                cpu_class(r.cpu_out),
                r.thread,
            )?;
        }
        if let Some(span) = self.witness_cpu.lock().unwrap().as_ref() {
            let by_name: Vec<String> = span
                .by_name
                .iter()
                .map(|(name, us)| format!(r#"{{"name":"{}","cpu_us":{}}}"#, sanitize(name), us))
                .collect();
            writeln!(
                out,
                r#"{{"t":"witness_cpu","wall_us":{},"cpu_us":{},"psi_cpu_some_us":{},"psi_cpu_full_us":{},"runnable_start":{},"runnable_end":{},"loops_entered":{},"max_active_loops":{},"chunks_done":{},"scan_us":{},"by_thread_name":[{}]}}"#,
                span.wall_us,
                span.cpu_us_total,
                span.psi_cpu_some_us,
                span.psi_cpu_full_us,
                span.runnable_at_start,
                span.runnable_at_end,
                span.loops_entered,
                span.max_active_loops,
                span.chunks_done,
                span.scan_us,
                by_name.join(",")
            )?;
        }
        let g = gauges();
        writeln!(
            out,
            r#"{{"t":"pool_builds","n":{},"total_ns":{},"max_ns":{},"in_witness_span":{}}}"#,
            g.pool_builds.load(Ordering::Relaxed),
            g.pool_build_ns.load(Ordering::Relaxed),
            g.pool_build_max_ns.load(Ordering::Relaxed),
            g.pool_builds_in_span.load(Ordering::Relaxed),
        )?;
        for s in samples {
            let freqs: Vec<String> = s.freq_khz.iter().map(|k| k.to_string()).collect();
            writeln!(
                out,
                r#"{{"t":"sample","offset_us":{},"psi_cpu_us":{},"psi_mem_us":{},"psi_io_us":{},"procs_running":{},"active_loops":{},"in_witness":{},"freq_khz":[{}]}}"#,
                s.offset_us,
                s.psi.cpu,
                s.psi.memory,
                s.psi.io,
                s.procs_running,
                s.active_loops,
                s.in_witness,
                freqs.join(",")
            )?;
        }
        writeln!(out, "{}", mul_stats().json())?;
        out.flush()
    }
}

fn thread_id() -> u64 {
    // SAFETY: gettid takes no arguments and only reads the caller's tid
    unsafe { libc::syscall(libc::SYS_gettid) as u64 }
}

/// Times a section and records it, if `trace` is `Some`. Returns the section's value.
///
/// Keeps the two `sched_getcpu` calls and the `Instant`s out of the call sites, and out of the
/// build entirely when the trace is `None`.
#[inline]
pub fn timed<T>(trace: &Option<Arc<Phase1Trace>>, instance_id: usize, step: ContribStep, f: impl FnOnce() -> T) -> T {
    match trace {
        None => f(),
        Some(t) => {
            let cpu_in = current_cpu();
            let started = Instant::now();
            let out = f();
            let cpu_out = current_cpu();
            t.record(instance_id, step, started, cpu_in, cpu_out);
            out
        }
    }
}

#[cfg(test)]
mod tests {
    use super::split_by_widest_gap;

    fn classify(freqs: &[u64]) -> Vec<bool> {
        let values: Vec<Option<u64>> = freqs.iter().map(|f| Some(*f)).collect();
        let mut distinct: Vec<u64> = freqs.to_vec();
        distinct.sort_unstable();
        distinct.dedup();
        split_by_widest_gap(&values, &distinct)
    }

    /// The bug this replaces: `== max` labelled only the two Turbo-Boost-Max favoured P-cores as P,
    /// reporting 4P+28E on a part that is 16P+16E.
    #[test]
    fn favoured_p_cores_are_not_a_class_of_their_own() {
        // cpuinfo_max_freq as an i9-14900K reports it: 2 favoured P, 6 other P (both HT), 16 E
        let mut freqs = vec![6_000_000; 4];
        freqs.extend(std::iter::repeat_n(5_700_000, 12));
        freqs.extend(std::iter::repeat_n(4_400_000, 16));

        let is_perf = classify(&freqs);
        assert_eq!(is_perf.iter().filter(|p| **p).count(), 16, "16 P threads");
        assert_eq!(is_perf.iter().filter(|p| !**p).count(), 16, "16 E threads");
        assert!(is_perf[0] && is_perf[15], "both favoured and ordinary P-cores are P");
        assert!(!is_perf[16], "E-cores are E");
    }

    /// cpu_capacity, when the kernel exposes it, is the two-value case
    #[test]
    fn capacity_two_classes() {
        let freqs = [1024, 1024, 1024, 1024, 631, 631, 631, 631];
        let is_perf = classify(&freqs);
        assert_eq!(is_perf, vec![true, true, true, true, false, false, false, false]);
    }

    /// A non-hybrid machine must not be split at some incidental turbo difference
    #[test]
    fn uniform_machine_is_all_perf() {
        let is_perf = classify(&[3_500_000; 8]);
        assert!(is_perf.iter().all(|p| *p));
    }
}
