//! Profiling for the contributions phase.
//!
//! Motivation: `CALCULATING_CONTRIBUTIONS` had exactly one interior marker
//! (`First GPU contribution queued`), and that marker fires from a worker thread as
//! soon as the *first* instance is committed — which on the GPU path happens during
//! `exec`, long before the phase's own timers start. So "after first queue" covered
//! the CPU witness computation of every instance, all GPU commits, the drain and the
//! challenge aggregation, with no way to tell them apart. An observed ~0.8s excess
//! could have been in any of them.
//!
//! This module fixes that in three layers:
//!
//! 1. **Milestones** — wall-clock boundaries between the phase's stages, so the
//!    excess can be attributed to a stage without any per-instance detail.
//! 2. **Host waits** — the blocking spins that produce no output at all today:
//!    trace-buffer starvation, witness thread-token acquisition, GPU stream
//!    reservation, trace-H2D completion, the pinned-staging mutex.
//! 3. **Per-commit records** — drained from the C++ collector: which stream each
//!    instance landed on, whether its const-pols slot was warm, whether the
//!    custom-fixed file was re-read, which H2D path it took, and the device times.
//!
//! Volume: one summary line per job. The per-instance table is printed only when the
//! phase exceeded `PROOFMAN_CONTRIB_PROFILE_MS` (default 2400ms — above the fast mode,
//! below the slow one), so the ~4% of jobs that matter get full detail and the rest
//! cost one line.

use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

use proofman_starks_lib_c::{
    contrib_profile_drain_c, contrib_profile_reset_c, contrib_profile_totals_c, CommitProfileRecord,
    ContribProfileTotals, CONTRIB_FLAG_CUSTOM_RELOAD, CONTRIB_FLAG_H2D_DIRECT, CONTRIB_FLAG_SLOT_WARM,
    CONTRIB_FLAG_UNPACK_FIXED, CONTRIB_FLAG_WITNESS_HINTS,
};

/// Default detail-dump threshold. The measured fast mode is ~2.06s and the slow mode
/// ~2.88s, so 2400ms sits in the empty valley between them.
const DEFAULT_DETAIL_THRESHOLD_MS: u64 = 2400;

fn detail_threshold() -> Duration {
    let ms = std::env::var("PROOFMAN_CONTRIB_PROFILE_MS")
        .ok()
        .and_then(|v| v.parse::<u64>().ok())
        .unwrap_or(DEFAULT_DETAIL_THRESHOLD_MS);
    Duration::from_millis(ms)
}

/// Stage boundaries within the contributions phase, recorded on the driving thread.
#[derive(Default)]
struct Milestones {
    /// Set when the phase starts; every other field is an offset from it.
    start: Option<Instant>,
    /// End of `exec`. Commits already flow during exec (the witness pipeline is armed
    /// before it runs), so this is where "GPU work overlapped with execution" ends and
    /// the phase the coordinator calls Witness begins.
    exec_done: Option<Duration>,
    witness_enqueued: Option<Duration>,
    witness_joined: Option<Duration>,
    tables_done: Option<Duration>,
    commits_drained: Option<Duration>,
    stream_proofs_done: Option<Duration>,
    challenge_done: Option<Duration>,
}

/// Collects everything about one contributions phase. Cheap to update from any
/// thread: milestones are only touched by the driving thread, waits are atomics.
pub struct ContribProfile {
    milestones: std::sync::Mutex<Milestones>,
    /// Time the witness handler and the main witness loop spent acquiring CPU thread
    /// tokens from `rx_threads`. A 1ms-granularity poll loop with no logging today.
    thread_token_wait_ns: AtomicU64,
    thread_token_waits: AtomicU64,
}

impl Default for ContribProfile {
    fn default() -> Self {
        Self::new()
    }
}

impl ContribProfile {
    pub fn new() -> Self {
        Self {
            milestones: std::sync::Mutex::new(Milestones::default()),
            thread_token_wait_ns: AtomicU64::new(0),
            thread_token_waits: AtomicU64::new(0),
        }
    }

    /// Start a phase: zero the host counters and the C++ collector.
    pub fn begin(&self) {
        *self.milestones.lock().unwrap() = Milestones { start: Some(Instant::now()), ..Default::default() };
        self.thread_token_wait_ns.store(0, Ordering::Relaxed);
        self.thread_token_waits.store(0, Ordering::Relaxed);
        contrib_profile_reset_c();
    }

    pub fn record_thread_token_wait(&self, elapsed: Duration) {
        self.thread_token_wait_ns.fetch_add(elapsed.as_nanos() as u64, Ordering::Relaxed);
        self.thread_token_waits.fetch_add(1, Ordering::Relaxed);
    }

    fn mark(&self, f: impl FnOnce(&mut Milestones, Duration)) {
        let mut m = self.milestones.lock().unwrap();
        if let Some(start) = m.start {
            let elapsed = start.elapsed();
            f(&mut m, elapsed);
        }
    }

    pub fn mark_exec_done(&self) {
        self.mark(|m, d| m.exec_done = Some(d));
    }
    pub fn mark_witness_enqueued(&self) {
        self.mark(|m, d| m.witness_enqueued = Some(d));
    }
    pub fn mark_witness_joined(&self) {
        self.mark(|m, d| m.witness_joined = Some(d));
    }
    pub fn mark_tables_done(&self) {
        self.mark(|m, d| m.tables_done = Some(d));
    }
    pub fn mark_commits_drained(&self) {
        self.mark(|m, d| m.commits_drained = Some(d));
    }
    pub fn mark_stream_proofs_done(&self) {
        self.mark(|m, d| m.stream_proofs_done = Some(d));
    }
    pub fn mark_challenge_done(&self) {
        self.mark(|m, d| m.challenge_done = Some(d));
    }

    /// Emit the report. `buffer_wait` comes from the `MemoryHandler`, which owns that
    /// counter. Always logs one summary line; adds the per-instance table when the
    /// phase ran long (see `detail_threshold`).
    pub fn report(&self, buffer_wait: (Duration, u64, Duration)) {
        let m = self.milestones.lock().unwrap();
        let Some(start) = m.start else { return };
        let total = m.challenge_done.unwrap_or_else(|| start.elapsed());

        let totals = contrib_profile_totals_c();
        let records = contrib_profile_drain_c();

        let (buf_wait, buf_wait_n, buf_wait_max) = buffer_wait;
        let token_wait = Duration::from_nanos(self.thread_token_wait_ns.load(Ordering::Relaxed));
        let token_waits = self.thread_token_waits.load(Ordering::Relaxed);

        // Per-commit aggregates. `select_wait` is the one that says "the GPU was the
        // bottleneck": it is time a finished witness spent waiting for a stream.
        let n = records.len().max(1) as f64;
        let select_wait_total: u64 = records.iter().map(|r| r.select_wait_ns).sum();
        let select_wait_max = records.iter().map(|r| r.select_wait_ns).max().unwrap_or(0);
        let select_retries: u64 = records.iter().map(|r| r.select_retries).sum();
        let gpu_commit_total: f32 = records.iter().map(|r| r.gpu_commit_ms).sum();
        let gpu_commit_max = records.iter().map(|r| r.gpu_commit_ms).fold(0.0f32, f32::max);
        let warm = records.iter().filter(|r| r.flags & CONTRIB_FLAG_SLOT_WARM != 0).count();
        let unpacked = records.iter().filter(|r| r.flags & CONTRIB_FLAG_UNPACK_FIXED != 0).count();
        let custom_reload = records.iter().filter(|r| r.flags & CONTRIB_FLAG_CUSTOM_RELOAD != 0).count();
        let staged = records.iter().filter(|r| r.flags & CONTRIB_FLAG_H2D_DIRECT == 0).count();
        let hinted = records.iter().filter(|r| r.flags & CONTRIB_FLAG_WITNESS_HINTS != 0).count();

        let ms = |d: Option<Duration>| d.map(|d| d.as_secs_f64() * 1000.0).unwrap_or(f64::NAN);

        tracing::info!(
            "CONTRIB_PROFILE total={:.1}ms | stages(ms, cumulative from pipeline arm): exec={:.1} \
             wc_enqueue={:.1} wc_join={:.1} tables={:.1} drain={:.1} stream_proofs={:.1} challenge={:.1}",
            total.as_secs_f64() * 1000.0,
            ms(m.exec_done),
            ms(m.witness_enqueued),
            ms(m.witness_joined),
            ms(m.tables_done),
            ms(m.commits_drained),
            ms(m.stream_proofs_done),
            ms(m.challenge_done),
        );
        tracing::info!(
            "CONTRIB_PROFILE waits(ms): trace_buf={:.1}/n{} max={:.1} threads={:.1}/n{} \
             stream_sel={:.1} max={:.1} retries={} h2d_done={:.1}/n{} pinned_lock={:.1}/n{} events={:.1}/n{}",
            buf_wait.as_secs_f64() * 1000.0,
            buf_wait_n,
            buf_wait_max.as_secs_f64() * 1000.0,
            token_wait.as_secs_f64() * 1000.0,
            token_waits,
            select_wait_total as f64 / 1e6,
            select_wait_max as f64 / 1e6,
            select_retries,
            totals.h2d_wait_ns as f64 / 1e6,
            totals.h2d_wait_count,
            totals.pinned_lock_ns as f64 / 1e6,
            totals.pinned_lock_count,
            totals.event_churn_ns as f64 / 1e6,
            totals.event_churn_count,
        );
        tracing::info!(
            "CONTRIB_PROFILE commits={} dropped={} gpu_commit_sum={:.1}ms max={:.1}ms avg={:.1}ms | \
             slot_warm={}/{} unpack_fixed={} custom_reload={} h2d_staged={} witness_hints={} | \
             vram_free min={} last={} of {}",
            records.len(),
            totals.dropped,
            gpu_commit_total,
            gpu_commit_max,
            gpu_commit_total as f64 / n,
            warm,
            hinted,
            unpacked,
            custom_reload,
            staged,
            hinted,
            fmt_bytes(totals.gpu_free_min_bytes),
            fmt_bytes(totals.gpu_free_last_bytes),
            fmt_bytes(totals.gpu_total_bytes),
        );

        // The MO count-and-plan borrow. On a single-GPU worker the borrow window is a
        // hard stop for every commit, and the release wipes each stream's const-pols
        // affinity — so both the window length and the blocked-select count are prime
        // suspects for a step change in contributions time.
        if totals.borrow_count > 0 {
            tracing::info!(
                "CONTRIB_PROFILE gpu_borrow n={} drain={:.1}ms acq_sync={:.1}ms window={:.1}ms \
                 rel_sync={:.1}ms blocked_selects={} (~{:.1}ms of commit stall)",
                totals.borrow_count,
                totals.borrow_drain_ns as f64 / 1e6,
                totals.borrow_acq_sync_ns as f64 / 1e6,
                totals.borrow_window_ns as f64 / 1e6,
                totals.borrow_rel_sync_ns as f64 / 1e6,
                totals.borrow_blocked_selects,
                // Each blocked retry costs one 300us sleep.
                totals.borrow_blocked_selects as f64 * 0.3,
            );
        }

        if total >= detail_threshold() {
            self.report_detail(&records, &totals, total);
        }
    }

    /// Per-instance table, printed only for a slow phase. Sorted by device commit time
    /// so the outlier is the first row.
    fn report_detail(&self, records: &[CommitProfileRecord], totals: &ContribProfileTotals, total: Duration) {
        tracing::info!(
            "CONTRIB_PROFILE_DETAIL phase={:.1}ms over threshold — {} commits (instance airgroup:air stream \
             flags select_wait_ms enqueue_ms stage_ms gpu_commit_ms h2d ntt merkle exprs)",
            total.as_secs_f64() * 1000.0,
            records.len(),
        );
        let mut sorted: Vec<&CommitProfileRecord> = records.iter().collect();
        sorted.sort_by(|a, b| b.gpu_commit_ms.partial_cmp(&a.gpu_commit_ms).unwrap_or(std::cmp::Ordering::Equal));
        for r in sorted {
            let mut flags = String::new();
            if r.flags & CONTRIB_FLAG_WITNESS_HINTS != 0 {
                flags.push('H');
            }
            if r.flags & CONTRIB_FLAG_SLOT_WARM != 0 {
                flags.push('W');
            }
            if r.flags & CONTRIB_FLAG_UNPACK_FIXED != 0 {
                flags.push('U');
            }
            if r.flags & CONTRIB_FLAG_CUSTOM_RELOAD != 0 {
                flags.push('C');
            }
            flags.push(if r.flags & CONTRIB_FLAG_H2D_DIRECT != 0 { 'D' } else { 'S' });
            tracing::info!(
                "CONTRIB_COMMIT i={:<4} {}:{:<3} s={:<3} {:<6} sel={:>8.3} enq={:>8.3} stage={:>8.3} \
                 gpu={:>8.3} h2d={:>7.3} ntt={:>7.3} mt={:>7.3} exp={:>7.3}",
                r.instance_id,
                r.airgroup_id,
                r.air_id,
                r.stream_id,
                flags,
                r.select_wait_ns as f64 / 1e6,
                r.enqueue_ns as f64 / 1e6,
                r.h2d_stage_ns as f64 / 1e6,
                r.gpu_commit_ms,
                r.gpu_h2d_ms,
                r.gpu_ntt_ms,
                r.gpu_merkle_ms,
                r.gpu_exprs_ms,
            );
        }
        if totals.dropped > 0 {
            tracing::warn!(
                "CONTRIB_PROFILE_DETAIL {} commit records dropped (collector capacity exceeded); \
                 the table above is incomplete",
                totals.dropped
            );
        }
    }
}

fn fmt_bytes(b: u64) -> String {
    if b == 0 {
        return "n/a".to_string();
    }
    format!("{:.2}GB", b as f64 / (1024.0 * 1024.0 * 1024.0))
}
