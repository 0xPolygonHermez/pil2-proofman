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
//! Volume: three summary lines per job. The per-instance table is printed only when
//! CALCULATING_CONTRIBUTIONS exceeded `PROOFMAN_CONTRIB_PROFILE_MS` (default 2400ms —
//! above the observed fast mode, below the slow one), so the slow minority gets full
//! detail and the rest stays cheap. Note the gate is the phase duration, not the
//! pipeline-arm-relative `total` on the summary line, which also covers exec.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::OnceLock;
use std::time::{Duration, Instant};

use proofman_starks_lib_c::{
    contrib_profile_drain_c, contrib_profile_reset_c, contrib_profile_totals_c, CommitProfileRecord,
    ContribProfileTotals, CONTRIB_FLAG_CUSTOM_RELOAD, CONTRIB_FLAG_H2D_DIRECT, CONTRIB_FLAG_SLOT_WARM,
    CONTRIB_FLAG_UNPACK_FIXED, CONTRIB_FLAG_WITNESS_HINTS,
};

/// Which logical CPUs are efficiency cores.
///
/// Built once from `cpuinfo_max_freq`. On a hybrid part (e.g. i9-14900K: P at 5.7-6.0GHz,
/// E at 4.4GHz) the max-frequency groups separate the classes cleanly, so we take the
/// lowest group as E when it is meaningfully below the highest. A homogeneous machine
/// yields an empty E set and the P/E fields become trivially all-P.
struct CoreClasses {
    is_e: Vec<bool>,
    n_p: usize,
    n_e: usize,
}

impl CoreClasses {
    fn detect() -> Self {
        let mut freqs: Vec<u64> = Vec::new();
        for cpu in 0.. {
            let path = format!("/sys/devices/system/cpu/cpu{cpu}/cpufreq/cpuinfo_max_freq");
            match std::fs::read_to_string(&path) {
                Ok(v) => freqs.push(v.trim().parse::<u64>().unwrap_or(0)),
                Err(_) => break,
            }
        }
        let hi = freqs.iter().copied().max().unwrap_or(0);
        let lo = freqs.iter().copied().filter(|f| *f > 0).min().unwrap_or(0);
        // 0.9 is comfortably below any hybrid P/E ratio (4.4/6.0 = 0.73) and comfortably
        // above the spread within a class (5.7/6.0 = 0.95), so turbo-bin variation among
        // P-cores is not mistaken for a second class.
        let hybrid = hi > 0 && lo > 0 && (lo as f64) < 0.9 * hi as f64;
        let is_e: Vec<bool> = freqs.iter().map(|f| hybrid && *f > 0 && (*f as f64) < 0.9 * hi as f64).collect();
        let n_e = is_e.iter().filter(|e| **e).count();
        let n_p = is_e.len() - n_e;
        Self { is_e, n_p, n_e }
    }

    fn get() -> &'static CoreClasses {
        static CLASSES: OnceLock<CoreClasses> = OnceLock::new();
        CLASSES.get_or_init(CoreClasses::detect)
    }

    fn is_e_core(&self, cpu: i32) -> bool {
        cpu >= 0 && self.is_e.get(cpu as usize).copied().unwrap_or(false)
    }
}

/// `(user, sys)` CPU consumed by the whole process so far, or `None` if unavailable.
///
/// The discriminator between the two remaining mechanisms for the process-internal
/// slowdown. Serial steps inflate by the same factor as the 24-thread work while a
/// separate process (the ASM emulator) is untouched, so it is neither CPU-count contention
/// nor host-wide bandwidth — it is something shared inside this address space. Splitting
/// the phase's CPU says which:
///
/// - `sys` balloons => kernel work: page faults, mmap/munmap, allocator syscalls. The fix
///   is buffer reuse, arena tuning, or a different allocator.
/// - `user` balloons => genuine IPC loss: cache or memory-bandwidth pressure from this
///   process's own footprint, and `perf stat` is then the right follow-up.
pub fn process_cpu_user_sys() -> Option<(Duration, Duration)> {
    #[cfg(target_os = "linux")]
    {
        let mut usage: libc::rusage = unsafe { std::mem::zeroed() };
        // SAFETY: writes only into the local rusage.
        if unsafe { libc::getrusage(libc::RUSAGE_SELF, &mut usage) } != 0 {
            return None;
        }
        let tv = |t: libc::timeval| Duration::new(t.tv_sec as u64, (t.tv_usec as u32) * 1000);
        Some((tv(usage.ru_utime), tv(usage.ru_stime)))
    }
    #[cfg(not(target_os = "linux"))]
    None
}

/// `(minor_faults, major_faults)` for the process so far, or `None` if unavailable.
///
/// The counter the earlier memory refutation missed. That work checked `pgmajfault` and
/// direct-reclaim rates — all frozen — but never *minor* faults, which is what
/// first-touching freshly-mapped anonymous memory produces. Each minor fault costs a
/// kernel entry plus a page zeroing, charged to the process as CPU with terrible IPC, so a
/// burst of them looks exactly like "same instructions, 3.6x the cycles".
///
/// Relevant because the collect path allocates one `Vec::with_capacity` per
/// (chunk, instance), fills it during the replay, and frees it afterwards. Whether the
/// allocator hands those pages back to the kernel between jobs is state-dependent — which
/// is the shape of a memoryless, few-percent event.
pub fn page_faults() -> Option<(u64, u64)> {
    #[cfg(target_os = "linux")]
    {
        let mut usage: libc::rusage = unsafe { std::mem::zeroed() };
        // SAFETY: writes only into the local rusage.
        let rc = unsafe { libc::getrusage(libc::RUSAGE_SELF, &mut usage) };
        if rc != 0 {
            return None;
        }
        Some((usage.ru_minflt as u64, usage.ru_majflt as u64))
    }
    #[cfg(not(target_os = "linux"))]
    None
}

/// glibc allocator state: `(mmapped_bytes, in_use_bytes, free_in_arena_bytes)`.
///
/// Distinguishes "the allocator reused its arena" (cheap, no faults) from "the allocator
/// returned the pages and had to remap them" (a fault storm). `hblkhd` moving between jobs
/// is the direct evidence of that churn.
pub fn allocator_stats() -> Option<(u64, u64, u64)> {
    #[cfg(all(target_os = "linux", target_env = "gnu"))]
    {
        // SAFETY: mallinfo2 takes no arguments and returns a plain struct by value.
        let mi = unsafe { libc::mallinfo2() };
        Some((mi.hblkhd as u64, mi.uordblks as u64, mi.fordblks as u64))
    }
    #[cfg(not(all(target_os = "linux", target_env = "gnu")))]
    None
}

/// Is this logical CPU an efficiency core? Exposed so callers that run work on their own
/// thread pools (the executor's rayon pools) can classify the threads that actually do the
/// work — measuring the dispatcher thread instead is misleading, because it is blocked in
/// `install` while the pool runs.
pub fn is_e_core(cpu: i32) -> bool {
    CoreClasses::get().is_e_core(cpu)
}

/// `(n_p, n_e)` for this host.
pub fn core_class_counts() -> (usize, usize) {
    let c = CoreClasses::get();
    (c.n_p, c.n_e)
}

/// Consumed CPU time (user+system) of the whole process, or `None` if unavailable.
///
/// This is the spin-vs-wait discriminator. Wall-clock and summed per-task elapsed both
/// grow the same way whether a stage spins or blocks; only consumed CPU separates them.
/// If the phase's wall time rises and this rises with it, something is burning CPU; if
/// this stays flat, the stage is waiting.
pub fn process_cpu_time() -> Option<Duration> {
    #[cfg(target_os = "linux")]
    {
        let mut ts = libc::timespec { tv_sec: 0, tv_nsec: 0 };
        // SAFETY: writes only into the local timespec.
        let rc = unsafe { libc::clock_gettime(libc::CLOCK_PROCESS_CPUTIME_ID, &mut ts) };
        if rc != 0 {
            return None;
        }
        Some(Duration::new(ts.tv_sec as u64, ts.tv_nsec as u32))
    }
    #[cfg(not(target_os = "linux"))]
    None
}

/// Consumed CPU time of the calling thread, or `None` if unavailable. Pairs with the
/// wall-clock elapsed of the same task so each task carries its own cpu/wall ratio.
pub fn thread_cpu_time() -> Option<Duration> {
    #[cfg(target_os = "linux")]
    {
        let mut ts = libc::timespec { tv_sec: 0, tv_nsec: 0 };
        // SAFETY: writes only into the local timespec.
        let rc = unsafe { libc::clock_gettime(libc::CLOCK_THREAD_CPUTIME_ID, &mut ts) };
        if rc != 0 {
            return None;
        }
        Some(Duration::new(ts.tv_sec as u64, ts.tv_nsec as u32))
    }
    #[cfg(not(target_os = "linux"))]
    None
}

/// The logical CPU the caller is running on, or -1 if unavailable.
pub fn current_cpu() -> i32 {
    // SAFETY: sched_getcpu takes no arguments and only reads the calling thread's state.
    #[cfg(target_os = "linux")]
    unsafe {
        libc::sched_getcpu()
    }
    #[cfg(not(target_os = "linux"))]
    -1
}

/// Where a pipeline task ran and for how long. Aggregated per class so one job's
/// placement is a handful of numbers rather than a per-task dump.
#[derive(Default)]
struct Placement {
    tasks: AtomicU64,
    tasks_e: AtomicU64,
    /// WALL-clock elapsed, split by the core class the task finished on. Summed across
    /// concurrent tasks, so it exceeds the phase wall time — and it grows identically for
    /// a spin and for a block. Use `cpu_ns` to tell those apart.
    ns_p: AtomicU64,
    ns_e: AtomicU64,
    /// CONSUMED CPU of the same tasks. cpu/wall near 1 means the task was running; well
    /// below 1 means it was blocked.
    cpu_ns: AtomicU64,
    /// Task started on one core class and finished on the other: the scheduler moved it
    /// mid-flight, so its time is split and the class attribution is approximate.
    migrations: AtomicU64,
}

impl Placement {
    fn record(&self, cpu_start: i32, cpu_end: i32, elapsed: Duration, cpu: Option<Duration>) {
        let classes = CoreClasses::get();
        let e_start = classes.is_e_core(cpu_start);
        let e_end = classes.is_e_core(cpu_end);
        self.tasks.fetch_add(1, Ordering::Relaxed);
        if e_start != e_end {
            self.migrations.fetch_add(1, Ordering::Relaxed);
        }
        // Attribute on the finishing core: for a task that migrated, that is where it did
        // its most recent work, and the migration counter flags the ambiguity.
        if let Some(cpu) = cpu {
            self.cpu_ns.fetch_add(cpu.as_nanos() as u64, Ordering::Relaxed);
        }
        let ns = elapsed.as_nanos() as u64;
        if e_end {
            self.tasks_e.fetch_add(1, Ordering::Relaxed);
            self.ns_e.fetch_add(ns, Ordering::Relaxed);
        } else {
            self.ns_p.fetch_add(ns, Ordering::Relaxed);
        }
    }

    fn reset(&self) {
        self.tasks.store(0, Ordering::Relaxed);
        self.tasks_e.store(0, Ordering::Relaxed);
        self.ns_p.store(0, Ordering::Relaxed);
        self.ns_e.store(0, Ordering::Relaxed);
        self.cpu_ns.store(0, Ordering::Relaxed);
        self.migrations.store(0, Ordering::Relaxed);
    }

    /// `(tasks, tasks_on_e, wall_ms_on_p, wall_ms_on_e, cpu_ms, migrations)`
    fn snapshot(&self) -> (u64, u64, f64, f64, f64, u64) {
        (
            self.tasks.load(Ordering::Relaxed),
            self.tasks_e.load(Ordering::Relaxed),
            self.ns_p.load(Ordering::Relaxed) as f64 / 1e6,
            self.ns_e.load(Ordering::Relaxed) as f64 / 1e6,
            self.cpu_ns.load(Ordering::Relaxed) as f64 / 1e6,
            self.migrations.load(Ordering::Relaxed),
        )
    }
}

/// Default detail-dump threshold. The measured fast mode is ~2.06s and the slow mode
/// ~2.88s, so 2400ms sits in the empty valley between them.
const DEFAULT_DETAIL_THRESHOLD_MS: u64 = 2400;

/// Sample one in N sub-threshold (fast) jobs into the detail dump, so the per-commit
/// numbers have a baseline. Without this, every detail block is a slow job and there is
/// nothing to compare it against. 0 disables fast sampling.
const DEFAULT_FAST_SAMPLE_EVERY: u64 = 100;

fn fast_sample_every() -> u64 {
    std::env::var("PROOFMAN_CONTRIB_PROFILE_SAMPLE")
        .ok()
        .and_then(|v| v.parse::<u64>().ok())
        .unwrap_or(DEFAULT_FAST_SAMPLE_EVERY)
}

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
    /// Process CPU consumed at phase start. Differenced at report time to give the phase's
    /// own CPU cost — the one number that separates a spin from a block.
    cpu_at_start: Option<Duration>,
    /// user/sys split and minor faults at phase start, differenced the same way. These say
    /// whether the phase's extra CPU is kernel work or user-space IPC loss.
    user_sys_at_start: Option<(Duration, Duration)>,
    faults_at_start: Option<(u64, u64)>,
    /// End of `exec`. Commits already flow during exec (the witness pipeline is armed
    /// before it runs), so this is where "GPU work overlapped with execution" ends and
    /// the phase the coordinator calls Witness begins.
    exec_done: Option<Duration>,
    /// `set_publics_custom_commits` and the process-instance list, both of which sit
    /// inside the `wc_enqueue` window. Broken out because that stage carries ~91% of the
    /// slow-mode excess, and without these the stage has unattributed time in it.
    publics_done: Option<Duration>,
    instances_listed: Option<Duration>,
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
    /// The bulk `pre_calculate_witness` call. This is the body of the `wc_enqueue` stage,
    /// which carries ~91% of the slow-mode excess, so it needs its own number rather than
    /// being inferred from the stage boundary.
    pre_calculate_ns: AtomicU64,
    /// Blocking on `witness_done` for every per-instance witness to complete. This is the
    /// rest of `wc_enqueue` and, on the cluster, the bulk of it: the stage is not doing
    /// work, it is waiting for the witness threads whose core placement is recorded below.
    witness_done_wait_ns: AtomicU64,
    /// Core placement of the two CPU-heavy task kinds. On a hybrid part an E-core
    /// placement costs 1.5-2.5x per thread, which is the size of the observed excess —
    /// so this is what distinguishes "the host was slow" from "the host ran on slow
    /// cores".
    witness_placement: Placement,
    commit_placement: Placement,
    /// Jobs seen since construction, for sampling fast jobs into the detail dump.
    jobs_seen: AtomicU64,
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
            pre_calculate_ns: AtomicU64::new(0),
            witness_done_wait_ns: AtomicU64::new(0),
            witness_placement: Placement::default(),
            commit_placement: Placement::default(),
            jobs_seen: AtomicU64::new(0),
        }
    }

    /// Start a phase: zero the host counters and the C++ collector.
    pub fn begin(&self) {
        *self.milestones.lock().unwrap() = Milestones {
            start: Some(Instant::now()),
            cpu_at_start: process_cpu_time(),
            user_sys_at_start: process_cpu_user_sys(),
            faults_at_start: page_faults(),
            ..Default::default()
        };
        self.thread_token_wait_ns.store(0, Ordering::Relaxed);
        self.thread_token_waits.store(0, Ordering::Relaxed);
        self.pre_calculate_ns.store(0, Ordering::Relaxed);
        self.witness_done_wait_ns.store(0, Ordering::Relaxed);
        self.witness_placement.reset();
        self.commit_placement.reset();
        self.jobs_seen.fetch_add(1, Ordering::Relaxed);
        contrib_profile_reset_c();
    }

    pub fn record_pre_calculate(&self, elapsed: Duration) {
        self.pre_calculate_ns.fetch_add(elapsed.as_nanos() as u64, Ordering::Relaxed);
    }

    pub fn record_witness_done_wait(&self, elapsed: Duration) {
        self.witness_done_wait_ns.fetch_add(elapsed.as_nanos() as u64, Ordering::Relaxed);
    }

    /// One per-instance witness computation, with the CPU it started and finished on.
    pub fn record_witness_task(&self, cpu_start: i32, cpu_end: i32, elapsed: Duration, cpu: Option<Duration>) {
        self.witness_placement.record(cpu_start, cpu_end, elapsed, cpu);
    }

    /// One contribution commit enqueue, with the CPU it started and finished on.
    pub fn record_commit_task(&self, cpu_start: i32, cpu_end: i32, elapsed: Duration, cpu: Option<Duration>) {
        self.commit_placement.record(cpu_start, cpu_end, elapsed, cpu);
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
    pub fn mark_publics_done(&self) {
        self.mark(|m, d| m.publics_done = Some(d));
    }
    pub fn mark_instances_listed(&self) {
        self.mark(|m, d| m.instances_listed = Some(d));
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
    /// counter. Always logs the summary lines; adds the per-instance table when
    /// CALCULATING_CONTRIBUTIONS ran long (see `detail_threshold`).
    pub fn report(&self, buffer_wait: (Duration, u64, Duration)) {
        let m = self.milestones.lock().unwrap();
        let Some(start) = m.start else { return };
        let total = m.challenge_done.unwrap_or_else(|| start.elapsed());
        // The detail gate must use CALCULATING_CONTRIBUTIONS, not `total`: `total` runs
        // from the pipeline arm and so includes exec (~0.5s on the cluster), which would
        // put every job over a threshold picked from the observed witness modes.
        let contrib_phase = total.saturating_sub(m.exec_done.unwrap_or_default());

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

        // Increments, not just the cumulative offsets: the dominant stage should be
        // readable at a glance instead of requiring the reader to difference the fields.
        // `total` spans the whole pipeline (exec included); `phase` is
        // CALCULATING_CONTRIBUTIONS alone and is the value that matches its timer line.
        let marks = [
            m.exec_done,
            m.publics_done,
            m.instances_listed,
            m.witness_enqueued,
            m.witness_joined,
            m.tables_done,
            m.commits_drained,
            m.stream_proofs_done,
            m.challenge_done,
        ];
        let mut incr = [f64::NAN; 9];
        let mut prev = Duration::ZERO;
        for (i, mark) in marks.iter().enumerate() {
            if let Some(d) = mark {
                incr[i] = d.saturating_sub(prev).as_secs_f64() * 1000.0;
                prev = *d;
            }
        }
        tracing::info!(
            "CONTRIB_PROFILE total={:.1}ms (incl exec) phase={:.1}ms (== CALCULATING_CONTRIBUTIONS) | \
             stage_ms(incremental): exec={:.1} publics={:.1} instances={:.1} wc_enqueue={:.1} \
             wc_join={:.1} tables={:.1} drain={:.1} stream_proofs={:.1} challenge={:.1} \
             | wc_enqueue splits into pre_calculate={:.1} + await_witness={:.1} \
             | phase_cpu_ms={:.1} (user={:.1} sys={:.1}) phase_minflt={} majflt={}",
            total.as_secs_f64() * 1000.0,
            contrib_phase.as_secs_f64() * 1000.0,
            incr[0],
            incr[1],
            incr[2],
            incr[3],
            incr[4],
            incr[5],
            incr[6],
            incr[7],
            incr[8],
            self.pre_calculate_ns.load(Ordering::Relaxed) as f64 / 1e6,
            self.witness_done_wait_ns.load(Ordering::Relaxed) as f64 / 1e6,
            m.cpu_at_start
                .and_then(|c0| process_cpu_time().map(|c1| c1.saturating_sub(c0).as_secs_f64() * 1000.0))
                .unwrap_or(f64::NAN),
            m.user_sys_at_start
                .zip(process_cpu_user_sys())
                .map(|((u0, _), (u1, _))| u1.saturating_sub(u0).as_secs_f64() * 1000.0)
                .unwrap_or(f64::NAN),
            m.user_sys_at_start
                .zip(process_cpu_user_sys())
                .map(|((_, s0), (_, s1))| s1.saturating_sub(s0).as_secs_f64() * 1000.0)
                .unwrap_or(f64::NAN),
            m.faults_at_start.zip(page_faults()).map(|((a0, _), (a1, _))| a1.saturating_sub(a0)).unwrap_or(0),
            m.faults_at_start.zip(page_faults()).map(|((_, b0), (_, b1))| b1.saturating_sub(b0)).unwrap_or(0),
        );
        tracing::info!(
            "CONTRIB_PROFILE waits(ms, CROSS-THREAD SUMS not wall-clock; /nN = count): \
             trace_buf={:.1}/n{} max={:.1} threads={:.1}/n{} \
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
            "CONTRIB_PROFILE commits={} gpu_commit_sum={:.1}ms max={:.1}ms avg={:.1}ms | \
             slot_warm={}/{} unpack_fixed={} h2d_staged={}",
            records.len(),
            gpu_commit_total,
            gpu_commit_max,
            gpu_commit_total as f64 / n,
            warm,
            hinted,
            unpacked,
            staged,
        );

        // Core placement. The reason this is on the info line: on a hybrid CPU an E-core
        // placement is a 1.5-2.5x per-thread slowdown, the same size as the observed
        // slow-mode excess, and nothing else in the log can distinguish that from the host
        // simply having more work to do.
        let classes = CoreClasses::get();
        let (wc_tasks, wc_e, wc_ms_p, wc_ms_e, wc_cpu_ms, wc_mig) = self.witness_placement.snapshot();
        let (cm_tasks, cm_e, cm_ms_p, cm_ms_e, cm_cpu_ms, cm_mig) = self.commit_placement.snapshot();
        let pct = |e: f64, p: f64| if e + p > 0.0 { 100.0 * e / (e + p) } else { 0.0 };
        // cpu/wall near 1 => the task was on-CPU the whole time (a spin or real work);
        // well under 1 => it was blocked. This is what makes the wall sums interpretable.
        let ratio = |cpu: f64, wall: f64| if wall > 0.0 { cpu / wall } else { 0.0 };
        tracing::info!(
            "CONTRIB_PROFILE cores(P={} E={}): witness tasks={} on_e={} wall_ms_p={:.1} wall_ms_e={:.1} \
             e_share={:.1}% cpu_ms={:.1} cpu/wall={:.2} migrations={} | commit tasks={} on_e={} \
             wall_ms_p={:.1} wall_ms_e={:.1} e_share={:.1}% cpu_ms={:.1} cpu/wall={:.2} migrations={}",
            classes.n_p,
            classes.n_e,
            wc_tasks,
            wc_e,
            wc_ms_p,
            wc_ms_e,
            pct(wc_ms_e, wc_ms_p),
            wc_cpu_ms,
            ratio(wc_cpu_ms, wc_ms_p + wc_ms_e),
            wc_mig,
            cm_tasks,
            cm_e,
            cm_ms_p,
            cm_ms_e,
            pct(cm_ms_e, cm_ms_p),
            cm_cpu_ms,
            ratio(cm_cpu_ms, cm_ms_p + cm_ms_e),
            cm_mig,
        );

        // Constant across every record of the 2026-08-04 run, so they are noise on the
        // info line — but they are exactly what refuted the VRAM and buffer-contention
        // hypotheses, so keep them at debug and escalate any that stops being nominal.
        tracing::debug!(
            "CONTRIB_PROFILE steady: dropped={} custom_reload={} witness_hints={} \
             vram_free min={} last={} of {}",
            totals.dropped,
            custom_reload,
            hinted,
            fmt_bytes(totals.gpu_free_min_bytes),
            fmt_bytes(totals.gpu_free_last_bytes),
            fmt_bytes(totals.gpu_total_bytes),
        );
        if totals.dropped > 0 || custom_reload > 0 {
            tracing::warn!(
                "CONTRIB_PROFILE no longer nominal: dropped={} custom_reload={} — these were 0 in \
                 the baseline run, so the profile or the reuse path has changed",
                totals.dropped,
                custom_reload,
            );
        }

        // The MO count-and-plan borrow. On a single-GPU worker the borrow window is a
        // hard stop for every commit, and the release wipes each stream's const-pols
        // affinity — so both the window length and the blocked-select count are prime
        // suspects for a step change in contributions time.
        // Nominal on the baseline run (drain/acq_sync ~0, blocked_selects 0), so this is
        // only worth an info line when the borrow actually stalled commits.
        let borrow_costly = totals.borrow_blocked_selects > 0
            || totals.borrow_drain_ns > 5_000_000
            || totals.borrow_acq_sync_ns > 5_000_000;
        if totals.borrow_count > 0 && borrow_costly {
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
        } else if totals.borrow_count > 0 {
            tracing::debug!(
                "CONTRIB_PROFILE gpu_borrow n={} window={:.1}ms (nominal: no commits blocked)",
                totals.borrow_count,
                totals.borrow_window_ns as f64 / 1e6,
            );
        }

        // Dump detail for every slow phase, plus a sample of fast ones so the slow blocks
        // have something to be compared against.
        let over = contrib_phase >= detail_threshold();
        let every = fast_sample_every();
        let sampled = !over && every > 0 && self.jobs_seen.load(Ordering::Relaxed).is_multiple_of(every);
        if over || sampled {
            self.report_detail(&records, &totals, contrib_phase, if over { "over threshold" } else { "fast sample" });
        }
    }

    /// Per-instance table. Sorted by device commit time so the outlier is the first row.
    /// `reason` distinguishes a slow phase from a sampled fast one — the baseline blocks
    /// must be identifiable, or they get pooled with the slow ones in analysis.
    fn report_detail(
        &self,
        records: &[CommitProfileRecord],
        totals: &ContribProfileTotals,
        contrib_phase: Duration,
        reason: &str,
    ) {
        tracing::info!(
            "CONTRIB_PROFILE_DETAIL phase={:.1}ms {} — {} commits (instance airgroup:air stream \
             flags select_wait_ms enqueue_ms stage_ms gpu_commit_ms h2d ntt merkle exprs)",
            contrib_phase.as_secs_f64() * 1000.0,
            reason,
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
