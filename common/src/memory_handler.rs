use crossbeam_channel::{bounded, Sender, Receiver};
use std::collections::{HashMap, HashSet};
use std::ffi::c_void;
use std::sync::{Arc, LazyLock, Mutex};
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::time::{Duration, Instant};
use crate::ProofCtx;
use proofman_fields::PrimeField64;
use crate::{ProofmanError, ProofmanResult};
use proofman_starks_lib_c::{register_host_memory_c, unregister_host_memory_c};

/// Ceiling on the re-check interval of a thread waiting for a pooled buffer. A released buffer
/// wakes it immediately, so this bounds cancel responsiveness, not pickup latency.
const MAX_POOL_WAIT_BACKOFF: Duration = Duration::from_millis(1);

/// Pool waits keyed by buffer pointer, not by thread: the block can happen on any worker a witness
/// component spawns, but the buffer always reaches one known instance.
static PENDING_WAITS: LazyLock<Mutex<HashMap<usize, Duration>>> = LazyLock::new(|| Mutex::new(HashMap::new()));

/// Called on release, so an entry nobody reads back cannot accumulate for the process's life.
fn forget_buffer_wait<F>(buffer: &[F]) {
    let key = buffer.as_ptr() as usize;
    PENDING_WAITS.lock().unwrap_or_else(|e| e.into_inner()).remove(&key);
}

/// Carry a wait onto a buffer that outlives the one it was incurred for, so a span that took
/// several pooled buffers reports their total.
pub fn charge_buffer_wait(ptr: *const u8, waited: Duration) {
    if waited.is_zero() {
        return;
    }
    let mut map = PENDING_WAITS.lock().unwrap_or_else(|e| e.into_inner());
    *map.entry(ptr as usize).or_insert(Duration::ZERO) += waited;
}

/// How long this buffer waited to be acquired, clearing the entry. `ZERO` if it never blocked.
pub fn take_buffer_wait(ptr: *const u8) -> Duration {
    let mut map = PENDING_WAITS.lock().unwrap_or_else(|e| e.into_inner());
    map.remove(&(ptr as usize)).unwrap_or(Duration::ZERO)
}

/// Read a buffer's charged wait without consuming it. `take_buffer_wait` is the consuming read one
/// timer makes; this is for a second reader that must not take the entry out from under it.
pub fn peek_buffer_wait(ptr: *const u8) -> Duration {
    let map = PENDING_WAITS.lock().unwrap_or_else(|e| e.into_inner());
    map.get(&(ptr as usize)).copied().unwrap_or(Duration::ZERO)
}

/// Round a host range out to page boundaries — `cudaHostRegister` requires the
/// region to cover whole pages.
fn aligned_host_range(ptr: usize, bytes: usize) -> Option<(usize, usize)> {
    if ptr == 0 || bytes == 0 {
        return None;
    }
    // `libc` is Linux-only (the only GPU-backend target); elsewhere pinning is a no-op, so a
    // default page size is fine.
    #[cfg(target_os = "linux")]
    let page_size = {
        let ps = unsafe { libc::sysconf(libc::_SC_PAGESIZE) };
        if ps > 0 {
            ps as usize
        } else {
            4096
        }
    };
    #[cfg(not(target_os = "linux"))]
    let page_size = 4096usize;
    let base = ptr & !(page_size - 1);
    let offset = ptr - base;
    let size = (bytes + offset + page_size - 1) & !(page_size - 1);
    Some((base, size))
}

/// Register a pool with all-or-nothing pinning: all buffers' pages pin (GPU) or none (CPU no-op);
/// a partial result (a `cudaHostRegister` failed) leaves an unpinned buffer in a pinned pool, so we
/// panic. Returns the distinct base pages (unregister each once on Drop); a registered buffer must
/// never be reallocated — registration pins a specific address (see `reset`). Dedups the base page
/// because `cudaHostRegister` rejects an already-registered one.
/// Pin a long-lived host buffer so H2D can DMA straight out of it instead of taking the staging path.
/// Returns the page base for [`unregister_host_buffer`]. Registration is not cheap: reused buffers only.
pub fn register_host_buffer<T>(buffer: &[T]) -> Option<usize> {
    let bytes = buffer.len().saturating_mul(std::mem::size_of::<T>());
    let (base, size) = aligned_host_range(buffer.as_ptr() as usize, bytes)?;
    register_host_memory_c(base as *mut c_void, size as u64).then_some(base)
}

/// Release a base from [`register_host_buffer`]. Must run before the buffer is freed.
pub fn unregister_host_buffer(base: usize) {
    unregister_host_memory_c(base as *mut c_void);
}

fn register_pool<F: PrimeField64>(buffers: &[Vec<F>]) -> Vec<usize> {
    let mut registered: Vec<usize> = Vec::with_capacity(buffers.len());
    let mut covered: HashSet<usize> = HashSet::with_capacity(buffers.len());
    let mut all_covered = true;
    for buffer in buffers.iter() {
        // Compute the page range once and use that exact base for both the dedup
        // check and the registration, so the two can never diverge.
        let bytes = buffer.len().saturating_mul(std::mem::size_of::<F>());
        match aligned_host_range(buffer.as_ptr() as usize, bytes) {
            // Page already pinned by an earlier buffer in this pool — skip the
            // duplicate cudaHostRegister; it is covered.
            Some((base, _)) if covered.contains(&base) => {}
            Some((base, size)) => {
                if register_host_memory_c(base as *mut c_void, size as u64) {
                    covered.insert(base);
                    registered.push(base);
                } else {
                    all_covered = false;
                }
            }
            None => all_covered = false,
        }
    }
    // Partial pinning is fatal, but unregister what we did pin before aborting — else those pages
    // stay locked (and under panic=abort, Drop never runs to release them).
    if !registered.is_empty() && !all_covered {
        for ptr in &registered {
            unregister_host_memory_c(*ptr as *mut c_void);
        }
        panic!(
            "MemoryHandler: host-memory pinning is all-or-nothing, but only {} of {} buffers' pages pinned. \
             The GPU backend is active and a cudaHostRegister failed — refusing to run with a \
             partially-pinned pool.",
            covered.len(),
            buffers.len()
        );
    }
    registered
}

/// Single fixed-size buffer pool over a bounded channel (internal to `MemoryHandlerRecursive`).
/// `take()` waits on the channel with a backoff timeout so the abort path (`cancelled`) can wake it.
struct Pool<F: PrimeField64 + Send + Sync + 'static> {
    /// Names this pool in the wait diagnostics.
    name: &'static str,
    sender: Sender<Vec<F>>,
    receiver: Receiver<Vec<F>>,
    n_buffers: usize,
    buffer_size: usize,
    /// Distinct page-locked base pages the pool registered. Empty iff pinning is
    /// disabled (CPU backend). Used only to unregister on Drop (one call per page).
    registered_buffers: Vec<usize>,
    /// Shared with the owning `MemoryHandlerRecursive`; set on the abort path so a
    /// blocking `take()` exits instead of parking forever on `recv`.
    cancelled: Arc<AtomicBool>,
    /// Data pointers of the pool's OWN buffers — only these are re-pooled on release (a cancel-escape
    /// buffer is freed instead), so the pool stays at exactly N pinned buffers and `release` never blocks.
    original_ptrs: HashSet<usize>,
    /// Blocked time and count. Unmeasured, it is charged to whichever timer wraps the caller.
    wait_ns: AtomicU64,
    wait_count: AtomicUsize,
    /// Buffers the last `reset` found missing. While non-zero a waiter gets a fresh buffer instead of
    /// parking on one that is never coming back; a later clean `reset` clears it.
    deficit: AtomicUsize,
}

impl<F: PrimeField64 + Send + Sync + 'static> Pool<F> {
    fn new(name: &'static str, n_buffers: usize, buffer_size: usize, pin: bool, cancelled: Arc<AtomicBool>) -> Self {
        let (sender, receiver) = bounded(n_buffers);
        let buffers: Vec<Vec<F>> = (0..n_buffers).map(|_| vec![F::ZERO; buffer_size]).collect();
        // A zero-sized pool has nothing to pin: an empty Vec's pointer is dangling, not a page.
        let registered_buffers: Vec<usize> = if pin && buffer_size > 0 { register_pool(&buffers) } else { Vec::new() };
        // Record our own buffers by pointer; they're never reallocated, so the pointers stay stable.
        let original_ptrs: HashSet<usize> = buffers.iter().map(|b| b.as_ptr() as usize).collect();
        for buffer in buffers {
            sender.send(buffer).unwrap();
        }
        Self {
            name,
            sender,
            receiver,
            n_buffers,
            buffer_size,
            registered_buffers,
            cancelled,
            original_ptrs,
            deficit: AtomicUsize::new(0),
            wait_ns: AtomicU64::new(0),
            wait_count: AtomicUsize::new(0),
        }
    }

    fn wait_stats(&self) -> (Duration, usize) {
        (Duration::from_nanos(self.wait_ns.load(Ordering::Relaxed)), self.wait_count.load(Ordering::Relaxed))
    }

    /// The callers' reported waits must sum to this; short means blocking is not reaching a span.
    fn log_wait(&self) {
        let (waited, blocked) = self.wait_stats();
        if blocked > 0 {
            tracing::debug!(
                "Pool '{}' ({} buffers): {} acquisitions blocked, {:.3}s total",
                self.name,
                self.n_buffers,
                blocked,
                waited.as_secs_f64()
            );
        }
    }

    /// Charge to the pool total and to the buffer, so the caller holding it can subtract the wait.
    fn charge_wait<T>(&self, buffer: &[T], waited: Duration) {
        self.wait_ns.fetch_add(waited.as_nanos() as u64, Ordering::Relaxed);
        self.wait_count.fetch_add(1, Ordering::Relaxed);
        charge_buffer_wait(buffer.as_ptr() as *const u8, waited);
    }

    fn take(&self) -> Vec<F> {
        // The timeout only paces the abort-flag re-check, so it escalates: a fixed 100us cost
        // 10k wakeups/s per waiter, exactly while the pool was exhausted.
        let mut backoff = Duration::from_micros(100);
        // Fast path first, so an uncontended acquisition costs no clock read.
        if let Ok(buffer) = self.receiver.try_recv() {
            return buffer;
        }
        let started = Instant::now();
        loop {
            match self.receiver.recv_timeout(backoff) {
                Ok(buffer) => {
                    let waited = started.elapsed();
                    self.charge_wait(&buffer, waited);
                    return buffer;
                }
                Err(crossbeam_channel::RecvTimeoutError::Timeout) => {
                    // on cancel, hand back a fresh buffer so teardown doesn't hang
                    if self.cancelled.load(Ordering::SeqCst) {
                        let (waited, buffer) = (started.elapsed(), vec![F::ZERO; self.buffer_size]);
                        self.charge_wait(&buffer, waited);
                        return buffer;
                    }
                    if let Some(short_by) = self.short_by() {
                        let (waited, buffer) = (started.elapsed(), vec![F::ZERO; self.buffer_size]);
                        self.charge_wait(&buffer, waited);
                        tracing::error!("Pool '{}': short by {short_by}; using an unpooled buffer", self.name);
                        return buffer;
                    }
                    backoff = (backoff * 2).min(MAX_POOL_WAIT_BACKOFF);
                }
                Err(crossbeam_channel::RecvTimeoutError::Disconnected) => {
                    panic!("Pool channel closed");
                }
            }
        }
    }

    /// Wait for a buffer, giving up after `timeout` so the caller can re-check its other wakeup
    /// sources. `None` on timeout or a closed channel (which `try_take` also tolerates).
    fn take_timeout(&self, timeout: Duration) -> Option<Vec<F>> {
        self.receiver.recv_timeout(timeout).ok()
    }

    /// Non-blocking channel poll (a pooled buffer or `None`). For callers that interleave the channel
    /// with another wakeup source in one loop and so can't use the blocking `take()`.
    fn try_take(&self) -> Option<Vec<F>> {
        self.receiver.try_recv().ok()
    }

    /// A fresh, unpooled buffer of the pool's size. Used on the abort path to
    /// unblock a waiter without drawing from the (possibly empty) channel.
    fn fresh_buffer(&self) -> Vec<F> {
        vec![F::ZERO; self.buffer_size]
    }

    fn is_cancelled(&self) -> bool {
        self.cancelled.load(Ordering::SeqCst)
    }

    /// How many buffers the last `reset` found missing, if any. A waiter must not park on those.
    fn short_by(&self) -> Option<usize> {
        match self.deficit.load(Ordering::SeqCst) {
            0 => None,
            n => Some(n),
        }
    }

    fn release(&self, buffer: Vec<F>) -> ProofmanResult<()> {
        // Every take charges this buffer's wait; whoever wanted it has read it by now.
        forget_buffer_wait(&buffer);
        let pooled = self.original_ptrs.contains(&(buffer.as_ptr() as usize));
        // An original must come back exactly as it left; a foreign one (cancel-escape, or the
        // larger trace a late-registered setup had to allocate) only has to reach the pool's size.
        if buffer.len() < self.buffer_size || (pooled && buffer.len() != self.buffer_size) {
            return Err(ProofmanError::ProofmanError(format!(
                "Pool::release: wrong size {} (expected {})",
                buffer.len(),
                self.buffer_size
            )));
        }
        // Only originals go back: pooling a foreign buffer would put an unpinned buffer in a pinned
        // pool, and would free a page `registered_buffers` still points at. This send can't block.
        if pooled {
            self.sender.send(buffer).expect("Pool channel closed");
        }
        Ok(())
    }

    /// Recover-only reset: every buffer must come back; we must NOT reallocate (a fresh buffer would
    /// be unpinned and leave a stale registration that Drop later frees against freed memory). A short
    /// count is reported as an error.
    ///
    /// NON-DESTRUCTIVE: whatever was drained goes back into the channel on every path, including the
    /// error one. Dropping the drained buffers instead would (a) turn a pool that is short by one into
    /// an empty pool, and (b) free pages that `registered_buffers` still points at, so `Pool::drop`
    /// would later unregister freed memory. Nothing here waits: the caller must have joined every
    /// worker that took a buffer, or this races them and reports a spurious leak.
    fn reset(&self) -> ProofmanResult<()> {
        // On abort, take() hands out fresh buffers, so a short pool is expected rather than a bug in
        // the release discipline: warn instead of erroring, so a real cancellation error is not
        // masked by a spurious invariant violation (mirrors MemoryHandler::reset).
        let cancelled = self.cancelled.load(Ordering::SeqCst);

        // Only originals are ever pooled (see `release`), so a wrong-size buffer here is impossible;
        // if one ever appears it is not a registered page, so dropping just that one is safe.
        let mut valid_buffers: Vec<Vec<F>> = Vec::with_capacity(self.n_buffers);
        let mut wrong_size = 0usize;
        while let Ok(buf) = self.receiver.try_recv() {
            if buf.len() != self.buffer_size {
                wrong_size += 1;
                continue;
            }
            valid_buffers.push(buf);
        }
        let recovered = valid_buffers.len();

        // Put everything back BEFORE deciding the outcome, so no exit path loses a buffer.
        for buf in valid_buffers {
            self.sender.send(buf).expect("Pool channel closed");
        }

        // A missing buffer is gone for good, so record it: `take` must stop parking on it (with
        // n_buffers == 1 it would park forever — nothing is left to release).
        self.deficit.store(self.n_buffers.saturating_sub(recovered), Ordering::SeqCst);

        if recovered == self.n_buffers && wrong_size == 0 {
            return Ok(());
        }

        let mut what = format!("recovered {} of {} buffers", recovered, self.n_buffers);
        if wrong_size > 0 {
            what.push_str(&format!(
                "; dropped {wrong_size} buffer(s) of unexpected size (expected {})",
                self.buffer_size
            ));
        }
        if cancelled {
            // Expected after an abort: take() hands out fresh buffers, so the release discipline can
            // legitimately be short here. Warn so a real leak is still traceable.
            tracing::warn!("Pool::reset (cancelled): {what}");
            return Ok(());
        }
        Err(ProofmanError::ProofmanError(format!("Pool::reset: {what}; a buffer was not released")))
    }

    fn total_bytes(&self) -> usize {
        // saturating_mul to match the rest of the file; can't overflow on 64-bit
        // with realistic sizes, but keeps the arithmetic uniform and panic-free.
        self.n_buffers.saturating_mul(self.buffer_size).saturating_mul(std::mem::size_of::<F>())
    }
}

impl<F: PrimeField64 + Send + Sync + 'static> Drop for Pool<F> {
    fn drop(&mut self) {
        // Runs once the last Arc is gone (all workers released their clones; on abort `cancel()`
        // unblocks their pooled take() so they exit and are joined — see `Drop for MemoryHandler`).
        // Runs before fields drop, so the pooled Vecs are still alive while we unregister their pages.
        for ptr in &self.registered_buffers {
            unregister_host_memory_c(*ptr as *mut c_void);
        }
    }
}

/// An instance whose contribution is already committed, so the pool may take its trace buffer back.
/// Reclaiming one is never free: every trace dropped here is recomputed from scratch in the proofs
/// phase, so `evict_key` is what that recomputation would cost.
struct PendingRelease {
    instance_id: usize,
    remove_from_calculated: bool,
    /// Measured witness time; zero when this instance was never measured.
    measured: Duration,
    /// Trace length in field elements. Only ranks the unmeasured against each other.
    trace_len: usize,
}

impl PendingRelease {
    /// Cheapest first, and anything unmeasured after everything measured: with no price on it,
    /// assume it is expensive rather than evict it on a zero.
    fn evict_key(&self) -> (bool, Duration, usize) {
        (self.measured.is_zero(), self.measured, self.trace_len)
    }
}

/// Instances whose trace buffer the pool is allowed to reclaim, and the order it reclaims them in.
///
/// The order is the whole point. Reclaiming is a choice between recomputations, and taking the
/// oldest (what a plain FIFO does) systematically picks the Main segments computed early — the work
/// the incremental Main advancement exists to save. Taking the most recent is worse still: the last
/// instances to commit are the expensive ones. So take the cheapest to recompute.
struct ReleaseQueue {
    /// Short by construction — one entry per committed instance still holding a pooled buffer — so a
    /// linear scan for the minimum beats maintaining a heap.
    pending: Mutex<Vec<PendingRelease>>,
    /// PROOFMAN_EVICT_FIFO=1 restores the previous oldest-first order, for A/B against this one.
    fifo: bool,
    evicted: AtomicUsize,
    evicted_cost_ns: AtomicU64,
}

impl ReleaseQueue {
    fn new() -> Self {
        Self::with_order(std::env::var("PROOFMAN_EVICT_FIFO").map(|v| v == "1" || v == "true").unwrap_or(false))
    }

    fn with_order(fifo: bool) -> Self {
        Self { pending: Mutex::new(Vec::new()), fifo, evicted: AtomicUsize::new(0), evicted_cost_ns: AtomicU64::new(0) }
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, Vec<PendingRelease>> {
        self.pending.lock().unwrap_or_else(|e| e.into_inner())
    }

    fn push(&self, entry: PendingRelease) {
        self.lock().push(entry);
    }

    /// The next instance to give up its buffer, or `None` if nothing is reclaimable.
    fn pop_victim(&self) -> Option<PendingRelease> {
        let mut pending = self.lock();
        if pending.is_empty() {
            return None;
        }
        let victim = if self.fifo {
            pending.remove(0)
        } else {
            let (idx, _) = pending.iter().enumerate().min_by_key(|(_, e)| e.evict_key())?;
            // Order carries no meaning once the minimum decides, so swap_remove is free.
            pending.swap_remove(idx)
        };
        self.evicted.fetch_add(1, Ordering::Relaxed);
        self.evicted_cost_ns.fetch_add(victim.measured.as_nanos() as u64, Ordering::Relaxed);
        Some(victim)
    }

    /// Drop what is still reclaimable without touching the counters: this runs between the two phases
    /// of the same proof, and the eviction tally is reported once, at the end of it.
    fn clear(&self) {
        self.lock().clear();
    }

    /// Start of a new proof: the tally is per proof.
    fn reset(&self) {
        self.clear();
        self.evicted.store(0, Ordering::Relaxed);
        self.evicted_cost_ns.store(0, Ordering::Relaxed);
    }

    /// What this proof paid to be short of buffers. Info, not debug: it is the figure to compare
    /// between runs, and it only prints when the pool actually ran out.
    fn log_summary(&self) {
        let evicted = self.evicted.load(Ordering::Relaxed);
        if evicted == 0 {
            return;
        }
        tracing::info!(
            "Trace pool: evicted {} committed trace(s), {}: {:.3}s of witness to recompute",
            evicted,
            if self.fifo { "oldest first" } else { "cheapest first" },
            Duration::from_nanos(self.evicted_cost_ns.load(Ordering::Relaxed)).as_secs_f64(),
        );
    }
}

pub struct MemoryHandler<F: PrimeField64 + Send + Sync + 'static> {
    pctx: Arc<ProofCtx<F>>,
    /// Committed instances whose buffer can be reclaimed, cheapest-to-recompute first.
    release_queue: ReleaseQueue,
    /// Channel + pinning + reset/Drop mechanics for the basic-trace buffers. The instance-release
    /// side-channel below and the `pctx` coupling are the only behavior layered on the shared pool.
    pool: Pool<F>,
    /// Set by `cancel()` so the `take_buffer` loop can exit instead of spinning on a buffer that
    /// will never be released. Shared with `pool` so one flag drives both the drain and channel poll.
    cancelled: Arc<AtomicBool>,
}

impl<F: PrimeField64 + Send + Sync + 'static> MemoryHandler<F> {
    pub fn new(pctx: Arc<ProofCtx<F>>, n_buffers: usize, buffer_size: usize) -> Self {
        let cancelled = Arc::new(AtomicBool::new(false));

        // Page-lock the basic-trace pool for direct H2D (trace is an H2D source; pairs with the
        // direct-copy fast path in goldilocks_tooling.cu). Relies on buffers never permanently
        // escaping the pool, which `reset` enforces.
        let pool = Pool::new("basic-trace", n_buffers, buffer_size, true, cancelled.clone());

        let total_memory = n_buffers * buffer_size * std::mem::size_of::<F>();
        tracing::info!("MemoryHandler::Total memory for basic traces: {}", crate::format_bytes(total_memory as f64));

        Self { pctx, release_queue: ReleaseQueue::new(), pool, cancelled }
    }

    /// Unblock any thread parked in `take_buffer`. Called on the abort path so a failed proof tears
    /// down cleanly instead of hanging on a buffer that will never be released.
    pub fn cancel(&self) {
        self.cancelled.store(true, Ordering::SeqCst);
    }

    /// Recover-only reset; see `Pool::reset` for the no-reallocate rationale. Sequencing requirement:
    /// all worker threads that took buffers must already be joined (so every buffer is back), or the
    /// `try_recv` drain below races a live worker and trips the `recovered N of M` error.
    pub fn reset(&self) -> ProofmanResult<()> {
        self.release_queue.reset();

        // Buffer recovery + integrity checks live in the shared pool. Run it while `cancelled` is
        // still visible there: on the abort path it warns about a short pool rather than erroring (a
        // spurious invariant violation would mask the real cancellation error), and either way it
        // restores everything it drained. Previously the abort path returned early and skipped the
        // pool entirely, so a leak left no trace at all.
        let result = self.pool.reset();

        // Clear the otherwise-sticky flag only AFTER the pool has read it (safe — workers are joined
        // by now). Left set, it turns the next run's take_buffer into an unbounded fresh allocator
        // (OOM, and unpinned buffers in a pinned pool).
        self.cancelled.store(false, Ordering::SeqCst);
        result
    }

    /// Take a basic-trace buffer. Waits on the pool channel with a backoff timeout that paces the
    /// two non-channel wakeup sources: the abort flag, and the soft-release queue that
    /// `to_be_released_buffer` fills without sending to the channel (so a bare parked `recv`
    /// would miss those). Was a 10µs sleep-poll — ~100k wakeups/s per waiter, inside
    /// CALCULATING_WITNESS, each iteration touching state shared with the releasing threads.
    pub fn take_buffer(&self) -> Vec<F> {
        let mut backoff = std::time::Duration::from_micros(50);
        let mut started: Option<Instant> = None;
        loop {
            if let Some(buffer) = self.pool.try_take() {
                if let Some(t) = started {
                    let waited = t.elapsed();
                    self.pool.charge_wait(&buffer, waited);
                }
                return buffer;
            }
            // Only now is this a wait; the first poll above is the uncontended path.
            let started = *started.get_or_insert_with(Instant::now);
            // Abort path: the awaited buffer may never be released (proof errored first), so return
            // a fresh buffer to unblock the worker and let the process tear down instead of spinning.
            if self.pool.is_cancelled() {
                let (waited, buffer) = (started.elapsed(), self.pool.fresh_buffer());
                self.pool.charge_wait(&buffer, waited);
                return buffer;
            }
            if let Some(victim) = self.release_queue.pop_victim() {
                let iid = victim.instance_id;
                if victim.remove_from_calculated {
                    self.pctx.dctx_reset_instance_calculated(iid);
                }
                let (is_shared, buf) = self.pctx.free_instance_traces(iid);
                if is_shared {
                    let waited = started.elapsed();
                    self.pool.charge_wait(&buf, waited);
                    return buf;
                }
                continue;
            }
            if let Some(buffer) = self.pool.take_timeout(backoff) {
                let waited = started.elapsed();
                self.pool.charge_wait(&buffer, waited);
                return buffer;
            }
            if let Some(short_by) = self.pool.short_by() {
                let buffer = self.pool.fresh_buffer();
                self.pool.charge_wait(&buffer, started.elapsed());
                tracing::error!("MemoryHandler::take_buffer: pool short by {short_by}; using an unpooled buffer");
                return buffer;
            }
            backoff = (backoff * 2).min(MAX_POOL_WAIT_BACKOFF);
        }
    }

    /// Log how long callers spent blocked on this pool. Their own timers include that wait, so
    /// without this line a queueing delay reads as witness compute.
    pub fn log_wait_summary(&self) {
        self.pool.log_wait();
        self.release_queue.log_summary();
    }

    pub fn release_buffer(&self, buffer: Vec<F>) -> ProofmanResult<()> {
        self.pool.release(buffer)
    }

    /// Offer this instance's trace buffer back to the pool. Nothing is freed here: the instance keeps
    /// its trace, and so its place in the proofs phase, until a thread that needs a buffer picks it as
    /// the cheapest one to recompute. Prices it now, while the trace is still there to measure.
    pub fn to_be_released_buffer(&self, instance_id: usize, remove_from_calculated: bool) {
        self.release_queue.push(PendingRelease {
            instance_id,
            remove_from_calculated,
            measured: self.pctx.witness_cost(instance_id).unwrap_or_default(),
            trace_len: self.pctx.air_instance_trace_len(instance_id),
        });
    }

    pub fn empty_queue_to_be_released(&self) {
        self.release_queue.clear();
    }
}

// No explicit Drop: `Pool::drop` unregisters the pinned pages once the last Arc<MemoryHandler> is
// gone. That hinges on workers being joined first (each holds an Arc clone; a parked take_buffer
// never releases it — `cancel()` unblocks it). Under panic=abort no Drop runs; the OS reclaims.

pub trait BufferPool<F: PrimeField64>: Send + Sync
where
    F: Send + Sync + 'static,
{
    fn take_buffer(&self) -> Vec<F>;
}

impl<F: PrimeField64 + Send + Sync + 'static> BufferPool<F> for MemoryHandler<F> {
    fn take_buffer(&self) -> Vec<F> {
        self.take_buffer()
    }
}

/// Buffers for in-flight recursive proofs: one pool per kind of buffer, not per kind of proof.
///
/// A compressor and a recursive proof draw from the same two, each sized at the larger of what the
/// two need. That is only cheap while the two sizes are close: every buffer pays the larger one, so
/// at `n` buffers the merge costs `n` times the difference, not one buffer's worth.
pub struct MemoryHandlerRecursive<F: PrimeField64 + Send + Sync + 'static> {
    trace: Pool<F>,
    signal_values: Option<SignalValuesPool>,
    witness_threads: usize,
    cancelled: Arc<AtomicBool>,
}

impl<F: PrimeField64 + Send + Sync + 'static> MemoryHandlerRecursive<F> {
    /// `buffer_size_trace` is the largest any proof kind needs, and `n_buffers` covers every
    /// recursive kind including the compressor: they share one pool, so a compressor and a
    /// recursive proof in flight together just take two of the same buffers.
    pub fn new(n_buffers: usize, buffer_size_trace: usize) -> Self {
        Self::new_with_signal_pool(n_buffers, buffer_size_trace, None, 8)
    }

    pub fn new_with_signal_pool(
        n_buffers: usize,
        buffer_size_trace: usize,
        signal_pool: Option<(usize, usize)>,
        witness_threads: usize,
    ) -> Self {
        let cancelled = Arc::new(AtomicBool::new(false));
        // One pool for every recursive kind, the compressor included. Traces are H2D sources so it
        // is pinned; signalValues is CPU-only scratch.
        let n = n_buffers.max(1);
        let trace = Pool::new("recursive-trace", n, buffer_size_trace, true, cancelled.clone());

        tracing::info!(
            "MemoryHandlerRecursive::Total memory for recursive traces: {} = {n} x trace {}",
            crate::format_bytes(trace.total_bytes() as f64),
            crate::format_bytes((buffer_size_trace * std::mem::size_of::<F>()) as f64),
        );

        let signal_values = signal_pool.map(|(cap, n)| SignalValuesPool::new(cap, n, cancelled.clone()));

        let witness_threads = witness_threads.max(1);
        tracing::info!("MemoryHandlerRecursive::circom solve threads per recursive witness: {}", witness_threads);

        Self { trace, signal_values, witness_threads, cancelled }
    }

    /// Unblock any thread parked in a pooled `take()`. Called on the abort path so a failed proof
    /// tears down cleanly instead of hanging on a buffer that will never be released.
    pub fn cancel(&self) {
        self.cancelled.store(true, Ordering::SeqCst);
    }

    pub fn witness_threads(&self) -> usize {
        self.witness_threads
    }

    pub fn take_buffer_signal_values(&self, needed: usize) -> Vec<u64> {
        match &self.signal_values {
            Some(pool) => pool.take(needed),
            None => Vec::new(),
        }
    }
    pub fn release_buffer_signal_values(&self, buffer: Vec<u64>) {
        if let Some(pool) = &self.signal_values {
            if !buffer.is_empty() {
                pool.release(buffer);
            }
        }
    }

    /// Reset both trace pools and report the first failure — never `?` out early. A short trace pool
    /// must not stop the compressor pool from being recovered, and above all must not skip clearing
    /// `cancelled`: left set, it turns the next run's `take()` into an unbounded fresh allocator
    /// handing out unpinned buffers. Each pool's own reset is non-destructive, so continuing past a
    /// failure cannot lose anything.
    pub fn reset(&self) -> ProofmanResult<()> {
        let results = [("trace", self.trace.reset())];
        // Re-arm AFTER both pool resets: the pools share this flag and each Pool::reset reads it
        // (cancelled-aware outcome), so clearing earlier would re-enable the hard checks mid-teardown.
        self.cancelled.store(false, Ordering::SeqCst);

        let mut first_err = None;
        for (name, result) in results {
            if let Err(e) = result {
                tracing::error!("MemoryHandlerRecursive::reset: {name} pool did not recover: {e}");
                first_err = first_err.or(Some(e));
            }
        }
        match first_err {
            Some(e) => Err(e),
            None => Ok(()),
        }
    }

    /// Same as [`MemoryHandler::log_wait_summary`], across the recursive pools.
    pub fn log_wait_summary(&self) {
        self.trace.log_wait();
        if let Some(pool) = &self.signal_values {
            pool.log_wait();
        }
    }

    pub fn take_buffer_trace(&self) -> Vec<F> {
        self.trace.take()
    }
    pub fn release_buffer_trace(&self, buffer: Vec<F>) -> ProofmanResult<()> {
        self.trace.release(buffer)
    }

    /// Take a recursive-proof trace buffer as a release-on-drop lease (see [`BufferLease`]).
    /// One pool serves every proof kind; see the field.
    pub fn take_trace_lease(&self) -> BufferLease<'_, F> {
        BufferLease { handler: self, buffer: Some(self.trace.take()), pool: RecursivePool::Trace }
    }

    /// Adopt an already-taken trace buffer (a `Proof`'s `trace`, taken from the pool when its witness
    /// was generated) into a release-on-drop lease so it returns to its pool on every exit path
    /// instead of leaking on cancel/error. `adopt`, not `take`: the buffer already left the pool.
    /// `compressor` selects the compressor trace pool.
    pub fn adopt_trace(&self, buffer: Vec<F>) -> BufferLease<'_, F> {
        BufferLease { handler: self, buffer: Some(buffer), pool: RecursivePool::Trace }
    }
}

/// Which of the two recursive trace pools a [`BufferLease`] returns its buffer to on drop.
#[derive(Clone, Copy)]
enum RecursivePool {
    Trace,
}

/// A recursive-proof buffer that returns itself to its pool when dropped (success, early `?`, or
/// panic) so it can't leak and shrink the pool. Obtain via `take_trace_lease` or `adopt_trace`;
/// derefs to `Vec<F>`. On GPU a trace is an async H2D source, so the caller must gate reuse on the
/// stream's commit event *before* the lease drops at scope exit.
pub struct BufferLease<'a, F: PrimeField64 + Send + Sync + 'static> {
    handler: &'a MemoryHandlerRecursive<F>,
    buffer: Option<Vec<F>>,
    pool: RecursivePool,
}

impl<F: PrimeField64 + Send + Sync + 'static> std::ops::Deref for BufferLease<'_, F> {
    type Target = Vec<F>;
    fn deref(&self) -> &Vec<F> {
        self.buffer.as_ref().expect("BufferLease used after release")
    }
}

impl<F: PrimeField64 + Send + Sync + 'static> std::ops::DerefMut for BufferLease<'_, F> {
    fn deref_mut(&mut self) -> &mut Vec<F> {
        self.buffer.as_mut().expect("BufferLease used after release")
    }
}

impl<F: PrimeField64 + Send + Sync + 'static> Drop for BufferLease<'_, F> {
    fn drop(&mut self) {
        if let Some(buffer) = self.buffer.take() {
            // Return the buffer to its pool. A destructor can't propagate a Result, but release can't
            // fail here (size matches, and the send never blocks — see Pool::release), so it's fine.
            let _ = match self.pool {
                RecursivePool::Trace => self.handler.release_buffer_trace(buffer),
            };
        }
    }
}

/// Pool of reusable `signalValues` (u64) buffers handed into getWitnessTrace so
/// Circom_CalcWit reuses them instead of allocating tens-to-hundreds of MB per
/// proof. Buffers are NOT zeroed on reuse: the circom solve is write-before-read
/// (only signalValues[0]=1, reset inside the ctor), validated end-to-end.
///
/// One size for every buffer, over a bounded channel (its depth caps concurrency;
/// `recv` blocks when empty). `cap` covers the largest circuit, so no proof allocates its own; the
/// oversize path below exists only for a circuit registered after the pool was sized.
///
/// Every blocking wait is cancel-aware, like the trace `Pool`.
pub struct SignalValuesPool {
    tx: Sender<Vec<u64>>,
    rx: Receiver<Vec<u64>>,
    cap: usize,
    /// Shared with the owning `MemoryHandlerRecursive` and its trace pools.
    cancelled: Arc<AtomicBool>,
    /// It replaced the witness pools, which did block, so leaving it unmeasured moves the blind spot.
    wait_ns: AtomicU64,
    wait_count: AtomicUsize,
}

impl SignalValuesPool {
    /// `cap` is in u64 elements (= getTotalSignalNo()).
    pub fn new(cap: usize, n: usize, cancelled: Arc<AtomicBool>) -> Self {
        let n = n.max(1);
        let (tx, rx) = bounded(n);
        for _ in 0..n {
            tx.send(vec![0u64; cap]).unwrap();
        }
        tracing::info!(
            "SignalValuesPool: {n} x {} = {}",
            crate::format_bytes((cap * 8) as f64),
            crate::format_bytes((cap * n * 8) as f64),
        );
        Self { tx, rx, cap, cancelled, wait_ns: AtomicU64::new(0), wait_count: AtomicUsize::new(0) }
    }

    /// Blocking receive that yields to `cancelled`, mirroring `Pool::take`: on the abort
    /// path it hands back a fresh buffer instead of parking on one nobody will return.
    fn recv_cancellable(&self) -> Vec<u64> {
        if let Ok(buffer) = self.rx.try_recv() {
            return buffer;
        }
        let start = Instant::now();
        let buffer = loop {
            match self.rx.recv_timeout(Duration::from_micros(100)) {
                Ok(buffer) => break buffer,
                Err(crossbeam_channel::RecvTimeoutError::Timeout) => {
                    if self.cancelled.load(Ordering::SeqCst) {
                        break vec![0u64; self.cap];
                    }
                }
                Err(crossbeam_channel::RecvTimeoutError::Disconnected) => {
                    panic!("SignalValuesPool channel closed");
                }
            }
        };
        let waited = start.elapsed();
        self.wait_ns.fetch_add(waited.as_nanos() as u64, Ordering::Relaxed);
        self.wait_count.fetch_add(1, Ordering::Relaxed);
        // Also charged to the buffer, so the caller can subtract it from its witness span.
        charge_buffer_wait(buffer.as_ptr() as *const u8, waited);
        buffer
    }

    /// Blocked time here, reported like the trace pools'.
    pub fn log_wait(&self) {
        let blocked = self.wait_count.load(Ordering::Relaxed);
        if blocked > 0 {
            tracing::debug!(
                "Pool 'signal-values': {} acquisitions blocked, {:.3}s total",
                blocked,
                Duration::from_nanos(self.wait_ns.load(Ordering::Relaxed)).as_secs_f64()
            );
        }
    }

    /// Empty when `needed` exceeds `cap`, which the caller turns into a null pointer so the C++
    /// solve allocates its own. `cap` is the largest circuit of the key, so this only fires for a
    /// recurser registered after the pool was sized.
    pub fn take(&self, needed: usize) -> Vec<u64> {
        if needed > self.cap {
            tracing::debug!(
                "SignalValuesPool: request of {} elements exceeds the pooled buffer ({}); \
                 falling back to a self-allocated signalValues buffer",
                needed,
                self.cap
            );
            return Vec::new();
        }
        self.recv_cancellable()
    }

    /// Pool only full-size buffers. The caller releases unconditionally, so this also gets the
    /// empty Vec that stood in for a self-allocating circuit; neither it nor the fresh buffers
    /// `take` mints on the abort path came from a slot, so dropping them keeps the count right.
    pub fn release(&self, buffer: Vec<u64>) {
        if buffer.len() >= self.cap {
            let _ = self.tx.try_send(buffer);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proofman_fields::{Field, Goldilocks};

    /// `(instance_id, witness ms, trace elements)` — `ms == 0` means never measured.
    fn queue(fifo: bool, entries: &[(usize, u64, usize)]) -> ReleaseQueue {
        let q = ReleaseQueue::with_order(fifo);
        for &(instance_id, ms, trace_len) in entries {
            q.push(PendingRelease {
                instance_id,
                remove_from_calculated: false,
                measured: Duration::from_millis(ms),
                trace_len,
            });
        }
        q
    }

    fn drain(q: &ReleaseQueue) -> Vec<usize> {
        std::iter::from_fn(|| q.pop_victim()).map(|e| e.instance_id).collect()
    }

    #[test]
    fn evicts_the_cheapest_witness_first() {
        // The shape that matters: an early Main committed first, cheap instances after it. FIFO would
        // throw the Main away; every trace evicted here is recomputed in the proofs phase.
        let q = queue(false, &[(0, 201, 640_000_000), (1, 1, 33_000_000), (2, 431, 475_000_000), (3, 59, 41_000_000)]);
        assert_eq!(drain(&q), vec![1, 3, 0, 2]);
    }

    #[test]
    fn an_unmeasured_instance_is_evicted_after_every_measured_one() {
        // A zero is "no price", not "free": evicting on it would pick exactly the instance we know
        // least about. It still ranks ahead of nothing, so it goes last, by trace length.
        let q = queue(false, &[(0, 0, 475_000_000), (1, 0, 33_000_000), (2, 431, 475_000_000)]);
        assert_eq!(drain(&q), vec![2, 1, 0]);
    }

    #[test]
    fn fifo_order_is_restorable() {
        // PROOFMAN_EVICT_FIFO=1 must reproduce the previous behavior exactly, for A/B runs.
        let q = queue(true, &[(0, 201, 640_000_000), (1, 1, 33_000_000), (2, 431, 475_000_000)]);
        assert_eq!(drain(&q), vec![0, 1, 2]);
    }

    #[test]
    fn the_tally_spans_both_phases_of_a_proof() {
        let q = queue(false, &[(0, 200, 1), (1, 50, 1)]);
        q.pop_victim().unwrap();
        // Between contributions and proofs the queue is emptied; the tally must survive it, since it
        // is reported once at the end of the proof.
        q.clear();
        assert_eq!(q.evicted.load(Ordering::Relaxed), 1);
        assert_eq!(q.evicted_cost_ns.load(Ordering::Relaxed), Duration::from_millis(50).as_nanos() as u64);
        assert!(q.pop_victim().is_none(), "clear drops what is still reclaimable");

        q.reset();
        assert_eq!(q.evicted.load(Ordering::Relaxed), 0, "a new proof starts from zero");
    }

    // These exercise the pool's accounting (take/release/reset, leak detection, cancel), not pinning,
    // so they pass on both backends (CPU register is a no-op; GPU pinning is transparent to accounting).
    type F = Goldilocks;

    fn handler(n: usize, size: usize) -> MemoryHandlerRecursive<F> {
        MemoryHandlerRecursive::new(n, size)
    }

    #[test]
    fn clean_round_trip_then_reset_succeeds() {
        let h = handler(3, 8);
        // One pool holding n + n_comp: take all three out, then release them all back.
        let t0 = h.take_buffer_trace();
        let t1 = h.take_buffer_trace();
        let t2 = h.take_buffer_trace();
        h.release_buffer_trace(t0).unwrap();
        h.release_buffer_trace(t1).unwrap();
        h.release_buffer_trace(t2).unwrap();
        // This is the gap-2 invariant: a clean round trip leaves full pools, so the
        // reset wired into ProofMan::reset() passes.
        h.reset().unwrap();
        // And it is idempotent across reuse.
        h.reset().unwrap();
    }

    #[test]
    fn reset_detects_a_leaked_buffer() {
        let h = handler(2, 8);
        // Simulate a leak: a worker took a buffer and never released it (e.g. an early `?` on a
        // non-cancelled path). reset() must surface the short pool rather than paper over it.
        let _leaked = h.take_buffer_trace();
        assert!(h.reset().is_err());
    }

    #[test]
    fn a_short_pool_hands_out_unpooled_buffers_instead_of_parking() {
        let h = handler(1, 8);
        std::mem::forget(h.take_buffer_trace()); // stranded: never released, address never reused
        assert!(h.reset().is_err(), "the loss is reported");
        // Nothing is left to release, so without the recorded deficit this would park forever.
        let fresh = h.take_buffer_trace();
        assert_eq!(fresh.len(), 8);
        h.release_buffer_trace(fresh).unwrap(); // unpooled: dropped, never pooled
                                                // A clean reset (the escapee's slot still missing) keeps reporting it.
        assert!(h.reset().is_err());
    }

    #[test]
    fn release_rejects_wrong_size_buffer() {
        let h = handler(1, 8);
        let _good = h.take_buffer_trace();
        // Release a buffer of the wrong length; the size check rejects it.
        assert!(h.release_buffer_trace(vec![F::ZERO; 7]).is_err());
    }

    /// What a setup registered after the pool was sized (a recurser) hands back.
    #[test]
    fn release_accepts_an_oversized_unpooled_buffer() {
        let h = handler(1, 8);
        let pooled = h.take_buffer_trace();
        h.release_buffer_trace(vec![F::ZERO; 32]).expect("an oversized unpooled trace releases cleanly");
        h.release_buffer_trace(pooled).unwrap();
        h.reset().expect("pool intact"); // the oversized one was dropped, not pooled

        assert_eq!(h.take_buffer_trace().len(), 8);
    }

    #[test]
    fn a_failed_reset_keeps_the_buffers_it_recovered() {
        // reset() reports a short pool, but must not DESTROY what came back: dropping the drained
        // buffers would turn "short by one" into "empty", and would free pages that
        // `registered_buffers` still points at, so Pool::drop would unregister freed memory.
        let h = handler(3, 8);
        let leaked = h.take_buffer_trace(); // never released
        assert!(h.reset().is_err(), "a missing buffer must still be reported");

        // The other two are still pooled and usable, and a third take does not block.
        let a = h.take_buffer_trace();
        let b = h.take_buffer_trace();
        assert_eq!((a.len(), b.len()), (8, 8));
        h.release_buffer_trace(a).unwrap();
        h.release_buffer_trace(b).unwrap();
        // Returning the escapee makes the pool whole again — impossible if reset() had dropped the rest.
        h.release_buffer_trace(leaked).unwrap();
        h.reset().expect("pool is whole once the escapee comes back");
    }

    #[test]
    fn the_pool_holds_both_kinds_counts_summed() {
        // Counts are the SUM, not the max: a compressor and a recursive proof can be in flight
        // together, and a pool one buffer short parks whichever asks second until the other finishes.
        let h = handler(2, 8);
        let a = h.take_buffer_trace();
        let b = h.take_buffer_trace();
        assert_eq!((a.len(), b.len()), (8, 8));
        h.release_buffer_trace(a).unwrap();
        h.release_buffer_trace(b).unwrap();
        h.reset().expect("pool whole");
    }

    #[test]
    fn a_failed_reset_still_clears_the_cancelled_flag() {
        // Left set, `cancelled` makes take() an unbounded fresh allocator handing out unpinned
        // buffers. That must not survive a reset that reported an error.
        let h = handler(2, 8);
        h.cancel();
        let escapee = h.take_buffer_trace(); // the pool's only buffer
        let _ = h.reset(); // cancelled path: warns rather than errors
        h.release_buffer_trace(escapee).unwrap();
        // Flag cleared, so the pool is authoritative again: it holds its one original buffer.
        let original = h.take_buffer_trace();
        h.release_buffer_trace(original).unwrap();
        h.reset().expect("pool whole and no longer in cancelled mode");
    }

    #[test]
    fn releasing_a_cancel_escape_buffer_does_not_pool_it() {
        // `release` pools only the pool's own (registered) buffers. A fresh buffer handed out by a
        // cancelled `take()` is unregistered, so pooling it would put an unpinned buffer in a pinned
        // pool — and would also push the channel past capacity. It must be dropped instead, while
        // the originals still come back.
        let h = handler(1, 8);
        let original = h.take_buffer_trace();
        h.cancel();
        let fresh = h.take_buffer_trace();
        // Release both, fresh first, so a mistakenly-pooled fresh buffer would occupy the one slot.
        h.release_buffer_trace(fresh).unwrap();
        h.release_buffer_trace(original).unwrap();
        h.reset().unwrap();
        // The pool is whole again and hands out its own buffer, not the escapee.
        assert_eq!(h.take_buffer_trace().len(), 8);
    }

    #[test]
    fn adopt_trace_returns_the_buffer_on_drop() {
        // Teardown recovery relies on this: an adopted buffer must come back when the lease drops,
        // or the pool silently shrinks on every cancelled proof.
        let h = handler(2, 8);
        let a = h.take_buffer_trace();
        let b = h.take_buffer_trace();
        drop(h.adopt_trace(a));
        drop(h.adopt_trace(b));
        h.reset().expect("pool whole after adopt-then-drop");
    }

    #[test]
    fn cancel_unblocks_take_and_skips_reset_checks() {
        let h = handler(1, 8);
        // Empty the pool, then cancel. A subsequent take must return a fresh buffer
        // instead of blocking forever, and reset must not flag the (now short) pool.
        let _taken = h.take_buffer_trace();
        h.cancel();
        let fresh = h.take_buffer_trace(); // would hang pre-cancel on an empty pool
        assert_eq!(fresh.len(), 8);
        h.reset().unwrap(); // cancelled path skips integrity checks
    }

    // ---- signalValues pool ----

    fn signal_handler(cap: usize, n: usize) -> MemoryHandlerRecursive<F> {
        MemoryHandlerRecursive::new_with_signal_pool(1, 8, Some((cap, n)), 4)
    }

    #[test]
    fn signal_pool_hands_out_one_size() {
        let h = signal_handler(64, 2);
        // Every buffer is `cap`, however little the caller asked for.
        let light = h.take_buffer_signal_values(16);
        assert_eq!(light.len(), 64);
        let heavy = h.take_buffer_signal_values(64);
        assert_eq!(heavy.len(), 64);
        h.release_buffer_signal_values(light);
        h.release_buffer_signal_values(heavy);
        // Round-trip is repeatable: both went back into the one channel.
        assert_eq!(h.take_buffer_signal_values(64).len(), 64);
    }

    #[test]
    fn signal_pool_request_larger_than_pool_yields_empty_buffer() {
        let h = signal_handler(64, 2);
        // The circuit the sizing left out (or a late-registered recurser): must not be handed
        // the too-short pooled buffer. Empty => null => C++ self-allocates.
        assert!(h.take_buffer_signal_values(65).is_empty());
        assert_eq!(h.take_buffer_signal_values(64).len(), 64);
    }

    #[test]
    fn signal_pool_absent_yields_empty_buffer() {
        // No pool configured => empty Vec (the null sentinel); releasing it is a no-op.
        let h = handler(1, 8);
        let buf = h.take_buffer_signal_values(1024);
        assert!(buf.is_empty());
        h.release_buffer_signal_values(buf);
    }

    #[test]
    fn signal_pool_never_pools_the_self_allocating_stand_in() {
        // The caller releases unconditionally, so the empty Vec that stood in for an oversized
        // circuit comes back here. Pooling it would hand the next solve a zero-length buffer.
        let h = signal_handler(64, 1);
        let stand_in = h.take_buffer_signal_values(65);
        assert!(stand_in.is_empty());
        h.release_buffer_signal_values(stand_in);
        assert_eq!(h.take_buffer_signal_values(64).len(), 64, "the empty stand-in reached the solve");
    }

    #[test]
    fn signal_pool_take_never_returns_a_short_buffer_after_an_abort() {
        // With the pool drained, `cancel` makes take() mint a fresh buffer, so live buffers
        // outnumber the channel and one release is dropped. A later request must still get a
        // buffer it can safely write `needed` elements into.
        let h = signal_handler(64, 1);
        let held = h.take_buffer_signal_values(64);
        h.cancel();
        let fresh = h.take_buffer_signal_values(16);
        h.release_buffer_signal_values(held);
        h.release_buffer_signal_values(fresh);
        h.reset().unwrap(); // re-arms `cancelled`, as the distributed worker does
        assert!(h.take_buffer_signal_values(64).len() >= 64, "handed the solve a short buffer");
    }

    #[test]
    fn signal_pool_cancel_unblocks_take() {
        let h = signal_handler(64, 1);
        // Drain the pool, then cancel: a further take must return a fresh full-size buffer
        // rather than parking forever.
        let _held = h.take_buffer_signal_values(64);
        h.cancel();
        assert_eq!(h.take_buffer_signal_values(16).len(), 64);
    }

    /// A blocked acquisition must be attributed to the pool, not left inside whichever timer wraps
    /// the caller -- that is what made a 6.4s buffer wait read as 6.4s of witness compute.
    #[test]
    fn a_blocked_acquisition_is_charged_to_the_pool() {
        use proofman_fields::Goldilocks;
        let cancelled = Arc::new(AtomicBool::new(false));
        let pool: Arc<Pool<Goldilocks>> = Arc::new(Pool::new("test", 1, 8, false, cancelled));

        // The only buffer, so the next take must block.
        let held = pool.take();
        assert_eq!(pool.wait_stats(), (Duration::ZERO, 0), "an uncontended take must not be charged");

        let returner = {
            let pool = pool.clone();
            std::thread::spawn(move || {
                std::thread::sleep(Duration::from_millis(150));
                pool.release(held).unwrap();
            })
        };
        let _second = pool.take();
        returner.join().unwrap();

        let (waited, blocked) = pool.wait_stats();
        assert_eq!(blocked, 1, "exactly one acquisition blocked");
        assert!(waited >= Duration::from_millis(100), "wait of {waited:?} should reflect the 150ms hold");
        assert!(waited < Duration::from_secs(5), "wait of {waited:?} is implausible");
    }
}
