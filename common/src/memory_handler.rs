use crossbeam_channel::{bounded, Sender, Receiver};
use std::collections::HashSet;
use std::ffi::c_void;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use crossbeam_queue::SegQueue;
use std::time::Duration;
use crate::ProofCtx;
use fields::PrimeField64;
use crate::{ProofmanError, ProofmanResult};
use proofman_starks_lib_c::{register_host_memory_c, unregister_host_memory_c};

/// Round a host range out to page boundaries — `cudaHostRegister` requires the
/// region to cover whole pages.
fn aligned_host_range(ptr: usize, bytes: usize) -> Option<(usize, usize)> {
    if ptr == 0 || bytes == 0 {
        return None;
    }
    // `libc` is only a dependency on Linux (the only target with the GPU backend
    // whose `cudaHostRegister` this range feeds). Elsewhere host pinning is a no-op,
    // so a plain default page size is fine.
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

/// Register a freshly-allocated pool, enforcing all-or-nothing pinning: either
/// every buffer's pages are pinned (GPU backend) or none are (CPU backend, where
/// pinning is a no-op). A partial result means a `cudaHostRegister` failed under
/// the GPU backend, leaving an unpinned buffer in a pool the GPU treats as pinned
/// — we refuse to start rather than run with that soundness hole. Returns the
/// distinct base pages pinned, so the caller can unregister each exactly once on
/// Drop. Registration pins a *specific address*, so a registered buffer must never
/// be reallocated for the life of the pool — see the recover-only `reset`.
///
/// Defensive dedup on the base page: if two buffers rounded down to the same base,
/// register it only once, because `cudaHostRegister` rejects an already-registered
/// page and the duplicate "failure" would trip the all-or-nothing panic spuriously.
///
/// Two *multi-page* allocations cannot share a base page (the second would have to
/// start inside the first, i.e. overlap), and every pool buffer here spans many
/// pages (circom witness/trace sizes), so this branch never fires for the real
/// workload. It only guards a hypothetical sub-page pool — for which base-keyed
/// dedup could under-cover a buffer that spills past the first's range, and proper
/// range-based coverage would be needed. We keep the cheap guard and the assumption.
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
    // Partial pinning is fatal (all-or-nothing), but before we abort, unregister
    // the pages we did pin — otherwise those stay locked for the rest of the
    // process (and, under panic=abort, Drop never runs to release them).
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

/// Single fixed-size buffer pool over a bounded channel. Internal helper used
/// by `MemoryHandlerRecursive`. `take()` polls the channel with a short
/// `recv_timeout` so the abort path (`cancelled`) can wake it; a released buffer
/// is picked up on the next poll.
struct Pool<F: PrimeField64 + Send + Sync + 'static> {
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
}

impl<F: PrimeField64 + Send + Sync + 'static> Pool<F> {
    fn new(n_buffers: usize, buffer_size: usize, pin: bool, cancelled: Arc<AtomicBool>) -> Self {
        let (sender, receiver) = bounded(n_buffers);
        let buffers: Vec<Vec<F>> = (0..n_buffers).map(|_| vec![F::ZERO; buffer_size]).collect();
        let registered_buffers: Vec<usize> = if pin { register_pool(&buffers) } else { Vec::new() };
        for buffer in buffers {
            sender.send(buffer).unwrap();
        }
        Self { sender, receiver, n_buffers, buffer_size, registered_buffers, cancelled }
    }

    fn take(&self) -> Vec<F> {
        // Poll with a timeout instead of a bare blocking recv so the abort path can
        // wake us; on cancel, hand back a fresh buffer so teardown doesn't hang.
        loop {
            match self.receiver.recv_timeout(Duration::from_micros(100)) {
                Ok(buffer) => return buffer,
                Err(crossbeam_channel::RecvTimeoutError::Timeout) => {
                    if self.cancelled.load(Ordering::SeqCst) {
                        return vec![F::ZERO; self.buffer_size];
                    }
                }
                Err(crossbeam_channel::RecvTimeoutError::Disconnected) => {
                    panic!("Pool channel closed");
                }
            }
        }
    }

    /// Non-blocking channel poll — returns a pooled buffer if one is immediately
    /// available, else `None`. Used by callers (e.g. `MemoryHandler`) that must
    /// interleave the channel with another wakeup source in a single loop and so
    /// can't use the blocking `take()`.
    fn try_take(&self) -> Option<Vec<F>> {
        self.receiver.try_recv().ok()
    }

    /// A fresh, unpooled buffer of the pool's size. Used on the abort path to
    /// unblock a waiter without drawing from the (possibly empty) channel.
    fn fresh_buffer(&self) -> Vec<F> {
        vec![F::ZERO; self.buffer_size]
    }

    /// Buffers currently sitting in the channel (not the pool capacity).
    fn available(&self) -> usize {
        self.receiver.len()
    }

    fn is_cancelled(&self) -> bool {
        self.cancelled.load(Ordering::SeqCst)
    }

    fn release(&self, buffer: Vec<F>) -> ProofmanResult<()> {
        if buffer.len() != self.buffer_size {
            return Err(ProofmanError::ProofmanError(format!(
                "Pool::release: wrong size {} (expected {})",
                buffer.len(),
                self.buffer_size
            )));
        }
        // On the abort path take() may hand out fresh buffers, so live buffers can
        // exceed channel capacity; a blocking send would park forever. Release
        // best-effort when cancelled. KNOWN HAZARD for the cancellation rework:
        // dropping a REGISTERED buffer leaves a live cudaHostRegister on recycled
        // pages — a later allocation there takes the un-gated direct-H2D fast path.
        if self.cancelled.load(Ordering::SeqCst) {
            let _ = self.sender.try_send(buffer);
            return Ok(());
        }
        self.sender.send(buffer).expect("Pool channel closed");
        Ok(())
    }

    /// Recover-only reset: every buffer must come back into the pool. We must NOT
    /// reallocate a replacement — a fresh allocation would be unpinned and leave a
    /// stale registration that Drop later unregisters against freed memory. A short
    /// count means a buffer escaped, so we surface it rather than paper over it.
    ///
    /// On the error path this is destructive: the buffers already drained into the
    /// local `valid_buffers` are dropped (not re-sent), so a failed reset empties the
    /// pool. That is fine — a failed reset means a buffer escaped, which only happens
    /// when the proof is already aborting and the pool is about to be torn down.
    ///
    /// Caller must have joined all workers that took buffers first (the drain below
    /// races a live worker and trips the `recovered N of M` error).
    fn reset(&self) -> ProofmanResult<()> {
        // On the abort path take() hands out a fresh (unregistered) buffer to
        // unblock workers, which may then be released back into the pool. Skip the
        // integrity checks when cancelled — they would mask the real cancellation
        // error with a spurious invariant violation (mirrors MemoryHandler::reset).
        if self.cancelled.load(Ordering::SeqCst) {
            return Ok(());
        }

        let mut valid_buffers: Vec<Vec<F>> = Vec::with_capacity(self.n_buffers);
        while let Ok(buf) = self.receiver.try_recv() {
            if buf.len() != self.buffer_size {
                return Err(ProofmanError::ProofmanError(format!(
                    "Pool::reset: buffer with unexpected size {} (expected {})",
                    buf.len(),
                    self.buffer_size
                )));
            }
            valid_buffers.push(buf);
        }
        if valid_buffers.len() != self.n_buffers {
            return Err(ProofmanError::ProofmanError(format!(
                "Pool::reset: recovered {} of {} buffers; a buffer was not released",
                valid_buffers.len(),
                self.n_buffers
            )));
        }
        for buf in valid_buffers {
            self.sender.send(buf).expect("Pool channel closed");
        }
        Ok(())
    }

    fn total_bytes(&self) -> usize {
        // saturating_mul to match the rest of the file; can't overflow on 64-bit
        // with realistic sizes, but keeps the arithmetic uniform and panic-free.
        self.n_buffers.saturating_mul(self.buffer_size).saturating_mul(std::mem::size_of::<F>())
    }
}

impl<F: PrimeField64 + Send + Sync + 'static> Drop for Pool<F> {
    fn drop(&mut self) {
        // Only runs once the last Arc holding this pool is gone, i.e. after all
        // worker threads have released their clones; on the abort path `cancel()`
        // unblocks the pooled take() so those threads can exit and be joined. See
        // the longer note on `Drop for MemoryHandler`.
        //
        // Runs before fields drop, so the pooled Vecs (held by the channel) are
        // alive while we unregister their pages.
        for ptr in &self.registered_buffers {
            unregister_host_memory_c(*ptr as *mut c_void);
        }
    }
}

pub struct MemoryHandler<F: PrimeField64 + Send + Sync + 'static> {
    pctx: Arc<ProofCtx<F>>,
    instance_ids_to_be_released: Arc<SegQueue<(usize, bool)>>,
    /// Channel + pinning + reset/Drop mechanics for the basic-trace buffers. The
    /// instance-release side-channel below and the `pctx` coupling are the only
    /// behavior layered on top of the shared pool.
    pool: Pool<F>,
    /// Set by `cancel()` on the abort path so the `take_buffer` loop can exit
    /// instead of spinning forever on a buffer that will never be released. Shared
    /// with `pool` so a single flag drives both the queue drain and the pooled
    /// channel poll.
    cancelled: Arc<AtomicBool>,
}

impl<F: PrimeField64 + Send + Sync + 'static> MemoryHandler<F> {
    pub fn new(pctx: Arc<ProofCtx<F>>, n_buffers: usize, buffer_size: usize) -> Self {
        let instance_ids_to_be_released = Arc::new(SegQueue::new());
        let cancelled = Arc::new(AtomicBool::new(false));

        // Page-lock the basic-trace buffer pool for direct H2D (pairs with the
        // direct-copy fast path in goldilocks_tooling.cu). The basic trace is an H2D
        // source, so it is pinned. Relies on pool buffers never permanently escaping —
        // shared-buffer traces recycle the same Vec back, and `reset` enforces it.
        let pool = Pool::new(n_buffers, buffer_size, true, cancelled.clone());

        let total_memory = n_buffers * buffer_size * std::mem::size_of::<F>();
        tracing::info!("MemoryHandler::Total memory for basic traces: {}", crate::format_bytes(total_memory as f64));

        Self { pctx, instance_ids_to_be_released, pool, cancelled }
    }

    /// Unblock any thread parked in `take_buffer`. Called on the abort path so a
    /// failed proof tears down cleanly instead of hanging on a buffer that will
    /// never be released.
    pub fn cancel(&self) {
        self.cancelled.store(true, Ordering::SeqCst);
    }

    /// Recover-only reset; see `Pool::reset` for the no-reallocate rationale.
    ///
    /// Sequencing requirement: all worker threads that took buffers must already be
    /// joined (so every buffer is back in the channel). Calling this while a worker
    /// is still running races the `try_recv` drain below and trips the
    /// `recovered N of M` error. Callers join workers — or `cancel()` then join —
    /// before resetting.
    pub fn reset(&self) -> ProofmanResult<()> {
        self.empty_queue_to_be_released();

        // swap, not load: reset() must also re-arm the flag, which is otherwise
        // sticky. The distributed worker cancels as housekeeping before every job;
        // left set, the flag turns the next run's take_buffer into an unbounded
        // fresh allocator (OOM on large blocks). Safe to clear here: workers are
        // joined before reset() (see doc above), so nothing is parked on the flag.
        //
        // On the abort path (flag set) we skip pool.reset's integrity checks — they
        // would mask the real cancellation error with a spurious invariant violation.
        // Clearing the flag here also makes the subsequent pool.reset() a no-op guard;
        // we return before calling it.
        if self.cancelled.swap(false, Ordering::SeqCst) {
            return Ok(());
        }

        // Buffer recovery + integrity checks live in the shared pool.
        self.pool.reset()
    }

    /// Take a basic-trace buffer. Polls both the pool channel and the soft-release
    /// SegQueue every 10µs. The pool's blocking `take()` won't work here:
    /// `to_be_released_buffer` enqueues without sending to the channel, so a parked
    /// `recv` would miss those wakeups — hence the non-blocking `try_take` in the loop.
    pub fn take_buffer(&self) -> Vec<F> {
        loop {
            if let Some(buffer) = self.pool.try_take() {
                return buffer;
            }
            // Abort path: a buffer this loop is waiting on may never be released
            // (the proof errored before releasing it). Return a fresh buffer so the
            // worker unblocks and the process can tear down instead of spinning.
            if self.pool.is_cancelled() {
                return self.pool.fresh_buffer();
            }
            if let Some((iid, remove_from_calculated)) = self.instance_ids_to_be_released.pop() {
                if remove_from_calculated {
                    self.pctx.dctx_reset_instance_calculated(iid);
                }
                let (is_shared, buf) = self.pctx.free_instance_traces(iid);
                if is_shared {
                    return buf;
                }
                continue;
            }
            std::thread::sleep(std::time::Duration::from_micros(10));
        }
    }

    pub fn release_buffer(&self, buffer: Vec<F>) -> ProofmanResult<()> {
        self.pool.release(buffer)
    }

    pub fn to_be_released_buffer(&self, instance_id: usize, remove_from_calculated: bool) {
        self.instance_ids_to_be_released.push((instance_id, remove_from_calculated));
    }

    pub fn get_n_buffers(&self) -> usize {
        self.pool.available()
    }

    pub fn empty_queue_to_be_released(&self) {
        while !self.instance_ids_to_be_released.is_empty() {
            self.instance_ids_to_be_released.pop();
        }
    }
}

// No explicit Drop: the `pool` field's `Pool::drop` unregisters the pinned pages
// when the last Arc<MemoryHandler> is gone. That teardown still hinges on workers
// being joined first — each holds a clone of the Arc, and a worker parked in
// take_buffer would never release it. `cancel()` unblocks take_buffer on the abort
// path so the thread exits and is joined; skip it and pinned pages may leak. (Under
// panic=abort no Drop runs at all; the OS reclaims the pages on process exit.)

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

pub struct MemoryHandlerRecursive<F: PrimeField64 + Send + Sync + 'static> {
    trace: Pool<F>,
    trace_compressor: Pool<F>,
    signal_values: Option<SignalValuesPool>,
    witness_threads: usize,
    cancelled: Arc<AtomicBool>,
}

impl<F: PrimeField64 + Send + Sync + 'static> MemoryHandlerRecursive<F> {
    pub fn new(
        n_buffers: usize,
        n_buffers_compressor: usize,
        buffer_size_trace: usize,
        buffer_size_trace_compressor: usize,
    ) -> Self {
        Self::new_with_signal_pool(
            n_buffers,
            n_buffers_compressor,
            buffer_size_trace,
            buffer_size_trace_compressor,
            None,
            8,
        )
    }

    pub fn new_with_signal_pool(
        n_buffers: usize,
        n_buffers_compressor: usize,
        buffer_size_trace: usize,
        buffer_size_trace_compressor: usize,
        signal_pool: Option<(usize, usize, usize)>,
        witness_threads: usize,
    ) -> Self {
        let cancelled = Arc::new(AtomicBool::new(false));
        // Trace pools are H2D sources so they are pinned; signalValues is CPU-only scratch.
        let trace = Pool::new(n_buffers, buffer_size_trace, true, cancelled.clone());
        let trace_compressor = Pool::new(n_buffers_compressor, buffer_size_trace_compressor, true, cancelled.clone());

        let total = trace.total_bytes() + trace_compressor.total_bytes();
        tracing::info!(
            "MemoryHandlerRecursive::Total memory for recursive traces: {}",
            crate::format_bytes(total as f64)
        );

        let signal_values = signal_pool.map(|(large_cap, small_cap, n_small)| {
            SignalValuesPool::new(large_cap, small_cap, n_small, cancelled.clone())
        });

        let witness_threads = witness_threads.max(1);
        tracing::info!("MemoryHandlerRecursive::circom solve threads per recursive witness: {}", witness_threads);

        Self { trace, trace_compressor, signal_values, witness_threads, cancelled }
    }

    /// Unblock any thread parked in a pooled `take()`. Called on the abort path so
    /// a failed proof tears down cleanly instead of hanging on a buffer that will
    /// never be released.
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

    pub fn reset(&self) -> ProofmanResult<()> {
        self.trace.reset()?;
        self.trace_compressor.reset()?;
        // Re-arm AFTER both pool resets: the pools share this flag, and each
        // Pool::reset relies on it (cancelled-aware early-out) — clearing earlier
        // would re-enable the integrity checks mid-teardown. Left set, the flag
        // turns take() into an unbounded fresh allocator on the next run.
        self.cancelled.store(false, Ordering::SeqCst);
        Ok(())
    }

    pub fn take_buffer_trace(&self) -> Vec<F> {
        self.trace.take()
    }
    pub fn release_buffer_trace(&self, buffer: Vec<F>) -> ProofmanResult<()> {
        self.trace.release(buffer)
    }

    pub fn take_buffer_trace_compressor(&self) -> Vec<F> {
        self.trace_compressor.take()
    }
    pub fn release_buffer_trace_compressor(&self, buffer: Vec<F>) -> ProofmanResult<()> {
        self.trace_compressor.release(buffer)
    }
}

/// Pool of reusable `signalValues` (u64) buffers handed into getWitnessTrace so
/// Circom_CalcWit reuses them instead of allocating tens-to-hundreds of MB per
/// proof. Buffers are NOT zeroed on reuse: the circom solve is write-before-read
/// (only signalValues[0]=1, reset inside the ctor), validated end-to-end.
///
/// Two size classes over bounded channels (a bounded channel's depth caps
/// concurrency; `recv` blocks when empty):
///
/// - `large`: one buffer sized to the biggest circuit (e.g. Keccakf compressor,
///   ~835MB). Depth 1 ⇒ at most one big-circuit witness allocates at a time.
/// - `small`: `n_small` buffers sized to the largest non-outlier circuit, for the
///   many lighter recursive proofs.
///
/// A `large` buffer fits any circuit, so a small proof may borrow it when smalls
/// are exhausted — but only if no big proof is queued for it (see `large_waiters`).
///
/// Every blocking wait is cancel-aware, like the trace `Pool`.
pub struct SignalValuesPool {
    large: Sender<Vec<u64>>,
    large_rx: Receiver<Vec<u64>>,
    small: Sender<Vec<u64>>,
    small_rx: Receiver<Vec<u64>>,
    large_cap: usize,
    small_cap: usize,
    /// Count of big proofs (needed > small_cap) currently blocked on the large
    /// buffer. While > 0, small proofs won't steal it — big circuits get priority.
    large_waiters: AtomicUsize,
    /// Shared with the owning `MemoryHandlerRecursive` and its trace pools.
    cancelled: Arc<AtomicBool>,
}

impl SignalValuesPool {
    /// `large_cap` / `small_cap` are in u64 elements (= get_total_signal_no()).
    pub fn new(large_cap: usize, small_cap: usize, n_small: usize, cancelled: Arc<AtomicBool>) -> Self {
        let (large, large_rx) = bounded(1);
        large.send(vec![0u64; large_cap]).unwrap();
        let (small, small_rx) = bounded(n_small.max(1));
        for _ in 0..n_small.max(1) {
            small.send(vec![0u64; small_cap]).unwrap();
        }
        let total = (large_cap + small_cap * n_small.max(1)) * 8;
        tracing::info!(
            "SignalValuesPool: 1 large ({}) + {} small ({} each) = {}",
            crate::format_bytes((large_cap * 8) as f64),
            n_small.max(1),
            crate::format_bytes((small_cap * 8) as f64),
            crate::format_bytes(total as f64),
        );
        Self { large, large_rx, small, small_rx, large_cap, small_cap, large_waiters: AtomicUsize::new(0), cancelled }
    }

    /// Blocking receive that yields to `cancelled`, mirroring `Pool::take`: on the abort
    /// path it hands back a fresh buffer instead of parking on one nobody will return.
    fn recv_cancellable(&self, rx: &Receiver<Vec<u64>>, cap: usize) -> Vec<u64> {
        loop {
            match rx.recv_timeout(Duration::from_micros(100)) {
                Ok(buffer) => return buffer,
                Err(crossbeam_channel::RecvTimeoutError::Timeout) => {
                    if self.cancelled.load(Ordering::SeqCst) {
                        return vec![0u64; cap];
                    }
                }
                Err(crossbeam_channel::RecvTimeoutError::Disconnected) => {
                    panic!("SignalValuesPool channel closed");
                }
            }
        }
    }

    /// Take a buffer with capacity >= `needed`. Big proofs (needed > small_cap)
    /// block on the large buffer with priority; small proofs prefer a small buffer
    /// and only borrow the large one when no big proof is waiting for it.
    pub fn take(&self, needed: usize) -> Vec<u64> {
        if needed > self.large_cap {
            // No pooled buffer fits — recursers are registered after the pool is sized.
            // Handing out the large one would overflow it; empty ⇒ null ⇒ C++ self-allocates.
            tracing::debug!(
                "SignalValuesPool: request of {} elements exceeds the largest pooled buffer ({}); \
                 falling back to a self-allocated signalValues buffer",
                needed,
                self.large_cap
            );
            return Vec::new();
        }
        let mut buffer = if needed > self.small_cap {
            // Only the large buffer fits. Register as a waiter so small proofs yield
            // it to us, then block until it's free.
            self.large_waiters.fetch_add(1, Ordering::SeqCst);
            let buf = self.recv_cancellable(&self.large_rx, self.large_cap);
            self.large_waiters.fetch_sub(1, Ordering::SeqCst);
            buf
        } else if let Ok(buf) = self.small_rx.try_recv() {
            buf
        } else if self.large_waiters.load(Ordering::SeqCst) == 0 {
            // Smalls exhausted, no big proof queued: borrow large non-blockingly;
            // if it's busy too, wait for a small rather than block a big proof.
            match self.large_rx.try_recv() {
                Ok(buf) => buf,
                Err(_) => self.recv_cancellable(&self.small_rx, self.small_cap),
            }
        } else {
            // A big proof is queued for the large buffer — don't contend; wait for a small.
            self.recv_cancellable(&self.small_rx, self.small_cap)
        };
        // The class a buffer sits in is not a hard guarantee of its length: on the abort
        // path `take` mints fresh buffers, so `release` can find a channel full and file a
        // short buffer under `large`. Never hand the C++ solve a buffer shorter than it
        // will write; growing it here also re-files it into the right class on release.
        if buffer.len() < needed {
            buffer.resize(needed, 0);
        }
        buffer
    }

    /// Return a buffer, preferring the class its length belongs to. Never blocks: with
    /// `large_cap == small_cap` every buffer looks "large" and the depth-1 large channel
    /// fills, so a blocking send would deadlock — fall back to the other class, and drop
    /// only if both are full (which means `take` minted extras on the abort path). A short
    /// buffer can therefore end up under `large`; `take` re-grows it rather than trusting
    /// the class.
    pub fn release(&self, buffer: Vec<u64>) {
        let (preferred, fallback) =
            if buffer.len() >= self.large_cap { (&self.large, &self.small) } else { (&self.small, &self.large) };
        match preferred.try_send(buffer) {
            Ok(()) => {}
            Err(crossbeam_channel::TrySendError::Full(buffer)) => {
                // On the abort path `take` hands out fresh buffers, so live buffers can
                // outnumber channel capacity; dropping the extra is correct there.
                let _ = fallback.try_send(buffer);
            }
            Err(crossbeam_channel::TrySendError::Disconnected(_)) => {
                panic!("SignalValuesPool channel closed");
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use fields::{Field, Goldilocks};

    // These exercise the pool's accounting (take/release/reset round-trip, leak
    // detection, cancel). They don't assert anything about pinning, so they pass on
    // both backends: CPU (register is a no-op) and GPU (the tiny buffers are page-
    // locked, but that's transparent to the accounting under test).
    type F = Goldilocks;

    fn handler(n: usize, n_comp: usize, size: usize, size_comp: usize) -> MemoryHandlerRecursive<F> {
        MemoryHandlerRecursive::new(n, n_comp, size, size_comp)
    }

    #[test]
    fn clean_round_trip_then_reset_succeeds() {
        let h = handler(2, 1, 8, 4);
        // Take every buffer out of each pool, then release them all back.
        let t0 = h.take_buffer_trace();
        let t1 = h.take_buffer_trace();
        let tc = h.take_buffer_trace_compressor();
        h.release_buffer_trace(t0).unwrap();
        h.release_buffer_trace(t1).unwrap();
        h.release_buffer_trace_compressor(tc).unwrap();
        // This is the gap-2 invariant: a clean round trip leaves full pools, so the
        // reset wired into ProofMan::reset() passes.
        h.reset().unwrap();
        // And it is idempotent across reuse.
        h.reset().unwrap();
    }

    #[test]
    fn reset_detects_a_leaked_buffer() {
        let h = handler(2, 0, 8, 0);
        // Simulate a leak: a worker took a buffer and never released it (e.g. an
        // early `?` return on a non-cancelled path). reset() must surface the short
        // pool rather than paper over it — this is exactly what the gap-2 reset wired
        // into ProofMan::reset() now catches.
        let _leaked = h.take_buffer_trace();
        assert!(h.reset().is_err());
    }

    #[test]
    fn release_rejects_wrong_size_buffer() {
        let h = handler(1, 0, 8, 0);
        let _good = h.take_buffer_trace();
        // Release a buffer of the wrong length; the size check rejects it.
        assert!(h.release_buffer_trace(vec![F::ZERO; 7]).is_err());
    }

    #[test]
    fn cancel_unblocks_take_and_skips_reset_checks() {
        let h = handler(1, 0, 8, 0);
        // Empty the pool, then cancel. A subsequent take must return a fresh buffer
        // instead of blocking forever, and reset must not flag the (now short) pool.
        let _taken = h.take_buffer_trace();
        h.cancel();
        let fresh = h.take_buffer_trace(); // would hang pre-cancel on an empty pool
        assert_eq!(fresh.len(), 8);
        h.reset().unwrap(); // cancelled path skips integrity checks
    }

    // ---- signalValues pool ----

    fn signal_handler(large: usize, small: usize, n_small: usize) -> MemoryHandlerRecursive<F> {
        MemoryHandlerRecursive::new_with_signal_pool(1, 0, 8, 0, Some((large, small, n_small)), 4)
    }

    #[test]
    fn signal_pool_sizes_buffers_by_class() {
        let h = signal_handler(64, 16, 2);
        // A request that fits a small gets a small; one that doesn't gets the large.
        let s = h.take_buffer_signal_values(16);
        assert!(s.len() >= 16);
        let l = h.take_buffer_signal_values(17);
        assert!(l.len() >= 17);
        h.release_buffer_signal_values(s);
        h.release_buffer_signal_values(l);
        // Round-trip is repeatable: the large buffer went back to the large class.
        let l2 = h.take_buffer_signal_values(64);
        assert_eq!(l2.len(), 64);
        h.release_buffer_signal_values(l2);
    }

    #[test]
    fn signal_pool_request_larger_than_pool_yields_empty_buffer() {
        let h = signal_handler(64, 16, 2);
        // Bigger than any pooled buffer (a late-registered recurser): must not be handed
        // the too-short large one. Empty => null => C++ self-allocates.
        assert!(h.take_buffer_signal_values(65).is_empty());
        assert_eq!(h.take_buffer_signal_values(64).len(), 64);
    }

    #[test]
    fn signal_pool_absent_yields_empty_buffer() {
        // No pool configured => empty Vec (the null sentinel); releasing it is a no-op.
        let h = handler(1, 0, 8, 0);
        let buf = h.take_buffer_signal_values(1024);
        assert!(buf.is_empty());
        h.release_buffer_signal_values(buf);
    }

    #[test]
    fn signal_pool_take_never_returns_a_short_buffer_after_an_abort() {
        // Reproduces the class-mixing the abort path can cause: with both classes drained,
        // `cancel` makes take() mint a fresh small buffer; releasing all three then finds
        // the small channel full and files a SHORT buffer under `large`. A later big
        // request must still get something it can safely write `needed` elements into.
        let h = signal_handler(64, 16, 1);
        let s = h.take_buffer_signal_values(16);
        let l = h.take_buffer_signal_values(64);
        h.cancel();
        let fresh = h.take_buffer_signal_values(16);
        h.release_buffer_signal_values(s);
        h.release_buffer_signal_values(fresh); // small full -> lands in `large`
        h.release_buffer_signal_values(l); // large full -> dropped
        h.reset().unwrap(); // re-arms `cancelled`, as the distributed worker does
        assert!(h.take_buffer_signal_values(64).len() >= 64, "handed the solve a short buffer");
    }

    #[test]
    fn signal_pool_cancel_unblocks_take() {
        let h = signal_handler(64, 16, 1);
        // Drain both classes, then cancel: a further take must return a fresh buffer of
        // the right class rather than parking forever.
        let _s = h.take_buffer_signal_values(16);
        let _l = h.take_buffer_signal_values(64);
        h.cancel();
        assert_eq!(h.take_buffer_signal_values(16).len(), 16);
        assert_eq!(h.take_buffer_signal_values(64).len(), 64);
    }
}
