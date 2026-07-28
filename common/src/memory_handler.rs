use crossbeam_channel::{bounded, Sender, Receiver};
use std::collections::HashSet;
use std::ffi::c_void;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;
use crossbeam_queue::SegQueue;
use crate::ProofCtx;
use fields::PrimeField64;
use crate::{ProofmanError, ProofmanResult};
use proofman_starks_lib_c::{register_host_memory_c, unregister_host_memory_c};

/// Ceiling on the re-check interval of a thread waiting for a pooled buffer. A released buffer
/// wakes it immediately, so this bounds cancel responsiveness, not pickup latency.
const MAX_POOL_WAIT_BACKOFF: Duration = Duration::from_millis(1);

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
}

impl<F: PrimeField64 + Send + Sync + 'static> Pool<F> {
    fn new(n_buffers: usize, buffer_size: usize, pin: bool, cancelled: Arc<AtomicBool>) -> Self {
        let (sender, receiver) = bounded(n_buffers);
        let buffers: Vec<Vec<F>> = (0..n_buffers).map(|_| vec![F::ZERO; buffer_size]).collect();
        let registered_buffers: Vec<usize> = if pin { register_pool(&buffers) } else { Vec::new() };
        // Record our own buffers by pointer; they're never reallocated, so the pointers stay stable.
        let original_ptrs: HashSet<usize> = buffers.iter().map(|b| b.as_ptr() as usize).collect();
        for buffer in buffers {
            sender.send(buffer).unwrap();
        }
        Self { sender, receiver, n_buffers, buffer_size, registered_buffers, cancelled, original_ptrs }
    }

    fn take(&self) -> Vec<F> {
        // The timeout only paces the abort-flag re-check, so it escalates: a fixed 100us cost
        // 10k wakeups/s per waiter, exactly while the pool was exhausted.
        let mut backoff = Duration::from_micros(100);
        loop {
            match self.receiver.recv_timeout(backoff) {
                Ok(buffer) => return buffer,
                Err(crossbeam_channel::RecvTimeoutError::Timeout) => {
                    // on cancel, hand back a fresh buffer so teardown doesn't hang
                    if self.cancelled.load(Ordering::SeqCst) {
                        return vec![F::ZERO; self.buffer_size];
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

    fn release(&self, buffer: Vec<F>) -> ProofmanResult<()> {
        if buffer.len() != self.buffer_size {
            return Err(ProofmanError::ProofmanError(format!(
                "Pool::release: wrong size {} (expected {})",
                buffer.len(),
                self.buffer_size
            )));
        }
        // Only the pool's own buffers go back to the channel; a fresh cancel-escape buffer is freed
        // here instead (pooling it would put an unpinned buffer in a pinned pool). Since exactly the
        // n_buffers originals are ever pooled, releasing one always finds room — this send can't block.
        // Dropping the fresh buffer is also what keeps a REGISTERED buffer from being freed while its
        // cudaHostRegister is live: only originals (the registered ones) are ever recycled.
        if self.original_ptrs.contains(&(buffer.as_ptr() as usize)) {
            self.sender.send(buffer).expect("Pool channel closed");
        }
        // else: a fresh cancel-escape buffer — let it drop (freed); never pooled.
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

pub struct MemoryHandler<F: PrimeField64 + Send + Sync + 'static> {
    pctx: Arc<ProofCtx<F>>,
    instance_ids_to_be_released: Arc<SegQueue<(usize, bool)>>,
    /// Channel + pinning + reset/Drop mechanics for the basic-trace buffers. The instance-release
    /// side-channel below and the `pctx` coupling are the only behavior layered on the shared pool.
    pool: Pool<F>,
    /// Set by `cancel()` so the `take_buffer` loop can exit instead of spinning on a buffer that
    /// will never be released. Shared with `pool` so one flag drives both the drain and channel poll.
    cancelled: Arc<AtomicBool>,
}

impl<F: PrimeField64 + Send + Sync + 'static> MemoryHandler<F> {
    pub fn new(pctx: Arc<ProofCtx<F>>, n_buffers: usize, buffer_size: usize) -> Self {
        let instance_ids_to_be_released = Arc::new(SegQueue::new());
        let cancelled = Arc::new(AtomicBool::new(false));

        // Page-lock the basic-trace pool for direct H2D (trace is an H2D source; pairs with the
        // direct-copy fast path in goldilocks_tooling.cu). Relies on buffers never permanently
        // escaping the pool, which `reset` enforces.
        let pool = Pool::new(n_buffers, buffer_size, true, cancelled.clone());

        let total_memory = n_buffers * buffer_size * std::mem::size_of::<F>();
        tracing::info!("MemoryHandler::Total memory for basic traces: {}", crate::format_bytes(total_memory as f64));

        Self { pctx, instance_ids_to_be_released, pool, cancelled }
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
        self.empty_queue_to_be_released();

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
    /// two non-channel wakeup sources: the abort flag, and the soft-release SegQueue that
    /// `to_be_released_buffer` enqueues without sending to the channel (so a bare parked `recv`
    /// would miss those). Was a 10µs sleep-poll — ~100k wakeups/s per waiter, inside
    /// CALCULATING_WITNESS, each iteration touching state shared with the releasing threads.
    pub fn take_buffer(&self) -> Vec<F> {
        let mut backoff = std::time::Duration::from_micros(50);
        loop {
            if let Some(buffer) = self.pool.try_take() {
                return buffer;
            }
            // Abort path: the awaited buffer may never be released (proof errored first), so return
            // a fresh buffer to unblock the worker and let the process tear down instead of spinning.
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
            if let Some(buffer) = self.pool.take_timeout(backoff) {
                return buffer;
            }
            backoff = (backoff * 2).min(MAX_POOL_WAIT_BACKOFF);
        }
    }

    pub fn release_buffer(&self, buffer: Vec<F>) -> ProofmanResult<()> {
        self.pool.release(buffer)
    }

    pub fn to_be_released_buffer(&self, instance_id: usize, remove_from_calculated: bool) {
        self.instance_ids_to_be_released.push((instance_id, remove_from_calculated));
    }

    pub fn empty_queue_to_be_released(&self) {
        while !self.instance_ids_to_be_released.is_empty() {
            self.instance_ids_to_be_released.pop();
        }
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

pub struct MemoryHandlerRecursive<F: PrimeField64 + Send + Sync + 'static> {
    witness: Pool<F>,
    witness_compressor: Pool<F>,
    trace: Pool<F>,
    trace_compressor: Pool<F>,
    cancelled: Arc<AtomicBool>,
}

impl<F: PrimeField64 + Send + Sync + 'static> MemoryHandlerRecursive<F> {
    pub fn new(
        n_buffers: usize,
        n_buffers_compressor: usize,
        buffer_size_witness: usize,
        buffer_size_witness_compressor: usize,
        buffer_size_trace: usize,
        buffer_size_trace_compressor: usize,
    ) -> Self {
        let cancelled = Arc::new(AtomicBool::new(false));
        let witness = Pool::new(n_buffers, buffer_size_witness, false, cancelled.clone());
        let witness_compressor =
            Pool::new(n_buffers_compressor, buffer_size_witness_compressor, false, cancelled.clone());
        let trace = Pool::new(n_buffers, buffer_size_trace, true, cancelled.clone());
        let trace_compressor = Pool::new(n_buffers_compressor, buffer_size_trace_compressor, true, cancelled.clone());

        let total = witness.total_bytes()
            + witness_compressor.total_bytes()
            + trace.total_bytes()
            + trace_compressor.total_bytes();
        tracing::info!(
            "MemoryHandlerRecursive::Total memory for recursive traces: {}",
            crate::format_bytes(total as f64)
        );

        Self { witness, witness_compressor, trace, trace_compressor, cancelled }
    }

    /// Unblock any thread parked in a pooled `take()`. Called on the abort path so a failed proof
    /// tears down cleanly instead of hanging on a buffer that will never be released.
    pub fn cancel(&self) {
        self.cancelled.store(true, Ordering::SeqCst);
    }

    /// Reset all four pools and report the first failure — never `?` out early. A short witness pool
    /// must not stop the compressor pools from being recovered, and above all must not skip clearing
    /// `cancelled`: left set, it turns the next run's `take()` into an unbounded fresh allocator
    /// handing out unpinned buffers. Each pool's own reset is non-destructive, so continuing past a
    /// failure cannot lose anything.
    pub fn reset(&self) -> ProofmanResult<()> {
        let results = [
            ("witness", self.witness.reset()),
            ("witness_compressor", self.witness_compressor.reset()),
            ("trace", self.trace.reset()),
            ("trace_compressor", self.trace_compressor.reset()),
        ];
        // Re-arm AFTER all four pool resets: the pools share this flag and each Pool::reset reads it
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

    pub fn take_buffer_witness(&self) -> Vec<F> {
        self.witness.take()
    }
    pub fn release_buffer_witness(&self, buffer: Vec<F>) -> ProofmanResult<()> {
        self.witness.release(buffer)
    }

    pub fn take_buffer_witness_compressor(&self) -> Vec<F> {
        self.witness_compressor.take()
    }
    pub fn release_buffer_witness_compressor(&self, buffer: Vec<F>) -> ProofmanResult<()> {
        self.witness_compressor.release(buffer)
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

    /// Take a recursive-proof trace buffer as a release-on-drop lease (see [`BufferLease`]).
    /// `compressor` selects the (differently-sized) compressor trace pool.
    pub fn take_trace_lease(&self, compressor: bool) -> BufferLease<'_, F> {
        let (buffer, pool) = if compressor {
            (self.trace_compressor.take(), RecursivePool::TraceCompressor)
        } else {
            (self.trace.take(), RecursivePool::Trace)
        };
        BufferLease { handler: self, buffer: Some(buffer), pool }
    }

    /// Adopt an already-taken witness buffer (a `Proof`'s `circom_witness`) into a release-on-drop
    /// lease so it returns to its pool on every exit path instead of leaking on cancel/error. `adopt`,
    /// not `take`: the buffer already left the pool. `compressor` selects the compressor witness pool.
    pub fn adopt_witness(&self, buffer: Vec<F>, compressor: bool) -> BufferLease<'_, F> {
        let pool = if compressor { RecursivePool::WitnessCompressor } else { RecursivePool::Witness };
        BufferLease { handler: self, buffer: Some(buffer), pool }
    }
}

/// Which of the four recursive pools a [`BufferLease`] returns its buffer to on drop.
#[derive(Clone, Copy)]
enum RecursivePool {
    Witness,
    WitnessCompressor,
    Trace,
    TraceCompressor,
}

/// A recursive-proof buffer that returns itself to its pool when dropped (success, early `?`, or
/// panic) so it can't leak and shrink the pool. Obtain via `take_trace_lease` or `adopt_witness`;
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
                RecursivePool::Witness => self.handler.release_buffer_witness(buffer),
                RecursivePool::WitnessCompressor => self.handler.release_buffer_witness_compressor(buffer),
                RecursivePool::Trace => self.handler.release_buffer_trace(buffer),
                RecursivePool::TraceCompressor => self.handler.release_buffer_trace_compressor(buffer),
            };
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use fields::{Field, Goldilocks};

    // These exercise the pool's accounting (take/release/reset, leak detection, cancel), not pinning,
    // so they pass on both backends (CPU register is a no-op; GPU pinning is transparent to accounting).
    type F = Goldilocks;

    fn handler(n: usize, n_comp: usize, size: usize, size_comp: usize) -> MemoryHandlerRecursive<F> {
        MemoryHandlerRecursive::new(n, n_comp, size, size_comp, size, size_comp)
    }

    #[test]
    fn clean_round_trip_then_reset_succeeds() {
        let h = handler(2, 1, 8, 4);
        // Take every buffer out of each pool, then release them all back.
        let w0 = h.take_buffer_witness();
        let w1 = h.take_buffer_witness();
        let t0 = h.take_buffer_trace();
        let t1 = h.take_buffer_trace();
        let wc = h.take_buffer_witness_compressor();
        let tc = h.take_buffer_trace_compressor();
        h.release_buffer_witness(w0).unwrap();
        h.release_buffer_witness(w1).unwrap();
        h.release_buffer_trace(t0).unwrap();
        h.release_buffer_trace(t1).unwrap();
        h.release_buffer_witness_compressor(wc).unwrap();
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
        // Simulate a leak: a worker took a buffer and never released it (e.g. an early `?` on a
        // non-cancelled path). reset() must surface the short pool rather than paper over it.
        let _leaked = h.take_buffer_witness();
        assert!(h.reset().is_err());
    }

    #[test]
    fn release_rejects_wrong_size_buffer() {
        let h = handler(1, 0, 8, 0);
        let _good = h.take_buffer_witness();
        // Release a buffer of the wrong length; the size check rejects it.
        assert!(h.release_buffer_witness(vec![F::ZERO; 7]).is_err());
    }

    #[test]
    fn a_failed_reset_keeps_the_buffers_it_recovered() {
        // reset() reports a short pool, but must not DESTROY what came back: dropping the drained
        // buffers would turn "short by one" into "empty", and would free pages that
        // `registered_buffers` still points at, so Pool::drop would unregister freed memory.
        let h = handler(3, 0, 8, 0);
        let leaked = h.take_buffer_witness(); // never released
        assert!(h.reset().is_err(), "a missing buffer must still be reported");

        // The other two are still pooled and usable, and a third take does not block.
        let a = h.take_buffer_witness();
        let b = h.take_buffer_witness();
        assert_eq!((a.len(), b.len()), (8, 8));
        h.release_buffer_witness(a).unwrap();
        h.release_buffer_witness(b).unwrap();
        // Returning the escapee makes the pool whole again — impossible if reset() had dropped the rest.
        h.release_buffer_witness(leaked).unwrap();
        h.reset().expect("pool is whole once the escapee comes back");
    }

    #[test]
    fn reset_covers_every_pool_even_when_an_earlier_one_fails() {
        // The compressor pools must be recovered (and `cancelled` cleared) even when the plain
        // witness pool is short — an early `?` used to skip both.
        let h = handler(1, 1, 8, 4);
        let leaked = h.take_buffer_witness();
        let comp = h.take_buffer_witness_compressor();
        h.release_buffer_witness_compressor(comp).unwrap();

        assert!(h.reset().is_err(), "the short witness pool is reported");
        // Compressor pool was still visited and is whole: taking from it must not block.
        let comp = h.take_buffer_witness_compressor();
        assert_eq!(comp.len(), 4);
        h.release_buffer_witness_compressor(comp).unwrap();
        h.release_buffer_witness(leaked).unwrap();
        h.reset().expect("all pools whole");
    }

    #[test]
    fn a_failed_reset_still_clears_the_cancelled_flag() {
        // Left set, `cancelled` makes take() an unbounded fresh allocator handing out unpinned
        // buffers. That must not survive a reset that reported an error.
        let h = handler(1, 1, 8, 4);
        h.cancel();
        let _escapee = h.take_buffer_witness(); // empty pool + cancelled -> fresh buffer
        let _ = h.reset(); // cancelled path: warns rather than errors
                           // Flag cleared, so the pool is authoritative again: it holds its one original buffer.
        let original = h.take_buffer_witness();
        h.release_buffer_witness(original).unwrap();
        h.reset().expect("pool whole and no longer in cancelled mode");
    }

    #[test]
    fn releasing_a_cancel_escape_buffer_does_not_pool_it() {
        // `release` pools only the pool's own (registered) buffers. A fresh buffer handed out by a
        // cancelled `take()` is unregistered, so pooling it would put an unpinned buffer in a pinned
        // pool — and would also push the channel past capacity. It must be dropped instead, while
        // the originals still come back.
        let h = handler(1, 0, 8, 0);
        let original = h.take_buffer_witness();
        h.cancel();
        let fresh = h.take_buffer_witness();
        // Release both, fresh first, so a mistakenly-pooled fresh buffer would occupy the one slot.
        h.release_buffer_witness(fresh).unwrap();
        h.release_buffer_witness(original).unwrap();
        h.reset().unwrap();
        // The pool is whole again and hands out its own buffer, not the escapee.
        assert_eq!(h.take_buffer_witness().len(), 8);
    }

    #[test]
    fn adopt_witness_returns_the_buffer_to_the_right_pool() {
        // Teardown recovery relies on this: a compressor witness must go back to the compressor pool
        // (the smallest one), keyed off the proof type, or that pool silently shrinks.
        let h = handler(1, 1, 8, 4);
        let plain = h.take_buffer_witness();
        let comp = h.take_buffer_witness_compressor();
        drop(h.adopt_witness(plain, false));
        drop(h.adopt_witness(comp, true));
        h.reset().expect("both witness pools whole after adopt-then-drop");
    }

    #[test]
    fn cancel_unblocks_take_and_skips_reset_checks() {
        let h = handler(1, 0, 8, 0);
        // Empty the pool, then cancel. A subsequent take must return a fresh buffer
        // instead of blocking forever, and reset must not flag the (now short) pool.
        let _taken = h.take_buffer_witness();
        h.cancel();
        let fresh = h.take_buffer_witness(); // would hang pre-cancel on an empty pool
        assert_eq!(fresh.len(), 8);
        h.reset().unwrap(); // cancelled path skips integrity checks
    }
}
