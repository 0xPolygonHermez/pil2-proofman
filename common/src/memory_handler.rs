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

/// Round a host range out to page boundaries — `cudaHostRegister` requires the
/// region to cover whole pages.
fn aligned_host_range(ptr: usize, bytes: usize) -> Option<(usize, usize)> {
    if ptr == 0 || bytes == 0 {
        return None;
    }
    let page_size = unsafe { libc::sysconf(libc::_SC_PAGESIZE) };
    let page_size = if page_size > 0 { page_size as usize } else { 4096 };
    let base = ptr & !(page_size - 1);
    let offset = ptr - base;
    let size = (bytes + offset + page_size - 1) & !(page_size - 1);
    Some((base, size))
}

/// Page-lock a buffer's backing pages for direct H2D. Returns `Some((base, bytes))`
/// so the caller can unregister by the same base pointer (CUDA requires it), or
/// `None` if the backend declined (CPU: no GPU to pin) or the page-lock failed.
/// Registration pins a *specific address*, so a registered buffer must not be
/// reallocated for the life of the pool — see the recover-only `reset`.
fn register_buffer<F: PrimeField64>(buffer: &mut [F]) -> Option<(usize, usize)> {
    let bytes = buffer.len().saturating_mul(std::mem::size_of::<F>());
    let (base, size) = aligned_host_range(buffer.as_mut_ptr() as usize, bytes)?;
    if register_host_memory_c(base as *mut c_void, size as u64) {
        Some((base, size))
    } else {
        None
    }
}

/// Register a freshly-allocated pool, enforcing all-or-nothing pinning: either
/// every buffer is pinned (GPU backend) or none are (CPU backend, where pinning
/// is a no-op). A partial result means a `cudaHostRegister` failed under the GPU
/// backend, leaving an unpinned buffer in a pool the GPU treats as pinned — we
/// refuse to start rather than run with that soundness hole.
fn register_pool<F: PrimeField64>(buffers: &mut [Vec<F>]) -> Vec<usize> {
    let mut registered: Vec<usize> = Vec::with_capacity(buffers.len());
    for buffer in buffers.iter_mut() {
        if let Some((base, _)) = register_buffer(buffer) {
            registered.push(base);
        }
    }
    // Partial pinning is fatal (all-or-nothing), but before we abort, unregister
    // the buffers we did pin — otherwise those pages stay locked for the rest of
    // the process (and, under panic=abort, Drop never runs to release them).
    if !registered.is_empty() && registered.len() != buffers.len() {
        for ptr in &registered {
            unregister_host_memory_c(*ptr as *mut c_void);
        }
        panic!(
            "MemoryHandler: host-memory pinning is all-or-nothing, but only {} of {} buffers pinned. \
             The GPU backend is active and a cudaHostRegister failed — refusing to run with a \
             partially-pinned pool.",
            registered.len(),
            buffers.len()
        );
    }
    registered
}

/// Re-assert the pinning invariant on `reset`: every recovered buffer must sit
/// at one of the originally-registered addresses (no-op when pinning is off). An
/// unknown address means a buffer escaped the pool or was reallocated, leaving it
/// unpinned and the original registration dangling.
fn verify_registered<F: PrimeField64>(
    registered: &HashSet<usize>,
    buffers: &[Vec<F>],
    ctx: &str,
) -> ProofmanResult<()> {
    if registered.is_empty() {
        return Ok(()); // pinning disabled (CPU backend) — nothing to verify
    }
    for buf in buffers {
        let bytes = buf.len().saturating_mul(std::mem::size_of::<F>());
        let base = aligned_host_range(buf.as_ptr() as usize, bytes).map(|(b, _)| b);
        if !base.map(|b| registered.contains(&b)).unwrap_or(false) {
            return Err(ProofmanError::ProofmanError(format!(
                "{ctx}: recovered a buffer at an unregistered address — the pinned-pool \
                 invariant was violated (a buffer escaped the pool or was reallocated)"
            )));
        }
    }
    Ok(())
}

/// Single fixed-size buffer pool over a bounded channel. Internal helper used
/// by `MemoryHandlerRecursive`. `take()` is a blocking channel recv — wakes
/// instantly when a buffer is released, no polling.
struct Pool<F: PrimeField64 + Send + Sync + 'static> {
    sender: Sender<Vec<F>>,
    receiver: Receiver<Vec<F>>,
    n_buffers: usize,
    buffer_size: usize,
    /// Page-locked base pointers of the pooled buffers: the authoritative set the
    /// pool owns. Empty iff pinning is disabled (CPU backend). Used to unregister
    /// on Drop and to verify the pool stays intact across `reset`.
    registered_buffers: HashSet<usize>,
    /// Shared with the owning `MemoryHandlerRecursive`; set on the abort path so a
    /// blocking `take()` exits instead of parking forever on `recv`.
    cancelled: Arc<AtomicBool>,
}

impl<F: PrimeField64 + Send + Sync + 'static> Pool<F> {
    fn new(n_buffers: usize, buffer_size: usize, cancelled: Arc<AtomicBool>) -> Self {
        let (sender, receiver) = bounded(n_buffers);
        let mut buffers: Vec<Vec<F>> = (0..n_buffers).map(|_| vec![F::ZERO; buffer_size]).collect();
        let registered_buffers: HashSet<usize> = register_pool(&mut buffers).into_iter().collect();
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

    fn release(&self, buffer: Vec<F>) -> ProofmanResult<()> {
        if buffer.len() != self.buffer_size {
            return Err(ProofmanError::ProofmanError(format!(
                "Pool::release: wrong size {} (expected {})",
                buffer.len(),
                self.buffer_size
            )));
        }
        // On the abort path take() may hand out freshly-allocated buffers without
        // drawing from the channel, so the live buffer count can exceed the channel
        // capacity. A blocking send would then park forever on a full channel,
        // reintroducing the teardown hang. Release best-effort when cancelled: the
        // dropped buffer is freed normally and the pool is about to be torn down.
        if self.cancelled.load(Ordering::SeqCst) {
            let _ = self.sender.try_send(buffer);
            return Ok(());
        }
        self.sender.send(buffer).expect("Pool channel closed");
        Ok(())
    }

    /// Recover-only reset that re-asserts the pinning invariant: every buffer must
    /// come back, and (when pinned) at one of the originally-registered addresses.
    /// We must NOT reallocate a replacement — a fresh allocation would be unpinned
    /// and leave a stale registration that Drop later unregisters against freed
    /// memory. A short count or unknown address means a buffer escaped, so we
    /// surface it rather than paper over it.
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
        verify_registered(&self.registered_buffers, &valid_buffers, "Pool::reset")?;
        for buf in valid_buffers {
            self.sender.send(buf).expect("Pool channel closed");
        }
        Ok(())
    }

    fn total_bytes(&self) -> usize {
        self.n_buffers * self.buffer_size * std::mem::size_of::<F>()
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
    sender: Sender<Vec<F>>,
    receiver: Receiver<Vec<F>>,
    n_buffers: usize,
    buffer_size: usize,
    /// Page-locked base pointers the pool owns (see `Pool::registered_buffers`).
    registered_buffers: HashSet<usize>,
    /// Set by `cancel()` on the abort path so the blocking `take_buffer` loop can
    /// exit instead of spinning forever on a buffer that will never be released.
    cancelled: Arc<AtomicBool>,
}

impl<F: PrimeField64 + Send + Sync + 'static> MemoryHandler<F> {
    pub fn new(pctx: Arc<ProofCtx<F>>, n_buffers: usize, buffer_size: usize) -> Self {
        let (tx_buffer_pool, rx_buffer_pool) = bounded(n_buffers);
        let instance_ids_to_be_released = Arc::new(SegQueue::new());

        // Page-lock the basic-trace buffer pool for direct H2D (pairs with the
        // direct-copy fast path in goldilocks_tooling.cu). All-or-nothing (see
        // register_pool). Relies on pool buffers never permanently escaping —
        // shared-buffer traces recycle the same Vec back, and `reset` enforces it.
        let mut buffers: Vec<Vec<F>> = (0..n_buffers).map(|_| vec![F::ZERO; buffer_size]).collect();
        let registered_buffers: HashSet<usize> = register_pool(&mut buffers).into_iter().collect();
        let registered_bytes: usize =
            registered_buffers.len().saturating_mul(buffer_size).saturating_mul(std::mem::size_of::<F>());
        for buffer in buffers {
            tx_buffer_pool.send(buffer).unwrap();
        }

        let total_memory = n_buffers * buffer_size * std::mem::size_of::<F>();
        tracing::info!("MemoryHandler::Total memory for basic traces: {}", crate::format_bytes(total_memory as f64));
        tracing::info!(
            "MemoryHandler::registered {} basic-trace buffers ({}) for direct H2D",
            registered_buffers.len(),
            crate::format_bytes(registered_bytes as f64)
        );

        Self {
            pctx,
            sender: tx_buffer_pool,
            receiver: rx_buffer_pool,
            instance_ids_to_be_released,
            n_buffers,
            buffer_size,
            registered_buffers,
            cancelled: Arc::new(AtomicBool::new(false)),
        }
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

        // On the abort path take_buffer hands out a fresh (unregistered) buffer to
        // unblock workers, which may then be released back into the pool. The pool
        // is about to be dropped, so don't run the integrity checks — they would
        // mask the real cancellation error with a spurious invariant violation.
        if self.cancelled.load(Ordering::SeqCst) {
            return Ok(());
        }

        let mut valid_buffers: Vec<Vec<F>> = Vec::with_capacity(self.n_buffers);
        while let Ok(buf) = self.receiver.try_recv() {
            if buf.len() != self.buffer_size {
                return Err(ProofmanError::ProofmanError(format!(
                    "MemoryHandler::reset: buffer with unexpected size {} (expected {})",
                    buf.len(),
                    self.buffer_size
                )));
            }
            valid_buffers.push(buf);
        }

        if valid_buffers.len() != self.n_buffers {
            return Err(ProofmanError::ProofmanError(format!(
                "MemoryHandler::reset: recovered {} of {} buffers; a buffer was not released",
                valid_buffers.len(),
                self.n_buffers
            )));
        }

        verify_registered(&self.registered_buffers, &valid_buffers, "MemoryHandler::reset")?;

        for buf in valid_buffers.into_iter() {
            self.sender.send(buf).unwrap();
        }

        Ok(())
    }

    /// Take a basic-trace buffer. Polls both the channel and the soft-release SegQueue
    /// every 10µs. A blocking `recv` won't work here: `to_be_released_buffer` enqueues
    /// without sending to the channel, so a parked `recv` would miss those wakeups.
    pub fn take_buffer(&self) -> Vec<F> {
        loop {
            if let Ok(buffer) = self.receiver.try_recv() {
                return buffer;
            }
            // Abort path: a buffer this loop is waiting on may never be released
            // (the proof errored before releasing it). Return a fresh buffer so the
            // worker unblocks and the process can tear down instead of spinning.
            if self.cancelled.load(Ordering::SeqCst) {
                return vec![F::ZERO; self.buffer_size];
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
        if buffer.len() != self.buffer_size {
            return Err(ProofmanError::ProofmanError(format!(
                "MemoryHandler::Trying to release buffer with unexpected size {} (expected {}).",
                buffer.len(),
                self.buffer_size
            )));
        }
        // On the abort path take_buffer may hand out freshly-allocated buffers
        // without drawing from the channel, so the live buffer count can exceed the
        // channel capacity. A blocking send would then park forever on a full
        // channel, reintroducing the teardown hang. Release best-effort when
        // cancelled: the dropped buffer is freed normally and the pool is about to
        // be torn down.
        if self.cancelled.load(Ordering::SeqCst) {
            let _ = self.sender.try_send(buffer);
            return Ok(());
        }
        self.sender.send(buffer).expect("Failed to send buffer back to pool");
        Ok(())
    }

    pub fn to_be_released_buffer(&self, instance_id: usize, remove_from_calculated: bool) {
        self.instance_ids_to_be_released.push((instance_id, remove_from_calculated));
    }

    pub fn get_n_buffers(&self) -> usize {
        self.receiver.len()
    }

    pub fn empty_queue_to_be_released(&self) {
        while !self.instance_ids_to_be_released.is_empty() {
            self.instance_ids_to_be_released.pop();
        }
    }
}

impl<F: PrimeField64 + Send + Sync + 'static> Drop for MemoryHandler<F> {
    fn drop(&mut self) {
        // Unregistration hinges on this Drop running, which only happens once the
        // LAST Arc<MemoryHandler> is gone. Worker threads each hold a clone moved
        // into their closure, so they must terminate and be joined first. A worker
        // parked in take_buffer would never release its Arc — that is exactly what
        // `cancel()` prevents on the abort path (it unblocks take_buffer so the
        // thread exits and is joined). If you stop calling `cancel()` before
        // joining, pinned pages may leak. (Under panic=abort no Drop runs at all;
        // the OS reclaims the pages on process exit.)
        //
        // Runs before fields drop, so the pooled Vecs (still held by the channel)
        // are alive while we unregister their pages.
        for ptr in &self.registered_buffers {
            unregister_host_memory_c(*ptr as *mut c_void);
        }
    }
}

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
        let witness = Pool::new(n_buffers, buffer_size_witness, cancelled.clone());
        let witness_compressor = Pool::new(n_buffers_compressor, buffer_size_witness_compressor, cancelled.clone());
        let trace = Pool::new(n_buffers, buffer_size_trace, cancelled.clone());
        let trace_compressor = Pool::new(n_buffers_compressor, buffer_size_trace_compressor, cancelled.clone());

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

    /// Unblock any thread parked in a pooled `take()`. Called on the abort path so
    /// a failed proof tears down cleanly instead of hanging on a buffer that will
    /// never be released.
    pub fn cancel(&self) {
        self.cancelled.store(true, Ordering::SeqCst);
    }

    pub fn reset(&self) -> ProofmanResult<()> {
        self.witness.reset()?;
        self.witness_compressor.reset()?;
        self.trace.reset()?;
        self.trace_compressor.reset()
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
}
