use crossbeam_channel::{bounded, Sender, Receiver};
use std::sync::Arc;
use std::sync::atomic::AtomicUsize;
use crossbeam_queue::SegQueue;
use crate::ProofCtx;
use fields::PrimeField64;
use crate::{ProofmanError, ProofmanResult};

/// Single fixed-size buffer pool over a bounded channel. Internal helper used
/// by `MemoryHandlerRecursive`. `take()` is a blocking channel recv — wakes
/// instantly when a buffer is released, no polling.
struct Pool<F: PrimeField64 + Send + Sync + 'static> {
    sender: Sender<Vec<F>>,
    receiver: Receiver<Vec<F>>,
    n_buffers: usize,
    buffer_size: usize,
}

impl<F: PrimeField64 + Send + Sync + 'static> Pool<F> {
    fn new(n_buffers: usize, buffer_size: usize) -> Self {
        let (sender, receiver) = bounded(n_buffers);
        for _ in 0..n_buffers {
            sender.send(vec![F::ZERO; buffer_size]).unwrap();
        }
        Self { sender, receiver, n_buffers, buffer_size }
    }

    fn take(&self) -> Vec<F> {
        self.receiver.recv().expect("Pool channel closed")
    }

    fn release(&self, buffer: Vec<F>) -> ProofmanResult<()> {
        if buffer.len() != self.buffer_size {
            return Err(ProofmanError::ProofmanError(format!(
                "Pool::release: wrong size {} (expected {})",
                buffer.len(),
                self.buffer_size
            )));
        }
        self.sender.send(buffer).expect("Pool channel closed");
        Ok(())
    }

    fn reset(&self) -> ProofmanResult<()> {
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
        while valid_buffers.len() < self.n_buffers {
            tracing::warn!("Pool::reset: only {} valid buffers; allocating a replacement", valid_buffers.len());
            valid_buffers.push(vec![F::ZERO; self.buffer_size]);
        }
        for buf in valid_buffers {
            self.sender.send(buf).expect("Pool channel closed");
        }
        Ok(())
    }

    fn total_bytes(&self) -> usize {
        self.n_buffers * self.buffer_size * std::mem::size_of::<F>()
    }
}

pub struct MemoryHandler<F: PrimeField64 + Send + Sync + 'static> {
    pctx: Arc<ProofCtx<F>>,
    instance_ids_to_be_released: Arc<SegQueue<(usize, bool)>>,
    sender: Sender<Vec<F>>,
    receiver: Receiver<Vec<F>>,
    n_buffers: usize,
    buffer_size: usize,
}

impl<F: PrimeField64 + Send + Sync + 'static> MemoryHandler<F> {
    pub fn new(pctx: Arc<ProofCtx<F>>, n_buffers: usize, buffer_size: usize) -> Self {
        let (tx_buffer_pool, rx_buffer_pool) = bounded(n_buffers);
        let instance_ids_to_be_released = Arc::new(SegQueue::new());
        for _ in 0..n_buffers {
            tx_buffer_pool.send(vec![F::ZERO; buffer_size]).unwrap();
        }

        let total_memory = n_buffers * buffer_size * std::mem::size_of::<F>();
        tracing::info!("MemoryHandler::Total memory for basic traces: {}", crate::format_bytes(total_memory as f64));

        Self {
            pctx,
            sender: tx_buffer_pool,
            receiver: rx_buffer_pool,
            instance_ids_to_be_released,
            n_buffers,
            buffer_size,
        }
    }

    pub fn reset(&self) -> ProofmanResult<()> {
        self.empty_queue_to_be_released();

        let mut current_buffers = Vec::new();
        while let Ok(buffer) = self.receiver.try_recv() {
            current_buffers.push(buffer);
        }

        let mut valid_buffers: Vec<Vec<F>> = Vec::with_capacity(self.n_buffers);
        for buf in current_buffers.into_iter() {
            if buf.len() == self.buffer_size {
                valid_buffers.push(buf);
            } else {
                return Err(ProofmanError::ProofmanError(format!(
                    "MemoryHandler::Found buffer with unexpected size {} (expected {}), replacing it.",
                    buf.len(),
                    self.buffer_size
                )));
            }
        }

        while valid_buffers.len() < self.n_buffers {
            tracing::warn!(
                "MemoryHandler::Not enough valid buffers (found {}), creating a new one.",
                valid_buffers.len()
            );
            valid_buffers.push(vec![F::ZERO; self.buffer_size]);
        }

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
        let trace = Pool::new(n_buffers, buffer_size_trace);
        let trace_compressor = Pool::new(n_buffers_compressor, buffer_size_trace_compressor);

        let total = trace.total_bytes() + trace_compressor.total_bytes();
        tracing::info!(
            "MemoryHandlerRecursive::Total memory for recursive traces: {}",
            crate::format_bytes(total as f64)
        );

        let signal_values =
            signal_pool.map(|(large_cap, small_cap, n_small)| SignalValuesPool::new(large_cap, small_cap, n_small));

        let witness_threads = witness_threads.max(1);
        tracing::info!("MemoryHandlerRecursive::circom solve threads per recursive witness: {}", witness_threads);

        Self { trace, trace_compressor, signal_values, witness_threads }
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
        self.trace_compressor.reset()
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
pub struct SignalValuesPool {
    large: Sender<Vec<u64>>,
    large_rx: Receiver<Vec<u64>>,
    small: Sender<Vec<u64>>,
    small_rx: Receiver<Vec<u64>>,
    large_cap: usize,
    small_cap: usize,
    /// Count of big proofs (needed > small_cap) currently blocked on the large
    /// buffer. While > 0, small proofs won't steal it — big circuits get priority.
    large_waiters: std::sync::atomic::AtomicUsize,
}

impl SignalValuesPool {
    /// `large_cap` / `small_cap` are in u64 elements (= get_total_signal_no()).
    pub fn new(large_cap: usize, small_cap: usize, n_small: usize) -> Self {
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
        Self { large, large_rx, small, small_rx, large_cap, small_cap, large_waiters: AtomicUsize::new(0) }
    }

    /// Take a buffer with capacity >= `needed`. Big proofs (needed > small_cap)
    /// block on the large buffer with priority; small proofs prefer a small buffer
    /// and only borrow the large one when no big proof is waiting for it.
    pub fn take(&self, needed: usize) -> Vec<u64> {
        use std::sync::atomic::Ordering;
        if needed > self.small_cap {
            // Only the large buffer fits. Register as a waiter so small proofs yield
            // it to us, then block until it's free.
            self.large_waiters.fetch_add(1, Ordering::SeqCst);
            let buf = self.large_rx.recv().expect("SignalValuesPool large channel closed");
            self.large_waiters.fetch_sub(1, Ordering::SeqCst);
            buf
        } else if let Ok(buf) = self.small_rx.try_recv() {
            buf
        } else if self.large_waiters.load(Ordering::SeqCst) == 0 {
            // Smalls exhausted, no big proof queued: borrow large non-blockingly;
            // if it's busy too, wait for a small rather than block a big proof.
            match self.large_rx.try_recv() {
                Ok(buf) => buf,
                Err(_) => self.small_rx.recv().expect("SignalValuesPool small channel closed"),
            }
        } else {
            // A big proof is queued for the large buffer — don't contend; wait for a small.
            self.small_rx.recv().expect("SignalValuesPool small channel closed")
        }
    }

    /// Return a buffer; routed back to its class by capacity.
    pub fn release(&self, buffer: Vec<u64>) {
        if buffer.len() >= self.large_cap {
            self.large.send(buffer).expect("SignalValuesPool large channel closed");
        } else {
            self.small.send(buffer).expect("SignalValuesPool small channel closed");
        }
    }
}
