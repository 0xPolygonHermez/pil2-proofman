use crossbeam_channel::{bounded, Sender, Receiver};
use std::sync::Arc;
use crossbeam_queue::SegQueue;
use crate::ProofCtx;
use proofman_fields::PrimeField64;
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
    witness: Pool<F>,
    witness_compressor: Pool<F>,
    trace: Pool<F>,
    trace_compressor: Pool<F>,
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
        let witness = Pool::new(n_buffers, buffer_size_witness);
        let witness_compressor = Pool::new(n_buffers_compressor, buffer_size_witness_compressor);
        let trace = Pool::new(n_buffers, buffer_size_trace);
        let trace_compressor = Pool::new(n_buffers_compressor, buffer_size_trace_compressor);

        let total = witness.total_bytes()
            + witness_compressor.total_bytes()
            + trace.total_bytes()
            + trace_compressor.total_bytes();
        tracing::info!(
            "MemoryHandlerRecursive::Total memory for recursive traces: {}",
            crate::format_bytes(total as f64)
        );

        Self { witness, witness_compressor, trace, trace_compressor }
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
