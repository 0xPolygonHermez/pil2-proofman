use crossbeam_channel::{bounded, Sender, Receiver};
use proofman_util::create_buffer_fast;
use std::sync::Arc;
use crossbeam_queue::SegQueue;
use crate::ProofCtx;
use fields::PrimeField64;
use crate::{ProofmanError, ProofmanResult};

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
            tx_buffer_pool.send(create_buffer_fast(buffer_size)).unwrap();
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
            valid_buffers.push(create_buffer_fast(self.buffer_size));
        }

        for buf in valid_buffers.into_iter() {
            self.sender.send(buf).unwrap();
        }

        Ok(())
    }

    pub fn take_buffer(&self) -> Vec<F> {
        loop {
            if let Ok(buffer) = self.receiver.try_recv() {
                return buffer;
            }
            if let Some((stored_instance_id, remove_from_calculated)) = self.instance_ids_to_be_released.pop() {
                if remove_from_calculated {
                    self.pctx.dctx_reset_instance_calculated(stored_instance_id);
                }
                let (is_shared_buffer, witness_buffer) = self.pctx.free_instance_traces(stored_instance_id);
                if is_shared_buffer {
                    return witness_buffer;
                }
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
    sender_witness: Sender<Vec<F>>,
    sender_witness_compressor: Sender<Vec<F>>,
    sender_trace: Sender<Vec<F>>,
    sender_trace_compressor: Sender<Vec<F>>,
    receiver_witness: Receiver<Vec<F>>,
    receiver_witness_compressor: Receiver<Vec<F>>,
    receiver_trace: Receiver<Vec<F>>,
    receiver_trace_compressor: Receiver<Vec<F>>,
    n_buffers_compressor: usize,
    n_buffers: usize,
    buffer_size_witness_compressor: usize,
    buffer_size_witness: usize,
    buffer_size_trace: usize,
    buffer_size_trace_compressor: usize,
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
        let (tx_witness, rx_witness) = bounded(n_buffers);
        let (tx_witness_compressor, rx_witness_compressor) = bounded(n_buffers_compressor);
        let (tx_trace, rx_trace) = bounded(n_buffers);
        let (tx_trace_compressor, rx_trace_compressor) = bounded(n_buffers_compressor);

        let total_witness_memory = n_buffers * buffer_size_witness * std::mem::size_of::<F>();
        let total_witness_compressor_memory =
            n_buffers_compressor * buffer_size_witness_compressor * std::mem::size_of::<F>();
        let total_trace_memory = n_buffers * buffer_size_trace * std::mem::size_of::<F>();
        let total_trace_compressor_memory =
            n_buffers_compressor * buffer_size_trace_compressor * std::mem::size_of::<F>();
        let total_memory =
            total_witness_memory + total_witness_compressor_memory + total_trace_memory + total_trace_compressor_memory;
        tracing::info!(
            "MemoryHandlerRecursive::Total memory for recursive traces: {}",
            crate::format_bytes(total_memory as f64)
        );

        for _ in 0..n_buffers {
            tx_witness.send(create_buffer_fast(buffer_size_witness)).unwrap();
            tx_trace.send(create_buffer_fast(buffer_size_trace)).unwrap();
        }

        for _ in 0..n_buffers_compressor {
            tx_witness_compressor.send(create_buffer_fast(buffer_size_witness_compressor)).unwrap();
            tx_trace_compressor.send(create_buffer_fast(buffer_size_trace_compressor)).unwrap();
        }

        Self {
            sender_witness: tx_witness,
            receiver_witness: rx_witness,
            sender_witness_compressor: tx_witness_compressor,
            receiver_witness_compressor: rx_witness_compressor,
            sender_trace: tx_trace,
            receiver_trace: rx_trace,
            sender_trace_compressor: tx_trace_compressor,
            receiver_trace_compressor: rx_trace_compressor,
            n_buffers,
            n_buffers_compressor,
            buffer_size_witness,
            buffer_size_witness_compressor,
            buffer_size_trace,
            buffer_size_trace_compressor,
        }
    }

    pub fn reset(&self) -> ProofmanResult<()> {
        // Reset witness buffers
        let mut current_buffers = Vec::new();
        while let Ok(buffer) = self.receiver_witness.try_recv() {
            current_buffers.push(buffer);
        }

        let mut valid_buffers: Vec<Vec<F>> = Vec::with_capacity(self.n_buffers);
        for buf in current_buffers.into_iter() {
            if buf.len() == self.buffer_size_witness {
                valid_buffers.push(buf);
            } else {
                return Err(ProofmanError::ProofmanError(format!(
                    "MemoryHandlerRecursive::Found witness buffer with unexpected size {} (expected {}).",
                    buf.len(),
                    self.buffer_size_witness
                )));
            }
        }

        while valid_buffers.len() < self.n_buffers {
            tracing::warn!(
                "MemoryHandlerRecursive::Not enough valid witness buffers (found {}), creating a new one.",
                valid_buffers.len()
            );
            valid_buffers.push(create_buffer_fast(self.buffer_size_witness));
        }

        for buf in valid_buffers.into_iter() {
            self.sender_witness.send(buf).unwrap();
        }

        // Reset witness compressor buffers
        let mut current_buffers = Vec::new();
        while let Ok(buffer) = self.receiver_witness_compressor.try_recv() {
            current_buffers.push(buffer);
        }

        let mut valid_buffers: Vec<Vec<F>> = Vec::with_capacity(self.n_buffers_compressor);
        for buf in current_buffers.into_iter() {
            if buf.len() == self.buffer_size_witness_compressor {
                valid_buffers.push(buf);
            } else {
                return Err(ProofmanError::ProofmanError(format!(
                    "MemoryHandlerRecursive::Found witness_compressor buffer with unexpected size {} (expected {}).",
                    buf.len(),
                    self.buffer_size_witness_compressor
                )));
            }
        }

        while valid_buffers.len() < self.n_buffers_compressor {
            tracing::warn!(
                "MemoryHandlerRecursive::Not enough valid witness_compressor buffers (found {}), creating a new one.",
                valid_buffers.len()
            );
            valid_buffers.push(create_buffer_fast(self.buffer_size_witness_compressor));
        }

        for buf in valid_buffers.into_iter() {
            self.sender_witness_compressor.send(buf).unwrap();
        }

        // Reset trace buffers
        let mut current_buffers = Vec::new();
        while let Ok(buffer) = self.receiver_trace.try_recv() {
            current_buffers.push(buffer);
        }

        let mut valid_buffers: Vec<Vec<F>> = Vec::with_capacity(self.n_buffers);
        for buf in current_buffers.into_iter() {
            if buf.len() == self.buffer_size_trace {
                valid_buffers.push(buf);
            } else {
                return Err(ProofmanError::ProofmanError(format!(
                    "MemoryHandlerRecursive::Found trace buffer with unexpected size {} (expected {}).",
                    buf.len(),
                    self.buffer_size_trace
                )));
            }
        }

        while valid_buffers.len() < self.n_buffers {
            tracing::warn!(
                "MemoryHandlerRecursive::Not enough valid trace buffers (found {}), creating a new one.",
                valid_buffers.len()
            );
            valid_buffers.push(create_buffer_fast(self.buffer_size_trace));
        }

        for buf in valid_buffers.into_iter() {
            self.sender_trace.send(buf).unwrap();
        }

        // Reset trace compressor buffers
        let mut current_buffers = Vec::new();
        while let Ok(buffer) = self.receiver_trace_compressor.try_recv() {
            current_buffers.push(buffer);
        }

        let mut valid_buffers: Vec<Vec<F>> = Vec::with_capacity(self.n_buffers_compressor);
        for buf in current_buffers.into_iter() {
            if buf.len() == self.buffer_size_trace_compressor {
                valid_buffers.push(buf);
            } else {
                return Err(ProofmanError::ProofmanError(format!(
                    "MemoryHandlerRecursive::Found trace_compressor buffer with unexpected size {} (expected {}).",
                    buf.len(),
                    self.buffer_size_trace_compressor
                )));
            }
        }

        while valid_buffers.len() < self.n_buffers_compressor {
            tracing::warn!(
                "MemoryHandlerRecursive::Not enough valid trace_compressor buffers (found {}), creating a new one.",
                valid_buffers.len()
            );
            valid_buffers.push(create_buffer_fast(self.buffer_size_trace_compressor));
        }

        for buf in valid_buffers.into_iter() {
            self.sender_trace_compressor.send(buf).unwrap();
        }

        Ok(())
    }

    pub fn take_buffer_witness(&self) -> Vec<F> {
        loop {
            if let Ok(buffer) = self.receiver_witness.try_recv() {
                return buffer;
            }
            std::thread::sleep(std::time::Duration::from_micros(10));
        }
    }

    pub fn release_buffer_witness(&self, buffer: Vec<F>) -> ProofmanResult<()> {
        if buffer.len() != self.buffer_size_witness {
            return Err(ProofmanError::ProofmanError(format!(
                "MemoryHandlerRecursive::Trying to release witness buffer with unexpected size {} (expected {}).",
                buffer.len(),
                self.buffer_size_witness
            )));
        }
        self.sender_witness.send(buffer).expect("Failed to send witness buffer back to pool");
        Ok(())
    }

    pub fn take_buffer_witness_compressor(&self) -> Vec<F> {
        loop {
            if let Ok(buffer) = self.receiver_witness_compressor.try_recv() {
                return buffer;
            }
            std::thread::sleep(std::time::Duration::from_micros(10));
        }
    }

    pub fn release_buffer_witness_compressor(&self, buffer: Vec<F>) -> ProofmanResult<()> {
        if buffer.len() != self.buffer_size_witness_compressor {
            return Err(ProofmanError::ProofmanError(format!(
                "MemoryHandlerRecursive::Trying to release witness_compressor buffer with unexpected size {} (expected {}).",
                buffer.len(),
                self.buffer_size_witness_compressor
            )));
        }
        self.sender_witness_compressor.send(buffer).expect("Failed to send witness_compressor buffer back to pool");
        Ok(())
    }

    pub fn take_buffer_trace(&self) -> Vec<F> {
        loop {
            if let Ok(buffer) = self.receiver_trace.try_recv() {
                return buffer;
            }
            std::thread::sleep(std::time::Duration::from_micros(10));
        }
    }

    pub fn release_buffer_trace(&self, buffer: Vec<F>) -> ProofmanResult<()> {
        if buffer.len() != self.buffer_size_trace {
            return Err(ProofmanError::ProofmanError(format!(
                "MemoryHandlerRecursive::Trying to release trace buffer with unexpected size {} (expected {}).",
                buffer.len(),
                self.buffer_size_trace
            )));
        }
        self.sender_trace.send(buffer).expect("Failed to send trace buffer back to pool");
        Ok(())
    }

    pub fn take_buffer_trace_compressor(&self) -> Vec<F> {
        loop {
            if let Ok(buffer) = self.receiver_trace_compressor.try_recv() {
                return buffer;
            }
            std::thread::sleep(std::time::Duration::from_micros(10));
        }
    }

    pub fn release_buffer_trace_compressor(&self, buffer: Vec<F>) -> ProofmanResult<()> {
        if buffer.len() != self.buffer_size_trace_compressor {
            return Err(ProofmanError::ProofmanError(format!(
                "MemoryHandlerRecursive::Trying to release trace_compressor buffer with unexpected size {} (expected {}).",
                buffer.len(),
                self.buffer_size_trace_compressor
            )));
        }
        self.sender_trace_compressor.send(buffer).expect("Failed to send trace_compressor buffer back to pool");
        Ok(())
    }
}
