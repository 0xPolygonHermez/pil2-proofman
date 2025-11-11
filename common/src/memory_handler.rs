use crossbeam_channel::{bounded, Sender, Receiver};
use proofman_util::create_buffer_fast;
use std::sync::Arc;
use crossbeam_queue::SegQueue;
use crate::ProofCtx;
use fields::PrimeField64;
use crate::{ProofmanError, ProofmanResult};

pub struct MemoryHandler<F: PrimeField64 + Send + Sync + 'static> {
    pctx: Arc<ProofCtx<F>>,
    instance_ids_to_be_released: Arc<SegQueue<usize>>,
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

        while !self.instance_ids_to_be_released.is_empty() {
            self.instance_ids_to_be_released.pop();
        }

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
            if let Some(stored_instance_id) = self.instance_ids_to_be_released.pop() {
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

    pub fn to_be_released_buffer(&self, instance_id: usize) {
        self.instance_ids_to_be_released.push(instance_id);
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
