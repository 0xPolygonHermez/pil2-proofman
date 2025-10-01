use crossbeam_channel::{bounded, Sender, Receiver};
use proofman_util::create_buffer_fast;
use std::sync::Arc;
use crossbeam_queue::SegQueue;
use crate::ProofCtx;
use fields::PrimeField64;

pub struct MemoryHandler<F: PrimeField64 + Send + Sync + 'static> {
    pctx: Arc<ProofCtx<F>>,
    instance_ids_to_be_released: Arc<SegQueue<usize>>,
    sender: Sender<Vec<F>>,
    receiver: Receiver<Vec<F>>,
}

impl<F: PrimeField64 + Send + Sync + 'static> MemoryHandler<F> {
    pub fn new(pctx: Arc<ProofCtx<F>>, n_buffers: usize, buffer_size: usize) -> Self {
        let (tx_buffer_pool, rx_buffer_pool) = bounded(n_buffers);
        let instance_ids_to_be_released = Arc::new(SegQueue::new());
        for _ in 0..n_buffers {
            tx_buffer_pool.send(create_buffer_fast(buffer_size)).unwrap();
        }

        Self { pctx, sender: tx_buffer_pool, receiver: rx_buffer_pool, instance_ids_to_be_released }
    }

    pub fn take_buffer(&self) -> Vec<F> {
        loop {
            if let Ok(buffer) = self.receiver.try_recv() {
                return buffer;
            }
            if let Some(stored_instance_id) = self.instance_ids_to_be_released.pop() {
                let (_, witness_buffer) = self.pctx.free_instance_traces(stored_instance_id);
                return witness_buffer;
            }
            std::thread::sleep(std::time::Duration::from_micros(10));
        }
    }

    pub fn release_buffer(&self, buffer: Vec<F>) {
        self.sender.send(buffer).expect("Failed to send buffer back to pool");
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
