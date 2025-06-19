use crossbeam_channel::{bounded, Sender, Receiver};
use proofman_util::create_buffer_fast;

pub struct MemoryHandler<F: Send + Sync + 'static> {
    sender: Sender<Vec<F>>,
    receiver: Receiver<Vec<F>>,
}

impl<F: Send + Sync + 'static> MemoryHandler<F> {
    pub fn new(n_buffers: usize, buffer_size: usize) -> Self {
        let (tx_buffer_pool, rx_buffer_pool) = bounded(n_buffers);
        for _ in 0..n_buffers {
            tx_buffer_pool.send(create_buffer_fast(buffer_size)).unwrap();
        }

        Self { sender: tx_buffer_pool, receiver: rx_buffer_pool }
    }

    pub fn take_buffer(&self) -> Vec<F> {
        self.receiver.recv().expect("Failed to receive buffer")
    }

    pub fn release_buffer(&self, buffer: Vec<F>) {
        self.sender.send(buffer).expect("Failed to send buffer back to pool");
    }
}

pub trait BufferPool<F>: Send + Sync
where
    F: Send + Sync + 'static,
{
    fn take_buffer(&self) -> Vec<F>;
}

impl<F: Send + Sync + 'static> BufferPool<F> for MemoryHandler<F> {
    fn take_buffer(&self) -> Vec<F> {
        self.take_buffer()
    }
}
