use std::sync::{Arc, Mutex, Condvar};
use std::sync::atomic::{AtomicUsize, Ordering};
use crossbeam_queue::ArrayQueue;

pub struct ProofExecutionManager {
    thread_state: Arc<(Mutex<Vec<bool>>, Condvar)>,
    instance_info: Arc<Box<[(AtomicUsize, AtomicUsize, AtomicUsize)]>>,
}

impl ProofExecutionManager {
    pub fn new(max_concurrent_proofs: usize) -> Self {
        let thread_state = Arc::new((Mutex::new(vec![true; max_concurrent_proofs]), Condvar::new()));

        let instance_info = Arc::new(
            (0..max_concurrent_proofs)
                .map(|_| (AtomicUsize::new(0), AtomicUsize::new(0), AtomicUsize::new(0)))
                .collect::<Vec<_>>()
                .into_boxed_slice(),
        );

        Self { thread_state, instance_info }
    }

    pub fn claim_thread(&self) -> usize {
        let (lock, cvar) = &*self.thread_state;
        let mut threads = lock.lock().unwrap();

        loop {
            if let Some(index) = threads.iter().position(|&available| available) {
                threads[index] = false;
                return index;
            }
            threads = cvar.wait(threads).unwrap(); // Wait for a signal
        }
    }

    pub fn release_thread(&self, thread_id: usize) {
        let (lock, cvar) = &*self.thread_state;
        let mut threads = lock.lock().unwrap();
        threads[thread_id] = true;
        cvar.notify_one();
    }

    pub fn set_instance_info(&self, thread_id: usize, airgroup_id: usize, air_id: usize, proof_type: usize) {
        self.instance_info[thread_id].0.store(airgroup_id, Ordering::Release);
        self.instance_info[thread_id].1.store(air_id, Ordering::Release);
        self.instance_info[thread_id].2.store(proof_type, Ordering::Release);
    }

    pub fn get_instance_info(&self, thread_id: usize) -> (usize, usize, usize) {
        (
            self.instance_info[thread_id].0.load(Ordering::Acquire),
            self.instance_info[thread_id].1.load(Ordering::Acquire),
            self.instance_info[thread_id].2.load(Ordering::Acquire),
        )
    }
}

#[derive(Debug)]
pub struct WitnessComputationManager {
    pool_state: Arc<(Mutex<Vec<bool>>, Condvar)>,
    pending_witness: Arc<(Mutex<usize>, Condvar)>,
}

impl WitnessComputationManager {
    pub fn new(max_concurrent_pools: usize) -> Self {
        let pool_state = Arc::new((Mutex::new(vec![true; max_concurrent_pools]), Condvar::new()));
        let pending_witness = Arc::new((Mutex::new(0), Condvar::new()));
        Self { pool_state, pending_witness }
    }

    pub fn claim_thread(&self) -> usize {
        let (lock, cvar) = &*self.pool_state;
        let mut pools = lock.lock().unwrap();

        loop {
            if let Some(index) = pools.iter().position(|&available| available) {
                pools[index] = false;
                return index;
            }
            pools = cvar.wait(pools).unwrap();
        }
    }

    pub fn release_thread(&self, thread_id: usize) {
        let (lock, cvar) = &*self.pool_state;
        let mut pools = lock.lock().unwrap();
        pools[thread_id] = true;
        cvar.notify_one();
    }

    pub fn set_pending_witness(&self) {
        let (lock, cvar) = &*self.pending_witness;
        let mut pending = lock.lock().unwrap();
        *pending += 1;
        cvar.notify_all();
    }

    pub fn set_witness_completed(&self) {
        let (lock, cvar) = &*self.pending_witness;
        let mut pending = lock.lock().unwrap();
        *pending -= 1;
        cvar.notify_all();
    }

    pub fn wait_until_tables_ready(&self) {
        let (lock, cvar) = &*self.pending_witness;
        let mut pending = lock.lock().unwrap();
        while *pending > 0 {
            pending = cvar.wait(pending).unwrap();
        }
    }
}

#[derive(Debug)]
pub struct WitnessBuffer {
    queue: ArrayQueue<usize>,
    condvar: Condvar,
    mutex: Mutex<BufferState>,
}

#[derive(Debug)]
struct BufferState {
    closed: bool,
}

impl WitnessBuffer {
    pub fn new(cap: usize) -> Self {
        Self { queue: ArrayQueue::new(cap), condvar: Condvar::new(), mutex: Mutex::new(BufferState { closed: false }) }
    }

    pub fn push(&self, item: usize) -> bool {
        let mut state = self.mutex.lock().unwrap();
        while self.queue.len() >= self.queue.capacity() {
            if state.closed {
                return false;
            }
            state = self.condvar.wait(state).unwrap();
        }

        let push_result = self.queue.push(item);
        self.condvar.notify_all();
        push_result.is_ok()
    }

    pub fn pop(&self) -> Option<usize> {
        let mut state = self.mutex.lock().unwrap();

        loop {
            if let Some(item) = self.queue.pop() {
                self.condvar.notify_all();
                return Some(item);
            }

            if state.closed {
                return None;
            }

            state = self.condvar.wait(state).unwrap();
        }
    }

    pub fn close(&self) {
        let mut state = self.mutex.lock().unwrap();
        state.closed = true;
        self.condvar.notify_all();
    }

    pub fn wait_until_below_capacity(&self) -> bool {
        let mut state = self.mutex.lock().unwrap();
        while self.queue.len() >= self.queue.capacity() && !state.closed {
            state = self.condvar.wait(state).unwrap();
        }
        !state.closed
    }

    pub fn is_closed(&self) -> bool {
        self.mutex.lock().unwrap().closed
    }
}
