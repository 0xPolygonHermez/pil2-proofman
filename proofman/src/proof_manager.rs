use std::sync::{Arc, Mutex, Condvar};
use std::sync::atomic::{AtomicUsize, Ordering};

pub struct ProofExecutionManager {
    thread_state: Arc<(Mutex<Vec<bool>>, Condvar)>,
    instance_info: Arc<Box<[(AtomicUsize, AtomicUsize)]>>,
}

impl ProofExecutionManager {
    pub fn new(max_concurrent_proofs: usize) -> Self {
        let thread_state = Arc::new((Mutex::new(vec![true; max_concurrent_proofs]), Condvar::new()));

        let instance_info = Arc::new(
            (0..max_concurrent_proofs)
                .map(|_| (AtomicUsize::new(0), AtomicUsize::new(0)))
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

    pub fn set_instance_info(&self, thread_id: usize, instance_id: usize, instance_size: usize) {
        self.instance_info[thread_id].0.store(instance_id, Ordering::Release);
        self.instance_info[thread_id].1.store(instance_size, Ordering::Release);
    }

    pub fn get_instance_info(&self, thread_id: usize) -> (usize, usize) {
        (
            self.instance_info[thread_id].0.load(Ordering::Acquire),
            self.instance_info[thread_id].1.load(Ordering::Acquire),
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
