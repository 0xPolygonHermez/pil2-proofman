use std::sync::atomic::{AtomicUsize, AtomicBool, Ordering};
use std::sync::Arc;

pub struct ProofExecutionManager {
    max_concurrent_proofs: usize,
    active_threads: Arc<AtomicUsize>,
    proofs_ready: Arc<AtomicUsize>,
    thread_available: Arc<Box<[AtomicBool]>>,
}

impl ProofExecutionManager {
    pub fn new(max_concurrent_proofs: usize) -> Self {
        let thread_available =
            Arc::new((0..max_concurrent_proofs).map(|_| AtomicBool::new(true)).collect::<Vec<_>>().into_boxed_slice());

        Self {
            max_concurrent_proofs,
            thread_available,
            active_threads: Arc::new(AtomicUsize::new(0)),
            proofs_ready: Arc::new(AtomicUsize::new(0)),
        }
    }

    pub fn try_claim_thread(&self, thread_id: usize) -> bool {
        if thread_id >= self.max_concurrent_proofs {
            return false;
        }
        if self.active_threads.load(Ordering::SeqCst) >= self.max_concurrent_proofs {
            return false;
        }
        if self.thread_available[thread_id].compare_exchange(true, false, Ordering::AcqRel, Ordering::Acquire).is_ok() {
            self.active_threads.fetch_add(1, Ordering::SeqCst);
            true
        } else {
            false
        }
    }

    pub fn get_n_active_threads(&self) -> usize {
        self.active_threads.load(Ordering::Acquire)
    }

    pub fn set_new_proof_ready(&self) {
        self.proofs_ready.fetch_add(1, Ordering::Release);
    }

    pub fn get_proofs_ready(&self) -> usize {
        self.proofs_ready.load(Ordering::Acquire)
    }

    pub fn proof_completed(&self, thread_id: usize) {
        self.thread_available[thread_id].store(true, Ordering::Release);
        self.active_threads.fetch_sub(1, Ordering::SeqCst);
        self.proofs_ready.fetch_sub(1, Ordering::SeqCst);
    }
}
