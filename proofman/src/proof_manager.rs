use std::sync::atomic::{AtomicUsize, AtomicBool, Ordering};
use std::sync::Arc;

pub struct ProofExecutionManager {
    pub max_concurrent_proofs: usize,
    thread_available: Arc<Box<[AtomicBool]>>,
    instance_info: Arc<Box<[(AtomicUsize, AtomicUsize)]>>,
}

impl ProofExecutionManager {
    pub fn new(max_concurrent_proofs: usize) -> Self {
        let thread_available =
            Arc::new((0..max_concurrent_proofs).map(|_| AtomicBool::new(true)).collect::<Vec<_>>().into_boxed_slice());

        let instance_info = Arc::new(
            (0..max_concurrent_proofs)
                .map(|_| (AtomicUsize::new(0), AtomicUsize::new(0)))
                .collect::<Vec<_>>()
                .into_boxed_slice(),
        );
        Self { max_concurrent_proofs, thread_available, instance_info }
    }

    pub fn try_claim_thread(&self) -> Option<usize> {
        (0..self.max_concurrent_proofs).find(|&thread_id| {
            self.thread_available[thread_id].compare_exchange(true, false, Ordering::AcqRel, Ordering::Acquire).is_ok()
        })
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

    pub fn proof_completed(&self, thread_id: usize) {
        self.thread_available[thread_id].store(true, Ordering::Release);
    }
}

#[derive(Debug)]
pub struct WitnessComputationManager {
    pub max_concurrent_pools: usize,
    pending_witness: Arc<AtomicUsize>,
    pub pools_available: Arc<Box<[AtomicBool]>>,
}

impl WitnessComputationManager {
    pub fn new(max_concurrent_pools: usize) -> Self {
        let pools_available =
            Arc::new((0..max_concurrent_pools).map(|_| AtomicBool::new(true)).collect::<Vec<_>>().into_boxed_slice());

        Self { max_concurrent_pools, pools_available, pending_witness: Arc::new(AtomicUsize::new(0)) }
    }

    pub fn try_claim_thread(&self) -> Option<usize> {
        (0..self.max_concurrent_pools).find(|&thread_id| {
            self.pools_available[thread_id].compare_exchange(true, false, Ordering::AcqRel, Ordering::Acquire).is_ok()
        })
    }

    pub fn set_thread_available(&self, thread_id: usize) {
        self.pools_available[thread_id].store(true, Ordering::SeqCst);
    }

    pub fn set_pending_witness(&self) -> usize {
        self.pending_witness.fetch_add(1, Ordering::SeqCst)
    }

    pub fn set_witness_completed(&self) {
        self.pending_witness.fetch_sub(1, Ordering::SeqCst);
    }

    pub fn are_tables_ready(&self) -> bool {
        self.pending_witness.load(Ordering::Acquire) == 0
    }
}
