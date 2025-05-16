use std::sync::{Arc, Mutex, Condvar};
use crossbeam_queue::ArrayQueue;
use proofman_starks_lib_c::{register_proof_done_callback_c};
use proofman_common::{ProofCtx, ProofType};
use p3_field::Field;
use std::sync::atomic::{Ordering, AtomicUsize};

pub struct Counter {
    counter: Mutex<usize>,
    cvar: Condvar,
}

impl Default for Counter {
    fn default() -> Self {
        Self::new()
    }
}

impl Counter {
    pub fn new() -> Self {
        Self { counter: Mutex::new(0), cvar: Condvar::new() }
    }

    pub fn increment(&self) -> usize {
        let mut count = self.counter.lock().unwrap();
        *count += 1;
        *count
    }

    pub fn decrement(&self) {
        let mut count = self.counter.lock().unwrap();
        *count -= 1;
        if *count == 0 {
            self.cvar.notify_all();
        }
    }

    pub fn wait_for_tables_ready(&self) {
        let mut count = self.counter.lock().unwrap();
        while *count > 0 {
            count = self.cvar.wait(count).unwrap();
        }
    }
}

#[derive(Debug)]
pub struct WitnessBuffer {
    queue: ArrayQueue<(usize, usize)>,
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

    pub fn push(&self, item: (usize, usize)) -> bool {
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

    pub fn pop(&self) -> Option<(usize, usize)> {
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

pub fn proofs_done_listener<F: Field>(
    pctx: Arc<ProofCtx<F>>,
    witness_recursive_tx: crossbeam_channel::Sender<(usize, usize, usize)>,
    proofs_counter: Arc<AtomicUsize>,
    aggregation: bool,
) -> std::thread::JoinHandle<()> {
    let (tx, rx) = crossbeam_channel::unbounded::<(u64, String)>();

    register_proof_done_callback_c(tx);

    std::thread::spawn(move || {
        let pctx_clone = pctx.clone();
        while let Ok((instance_id, proof_type)) = rx.recv() {
            let p: ProofType = proof_type.parse().unwrap();
            if aggregation {
                let new_proof_type = if p == ProofType::Basic {
                    let instances = pctx_clone.dctx_get_instances();
                    let (airgroup_id, air_id, _) = instances[instance_id as usize];
                    if pctx_clone.global_info.get_air_has_compressor(airgroup_id, air_id) {
                        ProofType::Compressor as usize
                    } else {
                        ProofType::Recursive1 as usize
                    }
                } else if p == ProofType::Compressor {
                    ProofType::Recursive1 as usize
                } else {
                    ProofType::Recursive2 as usize
                };
                witness_recursive_tx.send((instance_id as usize, p as usize, new_proof_type)).unwrap();
            } else {
                proofs_counter.fetch_sub(1, Ordering::SeqCst);
            }
        }
    })
}
