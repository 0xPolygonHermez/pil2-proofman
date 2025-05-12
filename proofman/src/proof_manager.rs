use std::sync::{Mutex, Condvar};
use std::sync::atomic::AtomicUsize;
use crossbeam_queue::ArrayQueue;

#[derive(Debug)]
pub struct ThreadInstanceInfo {
    pub airgroup_id: AtomicUsize,
    pub air_id: AtomicUsize,
    pub proof_type: AtomicUsize,
}

pub struct WitnessPendingCounter {
    counter: Mutex<usize>,
    cvar: Condvar,
}

impl WitnessPendingCounter {
    pub fn new() -> Self {
        Self {
            counter: Mutex::new(0),
            cvar: Condvar::new(),
        }
    }

    pub fn increment(&self) {
        let mut count = self.counter.lock().unwrap();
        *count += 1;
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
