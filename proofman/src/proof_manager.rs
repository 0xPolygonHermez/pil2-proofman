use std::sync::{Arc, Mutex, Condvar};
use proofman_starks_lib_c::{register_proof_done_callback_c};

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

    pub fn wait_until_and_check_streams<F: FnMut()>(&self, mut check_streams: F, threshold: usize) {
        let mut count = self.counter.lock().unwrap();
        while *count > threshold {
            check_streams();

            let (c, _) = self.cvar.wait_timeout(count, std::time::Duration::from_micros(100)).unwrap();
            count = c;
        }
    }

    pub fn wait_until_zero(&self) {
        let mut count = self.counter.lock().unwrap();
        while *count > 0 {
            count = self.cvar.wait(count).unwrap();
        }
    }

    pub fn wait_until_zero_and_check_streams<F: FnMut()>(&self, mut check_streams: F) {
        let mut count = self.counter.lock().unwrap();
        while *count > 0 {
            check_streams();

            let (c, _) = self.cvar.wait_timeout(count, std::time::Duration::from_micros(100)).unwrap();
            count = c;
        }
    }

    pub fn get_count(&self) -> usize {
        *self.counter.lock().unwrap()
    }
}

pub fn contributions_done_listener(contributions_counter: Arc<Counter>) -> std::thread::JoinHandle<()> {
    let (tx, rx) = crossbeam_channel::unbounded::<(u64, String)>();

    register_proof_done_callback_c(tx);

    std::thread::spawn(move || {
        while rx.recv().is_ok() {
            contributions_counter.decrement();
        }
    })
}
