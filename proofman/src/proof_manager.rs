use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Condvar, Mutex, RwLock,
};
use std::time::{Duration, Instant};

use crate::CancellationInfo;

pub const WAIT_TIMEOUT_SECONDS: u64 = 600;
pub const WAIT_STATUS_INTERVAL_SECONDS: u64 = 60;

struct WaitTimeoutTracker {
    start_time: Instant,
    last_status_print: Instant,
    last_seen_value: Option<usize>,
}

impl WaitTimeoutTracker {
    fn new() -> Self {
        let now = Instant::now();
        Self { start_time: now, last_status_print: now, last_seen_value: None }
    }

    /// Checks timeout and prints status. Returns true if should continue waiting, false if timeout.
    fn check_and_log(
        &mut self,
        current: usize,
        expected: usize,
        message_type: &str,
        cancellation_info: &RwLock<CancellationInfo>,
    ) -> bool {
        // Reset the 60s timer if counter has changed
        if self.last_seen_value != Some(current) {
            self.last_status_print = Instant::now();
            self.last_seen_value = Some(current);
        }

        let elapsed = self.start_time.elapsed();

        if elapsed.as_secs() >= WAIT_STATUS_INTERVAL_SECONDS
            && self.last_status_print.elapsed().as_secs() >= WAIT_STATUS_INTERVAL_SECONDS
        {
            tracing::warn!(
                "Counter still waiting {} after {}s - current: {}, expected: {}",
                message_type,
                elapsed.as_secs(),
                current,
                expected
            );
            self.last_status_print = Instant::now();
        }

        // Timeout after 10 minutes
        if elapsed.as_secs() >= 600 {
            tracing::error!(
                "Counter timeout after 10 minutes {} - current: {}, expected: {}. Cancelling.",
                message_type,
                current,
                expected
            );
            cancellation_info.write().unwrap().token.cancel();
            return false;
        }

        true
    }
}

pub struct Counter {
    counter: AtomicUsize,
    wait_lock: Mutex<()>,
    cvar: Condvar,
    threshold: usize,
}

impl Default for Counter {
    fn default() -> Self {
        Self::new()
    }
}

impl Counter {
    pub fn new() -> Self {
        Self { counter: AtomicUsize::new(0), wait_lock: Mutex::new(()), cvar: Condvar::new(), threshold: 0 }
    }

    pub fn new_with_threshold(threshold: usize) -> Self {
        Self { counter: AtomicUsize::new(0), wait_lock: Mutex::new(()), cvar: Condvar::new(), threshold }
    }

    #[inline(always)]
    pub fn increment(&self) -> usize {
        let new_val = self.counter.fetch_add(1, Ordering::Relaxed) + 1;

        if new_val >= self.threshold {
            let _guard = self.wait_lock.lock().unwrap();
            self.cvar.notify_all();
        }

        new_val
    }

    #[inline(always)]
    pub fn decrement(&self) -> usize {
        let new_val = self.counter.fetch_sub(1, Ordering::Release) - 1;

        if new_val == 0 {
            let _guard = self.wait_lock.lock().unwrap();
            self.cvar.notify_all();
        }

        new_val
    }

    pub fn wait_until_threshold_and_check_streams<F: FnMut()>(
        &self,
        mut check_streams: F,
        cancellation_info: &RwLock<CancellationInfo>,
    ) {
        let mut guard = self.wait_lock.lock().unwrap();
        let mut tracker = WaitTimeoutTracker::new();

        loop {
            if cancellation_info.read().unwrap().token.is_cancelled() {
                break;
            }

            let current = self.counter.load(Ordering::Acquire);
            if current >= self.threshold {
                break;
            }

            if !tracker.check_and_log(current, self.threshold, "for threshold", cancellation_info) {
                break;
            }

            check_streams();
            let (g, _) = self.cvar.wait_timeout(guard, Duration::from_micros(100)).unwrap();
            guard = g;
        }
    }

    pub fn wait_until_threshold(&self, cancellation_info: &RwLock<CancellationInfo>) {
        let mut guard = self.wait_lock.lock().unwrap();
        let mut tracker = WaitTimeoutTracker::new();

        loop {
            if cancellation_info.read().unwrap().token.is_cancelled() {
                break;
            }

            let current = self.counter.load(Ordering::Acquire);
            if current >= self.threshold {
                break;
            }

            if !tracker.check_and_log(current, self.threshold, "for threshold", cancellation_info) {
                break;
            }

            let (g, _) = self.cvar.wait_timeout(guard, Duration::from_millis(100)).unwrap();
            guard = g;
        }
    }

    pub fn wait_until_value_and_check_streams<F: FnMut()>(
        &self,
        value: usize,
        mut check_streams: F,
        cancellation_info: &RwLock<CancellationInfo>,
    ) {
        let mut guard = self.wait_lock.lock().unwrap();
        let mut tracker = WaitTimeoutTracker::new();

        loop {
            if cancellation_info.read().unwrap().token.is_cancelled() {
                break;
            }

            let current = self.counter.load(Ordering::Acquire);
            if current >= value {
                break;
            }

            if !tracker.check_and_log(current, value, "for value", cancellation_info) {
                break;
            }

            check_streams();
            let (g, _) = self.cvar.wait_timeout(guard, Duration::from_micros(100)).unwrap();
            guard = g;
        }
    }

    pub fn reset(&self) {
        self.counter.store(0, Ordering::Release);
    }

    pub fn wait_until_zero(&self, cancellation_info: &RwLock<CancellationInfo>) {
        let mut guard = self.wait_lock.lock().unwrap();
        let mut tracker = WaitTimeoutTracker::new();

        loop {
            if cancellation_info.read().unwrap().token.is_cancelled() {
                break;
            }

            let current = self.counter.load(Ordering::Acquire);
            if current == 0 {
                break;
            }

            if !tracker.check_and_log(current, 0, "to reach zero", cancellation_info) {
                break;
            }

            let (g, _) = self.cvar.wait_timeout(guard, Duration::from_millis(100)).unwrap();
            guard = g;
        }
    }

    pub fn wait_until_zero_and_check_streams<F: FnMut()>(
        &self,
        mut check_streams: F,
        cancellation_info: &RwLock<CancellationInfo>,
    ) {
        let mut guard = self.wait_lock.lock().unwrap();
        let mut tracker = WaitTimeoutTracker::new();

        loop {
            if cancellation_info.read().unwrap().token.is_cancelled() {
                break;
            }

            let current = self.counter.load(Ordering::Acquire);
            if current == 0 {
                break;
            }

            if !tracker.check_and_log(current, 0, "to reach zero", cancellation_info) {
                break;
            }

            check_streams();
            let (g, _) = self.cvar.wait_timeout(guard, Duration::from_micros(100)).unwrap();
            guard = g;
        }
    }

    #[inline(always)]
    pub fn get_count(&self) -> usize {
        self.counter.load(Ordering::Acquire)
    }
}
