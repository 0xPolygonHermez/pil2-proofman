use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Arc, Condvar, Mutex, RwLock,
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
        // Release: this counter gates readiness for waiters that use Acquire
        // loads. A Relaxed store would not synchronize-with those loads, so a
        // thread observing the counter reaching threshold would have no
        // happens-before with the work this increment represents (e.g. a
        // producer's multiplicity fetch_add). Today the producer joins mask
        // this, but the counter must carry its own ordering.
        let new_val = self.counter.fetch_add(1, Ordering::Release) + 1;

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

            let (g, _) = self.cvar.wait_timeout(guard, Duration::from_millis(1)).unwrap();
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

            let (g, _) = self.cvar.wait_timeout(guard, Duration::from_millis(1)).unwrap();
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

/// RAII balance guard for one unit of in-flight proof work. Its `decrement()`
/// runs on `Drop` for any exit path (success, `break`, `?`, cancellation, panic),
/// so the counter cannot leak by a forgotten decrement. Call [`PendingProof::commit`]
/// once the work is handed to the async pipeline, where the C proof-done callback
/// owns the decrement instead.
#[must_use = "dropping the guard immediately decrements; hold it until the work is handed off or has failed"]
pub struct PendingProof {
    counter: Arc<Counter>,
    armed: bool,
}

impl PendingProof {
    /// Adopt an already-counted outstanding unit (no increment); its single
    /// `decrement()` is now owed by this guard.
    #[inline(always)]
    pub fn from_outstanding(counter: &Arc<Counter>) -> PendingProof {
        PendingProof { counter: counter.clone(), armed: true }
    }

    /// Disarm: the async proof-done callback owns the decrement now.
    #[inline(always)]
    pub fn commit(mut self) {
        self.armed = false;
    }
}

impl Drop for PendingProof {
    #[inline(always)]
    fn drop(&mut self) {
        if self.armed {
            self.counter.decrement();
        }
    }
}

impl Counter {
    /// Increment and return an armed [`PendingProof`] whose drop balances it.
    #[inline(always)]
    pub fn pending(self: &Arc<Self>) -> PendingProof {
        self.increment();
        PendingProof::from_outstanding(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pending_increments_and_drop_decrements() {
        let counter = Arc::new(Counter::new());
        let guard = counter.pending();
        assert_eq!(counter.get_count(), 1);
        drop(guard);
        assert_eq!(counter.get_count(), 0);
    }

    #[test]
    fn commit_disarms_the_decrement() {
        let counter = Arc::new(Counter::new());
        let guard = counter.pending();
        guard.commit();
        assert_eq!(counter.get_count(), 1, "committed guard must not decrement on drop");
        counter.decrement(); // the "callback" settles it
        assert_eq!(counter.get_count(), 0);
    }

    #[test]
    fn from_outstanding_adopts_without_incrementing() {
        let counter = Arc::new(Counter::new());
        counter.increment(); // unit counted at submit time
        let guard = PendingProof::from_outstanding(&counter);
        assert_eq!(counter.get_count(), 1, "adoption must not increment");
        drop(guard);
        assert_eq!(counter.get_count(), 0);
    }

    #[test]
    fn drop_decrements_on_panic_unwind() {
        let counter = Arc::new(Counter::new());
        let c = counter.clone();
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(move || {
            let _guard = c.pending();
            panic!("simulated failure between increment and handoff");
        }));
        assert!(result.is_err());
        assert_eq!(counter.get_count(), 0, "guard must settle the unit on unwind");
    }

    #[test]
    fn wait_until_zero_returns_once_guards_settle() {
        let counter = Arc::new(Counter::new());
        let cancellation = RwLock::new(CancellationInfo::default());
        let guard = counter.pending();
        let c = counter.clone();
        let handle = std::thread::spawn(move || {
            std::thread::sleep(Duration::from_millis(20));
            drop(guard);
            drop(c);
        });
        counter.wait_until_zero(&cancellation);
        assert_eq!(counter.get_count(), 0);
        handle.join().unwrap();
    }
}
