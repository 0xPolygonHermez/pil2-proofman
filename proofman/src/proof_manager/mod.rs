//! Proof-lifecycle accounting: the monotone [`Counter`] (count up toward a known total) and the
//! job-scoped completion machinery in [`completion`] (count a set of outstanding units down to zero).
mod completion;
pub use completion::*;

use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Condvar, Mutex, RwLock,
};
use std::time::{Duration, Instant};

use crate::{CancellationInfo, CancellationInfoExt};

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

        if elapsed.as_secs() >= WAIT_TIMEOUT_SECONDS {
            tracing::error!(
                "Counter timeout after {}s {} - current: {}, expected: {}. Cancelling.",
                WAIT_TIMEOUT_SECONDS,
                message_type,
                current,
                expected
            );
            cancellation_info.write_recover().token.cancel();
            return false;
        }

        true
    }
}

/// A monotone up-counter with a cancel-aware, stream-pumping wait to reach a target value. Counts
/// completions toward a known total (`witness_done`, `total_outer_agg_proofs`, `recursive2_done`);
/// only increments, so — unlike the down-counting [`Ledger`](crate::Ledger) — it can't underflow.
pub struct Counter {
    counter: AtomicUsize,
    wait_lock: Mutex<()>,
    cvar: Condvar,
}

impl Default for Counter {
    fn default() -> Self {
        Self::new()
    }
}

impl Counter {
    pub fn new() -> Self {
        Self { counter: AtomicUsize::new(0), wait_lock: Mutex::new(()), cvar: Condvar::new() }
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

        // Notify under the lock so a waiter re-checks the predicate; increments are per-proof, so
        // the notify cost is negligible.
        let _guard = self.wait_lock.lock().unwrap();
        self.cvar.notify_all();

        new_val
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
            if cancellation_info.read_recover().token.is_cancelled() {
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
}
