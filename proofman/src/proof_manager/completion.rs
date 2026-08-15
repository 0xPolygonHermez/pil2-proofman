//! Proof-done completion accounting, modelled as an owned capability.
//!
//! A GPU proof is launched on one thread and completes asynchronously: the C++ harvest fires
//! `proof_done_callback`, which crosses the FFI into a Rust channel where a worker settles the unit.
//! The old bare [`Counter`](crate::Counter) plus process-global channel had two lifetime bugs: the
//! count could underflow (`fetch_sub` past zero wedged the phase until a 10-min timeout), and
//! single-owner-at-a-time was an unenforced convention. So [`DeviceCompletions`] owns the one
//! callback registration and hands out a [`CompletionOwner`] **one at a time**, made safe by:
//!
//! - **Idempotent settling, keyed by `(id, kind)`.** [`Ledger`] is a set of outstanding units, not a
//!   bare count (a basic proof and its recursive successor share a numeric id); settling an absent
//!   unit is a no-op, so a duplicate/late completion can neither double-count nor wrap past zero.
//! - **Drop drains before it releases, and the owner holds no sender.** The sender is moved into the
//!   C registration; clearing it in `Drop` disconnects the workers' receivers so they exit. Sharing
//!   only [`Arc<Ledger>`](Ledger), the owner never waits on a worker that is waiting on it.

use std::collections::HashSet;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex, RwLock};
use std::time::Duration;

use crossbeam_channel::{unbounded, Receiver};
use proofman_starks_lib_c::{
    clear_proof_done_callback_c, get_stream_proofs_c, get_stream_proofs_non_blocking_c, register_proof_done_callback_c,
    CompletionMsg,
};

use crate::CancellationInfo;

/// Poll cadence while draining outstanding units — unchanged from the previous busy-poll wait.
const POLL_INTERVAL: Duration = Duration::from_micros(100);
/// How long `Drop` pumps the non-blocking harvest for stragglers before giving up. Bounded because
/// on a cancelled job they may never complete; a late straggler is a no-op (idempotent settling).
const DRAIN_BUDGET: Duration = Duration::from_secs(5);
/// How long [`SlotToken::take`] waits before warning that the capability is still held. Only a
/// call-site ordering bug gets it here, and that would otherwise present as a silent hang.
const SLOT_WAIT_WARN_AFTER: Duration = Duration::from_secs(10);

/// One outstanding proof unit: `(id, ProofType as usize)`. Keying on the discriminant distinguishes
/// a basic proof from the recursive proof that reuses its id, without coupling to the enum.
type UnitKey = (u64, usize);

/// Process-lifetime owner of the single proof-done callback registration. Held by `ProofMan`; hands
/// out one [`CompletionOwner`] at a time via [`Self::acquire`], released when that owner drops.
pub struct DeviceCompletions {
    /// `true` while a `CompletionOwner` is alive. This is the "one at a time" guarantee.
    slot: Arc<Mutex<bool>>,
    /// Monotonic id handed to each owner (diagnostics only; see [`Ledger::epoch`]).
    next_epoch: AtomicU64,
}

impl Default for DeviceCompletions {
    fn default() -> Self {
        Self::new()
    }
}

impl DeviceCompletions {
    pub fn new() -> Self {
        Self { slot: Arc::new(Mutex::new(false)), next_epoch: AtomicU64::new(1) }
    }

    /// Take the right to receive proof-done completions. Blocks until any previous [`CompletionOwner`]
    /// has dropped — turning "stop the aggregation service before arming" from convention into an
    /// enforced invariant (the callers are already sequential, so it never really contends).
    ///
    /// `d_buffers` is used by the owner's `Drop` for the final blocking harvest; null on the CPU backend.
    pub fn acquire(&self, d_buffers: DeviceBuffersPtr) -> CompletionOwner {
        let slot = SlotToken::take(&self.slot);
        let epoch = self.next_epoch.fetch_add(1, Ordering::Relaxed);
        let ledger = Arc::new(Ledger::new(epoch));

        // The sender is MOVED into the C registration; owner and workers hold only receivers. That's
        // why clearing the registration in `Drop` disconnects every receiver.
        let (tx, rx) = unbounded::<CompletionMsg>();
        register_proof_done_callback_c(tx);

        CompletionOwner { ledger, rx, d_buffers, _slot: slot }
    }
}

/// Device-buffer pointer carried across threads for the final harvest in `Drop`. Owned by the device
/// layer (outlives every [`CompletionOwner`]); the wrapper only carries it without making the owner
/// non-`Send`.
#[derive(Clone, Copy)]
pub struct DeviceBuffersPtr(pub *mut std::ffi::c_void);

// SAFETY: the pointer refers to device-layer state that lives for the whole process and is only
// passed to FFI entry points that are internally synchronised.
unsafe impl Send for DeviceBuffersPtr {}
unsafe impl Sync for DeviceBuffersPtr {}

/// Proof that the holder owns the completion capability. A `bool` behind a mutex rather than a held
/// `MutexGuard`, so the capability can span an arbitrary scope without borrowing from
/// [`DeviceCompletions`]. Waiting is a short poll — the sites that take it are already sequential.
struct SlotToken {
    slot: Arc<Mutex<bool>>,
}

impl SlotToken {
    fn take(slot: &Arc<Mutex<bool>>) -> Self {
        let start = std::time::Instant::now();
        let mut warned = false;
        loop {
            {
                // A poisoned slot only means a previous holder panicked; the flag is still
                // meaningful, so recover rather than abort teardown.
                let mut held = slot.lock().unwrap_or_else(|p| p.into_inner());
                if !*held {
                    *held = true;
                    return Self { slot: Arc::clone(slot) };
                }
            }
            // The callers are sequential, so this never really contends: waiting for more than a
            // moment means a previous owner was never released (typically an aggregation service
            // left running before an `acquire`). There is nothing to recover here — the wait is
            // unbounded by design — but it must not look like a silent hang.
            if !warned && start.elapsed() >= SLOT_WAIT_WARN_AFTER {
                warned = true;
                tracing::warn!(
                    "Waiting >{}s for the proof-done completion capability: a previous CompletionOwner \
                     is still alive. Stop the outer-aggregation service before acquiring.",
                    SLOT_WAIT_WARN_AFTER.as_secs()
                );
            }
            std::thread::sleep(POLL_INTERVAL);
        }
    }
}

impl Drop for SlotToken {
    fn drop(&mut self) {
        let mut held = self.slot.lock().unwrap_or_else(|p| p.into_inner());
        *held = false;
    }
}

/// Shared accounting for one completion epoch: which units are outstanding, and how many. Cloned as
/// `Arc<Ledger>` into every worker to arm and settle units. Deliberately holds no channel sender, so
/// a worker never keeps the registration alive (see module docs).
pub struct Ledger {
    epoch: u64,
    /// Outstanding units. The set — not a bare count — is the source of truth, which is what makes
    /// [`Self::settle`] idempotent and underflow-free.
    outstanding: Mutex<HashSet<UnitKey>>,
    /// Mirrors `outstanding.len()` so waiters have a cheap predicate without taking the set lock.
    remaining: AtomicUsize,
    wait_lock: Mutex<()>,
    cvar: Condvar,
}

impl Ledger {
    fn new(epoch: u64) -> Self {
        Self {
            epoch,
            outstanding: Mutex::new(HashSet::new()),
            remaining: AtomicUsize::new(0),
            wait_lock: Mutex::new(()),
            cvar: Condvar::new(),
        }
    }

    /// Monotonic id for this owner's registration. Diagnostics only — cross-owner completions are
    /// handled structurally (channel disconnect + idempotent settling), not by filtering on it.
    pub fn epoch(&self) -> u64 {
        self.epoch
    }

    /// Number of units still outstanding.
    pub fn remaining(&self) -> usize {
        self.remaining.load(Ordering::Acquire)
    }

    /// Record a new unit as in flight, returning a guard that settles it if the caller never hands it
    /// to the async pipeline. Re-arming an already-outstanding unit is a no-op for the count (the set
    /// rejects the duplicate), so a double-arm cannot inflate the ledger.
    #[must_use = "dropping the token settles the unit immediately; hold it until the work is launched"]
    pub fn arm(self: &Arc<Self>, id: u64, kind: usize) -> ProofToken {
        let key = (id, kind);
        {
            // Update the set and its mirror count under one lock, so `remaining` is never observed
            // out of step with the set and an interleaved `settle` can't fetch_sub below its size.
            let mut set = self.outstanding.lock().unwrap_or_else(|p| p.into_inner());
            if set.insert(key) {
                self.remaining.fetch_add(1, Ordering::AcqRel);
            }
        }
        ProofToken { ledger: Arc::clone(self), key, armed: true }
    }

    /// Adopt settling of an *already*-outstanding unit without changing the count — a worker taking
    /// over an in-flight unit to launch. The returned guard settles on drop unless
    /// [`commit`](ProofToken::commit)ted, so a failed launch still balances the ledger.
    #[must_use = "dropping the token settles the unit immediately; hold it until the work is launched"]
    pub fn adopt(self: &Arc<Self>, id: u64, kind: usize) -> ProofToken {
        ProofToken { ledger: Arc::clone(self), key: (id, kind), armed: true }
    }

    /// Settle one outstanding unit. Idempotent: a unit that is not outstanding (already settled,
    /// never armed, or from a previous owner) leaves the ledger untouched — removing the old
    /// `fetch_sub(1) - 1` underflow and the ordered-`reset()` workaround that dodged the wrap.
    pub fn settle(&self, id: u64, kind: usize) {
        // Remove from the set and decrement the mirror count under one lock (see `arm`). The notify
        // is done after releasing it, holding only `wait_lock`, to keep the two locks unnested.
        let now = {
            let mut set = self.outstanding.lock().unwrap_or_else(|p| p.into_inner());
            if !set.remove(&(id, kind)) {
                return;
            }
            self.remaining.fetch_sub(1, Ordering::AcqRel) - 1
        };
        if now == 0 {
            let _g = self.wait_lock.lock().unwrap_or_else(|p| p.into_inner());
            self.cvar.notify_all();
        }
    }

    /// Block until every unit has settled, the job is cancelled, or `timeout` elapses; returns `true`
    /// only if the ledger reached zero. `pump` runs each iteration to drive the non-blocking harvest.
    /// The condvar is the mechanism; the 100 µs poll is only a backstop.
    pub fn wait_settled<P: FnMut()>(
        &self,
        mut pump: P,
        cancellation_info: &RwLock<CancellationInfo>,
        timeout: Option<Duration>,
    ) -> bool {
        let start = std::time::Instant::now();
        let mut guard = self.wait_lock.lock().unwrap_or_else(|p| p.into_inner());
        loop {
            if self.remaining.load(Ordering::Acquire) == 0 {
                return true;
            }
            let cancelled = {
                let info = cancellation_info.read().unwrap_or_else(|p| p.into_inner());
                info.token.is_cancelled()
            };
            if cancelled {
                return false;
            }
            if let Some(limit) = timeout {
                if start.elapsed() >= limit {
                    return false;
                }
            }
            pump();
            let (g, _) = self.cvar.wait_timeout(guard, POLL_INTERVAL).unwrap_or_else(|p| p.into_inner());
            guard = g;
        }
    }
}

/// The exclusive right to receive proof-done completions for one epoch (construct via
/// [`DeviceCompletions::acquire`]). Held by the phase's main thread only, never cloned into a worker,
/// so its `Drop` can tear the epoch down; workers instead get an [`Arc<Ledger>`](Ledger) and a receiver.
pub struct CompletionOwner {
    ledger: Arc<Ledger>,
    rx: Receiver<CompletionMsg>,
    /// Used by `Drop` to run the final blocking harvest before releasing the registration.
    d_buffers: DeviceBuffersPtr,
    _slot: SlotToken,
}

impl CompletionOwner {
    /// The shared ledger; clone into each worker so it can arm/adopt/settle.
    pub fn ledger(&self) -> Arc<Ledger> {
        Arc::clone(&self.ledger)
    }

    /// A receiver clone for a worker to consume completions from. All clones disconnect when this
    /// owner is dropped, which is how workers learn to exit (no sentinel counting).
    pub fn receiver(&self) -> Receiver<CompletionMsg> {
        self.rx.clone()
    }

    /// A monotonic id for this owner (diagnostics only; see [`Ledger::epoch`]).
    pub fn epoch(&self) -> u64 {
        self.ledger.epoch()
    }

    /// Units still outstanding.
    pub fn remaining(&self) -> usize {
        self.ledger.remaining()
    }

    /// See [`Ledger::wait_settled`].
    pub fn wait_settled<P: FnMut()>(
        &self,
        pump: P,
        cancellation_info: &RwLock<CancellationInfo>,
        timeout: Option<Duration>,
    ) -> bool {
        self.ledger.wait_settled(pump, cancellation_info, timeout)
    }
}

impl Drop for CompletionOwner {
    /// Drain, **then** release — in that order. Releasing first would drop still-in-flight
    /// completions (the lost-decrement bug); centralizing the order here means no call site can get
    /// it wrong. The drain is bounded because a cancelled job's units may never complete; giving up
    /// is safe because [`Ledger::settle`] is idempotent, so a late straggler is a no-op.
    fn drop(&mut self) {
        // (1) Give outstanding completions a bounded chance to land, pumping the non-blocking
        //     harvest so the GPU side can deliver them.
        let start = std::time::Instant::now();
        while self.ledger.remaining() > 0 && start.elapsed() < DRAIN_BUDGET {
            if !self.d_buffers.0.is_null() {
                get_stream_proofs_non_blocking_c(self.d_buffers.0);
            }
            std::thread::sleep(POLL_INTERVAL);
        }

        // (2) Final blocking harvest so anything already completed on the device is delivered.
        if !self.d_buffers.0.is_null() {
            get_stream_proofs_c(self.d_buffers.0);
        }

        let leaked = self.ledger.remaining();
        if leaked > 0 {
            // Expected on a cancelled job; noteworthy otherwise. Not fatal: the next owner starts
            // from an empty ledger, and late completions for this epoch are ignored.
            tracing::debug!(
                "CompletionOwner(epoch {}) released with {} unit(s) unsettled (expected after cancellation)",
                self.ledger.epoch(),
                leaked
            );
        }

        // (3) Release the registration. This drops the sole sender, so every receiver clone
        //     disconnects and its worker loop exits — no sentinel needed.
        clear_proof_done_callback_c();
    }
}

/// RAII guard for one in-flight unit. Dropping an uncommitted token settles its unit, so a unit
/// cannot leak on a failed launch, a cancellation break, or an unwind. [`ProofToken::commit`] hands
/// settlement to the async completion path once the work is genuinely launched.
#[must_use = "dropping the token settles the unit immediately; hold it until the work is launched"]
pub struct ProofToken {
    ledger: Arc<Ledger>,
    key: UnitKey,
    armed: bool,
}

impl ProofToken {
    /// The async completion now owns settling this unit.
    pub fn commit(mut self) {
        self.armed = false;
    }

    /// The proof id of this unit.
    pub fn id(&self) -> u64 {
        self.key.0
    }
}

impl Drop for ProofToken {
    fn drop(&mut self) {
        if self.armed {
            self.ledger.settle(self.key.0, self.key.1);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const BASIC: usize = 0;
    const RECURSIVE1: usize = 2;

    /// A CPU-backend owner: a null device pointer makes `Drop`'s harvest a no-op, so these tests
    /// exercise the ledger without touching the GPU.
    fn owner() -> CompletionOwner {
        DeviceCompletions::new().acquire(DeviceBuffersPtr(std::ptr::null_mut()))
    }

    #[test]
    fn arm_then_drop_settles() {
        let o = owner();
        let l = o.ledger();
        let t = l.arm(7, BASIC);
        assert_eq!(l.remaining(), 1);
        drop(t);
        assert_eq!(l.remaining(), 0);
    }

    #[test]
    fn commit_transfers_settlement_to_the_async_path() {
        let o = owner();
        let l = o.ledger();
        l.arm(7, BASIC).commit();
        assert_eq!(l.remaining(), 1, "a committed unit stays outstanding until it completes");
        l.settle(7, BASIC);
        assert_eq!(l.remaining(), 0);
    }

    #[test]
    fn adopt_does_not_change_the_count_but_settles_on_drop() {
        let o = owner();
        let l = o.ledger();
        l.arm(7, BASIC).commit();
        let taken = l.adopt(7, BASIC);
        assert_eq!(l.remaining(), 1, "adopt must not double-count an already-armed unit");
        drop(taken);
        assert_eq!(l.remaining(), 0, "an uncommitted adopt settles the unit it took over");
    }

    #[test]
    fn settling_an_unknown_unit_is_a_no_op() {
        // The case that used to wrap the counter to usize::MAX and wedge the prove phase.
        let o = owner();
        let l = o.ledger();
        l.settle(1234, BASIC);
        assert_eq!(l.remaining(), 0);
        l.arm(1, BASIC).commit();
        l.settle(999, BASIC);
        assert_eq!(l.remaining(), 1, "a stray completion must not settle someone else's unit");
    }

    #[test]
    fn the_same_id_with_a_different_kind_is_a_distinct_unit() {
        // A basic proof and the recursive proof derived from it can share a numeric id.
        let o = owner();
        let l = o.ledger();
        l.arm(5, BASIC).commit();
        l.arm(5, RECURSIVE1).commit();
        assert_eq!(l.remaining(), 2, "(5, Basic) and (5, Recursive1) are different units");
        l.settle(5, BASIC);
        assert_eq!(l.remaining(), 1, "settling the basic unit must not settle the recursive one");
        l.settle(5, RECURSIVE1);
        assert_eq!(l.remaining(), 0);
    }

    #[test]
    fn settling_twice_only_counts_once() {
        let o = owner();
        let l = o.ledger();
        l.arm(3, BASIC).commit();
        l.settle(3, BASIC);
        l.settle(3, BASIC);
        assert_eq!(l.remaining(), 0);
    }

    #[test]
    fn a_panicking_worker_still_settles_its_unit() {
        let o = owner();
        let l = o.ledger();
        let l2 = Arc::clone(&l);
        let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(move || {
            let _t = l2.arm(11, BASIC);
            panic!("failure between arming and launch");
        }));
        assert!(r.is_err());
        assert_eq!(l.remaining(), 0);
    }

    #[test]
    fn workers_learn_to_exit_when_the_owner_drops() {
        let o = owner();
        let rx = o.receiver();
        drop(o);
        assert!(rx.recv().is_err(), "dropping the owner disconnects consumers; no sentinel needed");
    }

    #[test]
    fn the_capability_is_exclusive() {
        let completions = DeviceCompletions::new();
        let null = DeviceBuffersPtr(std::ptr::null_mut());
        let first = completions.acquire(null);
        let epoch = first.epoch();
        drop(first);
        let second = completions.acquire(null);
        assert!(second.epoch() > epoch, "each owner gets a fresh epoch");
    }
}
