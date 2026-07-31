//! Key-affinity recursive scheduler.
//!
//! Centralizes the witness pick + CUDA stream reservation behind one lock, tracking which key
//! each stream holds resident (`stream_warm`) so a popular key drains on one stream instead of
//! reloading its const-tree on every free stream. Reservation picks in three passes: reuse a
//! warm free stream → fresh-load an unloaded key → (last resort) reload rather than idle.
//! Resident-tree basics are filler (never starve a ready compressor on the non-recursive pool).
//!
//! Wrap in a `Mutex`; the reserve FFIs are non-blocking, so the lock is held only briefly.

use std::collections::{BTreeMap, HashMap, HashSet, VecDeque};
use std::ffi::c_void;
use std::sync::{Condvar, Mutex};

use fields::PrimeField64;
use proofman_common::{MemoryHandlerRecursive, Proof, ProofType};

use crate::Ledger;
use proofman_starks_lib_c::{release_stream_reservation_c, reserve_best_stream_nonblock_c, reserve_stream_if_free_c};

/// Warm-affinity key: launches sharing it can reuse a resident const-tree.
type Key = (usize, usize, ProofType);

/// `true` for the aggregation proof types (recursive1/recursive2), which mirror the C
/// `aggregation` flag selecting the recursive stream pool. Compressor is not aggregation.
fn is_aggregation(t: ProofType) -> bool {
    t == ProofType::Recursive1 || t == ProofType::Recursive2
}

/// Recursive-stream priority: aggregate deeper first (rec2 is closer to the root than rec1).
/// Non-recursive streams handle compressors and basics separately (see `next_nonrecursive`).
const RECURSIVE_ORDER: [ProofType; 2] = [ProofType::Recursive2, ProofType::Recursive1];

/// A stream reserved (status=1) and not yet launched on. Every selection pass reads a reserved
/// stream as busy, so a caller that fails between pick and launch must hand the reservation back or
/// the slot is lost for the process lifetime. Dropping this guard does exactly that; [`Self::commit`]
/// transfers ownership to the launch, which leaves the stream at status=2 for the harvest to collect.
///
/// Constructed under the scheduler lock on every path that reserves ([`RecursiveScheduler::
/// pick_and_reserve`] and [`RecursiveScheduler::reserve_best`]), with nothing fallible in between,
/// so a reservation is never left unguarded.
pub struct StreamReservation {
    /// Opaque device-buffers handle (as usize for `Send`); only handed back to the FFI.
    d_buffers: usize,
    stream_id: u32,
    armed: bool,
}

// SAFETY: `d_buffers` is the opaque handle already shared across worker threads and only passed
// back to the C FFI, never dereferenced in Rust. The release entry point is internally locked.
unsafe impl Send for StreamReservation {}

impl StreamReservation {
    fn new(d_buffers: usize, stream_id: u32) -> Self {
        Self { d_buffers, stream_id, armed: true }
    }

    pub fn stream_id(&self) -> u32 {
        self.stream_id
    }

    /// The launch now owns this stream; stop releasing it on drop.
    pub fn commit(mut self) {
        self.armed = false;
    }
}

impl Drop for StreamReservation {
    fn drop(&mut self) {
        if self.armed {
            release_stream_reservation_c(self.d_buffers as *mut c_void, self.stream_id);
        }
    }
}

/// A single dispatch decision handed to a stream worker (chosen under the scheduler lock). Carrying
/// the [`StreamReservation`] rather than a bare id means a dropped pick returns its stream.
pub enum WorkerPick<F: PrimeField64> {
    /// A recursive/compressor witness to launch on the reserved stream.
    Recursive(Proof<F>, StreamReservation),
    /// A stored basic instance to launch on the reserved stream.
    Basic(usize, StreamReservation),
}

/// Candidate keys most-preferred first: restricted to `order`'s proof types (in that priority),
/// then backlog desc (drain big runs first → more reuse per load), then `(airgroup, air)`.
fn candidate_keys(ready: &[(Key, usize)], order: &[ProofType]) -> Vec<Key> {
    let mut out = Vec::new();
    for &t in order {
        let mut of_type: Vec<(Key, usize)> = ready.iter().copied().filter(|((_, _, kt), _)| *kt == t).collect();
        // backlog desc, then (airgroup, air) asc
        of_type.sort_by(|(ka, ba), (kb, bb)| bb.cmp(ba).then((ka.0, ka.1).cmp(&(kb.0, kb.1))));
        out.extend(of_type.into_iter().map(|(k, _)| k));
    }
    out
}

/// Centralized key-affinity scheduler for recursive/compressor launches (+ stored basics).
pub struct RecursiveScheduler<F: PrimeField64> {
    /// Opaque device-buffers handle (as usize for `Send`); only handed back to the FFI.
    d_buffers: usize,
    /// Ready recursive/compressor witnesses, bucketed by key.
    queues: HashMap<Key, VecDeque<Proof<F>>>,
    /// Ready "stored" basic instances (recompute path), bucketed by `(airgroup, air)`.
    /// Resident-in-GPU basics (skip_recalculation, pinned) never enter here.
    basic_queue: HashMap<(usize, usize), VecDeque<usize>>,
    /// Basic AIRs whose const-tree is resident (preallocated on-device). Held back as filler:
    /// they load nothing, and a big resident table draining first would starve ready
    /// compressors on the shared non-recursive streams.
    resident_keys: HashSet<Key>,
    /// physical stream -> key it currently holds resident (mirrors the CUDA side across
    /// `reset(false)`). Shared across basic and recursive. Ordered, not hashed: pass 1 scans it to
    /// choose among equally-warm free streams, and a `HashMap`'s arbitrary order would make that
    /// choice — and so the whole stream assignment — differ between otherwise identical runs.
    stream_warm: BTreeMap<usize, Key>,
}

// `Send` is derived, not asserted: `d_buffers` is stored as a `usize` (the opaque handle is only
// ever handed back to the C FFI, never dereferenced in Rust) and `PrimeField64: Field: Send`, so
// the auto impl covers the whole struct and the `F: Send` requirement stays compiler-checked.

impl<F: PrimeField64> RecursiveScheduler<F> {
    pub fn new(d_buffers: *mut c_void) -> Self {
        Self {
            d_buffers: d_buffers as usize,
            queues: HashMap::new(),
            basic_queue: HashMap::new(),
            resident_keys: HashSet::new(),
            stream_warm: BTreeMap::new(),
        }
    }

    fn d_buffers(&self) -> *mut c_void {
        self.d_buffers as *mut c_void
    }

    pub fn push(&mut self, w: Proof<F>) {
        let key = (w.airgroup_id, w.air_id, w.proof_type);
        self.queues.entry(key).or_default().push_back(w);
    }

    pub fn is_empty(&self) -> bool {
        self.queues.values().all(|q| q.is_empty())
    }

    fn pop_key(&mut self, key: Key) -> Option<Proof<F>> {
        let q = self.queues.get_mut(&key)?;
        let item = q.pop_front();
        if q.is_empty() {
            self.queues.remove(&key);
        }
        item
    }

    /// True if some physical stream currently holds `key` resident.
    fn is_warm_somewhere(&self, key: Key) -> bool {
        self.stream_warm.values().any(|k| *k == key)
    }

    /// Pick + reserve a stream among `order`'s proof types → `(witness, stream)`, or `None` if
    /// none is ready / no free stream. `force_recursive` restricts the reservation to a
    /// recursive stream. The returned stream is reserved (status=1): the caller MUST launch.
    fn next_of_types(&mut self, order: &[ProofType], force_recursive: bool) -> Option<(Proof<F>, StreamReservation)> {
        let ready: Vec<(Key, usize)> =
            self.queues.iter().filter(|(_, q)| !q.is_empty()).map(|(k, q)| (*k, q.len())).collect();
        let candidates = candidate_keys(&ready, order);
        if candidates.is_empty() {
            return None;
        }
        let (key, s) = self.pick_and_reserve(&candidates, force_recursive)?;
        // `s` is live from here on: if this `expect` ever fired, unwinding would release the stream.
        let w = self.pop_key(key).expect("candidate key non-empty");
        Some((w, s))
    }

    /// Force-recursive stream: rec2 > rec1 (never compressor), on a recursive stream.
    pub fn next_recursive(&mut self) -> Option<(Proof<F>, StreamReservation)> {
        self.next_of_types(&RECURSIVE_ORDER, true)
    }

    /// Non-recursive stream, one shot: compressor > basic > rec2/rec1. Deciding here (not via two
    /// racy gates) means a worker yields a basic only when it actually takes recursive work — no
    /// "yield-then-fail-to-place" idle. Resident basics are filler: eligible only when nothing
    /// recursive/compressor is queued.
    pub fn next_nonrecursive(&mut self) -> Option<WorkerPick<F>> {
        if let Some((w, s)) = self.next_of_types(&[ProofType::Compressor], false) {
            return Some(WorkerPick::Recursive(w, s));
        }
        let allow_resident = self.is_empty();
        if let Some((id, s)) = self.next_basic(allow_resident) {
            return Some(WorkerPick::Basic(id, s));
        }
        if let Some((w, s)) = self.next_of_types(&RECURSIVE_ORDER, false) {
            return Some(WorkerPick::Recursive(w, s));
        }
        None
    }

    /// Pick + reserve a stream for the next stored basic → `(instance_id, stream)`. Resident
    /// basics are excluded unless `include_resident` (held back as filler otherwise).
    pub fn next_basic(&mut self, include_resident: bool) -> Option<(usize, StreamReservation)> {
        let mut ready: Vec<((usize, usize), usize)> = self
            .basic_queue
            .iter()
            .filter(|(k, q)| {
                !q.is_empty() && (include_resident || !self.resident_keys.contains(&(k.0, k.1, ProofType::Basic)))
            })
            .map(|(k, q)| (*k, q.len()))
            .collect();
        if ready.is_empty() {
            return None;
        }
        // Plain key-affinity: backlog desc (drain big runs first), then key for determinism.
        ready.sort_by(|(ka, ba), (kb, bb)| bb.cmp(ba).then(ka.cmp(kb)));
        let candidates: Vec<Key> = ready.iter().map(|((ag, air), _)| (*ag, *air, ProofType::Basic)).collect();

        let (key, s) = self.pick_and_reserve(&candidates, false)?;
        let (ag, air, _) = key;
        let q = self.basic_queue.get_mut(&(ag, air)).expect("candidate basic key non-empty");
        let id = q.pop_front().expect("candidate basic key non-empty");
        if q.is_empty() {
            self.basic_queue.remove(&(ag, air));
        }
        Some((id, s))
    }

    /// Enqueue a stored basic instance for key-affinity dispatch. `resident` marks whether
    /// this AIR's basic const-tree is preallocated on-device (→ treated as filler).
    pub fn push_basic(&mut self, instance_id: usize, airgroup_id: usize, air_id: usize, resident: bool) {
        if resident {
            self.resident_keys.insert((airgroup_id, air_id, ProofType::Basic));
        }
        self.basic_queue.entry((airgroup_id, air_id)).or_default().push_back(instance_id);
    }

    pub fn basic_is_empty(&self) -> bool {
        self.basic_queue.values().all(|q| q.is_empty())
    }

    /// Three-pass reservation shared by `next_of_types`/`next_basic`: (1) reuse a candidate's warm
    /// free stream, (2) fresh-load a candidate on no stream, (3) last resort reload rather
    /// than idle. Returns the chosen `(key, reserved stream)`, or `None` if no free stream.
    fn pick_and_reserve(&mut self, candidates: &[Key], force_recursive: bool) -> Option<(Key, StreamReservation)> {
        let d = self.d_buffers();

        // Pass 1 — REUSE: a stream already holding this key, free right now.
        for &key in candidates {
            for (&s, &k) in self.stream_warm.iter() {
                if k == key && reserve_stream_if_free_c(d, s as u32, force_recursive) {
                    return Some((key, StreamReservation::new(self.d_buffers, s as u32)));
                }
            }
        }
        // Pass 2 — FRESH.
        for &key in candidates {
            if self.is_warm_somewhere(key) {
                continue; // loaded somewhere → not fresh
            }
            if let Some(s) = self.reserve_best(key, force_recursive) {
                return Some((key, s));
            }
        }
        // Pass 3 — REDUNDANT (last resort).
        for &key in candidates {
            if let Some(s) = self.reserve_best(key, force_recursive) {
                return Some((key, s));
            }
        }
        None
    }

    /// Reserve the best free stream for `key` (today's C selection), recording the
    /// assignment so future launches of `key` target this stream and stale entries for a
    /// repurposed stream are overwritten. Returns the reserved stream or `None`.
    fn reserve_best(&mut self, key: Key, force_recursive: bool) -> Option<StreamReservation> {
        let (ag, air, t) = key;
        let type_str: &'static str = t.into();
        let s = reserve_best_stream_nonblock_c(
            self.d_buffers(),
            ag as u64,
            air as u64,
            type_str,
            is_aggregation(t),
            force_recursive,
        );
        if s == u32::MAX {
            return None;
        }
        self.stream_warm.insert(s as usize, key);
        Some(StreamReservation::new(self.d_buffers, s))
    }

    /// Teardown drain: take everything still queued and leave the scheduler empty. Returns the
    /// queued recursive/compressor witnesses and the queued stored-basic instance ids.
    ///
    /// The caller MUST return each witness's `circom_witness` to its pool and settle its ledger
    /// unit: these were armed and committed at hand-off, and their buffers came out of the recursive
    /// witness pools, so dropping them shrinks those pools (the compressor pool most visibly, since
    /// it is the smallest) and leaves the pool-integrity check in `reset()` short. Only safe once
    /// every producer and consumer is joined — otherwise a live worker can push right after.
    pub fn drain_all(&mut self) -> (Vec<Proof<F>>, Vec<usize>) {
        let witnesses: Vec<Proof<F>> = self.queues.drain().flat_map(|(_, q)| q.into_iter()).collect();
        let basics: Vec<usize> = self.basic_queue.drain().flat_map(|(_, q)| q.into_iter()).collect();
        self.resident_keys.clear();
        (witnesses, basics)
    }
}

/// Scheduler shared across the witness and stream workers: the lock plus the condvar that
/// parks idle stream workers until new work arrives. Wrap in an `Arc` at the use site.
pub struct SharedScheduler<F: PrimeField64> {
    pub lock: Mutex<RecursiveScheduler<F>>,
    pub ready: Condvar,
}

impl<F: PrimeField64> SharedScheduler<F> {
    pub fn new(inner: RecursiveScheduler<F>) -> Self {
        Self { lock: Mutex::new(inner), ready: Condvar::new() }
    }

    /// Push a ready recursive/compressor witness and wake a parked stream worker.
    pub fn push(&self, w: Proof<F>) {
        self.lock.lock().unwrap().push(w);
        self.ready.notify_all();
    }
}

/// Recover witnesses taken out of the scheduler by [`RecursiveScheduler::drain_all`]: return each
/// `circom_witness` to the pool `generate_witness` took it from, and settle the ledger unit that was
/// armed for it at hand-off. Returns how many were recovered.
///
/// Split out of the teardown closure so the scheduler/ledger/pool interaction is directly testable:
/// the pool a witness goes back to is keyed off its `proof_type` (compressor witnesses come from the
/// separate, smaller compressor pool), and the ledger unit off `(global_idx, proof_type)` — the same
/// pair used to arm it. Getting either wrong silently shrinks a pool or strands a unit.
pub fn recover_drained_witnesses<F: PrimeField64 + Send + Sync + 'static>(
    witnesses: Vec<Proof<F>>,
    memory_handler_recursive_witness: &MemoryHandlerRecursive<F>,
    ledger: &Ledger,
) -> usize {
    let recovered = witnesses.len();
    for mut w in witnesses {
        let compressor = w.proof_type == ProofType::Compressor;
        // Adopt-then-drop returns the buffer to its pool.
        drop(memory_handler_recursive_witness.adopt_witness(std::mem::take(&mut w.circom_witness), compressor));
        if let Some(idx) = w.global_idx {
            ledger.settle(idx as u64, w.proof_type.as_usize());
        }
    }
    recovered
}

/// Sort key for the basic-proof schedule: `priority_tier` (front-load stored / has-compressor
/// AIRs to feed the recursive pipeline), then heaviest-per-proof first (LPT), then
/// `(airgroup_id, air_id)` to cluster each AIR's instances contiguously for const-tree reuse.
/// `proof_cost` is a per-AIR proxy, e.g. `(1 << n_bits) * n_cols`.
pub fn schedule_key(
    airgroup_id: usize,
    air_id: usize,
    is_stored: bool,
    has_compressor: bool,
    proof_cost: u64,
) -> (u8, std::cmp::Reverse<u64>, usize, usize) {
    let priority_tier: u8 = if is_stored && has_compressor {
        0
    } else if is_stored {
        1
    } else if has_compressor {
        2
    } else {
        3
    };
    (priority_tier, std::cmp::Reverse(proof_cost), airgroup_id, air_id)
}

#[cfg(test)]
mod tests {
    use super::{candidate_keys, is_aggregation, Key, RECURSIVE_ORDER};
    use proofman_common::ProofType;

    const FEED_ORDER: [ProofType; 3] = [ProofType::Compressor, ProofType::Recursive2, ProofType::Recursive1];

    fn k(ag: usize, air: usize, t: ProofType) -> Key {
        (ag, air, t)
    }

    #[test]
    fn aggregation_flag_matches_c() {
        assert!(is_aggregation(ProofType::Recursive1));
        assert!(is_aggregation(ProofType::Recursive2));
        assert!(!is_aggregation(ProofType::Compressor));
    }

    #[test]
    fn recursive_order_is_rec2_then_rec1() {
        assert_eq!(RECURSIVE_ORDER, [ProofType::Recursive2, ProofType::Recursive1]);
    }

    #[test]
    fn candidates_prefer_bigger_backlog_within_type() {
        let ready = vec![(k(0, 5, ProofType::Recursive1), 2), (k(0, 2, ProofType::Recursive1), 9)];
        let out = candidate_keys(&ready, &FEED_ORDER);
        assert_eq!(out, vec![k(0, 2, ProofType::Recursive1), k(0, 5, ProofType::Recursive1)]);
    }

    #[test]
    fn candidates_respect_type_priority() {
        let ready = vec![
            (k(0, 1, ProofType::Recursive1), 5),
            (k(0, 2, ProofType::Compressor), 1),
            (k(0, 3, ProofType::Recursive2), 1),
        ];
        let out = candidate_keys(&ready, &FEED_ORDER);
        assert_eq!(
            out,
            vec![k(0, 2, ProofType::Compressor), k(0, 3, ProofType::Recursive2), k(0, 1, ProofType::Recursive1),]
        );
    }

    #[test]
    fn candidates_recursive_order_drops_compressor() {
        let ready = vec![(k(0, 2, ProofType::Compressor), 9), (k(0, 1, ProofType::Recursive1), 1)];
        let out = candidate_keys(&ready, &RECURSIVE_ORDER);
        assert_eq!(out, vec![k(0, 1, ProofType::Recursive1)]);
    }

    #[test]
    fn candidates_tie_break_by_airgroup_air() {
        let ready = vec![(k(0, 7, ProofType::Recursive1), 3), (k(0, 2, ProofType::Recursive1), 3)];
        let out = candidate_keys(&ready, &FEED_ORDER);
        assert_eq!(out, vec![k(0, 2, ProofType::Recursive1), k(0, 7, ProofType::Recursive1)]);
    }
}

/// Teardown recovery: the scheduler holds pooled buffers and armed ledger units, so what happens to
/// its queues when a phase ends early is a correctness property, not bookkeeping. These use a null
/// device pointer throughout — `push`/`drain_all` never touch the FFI, and a null-pointer
/// `CompletionOwner` makes its harvest a no-op (see `completion.rs` tests).
#[cfg(test)]
mod drain_tests {
    use super::*;
    use crate::{DeviceBuffersPtr, DeviceCompletions};
    use fields::{Field, Goldilocks};
    use proofman_common::MemoryHandlerRecursive;

    type F = Goldilocks;

    const W_SIZE: usize = 8;
    const W_SIZE_COMPRESSOR: usize = 4;

    fn scheduler() -> RecursiveScheduler<F> {
        RecursiveScheduler::<F>::new(std::ptr::null_mut())
    }

    /// Two witness buffers per pool so a drained-and-returned buffer is distinguishable from a
    /// refilled one: `reset()` only passes if the originals come back.
    fn handler() -> MemoryHandlerRecursive<F> {
        MemoryHandlerRecursive::new(2, 2, W_SIZE, W_SIZE_COMPRESSOR, W_SIZE, W_SIZE_COMPRESSOR)
    }

    /// A witness as the hand-off builds it: `global_idx` set (the ledger keys off it) and holding a
    /// buffer drawn from the pool its `proof_type` selects.
    fn witness(h: &MemoryHandlerRecursive<F>, t: ProofType, global_idx: usize) -> Proof<F> {
        let buf = if t == ProofType::Compressor { h.take_buffer_witness_compressor() } else { h.take_buffer_witness() };
        Proof::new_witness(t, 0, 0, Some(global_idx), buf, 1)
    }

    #[test]
    fn drain_all_takes_everything_and_leaves_the_scheduler_empty() {
        let h = handler();
        let mut s = scheduler();
        s.push(witness(&h, ProofType::Compressor, 0));
        s.push(witness(&h, ProofType::Recursive1, 1));
        s.push(witness(&h, ProofType::Recursive2, 2));
        s.push_basic(10, 0, 0, false);
        s.push_basic(11, 0, 1, true);

        let (witnesses, basics) = s.drain_all();
        assert_eq!(witnesses.len(), 3);
        assert_eq!(basics.len(), 2);
        assert!(s.is_empty() && s.basic_is_empty(), "drain must leave nothing behind");

        // A second drain is a no-op, so a double teardown can't double-return a buffer.
        let (w2, b2) = s.drain_all();
        assert!(w2.is_empty() && b2.is_empty());

        recover_drained_witnesses(witnesses, &h, &DeviceCompletions::new().acquire(null_ptr()).ledger());
    }

    fn null_ptr() -> DeviceBuffersPtr {
        DeviceBuffersPtr(std::ptr::null_mut())
    }

    #[test]
    fn recovery_returns_compressor_buffers_to_the_compressor_pool() {
        // The pool that shrank in the field: compressor witnesses come from their own, smallest pool,
        // so returning one to the plain witness pool (or dropping it) leaves both pools wrong.
        let h = handler();
        let mut s = scheduler();
        s.push(witness(&h, ProofType::Compressor, 0));
        s.push(witness(&h, ProofType::Compressor, 1));
        // Both compressor buffers are out of the pool now; a drop here would lose them for good.
        let (witnesses, _) = s.drain_all();

        let owner = DeviceCompletions::new().acquire(null_ptr());
        assert_eq!(recover_drained_witnesses(witnesses, &h, &owner.ledger()), 2);

        // Every buffer is back in the pool it came from, so the integrity check passes. This is the
        // assertion that fails if a compressor buffer is dropped or filed under the wrong pool.
        h.reset().expect("all four pools must be whole after recovery");
    }

    #[test]
    fn recovery_settles_the_units_armed_at_hand_off() {
        let h = handler();
        let owner = DeviceCompletions::new().acquire(null_ptr());
        let ledger = owner.ledger();

        let mut s = scheduler();
        // Exactly what the hand-off does: arm by (child_id, kind), commit, then push.
        for (t, idx) in [(ProofType::Compressor, 0usize), (ProofType::Recursive1, 1), (ProofType::Recursive2, 2)] {
            ledger.arm(idx as u64, t.as_usize()).commit();
            s.push(witness(&h, t, idx));
        }
        assert_eq!(ledger.remaining(), 3);

        let (witnesses, _) = s.drain_all();
        recover_drained_witnesses(witnesses, &h, &ledger);
        assert_eq!(ledger.remaining(), 0, "a drained witness's unit must not stay outstanding");
        h.reset().expect("pools whole");
    }

    #[test]
    fn recovery_does_not_settle_a_unit_it_does_not_own() {
        // Guards the key pairing: a drained compressor must settle (id, Compressor), never the basic
        // unit that shares its numeric id and is still legitimately in flight.
        let h = handler();
        let owner = DeviceCompletions::new().acquire(null_ptr());
        let ledger = owner.ledger();
        ledger.arm(7, ProofType::Basic.as_usize()).commit();
        ledger.arm(7, ProofType::Compressor.as_usize()).commit();
        assert_eq!(ledger.remaining(), 2);

        let mut s = scheduler();
        s.push(witness(&h, ProofType::Compressor, 7));
        let (witnesses, _) = s.drain_all();
        recover_drained_witnesses(witnesses, &h, &ledger);

        assert_eq!(ledger.remaining(), 1, "only the compressor unit settles");
        drop(ledger.adopt(7, ProofType::Basic.as_usize())); // the basic unit is still the one left
        assert_eq!(ledger.remaining(), 0);
        h.reset().expect("pools whole");
    }

    #[test]
    fn a_witness_without_a_global_idx_still_returns_its_buffer() {
        // `global_idx` is set for every hand-off path today; if that ever regresses, the buffer must
        // still come back rather than the recovery panicking on an `unwrap`.
        let h = handler();
        let buf = h.take_buffer_witness();
        let mut s = scheduler();
        s.push(Proof::new_witness(ProofType::Recursive1, 0, 0, None, buf, 1));

        let (witnesses, _) = s.drain_all();
        recover_drained_witnesses(witnesses, &h, &DeviceCompletions::new().acquire(null_ptr()).ledger());
        h.reset().expect("buffer returned even with no unit to settle");
    }

    #[test]
    fn ledger_kind_spellings_agree() {
        // The ledger key mixes two spellings of the same number: units are armed with
        // `ProofType::X as usize` (the discriminant) and settled with `.as_usize()` (a match arm).
        // If those ever diverge, arm and settle land on different keys and every unit strands.
        for (t, discriminant) in [
            (ProofType::Basic, ProofType::Basic as usize),
            (ProofType::Compressor, ProofType::Compressor as usize),
            (ProofType::Recursive1, ProofType::Recursive1 as usize),
            (ProofType::Recursive2, ProofType::Recursive2 as usize),
            (ProofType::VadcopFinal, ProofType::VadcopFinal as usize),
            (ProofType::VadcopFinalCompressed, ProofType::VadcopFinalCompressed as usize),
            (ProofType::RecursiveF, ProofType::RecursiveF as usize),
            (ProofType::RecurserAggregator, ProofType::RecurserAggregator as usize),
        ] {
            assert_eq!(t.as_usize(), discriminant, "as_usize() must match `{t:?} as usize`");
        }
    }

    #[test]
    fn dropping_the_scheduler_undrained_is_what_loses_the_buffers() {
        // The failure this recovery exists to prevent, pinned down so the cost of skipping the drain
        // stays visible: without it the pool comes back short and `reset()` reports the leak.
        let h = handler();
        let mut s = scheduler();
        s.push(witness(&h, ProofType::Compressor, 0));
        s.queues.clear(); // drop the queued witness instead of draining it
        assert!(h.reset().is_err(), "a dropped witness permanently shrinks its pool");
        let _ = F::ZERO; // keep the Field import honest across backends
    }
}

#[cfg(test)]
mod schedule_tests {
    use super::schedule_key;
    use std::cmp::Reverse;

    // (airgroup, air, is_stored, has_compressor, proof_cost)
    fn sort_ids(items: &[(usize, usize, bool, bool, u64)]) -> Vec<usize> {
        let mut ids: Vec<usize> = (0..items.len()).collect();
        ids.sort_by_key(|&i| {
            let (ag, air, st, hc, cost) = items[i];
            schedule_key(ag, air, st, hc, cost)
        });
        ids
    }

    #[test]
    fn priority_tier_orders_stored_and_compressor_first() {
        // tiers: (stored+comp)=0, (stored)=1, (comp)=2, (neither)=3
        let items = [
            (9, 0, false, false, 100), // tier 3
            (8, 0, true, true, 100),   // tier 0
            (7, 0, false, true, 100),  // tier 2
            (6, 0, true, false, 100),  // tier 1
        ];
        let tiers: Vec<u8> = sort_ids(&items)
            .iter()
            .map(|&i| schedule_key(items[i].0, items[i].1, items[i].2, items[i].3, items[i].4).0)
            .collect();
        assert_eq!(tiers, vec![0, 1, 2, 3]);
    }

    #[test]
    fn lpt_orders_heavier_group_first_within_tier() {
        // Same tier (neither stored nor compressor); heavier proof_cost must come first.
        let items = [(0, 0, false, false, 50), (0, 1, false, false, 500)];
        let ordered = sort_ids(&items);
        assert_eq!(ordered, vec![1, 0]); // air (0,1) cost 500 before (0,0) cost 50
    }

    #[test]
    fn key_is_reverse_on_cost() {
        // Guards the LPT direction explicitly.
        let hi = schedule_key(0, 0, false, false, 500);
        let lo = schedule_key(0, 0, false, false, 50);
        assert!(hi < lo); // higher cost sorts earlier
        assert_eq!(hi.1, Reverse(500));
    }
}
