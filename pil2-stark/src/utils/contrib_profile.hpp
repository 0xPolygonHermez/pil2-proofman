#ifndef CONTRIB_PROFILE_HPP
#define CONTRIB_PROFILE_HPP

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstring>

// Per-commit profiling for the contributions phase.
//
// Why this exists: the only markers inside CALCULATING_CONTRIBUTIONS were
// "First GPU contribution queued" and the phase's own stop, so a ~0.8s stall in
// between was invisible. This records, for every commit_witness, the discrete
// branch points that can add a large chunk of work (const-pols slot miss, custom-
// fixed reload, staged vs direct H2D) plus how long the caller blocked waiting for
// a stream, and pairs them with the GPU event times the stream timer already
// measures but only logged at trace level.
//
// Hot-path constraints: fixed-capacity storage, no allocation, no locks, no
// syscalls. A record is two atomic ops and a struct store. Overflow is counted and
// dropped rather than resized — a profiler that reallocates under the thing it is
// timing measures itself.

// Flags on CommitProfileRecord::flags.
#define CONTRIB_FLAG_SLOT_WARM      (1u << 0)  // adoptFixedSlot hit: const pols already unpacked
#define CONTRIB_FLAG_UNPACK_FIXED   (1u << 1)  // unpack_fixed kernel ran (slot miss or alias)
#define CONTRIB_FLAG_CUSTOM_RELOAD  (1u << 2)  // custom-commits-fixed re-read from disk
#define CONTRIB_FLAG_H2D_DIRECT     (1u << 3)  // trace H2D took the pinned direct path
#define CONTRIB_FLAG_WITNESS_HINTS  (1u << 4)  // air has witness_calc hints (runs the expr block)

// Kept POD and #[repr(C)]-compatible: proofman drains these straight into Rust.
struct CommitProfileRecord {
    uint64_t instanceId;
    uint64_t airgroupId;
    uint64_t airId;
    uint64_t streamId;
    uint64_t flags;
    uint64_t select_wait_ns;   // blocked inside selectStream waiting for a free stream
    uint64_t select_retries;   // scan passes that found nothing (each costs a 300us sleep)
    uint64_t enqueue_ns;       // host wall time inside commit_witness_gpu
    uint64_t h2d_stage_ns;     // host wall time inside the staged (mutex_pinned) H2D path
    float gpu_commit_ms;       // STARK_GPU_COMMIT, device time
    float gpu_h2d_ms;
    float gpu_ntt_ms;
    float gpu_merkle_ms;
    float gpu_exprs_ms;
    float _pad;                // keep the struct 8-byte aligned on both sides
};

// The drain is a raw memcpy into a Rust #[repr(C)] struct, so the layout is load
// bearing: pin it here rather than discovering a mismatch as garbage timings.
static_assert(sizeof(CommitProfileRecord) == 96, "CommitProfileRecord layout changed; update the Rust mirror");
static_assert(alignof(CommitProfileRecord) == 8, "CommitProfileRecord alignment changed");

// One job's worth of commits on one worker: 204 global instances, so 256 covers a
// single-worker partition with headroom.
#define CONTRIB_PROFILE_CAPACITY 256

// Slots written by contrib_profile_totals, in order:
//  0 records, 1 dropped, 2 h2d_wait_ns, 3 h2d_wait_count, 4 pinned_lock_ns,
//  5 pinned_lock_count, 6 event_churn_ns, 7 event_churn_count,
//  8 gpu_free_min_bytes, 9 gpu_free_last_bytes, 10 gpu_total_bytes,
// 11 borrow_count, 12 borrow_drain_ns, 13 borrow_acq_sync_ns, 14 borrow_window_ns,
// 15 borrow_rel_sync_ns, 16 borrow_blocked_selects
#define CONTRIB_PROFILE_N_TOTALS 17

struct ContribProfileState {
    CommitProfileRecord records[CONTRIB_PROFILE_CAPACITY];
    std::atomic<uint64_t> count{0};
    std::atomic<uint64_t> dropped{0};

    // Aggregate waits that are not attributable to one commit.
    std::atomic<uint64_t> h2d_wait_ns{0};      // wait_trace_h2d_done: gating buffer reuse
    std::atomic<uint64_t> h2d_wait_count{0};
    std::atomic<uint64_t> pinned_lock_ns{0};   // blocked on the per-GPU mutex_pinned
    std::atomic<uint64_t> pinned_lock_count{0};
    std::atomic<uint64_t> event_churn_ns{0};   // cudaEventCreate/Destroy in the stream timers
    std::atomic<uint64_t> event_churn_count{0};

    // The first-GPU unified-buffer borrow taken by memory-ops count-and-plan during
    // exec. On a single-GPU box this is a hard stop for the whole commit pipeline:
    // stream selection skips every stream on the borrowed GPU, so contribution workers
    // can only spin. Release then invalidates every stream's context, so the commits
    // that follow all take the cold const-pols path.
    std::atomic<uint64_t> borrow_count{0};
    std::atomic<uint64_t> borrow_drain_ns{0};    // waiting for in-flight commits to finish
    std::atomic<uint64_t> borrow_acq_sync_ns{0}; // cudaDeviceSynchronize in acquire
    std::atomic<uint64_t> borrow_window_ns{0};   // acquire returns -> release called
    std::atomic<uint64_t> borrow_rel_sync_ns{0}; // cudaDeviceSynchronize in release
    std::atomic<uint64_t> borrow_blocked_selects{0}; // selectStream retries while borrowed
    // Set by acquire, read by release to close the window. Single borrower by
    // construction (count-and-plan is serialized), so a plain atomic suffices.
    std::atomic<uint64_t> borrow_open_ns{0};

    // Device memory sampled at each commit harvest. The box runs at ~95% VRAM
    // occupancy, which was the leading hypothesis for the stall; this makes the
    // headroom during the phase a measured number instead of an assumption.
    std::atomic<uint64_t> gpu_free_min_bytes{UINT64_MAX};
    std::atomic<uint64_t> gpu_free_last_bytes{0};
    std::atomic<uint64_t> gpu_total_bytes{0};

    void note_gpu_mem(uint64_t free_bytes, uint64_t total_bytes) {
        gpu_free_last_bytes.store(free_bytes, std::memory_order_relaxed);
        gpu_total_bytes.store(total_bytes, std::memory_order_relaxed);
        uint64_t prev = gpu_free_min_bytes.load(std::memory_order_relaxed);
        while (free_bytes < prev &&
               !gpu_free_min_bytes.compare_exchange_weak(prev, free_bytes, std::memory_order_relaxed)) {
        }
    }

    void reset() {
        count.store(0, std::memory_order_relaxed);
        dropped.store(0, std::memory_order_relaxed);
        h2d_wait_ns.store(0, std::memory_order_relaxed);
        h2d_wait_count.store(0, std::memory_order_relaxed);
        pinned_lock_ns.store(0, std::memory_order_relaxed);
        pinned_lock_count.store(0, std::memory_order_relaxed);
        event_churn_ns.store(0, std::memory_order_relaxed);
        event_churn_count.store(0, std::memory_order_relaxed);
        borrow_count.store(0, std::memory_order_relaxed);
        borrow_drain_ns.store(0, std::memory_order_relaxed);
        borrow_acq_sync_ns.store(0, std::memory_order_relaxed);
        borrow_window_ns.store(0, std::memory_order_relaxed);
        borrow_rel_sync_ns.store(0, std::memory_order_relaxed);
        borrow_blocked_selects.store(0, std::memory_order_relaxed);
        borrow_open_ns.store(0, std::memory_order_relaxed);
        gpu_free_min_bytes.store(UINT64_MAX, std::memory_order_relaxed);
        gpu_free_last_bytes.store(0, std::memory_order_relaxed);
        gpu_total_bytes.store(0, std::memory_order_relaxed);
    }

    void push(const CommitProfileRecord &rec) {
        uint64_t idx = count.fetch_add(1, std::memory_order_relaxed);
        if (idx >= CONTRIB_PROFILE_CAPACITY) {
            dropped.fetch_add(1, std::memory_order_relaxed);
            return;
        }
        records[idx] = rec;
    }
};

inline ContribProfileState &contribProfile() {
    static ContribProfileState state;
    return state;
}

// Host-side facts about the commit in flight, stashed on the stream between
// commit_witness_gpu (which knows the branch outcomes) and the harvest in
// get_commit_root (which is where the GPU event times become readable).
struct PendingCommitProfile {
    uint64_t flags = 0;
    uint64_t select_wait_ns = 0;
    uint64_t select_retries = 0;
    uint64_t enqueue_ns = 0;
    uint64_t h2d_stage_ns = 0;
    bool valid = false;

    void clear() { *this = PendingCommitProfile(); }
};

// selectStream reports its wait to whoever called it. Thread-local rather than an
// out-param so the several selectStream callers that do not profile stay untouched.
// A monotonic clock reading in ns, shared by every call site here so the borrow
// window and the per-commit timings are on the same timebase.
inline uint64_t contribNowNs() {
    return (uint64_t)std::chrono::duration_cast<std::chrono::nanoseconds>(
               std::chrono::steady_clock::now().time_since_epoch())
        .count();
}

inline thread_local uint64_t tl_select_wait_ns = 0;
inline thread_local uint64_t tl_select_retries = 0;

#endif
