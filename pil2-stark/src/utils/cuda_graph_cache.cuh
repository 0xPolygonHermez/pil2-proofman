#ifndef CUDA_GRAPH_CACHE_CUH
#define CUDA_GRAPH_CACHE_CUH

#ifdef USE_CUDA_GRAPH

#include <cuda_runtime.h>
#include <unordered_map>
#include <unordered_set>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cerrno>
#include <climits>
#include <atomic>
#include <mutex>

class CudaGraphCache;

namespace cudagraph {
    inline CudaGraphCache*& current() {
        static thread_local CudaGraphCache* ptr = nullptr;
        return ptr;
    }

    inline bool enabled() {
        static const bool v = [] {
            const char *e = getenv("CUDA_GRAPHS");
            return e == nullptr || e[0] != '0';
        }();
        return v;
    }

    // ---- Per-device graph memory budget: PROOFMAN_GPU_HEADROOM_MB - PROOFMAN_GPU_MIN_HEADROOM_MB.
    // Graph executables are driver allocations that live until the stream caches are freed; the
    // unified-buffer allocation sets what they may hold on each device (setBudget), every
    // instantiated graph is charged what cudaMemGetInfo saw it take, and once the next one does not
    // fit the device stops capturing for good and warns once.
    constexpr int MAX_DEVICES = 64;

    struct DeviceBudget {
        std::atomic<uint64_t> budget{0};     // bytes the graphs may hold on this device; 0 = none
        std::atomic<uint64_t> used{0};       // bytes charged; equal to budget once spent, for good
        std::mutex instantiate;              // one instantiation at a time per device: the charge is a memory delta
    };

    inline DeviceBudget& deviceBudget(int dev) {
        static DeviceBudget table[MAX_DEVICES];
        if (dev < 0 || dev >= MAX_DEVICES) {
            fprintf(stderr, "[cudaGraph] device ordinal %d is outside the budget table (%d devices)\n", dev, MAX_DEVICES);
            abort();
        }
        return table[dev];
    }

    inline void setBudget(int dev, uint64_t bytes) {
        DeviceBudget &b = deviceBudget(dev);
        b.used.store(0, std::memory_order_relaxed);
        b.budget.store(bytes, std::memory_order_release);
    }

    // The buffer is gone: so are the caches that held the graphs.
    inline void clearBudget(int dev) { setBudget(dev, 0); }

    inline bool captureAllowed() {
        int dev = 0;
        (void)cudaGetDevice(&dev);
        const DeviceBudget &b = deviceBudget(dev);
        return b.used.load(std::memory_order_acquire) < b.budget.load(std::memory_order_acquire);
    }

    // Charge an instantiated executable (under the device's `instantiate` mutex): `bytes` is what
    // the device lost across the cudaGraphInstantiate. True when it fits. The budget is spent for
    // good once it is full, by an exact fill or by a graph that does not fit (the caller destroys
    // that one, so the budget is never exceeded); the device warns once, at that moment.
    inline bool chargeGraph(int dev, uint64_t bytes) {
        DeviceBudget &b = deviceBudget(dev);
        const uint64_t budget = b.budget.load(std::memory_order_relaxed);
        const uint64_t used = b.used.load(std::memory_order_relaxed);
        if (used >= budget) return false;   // already spent: a capture that was in flight when it closed
        const bool fits = used + bytes <= budget;
        const uint64_t now = fits ? used + bytes : budget;
        b.used.store(now, std::memory_order_release);
        if (now >= budget) {
            fprintf(stderr, "[cudaGraph] GPU %d: graph budget spent (%llu of %llu MiB used%s): no more graphs are captured "
                            "on this device, further regions run directly (PROOFMAN_GPU_HEADROOM_MB - PROOFMAN_GPU_MIN_HEADROOM_MB)\n",
                    dev, (unsigned long long)((fits ? now : used) >> 20), (unsigned long long)(budget >> 20),
                    fits ? "" : ", the next graph did not fit");
        }
        return fits;
    }
}

// NOT internally synchronized. Thread safety is by ownership: each StreamData owns one
// instance, a stream is claimed under stream_selection_mutex and driven by a single host
// thread until release, and that mutex orders the handoff between successive owners. The
// capture state below (pending_key_, capturing_, poisoned_) is only valid between the
// owner's bind of cudagraph::current() (thread_local) and its guard-scoped unbind. Do not
// touch a stream's cache from CUDA callbacks or any thread that has not reserved the stream.
class CudaGraphCache {
    std::unordered_map<uint64_t, cudaGraphExec_t> cache_;
    std::unordered_map<uint64_t, uint32_t> hitCount_;
    // Keys whose region proved non-replayable (a launch inside staged per-proof data
    // through shared pinned slots — replaying such a graph would read another air's
    // values). Poisoned once, never captured again.
    std::unordered_set<uint64_t> blacklist_;
    uint64_t pending_key_ = 0;
    bool capturing_ = false;
    bool poisoned_ = false;

    // Captures are deferred until a key has been seen this many times. Overridable via
    // CUDA_GRAPH_CAPTURE_THRESHOLD (read once, thread-safe static init). Default 100,
    // chosen conservatively for a long-running prover service: capture+instantiate is
    // ~7 ms per graph (RTX 5090, 3+2-stream worker) so the deferral costs little, and a
    // key only proves itself hot after many executions. The old default (1000) left
    // every graph dormant for the first ~50-200 jobs per stream.
    // Floor of 10: a capture on a key's earliest executions could reach lazy one-time
    // init (e.g. the NTT twiddle tables' cudaMalloc + sync memcpy) whose calls are
    // illegal during stream capture and whose CHECKCUDAERR would abort the process;
    // after several executions all lazy init on the region's path has already run.
    static uint32_t captureThreshold() {
        static const uint32_t v = [] {
            constexpr uint32_t defaultThreshold = 100u;
            constexpr uint32_t floorThreshold = 10u;
            const char *e = getenv("CUDA_GRAPH_CAPTURE_THRESHOLD");
            if (e == nullptr || *e == '\0') return defaultThreshold;
            // Full-string numeric or it doesn't count: a typo must fall back to the
            // conservative default, not silently clamp to the aggressive floor.
            char *end = nullptr;
            errno = 0;
            unsigned long t = strtoul(e, &end, 10);
            if (end == e || *end != '\0' || errno == ERANGE || t > UINT32_MAX) {
                fprintf(stderr, "[cudaGraph] invalid CUDA_GRAPH_CAPTURE_THRESHOLD='%s', using default %u\n", e, defaultThreshold);
                return defaultThreshold;
            }
            return t < floorThreshold ? floorThreshold : (uint32_t)t;
        }();
        return v;
    }

    static void clearCudaError() {
        cudaGetLastError();
    }

public:
    CudaGraphCache() = default;

    ~CudaGraphCache() { clear(); }

    CudaGraphCache(const CudaGraphCache&) = delete;
    CudaGraphCache& operator=(const CudaGraphCache&) = delete;

    bool contains(uint64_t key) const {
        return cache_.find(key) != cache_.end();
    }

    bool tryLaunch(uint64_t key, cudaStream_t stream) {
        auto it = cache_.find(key);
        if (it == cache_.end()) return false;
        cudaError_t err = cudaGraphLaunch(it->second, stream);
        if (err != cudaSuccess) {
            // Never re-capture after a lost graph: the body's host state may already have been
            // skipped by earlier replays in this proof (a re-capture would bake it). The key runs
            // its body from now on; loud because this should not happen at steady state.
            fprintf(stderr, "[cudaGraph] launch failed key=%llx: %s -- key blacklisted\n",
                    (unsigned long long)key, cudaGetErrorString(err));
            clearCudaError();
            cudaGraphExecDestroy(it->second);
            cache_.erase(it);
            blacklist_.insert(key);
            return false;
        }
        return true;
    }

    bool shouldCapture(uint64_t key) {
        if (blacklist_.count(key)) return false;
        return ++hitCount_[key] >= captureThreshold();
    }

    bool beginCapture(uint64_t key, cudaStream_t stream) {
        pending_key_ = key;
        capturing_ = true;
        poisoned_ = false;
        cudaError_t err = cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal);
        if (err != cudaSuccess) {
            fprintf(stderr, "[cudaGraph] beginCapture failed: %s\n", cudaGetErrorString(err));
            clearCudaError();
            capturing_ = false;
            return false;
        }
        return true;
    }

    // Mark the in-flight capture as non-replayable (see blacklist_). The capture still
    // runs to endCaptureAndLaunch, which discards the graph and reports failure so the
    // caller re-executes the body directly.
    void poison() {
        if (capturing_) poisoned_ = true;
    }

    bool endCaptureAndLaunch(cudaStream_t stream) {
        if (!capturing_) return false;
        capturing_ = false;

        cudaGraph_t graph = nullptr;
        cudaError_t err = cudaStreamEndCapture(stream, &graph);
        if (err != cudaSuccess || graph == nullptr) {
            fprintf(stderr, "[cudaGraph] endCapture failed key=%llx: %s -- key blacklisted\n",
                    (unsigned long long)pending_key_, cudaGetErrorString(err));
            clearCudaError();
            if (graph) cudaGraphDestroy(graph);
            blacklist_.insert(pending_key_);
            return false;
        }

        if (poisoned_) {
            poisoned_ = false;
            blacklist_.insert(pending_key_);
            cudaGraphDestroy(graph);
            return false;
        }

        cudaGraphExec_t exec = nullptr;
        int dev = 0;
        (void)cudaGetDevice(&dev);
        {
            // One instantiation at a time per device, so the free-memory delta is this graph's (plus
            // whatever another stream allocated meanwhile: an overcharge, never an undercharge).
            std::lock_guard<std::mutex> lk(cudagraph::deviceBudget(dev).instantiate);
            size_t freeBefore = 0, freeAfter = 0, total = 0;
            const bool measured = cudaMemGetInfo(&freeBefore, &total) == cudaSuccess;
#if CUDART_VERSION >= 12000
            err = cudaGraphInstantiate(&exec, graph, 0);
#else
            err = cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0);
#endif
            const bool measuredAfter = measured && cudaMemGetInfo(&freeAfter, &total) == cudaSuccess;
            const uint64_t charged = measuredAfter && freeBefore > freeAfter ? freeBefore - freeAfter : 0;
            // Not kept when its cost is unknown (a failed memory query) or over budget: the region
            // runs directly (over budget: from now on, the device is closed).
            if (err == cudaSuccess && (!measuredAfter || !cudagraph::chargeGraph(dev, charged))) {
                cudaGraphExecDestroy(exec);
                cudaGraphDestroy(graph);
                return false;
            }
        }
        if (err != cudaSuccess) {
            fprintf(stderr, "[cudaGraph] instantiate failed key=%llx: %s -- key blacklisted\n",
                    (unsigned long long)pending_key_, cudaGetErrorString(err));
            clearCudaError();
            cudaGraphDestroy(graph);
            blacklist_.insert(pending_key_);
            return false;
        }

        err = cudaGraphLaunch(exec, stream);
        if (err != cudaSuccess) {
            fprintf(stderr, "[cudaGraph] first launch failed key=%llx: %s -- key blacklisted\n",
                    (unsigned long long)pending_key_, cudaGetErrorString(err));
            clearCudaError();
            cudaGraphExecDestroy(exec);
            cudaGraphDestroy(graph);
            blacklist_.insert(pending_key_);
            return false;
        }

        cache_[pending_key_] = exec;
        cudaGraphDestroy(graph);
        return true;
    }

    bool isCapturing() const { return capturing_; }

    void clear() {
        for (auto& kv : cache_) {
            cudaGraphExecDestroy(kv.second);
        }
        cache_.clear();
        hitCount_.clear();
        blacklist_.clear();
        capturing_ = false;
        poisoned_ = false;
        clearCudaError();
    }

    size_t size() const { return cache_.size(); }

    static uint64_t makeKey(uint64_t a, uint64_t b = 0, uint64_t c = 0,
                            uint64_t d = 0, uint64_t e = 0, uint64_t f = 0,
                            uint64_t g = 0) {
        uint64_t h = 0xcbf29ce484222325ULL;
        const uint64_t prime = 0x100000001b3ULL;
        auto mix = [&](uint64_t v) {
            h ^= v;
            h *= prime;
        };
        mix(a); mix(b); mix(c); mix(d); mix(e); mix(f); mix(g);
        return h;
    }
};

namespace cudagraph {
    inline uint64_t key(uint64_t a, uint64_t b = 0, uint64_t c = 0, uint64_t d = 0,
                        uint64_t e = 0, uint64_t f = 0, uint64_t g = 0) {
        return CudaGraphCache::makeKey(a, b, c, d, e, f, g);
    }

    // The ONE capture-region wrapper: replay a cached graph for `key`, else capture the
    // body above the threshold, else run it directly. A failed or poisoned capture
    // re-executes the body — safe because stream capture RECORDS work without executing
    // it. `countId` (pinned expression-slot cursor) is snapshotted across the capture
    // attempt: the recording pass advances it host-side, so the re-execution must not
    // consume a second set of slots. Callers without expression launches in the body
    // pass any dummy counter; the restore is then a no-op.
    //
    // Region-body rules (checked in review, not enforceable here):
    //  1. No host writes to shared pinned/host buffers inside the body — replays skip
    //     host code, so staging must happen before the region (or poison the capture,
    //     as stageExpsSlot does).
    //  2. Shape-determinism: every variable that changes WHICH work the body enqueues
    //     must be part of `key`.
    template <typename Body>
    inline void run(uint64_t key, uint64_t &countId, cudaStream_t stream, Body&& body) {
        CudaGraphCache *gc = enabled() ? current() : nullptr;
        if (gc) {
            if (gc->tryLaunch(key, stream)) return;
            if (gc->shouldCapture(key) && captureAllowed() && gc->beginCapture(key, stream)) {
                uint64_t countIdSnap = countId;
                body();
                if (gc->endCaptureAndLaunch(stream)) return;
                countId = countIdSnap;
                body();
                return;
            }
        }
        body();
    }
}

#else // !USE_CUDA_GRAPH

// No-op stubs so `cudagraph::run(...)` region call sites compile unchanged without the
// feature: run() degenerates to a plain body() call. Everything that touches the cache
// itself (current() binding, poison hook) lives under its own USE_CUDA_GRAPH guard at
// the use site, so no stub is needed for it here.
#include <cuda_runtime.h>
#include <cstdint>

namespace cudagraph {
    inline bool enabled() { return false; }
    inline void setBudget(int, uint64_t) {}
    inline void clearBudget(int) {}
    inline uint64_t key(uint64_t, uint64_t = 0, uint64_t = 0, uint64_t = 0,
                        uint64_t = 0, uint64_t = 0, uint64_t = 0) { return 0; }
    template <typename Body>
    inline void run(uint64_t, uint64_t&, cudaStream_t, Body&& body) { body(); }
}

#endif // USE_CUDA_GRAPH

#endif // CUDA_GRAPH_CACHE_CUH
