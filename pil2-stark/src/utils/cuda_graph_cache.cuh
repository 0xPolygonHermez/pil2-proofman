#ifndef CUDA_GRAPH_CACHE_CUH
#define CUDA_GRAPH_CACHE_CUH

#ifdef USE_CUDA_GRAPH

#include <cuda_runtime.h>
#include "cuda_utils.cuh"
#include <unordered_map>
#include <unordered_set>
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cerrno>
#include <climits>

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
}

// NOT internally synchronized. Thread safety is by ownership: each StreamData owns one
// instance, a stream is claimed under stream_selection_mutex and driven by a single host
// thread until release, and that mutex orders the handoff between successive owners. The
// capture state below (pending_key_, capturing_, poisoned_) is only valid between the
// owner's bind of cudagraph::current() (thread_local) and its guard-scoped unbind. Do not
// touch a stream's cache from CUDA callbacks or any thread that has not reserved the stream.
inline std::atomic<long long> &liveGraphExecs() { return gpuDiag().graphExecs; }
inline void graphExecDelta(int d, size_t cacheSize) {
    long long now = liveGraphExecs().fetch_add(d) + d;
    static std::atomic<long long> peak{0};
    if (d > 0 && now > peak.load() + 32) {
        peak.store(now);
        fprintf(stderr, "[GRAPH-LEAK] live cudaGraphExec = %lld (this stream's cache holds %zu)\n",
                now, cacheSize);
        fflush(stderr);
    }
}

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
            clearCudaError();
            cudaGraphExecDestroy(it->second);
            graphExecDelta(-1, cache_.size());
            cache_.erase(it);
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
        if (err == cudaSuccess && graph != nullptr) gpuDiag().graphs += 1;
        if (err != cudaSuccess || graph == nullptr) {
            clearCudaError();
            if (graph) { cudaGraphDestroy(graph); gpuDiag().graphs -= 1; }
            return false;
        }

        if (poisoned_) {
            poisoned_ = false;
            blacklist_.insert(pending_key_);
            cudaGraphDestroy(graph); gpuDiag().graphs -= 1;
            return false;
        }

        cudaGraphExec_t exec = nullptr;
#if CUDART_VERSION >= 12000
        err = cudaGraphInstantiate(&exec, graph, 0);
#else
        err = cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0);
#endif
        if (err != cudaSuccess) {
            clearCudaError();
            cudaGraphDestroy(graph); gpuDiag().graphs -= 1;
            return false;
        }

        err = cudaGraphLaunch(exec, stream);
        if (err != cudaSuccess) {
            clearCudaError();
            cudaGraphExecDestroy(exec);
            cudaGraphDestroy(graph); gpuDiag().graphs -= 1;
            return false;
        }

        cache_[pending_key_] = exec;
        graphExecDelta(1, cache_.size());
        cudaGraphDestroy(graph); gpuDiag().graphs -= 1;
        return true;
    }

    bool isCapturing() const { return capturing_; }

    void clear() {
        for (auto& kv : cache_) {
            cudaGraphExecDestroy(kv.second);
            graphExecDelta(-1, cache_.size());
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

    // Last capture region this thread entered. A sticky CUDA error (illegal address, invalidated
    // capture) is reported by whichever CHECKCUDAERR runs next, which is routinely in an unrelated
    // file -- the breadcrumb says which region actually produced it.
    inline const char *&phase() { return cudaDiagPhase(); }

    // Checked at every region boundary so a fault is attributed to the region that caused it.
    // Peek, never Get: clearing here would hide the error from the real CHECKCUDAERR.
    inline void checkBoundary(const char *label, const char *when, cudaStream_t stream) {
        cudaError_t err = cudaPeekAtLastError();
        cudaStreamCaptureStatus st = cudaStreamCaptureStatusNone;
        // cudaStreamIsCapturing: stable since CUDA 10.0, unlike cudaStreamGetCaptureInfo_v2.
        bool haveSt = cudaStreamIsCapturing(stream, &st) == cudaSuccess;
        if (err == cudaSuccess && (!haveSt || st != cudaStreamCaptureStatusInvalidated)) return;
        fprintf(stderr,
                "[GRAPH-DIAG] region '%s' %s: err=%s(%d) captureStatus=%d stream=%p prevPhase='%s'\n",
                label, when, cudaGetErrorString(err), (int)err, haveSt ? (int)st : -1,
                (void *)stream, phase());
        fflush(stderr);
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
    inline void run(uint64_t key, uint64_t &countId, cudaStream_t stream, const char *label, Body&& body) {
        checkBoundary(label, "on entry", stream);
        const char *prev = phase();
        phase() = label;
        struct Restore {
            const char *p; const char *l; cudaStream_t s;
            ~Restore() { checkBoundary(l, "on exit", s); phase() = p; }
        } restore{prev, label, stream};
        CudaGraphCache *gc = enabled() ? current() : nullptr;
        if (gc) {
            if (gc->tryLaunch(key, stream)) return;
            if (gc->shouldCapture(key) && gc->beginCapture(key, stream)) {
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
    inline uint64_t key(uint64_t, uint64_t = 0, uint64_t = 0, uint64_t = 0,
                        uint64_t = 0, uint64_t = 0, uint64_t = 0) { return 0; }
    inline const char *&phase() { return cudaDiagPhase(); }
    template <typename Body>
    inline void run(uint64_t, uint64_t&, cudaStream_t, const char *, Body&& body) { body(); }
}

#endif // USE_CUDA_GRAPH

#endif // CUDA_GRAPH_CACHE_CUH
