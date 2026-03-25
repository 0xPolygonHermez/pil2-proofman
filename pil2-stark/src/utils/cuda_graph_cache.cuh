#ifndef CUDA_GRAPH_CACHE_CUH
#define CUDA_GRAPH_CACHE_CUH

#ifdef USE_CUDA_GRAPH

#include <cuda_runtime.h>
#include <unordered_map>
#include <cstdint>
#include <cstdio>

class CudaGraphCache;

namespace cudagraph {
    inline CudaGraphCache*& current() {
        static thread_local CudaGraphCache* ptr = nullptr;
        return ptr;
    }
    inline bool& aggressive() {
        static thread_local bool val = false;
        return val;
    }
}

class CudaGraphCache {
    std::unordered_map<uint64_t, cudaGraphExec_t> cache_;
    std::unordered_map<uint64_t, uint32_t> hitCount_;
    uint64_t pending_key_ = 0;
    bool capturing_ = false;

    static constexpr uint32_t CAPTURE_THRESHOLD = 1000;

    static void clearCudaError() {
        cudaGetLastError();
    }

public:
    CudaGraphCache() = default;

    ~CudaGraphCache() { clear(); }

    CudaGraphCache(const CudaGraphCache&) = delete;
    CudaGraphCache& operator=(const CudaGraphCache&) = delete;

    bool tryLaunch(uint64_t key, cudaStream_t stream) {
        auto it = cache_.find(key);
        if (it == cache_.end()) return false;
        cudaError_t err = cudaGraphLaunch(it->second, stream);
        if (err != cudaSuccess) {
            clearCudaError();
            cudaGraphExecDestroy(it->second);
            cache_.erase(it);
            return false;
        }
        return true;
    }

    bool shouldCapture(uint64_t key) {
        return ++hitCount_[key] >= CAPTURE_THRESHOLD;
    }

    void beginCapture(uint64_t key, cudaStream_t stream) {
        pending_key_ = key;
        capturing_ = true;
        cudaError_t err = cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal);
        if (err != cudaSuccess) {
            clearCudaError();
            capturing_ = false;
        }
    }

    bool endCaptureAndLaunch(cudaStream_t stream) {
        if (!capturing_) return false;
        capturing_ = false;

        cudaGraph_t graph = nullptr;
        cudaError_t err = cudaStreamEndCapture(stream, &graph);
        if (err != cudaSuccess || graph == nullptr) {
            clearCudaError();
            if (graph) cudaGraphDestroy(graph);
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
            cudaGraphDestroy(graph);
            return false;
        }

        err = cudaGraphLaunch(exec, stream);
        if (err != cudaSuccess) {
            clearCudaError();
            cudaGraphExecDestroy(exec);
            cudaGraphDestroy(graph);
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
        capturing_ = false;
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

#endif // USE_CUDA_GRAPH

#endif // CUDA_GRAPH_CACHE_CUH
