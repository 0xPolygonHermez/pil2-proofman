#ifndef __CUDA_UTILS_CUH__
#define __CUDA_UTILS_CUH__

#include <cuda.h>
#include <stdio.h>
#include <assert.h>
#include <cuda_runtime.h>
#include <unordered_map>
#include <mutex>
#include <atomic>

// ---------------------------------------------------------------------------------------------
// Device-resource census. A slow device-memory leak is invisible in aggregate: cudaMemGetInfo says
// memory is going but not what took it. These counters track every resource class the prover
// allocates, with EXTERNAL linkage (plain `inline`, so the function-local static is one instance
// program-wide -- `static inline` would give each translation unit its own copy and creates would
// be counted separately from destroys). Printed as one greppable line: [GPU-CENSUS].
//
// Read it by elimination: whichever counter tracks the fall in `free` is the leak. If they are ALL
// flat while `free` falls, the allocation is not coming from this library.
// ---------------------------------------------------------------------------------------------
struct GpuDiagCounters {
    std::atomic<long long> events{0};       // live cudaEvent_t
    std::atomic<long long> streams{0};      // live cudaStream_t
    std::atomic<long long> graphs{0};       // live cudaGraph_t
    std::atomic<long long> graphExecs{0};   // live cudaGraphExec_t
    std::atomic<long long> devBytes{0};     // live bytes from tracked cudaMalloc
    std::atomic<long long> devBlocks{0};    // live tracked cudaMalloc blocks
    std::atomic<long long> hostBytes{0};    // live bytes from tracked cudaMallocHost
    std::atomic<long long> proofs{0};       // harvested proofs, for a per-proof rate
};

inline GpuDiagCounters &gpuDiag()
{
    static GpuDiagCounters c;
    return c;
}

// Sizes of tracked device allocations, so a free can subtract the right amount.
inline std::mutex &gpuDiagMapMutex()
{
    static std::mutex m;
    return m;
}

inline std::unordered_map<void *, size_t> &gpuDiagSizes()
{
    static std::unordered_map<void *, size_t> m;
    return m;
}

template <typename T>
inline cudaError_t diagCudaMalloc(T **p, size_t n)
{
    cudaError_t e = cudaMalloc(p, n);
    if (e == cudaSuccess && *p != nullptr) {
        std::lock_guard<std::mutex> lk(gpuDiagMapMutex());
        gpuDiagSizes()[(void *)*p] = n;
        gpuDiag().devBytes += (long long)n;
        gpuDiag().devBlocks += 1;
    }
    return e;
}

inline cudaError_t diagCudaFree(void *p)
{
    if (p != nullptr) {
        std::lock_guard<std::mutex> lk(gpuDiagMapMutex());
        auto it = gpuDiagSizes().find(p);
        if (it != gpuDiagSizes().end()) {
            gpuDiag().devBytes -= (long long)it->second;
            gpuDiag().devBlocks -= 1;
            gpuDiagSizes().erase(it);
        }
    }
    return cudaFree(p);
}

template <typename T>
inline cudaError_t diagCudaMallocHost(T **p, size_t n)
{
    cudaError_t e = cudaMallocHost(p, n);
    if (e == cudaSuccess) gpuDiag().hostBytes += (long long)n;
    return e;
}

inline cudaError_t diagCudaStreamCreate(cudaStream_t *s)
{
    cudaError_t e = cudaStreamCreate(s);
    if (e == cudaSuccess) gpuDiag().streams += 1;
    return e;
}

inline cudaError_t diagCudaStreamCreateWithPriority(cudaStream_t *s, unsigned int flags, int priority)
{
    cudaError_t e = cudaStreamCreateWithPriority(s, flags, priority);
    if (e == cudaSuccess) gpuDiag().streams += 1;
    return e;
}

inline cudaError_t diagCudaStreamDestroy(cudaStream_t s)
{
    gpuDiag().streams -= 1;
    return cudaStreamDestroy(s);
}

// One line with everything. `where` names the call site so interleaved reports stay readable.
inline void gpuDiagCensus(const char *where)
{
    size_t freeB = 0, totalB = 0;
    if (cudaMemGetInfo(&freeB, &totalB) != cudaSuccess) { cudaGetLastError(); return; }
    unsigned long long poolReserved = 0;
#if CUDART_VERSION >= 11020
    cudaMemPool_t pool = nullptr;
    int dev = 0;
    if (cudaGetDevice(&dev) == cudaSuccess && cudaDeviceGetDefaultMemPool(&pool, dev) == cudaSuccess)
        cudaMemPoolGetAttribute(pool, cudaMemPoolAttrReservedMemCurrent, &poolReserved);
    cudaGetLastError();
#endif
    GpuDiagCounters &c = gpuDiag();
    fprintf(stderr,
            "[GPU-CENSUS] at=%s proofs=%lld freeMB=%.1f poolMB=%.1f events=%lld streams=%lld "
            "graphs=%lld graphExecs=%lld devMB=%.1f devBlocks=%lld pinnedMB=%.1f\n",
            where, (long long)c.proofs.load(), freeB / 1048576.0, poolReserved / 1048576.0,
            (long long)c.events.load(), (long long)c.streams.load(), (long long)c.graphs.load(),
            (long long)c.graphExecs.load(), c.devBytes.load() / 1048576.0,
            (long long)c.devBlocks.load(), c.hostBytes.load() / 1048576.0);
    fflush(stderr);
}

// Fault attribution. A CUDA error is sticky and asynchronous: it is reported by whichever
// CHECKCUDAERR runs next, routinely in an unrelated file, so the abort site alone does not say what
// produced it. These two breadcrumbs are kept current by the prover -- the capture region
// (cudagraph::run) and the launch identity (gen_*_proof_gpu) -- and printed on abort.
__host__ inline const char *&cudaDiagPhase()
{
    static thread_local const char *p = "none";
    return p;
}

__host__ inline char *cudaDiagContext()
{
    static thread_local char buf[256] = "none";
    return buf;
}

__host__ inline void checkCudaError(cudaError_t code, const char* expr, const char *file, int line)
{
   if (code != cudaSuccess) {
        fprintf(stderr,
                "[CUDA] %s failed due to: %s (%d) at %s:%d\n",
                expr, cudaGetErrorString(code), static_cast<int>(code), file, line);
        fprintf(stderr, "[CUDA] WHERE: region='%s' launch=%s\n", cudaDiagPhase(), cudaDiagContext());

        // Also report the last sticky error (useful after kernel launches)
        const cudaError_t last = cudaGetLastError();
        if (last != cudaSuccess && last != code) {
            fprintf(stderr,
                    "[CUDA] sticky last error: %s (%d)\n",
                    cudaGetErrorString(last), static_cast<int>(last));
        }
        fflush(stderr);
        std::abort(); // don't use assert(0) here
    }
}
#define CHECKCUDAERR(ans) checkCudaError((ans), #ans, __FILE__, __LINE__)

__device__ __forceinline__ void mymemcpy(uint64_t* dst, uint64_t* src, size_t n)
{
    for (uint32_t i = 0; i < n; i++)
    {
        dst[i] = src[i];
    }
}

__device__ __forceinline__ void mymemset(uint64_t* dst, uint64_t v, size_t n)
{
    for (uint32_t i = 0; i < n; i++)
    {
        dst[i] = v;
    }
}

#endif  // __CUDA_UTILS_CUH__
