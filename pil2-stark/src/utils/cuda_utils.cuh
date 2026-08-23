#ifndef __CUDA_UTILS_CUH__
#define __CUDA_UTILS_CUH__

#include <cuda.h>
#include <stdio.h>
#include <assert.h>

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
