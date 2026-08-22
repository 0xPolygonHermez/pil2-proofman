#ifndef __CUDA_UTILS_CUH__
#define __CUDA_UTILS_CUH__

#include <cuda.h>
#include <stdio.h>
#include <assert.h>

// Diagnostic hook run once before a fatal CUDA error aborts the process (set by the
// device-buffers owner to dump every stream's air/instance/status): a sticky error
// surfaces at whatever sync runs next, so file:line alone rarely names the culprit.
inline void (*&cudaErrorDumpHook())() {
    static void (*hook)() = nullptr;
    return hook;
}

__host__ inline void checkCudaError(cudaError_t code, const char* expr, const char *file, int line)
{
   if (code != cudaSuccess) {
        fprintf(stderr,
                "[CUDA] %s failed due to: %s (%d) at %s:%d\n",
                expr, cudaGetErrorString(code), static_cast<int>(code), file, line);
        static bool dumping = false;
        if (cudaErrorDumpHook() != nullptr && !dumping) {
            dumping = true;
            cudaErrorDumpHook()();
        }

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
