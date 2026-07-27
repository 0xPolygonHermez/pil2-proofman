#include "proofman_sumcheck.cuh"
#include "cuda_utils.cuh"

#include <cstdio>
#include <cstdlib>
#include <cstdarg>

// See proofman_sumcheck.cuh for the contract. Implementation isolated here so the
// prover TUs (starks_api.cu / starks_gpu.cu) only carry the include + the terse macro.

thread_local uint64_t g_sumcheck_inst = 0;
thread_local uint64_t g_sumcheck_airgroup = 0;
thread_local uint64_t g_sumcheck_air = 0;

__global__ static void proofman_sumcheck_kernel(const uint64_t *d, uint64_t n, unsigned long long *out) {
    uint64_t idx = blockIdx.x * (uint64_t)blockDim.x + threadIdx.x;
    uint64_t stride = (uint64_t)gridDim.x * blockDim.x;
    unsigned long long local = 0;
    for (uint64_t i = idx; i < n; i += stride) {
        uint64_t h = d[i] + 0x9E3779B97F4A7C15ULL * (i + 1);
        h ^= h >> 33; h *= 0xff51afd7ed558ccdULL; h ^= h >> 33;
        local += (unsigned long long)h;
    }
    atomicAdd(out, local);
}

void proofman_sumcheck_impl(const void *d_ptr, uint64_t n_u64, cudaStream_t stream,
                            uint64_t instanceId, uint64_t airgroupId, uint64_t airId,
                            const char *stage, ...) {
    static int en = -1;
    if (en < 0) { const char *e = getenv("PROOFMAN_SUMCHECK"); en = (e && e[0] == '1') ? 1 : 0; }
    if (!en || n_u64 == 0) return;   // gate before formatting: disabled probes do no work

    char label[64];
    va_list ap;
    va_start(ap, stage);
    vsnprintf(label, sizeof(label), stage, ap);
    va_end(ap);

    unsigned long long *d_out = nullptr;
    if (cudaMalloc(&d_out, sizeof(unsigned long long)) != cudaSuccess) return;
    CHECKCUDAERR(cudaMemsetAsync(d_out, 0, sizeof(unsigned long long), stream));
    proofman_sumcheck_kernel<<<256, 256, 0, stream>>>((const uint64_t *)d_ptr, n_u64, d_out);
    unsigned long long h = 0;
    CHECKCUDAERR(cudaMemcpyAsync(&h, d_out, sizeof(unsigned long long), cudaMemcpyDeviceToHost, stream));
    CHECKCUDAERR(cudaStreamSynchronize(stream));
    cudaFree(d_out);
    printf("[SUMCHECK] inst=%lu airgroup=%lu air=%lu stage=%-22s n=%lu cksum=%016llx\n",
           (unsigned long)instanceId, (unsigned long)airgroupId, (unsigned long)airId, label,
           (unsigned long)n_u64, h);
    fflush(stdout);
}
