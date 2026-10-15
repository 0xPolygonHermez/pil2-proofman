#ifndef GRINDING_GPU_CUH
#define GRINDING_GPU_CUH

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cuda_runtime.h>
#include "cuda_utils.cuh"

#define GRIND_TPB 256
#define GRIND_BLOCKS_PER_SM 2   // more starves the youngest warps for the whole grind; 2 already saturates the ALUs

// Warps take 32 * Hash::ROUNDS-nonce chunks in increasing order and stop past the best hit, so every
// nonce below the result is tried: the result is the minimal valid nonce. hash(n) = first output limb.
template <class Hash>
__global__ void __launch_bounds__(GRIND_TPB) grindMinKernel(uint64_t *nonce, uint64_t *counter,
                                                             const uint64_t *__restrict__ input,
                                                             uint64_t level, uint64_t limit)
{
    Hash hash(input);
    const uint32_t lane = threadIdx.x & 31;
    while (true)
    {
        uint64_t base = 0, best = 0;
        if (lane == 0)
        {
            base = atomicAdd((unsigned long long *)counter, 32ull * Hash::ROUNDS);
            asm volatile("ld.relaxed.gpu.global.u64 %0, [%1];" : "=l"(best) : "l"(nonce));
        }
        base = __shfl_sync(0xffffffffu, base, 0);
        best = __shfl_sync(0xffffffffu, best, 0);
        if (base >= best || base >= limit) return;
#pragma unroll 1
        for (uint32_t r = 0; r < Hash::ROUNDS; ++r)
        {
            const uint64_t n = base + 32 * r + lane;
            const uint32_t hit = __ballot_sync(0xffffffffu, n < limit && hash(n) < level);
            if (hit)   // later rounds and chunks only hold larger nonces
            {
                if (lane == 0) atomicMin((unsigned long long *)nonce, (unsigned long long)(base + 32 * r + __ffs(hit) - 1));
                return;
            }
        }
    }
}

// Capturable, allocation-free; d_counter is one word of scratch (the caller's nonce-blocks region).
template <class Hash>
inline void grindMin(uint64_t *d_nonce, uint64_t *d_counter, const uint64_t *d_in, uint32_t n_bits, cudaStream_t stream)
{
    if (n_bits == 0)
    {
        // every nonce passes (and 1 << 64 below is UB)
        CHECKCUDAERR(cudaMemsetAsync(d_nonce, 0, sizeof(uint64_t), stream));
        return;
    }
    // The CPU grinding's bound: a valid nonce exists below it except with probability 2^-128.
    const double eps = std::ldexp(1.0, -int(n_bits));
    const uint64_t limit = (uint64_t)std::ceil(-128.0 * std::log(2.0) / std::log1p(-eps));

    int dev = 0, sms = 0, perSm = 0;
    CHECKCUDAERR(cudaGetDevice(&dev));
    CHECKCUDAERR(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev));
    CHECKCUDAERR(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&perSm, grindMinKernel<Hash>, GRIND_TPB, 0));

    CHECKCUDAERR(cudaMemsetAsync(d_nonce, 0xFF, sizeof(uint64_t), stream));
    CHECKCUDAERR(cudaMemsetAsync(d_counter, 0, sizeof(uint64_t), stream));
    const int blocks = sms * std::max(1, std::min(perSm, GRIND_BLOCKS_PER_SM));
    grindMinKernel<Hash><<<blocks, GRIND_TPB, 0, stream>>>(
        d_nonce, d_counter, d_in, uint64_t(1) << (64 - n_bits), limit);
    CHECKCUDAERR(cudaGetLastError());
}

#endif
