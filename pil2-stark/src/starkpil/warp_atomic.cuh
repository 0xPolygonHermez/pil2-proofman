#ifndef WARP_ATOMIC_CUH
#define WARP_ATOMIC_CUH

#include <cuda_runtime.h>
#include <cstdint>

// One atomic per counter per warp instead of per lane: lookup counts cluster on few counters.
// Group sums must stay constant-time (a peer-mask loop gets slower as locality improves), hence the
// two fast paths. Every lane here must be active in `mask`, and `key` must be the counter's global
// index, or lanes counting different tables would merge.
__device__ __forceinline__ void warpAggregatedAdd(unsigned mask, uint64_t key, uint64_t value,
                                                  unsigned long long* counter) {
    const unsigned peers = __match_any_sync(mask, key);
    const int leader = __ffs(peers) - 1;
    uint64_t total;
    const uint64_t lead = __shfl_sync(peers, value, leader);
    if (__all_sync(peers, value == lead)) total = lead * (uint64_t)__popc(peers);
    else {
        total = value;
        for (unsigned rest = peers & ~(1u << (unsigned)leader); rest != 0; rest &= rest - 1)
            total += __shfl_sync(peers, value, __ffs(rest) - 1);
    }
    if ((int)(threadIdx.x & 31) == leader) atomicAdd(counter, (unsigned long long)total);
}

// Constant-value case: no shuffle, the group's contribution is its population count.
__device__ __forceinline__ void warpAggregatedInc(unsigned mask, uint64_t key,
                                                  unsigned long long* counter) {
    const unsigned peers = __match_any_sync(mask, key);
    if ((int)(threadIdx.x & 31) == __ffs(peers) - 1)
        atomicAdd(counter, (unsigned long long)__popc(peers));
}

#endif
