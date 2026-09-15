#ifndef WARP_ATOMIC_CUH
#define WARP_ATOMIC_CUH

#include <cuda_runtime.h>
#include <cstdint>

// One atomic per counter per warp instead of one per lane: lookup counting is heavily clustered,
// so a warp's lanes land on a handful of counters and per-lane adds serialise on them.
//
// Summing a group MUST stay constant-time -- a loop over the peer mask costs one shuffle per peer,
// which makes the kernel slower the better the locality gets. Hence the two fast paths; the loop
// is a last resort for a selector that genuinely differs within a group.
//
// Every lane reaching this must be active in `mask`, and `key` must be the counter's global index,
// or lanes counting different tables would merge.
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
