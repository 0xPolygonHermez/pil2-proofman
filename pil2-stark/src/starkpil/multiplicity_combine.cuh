#ifndef MULTIPLICITY_COMBINE_CUH
#define MULTIPLICITY_COMBINE_CUH

#include <cuda_runtime.h>
#include <cstdint>

// Block-level combining for the scatter's atomics. Counts pile onto a few hot counters, so
// warps serialise on the same global addresses even after warp aggregation.
//
// A direct-mapped, write-once cache in shared memory keyed by accumulator index: a key claims a
// slot on first sight, later hits in the block add in shared memory, and the block flushes live
// slots to global at the end. A key whose slot is owned by another falls through to the global
// counter, so this is never worse than plain atomics.
//
// How: a direct-mapped, write-once cache in shared memory, keyed by the ACCUMULATOR INDEX. A hot
// key claims a slot on its first appearance and every later hit in this block is a shared-memory
// add; the block flushes each live slot to global once at the end. A key whose slot is already
// owned by another key falls straight through to the global counter -- exactly today's behaviour
// for that key, so this is never worse, only better for whatever is hot.
//
// Nothing here knows about tables, ranges or ids: it keys on the accumulator index alone, so a
// table added tomorrow is combined on the same terms.
#define MUL_COMBINE_SLOTS 1024u
#define MUL_COMBINE_EMPTY 0xFFFFFFFFFFFFFFFFULL

struct MulCombine {
    uint64_t*           key;   // shared, write-once: EMPTY or the accumulator index it owns
    unsigned long long* val;   // shared, the pending count for that index
};

__device__ __forceinline__ void mulCombineInit(MulCombine c) {
    for (uint32_t i = threadIdx.x; i < MUL_COMBINE_SLOTS; i += blockDim.x) {
        c.key[i] = MUL_COMBINE_EMPTY;
        c.val[i] = 0ULL;
    }
    __syncthreads();
}

// The hot counters of one table are contiguous, so the low bits alone would collide; mix first.
__device__ __forceinline__ uint32_t mulCombineSlot(uint64_t key) {
    uint64_t h = key * 0x9E3779B97F4A7C15ULL;
    h ^= h >> 29;
    return (uint32_t)(h & (MUL_COMBINE_SLOTS - 1));
}

// One lane of a matched group deposits the group's total. Slots are write-once, so the plain read
// below is safe: a slot holding our key can never become another's.
__device__ __forceinline__ void mulCombineDeposit(MulCombine c, uint64_t key, uint64_t total,
                                                  unsigned long long* acc) {
    const uint32_t s = mulCombineSlot(key);
    uint64_t prev = c.key[s];
    if (prev != key)
        prev = (uint64_t)atomicCAS((unsigned long long*)&c.key[s], MUL_COMBINE_EMPTY,
                                   (unsigned long long)key);
    if (prev == MUL_COMBINE_EMPTY || prev == key) atomicAdd(&c.val[s], (unsigned long long)total);
    else                                          atomicAdd(&acc[key], (unsigned long long)total);
}

// Warp-aggregate exactly as warp_atomic.cuh does, then deposit once per matched group.
__device__ __forceinline__ void mulCombineAdd(MulCombine c, unsigned mask, uint64_t key,
                                              uint64_t value, unsigned long long* acc) {
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
    if ((int)(threadIdx.x & 31) == leader) mulCombineDeposit(c, key, total, acc);
}

__device__ __forceinline__ void mulCombineInc(MulCombine c, unsigned mask, uint64_t key,
                                              unsigned long long* acc) {
    const unsigned peers = __match_any_sync(mask, key);
    if ((int)(threadIdx.x & 31) == __ffs(peers) - 1)
        mulCombineDeposit(c, key, (uint64_t)__popc(peers), acc);
}

// Every thread of the block must reach this, so the scatter loop is written with a uniform trip
// count rather than letting threads leave early.
__device__ __forceinline__ void mulCombineFlush(MulCombine c, unsigned long long* acc) {
    __syncthreads();
    for (uint32_t i = threadIdx.x; i < MUL_COMBINE_SLOTS; i += blockDim.x) {
        const uint64_t k = c.key[i];
        if (k != MUL_COMBINE_EMPTY && c.val[i] != 0ULL) atomicAdd(&acc[k], c.val[i]);
    }
}

#endif
