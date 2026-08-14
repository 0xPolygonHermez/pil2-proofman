#ifndef BLAKE3_GOLDILOCKS_CUH
#define BLAKE3_GOLDILOCKS_CUH

// ---------------------------------------------------------------------------
// Blake3GoldilocksGPU -- GPU BLAKE3 hashing over Goldilocks, exposing the merkle
// tree / reduce / grinding entry points used by the GPU prover. Forced arity 2.
// Built on the shared blake3_core so it is bit-identical to Blake3Goldilocks (CPU).
// ---------------------------------------------------------------------------

#include <cstdint>
#include <cuda_runtime.h>
#include "blake3_core.hpp"
#include "poseidon_gpu_common.cuh"  // Layout enum + dynamic-shared `scratchpad`

#include "grinding_launch.hpp"

// Grinding launch geometry for BLAKE3.
#define BLAKE3_GRIND_BITS   19
#define BLAKE3_GRIND_BLOCKS 512
#define BLAKE3_GRIND_GRID \
    ((((1ULL << BLAKE3_GRIND_BITS) + BLAKE3_GRIND_BLOCKS - 1) / BLAKE3_GRIND_BLOCKS))

static_assert(BLAKE3_GRIND_GRID <= GRIND_NONCE_BLOCKS_MAX,
              "BLAKE3 grinding grid exceeds the reserved nonce_blocks region");

class Blake3GoldilocksGPU
{
public:
    static constexpr uint32_t CAPACITY = 4;

    static void merkletree(uint32_t arity, uint64_t *d_tree, uint64_t *d_input,
                           uint64_t num_cols, uint64_t num_rows,
                           Layout layout, cudaStream_t stream);

    static void linearHash(uint64_t *d_hash_output, uint64_t *d_trace,
                           uint64_t num_cols, uint64_t num_rows,
                           Layout layout, cudaStream_t stream);

    static void merkletreeReduce(uint64_t *d_root, uint64_t *d_input,
                                 uint64_t num_elements, uint64_t arity,
                                 cudaStream_t stream);

    static void grinding(uint64_t *d_nonce, uint64_t *d_nonceBlock,
                         const uint64_t *d_in, const uint32_t n_bits,
                         cudaStream_t stream);
};

#endif  // BLAKE3_GOLDILOCKS_CUH
