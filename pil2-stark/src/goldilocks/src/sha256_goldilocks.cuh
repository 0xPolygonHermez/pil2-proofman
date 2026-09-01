#ifndef SHA256_GOLDILOCKS_CUH
#define SHA256_GOLDILOCKS_CUH

// GPU SHA-256 over Goldilocks: the merkle tree / reduce / grinding entry points
// the GPU prover calls. Forced arity 2, on the shared sha256_core so it is
// bit-identical to the CPU class. Leaves FIPS, nodes domain-separated.

#include <cstdint>
#include <cuda_runtime.h>
#include "sha256_core.hpp"
#include "poseidon_gpu_common.cuh"  // Layout enum + dynamic-shared `scratchpad`

#include "grinding_launch.hpp"

// Same launch geometry as BLAKE3: SHA-256's ~2x per-compression cost lengthens the
// search rather than changing the grid.
#define SHA256_GRIND_BITS   19
#define SHA256_GRIND_BLOCKS 512
#define SHA256_GRIND_GRID \
    ((((1ULL << SHA256_GRIND_BITS) + SHA256_GRIND_BLOCKS - 1) / SHA256_GRIND_BLOCKS))

static_assert(SHA256_GRIND_GRID <= GRIND_NONCE_BLOCKS_MAX,
              "SHA-256 grinding grid exceeds the reserved nonce_blocks region");

class Sha256GoldilocksGPU
{
public:
    static constexpr uint32_t CAPACITY = 4;
    static constexpr uint32_t ARITY    = 2;

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

#endif  // SHA256_GOLDILOCKS_CUH
