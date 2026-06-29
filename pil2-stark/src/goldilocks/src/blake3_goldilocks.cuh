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

class Blake3GoldilocksGPU
{
public:
    static constexpr uint32_t CAPACITY = 4;

    // Leaf linear hash + arity-ary reduction. `layout` selects how the input
    // trace rows are addressed (Tiles = column-major-within-tiles, RowMajor).
    static void merkletree(uint32_t arity, uint64_t *d_tree, uint64_t *d_input,
                           uint64_t num_cols, uint64_t num_rows,
                           Layout layout, cudaStream_t stream);

    // Leaf linear hash only (RowMajor/Tiles), digests written row-contiguous.
    static void linearHash(uint64_t *d_hash_output, uint64_t *d_trace,
                           uint64_t num_cols, uint64_t num_rows,
                           Layout layout, cudaStream_t stream);

    // Reduce num_elements pre-hashed digests to a single 4-word root.
    static void merkletreeReduce(uint64_t *d_root, uint64_t *d_input,
                                 uint64_t num_elements, uint64_t arity,
                                 cudaStream_t stream);

    // Proof-of-work over the BLAKE3 transcript permutation (8->8).
    static void grinding(uint64_t *d_nonce, uint64_t *d_nonceBlock,
                         const uint64_t *d_in, const uint32_t n_bits,
                         cudaStream_t stream);
};

#endif  // BLAKE3_GOLDILOCKS_CUH
