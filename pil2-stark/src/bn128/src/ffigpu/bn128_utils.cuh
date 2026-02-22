#ifndef BN128_UTILS_CUH
#define BN128_UTILS_CUH

#include "bn128.cuh"
#include "data_layout.cuh"

#ifndef GOLDILOCKS_PRIME
#define GOLDILOCKS_PRIME 0xFFFFFFFF00000001ULL
#endif

// Reduce a Goldilocks value to canonical form [0, MOD)
// Input must be in range [0, 2^64), outputs [0, MOD)
__device__ __forceinline__ uint64_t gl64_reduce(uint64_t val) {
    return (val >= GOLDILOCKS_PRIME) ? (val - GOLDILOCKS_PRIME) : val;
}

// Load 3 Goldilocks elements into a BN128 Fr element (in registers)
// Supports both row-major (TILED=false) and tiled (TILED=true) layouts
template<bool TILED>
__device__ __forceinline__ void bn128_load_fr_from_gl(
    BN128GPUScalarField::Element &elem,
    const uint64_t *input,
    uint32_t row,
    uint64_t col,
    uint64_t num_rows,
    uint64_t num_cols
) {
    elem[0] = elem[1] = elem[2] = elem[3] = 0;
    elem[4] = elem[5] = elem[6] = elem[7] = 0;

    #pragma unroll
    for (uint32_t k = 0; k < 3; k++) {
        uint64_t col_k = col + k;
        if (col_k < num_cols) {
            uint64_t idx;
            if constexpr (TILED) {
                idx = getBufferOffset(row, col_k, num_rows, num_cols);
            } else {
                idx = row * num_cols + col_k;
            }
            uint64_t gl_val = gl64_reduce(input[idx]);
            elem[k * 2] = (uint32_t)gl_val;
            elem[k * 2 + 1] = (uint32_t)(gl_val >> 32);
        }
    }
}

#endif // BN128_UTILS_CUH
