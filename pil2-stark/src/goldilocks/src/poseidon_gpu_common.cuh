#ifndef POSEIDON_GPU_COMMON_CUH
#define POSEIDON_GPU_COMMON_CUH


#include <cstdint>
#include "gl64_t.cuh"


extern __shared__ gl64_t scratchpad[];


enum class Layout : uint8_t {
    RowMajor,
    Tiles,
};


__device__ __forceinline__ void pow7(gl64_t &x)
{
    gl64_t x2 = x * x;
    gl64_t x3 = x * x2;
    gl64_t x4 = x2 * x2;
    x = x3 * x4;
}

// Warp shuffle of a Goldilocks element: copies the raw 64-bit representation
// from lane `src` with no field reduction (gl64_t operators normalize lazily).
// `mask` must name exactly the participating (active) lanes.
__device__ __forceinline__ gl64_t shfl_gl(uint32_t mask, gl64_t v, int src)
{
    gl64_t r;
    r[0] = (uint64_t)__shfl_sync(mask, (unsigned long long)v[0], src);
    return r;
}

// Sum of one element per lane across lanes 0..W-1, returned to every lane.
// For power-of-two W this is a butterfly all-reduce: log2(W) shuffles instead
// of a linear W-element gather. (W=12 falls back to the linear gather, since an
// XOR butterfly would read lanes >= W that are not in `mask`.) Goldilocks add is
// associative and exact, so the result is bit-identical to the linear order.
// `mask` must name exactly the active lanes (0..W-1).
template<uint32_t W>
__device__ __forceinline__ gl64_t warp_allreduce_add(gl64_t x, uint32_t mask)
{
    if constexpr ((W & (W - 1)) == 0) {   // power of two
#pragma unroll
        for (uint32_t off = W >> 1; off > 0; off >>= 1) {
            gl64_t y;
            y[0] = (uint64_t)__shfl_xor_sync(mask, (unsigned long long)x[0], off);
            x = x + y;
        }
        return x;
    } else {
        gl64_t acc = shfl_gl(mask, x, 0);
#pragma unroll
        for (uint32_t j = 1; j < W; ++j)
            acc = acc + shfl_gl(mask, x, j);
        return acc;
    }
}

#endif
