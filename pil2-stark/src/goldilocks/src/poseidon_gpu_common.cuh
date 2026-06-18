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

#endif
