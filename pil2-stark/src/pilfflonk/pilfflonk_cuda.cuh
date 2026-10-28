#ifndef PILFFLONK_CUDA_CUH
#define PILFFLONK_CUDA_CUH

// What the kernels of pilfflonk's GPU path (pilfflonk_*.cu) share: BN128 scalars in sppark's
// Montgomery arithmetic (BN128GPUScalarField), which keeps every result fully reduced, as ffiasm
// does; and the grid of an elementwise launch, blocks of THREADS threads, at most MAX_BLOCKS of them
// along x, over which a kernel strides. For the .cu files only: each has these definitions to
// itself, in an unnamed namespace.
#include <cuda.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>

#ifndef FEATURE_BN254
#define FEATURE_BN254
#endif

#include "../bn128/src/ffigpu/fr.cuh"
#include "cuda_utils.cuh"

namespace {

using Fr = BN128GPUScalarField;
using Element = Fr::Element;

constexpr uint32_t THREADS = 256;
// The most blocks of a launch along x: a kernel strides over what they do not cover.
constexpr uint64_t MAX_BLOCKS = uint64_t(1) << 20;

uint32_t blocksFor(uint64_t n) { return static_cast<uint32_t>(std::min((n + THREADS - 1) / THREADS, MAX_BLOCKS)); }

__device__ __forceinline__ uint64_t firstIndex() { return uint64_t(blockIdx.x) * blockDim.x + threadIdx.x; }

__device__ __forceinline__ uint64_t stride() { return uint64_t(gridDim.x) * blockDim.x; }

__device__ __forceinline__ bool isZero(const Element &a) {
    uint32_t bits = 0;
    for (int limb = 0; limb < 8; ++limb) {
        bits |= a[limb];
    }
    return bits == 0;
}

} // namespace

#endif
