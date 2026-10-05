// The kernels of the LDE on the device (pilfflonk_lde_kernels.hpp), in the GPU library only. They
// are elementwise, over BN128 scalars in sppark's Montgomery arithmetic (BN128GPUScalarField), which
// keeps every result fully reduced, as ffiasm does: each value is the same field element as the
// CPU's (pilfflonk_lde.cpp), and so the same bytes, whatever the order of the products.
#include <cuda.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>

#ifndef FEATURE_BN254
#define FEATURE_BN254
#endif

#include "../bn128/src/ffigpu/fr.cuh"
#include "cuda_utils.cuh"
#include "pilfflonk_lde_kernels.hpp"

namespace {

using Fr = BN128GPUScalarField;
using Element = Fr::Element;

constexpr uint32_t THREADS = 256;
// The most blocks of a launch along x: a kernel strides over what they do not cover.
constexpr uint64_t MAX_BLOCKS = uint64_t(1) << 20;
// The columns of one launch of foldByPowers, one per row of blocks, passed by value.
constexpr uint32_t FOLD_COLUMNS = 32;

uint32_t blocksFor(uint64_t n) { return static_cast<uint32_t>(std::min((n + THREADS - 1) / THREADS, MAX_BLOCKS)); }

__device__ __forceinline__ uint64_t firstIndex() { return uint64_t(blockIdx.x) * blockDim.x + threadIdx.x; }

__device__ __forceinline__ uint64_t stride() { return uint64_t(gridDim.x) * blockDim.x; }

__device__ __forceinline__ Element powerOf(const Element *blocks, const Element *powers, uint64_t i) {
    return Fr::mul(blocks[i >> 8], powers[i & 255]);
}

struct FoldColumns {
    const Element *source[FOLD_COLUMNS];
    uint64_t length[FOLD_COLUMNS];
};

// Row y of blocks folds column y; each thread the residues r of its stride, as foldByPowers does on
// the CPU: c^r from the tables, then a factor c^s from each term to the next.
__global__ void foldByPowers(Element *dst, uint64_t s, FoldColumns columns, const Element *blocks,
                             const Element *powers, Element cS) {
    const Element *src = columns.source[blockIdx.y];
    const uint64_t n = columns.length[blockIdx.y];
    Element *out = dst + blockIdx.y * s;
    for (uint64_t r = firstIndex(); r < s; r += stride()) {
        if (r >= n) {
            out[r] = Fr::zero();
            continue;
        }
        Element factor = powerOf(blocks, powers, r);
        Element acc = Fr::mul(src[r], factor);
        for (uint64_t j = r + s; j < n; j += s) {
            factor = Fr::mul(factor, cS);
            acc = Fr::add(acc, Fr::mul(src[j], factor));
        }
        out[r] = acc;
    }
}

__global__ void mulByPowers(Element *data, uint64_t n, const Element *blocks, const Element *powers) {
    for (uint64_t i = firstIndex(); i < n; i += stride()) {
        data[i] = Fr::mul(data[i], powerOf(blocks, powers, i));
    }
}

} // namespace

extern "C" void pilfflonk_gpu_fold_by_powers(void *dst, uint64_t s, const void *const *sources, const uint64_t *lengths,
                                             uint64_t nCols, const void *blocks, const void *powers, const void *cS) {
    if (s == 0) {
        return;
    }
    const Element factor = *static_cast<const Element *>(cS);
    for (uint64_t first = 0; first < nCols; first += FOLD_COLUMNS) {
        const uint64_t count = std::min<uint64_t>(FOLD_COLUMNS, nCols - first);
        FoldColumns columns{};
        for (uint64_t t = 0; t < count; ++t) {
            columns.source[t] = static_cast<const Element *>(sources[first + t]);
            columns.length[t] = lengths[first + t];
        }
        const dim3 grid(blocksFor(s), static_cast<uint32_t>(count));
        foldByPowers<<<grid, THREADS>>>(static_cast<Element *>(dst) + first * s, s, columns,
                                        static_cast<const Element *>(blocks), static_cast<const Element *>(powers),
                                        factor);
        CHECKCUDAERR(cudaGetLastError());
    }
}

extern "C" void pilfflonk_gpu_mul_by_powers(void *data, uint64_t n, const void *blocks, const void *powers) {
    if (n == 0) {
        return;
    }
    mulByPowers<<<blocksFor(n), THREADS>>>(static_cast<Element *>(data), n, static_cast<const Element *>(blocks),
                                           static_cast<const Element *>(powers));
    CHECKCUDAERR(cudaGetLastError());
}
