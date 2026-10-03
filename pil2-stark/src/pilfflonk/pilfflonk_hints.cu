// The kernels of the std's prover hints on the device (pilfflonk_hints_kernels.hpp), in the GPU
// library only, over BN254 scalars in sppark's Montgomery arithmetic (BN128GPUScalarField), which
// keeps every result fully reduced, as ffiasm does: the quotient and the running sums are the same
// field elements as the CPU's (Instance::computeHintColumns), and so the same bytes, whatever the
// order of the operations: an inverse is unique, and so is a sum.
#include <cuda.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstring>

#ifndef FEATURE_BN254
#define FEATURE_BN254
#endif

#include "../bn128/src/ffigpu/fr.cuh"
#include "cuda_utils.cuh"
#include "pilfflonk_hints_kernels.hpp"

namespace {

using Fr = BN128GPUScalarField;
using Element = Fr::Element;

static_assert(sizeof(Element) == sizeof(HintOperand::number), "a number is one element");

constexpr uint32_t THREADS = 256;
// The most blocks of a launch: a kernel strides over what they do not cover.
constexpr uint64_t MAX_BLOCKS = uint64_t(1) << 20;
// The scan's blocks, as the PLONK GPU prover's mulScan*: 256 threads of 4 elements each.
constexpr uint32_t SCAN_THREADS = 256;
constexpr uint32_t SCAN_PER_THREAD = 4;
constexpr uint64_t SCAN_BLOCK = SCAN_THREADS * SCAN_PER_THREAD;

uint32_t blocksFor(uint64_t n) { return static_cast<uint32_t>(std::min((n + THREADS - 1) / THREADS, MAX_BLOCKS)); }

__device__ __forceinline__ uint64_t stride() { return uint64_t(gridDim.x) * blockDim.x; }

__device__ __forceinline__ bool isZero(const Element &a) {
    uint32_t bits = 0;
    for (int limb = 0; limb < 8; ++limb) {
        bits |= a[limb];
    }
    return bits == 0;
}

__device__ __forceinline__ Element operandAt(const HintOperand &op, uint64_t i, uint64_t mask) {
    Element value;
    if (op.values == nullptr) {
        memcpy(&value, op.number, sizeof(value));
    } else {
        value = static_cast<const Element *>(op.values)[(i + op.shift) & mask];
    }
    return value;
}

// Every thread of a block runs every round of the loop, past the end too, on row 0, and reaches the
// reciprocal before it branches: sppark's pairs the lanes of a whole warp. Its inverse of 0 is 0.
__global__ void hintQuotient(Element *dest, uint64_t n, const HintOperand numerator, const HintOperand denominator,
                             unsigned long long *firstZero) {
    const uint64_t mask = n - 1;
    for (uint64_t first = uint64_t(blockIdx.x) * blockDim.x; first < n; first += stride()) {
        const uint64_t i = first + threadIdx.x;
        const bool inside = i < n;
        const uint64_t row = inside ? i : 0;
        const Element d = operandAt(denominator, row, mask);
        const Element inverse = Fr::reciprocal(d);
        if (inside) {
            if (isZero(d)) {
                atomicMin(firstZero, static_cast<unsigned long long>(row));
            }
            dest[i] = Fr::mul(operandAt(numerator, row, mask), inverse);
        }
    }
}

// The running sum of each block of SCAN_BLOCK elements, in place, as mulScanBlockKernel computes the
// running product: each thread the sums of its 4 elements, then a Hillis–Steele scan of the
// threads' totals in shared memory. With blockTotals, the block's total goes to
// blockTotals[blockIdx.x].
__global__ void addScanBlock(Element *data, Element *blockTotals, uint64_t n) {
    __shared__ __align__(alignof(Element)) unsigned char shared[SCAN_THREADS * sizeof(Element)];
    Element *totals = reinterpret_cast<Element *>(shared);
    const uint32_t t = threadIdx.x;
    const uint64_t start = uint64_t(blockIdx.x) * SCAN_BLOCK + t * SCAN_PER_THREAD;
    Element local[SCAN_PER_THREAD];
    for (uint32_t k = 0; k < SCAN_PER_THREAD; ++k) {
        local[k] = start + k < n ? data[start + k] : Fr::zero();
    }
    for (uint32_t k = 1; k < SCAN_PER_THREAD; ++k) {
        local[k] = Fr::add(local[k - 1], local[k]);
    }
    totals[t] = local[SCAN_PER_THREAD - 1];
    __syncthreads();
    for (uint32_t step = 1; step < SCAN_THREADS; step <<= 1) {
        const Element value = t >= step ? Fr::add(totals[t - step], totals[t]) : totals[t];
        __syncthreads();
        totals[t] = value;
        __syncthreads();
    }
    if (t == SCAN_THREADS - 1 && blockTotals != nullptr) {
        blockTotals[blockIdx.x] = totals[t];
    }
    const Element before = t > 0 ? totals[t - 1] : Fr::zero();
    for (uint32_t k = 0; k < SCAN_PER_THREAD; ++k) {
        if (start + k < n) {
            data[start + k] = Fr::add(before, local[k]);
        }
    }
}

// Block b of data adds prefixes[b], the sum of every block before it (data starts at the second).
__global__ void addScanPropagate(Element *data, const Element *prefixes, uint64_t n) {
    const Element prefix = prefixes[blockIdx.x];
    const uint64_t start = uint64_t(blockIdx.x) * SCAN_BLOCK + threadIdx.x * SCAN_PER_THREAD;
    for (uint32_t k = 0; k < SCAN_PER_THREAD; ++k) {
        if (start + k < n) {
            data[start + k] = Fr::add(prefix, data[start + k]);
        }
    }
}

// As mulScanRecursive: the blocks' running sums and their totals, the totals' running sum (one
// level down, in the work after them), and each block but the first plus the sum before it.
void addScan(Element *data, uint64_t n, Element *work) {
    if (n <= 1) {
        return;
    }
    const uint64_t blocks = (n + SCAN_BLOCK - 1) / SCAN_BLOCK;
    if (blocks == 1) {
        addScanBlock<<<1, SCAN_THREADS>>>(data, nullptr, n);
        CHECKCUDAERR(cudaGetLastError());
        return;
    }
    addScanBlock<<<static_cast<uint32_t>(blocks), SCAN_THREADS>>>(data, work, n);
    CHECKCUDAERR(cudaGetLastError());
    addScan(work, blocks, work + blocks);
    addScanPropagate<<<static_cast<uint32_t>(blocks - 1), SCAN_THREADS>>>(data + SCAN_BLOCK, work, n - SCAN_BLOCK);
    CHECKCUDAERR(cudaGetLastError());
}

} // namespace

extern "C" void pilfflonk_gpu_hint_quotient(void *dest, uint64_t n, const HintOperand *numerator,
                                            const HintOperand *denominator, uint64_t *firstZero) {
    static_assert(sizeof(unsigned long long) == sizeof(uint64_t), "atomicMin finds the row in 64 bits");
    CHECKCUDAERR(cudaMemsetAsync(firstZero, 0xff, sizeof(uint64_t), 0));
    if (n == 0) {
        return;
    }
    hintQuotient<<<blocksFor(n), THREADS>>>(static_cast<Element *>(dest), n, *numerator, *denominator,
                                            reinterpret_cast<unsigned long long *>(firstZero));
    CHECKCUDAERR(cudaGetLastError());
}

extern "C" void pilfflonk_gpu_prefix_scan_add(void *data, uint64_t n, void *work) {
    addScan(static_cast<Element *>(data), n, static_cast<Element *>(work));
}

extern "C" uint64_t pilfflonk_gpu_prefix_scan_work_elements(uint64_t n) {
    uint64_t elements = 0;
    while (n > SCAN_BLOCK) {
        n = (n + SCAN_BLOCK - 1) / SCAN_BLOCK;
        elements += n;
    }
    return elements;
}
