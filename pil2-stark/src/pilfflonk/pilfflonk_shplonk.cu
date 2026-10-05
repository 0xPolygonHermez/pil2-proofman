// The kernels of SHPLONK's opening on the device (pilfflonk_shplonk.hpp), in the GPU library only,
// over BN128 scalars in sppark's Montgomery arithmetic (BN128GPUScalarField), which keeps every
// result fully reduced, as ffiasm does. Their sums are in another order than the CPU's, which in an
// exact field gives the same element, and so the same bytes.
#include "pilfflonk_cuda.cuh"
#include "pilfflonk_shplonk.hpp"

namespace {

// The blocks of each evaluation of pilfflonk_gpu_evaluate, whose partial sums its scratch holds.
constexpr uint32_t EVALUATION_BLOCKS = 128;
// The most evaluations a launch covers at once, one per row of blocks.
constexpr uint64_t MAX_ROWS = 65535;

uint32_t rowsFor(uint64_t n) { return static_cast<uint32_t>(std::min(n, MAX_ROWS)); }

// The sum of the THREADS values of `sums` in sums[0]. Every thread of the block reaches each barrier.
__device__ void blockSum(Element *sums) {
    __syncthreads();
    for (uint32_t s = THREADS / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            sums[threadIdx.x] = Fr::add(sums[threadIdx.x], sums[threadIdx.x + s]);
        }
        __syncthreads();
    }
}

// Each row of blocks one evaluation at a time; each block the sum of its strided terms, into
// partials[e·EVALUATION_BLOCKS + blockIdx.x].
__global__ void evaluateTerms(Element *partials, const PilfflonkGpuEvaluation *evaluations, uint64_t nEvaluations) {
    __shared__ __align__(alignof(Element)) unsigned char shared[THREADS * sizeof(Element)];
    Element *sums = reinterpret_cast<Element *>(shared);
    for (uint64_t e = blockIdx.y; e < nEvaluations; e += gridDim.y) {
        const PilfflonkGpuEvaluation evaluation = evaluations[e];
        const Element *coefs = static_cast<const Element *>(evaluation.coefs);
        const Element *blocks = static_cast<const Element *>(evaluation.blocks);
        const Element *powers = static_cast<const Element *>(evaluation.powers);
        Element sum = Fr::zero();
        for (uint64_t i = firstIndex(); i < evaluation.n; i += stride()) {
            sum = Fr::add(sum, Fr::mul(coefs[i], Fr::mul(blocks[i >> 8], powers[i & 255])));
        }
        sums[threadIdx.x] = sum;
        blockSum(sums);
        if (threadIdx.x == 0) {
            partials[e * EVALUATION_BLOCKS + blockIdx.x] = sums[0];
        }
        // sums[0] is read before the next evaluation writes sums.
        __syncthreads();
    }
}

// Each block one evaluation at a time: the sum of its EVALUATION_BLOCKS partial sums.
__global__ void sumTerms(Element *results, const Element *partials, uint64_t nEvaluations) {
    __shared__ __align__(alignof(Element)) unsigned char shared[THREADS * sizeof(Element)];
    Element *sums = reinterpret_cast<Element *>(shared);
    for (uint64_t e = blockIdx.x; e < nEvaluations; e += gridDim.x) {
        Element sum = Fr::zero();
        for (uint32_t t = threadIdx.x; t < EVALUATION_BLOCKS; t += blockDim.x) {
            sum = Fr::add(sum, partials[e * EVALUATION_BLOCKS + t]);
        }
        sums[threadIdx.x] = sum;
        blockSum(sums);
        if (threadIdx.x == 0) {
            results[e] = sums[0];
        }
        __syncthreads();
    }
}

__global__ void componentMinus(Element *out, uint64_t n, const Element *p, uint64_t length, const Element *r,
                               uint64_t rLength, uint64_t k, uint64_t j) {
    for (uint64_t c = firstIndex(); c < n; c += stride()) {
        Element value = c < length ? p[c] : Fr::zero();
        const uint64_t at = c * k + j;
        if (at < rLength) {
            value = Fr::sub(value, r[at]);
        }
        out[c] = value;
    }
}

__global__ void addComponent(Element *out, const Element *q, uint64_t n, uint64_t k, uint64_t j, Element s) {
    for (uint64_t c = firstIndex(); c < n; c += stride()) {
        Element &target = out[c * k + j];
        target = Fr::add(target, Fr::mul(s, q[c]));
    }
}

__global__ void scaleAddConstant(Element *data, uint64_t n, Element s, Element constant) {
    for (uint64_t i = firstIndex(); i < n; i += stride()) {
        Element value = Fr::mul(s, data[i]);
        if (i == 0) {
            value = Fr::add(value, constant);
        }
        data[i] = value;
    }
}

} // namespace

extern "C" uint64_t pilfflonk_gpu_evaluation_scratch(uint64_t nEvaluations) {
    return nEvaluations * EVALUATION_BLOCKS;
}

extern "C" void pilfflonk_gpu_evaluate(void *results, const PilfflonkGpuEvaluation *evaluations, uint64_t nEvaluations,
                                       void *scratch) {
    if (nEvaluations == 0) {
        return;
    }
    Element *partials = static_cast<Element *>(scratch);
    evaluateTerms<<<dim3(EVALUATION_BLOCKS, rowsFor(nEvaluations)), THREADS>>>(partials, evaluations, nEvaluations);
    CHECKCUDAERR(cudaGetLastError());
    sumTerms<<<rowsFor(nEvaluations), THREADS>>>(static_cast<Element *>(results), partials, nEvaluations);
    CHECKCUDAERR(cudaGetLastError());
}

extern "C" void pilfflonk_gpu_component_minus(void *out, uint64_t n, const void *p, uint64_t length, const void *r,
                                              uint64_t rLength, uint64_t k, uint64_t j) {
    if (n == 0) {
        return;
    }
    componentMinus<<<blocksFor(n), THREADS>>>(static_cast<Element *>(out), n, static_cast<const Element *>(p), length,
                                              static_cast<const Element *>(r), rLength, k, j);
    CHECKCUDAERR(cudaGetLastError());
}

extern "C" void pilfflonk_gpu_add_component(void *out, const void *q, uint64_t n, uint64_t k, uint64_t j,
                                            const void *hostScalar) {
    if (n == 0) {
        return;
    }
    addComponent<<<blocksFor(n), THREADS>>>(static_cast<Element *>(out), static_cast<const Element *>(q), n, k, j,
                                            *static_cast<const Element *>(hostScalar));
    CHECKCUDAERR(cudaGetLastError());
}

extern "C" void pilfflonk_gpu_scale_add_constant(void *data, uint64_t n, const void *hostScale,
                                                 const void *hostConstant) {
    if (n == 0) {
        return;
    }
    scaleAddConstant<<<blocksFor(n), THREADS>>>(static_cast<Element *>(data), n,
                                                *static_cast<const Element *>(hostScale),
                                                *static_cast<const Element *>(hostConstant));
    CHECKCUDAERR(cudaGetLastError());
}
