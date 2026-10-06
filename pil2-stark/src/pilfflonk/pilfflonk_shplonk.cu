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

// Elements per thread of batchInverse: one inversion each.
constexpr uint64_t INVERSE_CHUNK = 16;

// Montgomery's trick on each thread's chunk, prefixes in `prefix`; *zero = 1 if some value is 0. Every
// thread of a block runs every round: sppark's reciprocal pairs the lanes of a warp.
__global__ void batchInverse(Element *data, Element *prefix, uint64_t n, uint32_t *zero) {
    for (uint64_t first = uint64_t(blockIdx.x) * blockDim.x * INVERSE_CHUNK; first < n;
         first += stride() * INVERSE_CHUNK) {
        const uint64_t start = first + threadIdx.x * INVERSE_CHUNK, end = min(start + INVERSE_CHUNK, n);
        Element acc = Fr::one();
        for (uint64_t i = start; i < end; ++i) {
            prefix[i] = acc;
            acc = Fr::mul(acc, data[i]);
        }
        const bool vanishes = isZero(acc);
        if (vanishes) {
            *zero = 1;
        }
        Element inv = Fr::reciprocal(vanishes ? Fr::one() : acc);
        for (uint64_t i = vanishes ? start : end; i > start; --i) {
            const Element value = data[i - 1];
            data[i - 1] = Fr::mul(inv, prefix[i - 1]);
            inv = Fr::mul(inv, value);
        }
    }
}

// Coefficients per chunk of the division by Y − β, and the threads of its carries' one block.
constexpr uint64_t DIVISION_CHUNK = 128;
constexpr uint32_t CARRY_THREADS = 256;

// T[b] = Σ_{d in chunk b} a_d·β^(d − b·L).
__global__ void chunkSums(Element *T, const Element *a, uint64_t n, uint64_t nChunks, const Element beta) {
    for (uint64_t b = firstIndex(); b < nChunks; b += stride()) {
        const uint64_t lo = b * DIVISION_CHUNK, hi = min(lo + DIVISION_CHUNK, n);
        Element v = Fr::zero();
        for (uint64_t d = hi; d > lo; --d) {
            v = Fr::add(a[d - 1], Fr::mul(beta, v));
        }
        T[b] = v;
    }
}

// S[b] = Σ_{b' >= b} T[b']·γ^(b' − b), S[nChunks] = 0, γ = β^L: thread j's `per` chunks, and
// across threads a suffix scan with Γ = γ^per.
__global__ void chunkCarries(Element *S, const Element *T, uint64_t nChunks, uint32_t per, const Element beta) {
    const Element gamma = Fr::pow(beta, static_cast<uint32_t>(DIVISION_CHUNK)), Gamma = Fr::pow(gamma, per);
    __shared__ __align__(alignof(Element)) unsigned char shared[CARRY_THREADS * sizeof(Element)];
    Element *P = reinterpret_cast<Element *>(shared);
    const uint32_t j = threadIdx.x;
    const uint64_t lo = min(uint64_t(j) * per, nChunks), hi = min(lo + per, nChunks);
    Element u = Fr::zero();
    for (uint64_t b = hi; b > lo; --b) {
        u = Fr::add(T[b - 1], Fr::mul(gamma, u));
    }
    P[j] = u;
    __syncthreads();
    Element g = Gamma;
    for (uint32_t s = 1; s < CARRY_THREADS; s <<= 1) {
        const Element above = j + s < CARRY_THREADS ? Fr::mul(g, P[j + s]) : Fr::zero();
        __syncthreads();
        P[j] = Fr::add(P[j], above);
        __syncthreads();
        g = Fr::square(g);
    }
    Element v = j + 1 < CARRY_THREADS ? P[j + 1] : Fr::zero();
    for (uint64_t b = hi; b > lo; --b) {
        v = Fr::add(T[b - 1], Fr::mul(gamma, v));
        S[b - 1] = v;
    }
    if (j == 0) {
        S[nChunks] = Fr::zero();
    }
}

// q_{c−1} = a_c + β·q_c down each chunk from q_{hi−1} = S[b + 1]; *flag = 1 if a_0 + β·q_0 is not 0.
__global__ void chunkQuotients(Element *q, const Element *a, const Element *S, uint64_t n, uint64_t nChunks,
                               const Element beta, uint32_t *flag) {
    for (uint64_t b = firstIndex(); b < nChunks; b += stride()) {
        const uint64_t lo = b * DIVISION_CHUNK, hi = min(lo + DIVISION_CHUNK, n);
        Element v = S[b + 1];
        for (uint64_t c = hi; c > lo; --c) {
            v = Fr::add(a[c - 1], Fr::mul(beta, v));
            if (c > 1) {
                q[c - 2] = v;
            } else if (!isZero(v)) {
                *flag = 1;
            }
        }
    }
}

__global__ void mulPointwise(Element *data, const Element *other, uint64_t n) {
    for (uint64_t i = firstIndex(); i < n; i += stride()) {
        data[i] = Fr::mul(data[i], other[i]);
    }
}

} // namespace

extern "C" void pilfflonk_gpu_batch_inverse(void *data, void *prefix, uint64_t n, uint32_t *zero) {
    if (n == 0) {
        return;
    }
    const uint64_t threads = (n + INVERSE_CHUNK - 1) / INVERSE_CHUNK;
    batchInverse<<<blocksFor(threads), THREADS>>>(static_cast<Element *>(data), static_cast<Element *>(prefix), n,
                                                  zero);
    CHECKCUDAERR(cudaGetLastError());
}

extern "C" uint64_t pilfflonk_gpu_division_chunks(uint64_t n) { return (n + DIVISION_CHUNK - 1) / DIVISION_CHUNK; }

extern "C" void pilfflonk_gpu_divide_linear(void *q, const void *a, uint64_t n, const void *hostBeta, void *T,
                                            void *S, uint32_t *flag) {
    if (n == 0) {
        return;
    }
    const uint64_t nChunks = pilfflonk_gpu_division_chunks(n), per = (nChunks + CARRY_THREADS - 1) / CARRY_THREADS;
    const Element beta = *static_cast<const Element *>(hostBeta);
    Element *t = static_cast<Element *>(T), *s = static_cast<Element *>(S);
    chunkSums<<<blocksFor(nChunks), THREADS>>>(t, static_cast<const Element *>(a), n, nChunks, beta);
    CHECKCUDAERR(cudaGetLastError());
    chunkCarries<<<1, CARRY_THREADS>>>(s, t, nChunks, static_cast<uint32_t>(per), beta);
    CHECKCUDAERR(cudaGetLastError());
    chunkQuotients<<<blocksFor(nChunks), THREADS>>>(static_cast<Element *>(q), static_cast<const Element *>(a), s, n,
                                                    nChunks, beta, flag);
    CHECKCUDAERR(cudaGetLastError());
}

extern "C" void pilfflonk_gpu_mul_pointwise(void *data, const void *other, uint64_t n) {
    if (n == 0) {
        return;
    }
    mulPointwise<<<blocksFor(n), THREADS>>>(static_cast<Element *>(data), static_cast<const Element *>(other), n);
    CHECKCUDAERR(cudaGetLastError());
}

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
