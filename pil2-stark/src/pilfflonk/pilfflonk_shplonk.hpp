#ifndef PILFFLONK_SHPLONK_HPP
#define PILFFLONK_SHPLONK_HPP

#include <cstdint>

// The kernels of SHPLONK's opening on the device (pilfflonk_shplonk.cu), with C linkage, as those of
// pilfflonk_kernels.hpp, whose conventions they keep: device pointers but those named host ones,
// 32-byte BN128 scalars in Montgomery form, polynomials in increasing degree, 64-bit indices, the
// legacy default stream, nothing launched for an empty range, and a CUDA failure aborting the
// process. Their MSMs are GpuKey::commit's.
extern "C" {

// One evaluation of pilfflonk_gpu_evaluate: Σ_{i<n} coefs[i]·x^i, n >= 1, with x^i =
// blocks[i >> 8]·powers[i & 255], the tables gpu_plonk_precompute_omega_tables_async computes with
// base x and block size 256 (blocks[b] = x^(256·b) for b <= (n − 1)/256, powers[t] = x^t).
struct PilfflonkGpuEvaluation {
    const void *coefs;
    uint64_t n;
    const void *blocks;
    const void *powers;
};

// The elements of scratch pilfflonk_gpu_evaluate needs for nEvaluations evaluations.
uint64_t pilfflonk_gpu_evaluation_scratch(uint64_t nEvaluations);

// results[e] is evaluations[e]'s value for e < nEvaluations (evaluations on the device too), with
// `scratch` of pilfflonk_gpu_evaluation_scratch(nEvaluations) elements.
void pilfflonk_gpu_evaluate(void *results, const PilfflonkGpuEvaluation *evaluations, uint64_t nEvaluations,
                            void *scratch);

// Component j of f(X) − r(X) for the packing f(X) = Σ_{j<k} p_j(X^k)·X^j, n coefficients of it into
// out: out[c] = p_j[c] − r[c·k + j] for c < n, with p_j = p, of `length` coefficients (0 above
// them), and r of rLength (0 above).
void pilfflonk_gpu_component_minus(void *out, uint64_t n, const void *p, uint64_t length, const void *r,
                                   uint64_t rLength, uint64_t k, uint64_t j);

// out[c·k + j] += s·q[c] for c < n, s at hostScalar: q as component j of a packing of k, scaled and
// added to out.
void pilfflonk_gpu_add_component(void *out, const void *q, uint64_t n, uint64_t k, uint64_t j,
                                 const void *hostScalar);

// data[i] = s·data[i] for i < n, and then data[0] += c, with s at hostScale and c at hostConstant.
void pilfflonk_gpu_scale_add_constant(void *data, uint64_t n, const void *hostScale, const void *hostConstant);

// data[i] = 1/data[i] for i < n, in place, with n elements of `prefix` as scratch; sets the device
// uint32 at *zero to 1 (and leaves its chunk of data unspecified) if some data[i] is 0.
void pilfflonk_gpu_batch_inverse(void *data, void *prefix, uint64_t n, uint32_t *zero);

// The chunks of pilfflonk_gpu_divide_linear's T and S for n coefficients (S has one more).
uint64_t pilfflonk_gpu_division_chunks(uint64_t n);

// The n coefficients at a divided by Y − β (β at hostBeta), the quotient's n − 1 at q (q ≠ a), in two
// passes over chunks and one block of carries, T and S their scratch; sets the device uint32 at *flag
// to 1 if the remainder is not 0. Writes nothing to q for n = 1.
void pilfflonk_gpu_divide_linear(void *q, const void *a, uint64_t n, const void *hostBeta, void *T, void *S,
                                 uint32_t *flag);

// data[i] = data[i]·other[i] for i < n.
void pilfflonk_gpu_mul_pointwise(void *data, const void *other, uint64_t n);

} // extern "C"

#endif
