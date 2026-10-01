#ifndef PILFFLONK_SHPLONK_HPP
#define PILFFLONK_SHPLONK_HPP

#include <cstdint>

// The kernels of SHPLONK's opening on the device (pilfflonk_shplonk.cu), with C linkage, as those of
// pilfflonk_kernels.hpp, whose conventions they keep: device pointers but those named host ones,
// 32-byte BN254 scalars in Montgomery form, polynomials in increasing degree, 64-bit indices, the
// legacy default stream, nothing launched for an empty range, and a CUDA failure aborting the
// process. Their divisions are the PLONK GPU prover's (gpu_plonk_compute_div_zerofier), and their
// MSMs GpuKey::commit's.
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

} // extern "C"

#endif
