#ifndef FRI_EXPRESSION_CUH
#define FRI_EXPRESSION_CUH

// The DEEP/FRI polynomial over the extended domain:
//
//   fri[r] = SUM_o vf1^(O-1-o) / (x[r] - xi_o) * SUM_{j in o} vf2^(n_o-1-j) * (p_j[r] - e_j)
//
// Every opening point is xi_o = xi w^o and x[r] = s wExt^r with w = wExt^B (B the blowup), so
// 1 / (x[r] - xi_o) = w^-o * D[r - o B] with D[q] = 1 / (x[q] - xi). With every per-proof power folded
// into constants (computeFRIFoldedConstants):
//
//   fri[r] = SUM_o D[r - o B] * (SUM_j coef_j * p_j[r] + K_o)
//   coef_j = w^-o vf1^(O-1-o) vf2^(n_o-1-j)        K_o = -SUM_j coef_j * e_j
//
// Self-contained (cubic extension and the POD FriTerm) so the kernels are unit tested against a host
// reference without the SetupCtx translation units.
#include "eval_info.hpp"
#include "goldilocks_cubic_extension.cuh"

// One thread per opening. coef lines up with the terms, K with the openings.
static __global__ void computeFRIFoldedConstants(uint64_t nOpenings, const int64_t *d_openingPoints, uint64_t wTrace,
                                                const uint64_t *d_termStart, const FriTerm *d_terms,
                                                gl64_t *d_evals, gl64_t *vf1, gl64_t *vf2, gl64_t *d_coef,
                                                gl64_t *d_k)
{
    const uint64_t o = blockIdx.x * blockDim.x + threadIdx.x;
    if (o >= nOpenings) return;

    Goldilocks3GPU::Element &vf2e = *(Goldilocks3GPU::Element *)vf2;
    // pow() squares its base in place, so it must never see the shared vf1 buffer.
    Goldilocks3GPU::Element base, c, k, t;
    Goldilocks3GPU::copy(base, *(Goldilocks3GPU::Element *)vf1);
    Goldilocks3GPU::pow(base, nOpenings - 1 - o, c);
    const int64_t op = d_openingPoints[o];
    gl64_t w = gl64_t(wTrace) ^ (uint32_t)(op < 0 ? -op : op);
    if (op > 0) w = w.reciprocal();
    Goldilocks3GPU::mul(c, c, w);
    Goldilocks3GPU::zero(k);

    // Walk the terms downwards so the vf2 exponent runs 0, 1, 2, ...
    for (uint64_t j = d_termStart[o + 1]; j-- > d_termStart[o];) {
        Goldilocks3GPU::copy(*(Goldilocks3GPU::Element *)(d_coef + j * FIELD_EXTENSION), c);
        Goldilocks3GPU::mul(t, c, *(Goldilocks3GPU::Element *)(d_evals + (uint64_t)d_terms[j].evalPos * FIELD_EXTENSION));
        Goldilocks3GPU::sub(k, k, t);
        Goldilocks3GPU::mul(c, c, vf2e);
    }
    Goldilocks3GPU::copy(*(Goldilocks3GPU::Element *)(d_k + o * FIELD_EXTENSION), k);
}

// K_o + SUM_j coef_j * p_j[r] over opening o's terms [tBegin, tEnd), shared by both kernels.
static __device__ __forceinline__ void friOpeningSum(Goldilocks3GPU::Element &accum, uint64_t r, uint64_t domainSize,
                                                     uint64_t tBegin, uint64_t tEnd, const FriTerm *d_terms,
                                                     gl64_t *d_coef, gl64_t *d_kO, const gl64_t *d_cmPols,
                                                     const gl64_t *d_customCommits, const gl64_t *d_fixedPols)
{
    Goldilocks3GPU::copy(accum, *(Goldilocks3GPU::Element *)d_kO);
    Goldilocks3GPU::Element term;
    for (uint64_t t = tBegin; t < tEnd; ++t) {
        const FriTerm m = d_terms[t];
        const gl64_t *pol = (m.src == 0 ? d_cmPols : m.src == 1 ? d_customCommits : d_fixedPols) + m.col + r;
        Goldilocks3GPU::Element &coef = *(Goldilocks3GPU::Element *)(d_coef + t * FIELD_EXTENSION);
        if (m.dim == 1) {
            gl64_t v = pol[0];
            Goldilocks3GPU::mul(term, coef, v);
        } else {
            Goldilocks3GPU::Element v = {pol[0], pol[domainSize], pol[2 * domainSize]};
            Goldilocks3GPU::mul(term, coef, v);
        }
        Goldilocks3GPU::add(accum, accum, term);
    }
}

// One block per blockDim.x rows (grid = domainSize / blockDim.x). The block batch-inverts the window of D
// its rows reach into shared memory (window * 24 bytes, friShiftedWindow), one inversion per row.
static __global__ void computeFRIExpressionShifted(uint64_t domainSize, uint64_t extendBits, uint64_t nOpenings,
                                                   const int64_t *d_openingPoints, int64_t oMax, uint64_t window,
                                                   const uint64_t *d_termStart, const FriTerm *d_terms,
                                                   gl64_t *d_coef, gl64_t *d_k, const gl64_t *d_cmPols,
                                                   const gl64_t *d_customCommits, const gl64_t *d_fixedPols,
                                                   gl64_t *d_xi, const gl64_t *d_x, gl64_t *d_fri)
{
    extern __shared__ Goldilocks3GPU::Element sD[];   // sD[t] = D[q0 + t]
    const uint64_t mask = domainSize - 1;
    const uint64_t r0 = (uint64_t)blockIdx.x * blockDim.x;
    const uint64_t q0 = (r0 - ((uint64_t)oMax << extendBits)) & mask;
    Goldilocks3GPU::Element &xi = *(Goldilocks3GPU::Element *)d_xi;

    // Prefix products over this thread's own entries, in place; one inversion; unwind.
    Goldilocks3GPU::Element den, inv;
    uint64_t last = threadIdx.x;
    Goldilocks3GPU::sub(sD[last], d_x[(q0 + last) & mask], xi);
    for (uint64_t t = last + blockDim.x; t < window; t += blockDim.x) {
        Goldilocks3GPU::sub(den, d_x[(q0 + t) & mask], xi);
        Goldilocks3GPU::mul(sD[t], sD[t - blockDim.x], den);
        last = t;
    }
    Goldilocks3GPU::inv(inv, sD[last]);
    for (uint64_t t = last; t > threadIdx.x; t -= blockDim.x) {
        Goldilocks3GPU::sub(den, d_x[(q0 + t) & mask], xi);
        Goldilocks3GPU::mul(sD[t], inv, sD[t - blockDim.x]);
        Goldilocks3GPU::mul(inv, inv, den);
    }
    Goldilocks3GPU::copy(sD[threadIdx.x], inv);
    __syncthreads();

    const uint64_t r = r0 + threadIdx.x;
    Goldilocks3GPU::Element fri, accum;
    Goldilocks3GPU::zero(fri);
    for (uint64_t o = 0; o < nOpenings; ++o) {
        friOpeningSum(accum, r, domainSize, d_termStart[o], d_termStart[o + 1], d_terms, d_coef,
                      d_k + o * FIELD_EXTENSION, d_cmPols, d_customCommits, d_fixedPols);
        Goldilocks3GPU::mul(accum, accum, sD[threadIdx.x + ((uint64_t)(oMax - d_openingPoints[o]) << extendBits)]);
        Goldilocks3GPU::add(fri, fri, accum);
    }
    Goldilocks3GPU::copy(*(Goldilocks3GPU::Element *)(d_fri + r * FIELD_EXTENSION), fri);
}

// Fallback for opening ranges whose window does not fit (friShiftedWindow == 0): each row batch-inverts
// its own D entries, four openings at a time.
static __global__ void computeFRIExpressionFolded(uint64_t domainSize, uint64_t extendBits, uint64_t nOpenings,
                                                  const int64_t *d_openingPoints, const uint64_t *d_termStart,
                                                  const FriTerm *d_terms, gl64_t *d_coef, gl64_t *d_k,
                                                  const gl64_t *d_cmPols, const gl64_t *d_customCommits,
                                                  const gl64_t *d_fixedPols, gl64_t *d_xi, const gl64_t *d_x,
                                                  gl64_t *d_fri)
{
    const uint64_t mask = domainSize - 1;
    Goldilocks3GPU::Element &xi = *(Goldilocks3GPU::Element *)d_xi;
    for (uint64_t r = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x; r < domainSize; r += (uint64_t)blockDim.x * gridDim.x) {
        Goldilocks3GPU::Element fri, accum, den, inv, D[4];
        Goldilocks3GPU::zero(fri);
        for (uint64_t og = 0; og < nOpenings; og += 4) {
            const uint32_t gn = nOpenings - og < 4 ? (uint32_t)(nOpenings - og) : 4u;
            auto row = [&](uint32_t k) { return (r - ((uint64_t)d_openingPoints[og + k] << extendBits)) & mask; };
            for (uint32_t k = 0; k < gn; ++k) {
                Goldilocks3GPU::sub(den, d_x[row(k)], xi);
                if (k == 0) Goldilocks3GPU::copy(D[0], den);
                else Goldilocks3GPU::mul(D[k], D[k - 1], den);
            }
            Goldilocks3GPU::inv(inv, D[gn - 1]);
            for (uint32_t k = gn - 1; k > 0; --k) {
                Goldilocks3GPU::sub(den, d_x[row(k)], xi);
                Goldilocks3GPU::mul(D[k], inv, D[k - 1]);
                Goldilocks3GPU::mul(inv, inv, den);
            }
            Goldilocks3GPU::copy(D[0], inv);
            for (uint32_t k = 0; k < gn; ++k) {
                const uint64_t o = og + k;
                friOpeningSum(accum, r, domainSize, d_termStart[o], d_termStart[o + 1], d_terms, d_coef,
                              d_k + o * FIELD_EXTENSION, d_cmPols, d_customCommits, d_fixedPols);
                Goldilocks3GPU::mul(accum, accum, D[k]);
                Goldilocks3GPU::add(fri, fri, accum);
            }
        }
        Goldilocks3GPU::copy(*(Goldilocks3GPU::Element *)(d_fri + r * FIELD_EXTENSION), fri);
    }
}

#endif
