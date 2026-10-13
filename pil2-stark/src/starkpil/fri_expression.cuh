#ifndef FRI_EXPRESSION_CUH
#define FRI_EXPRESSION_CUH

// The DEEP/FRI polynomial over the extended domain:
//
//   fri[r] = SUM_o vf1^(O-1-o) / (x[r] - xi_o) * SUM_{c in o} vf2^c * (p_c[r] - e_(o,c))
//
// With xi_o = xi w^o, x[r] = s wExt^r and w = wExt^B (B the blowup), 1 / (x[r] - xi_o) = w^-o D[r - o B] for
// D[q] = 1 / (x[q] - xi). Grouping the polynomials G by the set of openings they are in:
//
//   fri[r] = SUM_o D[r - o B] a_o (SUM_{G with o} S_G[r] + K_o)      S_G[r] = SUM_{c in G} b_c p_c[r]
//   b_c = vf2^c     a_o = w^-o vf1^e_o     K_o = -SUM_{c in o} vf2^c e_(o,c)
//
// e_o counts the later openings that have evaluations, as the setup's Horner on vf1 (fri_poly.rs).
//
// D is one inversion per row of the domain, computed once (computeFRIDenominators) and read by every opening
// at its own offset: the reads of a warp are contiguous, and the offsets of one row span a few KiB, so they
// come from L1/L2 rather than being recomputed per block.
//
// Self-contained (cubic extension and the POD FriTerm) so the kernels are unit tested against a host
// reference without the SetupCtx translation units.
#include "eval_info.hpp"
#include "goldilocks_cubic_extension.cuh"

// b_c for every polynomial and a_o for every opening, one thread each.
static __global__ void computeFRIPowers(uint64_t nOpenings, uint64_t nPols, const int64_t *d_openingPoints, uint64_t wTrace,
                                        const uint64_t *d_termStart, gl64_t *vf1, gl64_t *vf2, gl64_t *d_b, gl64_t *d_a)
{
    const uint64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    // pow() squares its base in place, so it must never see the shared challenge buffers.
    Goldilocks3GPU::Element base;
    if (i < nPols) {
        Goldilocks3GPU::copy(base, *(Goldilocks3GPU::Element *)vf2);
        Goldilocks3GPU::pow(base, i, *(Goldilocks3GPU::Element *)(d_b + i * FIELD_EXTENSION));
    }
    if (i >= nOpenings) return;
    Goldilocks3GPU::Element &a = *(Goldilocks3GPU::Element *)(d_a + i * FIELD_EXTENSION);
    uint64_t e = 0;
    for (uint64_t o = i + 1; o < nOpenings; ++o) e += d_termStart[o + 1] > d_termStart[o];
    Goldilocks3GPU::copy(base, *(Goldilocks3GPU::Element *)vf1);
    Goldilocks3GPU::pow(base, e, a);
    const int64_t op = d_openingPoints[i];
    gl64_t w = gl64_t(wTrace) ^ (uint32_t)(op < 0 ? -op : op);
    if (op > 0) w = w.reciprocal();
    Goldilocks3GPU::mul(a, a, w);
}

// K_o for every opening, one thread each.
static __global__ void computeFRIK(uint64_t nOpenings, const uint64_t *d_termStart, const FriTerm *d_terms,
                                   gl64_t *d_evals, gl64_t *d_b, gl64_t *d_k)
{
    const uint64_t o = blockIdx.x * blockDim.x + threadIdx.x;
    if (o >= nOpenings) return;
    Goldilocks3GPU::Element k, t;
    Goldilocks3GPU::zero(k);
    for (uint64_t j = d_termStart[o]; j < d_termStart[o + 1]; ++j) {
        Goldilocks3GPU::mul(t, *(Goldilocks3GPU::Element *)(d_b + (uint64_t)d_terms[j].vf2Exp * FIELD_EXTENSION),
                            *(Goldilocks3GPU::Element *)(d_evals + (uint64_t)d_terms[j].evalPos * FIELD_EXTENSION));
        Goldilocks3GPU::sub(k, k, t);
    }
    Goldilocks3GPU::copy(*(Goldilocks3GPU::Element *)(d_k + o * FIELD_EXTENSION), k);
}

// The per-proof constants.
static void computeFRIConstants(uint64_t nOpenings, uint64_t nPols, const int64_t *d_openingPoints, uint64_t wTrace,
                                const uint64_t *d_termStart, const FriTerm *d_terms, gl64_t *d_evals, gl64_t *vf1,
                                gl64_t *vf2, gl64_t *d_b, gl64_t *d_a, gl64_t *d_k, cudaStream_t stream = 0)
{
    const uint64_t n = nOpenings > nPols ? nOpenings : nPols;
    computeFRIPowers<<<(n + 63) / 64, 64, 0, stream>>>(nOpenings, nPols, d_openingPoints, wTrace, d_termStart, vf1, vf2, d_b, d_a);
    computeFRIK<<<(nOpenings + 63) / 64, 64, 0, stream>>>(nOpenings, d_termStart, d_terms, d_evals, d_b, d_k);
}

// D[q] = 1 / (x[q] - xi) over the whole domain, as three planes of domainSize (D0 | D1 | D2) so the expression
// kernel's reads coalesce. A thread owns FRI_DEN_ROWS rows blockDim apart: prefix products of their
// denominators, one inversion, unwind (Montgomery's trick). The rows' x are walked back with wStepInv rather
// than kept, which halves the live registers. wStep = wExt^blockDim. Rows past the domain (a block's tail on a
// domain smaller than its rows) are computed but not written: x wraps, so their denominators stay nonzero.
#define FRI_DEN_ROWS 16
static __global__ void computeFRIDenominators(uint64_t domainSize, uint64_t shift, uint64_t wExt, uint64_t wStep,
                                              uint64_t wStepInv, gl64_t *d_xi, gl64_t *d_den)
{
    const uint32_t tid = threadIdx.x, nT = blockDim.x;
    const uint64_t r0 = (uint64_t)blockIdx.x * nT * FRI_DEN_ROWS + tid;
    Goldilocks3GPU::Element &xi = *(Goldilocks3GPU::Element *)d_xi;
    gl64_t x = gl64_t(shift) * (gl64_t(wExt) ^ (uint32_t)r0);
    Goldilocks3GPU::Element pre[FRI_DEN_ROWS], den, inv;
#pragma unroll
    for (uint32_t k = 0; k < FRI_DEN_ROWS; ++k, x = x * gl64_t(wStep)) {
        Goldilocks3GPU::sub(den, x, xi);
        if (k == 0) Goldilocks3GPU::copy(pre[0], den);
        else Goldilocks3GPU::mul(pre[k], pre[k - 1], den);
    }
    Goldilocks3GPU::inv(inv, pre[FRI_DEN_ROWS - 1]);
#pragma unroll
    for (uint32_t k = FRI_DEN_ROWS; k-- > 0;) {
        x = x * gl64_t(wStepInv);
        Goldilocks3GPU::Element d;
        if (k == 0) {
            Goldilocks3GPU::copy(d, inv);
        } else {
            Goldilocks3GPU::mul(d, inv, pre[k - 1]);
            Goldilocks3GPU::sub(den, x, xi);
            Goldilocks3GPU::mul(inv, inv, den);
        }
        const uint64_t r = r0 + (uint64_t)k * nT;
        if (r >= domainSize) continue;
        d_den[r] = d[0];
        d_den[domainSize + r] = d[1];
        d_den[2 * domainSize + r] = d[2];
    }
}

// The launch shape of computeFRIDenominators for a domain: 256 threads, or fewer on a domain under 4096 rows.
static inline uint32_t friDenominatorThreads(uint64_t domainSize)
{
    const uint64_t t = domainSize / FRI_DEN_ROWS;
    return (uint32_t)(t < 256 ? (t == 0 ? 1 : t) : 256);
}
static inline uint32_t friDenominatorBlocks(uint64_t domainSize, uint32_t nThreads)
{
    const uint64_t rows = (uint64_t)nThreads * FRI_DEN_ROWS;
    return (uint32_t)((domainSize + rows - 1) / rows);
}

// One thread per row; the S_G are kept FRI_GROUP_BATCH groups at a time (a local array, since the openings index
// it by group id). Opening o reads D[r - o B] once per batch, so fewer batches is cheaper until the array spills
// past L1: measured 8 < 16 < 32 > 64 on 35-40 groups.
#define FRI_GROUP_BATCH 32
static __global__ void computeFRIExpression(uint64_t domainSize, uint64_t nOpenings, uint64_t extendBits,
                                            const int64_t *d_openingPoints, const gl64_t *d_den, uint32_t nGroups,
                                            const uint32_t *d_colStart, const FriTerm *d_cols, const uint32_t *d_opStart,
                                            const uint32_t *d_opGroups, gl64_t *d_b, gl64_t *d_a, gl64_t *d_k,
                                            const gl64_t *d_cmPols, const gl64_t *d_customCommits, const gl64_t *d_fixedPols,
                                            gl64_t *d_fri)
{
    const uint64_t r = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const uint64_t mask = domainSize - 1;
    auto at = [](gl64_t *v, uint64_t i) -> Goldilocks3GPU::Element & { return *(Goldilocks3GPU::Element *)(v + i * FIELD_EXTENSION); };

    Goldilocks3GPU::Element fri, S[FRI_GROUP_BATCH], W, t, D;
    Goldilocks3GPU::zero(fri);
    for (uint32_t g0 = 0; g0 < nGroups || g0 == 0; g0 += FRI_GROUP_BATCH) {
        const uint32_t g1 = min(g0 + FRI_GROUP_BATCH, nGroups);
        for (uint32_t g = g0; g < g1; ++g) {
            Goldilocks3GPU::Element acc;
            Goldilocks3GPU::zero(acc);
            for (uint32_t c = d_colStart[g]; c < d_colStart[g + 1]; ++c) {
                const FriTerm m = d_cols[c];
                const gl64_t *pol = (m.src == 0 ? d_cmPols : m.src == 1 ? d_customCommits : d_fixedPols) + m.col + r;
                if (m.dim == 1) {
                    gl64_t v = pol[0];
                    Goldilocks3GPU::mul(t, at(d_b, m.vf2Exp), v);
                } else {
                    Goldilocks3GPU::Element v = {pol[0], pol[domainSize], pol[2 * domainSize]};
                    Goldilocks3GPU::mul(t, at(d_b, m.vf2Exp), v);
                }
                Goldilocks3GPU::add(acc, acc, t);
            }
            Goldilocks3GPU::copy(S[g - g0], acc);
        }
        for (uint64_t o = 0; o < nOpenings; ++o) {
            // K_o goes in with the first batch.
            bool any = g0 == 0;
            if (any) Goldilocks3GPU::copy(W, at(d_k, o));
            else Goldilocks3GPU::zero(W);
            for (uint32_t j = d_opStart[o]; j < d_opStart[o + 1]; ++j) {
                const uint32_t g = d_opGroups[j];
                if (g < g0 || g >= g1) continue;
                Goldilocks3GPU::add(W, W, S[g - g0]);
                any = true;
            }
            if (!any) continue;
            const uint64_t q = (r - (uint64_t)(d_openingPoints[o] * (int64_t)(1ULL << extendBits))) & mask;
            D[0] = d_den[q];
            D[1] = d_den[domainSize + q];
            D[2] = d_den[2 * domainSize + q];
            Goldilocks3GPU::mul(W, W, at(d_a, o));
            Goldilocks3GPU::mul(t, W, D);
            Goldilocks3GPU::add(fri, fri, t);
        }
    }
    Goldilocks3GPU::copy(at(d_fri, r), fri);
}

#endif
