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

// One thread per row, nThreads rows per block. The block batch-inverts the rows of D it reads (the FriWindow
// segments) into shared memory; the S_G are kept FRI_GROUP_BATCH groups at a time.
#define FRI_GROUP_BATCH 16
static __global__ void computeFRIExpression(uint64_t domainSize, uint64_t nOpenings, uint32_t nSegments,
                                            const FriSegment *d_segments, const uint32_t *d_opBase, uint64_t shift,
                                            uint64_t wExt, uint64_t wStep, uint64_t wStepInv, gl64_t *d_xi,
                                            uint32_t nGroups, const uint32_t *d_colStart, const FriTerm *d_cols,
                                            const uint32_t *d_opStart, const uint32_t *d_opGroups, gl64_t *d_b,
                                            gl64_t *d_a, gl64_t *d_k, const gl64_t *d_cmPols,
                                            const gl64_t *d_customCommits, const gl64_t *d_fixedPols, gl64_t *d_fri)
{
    extern __shared__ Goldilocks3GPU::Element sD[];
    const uint32_t tid = threadIdx.x, nT = blockDim.x;
    const uint64_t r = (uint64_t)blockIdx.x * nT + tid;
    auto at = [](gl64_t *v, uint64_t i) -> Goldilocks3GPU::Element & { return *(Goldilocks3GPU::Element *)(v + i * FIELD_EXTENSION); };

    // x[r0 + offset + i] = s wStep^blockIdx wExt^offset wExt^i. Every segment is at least nT long, so the thread
    // has entries tid + j nT in all of them: prefix products over its entries, one inversion, unwind.
    __shared__ gl64_t xBlock;
    if (tid == 0) xBlock = gl64_t(shift) * (gl64_t(wStep) ^ blockIdx.x);
    __syncthreads();
    const gl64_t xThread = xBlock * (gl64_t(wExt) ^ tid);
    auto lastOf = [&](const FriSegment &seg) { return seg.base + tid + (seg.len - 1 - tid) / nT * nT; };
    Goldilocks3GPU::Element &xi = *(Goldilocks3GPU::Element *)d_xi;
    Goldilocks3GPU::Element den, inv;
    uint32_t prev = 0;
    for (uint32_t s = 0; s < nSegments; ++s) {
        const FriSegment seg = d_segments[s];
        gl64_t x = xThread * gl64_t(seg.w);
        for (uint32_t i = seg.base + tid; i < seg.base + seg.len; i += nT, x = x * gl64_t(wStep)) {
            Goldilocks3GPU::sub(den, x, xi);
            if (i == tid) Goldilocks3GPU::copy(sD[i], den);
            else Goldilocks3GPU::mul(sD[i], sD[prev], den);
            prev = i;
        }
    }
    Goldilocks3GPU::inv(inv, sD[prev]);
    for (uint32_t s = nSegments; s-- > 0;) {
        const FriSegment seg = d_segments[s];
        uint32_t i = lastOf(seg);
        gl64_t x = xThread * gl64_t(seg.w) * (gl64_t(wStep) ^ ((i - seg.base) / nT));
        for (; i != tid; i -= nT, x = x * gl64_t(wStepInv)) {
            const bool first = i == seg.base + tid;
            Goldilocks3GPU::sub(den, x, xi);
            Goldilocks3GPU::mul(sD[i], inv, sD[first ? lastOf(d_segments[s - 1]) : i - nT]);
            Goldilocks3GPU::mul(inv, inv, den);
            if (first) break;
        }
    }
    Goldilocks3GPU::copy(sD[tid], inv);
    __syncthreads();

    Goldilocks3GPU::Element fri, S[FRI_GROUP_BATCH], W, t;
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
            Goldilocks3GPU::mul(W, W, at(d_a, o));
            Goldilocks3GPU::mul(t, W, sD[d_opBase[o] + tid]);
            Goldilocks3GPU::add(fri, fri, t);
        }
    }
    Goldilocks3GPU::copy(at(d_fri, r), fri);
}

#endif
