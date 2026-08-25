#ifndef FRI_EXPRESSION_CUH
#define FRI_EXPRESSION_CUH

// The DEEP/FRI polynomial evaluated over the extended domain:
//
//   fri = SUM_o  1/(x[r] - xi_o) * SUM_{j in o} vf2^(n_o-1-j) * (p_j[r] - e_j) * vf1^(O-1-o)
//
// Self-contained on purpose (only the cubic-extension helpers, the trace layout
// and the POD EvalInfo) so the kernels below can be unit tested against a host
// reference without the SetupCtx/StarkInfo translation units.
#include "eval_info.hpp"
#include "goldilocks_cubic_extension.cuh"
#include "goldilocks_trace_layout.cuh"

// Every vf1/vf2 power and every opening evaluation e_j is a per-proof constant,
// not a per-row value, so the two Horner chains above collapse to
//
//   fri[r] = SUM_o 1/(x[r] - xi_o) * ( SUM_j coef_j * p_j[r] + K_o )
//   coef_j = vf1^(O-1-o) * vf2^(n_o-1-j)        K_o = -SUM_j coef_j * e_j
//
// which is one mul31 + one add33 per base-field column instead of a sub33, a
// mul33 and an add33 -- half the muls and half the adds -- and drops the evals
// load out of the row loop entirely. The constants are precomputed once per
// proof by computeFRIFoldedConstants.
//
// Replaced a Horner form (accum = accum*vf2 + (p_j - e_j), then fri = fri*vf1 +
// accum*inv_g) that recomputed all of that per row; measured 10-37% faster
// across the zisk AIR shapes.
//
// The 1/(x[r] - xi_o) factors fold too: carrying the row as a single fraction costs ONE cubic
// inversion per row instead of one per group of four.

// One thread per opening point. coef is indexed by evalPos so it lines up with
// the eval map; K is indexed by opening.
static __global__ void computeFRIFoldedConstants(uint64_t nOpeningPoints, uint64_t *d_countsPerOpeningPos,
                                                EvalInfo **d_evalInfoPerOpening, gl64_t *d_evals,
                                                gl64_t *vf1, gl64_t *vf2, gl64_t *d_coef, gl64_t *d_k)
{
    uint64_t o = blockIdx.x * blockDim.x + threadIdx.x;
    if (o >= nOpeningPoints) return;

    Goldilocks3GPU::Element &vf1e = *(Goldilocks3GPU::Element *)vf1;
    Goldilocks3GPU::Element &vf2e = *(Goldilocks3GPU::Element *)vf2;

    // pow() squares its base in place, so it must never see the shared vf1 buffer.
    Goldilocks3GPU::Element base, c, k, t;
    Goldilocks3GPU::copy(base, vf1e);
    Goldilocks3GPU::pow(base, nOpeningPoints - 1 - o, c);
    Goldilocks3GPU::zero(k);

    const uint64_t n = d_countsPerOpeningPos[o];
    // Walk j downwards so the vf2 exponent runs 0,1,2,... and matches vf2^(n-1-j).
    for (uint64_t s = 0; s < n; ++s) {
        const EvalInfo evalInfo = d_evalInfoPerOpening[o][n - 1 - s];
        gl64_t *dst = &d_coef[evalInfo.evalPos * FIELD_EXTENSION];
        dst[0] = c[0];
        dst[1] = c[1];
        dst[2] = c[2];

        Goldilocks3GPU::Element &eval = *(Goldilocks3GPU::Element *)(d_evals + evalInfo.evalPos * FIELD_EXTENSION);
        Goldilocks3GPU::mul(t, c, eval);
        Goldilocks3GPU::sub(k, k, t);
        Goldilocks3GPU::mul(c, c, vf2e);
    }

    gl64_t *kd = &d_k[o * FIELD_EXTENSION];
    kd[0] = k[0];
    kd[1] = k[1];
    kd[2] = k[2];
}

static __global__ void computeFRIExpressionFolded(uint64_t domainSize, uint64_t nBits, uint64_t nOpeningPoints,
                                                 gl64_t *d_fri, uint64_t *d_countsPerOpeningPos,
                                                 EvalInfo **d_evalInfoPerOpening, gl64_t *d_coef, gl64_t *d_k,
                                                 gl64_t *d_cmPols, gl64_t *d_xDivXSub, gl64_t *d_x,
                                                 gl64_t *d_fixedPols, gl64_t *d_customComits)
{
    int chunk_idx = blockIdx.x;
    uint64_t nchunks = domainSize / blockDim.x;

    while (chunk_idx < nchunks) {
        Goldilocks3GPU::Element accum, res, term, val, num, den, d, t;

        uint64_t i = chunk_idx * blockDim.x;
        uint64_t r = i + threadIdx.x;

        // num/den += accum/d  <=>  num = num*d + accum*den, den = den*d. One inversion per ROW
        // instead of one per group of four, and no inv_g[] batch in registers.
        // A zero d_o (challenge on the domain) poisons the row either way, as before.
        for (uint64_t o = 0; o < nOpeningPoints; ++o) {
            // K_o carries the folded evaluations; an opening with no columns
            // contributes K_o = 0.
            Goldilocks3GPU::copy(accum, *(Goldilocks3GPU::Element *)(d_k + o * FIELD_EXTENSION));

            for (uint64_t j = 0; j < d_countsPerOpeningPos[o]; ++j) {
                EvalInfo evalInfo = d_evalInfoPerOpening[o][j];
                Goldilocks3GPU::Element &coef =
                    *(Goldilocks3GPU::Element *)(d_coef + evalInfo.evalPos * FIELD_EXTENSION);
                // cm sections (type 0) follow resolveLayout (keyed on the small domain nBits); custom
                // commits (1) and fixed/const (2) follow fixedLayout().
                gl64_t *pol;
                Layout polLayout;
                if (evalInfo.type == 0) {
                    pol = d_cmPols;
                    polLayout = resolveLayout(nBits, evalInfo.stageCols);
                } else if (evalInfo.type == 1) {
                    pol = d_customComits;
                    polLayout = fixedLayout();
                } else {
                    pol = d_fixedPols;
                    polLayout = fixedLayout();
                }

                if (evalInfo.dim == 1) {
                    gl64_t v = pol[evalInfo.offset + getBufferOffset(r, evalInfo.stagePos, domainSize, evalInfo.stageCols, polLayout)];
                    Goldilocks3GPU::mul(term, coef, v);
                } else {
                    val[0] = pol[evalInfo.offset + getBufferOffset(r, evalInfo.stagePos, domainSize, evalInfo.stageCols, polLayout)];
                    val[1] = pol[evalInfo.offset + getBufferOffset(r, evalInfo.stagePos + 1, domainSize, evalInfo.stageCols, polLayout)];
                    val[2] = pol[evalInfo.offset + getBufferOffset(r, evalInfo.stagePos + 2, domainSize, evalInfo.stageCols, polLayout)];
                    Goldilocks3GPU::mul(term, coef, val);
                }
                Goldilocks3GPU::add(accum, accum, term);
            }

            Goldilocks3GPU::Element &xdiv = *(Goldilocks3GPU::Element *)(&d_xDivXSub[o * FIELD_EXTENSION]);
            Goldilocks3GPU::sub(d, d_x[r], xdiv);

            if (o == 0) {
                Goldilocks3GPU::copy(num, accum);
                Goldilocks3GPU::copy(den, d);
            } else {
                Goldilocks3GPU::mul(num, num, d);
                Goldilocks3GPU::mul(t, accum, den);
                Goldilocks3GPU::add(num, num, t);
                Goldilocks3GPU::mul(den, den, d);
            }
        }

        Goldilocks3GPU::inv(t, den);
        Goldilocks3GPU::mul(res, num, t);

        d_fri[r * FIELD_EXTENSION] = res[0];
        d_fri[r * FIELD_EXTENSION + 1] = res[1];
        d_fri[r * FIELD_EXTENSION + 2] = res[2];
        chunk_idx += gridDim.x;
    }
}

#endif
