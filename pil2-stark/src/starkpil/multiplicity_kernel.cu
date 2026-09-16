#include "multiplicity_kernel.cuh"
#include "multiplicity_decoders.hpp"
#include <algorithm>
#include "warp_atomic.cuh"
#include "multiplicity_combine.cuh"

// Dedicated range-check scatter: tuple, selector and bus id are linear forms, so a thread
// evaluates all three from registers instead of running `computeExpressions_` once per hint.

// Named members, NOT an array: indexing an array by a runtime value spills the whole struct to
// local memory and doubles every term load.
struct MulBases {
    const uint64_t *constPols, *trace, *aux, *publics, *airValues, *proofValues, *airgroupValues,
                   *customFixed;
};

__device__ __forceinline__ const uint64_t* mulBaseFor(const MulBases& b, uint32_t src) {
    switch (src) {
        case MUL_SRC_CONST:      return b.constPols;
        case MUL_SRC_TRACE:      return b.trace;
        case MUL_SRC_AUX:        return b.aux;
        case MUL_SRC_PUBLIC:     return b.publics;
        case MUL_SRC_AIRVALUE:   return b.airValues;
        case MUL_SRC_PROOFVALUE: return b.proofValues;
        case MUL_SRC_AIRGROUPVALUE: return b.airgroupValues;
        default:                 return b.customFixed;
    }
}

__device__ __forceinline__ uint64_t mulEvalTerms(const MulTermDev* t_, uint32_t n, uint64_t konst,
                                                 const MulBases& bases, uint64_t row, uint64_t rowMask) {
    uint64_t acc = konst;
    for (uint32_t i = 0; i < n; ++i) {
        const MulTermDev t = t_[i];
        // domainSize is a power of two, so the mask wraps `'`-shifted references for free.
        // Column-major: this backend stores a section as `col * nRows + row`.
        const uint64_t off = MUL_SRC_IS_UNIFORM(t.src)
                           ? t.sectionOffset
                           : t.sectionOffset + (uint64_t)t.col * (rowMask + 1)
                             + ((row + (uint64_t)t.rowStride) & rowMask);
        const uint64_t v = mulCanonHD(__ldg(&mulBaseFor(bases, t.src)[off]));
        acc = mulAddFE(acc, t.coef == 1 ? v : mulMulFE(v, t.coef));
    }
    return acc;
}


__device__ __forceinline__ uint64_t mulEvalForm(const MulFormDev& f, const MulBases& bases,
                                                uint64_t row, uint64_t rowMask) {
    const uint64_t a = mulEvalTerms(f.t, f.n, f.konst, bases, row, rowMask);
    if (!f.hasProduct) return a;
    return mulMulFE(a, mulEvalTerms(f.t2, f.n2, f.konst2, bases, row, rowMask));
}

// A sum of products: the single-product evaluator above, summed.
__device__ __forceinline__ uint64_t mulEvalPoly(const MulPolyDev& q, const MulBases& bases,
                                                uint64_t row, uint64_t rowMask) {
    uint64_t acc = 0;
    for (uint32_t i = 0; i < q.nProd; ++i) acc = mulAddFE(acc, mulEvalForm(q.p[i], bases, row, rowMask));
    return acc;
}



// Evaluate a compiled program for one row. The temporaries are per-thread; MUL_PROG_MAX_TEMP is
// sized from the deepest expression in the PIL (measured: 6), so this array stays small enough for
// the compiler to keep in local memory backed by L1 rather than spilling anything wider.
__device__ __forceinline__ uint64_t mulOperandVal(const MulOperandDev& o, const MulBases& bases,
                                                  uint64_t row, uint64_t rowMask,
                                                  const uint64_t* tmp){
    if (o.kind == MUL_OPND_TEMP)  return tmp[o.tmp];
    if (o.kind == MUL_OPND_CONST) return o.konst;
    const MulTermDev& t = o.term;
    const uint64_t off = MUL_SRC_IS_UNIFORM(t.src)
                       ? t.sectionOffset
                       : t.sectionOffset + (uint64_t)t.col * (rowMask + 1)
                         + ((row + (uint64_t)t.rowStride) & rowMask);
    return mulCanonHD(__ldg(&mulBaseFor(bases, t.src)[off]));
}

// The temporaries live in a stack array, which is local memory: `tmp[i]` is dynamically indexed, so
// nvcc cannot keep it in registers. That matters because these programs are chains -- Keccakf's
// selector is 119 instructions of `t0 = col + t0` -- so a naive evaluator does two local-memory
// accesses per instruction where a closed form would accumulate in a register.
//
// The most recently written temporary is therefore held in one, and the array is touched only when
// a program actually juggles several. Measured over the zisk PIL no expression uses more than 6,
// and the ones that cost anything use exactly one, so the array is typically never written at all.
__device__ __forceinline__ uint64_t mulEvalProgram(const MulInsnDev* __restrict__ prog, uint32_t n,
                                                   const MulBases& bases, uint64_t row,
                                                   uint64_t rowMask){
    uint64_t tmp[MUL_PROG_MAX_TEMP];
    #pragma unroll
    for (uint32_t i = 0; i < MUL_PROG_MAX_TEMP; ++i) tmp[i] = 0;
    uint32_t hotIdx = 0xFFFFFFFFu;   // no temporary written yet
    uint64_t hotVal = 0;

    uint64_t last = 0;
    for (uint32_t k = 0; k < n; ++k) {
        const MulInsnDev in = prog[k];
        const uint64_t a = in.a.kind == MUL_OPND_TEMP && in.a.tmp == hotIdx
                         ? hotVal : mulOperandVal(in.a, bases, row, rowMask, tmp);
        const uint64_t b = in.b.kind == MUL_OPND_TEMP && in.b.tmp == hotIdx
                         ? hotVal : mulOperandVal(in.b, bases, row, rowMask, tmp);
        uint64_t r;
        switch (in.op) {
            case 0:  r = mulAddFE(a, b); break;
            case 1:  r = mulSubFEHD(a, b); break;
            case 2:  r = mulMulFE(a, b); break;
            default: r = mulSubFEHD(b, a); break;   // rsub
        }
        if (k + 1 == n) { last = r; break; }
        // Writing a different temporary evicts the held one, so the array stays authoritative for
        // every temporary except the one in the register.
        if (hotIdx != in.dst && hotIdx != 0xFFFFFFFFu) tmp[hotIdx] = hotVal;
        hotIdx = in.dst;
        hotVal = r;
    }
    return last;
}

// A field is either a compiled program or a closed form, never both.
__device__ __forceinline__ uint64_t mulEvalField(const MulPolyDev& q, uint32_t progOff,
                                                 uint32_t progLen, const MulInsnDev* __restrict__ prog,
                                                 const MulBases& bases, uint64_t row, uint64_t rowMask) {
    if (progLen != 0) return mulEvalProgram(prog + progOff, progLen, bases, row, rowMask);
    return mulEvalPoly(q, bases, row, rowMask);
}


// The second bound is the occupancy the shared combining cache already allows (17 KB a block, so
// five blocks an SM). Without it the compiler budgets registers per-kernel and the digit decoder's
// extra live values pushed it to 60, which caps an SM at 1092 threads -- below the 1280 the shared
// memory permits, so every air slowed down for a decoder only some tables use.
__global__ __launch_bounds__(MUL_SCATTER_BLOCK, 5)
void mul_scatter_kernel_rows(const MulJobDev* __restrict__ jobs, uint32_t nJobs, MulBases bases,
                             uint64_t rowMask, uint64_t rows, uint64_t* __restrict__ acc,
                             uint64_t* __restrict__ oob, uint64_t air,
                             const MulInsnDev* __restrict__ prog) {
    __shared__ uint64_t           sKey[MUL_COMBINE_SLOTS];
    __shared__ unsigned long long sVal[MUL_COMBINE_SLOTS];
    MulCombine comb{sKey, sVal};
    mulCombineInit(comb);
    unsigned long long* gacc = (unsigned long long*)acc;

    // Persistent grid: the block is launched once per SM-wave and walks the domain, so the shared
    // table it builds is amortised over many rows instead of one block's worth. The trip count is
    // uniform across the block -- every thread must reach the flush barrier.
    const uint64_t stride = (uint64_t)gridDim.x * MUL_SCATTER_BLOCK;
    const uint64_t base   = (uint64_t)blockIdx.x * MUL_SCATTER_BLOCK + threadIdx.x;
    const uint64_t nIter  = (rows + stride - 1) / stride;

    for (uint64_t it = 0; it < nIter; ++it) {
        const uint64_t row = base + it * stride;
        if (row >= rows) continue;

        // Reset per row: the cached selector is only valid for the row it was computed on.
        uint32_t lastSelOff = 0xFFFFFFFFu;
        uint64_t lastSelVal = 0;

        for (uint32_t k = 0; k < nJobs; ++k) {
            const MulJobDev& j = jobs[k];
            // A degree-0 job contributes once per instance, not once per row.
            if (row >= j.rows) continue;

            uint64_t sel = 1;
            if (!j.selConstOne) {
                // Lookups of one air share selectors heavily -- Keccakf's 22 compiled jobs all read
                // the same 121-instruction clock predicate, which is 2.7 billion instruction
                // evaluations if each job re-runs it. The plan memoises identical programs to one
                // offset and orders the jobs by it, so carrying the last result across the loop
                // collapses a run of them into a single evaluation.
                if (j.selProgLen != 0 && j.selProgOff == lastSelOff) {
                    sel = lastSelVal;
                } else {
                    sel = mulEvalField(j.sel, j.selProgOff, j.selProgLen, prog, bases, row, rowMask);
                    if (j.selProgLen != 0) { lastSelOff = j.selProgOff; lastSelVal = sel; }
                }
                if (sel == 0) continue;
            }

            // Dynamic opid (multi_range_check): the bus id selects the table per row.
            if (j.hasBus && mulEvalField(j.bus, j.busProgOff, j.busProgLen, prog, bases, row,
                                         rowMask) != (uint64_t)j.tableId) continue;

            uint64_t key[MUL_MAX_TUPLE];
            if (j.mapSlots == 0 && j.digitCols == 0) {
                key[0] = mulAddFE(mulEvalField(j.value, j.valProgOff, j.valProgLen, prog,
                                               bases, row, rowMask), j.biasFE);
            } else {
                for (uint32_t c = 0; c < j.nKey; ++c)
                    key[c] = mulEvalField(j.key[c], j.keyProgOff[c], j.keyProgLen[c], prog,
                                          bases, row, rowMask);
            }
            uint64_t idx;
            if (!mulResolveRow(key, j.nKey, j.mapSlots, j.mapKV, idx,
                               j.digitCols, j.digitTab, j.dec) || idx >= j.nTableRows) {
                // j.nKey is 0 for a plain affine job (there is no tuple, just the one folded value
                // already sitting in key[0]), so floor it at 1 rather than logging an empty key.
                mulRecordOob(oob, j.tableId, air, key, j.nKey ? j.nKey : 1u);
                continue;
            }
            const uint64_t slot = j.accBase + idx;
            if (j.selConstOne) mulCombineInc(comb, __activemask(), slot, gacc);
            else               mulCombineAdd(comb, __activemask(), slot, sel, gacc);
        }
    }
    mulCombineFlush(comb, gacc);
}

__device__ __forceinline__ uint64_t mulEvalTermsTile(const MulTermDev* t_, uint32_t n, uint64_t konst,
                                                     const MulBases& bases, uint64_t localRow,
                                                     uint64_t globalRow, uint64_t traceRows,
                                                     uint64_t fullRows) {
    uint64_t acc = konst;
    for (uint32_t i = 0; i < n; ++i) {
        const MulTermDev t = t_[i];
        // cm1 lives in the tile, at its own height; everything else is still full-height.
        const uint64_t off = MUL_SRC_IS_UNIFORM(t.src)
                           ? t.sectionOffset
                           : (t.src == MUL_SRC_TRACE
                                  ? (uint64_t)t.col * traceRows + localRow
                                  : t.sectionOffset + (uint64_t)t.col * fullRows + globalRow);
        const uint64_t v = mulCanonHD(__ldg(&mulBaseFor(bases, t.src)[off]));
        acc = mulAddFE(acc, t.coef == 1 ? v : mulMulFE(v, t.coef));
    }
    return acc;
}

__device__ __forceinline__ uint64_t mulEvalFormTile(const MulFormDev& f, const MulBases& bases,
                                                    uint64_t localRow, uint64_t globalRow,
                                                    uint64_t traceRows, uint64_t fullRows) {
    const uint64_t a = mulEvalTermsTile(f.t, f.n, f.konst, bases, localRow, globalRow, traceRows, fullRows);
    if (!f.hasProduct) return a;
    return mulMulFE(a, mulEvalTermsTile(f.t2, f.n2, f.konst2, bases, localRow, globalRow, traceRows, fullRows));
}

__device__ __forceinline__ uint64_t mulEvalPolyTile(const MulPolyDev& q, const MulBases& bases,
                                                    uint64_t localRow, uint64_t globalRow,
                                                    uint64_t traceRows, uint64_t fullRows) {
    uint64_t acc = 0;
    for (uint32_t i = 0; i < q.nProd; ++i)
        acc = mulAddFE(acc, mulEvalFormTile(q.p[i], bases, localRow, globalRow, traceRows, fullRows));
    return acc;
}

__global__ __launch_bounds__(MUL_SCATTER_BLOCK)
void mul_scatter_kernel_tile(const MulJobDev* __restrict__ jobs, MulBases bases,
                             uint64_t rows, uint64_t rowBegin, uint64_t traceRows,
                             uint64_t fullRows, uint64_t* __restrict__ acc,
                             uint64_t* __restrict__ oob, uint64_t air) {
    const MulJobDev& j = jobs[blockIdx.y];
    const uint64_t localRow = (uint64_t)blockIdx.x * MUL_SCATTER_BLOCK + threadIdx.x;
    if (localRow >= rows) return;
    const uint64_t globalRow = rowBegin + localRow;
    // A degree-0 job contributes once per instance, not once per tile.
    if (globalRow >= j.rows) return;

    uint64_t sel = 1;
    if (!j.selConstOne) {
        sel = mulEvalPolyTile(j.sel, bases, localRow, globalRow, traceRows, fullRows);
        if (sel == 0) return;
    }
    if (j.hasBus && mulEvalPolyTile(j.bus, bases, localRow, globalRow, traceRows, fullRows)
                        != (uint64_t)j.tableId) return;

    const uint64_t idx = mulAddFE(mulEvalPolyTile(j.value, bases, localRow, globalRow, traceRows, fullRows),
                                  j.biasFE);
    if (idx >= j.nTableRows) {
        if (oob != nullptr) {
            unsigned long long* o = (unsigned long long*)oob;
            if (atomicAdd(o, 1ULL) == 0) { o[1] = j.tableId; o[2] = air; o[3] = idx; }
        }
        return;
    }
    const uint64_t key = j.accBase + idx;
    unsigned long long* counter = (unsigned long long*)&acc[key];
    if (j.selConstOne) warpAggregatedInc(__activemask(), key, counter);
    else               warpAggregatedAdd(__activemask(), key, sel, counter);
}

void mul_scatter_launch_tile(const MulJobDev* d_jobs, uint32_t nJobs, uint64_t rows,
                             uint64_t rowBegin, const uint64_t* const* bases_, uint64_t traceRows,
                             uint64_t fullRows, uint64_t* acc, uint64_t* oob, uint64_t air,
                             cudaStream_t stream) {
    if (d_jobs == nullptr || nJobs == 0 || rows == 0) return;
    const MulBases bases = { bases_[MUL_SRC_CONST],  bases_[MUL_SRC_TRACE],
                             bases_[MUL_SRC_AUX],    bases_[MUL_SRC_PUBLIC],
                             bases_[MUL_SRC_AIRVALUE], bases_[MUL_SRC_PROOFVALUE],
                             bases_[MUL_SRC_AIRGROUPVALUE], bases_[MUL_SRC_CUSTOM] };
    // A 2D grid here rather than the flat map the full-domain launch uses: within a tile every job
    // spans the same rows, so there is nothing for the map to save.
    dim3 grid((uint32_t)((rows + MUL_SCATTER_BLOCK - 1) / MUL_SCATTER_BLOCK), nJobs);
    mul_scatter_kernel_tile<<<grid, MUL_SCATTER_BLOCK, 0, stream>>>(
        d_jobs, bases, rows, rowBegin, traceRows, fullRows, acc, oob, air);
}

void mul_scatter_launch_rows(const MulJobDev* d_jobs, uint32_t nJobs, const uint64_t* const* bases_,
                             uint64_t domainSize, uint64_t maxRows, uint64_t* acc, uint64_t* oob,
                             uint64_t air, const MulInsnDev* d_prog, cudaStream_t stream) {
    if (d_jobs == nullptr || nJobs == 0 || maxRows == 0) return;
    const MulBases bases = { bases_[MUL_SRC_CONST],  bases_[MUL_SRC_TRACE],
                             bases_[MUL_SRC_AUX],    bases_[MUL_SRC_PUBLIC],
                             bases_[MUL_SRC_AIRVALUE], bases_[MUL_SRC_PROOFVALUE],
                             bases_[MUL_SRC_AIRGROUPVALUE], bases_[MUL_SRC_CUSTOM] };
    // One block per row chunk. The grid saturates the machine well before the domain ends, so a
    // persistent grid bought nothing measurable; the flush only touches slots a block actually
    // claimed, so a wider grid costs proportionally more atomics only where it did more work.
    const uint32_t blocks = (uint32_t)((maxRows + MUL_SCATTER_BLOCK - 1) / MUL_SCATTER_BLOCK);
    mul_scatter_kernel_rows<<<blocks, MUL_SCATTER_BLOCK, 0, stream>>>(
        d_jobs, nJobs, bases, domainSize - 1, maxRows, acc, oob, air, d_prog);
}
