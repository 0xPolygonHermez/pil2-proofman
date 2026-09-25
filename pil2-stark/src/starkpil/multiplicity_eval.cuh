#ifndef MULTIPLICITY_EVAL_CUH
#define MULTIPLICITY_EVAL_CUH

#include "multiplicity_job.hpp"
#include "multiplicity_job.hpp"
#include "multiplicity_decoders.hpp"

// Per-row device evaluation of a lookup field. The single copy shared by the multiplicity scatter
// and the slot's witness_calc hints: a new source belongs in mulTermValue, never in a second copy.

struct MulBases {
    const uint64_t *constPols, *trace, *aux, *publics, *airValues, *proofValues, *airgroupValues,
                   *customFixed;
    // cm1 may be the whole domain or a tile. `traceRows` is its column stride, `rowBegin` the
    // global row of its first row; (domainSize, 0) for a full trace.
    uint64_t traceRows, rowBegin;
    // First global row this launch COUNTS (a haloed tile's window starts earlier, at rowBegin).
    uint64_t rowStart;
    // Packed witness (read before unpack) and the side buffer of prover-computed stage-1 columns
    // (witness_hints_slot.hpp). Last and defaulted so brace initialisers are unaffected.
    const uint64_t *packed = nullptr, *side = nullptr;
    uint64_t wordsPerRow = 0;
    // Indexed rows: the instruction table and its geometry, air-uniform.
    const uint64_t *table = nullptr;
    uint64_t wordsPerEntry = 0, numEntries = 0, indexBits = 0;
    // If set, `packed` is column-major: word w of row r at packed[w * traceRows + r].
    uint32_t packedColMajor = 0;
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
        // PACKED and HINTCOL never reach here; mulTermValue handles them.
        default:                 return b.customFixed;
    }
}

// One bit field out of a packed row. Must match unpack_trace, or the value read here differs from
// the one committed.
__device__ __forceinline__ uint64_t mulPackedAt(const uint64_t* __restrict__ packed,
                                                uint64_t wordsPerRow, uint64_t row,
                                                uint64_t bit, uint64_t nbits) {
    const uint64_t* r = packed + row * wordsPerRow;
    const uint64_t widx = bit >> 6, boff = bit & 63;
    uint64_t v = __ldg(&r[widx]) >> boff;
    // Straddles only when boff > 0, so the shift below is always < 64.
    if (boff + nbits > 64) v |= __ldg(&r[widx + 1]) << (64 - boff);
    return (nbits < 64) ? (v & ((1ull << nbits) - 1ull)) : v;
}

__device__ __forceinline__ uint64_t mulTermOffset(const MulTermDev& t, const MulBases& b,
                                                  uint64_t row, uint64_t rowMask) {
    if (MUL_SRC_IS_UNIFORM(t.src)) return t.sectionOffset;
    const uint64_t r = (row + (uint64_t)t.rowStride) & rowMask;
    // Modular: a tile's halo wraps past row 0 / the last row.
    if (t.src == MUL_SRC_TRACE)
        return (uint64_t)t.col * b.traceRows + ((r - b.rowBegin) & rowMask);
    // Column-major: this backend stores a section as `col * nRows + row`.
    return t.sectionOffset + (uint64_t)t.col * (rowMask + 1) + r;
}

// Column-major variant: a straddling field's two words are nRows apart.
__device__ __forceinline__ uint64_t mulPackedAtCol(const uint64_t* __restrict__ packed,
                                                   uint64_t nRows, uint64_t row,
                                                   uint64_t bit, uint64_t nbits) {
    const uint64_t widx = bit >> 6, boff = bit & 63;
    uint64_t v = __ldg(&packed[widx * nRows + row]) >> boff;
    if (boff + nbits > 64) v |= __ldg(&packed[(widx + 1) * nRows + row]) << (64 - boff);
    return (nbits < 64) ? (v & ((1ull << nbits) - 1ull)) : v;
}

// A term's value, given the GLOBAL row (the power-of-two mask wraps `'`-shifts). A tile holds no
// neighbours, so mulPlanStreamable refuses airs with shifted cm1 terms.
__device__ __forceinline__ uint64_t mulTermValue(const MulTermDev& t, const MulBases& b,
                                                 uint64_t row, uint64_t rowMask) {
    if (t.src == MUL_SRC_PACKED) {
        const uint64_t r = (row + (uint64_t)t.rowStride) & rowMask;
        if (b.packedColMajor)
            return mulPackedAtCol(b.packed, b.traceRows, r, t.sectionOffset, t.nCols);
        return mulPackedAt(b.packed, b.wordsPerRow, r, t.sectionOffset, t.nCols);
    }
    if (t.src == MUL_SRC_PACKED_IDX) {
        const uint64_t r = (row + (uint64_t)t.rowStride) & rowMask;
        // Lane index from the row header, then the field from that table entry.
        const uint64_t index = mulPackedAt(b.packed, b.wordsPerRow, r,
                                           (uint64_t)t.col * b.indexBits, b.indexBits);
        if (index >= b.numEntries) return 0;   // in bounds; a wrong root is the signal we want
        return mulPackedAt(b.table, b.wordsPerEntry, index, t.sectionOffset, t.nCols);
    }
    if (t.src == MUL_SRC_HINTCOL)
        return mulCanonHD(__ldg(&b.side[t.sectionOffset * (rowMask + 1)
                                        + ((row + (uint64_t)t.rowStride) & rowMask)]));
    return mulCanonHD(__ldg(&mulBaseFor(b, t.src)[mulTermOffset(t, b, row, rowMask)]));
}







// One operand of a compiled program instruction.
__device__ __forceinline__ uint64_t mulOperandVal(const MulOperandDev& o, const MulBases& bases,
                                                  uint64_t row, uint64_t rowMask,
                                                  const uint64_t* tmp){
    if (o.kind == MUL_OPND_TEMP)  return tmp[o.tmp];
    if (o.kind == MUL_OPND_CONST) return o.konst;
    return mulTermValue(o.term, bases, row, rowMask);
}

// Evaluate a compiled program for one row. `tmp` is dynamically indexed, so it lives in local
// memory; the most recent temporary is held in a register and the array is touched only when a
// program juggles several (most chains use exactly one).
__device__ __forceinline__ uint64_t mulEvalProgram(const MulInsnDev* __restrict__ prog, uint32_t n,
                                                   const MulBases& bases, uint64_t row,
                                                   uint64_t rowMask){
    // Deliberately NOT zeroed: the bytecode is SSA, so every read follows an eviction below.
    uint64_t tmp[MUL_PROG_MAX_TEMP];
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
        // Writing a different temporary evicts the held one into the array.
        if (hotIdx != in.dst && hotIdx != 0xFFFFFFFFu) tmp[hotIdx] = hotVal;
        hotIdx = in.dst;
        hotVal = r;
    }
    return last;
}

// Evaluate one job field (always a compiled program).
__device__ __forceinline__ uint64_t mulEvalField(uint32_t progOff, uint32_t progLen,
                                                 const MulInsnDev* __restrict__ prog,
                                                 const MulBases& bases, uint64_t row, uint64_t rowMask) {
    return mulEvalProgram(prog + progOff, progLen, bases, row, rowMask);
}


#endif
