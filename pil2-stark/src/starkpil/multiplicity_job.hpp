#ifndef MULTIPLICITY_JOB_HPP
#define MULTIPLICITY_JOB_HPP

#include <cstdint>
#include <vector>
#include "multiplicity_decoders.hpp"

// A column reference as the expression bytecode names it, before its address is resolved.
// `type` is the operand type load__ decodes: below bufferCommitSize a pol buffer (0 const, 1.. the
// committed stages), at or above it one of the uniform pools.
struct MulLinTerm {
    uint16_t type   = 0;
    uint16_t argIdx = 0;   // column id within that buffer
    uint16_t rowOff = 0;   // index into nextStridesExps, for `'`-shifted references
};

// The uniform pools, in load__'s operand-type order. No challenges or evals: they do not exist
// yet when the witness commits.
inline bool mulIsUniformType(uint16_t type, uint32_t base) {
    return type == base + 2 || type == base + 4 || type == base + 5 || type == base + 6;
}

// One range-check lookup, resolved. Backend-neutral: the GPU kernel and the CPU scatter evaluate
// the SAME jobs. Addresses are resolved at registration against flat column-major buffers
// (goldilocks_trace_layout.cuh). Row-varying sources index by `elemBase + row`, uniform ones
// (publics, value pools) by `elemBase`, known only once the proof starts.
enum MulSrc : uint32_t {
    MUL_SRC_CONST = 0, MUL_SRC_TRACE = 1, MUL_SRC_AUX = 2,
    MUL_SRC_PUBLIC = 3, MUL_SRC_AIRVALUE = 4, MUL_SRC_PROOFVALUE = 5, MUL_SRC_AIRGROUPVALUE = 6,
    // A custom commit's fixed section. Row-addressed like the trace, in its own buffer.
    MUL_SRC_CUSTOM = 7,
    // Slot-only sources for cm1, where no unpacked trace exists (see witness_hints_slot.hpp).
    //   PACKED  -- bit field of a packed row: `sectionOffset` = BIT offset in the row, `nCols` = width.
    //   HINTCOL -- stage-1 column from a witness_calc hint: `sectionOffset` = its side-buffer slot.
    MUL_SRC_PACKED = 8,
    MUL_SRC_HINTCOL = 9,
    //   PACKED_IDX -- indexed air column in the instruction table: `col` = lane, `sectionOffset` =
    //              bit offset, `nCols` = width. Must match unpackIndexedRow's walk.
    MUL_SRC_PACKED_IDX = 10,
    MUL_SRC_N = 11
};
// Publics and the value pools (no row term). A range, not a floor: slot sources sit above.
#define MUL_SRC_IS_UNIFORM(s) ((s) >= MUL_SRC_PUBLIC && (s) <= MUL_SRC_AIRGROUPVALUE)

// 24 bytes, read twice per instruction, so its size is scatter L2 traffic.
// Column and section offset stay separate: the GPU is column-major (`col * nRows + row`), the CPU
// row-major (`row * nCols + col`, expressions_pack.hpp); one baked offset breaks one backend.
struct MulTermDev {
    uint64_t sectionOffset;  // start of the section inside its buffer; also the index of a uniform
    int32_t  rowStride;      // nextStridesExps[rowOffsetIndex], for `'`-shifted references
    uint32_t src;            // MulSrc
    uint32_t col;            // column within the section
    uint32_t nCols;          // columns in the section, for the row-major backend
};
static_assert(sizeof(MulTermDev) == 24, "MulTermDev size drifted -- update the L1-traffic comment above");

// A lookup field, compiled: `computeExpressions_`'s instruction stream is linear SSA (`args[i]`
// op, `args[i+1]` dst temp, `args[i+2..7]` operands) and transliterates one-for-one into
// instructions the scatter evaluates inline per row. The interpreter remains only for operands
// this cannot address (dim3 temporaries, challenges).
//
// Temporaries are per-thread registers paid by every row; exceeding the cap below fails loudly.
#define MUL_PROG_MAX_TEMP 12

enum MulOperandKind : uint8_t {
    MUL_OPND_TEMP  = 0,   // a previously written temporary
    MUL_OPND_COL   = 1,   // a column or uniform, addressed by `term`
    MUL_OPND_CONST = 2,   // a canonical field element
};

struct MulOperandDev {
    MulTermDev term;      // valid when kind == MUL_OPND_COL
    uint64_t   konst;     // valid when kind == MUL_OPND_CONST
    uint16_t   tmp;       // valid when kind == MUL_OPND_TEMP
    uint8_t    kind;
    uint8_t    pad[5];
};

// `dst` is written unless this is the last instruction, whose result is the field's value.
struct MulInsnDev {
    MulOperandDev a, b;
    uint16_t      dst;
    uint8_t       op;     // 0 add, 1 sub, 2 mul, 3 rsub (b - a)
    uint8_t       pad[5];
};

struct MulProgram {
    std::vector<MulInsnDev> insns;
    bool     ok    = false;
};

inline MulOperandDev mulOpConst(uint64_t k) {
    MulOperandDev o{}; o.kind = MUL_OPND_CONST; o.konst = k; return o;
}
inline MulOperandDev mulOpTemp(uint16_t t) {
    MulOperandDev o{}; o.kind = MUL_OPND_TEMP; o.tmp = t; return o;
}
inline void mulEmit(MulProgram& out, uint16_t dst, const MulOperandDev& a, uint8_t op,
                    const MulOperandDev& b) {
    MulInsnDev in{}; in.a = a; in.b = b; in.op = op; in.dst = dst;
    out.insns.push_back(in);
}

struct MulJobDev {
    // Hot fields first: every thread reads them for every job on every row, so keep them in one or
    // two cache lines.
    uint64_t   rows;       // 1 for a degree-0 term, else the air's row count
    uint64_t   accBase;    // first counter of this table inside the air's accumulator
    uint64_t   nTableRows;
    uint64_t   biasFE;     // -min, in the field
    uint64_t   mapSlots;
    const uint64_t* mapKV;
    // Digit recoding of the first tuple column; see MulDecoder. Needs one column, not the tuple.
    const uint64_t* digitTab;
    uint32_t   digitCols;
    uint64_t   hostAirId;  // the air whose virtual table holds this lookup's counters
    uint32_t   nKey;
    uint32_t   tableId;
    uint32_t   selConstOne;
    uint32_t   hasBus;
    // Every field is a compiled program: offsets into the air's single instruction buffer.
    uint32_t   valProgOff, valProgLen;
    uint32_t   selProgOff, selProgLen;
    uint32_t   busProgOff, busProgLen;
    uint32_t   keyProgOff[MUL_MAX_TUPLE], keyProgLen[MUL_MAX_TUPLE];

};

// `bases` is indexed by MulSrc; a null entry is fine as long as no term names it.

// Rows one block covers. Here, not with the kernel, because the plan sizes its grid from it.
#define MUL_SCATTER_BLOCK 256

#endif
