#ifndef MULTIPLICITY_PROGRAM_HPP
#define MULTIPLICITY_PROGRAM_HPP

#include <cstdint>
#include <vector>
#include "multiplicity_job.hpp"
#include "multiplicity_bytecode.hpp"
#include "setup_ctx.hpp"

// A lookup field evaluated as the PIL wrote it, rather than reduced to a closed form.
//
// `MulPoly` can only hold `sum of <= MUL_MAX_PRODUCTS products of <= 2 LINEAR factors`. Keccakf's
// selector is neither: measured, it is 121 operations whose shape is a product of two non-linear
// factors, and no value of MUL_MAX_PRODUCTS expresses it -- raising the cap only moved the failure
// from "too many product terms" to "product of non-linear factors". Those 22 selectors were the
// whole of the interpreter's 494 ms, which is 70% of all multiplicity GPU time.
//
// So this rung does not reduce anything. `computeExpressions_`'s own instruction stream is already
// linear SSA -- `args[i]` the operation, `args[i+1]` the destination temporary, `args[i+2..7]` the
// two operands -- so it transliterates one-for-one into instructions the scatter evaluates inline,
// per row, fused into the same kernel. There is no shape it cannot express and no cap to tune; the
// interpreter stays only for operands this cannot address at all (dim3 temporaries, challenges).
//
// Temporaries are per-thread, so their count is a register/local-memory cost paid by every row.
// Measured over every expression in the zisk PIL the maximum is 6; the cap below leaves room and
// fails loudly rather than silently truncating.
#define MUL_PROG_MAX_TEMP 12

enum MulOperandKind : uint8_t {
    MUL_OPND_TEMP  = 0,   // a previously written temporary
    MUL_OPND_COL   = 1,   // a column or uniform, addressed by `term`
    MUL_OPND_CONST = 2,   // a canonical field element
};

struct MulOperandDev {
    MulTermDev term;      // valid when kind == MUL_OPND_COL; `coef` is always 1 here
    uint64_t   konst;     // valid when kind == MUL_OPND_CONST
    uint16_t   tmp;       // valid when kind == MUL_OPND_TEMP
    uint8_t    kind;
    uint8_t    pad[5];
};

// `dst` is written unless this is the last instruction, whose result is the field's value --
// exactly the convention mulWalkBytecode follows.
struct MulInsnDev {
    MulOperandDev a, b;
    uint16_t      dst;
    uint8_t       op;     // 0 add, 1 sub, 2 mul, 3 rsub (b - a)
    uint8_t       pad[5];
};

struct MulProgram {
    std::vector<MulInsnDev> insns;
    uint32_t nTemp = 0;
    bool     ok    = false;
};


// Why the last compile gave up, for the same reason mulWalkFailReason exists: a field that reaches
// the interpreter is a performance cliff, so say what stopped it.
inline const char*& mulProgFailReason() { static const char* r = nullptr; return r; }

// A term's address, resolved: every prover buffer is flat column-major (goldilocks_trace_layout.cuh),
// so a column reference is one element offset plus the row.
inline bool mulTermToDev(SetupCtx& setupCtx, const MulLinTerm& t, uint64_t domainSize, MulTermDev& out) {
    (void)domainSize;
    const auto& si = setupCtx.starkInfo;
    const uint32_t base = (uint32_t)(1 + si.nStages + 3 + si.customCommits.size());
    out = MulTermDev{};
    out.coef = t.coef;

    if (mulIsUniformType(t.type, base)) {       // publics and the three value pools: no row term
        out.src = t.type == base + 2 ? MUL_SRC_PUBLIC
                : t.type == base + 4 ? MUL_SRC_AIRVALUE
                : t.type == base + 5 ? MUL_SRC_PROOFVALUE
                                     : MUL_SRC_AIRGROUPVALUE;
        out.sectionOffset = t.argIdx;           // a flat index into the pool
        return true;
    }

    const uint32_t customBase = (uint32_t)(si.nStages + 4);
    if (t.type >= customBase && t.type < customBase + si.customCommits.size()) {
        const uint64_t idx = t.type - customBase;
        const std::string sec = si.customCommits[idx].name + "0";
        auto it = si.mapOffsets.find(std::make_pair(sec, false));
        auto in = si.mapSectionsN.find(sec);
        if (it == si.mapOffsets.end() || in == si.mapSectionsN.end()) return false;
        if (t.rowOff >= si.openingPoints.size()) return false;
        out.rowStride = si.verify ? 0 : (int32_t)si.openingPoints[t.rowOff];
        out.src = MUL_SRC_CUSTOM;
        out.sectionOffset = it->second;
        out.col = t.argIdx;
        out.nCols = (uint32_t)in->second;
        return true;
    }
    if (t.rowOff >= si.openingPoints.size()) return false;
    out.rowStride = si.verify ? 0 : (int32_t)si.openingPoints[t.rowOff];
    out.col = t.argIdx;
    if (t.type == 0) {                          // const pols: their own buffer, no offset
        out.src = MUL_SRC_CONST;
        out.nCols = (uint32_t)si.nConstants;
    } else if (t.type == 1) {                   // stage 1: the trace buffer, no offset
        out.src = MUL_SRC_TRACE;
        out.nCols = (uint32_t)si.mapSectionsN.at("cm1");
    } else {                                    // later stages: a section of aux_trace
        const std::string sec = "cm" + std::to_string(t.type);
        auto it = si.mapOffsets.find(std::make_pair(sec, false));
        auto in = si.mapSectionsN.find(sec);
        if (it == si.mapOffsets.end() || in == si.mapSectionsN.end()) return false;
        out.src = MUL_SRC_AUX;
        out.sectionOffset = it->second;
        out.nCols = (uint32_t)in->second;
    }
    return true;
}

#endif
