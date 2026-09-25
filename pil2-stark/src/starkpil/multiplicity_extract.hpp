#ifndef MULTIPLICITY_EXTRACT_HPP
#define MULTIPLICITY_EXTRACT_HPP

#include <cstdint>
#include <vector>
#include <string>
#include "multiplicity_job.hpp"
#include "expressions_bin.hpp"
#include "setup_ctx.hpp"

// Why the last compile gave up (a field on the interpreter is a performance cliff).
inline const char*& mulProgFailReason() { static const char* r = nullptr; return r; }

// A term's address: prover buffers are flat column-major, so one element offset plus the row.
inline bool mulTermToDev(SetupCtx& setupCtx, const MulLinTerm& t, uint64_t domainSize, MulTermDev& out) {
    (void)domainSize;
    const auto& si = setupCtx.starkInfo;
    const uint32_t base = (uint32_t)(1 + si.nStages + 3 + si.customCommits.size());
    out = MulTermDev{};

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

// Compile a hint field into the instruction stream the scatter evaluates, once per (air, hint) at
// registration -- never per row.
//
// Layout, mirroring the `ops[kk] == 0` case of computeExpressions_:
//   ops[kk]        dimension case; 0 is dim1-op-dim1, the only shape this compiles
//   args[i+0]      arithmetic op: 0 add, 1 sub, 2 mul, 3 rsub
//   args[i+1]      destination temp index
//   args[i+2..4]   src0 as (type, argIdx, argOffset)
//   args[i+5..7]   src1
// `type` decodes as in load__: at or below `nSections` it names a pol buffer (0 const, 1.. the
// committed stages), `base` and `base+1` are the dim1/dim3 temporaries, and `base+2` upwards are
// the constant pools -- of which only `base+3` (numbers) is known before the proof starts.
struct MulByteCode {
    const uint8_t*  ops     = nullptr;
    const uint16_t* args    = nullptr;
    const uint64_t* numbers = nullptr;   // canonical field elements
    uint32_t nOps      = 0;
    uint32_t nTemp1    = 0;
    uint32_t base      = 0;              // bufferCommitSize
    uint32_t nSections = 0;              // highest valid pol-buffer type
};

// Transliterate the instruction stream one-for-one. Anything not understood returns false with a
// reason, and the caller keeps the interpreter for it.
inline bool mulCompileBytecode(SetupCtx& setupCtx, const MulByteCode& bc, uint64_t domainSize,
                               MulProgram& out) {
    out = MulProgram{};
    mulProgFailReason() = nullptr;
    auto bail = [&](const char* why) { mulProgFailReason() = why; return false; };
    if (bc.nOps == 0 || bc.ops == nullptr || bc.args == nullptr) return bail("no bytecode");
    if (bc.nTemp1 + 1 > MUL_PROG_MAX_TEMP) return bail("too many temporaries");

    auto operand = [&](uint16_t type, uint16_t argIdx, uint16_t argOff, MulOperandDev& o) -> bool {
        o = MulOperandDev{};
        if (type == bc.base) {                                   // dim1 temporary
            if (argIdx > bc.nTemp1) return bail("temp index out of range");
            o.kind = MUL_OPND_TEMP;
            o.tmp  = argIdx;
            return true;
        }
        if (type == bc.base + 1) return bail("dim3 temporary");
        if (type == bc.base + 3 && bc.numbers != nullptr) {      // numbers pool
            o.kind  = MUL_OPND_CONST;
            o.konst = mulCanonHD(bc.numbers[argIdx]);
            return true;
        }
        if (!mulIsUniformType(type, bc.base) && type >= bc.base + 2) return bail("challenge/eval operand");
        if (!mulIsUniformType(type, bc.base) && type > bc.nSections) return bail("zi / xDivXSub operand");
        o.kind = MUL_OPND_COL;
        if (!mulTermToDev(setupCtx, MulLinTerm{ type, argIdx, argOff }, domainSize, o.term))
            return bail("operand address not resolvable");
        return true;
    };

    uint64_t i = 0;
    for (uint32_t k = 0; k < bc.nOps; ++k) {
        if (bc.ops[k] != 0) return bail("non dim1-op-dim1 operation");
        MulInsnDev in{};
        if (!operand(bc.args[i + 2], bc.args[i + 3], bc.args[i + 4], in.a)) return false;
        if (!operand(bc.args[i + 5], bc.args[i + 6], bc.args[i + 7], in.b)) return false;
        in.op = (uint8_t)bc.args[i];
        if (in.op > 3) return bail("unknown arithmetic op");
        in.dst = bc.args[i + 1];
        if (k + 1 != bc.nOps && in.dst > bc.nTemp1) return bail("temp index out of range");
        out.insns.push_back(in);
        i += 8;
    }
    out.ok = true;
    return true;
}

inline bool mulCompileExpr(SetupCtx& setupCtx, uint64_t expId, uint64_t bufferCommitSize,
                           uint64_t domainSize, MulProgram& out) {
    auto it = setupCtx.expressionsBin.expressionsInfo.find(expId);
    if (it == setupCtx.expressionsBin.expressionsInfo.end()) { mulProgFailReason() = "no expression"; return false; }
    const ParserParams& pp = it->second;
    if (pp.destDim != 1) { mulProgFailReason() = "destination is not dim1"; return false; }
    const ParserArgs& pa = setupCtx.expressionsBin.expressionsBinArgsExpressions;

    MulByteCode bc;
    bc.ops       = &pa.ops[pp.opsOffset];
    bc.args      = &pa.args[pp.argsOffset];
    bc.numbers   = (const uint64_t*)pa.numbers;
    bc.nOps      = pp.nOps;
    bc.nTemp1    = pp.nTemp1;
    bc.base      = (uint32_t)bufferCommitSize;
    bc.nSections = (uint32_t)(setupCtx.starkInfo.nStages + 1);
    return mulCompileBytecode(setupCtx, bc, domainSize, out);
}

// A field that is not an expression carries no bytecode at all (caseNoOperations__): it is a bare
// column, number or uniform, which resolves straight to one operand.
inline bool mulOperandOfField(SetupCtx& setupCtx, const HintFieldValue& v, uint64_t bufferCommitSize,
                              uint64_t domainSize, MulOperandDev& out) {
    const StarkInfo& si = setupCtx.starkInfo;
    MulLinTerm t{};
    bool modelled = false;
    switch (v.operand) {
        case opType::cm: {
            const PolMap& p = si.cmPolsMap[v.id];
            if (p.dim != 1) break;
            t = MulLinTerm{ (uint16_t)p.stage, (uint16_t)p.stagePos, (uint16_t)v.rowOffsetIndex };
            modelled = true;
            break;
        }
        case opType::const_: {
            const PolMap& p = si.constPolsMap[v.id];
            if (p.dim != 1) break;
            t = MulLinTerm{ 0, (uint16_t)p.stagePos, (uint16_t)v.rowOffsetIndex };
            modelled = true;
            break;
        }
        case opType::custom: {
            const PolMap& p = si.customCommitsMap[v.commitId][v.id];
            if (p.dim != 1) break;
            // Encoded in the operand space the bytecode uses, so both paths resolve identically.
            t = MulLinTerm{ (uint16_t)(si.nStages + 4 + v.commitId), (uint16_t)p.stagePos,
                            (uint16_t)v.rowOffsetIndex };
            modelled = true;
            break;
        }
        case opType::number:
            out = mulOpConst(mulCanonHD(v.value));
            return true;
        case opType::public_:
            t = MulLinTerm{ (uint16_t)(bufferCommitSize + 2), (uint16_t)v.id, 0 };
            modelled = true;
            break;
        case opType::airgroupvalue:
            t = MulLinTerm{ (uint16_t)(bufferCommitSize + 6), (uint16_t)v.id, 0 };
            modelled = true;
            break;
        case opType::airvalue: {
            // airValues is laid out by stage: stage 1 takes one slot, later stages three
            // (as in addHintFieldAt).
            if (si.airValuesMap[v.id].stage != 1) break;
            uint64_t pos = 0;
            for (uint64_t i = 0; i < v.id; ++i)
                pos += si.airValuesMap[i].stage == 1 ? 1 : FIELD_EXTENSION;
            t = MulLinTerm{ (uint16_t)(bufferCommitSize + 4), (uint16_t)pos, 0 };
            modelled = true;
            break;
        }
        default: break;
    }
    if (!modelled) { mulProgFailReason() = "operand kind not modelled"; return false; }
    out = MulOperandDev{};
    out.kind = MUL_OPND_COL;
    if (!mulTermToDev(setupCtx, t, domainSize, out.term)) {
        mulProgFailReason() = "operand address not resolvable";
        return false;
    }
    return true;
}

// mulEvalProgram returns the last instruction's result, so a bare operand becomes `x + 0`.
inline bool mulCompileField(SetupCtx& setupCtx, const HintFieldValue& v, uint64_t bufferCommitSize,
                            uint64_t domainSize, MulProgram& out) {
    if (v.operand == opType::tmp)
        return mulCompileExpr(setupCtx, v.id, bufferCommitSize, domainSize, out);
    MulOperandDev o{};
    if (!mulOperandOfField(setupCtx, v, bufferCommitSize, domainSize, o)) return false;
    out = MulProgram{};
    mulEmit(out, 0, o, 0 /*add*/, mulOpConst(0));
    out.ok = true;
    return true;
}

// `konst + sum(coef[e] * element_e)`, the row map of a fitted virtual table. Bare-column elements
// are used as operands directly; expression elements are spliced in as sub-programs.
inline bool mulCompileFold(SetupCtx& setupCtx, const std::vector<HintFieldValue>& values,
                           const uint64_t* coef, uint8_t nCoef, uint64_t konst,
                           uint64_t bufferCommitSize, uint64_t domainSize, MulProgram& out) {
    out = MulProgram{};
    // Top of the temp space, so a spliced sub-program keeps its own numbering unremapped.
    const uint16_t SUM = MUL_PROG_MAX_TEMP - 1, SCR = MUL_PROG_MAX_TEMP - 2;
    bool haveSum = false;
    for (uint8_t e = 0; e < nCoef; ++e) {
        if (coef[e] == 0) continue;              // a column the fit found irrelevant
        if (e >= values.size()) { mulProgFailReason() = "fold wider than the tuple"; return false; }
        MulOperandDev o{};
        if (values[e].operand == opType::tmp) {
            // Splice the sub-program; its last dst holds the result. Its temps are consumed
            // before the next term, so terms may reuse them.

            MulProgram sub;
            if (!mulCompileField(setupCtx, values[e], bufferCommitSize, domainSize, sub))
                return false;                       // reason already set by the compiler
            if (sub.insns.empty()) { mulProgFailReason() = "fold term compiled to nothing"; return false; }
            for (const MulInsnDev& in : sub.insns) {
                if (in.dst >= SCR) { mulProgFailReason() = "fold term needs the accumulator's temps"; return false; }
                if ((in.a.kind == MUL_OPND_TEMP && in.a.tmp >= SCR) ||
                    (in.b.kind == MUL_OPND_TEMP && in.b.tmp >= SCR)) {
                    mulProgFailReason() = "fold term reads the accumulator's temps"; return false;
                }
            }
            out.insns.insert(out.insns.end(), sub.insns.begin(), sub.insns.end());
            o = mulOpTemp(sub.insns.back().dst);
        } else if (!mulOperandOfField(setupCtx, values[e], bufferCommitSize, domainSize, o)) {
            return false;
        }
        if (coef[e] != 1) { mulEmit(out, SCR, o, 2 /*mul*/, mulOpConst(coef[e])); o = mulOpTemp(SCR); }
        if (!haveSum) { mulEmit(out, SUM, o, 0 /*add*/, mulOpConst(0)); haveSum = true; }
        else          { mulEmit(out, SUM, mulOpTemp(SUM), 0, o); }
    }
    if (!haveSum) { mulProgFailReason() = "fold has no terms"; return false; }
    if (konst != 0) mulEmit(out, SUM, mulOpTemp(SUM), 0, mulOpConst(konst));
    out.ok = true;
    return true;
}

#endif
