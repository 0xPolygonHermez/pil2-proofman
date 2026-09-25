#ifndef MULTIPLICITY_EXTRACT_HPP
#define MULTIPLICITY_EXTRACT_HPP

#include <cstdint>
#include <vector>
#include <map>
#include <string>
#include "multiplicity_bytecode.hpp"
#include "multiplicity_program.hpp"
#include "expressions_bin.hpp"
#include "setup_ctx.hpp"

// Reduce a hint field to `sum(coef_i * col_i) + konst`, once per (air, hint) at registration --
// never per row. The walk itself is in multiplicity_bytecode.hpp; this is the SetupCtx adapter.
inline MulPoly mulExtractExpr(SetupCtx& setupCtx, uint64_t expId, uint64_t bufferCommitSize) {
    auto it = setupCtx.expressionsBin.expressionsInfo.find(expId);
    if (it == setupCtx.expressionsBin.expressionsInfo.end()) return MulPoly{};
    const ParserParams& pp = it->second;
    if (pp.destDim != 1) return MulPoly{};
    const ParserArgs& pa = setupCtx.expressionsBin.expressionsBinArgsExpressions;

    MulByteCode bc;
    bc.ops       = &pa.ops[pp.opsOffset];
    bc.args      = &pa.args[pp.argsOffset];
    bc.numbers   = (const uint64_t*)pa.numbers;
    bc.nOps      = pp.nOps;
    bc.nTemp1    = pp.nTemp1;
    bc.base      = (uint32_t)bufferCommitSize;
    bc.nSections = (uint32_t)(setupCtx.starkInfo.nStages + 1);
    bc.customBase = (uint32_t)(setupCtx.starkInfo.nStages + 4);
    bc.nCustom = (uint32_t)setupCtx.starkInfo.customCommits.size();
    return mulWalkBytecode(bc);
}

// The same reduction for a whole hint field value. `caseNoOperations__` is why the operand kinds
// are handled here and not only in the bytecode: a bare column or number carries no bytecode at all.
inline MulPoly mulExtractField(SetupCtx& setupCtx, const HintFieldValue& v, uint64_t bufferCommitSize) {
    switch (v.operand) {
        case opType::cm: {
            const PolMap& p = setupCtx.starkInfo.cmPolsMap[v.id];
            if (p.dim != 1) break;
            return mulPolyOf(mulLinColumn((uint16_t)p.stage, (uint16_t)p.stagePos, (uint16_t)v.rowOffsetIndex));
        }
        case opType::const_: {
            const PolMap& p = setupCtx.starkInfo.constPolsMap[v.id];
            if (p.dim != 1) break;
            return mulPolyOf(mulLinColumn(0, (uint16_t)p.stagePos, (uint16_t)v.rowOffsetIndex));
        }
        case opType::custom: {
            const PolMap& p = setupCtx.starkInfo.customCommitsMap[v.commitId][v.id];
            if (p.dim != 1) break;
            // Encoded in the operand space the bytecode uses, so both paths resolve identically.
            const uint16_t type = (uint16_t)(setupCtx.starkInfo.nStages + 4 + v.commitId);
            return mulPolyOf(mulLinColumn(type, (uint16_t)p.stagePos, (uint16_t)v.rowOffsetIndex));
        }
        case opType::number:
            return mulPolyOf(mulLinConst(mulCanonHD(v.value)));
        case opType::public_:
            return mulPolyOf(mulLinColumn((uint16_t)(bufferCommitSize + 2), (uint16_t)v.id, 0));
        case opType::airgroupvalue:
            return mulPolyOf(mulLinColumn((uint16_t)(bufferCommitSize + 6), (uint16_t)v.id, 0));
        case opType::airvalue: {
            // airValues is laid out by stage: a stage-1 value takes one slot, a later one three.
            // Same walk addHintFieldAt does, and the reason the id alone is not the index.
            if (setupCtx.starkInfo.airValuesMap[v.id].stage != 1) break;
            uint64_t pos = 0;
            for (uint64_t i = 0; i < v.id; ++i)
                pos += setupCtx.starkInfo.airValuesMap[i].stage == 1 ? 1 : FIELD_EXTENSION;
            return mulPolyOf(mulLinColumn((uint16_t)(bufferCommitSize + 4), (uint16_t)pos, 0));
        }
        case opType::tmp:
            return mulExtractExpr(setupCtx, v.id, bufferCommitSize);
        default: break;
    }
    return MulPoly{};
}


// ---------------------------------------------------------------------------------------------
// Rung below the closed form: compile the expression instead of reducing it.
// ---------------------------------------------------------------------------------------------

// Transliterate `computeExpressions_`'s instruction stream one-for-one. Operand decoding mirrors
// mulWalkBytecode exactly -- the two must agree on what a `type` means or the same PIL would be
// read differently by the two rungs.
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
        // `coef` is 1: the program multiplies explicitly, it does not fold coefficients.
        if (!mulTermToDev(setupCtx, MulLinTerm{ type, argIdx, argOff, 1 }, domainSize, o.term))
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
    out.nTemp = bc.nTemp1 + 1;
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
    bc.customBase = (uint32_t)(setupCtx.starkInfo.nStages + 4);
    bc.nCustom = (uint32_t)setupCtx.starkInfo.customCommits.size();
    return mulCompileBytecode(setupCtx, bc, domainSize, out);
}

// A field that is a bare column or number carries no bytecode at all (caseNoOperations__), so it
// compiles to the single instruction `x + 0` rather than needing a special case in the kernel.
inline bool mulCompileField(SetupCtx& setupCtx, const HintFieldValue& v, uint64_t bufferCommitSize,
                            uint64_t domainSize, MulProgram& out) {
    if (v.operand == opType::tmp)
        return mulCompileExpr(setupCtx, v.id, bufferCommitSize, domainSize, out);

    const MulPoly q = mulExtractField(setupCtx, v, bufferCommitSize);
    if (!q.ok || q.nProd != 1 || q.p[0].hasProduct || q.p[0].n > 1) {
        mulProgFailReason() = "operand kind not modelled";
        return false;
    }
    out = MulProgram{};
    MulInsnDev in{};
    in.op = 0;
    if (q.p[0].n == 1) {
        in.a.kind = MUL_OPND_COL;
        if (!mulTermToDev(setupCtx, q.p[0].t[0], domainSize, in.a.term)) {
            mulProgFailReason() = "operand address not resolvable";
            return false;
        }
    } else {
        in.a.kind = MUL_OPND_CONST;
        in.a.konst = q.p[0].konst;
    }
    in.b.kind = MUL_OPND_CONST;
    in.b.konst = q.p[0].n == 1 ? q.p[0].konst : 0;
    out.insns.push_back(in);
    out.nTemp = 1;
    out.ok = true;
    return true;
}


#endif
