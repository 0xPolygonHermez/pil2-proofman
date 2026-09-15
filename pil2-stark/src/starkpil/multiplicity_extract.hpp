#ifndef MULTIPLICITY_EXTRACT_HPP
#define MULTIPLICITY_EXTRACT_HPP

#include <cstdint>
#include <vector>
#include "multiplicity_bytecode.hpp"
#include "expressions_bin.hpp"
#include "setup_ctx.hpp"

// Reduce a hint field to `sum(coef_i * col_i) + konst`, once per (air, hint) at registration --
// never per row. The walk itself is in multiplicity_bytecode.hpp; this is the SetupCtx adapter.
inline MulLinForm mulExtractExpr(SetupCtx& setupCtx, uint64_t expId, uint64_t bufferCommitSize) {
    auto it = setupCtx.expressionsBin.expressionsInfo.find(expId);
    if (it == setupCtx.expressionsBin.expressionsInfo.end()) return MulLinForm{};
    const ParserParams& pp = it->second;
    if (pp.destDim != 1) return MulLinForm{};
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
inline MulLinForm mulExtractField(SetupCtx& setupCtx, const HintFieldValue& v, uint64_t bufferCommitSize) {
    switch (v.operand) {
        case opType::cm: {
            const PolMap& p = setupCtx.starkInfo.cmPolsMap[v.id];
            if (p.dim != 1) break;
            return mulLinColumn((uint16_t)p.stage, (uint16_t)p.stagePos, (uint16_t)v.rowOffsetIndex);
        }
        case opType::const_: {
            const PolMap& p = setupCtx.starkInfo.constPolsMap[v.id];
            if (p.dim != 1) break;
            return mulLinColumn(0, (uint16_t)p.stagePos, (uint16_t)v.rowOffsetIndex);
        }
        case opType::custom: {
            const PolMap& p = setupCtx.starkInfo.customCommitsMap[v.commitId][v.id];
            if (p.dim != 1) break;
            // Encoded in the operand space the bytecode uses, so both paths resolve identically.
            const uint16_t type = (uint16_t)(setupCtx.starkInfo.nStages + 4 + v.commitId);
            return mulLinColumn(type, (uint16_t)p.stagePos, (uint16_t)v.rowOffsetIndex);
        }
        case opType::number:
            return mulLinConst(mulCanonHD(v.value));
        case opType::public_:
            return mulLinColumn((uint16_t)(bufferCommitSize + 2), (uint16_t)v.id, 0);
        case opType::airgroupvalue:
            return mulLinColumn((uint16_t)(bufferCommitSize + 6), (uint16_t)v.id, 0);
        case opType::airvalue: {
            // airValues is laid out by stage: a stage-1 value takes one slot, a later one three.
            // Same walk addHintFieldAt does, and the reason the id alone is not the index.
            if (setupCtx.starkInfo.airValuesMap[v.id].stage != 1) break;
            uint64_t pos = 0;
            for (uint64_t i = 0; i < v.id; ++i)
                pos += setupCtx.starkInfo.airValuesMap[i].stage == 1 ? 1 : FIELD_EXTENSION;
            return mulLinColumn((uint16_t)(bufferCommitSize + 4), (uint16_t)pos, 0);
        }
        case opType::tmp:
            return mulExtractExpr(setupCtx, v.id, bufferCommitSize);
        default: break;
    }
    return MulLinForm{};
}

#endif
