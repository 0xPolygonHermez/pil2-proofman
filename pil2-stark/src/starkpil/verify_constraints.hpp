#include "expressions_ctx.hpp"

struct ConstraintRowInfo {
    uint64_t row;
    uint64_t dim;
    uint64_t value[3];
};

struct ConstraintInfo {
    uint64_t id;
    uint64_t stage;
    bool imPol;
    const char* line;
    uint64_t nrows;
    ConstraintRowInfo rows[10];
};

struct ConstraintsResults {
    uint64_t nConstraints;
    ConstraintInfo* constraintInfo;
};

std::tuple<bool, ConstraintRowInfo> checkConstraint(Goldilocks::Element* dest, ParserParams& parserParams, uint64_t row) {
    bool isValid = true;
    ConstraintRowInfo rowInfo;
    rowInfo.row = row;
    rowInfo.dim = parserParams.destDim;
    if(row < parserParams.firstRow || row > parserParams.lastRow) {
            rowInfo.value[0] = 0;
            rowInfo.value[1] = 0;
            rowInfo.value[2] = 0;
    } else {
            if(parserParams.destDim == 1) {
            rowInfo.value[0] = Goldilocks::toU64(dest[row]);
            rowInfo.value[1] = 0;
            rowInfo.value[2] = 0;
            if(rowInfo.value[0] != 0) isValid = false;
        } else if(parserParams.destDim == FIELD_EXTENSION) {
            rowInfo.value[0] = Goldilocks::toU64(dest[FIELD_EXTENSION*row]);
            rowInfo.value[1] = Goldilocks::toU64(dest[FIELD_EXTENSION*row + 1]);
            rowInfo.value[2] = Goldilocks::toU64(dest[FIELD_EXTENSION*row + 2]);
            if(rowInfo.value[0] != 0 || rowInfo.value[1] != 0 || rowInfo.value[2] != 0) isValid = false;
        } else {
            exitProcess();
            exit(-1);
        }
    }
    

    return std::make_tuple(isValid, rowInfo);
}


ConstraintInfo verifyConstraint(SetupCtx& setupCtx, Goldilocks::Element* dest, uint64_t constraintId) {        
    ConstraintInfo constraintInfo;
    constraintInfo.id = constraintId;
    constraintInfo.stage = setupCtx.expressionsBin.constraintsInfoDebug[constraintId].stage;
    constraintInfo.imPol = setupCtx.expressionsBin.constraintsInfoDebug[constraintId].imPol;
    constraintInfo.line = setupCtx.expressionsBin.constraintsInfoDebug[constraintId].line.c_str();
    constraintInfo.nrows = 0;

    uint64_t N = (1 << setupCtx.starkInfo.starkStruct.nBits);

    std::vector<ConstraintRowInfo> constraintInvalidRows;
    for(uint64_t i = 0; i < N; ++i) {
        auto [isValid, rowInfo] = checkConstraint(dest, setupCtx.expressionsBin.constraintsInfoDebug[constraintId], i);
        if(!isValid) {
            constraintInvalidRows.push_back(rowInfo);
            constraintInfo.nrows++;
        }
    }

    uint64_t num_rows = std::min(constraintInfo.nrows, uint64_t(10));
    uint64_t h = num_rows / 2;
    for(uint64_t i = 0; i < h; ++i) {
        constraintInfo.rows[i] = constraintInvalidRows[i];
    }

    for(uint64_t i = h; i < num_rows; ++i) {
        if(constraintInfo.nrows > num_rows) {
            constraintInfo.rows[i] = constraintInvalidRows[constraintInvalidRows.size() - num_rows + i];
        } else {
            constraintInfo.rows[i] = constraintInvalidRows[i];
        }
    }

    return constraintInfo;
}

ConstraintsResults *verifyConstraints(SetupCtx& setupCtx, StepsParams &params) {
    
    ConstraintsResults *constraintsInfo = new ConstraintsResults();
    constraintsInfo->nConstraints = setupCtx.expressionsBin.constraintsInfoDebug.size();
    constraintsInfo->constraintInfo = new ConstraintInfo[constraintsInfo->nConstraints];

    uint64_t N = (1 << setupCtx.starkInfo.starkStruct.nBits);

    uint64_t nPols = 0;
    for(uint64_t stage = 1; stage <= setupCtx.starkInfo.nStages; stage++) {
        nPols += setupCtx.starkInfo.mapSectionsN["cm" + to_string(stage)];
    }

    // TODO: REUSE MEMORY
    Goldilocks::Element* pBuffer = new Goldilocks::Element[setupCtx.expressionsBin.constraintsInfoDebug.size() * N * FIELD_EXTENSION];

    std::vector<Dest> dests;
    for (uint64_t i = 0; i < setupCtx.expressionsBin.constraintsInfoDebug.size(); i++) {
        Dest constraintDest(&pBuffer[i*FIELD_EXTENSION*N]);
        constraintDest.addParams(setupCtx.expressionsBin.constraintsInfoDebug[i]);
        dests.push_back(constraintDest);
    }

#ifdef __AVX512__
    ExpressionsAvx512 expressionsCtx(setupCtx);
#elif defined(__AVX2__)
    ExpressionsAvx expressionsCtx(setupCtx);
#else
    ExpressionsPack expressionsCtx(setupCtx);
#endif

    expressionsCtx.calculateExpressions(params, setupCtx.expressionsBin.expressionsBinArgsConstraints, dests, uint64_t(1 << setupCtx.starkInfo.starkStruct.nBits));

#pragma omp parallel for
    for (uint64_t i = 0; i < setupCtx.expressionsBin.constraintsInfoDebug.size(); i++) {
        auto constraintInfo = verifyConstraint(setupCtx, dests[i].dest, i);
        constraintsInfo->constraintInfo[i] = constraintInfo;
    }
    
    delete pBuffer;
    return constraintsInfo;
}
