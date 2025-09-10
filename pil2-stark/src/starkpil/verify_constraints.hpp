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
    uint64_t nrows;
    bool skip;
    uint64_t n_print_constraints;
    ConstraintRowInfo *rows;
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


void verifyConstraint(SetupCtx& setupCtx, Goldilocks::Element* dest, uint64_t constraintId, ConstraintInfo& constraintInfo) {        
    constraintInfo.nrows = 0;

    uint64_t N = (1 << setupCtx.starkInfo.starkStruct.nBits);

    std::vector<ConstraintRowInfo> constraintInvalidRows;
    std::vector<bool> invalidRow(N, false);

#pragma omp parallel for
    for(uint64_t i = 0; i < N; ++i) {
        auto [isValid, rowInfo] = checkConstraint(dest, setupCtx.expressionsBin.constraintsInfoDebug[constraintId], i);
        if (!isValid) {
            invalidRow[i] = true;
            #pragma omp atomic
            constraintInfo.nrows++;
        }
    }
    
    uint64_t invalid_num_rows_print = std::min(constraintInfo.nrows, uint64_t(constraintInfo.n_print_constraints));
    uint64_t num_rows = invalid_num_rows_print;
    uint64_t h = num_rows / 2;
    uint64_t count = 0;
    uint64_t found = 0;
    while (num_rows > h) {
        if (invalidRow[count]) {
            auto [_, rowInfo] = checkConstraint(dest, setupCtx.expressionsBin.constraintsInfoDebug[constraintId], count);
            constraintInfo.rows[found++] = rowInfo;
            num_rows--;
        }
        count++;
    }
    
    count = N - 1;
    while(num_rows > 0) {
        if (invalidRow[count]) {
            auto [_, rowInfo] = checkConstraint(dest, setupCtx.expressionsBin.constraintsInfoDebug[constraintId], count);
            constraintInfo.rows[found++] = rowInfo;
            num_rows--;
        }
        if(count == 0) break;
        count--;
    }

    std::reverse(
        constraintInfo.rows   + h,
        constraintInfo.rows   + invalid_num_rows_print
    );
}

void verifyConstraints(SetupCtx& setupCtx, StepsParams &params, ConstraintInfo *constraintsInfo) {
    
    uint64_t N = (1 << setupCtx.starkInfo.starkStruct.nBits);

    Goldilocks::Element* pBuffer = &params.aux_trace[setupCtx.starkInfo.mapOffsets[std::make_pair("q", true)]];

    ProverHelpers proverHelpers;
    ExpressionsPack expressionsCtx(setupCtx, &proverHelpers);

    for (uint64_t i = 0; i < setupCtx.expressionsBin.constraintsInfoDebug.size(); i++) {
        constraintsInfo[i].id = i;
        constraintsInfo[i].stage = setupCtx.expressionsBin.constraintsInfoDebug[i].stage;
        constraintsInfo[i].imPol = setupCtx.expressionsBin.constraintsInfoDebug[i].imPol;

        if(!constraintsInfo[i].skip) {
            Dest constraintDest(pBuffer, N);
            constraintDest.addParams(i, setupCtx.expressionsBin.constraintsInfoDebug[i].destDim);
            expressionsCtx.calculateExpressions(params, constraintDest, N, false, false, false, true);
            verifyConstraint(setupCtx, pBuffer, i, constraintsInfo[i]);
        }
    }    
}
