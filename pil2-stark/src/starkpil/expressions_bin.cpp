#include "expressions_bin.hpp"

ExpressionsBin::ExpressionsBin(string file, bool globalBin, bool verifierBin) {
    std::unique_ptr<BinFileUtils::BinFile> binFile = BinFileUtils::openExisting(file, "chps", 1);

    if(globalBin) {
        loadGlobalBin(binFile.get());
    } else if(verifierBin) {
        loadVerifierBin(binFile.get());
    } else {
        loadExpressionsBin(binFile.get());
    }
}

void ExpressionsBin::loadExpressionsBin(BinFileUtils::BinFile *expressionsBin) {
    expressionsBin->startReadSection(BINARY_EXPRESSIONS_SECTION);

    maxTmp1 = expressionsBin->readU32LE();
    maxTmp3 = expressionsBin->readU32LE();
    maxArgs = expressionsBin->readU32LE();
    maxOps = expressionsBin->readU32LE();

    uint32_t nOpsExpressions = expressionsBin->readU32LE();
    nOpsTotal = nOpsExpressions;
    uint32_t nArgsExpressions = expressionsBin->readU32LE();
    nArgsTotal = nArgsExpressions;
    uint32_t nNumbersExpressions = expressionsBin->readU32LE();
    
    expressionsBinArgsExpressions.ops = new uint8_t[nOpsExpressions];
    expressionsBinArgsExpressions.args = new uint16_t[nArgsExpressions];
    expressionsBinArgsExpressions.numbers = new Goldilocks::Element[nNumbersExpressions];
    expressionsBinArgsExpressions.nNumbers = nNumbersExpressions;

    uint64_t nExpressions = expressionsBin->readU32LE();

    for(uint64_t i = 0; i < nExpressions; ++i) {
        ParserParams parserParamsExpression;

        uint32_t expId = expressionsBin->readU32LE();
        
        parserParamsExpression.expId = expId;
        parserParamsExpression.destDim = expressionsBin->readU32LE();
        parserParamsExpression.destId = expressionsBin->readU32LE();
        parserParamsExpression.stage = expressionsBin->readU32LE();

        parserParamsExpression.nTemp1 = expressionsBin->readU32LE();
        parserParamsExpression.nTemp3 = expressionsBin->readU32LE();

        parserParamsExpression.nOps = expressionsBin->readU32LE();
        parserParamsExpression.opsOffset = expressionsBin->readU32LE();

        parserParamsExpression.nArgs = expressionsBin->readU32LE();
        parserParamsExpression.argsOffset = expressionsBin->readU32LE();

        parserParamsExpression.line = expressionsBin->readString();

        expressionsInfo[expId] = parserParamsExpression;
    }

    for(uint64_t j = 0; j < nOpsExpressions; ++j) {
        expressionsBinArgsExpressions.ops[j] = expressionsBin->readU8LE();
    }
    for(uint64_t j = 0; j < nArgsExpressions; ++j) {
        expressionsBinArgsExpressions.args[j] = expressionsBin->readU16LE();
    }
    for(uint64_t j = 0; j < nNumbersExpressions; ++j) {
        expressionsBinArgsExpressions.numbers[j] = Goldilocks::fromU64(expressionsBin->readU64LE());
    }

    expressionsBin->endReadSection();
    expressionsBin->startReadSection(BINARY_CONSTRAINTS_SECTION);

    uint32_t nOpsDebug = expressionsBin->readU32LE();
    uint32_t nArgsDebug = expressionsBin->readU32LE();
    uint32_t nNumbersDebug = expressionsBin->readU32LE();
   
    expressionsBinArgsConstraints.ops = new uint8_t[nOpsDebug];
    expressionsBinArgsConstraints.args = new uint16_t[nArgsDebug];
    expressionsBinArgsConstraints.numbers = new Goldilocks::Element[nNumbersDebug];
    expressionsBinArgsConstraints.nNumbers = nNumbersDebug;
    
    uint32_t nConstraints = expressionsBin->readU32LE();

    for(uint64_t i = 0; i < nConstraints; ++i) {
        ParserParams parserParamsConstraint;

        uint32_t stage = expressionsBin->readU32LE();
        parserParamsConstraint.stage = stage;
        parserParamsConstraint.expId = 0;
        
        parserParamsConstraint.destDim = expressionsBin->readU32LE();
        parserParamsConstraint.destId = expressionsBin->readU32LE();

        parserParamsConstraint.firstRow = expressionsBin->readU32LE();
        parserParamsConstraint.lastRow = expressionsBin->readU32LE();

        parserParamsConstraint.nTemp1 = expressionsBin->readU32LE();
        parserParamsConstraint.nTemp3 = expressionsBin->readU32LE();

        parserParamsConstraint.nOps = expressionsBin->readU32LE();
        parserParamsConstraint.opsOffset = expressionsBin->readU32LE();

        parserParamsConstraint.nArgs = expressionsBin->readU32LE();
        parserParamsConstraint.argsOffset = expressionsBin->readU32LE();

        parserParamsConstraint.imPol = bool(expressionsBin->readU32LE());
        parserParamsConstraint.line = expressionsBin->readString();

        constraintsInfoDebug.push_back(parserParamsConstraint);
    }


    for(uint64_t j = 0; j < nOpsDebug; ++j) {
        expressionsBinArgsConstraints.ops[j] = expressionsBin->readU8LE();
    }
    for(uint64_t j = 0; j < nArgsDebug; ++j) {
        expressionsBinArgsConstraints.args[j] = expressionsBin->readU16LE();
    }
    for(uint64_t j = 0; j < nNumbersDebug; ++j) {
        expressionsBinArgsConstraints.numbers[j] = Goldilocks::fromU64(expressionsBin->readU64LE());
    }

    expressionsBin->endReadSection();
    expressionsBin->startReadSection(BINARY_HINTS_SECTION);

    uint32_t nHints = expressionsBin->readU32LE();

    for(uint64_t h = 0; h < nHints; h++) {
        Hint hint;
        hint.name = expressionsBin->readString();

        uint32_t nFields = expressionsBin->readU32LE();

        for(uint64_t f = 0; f < nFields; f++) {
            HintField hintField;
            std::string name = expressionsBin->readString();
            hintField.name = name;

            uint64_t nValues = expressionsBin->readU32LE();
            for(uint64_t v = 0; v < nValues; v++) {
                HintFieldValue hintFieldValue;
                std::string operand = expressionsBin->readString();
                hintFieldValue.operand = string2opType(operand);
                if(hintFieldValue.operand == opType::number) {
                    hintFieldValue.value = expressionsBin->readU64LE();
                } else if(hintFieldValue.operand == opType::string_) {
                    hintFieldValue.stringValue = expressionsBin->readString();
                } else {
                    hintFieldValue.id = expressionsBin->readU32LE();
                }
                
                if(hintFieldValue.operand == opType::custom || hintFieldValue.operand == opType::const_ || hintFieldValue.operand == opType::cm) {
                    hintFieldValue.rowOffsetIndex = expressionsBin->readU32LE();
                }

                if(hintFieldValue.operand == opType::tmp) {
                    hintFieldValue.dim = expressionsBin->readU32LE();
                }
                if(hintFieldValue.operand == opType::custom) {
                    hintFieldValue.commitId = expressionsBin->readU32LE();
                }
                uint64_t nPos = expressionsBin->readU32LE();
                for(uint64_t p = 0; p < nPos; ++p) {
                    uint32_t pos = expressionsBin->readU32LE();
                    hintFieldValue.pos.push_back(pos);
                }
                hintField.values.push_back(hintFieldValue);
            }
            
            hint.fields.push_back(hintField);
        }

        hints.push_back(hint);
    }

    expressionsBin->endReadSection();
}

void ExpressionsBin::loadVerifierBin(BinFileUtils::BinFile *expressionsBin) {
    expressionsBin->startReadSection(BINARY_EXPRESSIONS_SECTION);
    
    maxTmp1 = expressionsBin->readU32LE();
    maxTmp3 = expressionsBin->readU32LE();
    maxArgs = expressionsBin->readU32LE();
    maxOps = expressionsBin->readU32LE();

    uint32_t nOpsExpressions = expressionsBin->readU32LE();
    uint32_t nArgsExpressions = expressionsBin->readU32LE();
    uint32_t nNumbersExpressions = expressionsBin->readU32LE();
   
    expressionsBinArgsExpressions.ops = new uint8_t[nOpsExpressions];
    expressionsBinArgsExpressions.args = new uint16_t[nArgsExpressions];
    expressionsBinArgsExpressions.numbers = new Goldilocks::Element[nNumbersExpressions];
    expressionsBinArgsExpressions.nNumbers = nNumbersExpressions;

    uint64_t nExpressions = expressionsBin->readU32LE();

    for(uint64_t i = 0; i < nExpressions; ++i) {
        ParserParams parserParamsExpression;

        uint32_t expId = expressionsBin->readU32LE();
        
        parserParamsExpression.expId = expId;
        parserParamsExpression.destDim = expressionsBin->readU32LE();
        parserParamsExpression.destId = expressionsBin->readU32LE();
        parserParamsExpression.stage = expressionsBin->readU32LE();

        parserParamsExpression.nTemp1 = expressionsBin->readU32LE();
        parserParamsExpression.nTemp3 = expressionsBin->readU32LE();

        parserParamsExpression.nOps = expressionsBin->readU32LE();
        parserParamsExpression.opsOffset = expressionsBin->readU32LE();

        parserParamsExpression.nArgs = expressionsBin->readU32LE();
        parserParamsExpression.argsOffset = expressionsBin->readU32LE();

        parserParamsExpression.line = expressionsBin->readString();

        expressionsInfo[expId] = parserParamsExpression;
    }

    for(uint64_t j = 0; j < nOpsExpressions; ++j) {
        expressionsBinArgsExpressions.ops[j] = expressionsBin->readU8LE();
    }
    for(uint64_t j = 0; j < nArgsExpressions; ++j) {
        expressionsBinArgsExpressions.args[j] = expressionsBin->readU16LE();
    }
    for(uint64_t j = 0; j < nNumbersExpressions; ++j) {
        expressionsBinArgsExpressions.numbers[j] = Goldilocks::fromU64(expressionsBin->readU64LE());
    }

    expressionsBin->endReadSection();
}

void ExpressionsBin::loadGlobalBin(BinFileUtils::BinFile *globalBin) {
    
    globalBin->startReadSection(GLOBAL_CONSTRAINTS_SECTION);

    uint32_t nOpsDebug = globalBin->readU32LE();
    uint32_t nArgsDebug = globalBin->readU32LE();
    uint32_t nNumbersDebug = globalBin->readU32LE();

    expressionsBinArgsConstraints.ops = new uint8_t[nOpsDebug];
    expressionsBinArgsConstraints.args = new uint16_t[nArgsDebug];
    expressionsBinArgsConstraints.numbers = new Goldilocks::Element[nNumbersDebug];
    expressionsBinArgsConstraints.nNumbers = nNumbersDebug;

    uint32_t nGlobalConstraints = globalBin->readU32LE();

    for(uint64_t i = 0; i < nGlobalConstraints; ++i) {
        ParserParams parserParamsConstraint;

        parserParamsConstraint.destDim = globalBin->readU32LE();
        parserParamsConstraint.destId = globalBin->readU32LE();

        parserParamsConstraint.nTemp1 = globalBin->readU32LE();
        parserParamsConstraint.nTemp3 = globalBin->readU32LE();

        parserParamsConstraint.nOps = globalBin->readU32LE();
        parserParamsConstraint.opsOffset = globalBin->readU32LE();

        parserParamsConstraint.nArgs = globalBin->readU32LE();
        parserParamsConstraint.argsOffset = globalBin->readU32LE();


        parserParamsConstraint.line = globalBin->readString();

        constraintsInfoDebug.push_back(parserParamsConstraint);
    }


    for(uint64_t j = 0; j < nOpsDebug; ++j) {
        expressionsBinArgsConstraints.ops[j] = globalBin->readU8LE();
    }
    for(uint64_t j = 0; j < nArgsDebug; ++j) {
        expressionsBinArgsConstraints.args[j] = globalBin->readU16LE();
    }
    for(uint64_t j = 0; j < nNumbersDebug; ++j) {
        expressionsBinArgsConstraints.numbers[j] = Goldilocks::fromU64(globalBin->readU64LE());
    }

    globalBin->endReadSection();

    globalBin->startReadSection(GLOBAL_HINTS_SECTION);

    uint32_t nHints = globalBin->readU32LE();

    for(uint64_t h = 0; h < nHints; h++) {
        Hint hint;
        hint.name = globalBin->readString();

        uint32_t nFields = globalBin->readU32LE();

        for(uint64_t f = 0; f < nFields; f++) {
            HintField hintField;
            std::string name = globalBin->readString();
            hintField.name = name;

            uint64_t nValues = globalBin->readU32LE();
            for(uint64_t v = 0; v < nValues; v++) {
                HintFieldValue hintFieldValue;
                std::string operand = globalBin->readString();
                hintFieldValue.operand = string2opType(operand);
                if(hintFieldValue.operand == opType::number) {
                    hintFieldValue.value = globalBin->readU64LE();
                } else if(hintFieldValue.operand == opType::string_) {
                    hintFieldValue.stringValue = globalBin->readString();
                } else if(hintFieldValue.operand == opType::airgroupvalue || hintFieldValue.operand == opType::airvalue) {
                    hintFieldValue.dim = globalBin->readU32LE();
                    hintFieldValue.id = globalBin->readU32LE();
                } else if(hintFieldValue.operand == opType::tmp || hintFieldValue.operand == opType::public_ || hintFieldValue.operand == opType::proofvalue) {
                    hintFieldValue.id = globalBin->readU32LE();
                } else {
                    throw new std::invalid_argument("Invalid file type");
                }
      
                uint64_t nPos = globalBin->readU32LE();
                for(uint64_t p = 0; p < nPos; ++p) {
                    uint32_t pos = globalBin->readU32LE();
                    hintFieldValue.pos.push_back(pos);
                }
                hintField.values.push_back(hintFieldValue);
            }
            
            hint.fields.push_back(hintField);
        }

        hints.push_back(hint);
    }

    globalBin->endReadSection();

}

void ExpressionsBin::getHintIdsByName(uint64_t* hintIds, std::string name) {
    uint64_t c = 0;
    for (uint64_t i = 0; i < hints.size(); ++i) {
        if (hints[i].name == name) {
            hintIds[c++] = i;
        }
    }
}


uint64_t ExpressionsBin::getNumberHintIdsByName(std::string name) {

    uint64_t nHints = 0;
    for (uint64_t i = 0; i < hints.size(); ++i) {
        if (hints[i].name == name) {
            nHints++;
        }
    }

    return nHints;
}