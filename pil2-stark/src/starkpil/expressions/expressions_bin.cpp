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

ExpressionsBin::ExpressionsBin(string starkInfoFile, string expressionsInfoFile, string expressionsBinFile, bool globalBin, bool verifierBin) {
    write = true;
    if(globalBin) {
        ExpressionsInfo expsInfo(expressionsInfoFile);
        writeGlobalExpressionsBin(expressionsBinFile, expsInfo);
    } else if(verifierBin) {
        ExpressionsInfo expsInfo(starkInfoFile, expressionsInfoFile, true);
        writeVerifierBin(expressionsBinFile, expsInfo);
    } else {
        ExpressionsInfo expsInfo(starkInfoFile, expressionsInfoFile);
        writeExpressionsBin(expressionsBinFile, expsInfo);
    }
};

void ExpressionsBin::writeGlobalExpressionsBin(string binFile, ExpressionsInfo& expsInfo) {
    BinFileUtils::BinFileWriter fdBinFile(binFile, "chps", 1, N_GLOBAL_SECTIONS);

    // Write ConstraintsSection
    writeGlobalConstraintsSection(fdBinFile, GLOBAL_CONSTRAINTS_SECTION, expsInfo.constraintsInfo, expsInfo.numbersConstraints);

    // Write HintsSection
    writeGlobalHintsSection(fdBinFile, GLOBAL_HINTS_SECTION, expsInfo.hintsInfo);
}


void ExpressionsBin::writeExpressionsBin(string binFile, ExpressionsInfo& expsInfo) {
    BinFileUtils::BinFileWriter fdBinFile(binFile, "chps", 1, N_SECTIONS);

    // Write ExpressionsSection
    writeExpressionsSection(fdBinFile, EXPRESSIONS_SECTION, expsInfo.expressionsInfo, expsInfo.numbersExps, expsInfo.maxTmp1, expsInfo.maxTmp3, expsInfo.maxArgs, expsInfo.maxOps);
    
    // Write ConstraintsSection
    writeConstraintsSection(fdBinFile, CONSTRAINTS_SECTION, expsInfo.constraintsInfo, expsInfo.numbersConstraints);

    // Write HintsSection
    writeHintsSection(fdBinFile, HINTS_SECTION, expsInfo.hintsInfo);
}

void ExpressionsBin::writeVerifierBin(string binFile, ExpressionsInfo& expsInfo) {
    BinFileUtils::BinFileWriter fdBinFile(binFile, "chps", 1, 1);

    // Write ExpressionsSection
    writeExpressionsSection(fdBinFile, EXPRESSIONS_SECTION, expsInfo.expressionsInfo, expsInfo.numbersExps, expsInfo.maxTmp1, expsInfo.maxTmp3, expsInfo.maxArgs, expsInfo.maxOps);
}

void ExpressionsBin::writeExpressionsSection(BinFileUtils::BinFileWriter &binFile, int section, std::vector<ExpInfoBin> expressionsInfo, std::vector<uint64_t> numbersExps, uint64_t maxTmp1, uint64_t maxTmp3, uint64_t maxArgs, uint64_t maxOps) {
    binFile.startWriteSection(section);

    std::vector<uint8_t> opsExpressions;
    std::vector<uint16_t> argsExpressions;

    std::vector<uint32_t> opsExpressionsOffset, argsExpressionsOffset;

    for (uint64_t i = 0; i < expressionsInfo.size(); ++i) {
        if (i == 0) {
            opsExpressionsOffset.push_back(0);
            argsExpressionsOffset.push_back(0);
        } else {
            opsExpressionsOffset.push_back(opsExpressionsOffset[i - 1] + expressionsInfo[i - 1].ops.size());
            argsExpressionsOffset.push_back(argsExpressionsOffset[i - 1] + expressionsInfo[i - 1].args.size());
        }

        opsExpressions.insert(opsExpressions.end(), expressionsInfo[i].ops.begin(), expressionsInfo[i].ops.end());
        argsExpressions.insert(argsExpressions.end(), expressionsInfo[i].args.begin(), expressionsInfo[i].args.end());
    }

    binFile.writeU32LE(maxTmp1);
    binFile.writeU32LE(maxTmp3);
    binFile.writeU32LE(maxArgs);
    binFile.writeU32LE(maxOps);
    binFile.writeU32LE(opsExpressions.size());
    binFile.writeU32LE(argsExpressions.size());
    binFile.writeU32LE(numbersExps.size());

    uint64_t nExpressions = expressionsInfo.size();

    binFile.writeU32LE(nExpressions);

    for (uint64_t i = 0; i < nExpressions; ++i) {
        const ExpInfoBin& expInfo = expressionsInfo[i];
        
        // Write expression metadata
        binFile.writeU32LE(expInfo.expId);
        binFile.writeU32LE(expInfo.destDim);
        binFile.writeU32LE(expInfo.destId);
        binFile.writeU32LE(expInfo.stage);
        binFile.writeU32LE(expInfo.nTemp1);
        binFile.writeU32LE(expInfo.nTemp3);

        // Write ops information
        binFile.writeU32LE(expInfo.ops.size());
        binFile.writeU32LE(opsExpressionsOffset[i]);

        // Write args information
        binFile.writeU32LE(expInfo.args.size());
        binFile.writeU32LE(argsExpressionsOffset[i]);

        // Write the line string
        binFile.writeString(expInfo.line);
    }

    for(uint64_t j = 0; j < opsExpressions.size(); ++j) {
        binFile.writeU8LE(opsExpressions[j]);
    }

    for(uint64_t j = 0; j < argsExpressions.size(); ++j) {
        binFile.writeU16LE(argsExpressions[j]);
    }

    for(uint64_t j = 0; j < numbersExps.size(); ++j) {
        binFile.writeU64LE(numbersExps[j]);
    }

    binFile.endWriteSection();
}

void ExpressionsBin::writeConstraintsSection(BinFileUtils::BinFileWriter &binFile, int section, std::vector<ExpInfoBin> constraintsInfo, std::vector<uint64_t> numbersConstraints) {
    
    binFile.startWriteSection(section);

    std::vector<uint8_t> opsDebug;
    std::vector<uint16_t> argsDebug;
   
    std::vector<uint32_t> opsDebugOffset, argsDebugOffset;
   
    for (uint64_t i = 0; i < constraintsInfo.size(); ++i) {
        if (i == 0) {
            opsDebugOffset.push_back(0);
            argsDebugOffset.push_back(0);
        } else {
            opsDebugOffset.push_back(opsDebugOffset[i - 1] + constraintsInfo[i - 1].ops.size());
            argsDebugOffset.push_back(argsDebugOffset[i - 1] + constraintsInfo[i - 1].args.size());
        }

        opsDebug.insert(opsDebug.end(), constraintsInfo[i].ops.begin(), constraintsInfo[i].ops.end());
        argsDebug.insert(argsDebug.end(), constraintsInfo[i].args.begin(), constraintsInfo[i].args.end());
    }

    binFile.writeU32LE(opsDebug.size());
    binFile.writeU32LE(argsDebug.size());
    binFile.writeU32LE(numbersConstraints.size());

    uint64_t nConstraints = constraintsInfo.size();

    binFile.writeU32LE(nConstraints);

    for (uint64_t i = 0; i < nConstraints; ++i) {
        const ExpInfoBin& constraintInfo = constraintsInfo[i];
        
        // Write expression metadata
        binFile.writeU32LE(constraintInfo.stage);
        binFile.writeU32LE(constraintInfo.destDim);
        binFile.writeU32LE(constraintInfo.destId);
        binFile.writeU32LE(constraintInfo.firstRow);
        binFile.writeU32LE(constraintInfo.lastRow);
        binFile.writeU32LE(constraintInfo.nTemp1);
        binFile.writeU32LE(constraintInfo.nTemp3);

        // Write ops information
        binFile.writeU32LE(constraintInfo.ops.size());
        binFile.writeU32LE(opsDebugOffset[i]);

        // Write args information
        binFile.writeU32LE(constraintInfo.args.size());
        binFile.writeU32LE(argsDebugOffset[i]);

        binFile.writeU32LE(constraintInfo.imPol);

        // Write the line string
        binFile.writeString(constraintInfo.line);
    }

    for(uint64_t j = 0; j < opsDebug.size(); ++j) {
        binFile.writeU8LE(opsDebug[j]);
    }

    for(uint64_t j = 0; j < argsDebug.size(); ++j) {
        binFile.writeU16LE(argsDebug[j]);
    }

    for(uint64_t j = 0; j < numbersConstraints.size(); ++j) {
        binFile.writeU64LE(numbersConstraints[j]);
    }

    binFile.endWriteSection();
}

void ExpressionsBin::writeHintsSection(BinFileUtils::BinFileWriter &binFile, int section, std::vector<HintInfo> hintsInfo) {
    binFile.startWriteSection(section);

    uint64_t nHints = hintsInfo.size();
    binFile.writeU32LE(nHints);
    
    for(uint64_t h = 0; h < nHints; h++) {
        HintInfo hint = hintsInfo[h];
        binFile.writeString(hint.name);

        uint32_t nFields = hint.fields.size();
        binFile.writeU32LE(nFields);
        for(uint64_t f = 0; f < nFields; f++) {
            binFile.writeString(hint.fields[f].name);
            uint64_t nValues = hint.fields[f].values.size();
            binFile.writeU32LE(nValues);
            for(uint64_t v = 0; v < nValues; v++) {
                HintValues value = hint.fields[f].values[v];
                binFile.writeString(opType2string(value.op));
                if(value.op == opType::number) {
                    binFile.writeU64LE(value.value);
                } else if(value.op == opType::string_) {
                    binFile.writeString(value.string_);
                } else {
                    binFile.writeU32LE(value.id);
                }
                
                if(value.op == opType::custom || value.op == opType::const_ || value.op == opType::cm) {
                    binFile.writeU32LE(value.rowOffsetIndex);
                }

                if(value.op == opType::tmp) {
                    binFile.writeU32LE(value.dim);
                }
                if(value.op == opType::custom) {
                    binFile.writeU32LE(value.commitId);
                }

                binFile.writeU32LE(value.pos.size());
                for(uint64_t p = 0; p < value.pos.size(); ++p) {
                    binFile.writeU32LE(value.pos[p]);
                }
                
            }
        }
    }


    binFile.endWriteSection();
}


void ExpressionsBin::writeGlobalConstraintsSection(BinFileUtils::BinFileWriter &binFile, int section, std::vector<ExpInfoBin> constraintsInfo, std::vector<uint64_t> numbersConstraints) {
    
    binFile.startWriteSection(section);

    std::vector<uint8_t> opsDebug;
    std::vector<uint16_t> argsDebug;

    std::vector<uint32_t> opsDebugOffset, argsDebugOffset;

    for (uint64_t i = 0; i < constraintsInfo.size(); ++i) {
        if (i == 0) {
            opsDebugOffset.push_back(0);
            argsDebugOffset.push_back(0);
        } else {
            opsDebugOffset.push_back(opsDebugOffset[i - 1] + constraintsInfo[i - 1].ops.size());
            argsDebugOffset.push_back(argsDebugOffset[i - 1] + constraintsInfo[i - 1].args.size());
        }

        opsDebug.insert(opsDebug.end(), constraintsInfo[i].ops.begin(), constraintsInfo[i].ops.end());
        argsDebug.insert(argsDebug.end(), constraintsInfo[i].args.begin(), constraintsInfo[i].args.end());
    }

    binFile.writeU32LE(opsDebug.size());
    binFile.writeU32LE(argsDebug.size());
    binFile.writeU32LE(numbersConstraints.size());

    uint64_t nConstraints = constraintsInfo.size();

    binFile.writeU32LE(nConstraints);

    for (uint64_t i = 0; i < nConstraints; ++i) {
        const ExpInfoBin& constraintInfo = constraintsInfo[i];
        
        // Write expression metadata
        binFile.writeU32LE(constraintInfo.destDim);
        binFile.writeU32LE(constraintInfo.destId);
        binFile.writeU32LE(constraintInfo.nTemp1);
        binFile.writeU32LE(constraintInfo.nTemp3);

        // Write ops information
        binFile.writeU32LE(constraintInfo.ops.size());
        binFile.writeU32LE(opsDebugOffset[i]);

        // Write args information
        binFile.writeU32LE(constraintInfo.args.size());
        binFile.writeU32LE(argsDebugOffset[i]);

        // Write the line string
        binFile.writeString(constraintInfo.line);
    }

    for(uint64_t j = 0; j < opsDebug.size(); ++j) {
        binFile.writeU8LE(opsDebug[j]);
    }

    for(uint64_t j = 0; j < argsDebug.size(); ++j) {
        binFile.writeU16LE(argsDebug[j]);
    }

    for(uint64_t j = 0; j < numbersConstraints.size(); ++j) {
        binFile.writeU64LE(numbersConstraints[j]);
    }

    binFile.endWriteSection();
}

void ExpressionsBin::writeGlobalHintsSection(BinFileUtils::BinFileWriter &binFile, int section, std::vector<HintInfo> hintsInfo) {
    binFile.startWriteSection(section);

    uint64_t nHints = hintsInfo.size();
    binFile.writeU32LE(nHints);
    
    for(uint64_t h = 0; h < nHints; h++) {
        HintInfo hint = hintsInfo[h];
        binFile.writeString(hint.name);

        uint32_t nFields = hint.fields.size();
        binFile.writeU32LE(nFields);
        for(uint64_t f = 0; f < nFields; f++) {
            binFile.writeString(hint.fields[f].name);
            uint64_t nValues = hint.fields[f].values.size();
            binFile.writeU32LE(nValues);
            for(uint64_t v = 0; v < nValues; v++) {
                HintValues value = hint.fields[f].values[v];
                binFile.writeString(opType2string(value.op));
                if(value.op == opType::number) {
                    binFile.writeU64LE(value.value);
                } else if(value.op == opType::string_) {
                    binFile.writeString(value.string_);
                } else if(value.op == opType::airgroupvalue){ 
                    binFile.writeU32LE(value.airgroupId);
                    binFile.writeU32LE(value.id);
                } else {
                    binFile.writeU32LE(value.id);
                }
                
                binFile.writeU32LE(value.pos.size());
                for(uint64_t p = 0; p < value.pos.size(); ++p) {
                    binFile.writeU32LE(value.pos[p]);
                }
                
            }
        }
    }


    binFile.endWriteSection();
}


void ExpressionsBin::loadExpressionsBin(BinFileUtils::BinFile *expressionsBin) {
    expressionsBin->startReadSection(EXPRESSIONS_SECTION);

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
    expressionsBin->startReadSection(CONSTRAINTS_SECTION);

    nOpsDebug = expressionsBin->readU32LE();
    nArgsDebug = expressionsBin->readU32LE();
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
    expressionsBin->startReadSection(HINTS_SECTION);

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
    expressionsBin->startReadSection(EXPRESSIONS_SECTION);
    
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