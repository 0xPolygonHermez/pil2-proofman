#include "expressions_bin.hpp"

ExpressionsBin::ExpressionsBin(string file, bool globalBin) {
    std::unique_ptr<BinFileUtils::BinFile> binFile = BinFileUtils::openExisting(file, "chps", 1);

    if(globalBin) {
        loadGlobalBin(binFile.get());
    } else {
        loadExpressionsBin(binFile.get());
    }
}

ExpressionsBin::ExpressionsBin(string starkInfoFile, string expressionsInfoFile, string expressionsBinFile, bool globalBin) {
    write = true;
    if(globalBin) {
        ExpressionsInfo expsInfo(expressionsInfoFile);
        writeGlobalExpressionsBin(expressionsBinFile, expsInfo);
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
    writeExpressionsSection(fdBinFile, EXPRESSIONS_SECTION, expsInfo.expressionsInfo, expsInfo.numbersExps);
    
    // Write ConstraintsSection
    writeConstraintsSection(fdBinFile, CONSTRAINTS_SECTION, expsInfo.constraintsInfo, expsInfo.numbersConstraints);

    // Write HintsSection
    writeHintsSection(fdBinFile, HINTS_SECTION, expsInfo.hintsInfo);
}

void ExpressionsBin::writeExpressionsSection(BinFileUtils::BinFileWriter &binFile, int section, std::vector<ExpInfoBin> expressionsInfo, std::vector<uint64_t> numbersExps) {
    uint64_t nCustomCommits = expressionsInfo[0].customValuesIds.size();

    binFile.startWriteSection(section);

    std::vector<uint8_t> opsExpressions;
    std::vector<uint16_t> argsExpressions, constPolsIdsExpressions, cmPolsIdsExpressions;
    std::vector<uint16_t> challengesIdsExpressions, publicsIdsExpressions, airgroupValuesIdsExpressions, airValuesIdsExpressions;
    std::vector<uint16_t> customCommitsIdsExpressions;

    std::vector<uint32_t> opsExpressionsOffset, argsExpressionsOffset, constPolsIdsExpressionsOffset;
    std::vector<uint32_t> cmPolsIdsExpressionsOffset, challengesIdsExpressionsOffset, publicsIdsExpressionsOffset;
    std::vector<uint32_t> airgroupValuesIdsExpressionsOffset, airValuesIdsExpressionsOffset;
    std::vector<std::vector<uint32_t>> customCommitsIdsExpressionsOffset;

    for (uint64_t i = 0; i < expressionsInfo.size(); ++i) {
        if (i == 0) {
            opsExpressionsOffset.push_back(0);
            argsExpressionsOffset.push_back(0);
            constPolsIdsExpressionsOffset.push_back(0);
            cmPolsIdsExpressionsOffset.push_back(0);
            challengesIdsExpressionsOffset.push_back(0);
            publicsIdsExpressionsOffset.push_back(0);
            airgroupValuesIdsExpressionsOffset.push_back(0);
            airValuesIdsExpressionsOffset.push_back(0);

            customCommitsIdsExpressionsOffset.emplace_back(nCustomCommits, 0);
        } else {
            opsExpressionsOffset.push_back(opsExpressionsOffset[i - 1] + expressionsInfo[i - 1].ops.size());
            argsExpressionsOffset.push_back(argsExpressionsOffset[i - 1] + expressionsInfo[i - 1].args.size());
            constPolsIdsExpressionsOffset.push_back(constPolsIdsExpressionsOffset[i - 1] + expressionsInfo[i - 1].constPolsIds.size());
            cmPolsIdsExpressionsOffset.push_back(cmPolsIdsExpressionsOffset[i - 1] + expressionsInfo[i - 1].cmPolsIds.size());
            challengesIdsExpressionsOffset.push_back(challengesIdsExpressionsOffset[i - 1] + expressionsInfo[i - 1].challengeIds.size());
            publicsIdsExpressionsOffset.push_back(publicsIdsExpressionsOffset[i - 1] + expressionsInfo[i - 1].publicsIds.size());
            airgroupValuesIdsExpressionsOffset.push_back(airgroupValuesIdsExpressionsOffset[i - 1] + expressionsInfo[i - 1].airgroupValuesIds.size());
            airValuesIdsExpressionsOffset.push_back(airValuesIdsExpressionsOffset[i - 1] + expressionsInfo[i - 1].airValuesIds.size());

            std::vector<uint32_t> customOffset;
            for (uint32_t j = 0; j < nCustomCommits; ++j) {
                customOffset.push_back(customCommitsIdsExpressionsOffset[i - 1][j] + expressionsInfo[i - 1].customValuesIds[j].size());
            }
            customCommitsIdsExpressionsOffset.push_back(customOffset);
        }

        opsExpressions.insert(opsExpressions.end(), expressionsInfo[i].ops.begin(), expressionsInfo[i].ops.end());
        argsExpressions.insert(argsExpressions.end(), expressionsInfo[i].args.begin(), expressionsInfo[i].args.end());
        constPolsIdsExpressions.insert(constPolsIdsExpressions.end(), expressionsInfo[i].constPolsIds.begin(), expressionsInfo[i].constPolsIds.end());
        cmPolsIdsExpressions.insert(cmPolsIdsExpressions.end(), expressionsInfo[i].cmPolsIds.begin(), expressionsInfo[i].cmPolsIds.end());
        challengesIdsExpressions.insert(challengesIdsExpressions.end(), expressionsInfo[i].challengeIds.begin(), expressionsInfo[i].challengeIds.end());
        publicsIdsExpressions.insert(publicsIdsExpressions.end(), expressionsInfo[i].publicsIds.begin(), expressionsInfo[i].publicsIds.end());
        airgroupValuesIdsExpressions.insert(airgroupValuesIdsExpressions.end(), expressionsInfo[i].airgroupValuesIds.begin(), expressionsInfo[i].airgroupValuesIds.end());
        airValuesIdsExpressions.insert(airValuesIdsExpressions.end(), expressionsInfo[i].airValuesIds.begin(), expressionsInfo[i].airValuesIds.end());
    
        for (uint32_t j = 0; j < nCustomCommits; ++j) {
            customCommitsIdsExpressions.insert(customCommitsIdsExpressions.end(), expressionsInfo[i].customValuesIds[j].begin(), expressionsInfo[i].customValuesIds[j].end());
        }
    }

    binFile.writeU32LE(opsExpressions.size());
    binFile.writeU32LE(argsExpressions.size());
    binFile.writeU32LE(numbersExps.size());
    binFile.writeU32LE(constPolsIdsExpressions.size());
    binFile.writeU32LE(cmPolsIdsExpressions.size());
    binFile.writeU32LE(challengesIdsExpressions.size());
    binFile.writeU32LE(publicsIdsExpressions.size());
    binFile.writeU32LE(airgroupValuesIdsExpressions.size());
    binFile.writeU32LE(airValuesIdsExpressions.size());
    binFile.writeU32LE(customCommitsIdsExpressions.size());

    uint64_t nExpressions = expressionsInfo.size();

    binFile.writeU32LE(nCustomCommits);

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

        // Write constPolsIds information
        binFile.writeU32LE(expInfo.constPolsIds.size());
        binFile.writeU32LE(constPolsIdsExpressionsOffset[i]);

        // Write cmPolsIds information
        binFile.writeU32LE(expInfo.cmPolsIds.size());
        binFile.writeU32LE(cmPolsIdsExpressionsOffset[i]);

        // Write challengesIds information
        binFile.writeU32LE(expInfo.challengeIds.size());
        binFile.writeU32LE(challengesIdsExpressionsOffset[i]);

        // Write publicsIds information
        binFile.writeU32LE(expInfo.publicsIds.size());
        binFile.writeU32LE(publicsIdsExpressionsOffset[i]);

        // Write airgroupValuesIds information
        binFile.writeU32LE(expInfo.airgroupValuesIds.size());
        binFile.writeU32LE(airgroupValuesIdsExpressionsOffset[i]);

        // Write airValuesIds information
        binFile.writeU32LE(expInfo.airValuesIds.size());
        binFile.writeU32LE(airValuesIdsExpressionsOffset[i]);

        // Write customCommitsIds information
        for (uint64_t j = 0; j < nCustomCommits; ++j) {
            binFile.writeU32LE(expInfo.customValuesIds[j].size());
            binFile.writeU32LE(customCommitsIdsExpressionsOffset[i][j]);
        }

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

    for(uint64_t j = 0; j < constPolsIdsExpressions.size(); ++j) {
        binFile.writeU16LE(constPolsIdsExpressions[j]);
    }

    for(uint64_t j = 0; j < cmPolsIdsExpressions.size(); ++j) {
        binFile.writeU16LE(cmPolsIdsExpressions[j]);
    }
    for(uint64_t j = 0; j < challengesIdsExpressions.size(); ++j) {
        binFile.writeU16LE(challengesIdsExpressions[j]);
    }

    for(uint64_t j = 0; j < publicsIdsExpressions.size(); ++j) {
        binFile.writeU16LE(publicsIdsExpressions[j]);
    }

    for(uint64_t j = 0; j < airgroupValuesIdsExpressions.size(); ++j) {
        binFile.writeU16LE(airgroupValuesIdsExpressions[j]);
    }

    for(uint64_t j = 0; j < airValuesIdsExpressions.size(); ++j) {
        binFile.writeU16LE(airValuesIdsExpressions[j]);
    }

    for(uint64_t j = 0; j < customCommitsIdsExpressions.size(); ++j) {
        binFile.writeU16LE(customCommitsIdsExpressions[j]);
    }

    binFile.endWriteSection();
}

void ExpressionsBin::writeConstraintsSection(BinFileUtils::BinFileWriter &binFile, int section, std::vector<ExpInfoBin> constraintsInfo, std::vector<uint64_t> numbersConstraints) {
    
    uint64_t nCustomCommits = constraintsInfo[0].customValuesIds.size();

    binFile.startWriteSection(section);

    std::vector<uint8_t> opsDebug;
    std::vector<uint16_t> argsDebug;
    std::vector<uint16_t> constPolsIdsDebug, cmPolsIdsDebug, customCommitsIdsDebug, challengesIdsDebug, publicsIdsDebug, airgroupValuesIdsDebug, airValuesIdsDebug;

    std::vector<uint32_t> opsDebugOffset, argsDebugOffset;
    std::vector<uint32_t> constPolsIdsDebugOffset, cmPolsIdsDebugOffset, challengesIdsDebugOffset, publicsIdsDebugOffset;
    std::vector<uint32_t> airgroupValuesIdsDebugOffset, airValuesIdsDebugOffset;
    std::vector<std::vector<uint32_t>> customCommitsIdsDebugOffset;

    for (uint64_t i = 0; i < constraintsInfo.size(); ++i) {
        if (i == 0) {
            opsDebugOffset.push_back(0);
            argsDebugOffset.push_back(0);
            constPolsIdsDebugOffset.push_back(0);
            cmPolsIdsDebugOffset.push_back(0);
            challengesIdsDebugOffset.push_back(0);
            publicsIdsDebugOffset.push_back(0);
            airgroupValuesIdsDebugOffset.push_back(0);
            airValuesIdsDebugOffset.push_back(0);

            customCommitsIdsDebugOffset.emplace_back(nCustomCommits, 0);
        } else {
            opsDebugOffset.push_back(opsDebugOffset[i - 1] + constraintsInfo[i - 1].ops.size());
            argsDebugOffset.push_back(argsDebugOffset[i - 1] + constraintsInfo[i - 1].args.size());
            constPolsIdsDebugOffset.push_back(constPolsIdsDebugOffset[i - 1] + constraintsInfo[i - 1].constPolsIds.size());
            cmPolsIdsDebugOffset.push_back(cmPolsIdsDebugOffset[i - 1] + constraintsInfo[i - 1].cmPolsIds.size());
            challengesIdsDebugOffset.push_back(challengesIdsDebugOffset[i - 1] + constraintsInfo[i - 1].challengeIds.size());
            publicsIdsDebugOffset.push_back(publicsIdsDebugOffset[i - 1] + constraintsInfo[i - 1].publicsIds.size());
            airgroupValuesIdsDebugOffset.push_back(airgroupValuesIdsDebugOffset[i - 1] + constraintsInfo[i - 1].airgroupValuesIds.size());
            airValuesIdsDebugOffset.push_back(airValuesIdsDebugOffset[i - 1] + constraintsInfo[i - 1].airValuesIds.size());

            std::vector<uint32_t> customOffset;
            for (uint32_t j = 0; j < nCustomCommits; ++j) {
                customOffset.push_back(customCommitsIdsDebugOffset[i - 1][j] + constraintsInfo[i - 1].customValuesIds[j].size());
            }
            customCommitsIdsDebugOffset.push_back(customOffset);
        }

        opsDebug.insert(opsDebug.end(), constraintsInfo[i].ops.begin(), constraintsInfo[i].ops.end());
        argsDebug.insert(argsDebug.end(), constraintsInfo[i].args.begin(), constraintsInfo[i].args.end());
        constPolsIdsDebug.insert(constPolsIdsDebug.end(), constraintsInfo[i].constPolsIds.begin(), constraintsInfo[i].constPolsIds.end());
        cmPolsIdsDebug.insert(cmPolsIdsDebug.end(), constraintsInfo[i].cmPolsIds.begin(), constraintsInfo[i].cmPolsIds.end());
        challengesIdsDebug.insert(challengesIdsDebug.end(), constraintsInfo[i].challengeIds.begin(), constraintsInfo[i].challengeIds.end());
        publicsIdsDebug.insert(publicsIdsDebug.end(), constraintsInfo[i].publicsIds.begin(), constraintsInfo[i].publicsIds.end());
        airgroupValuesIdsDebug.insert(airgroupValuesIdsDebug.end(), constraintsInfo[i].airgroupValuesIds.begin(), constraintsInfo[i].airgroupValuesIds.end());
        airValuesIdsDebug.insert(airValuesIdsDebug.end(), constraintsInfo[i].airValuesIds.begin(), constraintsInfo[i].airValuesIds.end());
    
        for (uint32_t j = 0; j < nCustomCommits; ++j) {
            customCommitsIdsDebug.insert(customCommitsIdsDebug.end(), constraintsInfo[i].customValuesIds[j].begin(), constraintsInfo[i].customValuesIds[j].end());
        }
    }

    binFile.writeU32LE(opsDebug.size());
    binFile.writeU32LE(argsDebug.size());
    binFile.writeU32LE(numbersConstraints.size());
    binFile.writeU32LE(constPolsIdsDebug.size());
    binFile.writeU32LE(cmPolsIdsDebug.size());
    binFile.writeU32LE(challengesIdsDebug.size());
    binFile.writeU32LE(publicsIdsDebug.size());
    binFile.writeU32LE(airgroupValuesIdsDebug.size());
    binFile.writeU32LE(airValuesIdsDebug.size());
    binFile.writeU32LE(customCommitsIdsDebug.size());

    uint64_t nConstraints = constraintsInfo.size();

    binFile.writeU32LE(nCustomCommits);

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

        // Write constPolsIds information
        binFile.writeU32LE(constraintInfo.constPolsIds.size());
        binFile.writeU32LE(constPolsIdsDebugOffset[i]);

        // Write cmPolsIds information
        binFile.writeU32LE(constraintInfo.cmPolsIds.size());
        binFile.writeU32LE(cmPolsIdsDebugOffset[i]);

        // Write challengesIds information
        binFile.writeU32LE(constraintInfo.challengeIds.size());
        binFile.writeU32LE(challengesIdsDebugOffset[i]);

        // Write publicsIds information
        binFile.writeU32LE(constraintInfo.publicsIds.size());
        binFile.writeU32LE(publicsIdsDebugOffset[i]);

        // Write airgroupValuesIds information
        binFile.writeU32LE(constraintInfo.airgroupValuesIds.size());
        binFile.writeU32LE(airgroupValuesIdsDebugOffset[i]);

        // Write airValuesIds information
        binFile.writeU32LE(constraintInfo.airValuesIds.size());
        binFile.writeU32LE(airValuesIdsDebugOffset[i]);

        // Write customCommitsIds information
        for (uint64_t j = 0; j < nCustomCommits; ++j) {
            binFile.writeU32LE(constraintInfo.customValuesIds[j].size());
            binFile.writeU32LE(customCommitsIdsDebugOffset[i][j]);
        }

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

    for(uint64_t j = 0; j < constPolsIdsDebug.size(); ++j) {
        binFile.writeU16LE(constPolsIdsDebug[j]);
    }

    for(uint64_t j = 0; j < cmPolsIdsDebug.size(); ++j) {
        binFile.writeU16LE(cmPolsIdsDebug[j]);
    }

    for(uint64_t j = 0; j < challengesIdsDebug.size(); ++j) {
        binFile.writeU16LE(challengesIdsDebug[j]);
    }

    for(uint64_t j = 0; j < publicsIdsDebug.size(); ++j) {
        binFile.writeU16LE(publicsIdsDebug[j]);
    }

    for(uint64_t j = 0; j < airgroupValuesIdsDebug.size(); ++j) {
        binFile.writeU16LE(airgroupValuesIdsDebug[j]);
    }

    for(uint64_t j = 0; j < airValuesIdsDebug.size(); ++j) {
        binFile.writeU16LE(airValuesIdsDebug[j]);
    }

    for(uint64_t j = 0; j < customCommitsIdsDebug.size(); ++j) {
        binFile.writeU16LE(customCommitsIdsDebug[j]);
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
        cout << j << " " << argsDebug[j] << endl;
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

    uint32_t nOpsExpressions = expressionsBin->readU32LE();
    uint32_t nArgsExpressions = expressionsBin->readU32LE();
    uint32_t nNumbersExpressions = expressionsBin->readU32LE();
    uint32_t nConstPolsIdsExpressions = expressionsBin->readU32LE();
    uint32_t nCmPolsIdsExpressions = expressionsBin->readU32LE();
    uint32_t nChallengesIdsExpressions = expressionsBin->readU32LE();
    uint32_t nPublicsIdsExpressions = expressionsBin->readU32LE();
    uint32_t nAirgroupValuesIdsExpressions = expressionsBin->readU32LE();
    uint32_t nAirValuesIdsExpressions = expressionsBin->readU32LE();
    uint64_t nCustomCommitsPolsIdsExpressions = expressionsBin->readU32LE();

    expressionsBinArgsExpressions.ops = new uint8_t[nOpsExpressions];
    expressionsBinArgsExpressions.args = new uint16_t[nArgsExpressions];
    expressionsBinArgsExpressions.numbers = new uint64_t[nNumbersExpressions];
    expressionsBinArgsExpressions.constPolsIds = new uint16_t[nConstPolsIdsExpressions];
    expressionsBinArgsExpressions.cmPolsIds = new uint16_t[nCmPolsIdsExpressions];
    expressionsBinArgsExpressions.challengesIds = new uint16_t[nChallengesIdsExpressions];
    expressionsBinArgsExpressions.publicsIds = new uint16_t[nPublicsIdsExpressions];
    expressionsBinArgsExpressions.airgroupValuesIds = new uint16_t[nAirgroupValuesIdsExpressions];
    expressionsBinArgsExpressions.airValuesIds = new uint16_t[nAirValuesIdsExpressions];
    expressionsBinArgsExpressions.customCommitsPolsIds = new uint16_t[nCustomCommitsPolsIdsExpressions];
    expressionsBinArgsExpressions.nNumbers = nNumbersExpressions;

    uint64_t nCustomCommits = expressionsBin->readU32LE();
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

        parserParamsExpression.nConstPolsUsed = expressionsBin->readU32LE();
        parserParamsExpression.constPolsOffset = expressionsBin->readU32LE();

        parserParamsExpression.nCmPolsUsed = expressionsBin->readU32LE();
        parserParamsExpression.cmPolsOffset = expressionsBin->readU32LE();

        parserParamsExpression.nChallengesUsed = expressionsBin->readU32LE();
        parserParamsExpression.challengesOffset = expressionsBin->readU32LE();

        parserParamsExpression.nPublicsUsed = expressionsBin->readU32LE();
        parserParamsExpression.publicsOffset = expressionsBin->readU32LE();

        parserParamsExpression.nAirgroupValuesUsed = expressionsBin->readU32LE();
        parserParamsExpression.airgroupValuesOffset = expressionsBin->readU32LE();

        parserParamsExpression.nAirValuesUsed = expressionsBin->readU32LE();
        parserParamsExpression.airValuesOffset = expressionsBin->readU32LE();
        
        std::vector<uint32_t> nCustomCommitsPolsUsed(nCustomCommits);
        std::vector<uint32_t> customCommitsOffset(nCustomCommits);
        for(uint64_t j = 0; j < nCustomCommits; ++j) {
            nCustomCommitsPolsUsed[j] = expressionsBin->readU32LE();
            customCommitsOffset[j] = expressionsBin->readU32LE();
        }
        parserParamsExpression.nCustomCommitsPolsUsed = nCustomCommitsPolsUsed;
        parserParamsExpression.customCommitsOffset = customCommitsOffset;

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
        expressionsBinArgsExpressions.numbers[j] = expressionsBin->readU64LE();
    }

    for(uint64_t j = 0; j < nConstPolsIdsExpressions; ++j) {
        expressionsBinArgsExpressions.constPolsIds[j] = expressionsBin->readU16LE();
    }

    for(uint64_t j = 0; j < nCmPolsIdsExpressions; ++j) {
        expressionsBinArgsExpressions.cmPolsIds[j] = expressionsBin->readU16LE();
    }

    for(uint64_t j = 0; j < nChallengesIdsExpressions; ++j) {
        expressionsBinArgsExpressions.challengesIds[j] = expressionsBin->readU16LE();
    }

    for(uint64_t j = 0; j < nPublicsIdsExpressions; ++j) {
        expressionsBinArgsExpressions.publicsIds[j] = expressionsBin->readU16LE();
    }

    for(uint64_t j = 0; j < nAirgroupValuesIdsExpressions; ++j) {
        expressionsBinArgsExpressions.airgroupValuesIds[j] = expressionsBin->readU16LE();
    }

    for(uint64_t j = 0; j < nAirValuesIdsExpressions; ++j) {
        expressionsBinArgsExpressions.airValuesIds[j] = expressionsBin->readU16LE();
    }

    for(uint64_t j = 0; j < nCustomCommitsPolsIdsExpressions; ++j) {
        expressionsBinArgsExpressions.customCommitsPolsIds[j] = expressionsBin->readU16LE();
    }

    expressionsBin->endReadSection();

    expressionsBin->startReadSection(CONSTRAINTS_SECTION);

    uint32_t nOpsDebug = expressionsBin->readU32LE();
    uint32_t nArgsDebug = expressionsBin->readU32LE();
    uint32_t nNumbersDebug = expressionsBin->readU32LE();
    uint32_t nConstPolsIdsDebug = expressionsBin->readU32LE();
    uint32_t nCmPolsIdsDebug = expressionsBin->readU32LE();
    uint32_t nChallengesIdsDebug = expressionsBin->readU32LE();
    uint32_t nPublicsIdsDebug = expressionsBin->readU32LE();
    uint32_t nAirgroupValuesIdsDebug = expressionsBin->readU32LE();
    uint32_t nAirValuesIdsDebug = expressionsBin->readU32LE();
    uint64_t nCustomCommitsPolsIdsDebug = expressionsBin->readU32LE();


    expressionsBinArgsConstraints.ops = new uint8_t[nOpsDebug];
    expressionsBinArgsConstraints.args = new uint16_t[nArgsDebug];
    expressionsBinArgsConstraints.numbers = new uint64_t[nNumbersDebug];
    expressionsBinArgsConstraints.constPolsIds = new uint16_t[nConstPolsIdsDebug];
    expressionsBinArgsConstraints.cmPolsIds = new uint16_t[nCmPolsIdsDebug];
    expressionsBinArgsConstraints.challengesIds = new uint16_t[nChallengesIdsDebug];
    expressionsBinArgsConstraints.publicsIds = new uint16_t[nPublicsIdsDebug];
    expressionsBinArgsConstraints.airgroupValuesIds = new uint16_t[nAirgroupValuesIdsDebug];
    expressionsBinArgsConstraints.airValuesIds = new uint16_t[nAirValuesIdsDebug];
    expressionsBinArgsConstraints.customCommitsPolsIds = new uint16_t[nCustomCommitsPolsIdsDebug];
    expressionsBinArgsConstraints.nNumbers = nNumbersDebug;
    
    uint64_t nCustomCommitsC = expressionsBin->readU32LE();

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

        parserParamsConstraint.nConstPolsUsed = expressionsBin->readU32LE();
        parserParamsConstraint.constPolsOffset = expressionsBin->readU32LE();

        parserParamsConstraint.nCmPolsUsed = expressionsBin->readU32LE();
        parserParamsConstraint.cmPolsOffset = expressionsBin->readU32LE();

        parserParamsConstraint.nChallengesUsed = expressionsBin->readU32LE();
        parserParamsConstraint.challengesOffset = expressionsBin->readU32LE();

        parserParamsConstraint.nPublicsUsed = expressionsBin->readU32LE();
        parserParamsConstraint.publicsOffset = expressionsBin->readU32LE();

        parserParamsConstraint.nAirgroupValuesUsed = expressionsBin->readU32LE();
        parserParamsConstraint.airgroupValuesOffset = expressionsBin->readU32LE();

        parserParamsConstraint.nAirValuesUsed = expressionsBin->readU32LE();
        parserParamsConstraint.airValuesOffset = expressionsBin->readU32LE();

        std::vector<uint32_t> nCustomCommitsPolsUsedC(nCustomCommitsC);
        std::vector<uint32_t> customCommitsOffsetC(nCustomCommitsC);
        for(uint64_t j = 0; j < nCustomCommitsC; ++j) {
            nCustomCommitsPolsUsedC[j] = expressionsBin->readU32LE();
            customCommitsOffsetC[j] = expressionsBin->readU32LE();
        }
        parserParamsConstraint.nCustomCommitsPolsUsed = nCustomCommitsPolsUsedC;
        parserParamsConstraint.customCommitsOffset = customCommitsOffsetC;

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
        expressionsBinArgsConstraints.numbers[j] = expressionsBin->readU64LE();
    }

    for(uint64_t j = 0; j < nConstPolsIdsDebug; ++j) {
        expressionsBinArgsConstraints.constPolsIds[j] = expressionsBin->readU16LE();
    }

    for(uint64_t j = 0; j < nCmPolsIdsDebug; ++j) {
        expressionsBinArgsConstraints.cmPolsIds[j] = expressionsBin->readU16LE();
    }

    for(uint64_t j = 0; j < nChallengesIdsDebug; ++j) {
        expressionsBinArgsConstraints.challengesIds[j] = expressionsBin->readU16LE();
    }

    for(uint64_t j = 0; j < nPublicsIdsDebug; ++j) {
        expressionsBinArgsConstraints.publicsIds[j] = expressionsBin->readU16LE();
    }

    for(uint64_t j = 0; j < nAirgroupValuesIdsDebug; ++j) {
        expressionsBinArgsConstraints.airgroupValuesIds[j] = expressionsBin->readU16LE();
    }

    for(uint64_t j = 0; j < nAirValuesIdsDebug; ++j) {
        expressionsBinArgsConstraints.airValuesIds[j] = expressionsBin->readU16LE();
    }

    for(uint64_t j = 0; j < nCustomCommitsPolsIdsDebug; ++j) {
        expressionsBinArgsConstraints.customCommitsPolsIds[j] = expressionsBin->readU16LE();
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


void ExpressionsBin::loadGlobalBin(BinFileUtils::BinFile *globalBin) {
    
    globalBin->startReadSection(GLOBAL_CONSTRAINTS_SECTION);

    uint32_t nOpsDebug = globalBin->readU32LE();
    uint32_t nArgsDebug = globalBin->readU32LE();
    uint32_t nNumbersDebug = globalBin->readU32LE();

    expressionsBinArgsConstraints.ops = new uint8_t[nOpsDebug];
    expressionsBinArgsConstraints.args = new uint16_t[nArgsDebug];
    expressionsBinArgsConstraints.numbers = new uint64_t[nNumbersDebug];
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
        expressionsBinArgsConstraints.numbers[j] = globalBin->readU64LE();
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
                } else if(hintFieldValue.operand == opType::tmp || hintFieldValue.operand == opType::public_) {
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


VecU64Result ExpressionsBin::getHintIdsByName(std::string name) {
    VecU64Result hintIds;

    hintIds.nElements = 0;
    for (uint64_t i = 0; i < hints.size(); ++i) {
        if (hints[i].name == name) {
            hintIds.nElements++;
        }
    }

    uint64_t c = 0;
    hintIds.ids = new uint64_t[hintIds.nElements];
    for (uint64_t i = 0; i < hints.size(); ++i) {
        if (hints[i].name == name) {
            hintIds.ids[c++] = i;
        }
    }

    return hintIds;
}