#ifndef EXPRESSIONS_PACK_HPP
#define EXPRESSIONS_PACK_HPP
#include "expressions_ctx.hpp"

class ExpressionsPack : public ExpressionsCtx {
public:
    uint64_t nrowsPack;
    uint64_t nCols;
    vector<uint64_t> nColsStages;
    vector<uint64_t> nColsStagesAcc;
    vector<uint64_t> offsetsStages;
    ExpressionsPack(SetupCtx& setupCtx, uint64_t nrowsPack_ = 4) : ExpressionsCtx(setupCtx), nrowsPack(nrowsPack_) {};

    void setBufferTInfo(bool domainExtended, int64_t expId) {
        uint64_t nOpenings = setupCtx.starkInfo.verify ? 1 : setupCtx.starkInfo.openingPoints.size();
        uint64_t ns = 2 + setupCtx.starkInfo.nStages + setupCtx.starkInfo.customCommits.size();
        offsetsStages = vector<uint64_t>(ns * nOpenings + 1, 0);
        nColsStages = vector<uint64_t>(ns * nOpenings + 1, 0);
        nColsStagesAcc = vector<uint64_t>(ns * nOpenings + 1, 0);

        nCols = setupCtx.starkInfo.nConstants;

        for(uint64_t o = 0; o < nOpenings; ++o) {
            for(uint64_t stage = 0; stage < ns; ++stage) {
                if(stage == 0) {
                    offsetsStages[ns*o] = 0;
                    nColsStages[ns*o] = setupCtx.starkInfo.mapSectionsN["const"];
                    nColsStagesAcc[ns*o] = o == 0 ? 0 : nColsStagesAcc[ns*o + stage - 1] + nColsStages[stage - 1];
                } else if(stage < 2 + setupCtx.starkInfo.nStages) {
                    std::string section = "cm" + to_string(stage);
                    offsetsStages[ns*o + stage] = setupCtx.starkInfo.mapOffsets[std::make_pair(section, domainExtended)];
                    nColsStages[ns*o + stage] = setupCtx.starkInfo.mapSectionsN[section];
                    nColsStagesAcc[ns*o + stage] = nColsStagesAcc[ns*o + stage - 1] + nColsStages[stage - 1];
                } else {
                    uint64_t index = stage - setupCtx.starkInfo.nStages - 2;
                    std::string section = setupCtx.starkInfo.customCommits[index].name + "0";
                    offsetsStages[ns*o + stage] = setupCtx.starkInfo.mapOffsets[std::make_pair(section, domainExtended)];
                    nColsStages[ns*o + stage] = setupCtx.starkInfo.mapSectionsN[section];
                    nColsStagesAcc[ns*o + stage] = nColsStagesAcc[ns*o + stage - 1] + nColsStages[stage - 1];
                }
            }
        }

        nColsStagesAcc[ns*nOpenings] = nColsStagesAcc[ns*nOpenings - 1] + nColsStages[ns*nOpenings - 1];
        if(expId == int64_t(setupCtx.starkInfo.cExpId)) {
            nCols = nColsStagesAcc[ns*nOpenings] + setupCtx.starkInfo.boundaries.size() + 1;
        } else if(expId == int64_t(setupCtx.starkInfo.friExpId)) {
            nCols = nColsStagesAcc[ns*nOpenings] + nOpenings*FIELD_EXTENSION;
        } else {
            nCols = nColsStagesAcc[ns*nOpenings] + 1;
        }
    }

    inline void loadPolynomials(StepsParams& params, ParserArgs &parserArgs, std::vector<Dest> &dests, Goldilocks::Element *bufferT_, uint64_t row, uint64_t domainSize, bool domainExtended) {
        uint64_t nOpenings = setupCtx.starkInfo.verify ? 1 : setupCtx.starkInfo.openingPoints.size();
        uint64_t ns = 2 + setupCtx.starkInfo.nStages + setupCtx.starkInfo.customCommits.size();

        uint64_t extendBits = (setupCtx.starkInfo.starkStruct.nBitsExt - setupCtx.starkInfo.starkStruct.nBits);
        int64_t extend = domainExtended ? (1 << extendBits) : 1;
        uint64_t nextStrides[nOpenings];
        for(uint64_t i = 0; i < nOpenings; ++i) {
            uint64_t opening = setupCtx.starkInfo.verify ? 0 : setupCtx.starkInfo.openingPoints[i] < 0 ? setupCtx.starkInfo.openingPoints[i] + domainSize : setupCtx.starkInfo.openingPoints[i];
            nextStrides[i] = opening * extend;
        }

        Goldilocks::Element *constPols = domainExtended ? &params.pConstPolsExtendedTreeAddress[2] : params.pConstPolsAddress;

        std::vector<bool> constPolsUsed(setupCtx.starkInfo.constPolsMap.size(), false);
        std::vector<bool> cmPolsUsed(setupCtx.starkInfo.cmPolsMap.size(), false);
        std::vector<std::vector<bool>> customCommitsUsed(setupCtx.starkInfo.customCommits.size());
        for(uint64_t i = 0; i < setupCtx.starkInfo.customCommits.size(); ++i) {
            customCommitsUsed[i] = std::vector<bool>(setupCtx.starkInfo.customCommits[i].stageWidths[0], false);
        }

        for(uint64_t i = 0; i < dests.size(); ++i) {
            for(uint64_t j = 0; j < dests[i].params.size(); ++j) {
                if(dests[i].params[j].op == opType::cm) {
                    cmPolsUsed[dests[i].params[j].polsMapId] = true;
                } else if (dests[i].params[j].op == opType::const_) {
                    constPolsUsed[dests[i].params[j].polsMapId] = true;
                } else if(dests[i].params[j].op == opType::tmp) {
                    uint16_t* cmUsed = &parserArgs.cmPolsIds[dests[i].params[j].parserParams.cmPolsOffset];
                    uint16_t* constUsed = &parserArgs.constPolsIds[dests[i].params[j].parserParams.constPolsOffset];

                    for(uint64_t k = 0; k < dests[i].params[j].parserParams.nConstPolsUsed; ++k) {
                        constPolsUsed[constUsed[k]] = true;
                    }

                    for(uint64_t k = 0; k < dests[i].params[j].parserParams.nCmPolsUsed; ++k) {
                        cmPolsUsed[cmUsed[k]] = true;
                    }

                    for(uint64_t k = 0; k < setupCtx.starkInfo.customCommits.size(); ++k) {
                        uint16_t* customCmUsed = &parserArgs.customCommitsPolsIds[dests[i].params[j].parserParams.customCommitsOffset[k]];
                        for(uint64_t l = 0; l < dests[i].params[j].parserParams.nCustomCommitsPolsUsed[k]; ++l) {
                            customCommitsUsed[k][customCmUsed[l]] = true;
                        }
                    }
                }
            }
        }
        for(uint64_t k = 0; k < constPolsUsed.size(); ++k) {
            if(!constPolsUsed[k]) continue;
            for(uint64_t o = 0; o < nOpenings; ++o) {
                for(uint64_t j = 0; j < nrowsPack; ++j) {
                    uint64_t l = (row + j + nextStrides[o]) % domainSize;
                    bufferT_[(nColsStagesAcc[ns*o] + k)*nrowsPack + j] = constPols[l * nColsStages[0] + k];
                }
            }
        }

        for(uint64_t k = 0; k < cmPolsUsed.size(); ++k) {
            if(!cmPolsUsed[k]) continue;
            PolMap polInfo = setupCtx.starkInfo.cmPolsMap[k];
            uint64_t stage = polInfo.stage;
            uint64_t stagePos = polInfo.stagePos;
            for(uint64_t d = 0; d < polInfo.dim; ++d) {
                for(uint64_t o = 0; o < nOpenings; ++o) {
                    for(uint64_t j = 0; j < nrowsPack; ++j) {
                        uint64_t l = (row + j + nextStrides[o]) % domainSize;
                        if(stage == 1 && !domainExtended) {
                            bufferT_[(nColsStagesAcc[ns*o + stage] + (stagePos + d))*nrowsPack + j] = params.trace[l * nColsStages[stage] + stagePos + d];
                        } else {
                            bufferT_[(nColsStagesAcc[ns*o + stage] + (stagePos + d))*nrowsPack + j] = params.aux_trace[offsetsStages[stage] + l * nColsStages[stage] + stagePos + d];
                        }
                    }
                }
            }
        }

        for(uint64_t i = 0; i < setupCtx.starkInfo.customCommits.size(); ++i) {
            for(uint64_t j = 0; j < setupCtx.starkInfo.customCommits[i].stageWidths[0]; ++j) {
                if(!customCommitsUsed[i][j]) continue;
                PolMap polInfo = setupCtx.starkInfo.customCommitsMap[i][j];
                uint64_t stage = setupCtx.starkInfo.nStages + 2 + i;
                uint64_t stagePos = polInfo.stagePos;
                for(uint64_t d = 0; d < polInfo.dim; ++d) {
                    for(uint64_t o = 0; o < nOpenings; ++o) {
                        for(uint64_t j = 0; j < nrowsPack; ++j) {
                            uint64_t l = (row + j + nextStrides[o]) % domainSize;
                            bufferT_[(nColsStagesAcc[ns*o + stage] + (stagePos + d))*nrowsPack + j] = params.pCustomCommitsFixed[offsetsStages[stage] + l * nColsStages[stage] + stagePos + d];
                        }
                    }
                }
            }
        }

        if(dests[0].params[0].parserParams.expId == int64_t(setupCtx.starkInfo.cExpId)) {
            for(uint64_t d = 0; d < setupCtx.starkInfo.boundaries.size(); ++d) {
                for(uint64_t j = 0; j < nrowsPack; ++j) {
                    if(setupCtx.starkInfo.verify) {
                        for(uint64_t e = 0; e < FIELD_EXTENSION; ++e) {
                            bufferT_[((nColsStagesAcc[ns*nOpenings] + d + FIELD_EXTENSION)*nrowsPack + j) + e] = setupCtx.proverHelpers.zi[d*FIELD_EXTENSION + e];
                        }
                    } else {
                        bufferT_[(nColsStagesAcc[ns*nOpenings] + d + 1)*nrowsPack + j] = setupCtx.proverHelpers.zi[row + j + d*domainSize];
                    }
                }
            }
            for(uint64_t j = 0; j < nrowsPack; ++j) {
                if(setupCtx.starkInfo.verify) {
                    for(uint64_t e = 0; e < FIELD_EXTENSION; ++e) {
                        bufferT_[((nColsStagesAcc[ns*nOpenings])*nrowsPack + j) + e] = setupCtx.proverHelpers.x_n[e];
                    }
                } else {
                    bufferT_[(nColsStagesAcc[ns*nOpenings])*nrowsPack + j] = setupCtx.proverHelpers.x_2ns[row + j];
                }
            }
        } else if(dests[0].params[0].parserParams.expId == int64_t(setupCtx.starkInfo.friExpId)) {
            for(uint64_t d = 0; d < setupCtx.starkInfo.openingPoints.size(); ++d) {
               for(uint64_t k = 0; k < FIELD_EXTENSION; ++k) {
                    for(uint64_t j = 0; j < nrowsPack; ++j) {
                        bufferT_[(nColsStagesAcc[ns*nOpenings] + d*FIELD_EXTENSION + k)*nrowsPack + j] = params.xDivXSub[((row + j)*setupCtx.starkInfo.openingPoints.size() + d)*FIELD_EXTENSION + k];
                    }
                }
            }
        } else {
            for(uint64_t j = 0; j < nrowsPack; ++j) {
                bufferT_[(nColsStagesAcc[ns*nOpenings])*nrowsPack + j] = setupCtx.proverHelpers.x_n[row + j];
            }
        }
    }

    inline void copyPolynomial(Goldilocks::Element* destVals, bool inverse, bool batch, uint64_t dim, Goldilocks::Element* tmp) {
        if(dim == 1) {
            if(inverse) {
                if(batch) {
                    Goldilocks::batchInverse(&destVals[0], &tmp[0], nrowsPack);
                } else {
                    for(uint64_t i = 0; i < nrowsPack; ++i) {
                        Goldilocks::inv(destVals[i], tmp[i]);
                    }
                }
            } else {
                Goldilocks::copy_pack(nrowsPack, &destVals[0], &tmp[0]);
            }
        } else if(dim == FIELD_EXTENSION) {
            if(inverse) {
                Goldilocks::Element buff[FIELD_EXTENSION*nrowsPack];
                Goldilocks::copy_pack(nrowsPack, &buff[0], uint64_t(FIELD_EXTENSION), &tmp[0]);
                Goldilocks::copy_pack(nrowsPack, &buff[1], uint64_t(FIELD_EXTENSION), &tmp[nrowsPack]);
                Goldilocks::copy_pack(nrowsPack, &buff[2], uint64_t(FIELD_EXTENSION), &tmp[2*nrowsPack]);
                if(batch) {
                    Goldilocks3::batchInverse((Goldilocks3::Element *)buff, (Goldilocks3::Element *)buff, nrowsPack);
                } else {
                    for(uint64_t i = 0; i < nrowsPack; ++i) {
                        Goldilocks3::inv((Goldilocks3::Element &)buff[i*FIELD_EXTENSION], (Goldilocks3::Element &)buff[i*FIELD_EXTENSION]);
                    }
                }
                Goldilocks::copy_pack(nrowsPack, &destVals[0], &buff[0], uint64_t(FIELD_EXTENSION));
                Goldilocks::copy_pack(nrowsPack, &destVals[nrowsPack], &buff[1], uint64_t(FIELD_EXTENSION));
                Goldilocks::copy_pack(nrowsPack, &destVals[2*nrowsPack], &buff[2], uint64_t(FIELD_EXTENSION));
            } else {
                Goldilocks::copy_pack(nrowsPack, &destVals[0], &tmp[0]);
                Goldilocks::copy_pack(nrowsPack, &destVals[nrowsPack], &tmp[nrowsPack]);
                Goldilocks::copy_pack(nrowsPack, &destVals[2*nrowsPack], &tmp[2*nrowsPack]);
            }
        }
    }

    inline void multiplyPolynomials(Dest &dest, Goldilocks::Element* destVals) {
        if(dest.dim == 1) {
            Goldilocks::op_pack(nrowsPack, 2, &destVals[0], &destVals[0], &destVals[FIELD_EXTENSION*nrowsPack]);
        } else {
            Goldilocks::Element vals[FIELD_EXTENSION*nrowsPack];
            if(dest.params[0].dim == FIELD_EXTENSION && dest.params[1].dim == FIELD_EXTENSION) {
                Goldilocks3::op_pack(nrowsPack, 2, &vals[0], &destVals[0], &destVals[FIELD_EXTENSION*nrowsPack]);
            } else if(dest.params[0].dim == FIELD_EXTENSION && dest.params[1].dim == 1) {
                Goldilocks3::op_31_pack(nrowsPack, 2, &vals[0], &destVals[0], &destVals[FIELD_EXTENSION*nrowsPack]);
            } else {
                Goldilocks3::op_31_pack(nrowsPack, 2, &vals[0], &destVals[FIELD_EXTENSION*nrowsPack], &destVals[0]);
            }
            Goldilocks::copy_pack(nrowsPack, &destVals[0], &vals[0]);
            Goldilocks::copy_pack(nrowsPack, &destVals[nrowsPack], &vals[nrowsPack]);
            Goldilocks::copy_pack(nrowsPack, &destVals[2*nrowsPack], &vals[2*nrowsPack]);
        }
    }

    inline void storePolynomial(std::vector<Dest> dests, Goldilocks::Element** destVals, uint64_t row) {
        for(uint64_t i = 0; i < dests.size(); ++i) {
            if(row >= dests[i].domainSize) continue;
            if(dests[i].dim == 1) {
                uint64_t offset = dests[i].offset != 0 ? dests[i].offset : 1;
                Goldilocks::copy_pack(nrowsPack, &dests[i].dest[row*offset], uint64_t(offset), &destVals[i][0]);
            } else {
                uint64_t offset = dests[i].offset != 0 ? dests[i].offset : FIELD_EXTENSION;
                Goldilocks::copy_pack(nrowsPack, &dests[i].dest[row*offset], uint64_t(offset), &destVals[i][0]);
                Goldilocks::copy_pack(nrowsPack, &dests[i].dest[row*offset + 1], uint64_t(offset), &destVals[i][nrowsPack]);
                Goldilocks::copy_pack(nrowsPack, &dests[i].dest[row*offset + 2], uint64_t(offset), &destVals[i][2*nrowsPack]);
            }
        }
    }

    inline void printTmp1(uint64_t row, Goldilocks::Element* tmp) {
        Goldilocks::Element buff[nrowsPack];
        Goldilocks::copy_pack(nrowsPack, buff, tmp);
        for(uint64_t i = 0; i < nrowsPack; ++i) {
            cout << "Value at row " << row + i << " is " << Goldilocks::toString(buff[i]) << endl;
        }
    }

    inline void printTmp3(uint64_t row, Goldilocks::Element* tmp) {
        for(uint64_t i = 0; i < nrowsPack; ++i) {
            cout << "Value at row " << row + i << " is [" << Goldilocks::toString(tmp[i]) << ", " << Goldilocks::toString(tmp[nrowsPack + i]) << ", " << Goldilocks::toString(tmp[2*nrowsPack + i]) << "]" << endl;
        }
    }

    inline void printCommit(uint64_t row, Goldilocks::Element* bufferT, bool extended) {
        if(extended) {
            Goldilocks::Element buff[FIELD_EXTENSION*nrowsPack];
            Goldilocks::copy_pack(nrowsPack, &buff[0], uint64_t(FIELD_EXTENSION), &bufferT[0]);
            Goldilocks::copy_pack(nrowsPack, &buff[1], uint64_t(FIELD_EXTENSION), &bufferT[setupCtx.starkInfo.openingPoints.size()]);
            Goldilocks::copy_pack(nrowsPack, &buff[2], uint64_t(FIELD_EXTENSION), &bufferT[2*setupCtx.starkInfo.openingPoints.size()]);
            for(uint64_t i = 0; i < nrowsPack; ++i) {
                cout << "Value at row " << row + i << " is [" << Goldilocks::toString(buff[FIELD_EXTENSION*i]) << ", " << Goldilocks::toString(buff[FIELD_EXTENSION*i + 1]) << ", " << Goldilocks::toString(buff[FIELD_EXTENSION*i + 2]) << "]" << endl;
            }
        } else {
            Goldilocks::Element buff[nrowsPack];
            Goldilocks::copy_pack(nrowsPack, &buff[0], &bufferT[0]);
            for(uint64_t i = 0; i < nrowsPack; ++i) {
                cout << "Value at row " << row + i << " is " << Goldilocks::toString(buff[i]) << endl;
            }
        }
    }

    void calculateExpressions(StepsParams& params, ParserArgs &parserArgs, std::vector<Dest> dests, uint64_t domainSize, bool compilation_time) override {
        uint64_t nOpenings = setupCtx.starkInfo.openingPoints.size();
        uint64_t ns = 2 + setupCtx.starkInfo.nStages + setupCtx.starkInfo.customCommits.size();
        bool domainExtended = !setupCtx.starkInfo.verify && domainSize == uint64_t(1 << setupCtx.starkInfo.starkStruct.nBitsExt) ? true : false;

        uint64_t expId = dests[0].params[0].op == opType::tmp ? dests[0].expId : 0;
        setBufferTInfo(domainExtended, expId);

        Goldilocks::Element challenges[setupCtx.starkInfo.challengesMap.size()*FIELD_EXTENSION*nrowsPack];
        if(!compilation_time) {
            for(uint64_t i = 0; i < setupCtx.starkInfo.challengesMap.size(); ++i) {
                for(uint64_t j = 0; j < nrowsPack; ++j) {
                    challenges[(i*FIELD_EXTENSION)*nrowsPack + j] = params.challenges[i * FIELD_EXTENSION];
                    challenges[(i*FIELD_EXTENSION + 1)*nrowsPack + j] = params.challenges[i * FIELD_EXTENSION + 1];
                    challenges[(i*FIELD_EXTENSION + 2)*nrowsPack + j] = params.challenges[i * FIELD_EXTENSION + 2];
                }
            }
        }

        Goldilocks::Element *numbers_ = new Goldilocks::Element[parserArgs.nNumbers*nrowsPack];
        for(uint64_t i = 0; i < parserArgs.nNumbers; ++i) {
            for(uint64_t k = 0; k < nrowsPack; ++k) {
                numbers_[i*nrowsPack + k] = Goldilocks::fromU64(parserArgs.numbers[i]);
            }
        }

        Goldilocks::Element publics[setupCtx.starkInfo.nPublics*nrowsPack];
        if(!compilation_time) {
            for(uint64_t i = 0; i < setupCtx.starkInfo.nPublics; ++i) {
                for(uint64_t j = 0; j < nrowsPack; ++j) {
                    publics[i*nrowsPack + j] = params.publicInputs[i];
                }
            }
        }

        Goldilocks::Element evals[setupCtx.starkInfo.evMap.size()*FIELD_EXTENSION*nrowsPack];
        if(!compilation_time) {
            for(uint64_t i = 0; i < setupCtx.starkInfo.evMap.size(); ++i) {
                for(uint64_t j = 0; j < nrowsPack; ++j) {
                    evals[(i*FIELD_EXTENSION)*nrowsPack + j] = params.evals[i * FIELD_EXTENSION];
                    evals[(i*FIELD_EXTENSION + 1)*nrowsPack + j] = params.evals[i * FIELD_EXTENSION + 1];
                    evals[(i*FIELD_EXTENSION + 2)*nrowsPack + j] = params.evals[i * FIELD_EXTENSION + 2];
                }
            }
        }

        Goldilocks::Element airgroupValues[setupCtx.starkInfo.airgroupValuesMap.size()*FIELD_EXTENSION*nrowsPack];
        if(!compilation_time) {
            uint64_t p = 0;
            for(uint64_t i = 0; i < setupCtx.starkInfo.airgroupValuesMap.size(); ++i) {
                for(uint64_t j = 0; j < nrowsPack; ++j) {
                    if(setupCtx.starkInfo.airgroupValuesMap[i].stage == 1) {
                        airgroupValues[(i*FIELD_EXTENSION)*nrowsPack + j] = params.airgroupValues[p];
                        airgroupValues[(i*FIELD_EXTENSION + 1)*nrowsPack + j] = Goldilocks::zero();
                        airgroupValues[(i*FIELD_EXTENSION + 2)*nrowsPack + j] = Goldilocks::zero();
                    } else {
                        airgroupValues[(i*FIELD_EXTENSION)*nrowsPack + j] = params.airgroupValues[p];
                        airgroupValues[(i*FIELD_EXTENSION + 1)*nrowsPack + j] = params.airgroupValues[p + 1];
                        airgroupValues[(i*FIELD_EXTENSION + 2)*nrowsPack + j] = params.airgroupValues[p + 2];
                    }
                }
                if(setupCtx.starkInfo.airgroupValuesMap[i].stage == 1) {
                    p += 1;
                } else {
                    p += 3;
                }
            }
        }

        Goldilocks::Element proofValues[setupCtx.starkInfo.proofValuesMap.size()*FIELD_EXTENSION*nrowsPack];
        if(!compilation_time) {
            uint64_t p = 0;
            for(uint64_t i = 0; i < setupCtx.starkInfo.proofValuesMap.size(); ++i) {
                for(uint64_t j = 0; j < nrowsPack; ++j) {
                    if(setupCtx.starkInfo.proofValuesMap[i].stage == 1) {
                        proofValues[(i*FIELD_EXTENSION)*nrowsPack + j] = params.proofValues[p];
                        proofValues[(i*FIELD_EXTENSION + 1)*nrowsPack + j] = Goldilocks::zero();
                        proofValues[(i*FIELD_EXTENSION + 2)*nrowsPack + j] = Goldilocks::zero();
                    } else {
                        proofValues[(i*FIELD_EXTENSION)*nrowsPack + j] = params.proofValues[p];
                        proofValues[(i*FIELD_EXTENSION + 1)*nrowsPack + j] = params.proofValues[p + 1];
                        proofValues[(i*FIELD_EXTENSION + 2)*nrowsPack + j] = params.proofValues[p + 2];
                    }
                }
                if(setupCtx.starkInfo.proofValuesMap[i].stage == 1) {
                    p += 1;
                } else {
                    p += 3;
                }
            }
        }

        Goldilocks::Element airValues[setupCtx.starkInfo.airValuesMap.size()*FIELD_EXTENSION*nrowsPack];
        if(!compilation_time) {
            uint64_t p = 0;
            for(uint64_t i = 0; i < setupCtx.starkInfo.airValuesMap.size(); ++i) {
                for(uint64_t j = 0; j < nrowsPack; ++j) {
                    if(setupCtx.starkInfo.airValuesMap[i].stage == 1) {
                        airValues[(i*FIELD_EXTENSION)*nrowsPack + j] = params.airValues[p];
                        airValues[(i*FIELD_EXTENSION + 1)*nrowsPack + j] = Goldilocks::zero();
                        airValues[(i*FIELD_EXTENSION + 2)*nrowsPack + j] = Goldilocks::zero();
                    } else {
                        airValues[(i*FIELD_EXTENSION)*nrowsPack + j] = params.airValues[p];
                        airValues[(i*FIELD_EXTENSION + 1)*nrowsPack + j] = params.airValues[p + 1];
                        airValues[(i*FIELD_EXTENSION + 2)*nrowsPack + j] = params.airValues[p + 2];
                    }
                }
                if(setupCtx.starkInfo.airValuesMap[i].stage == 1) {
                    p += 1;
                } else {
                    p += 3;
                }
            }
        }

        Goldilocks::Element *bufferT__ = new Goldilocks::Element[omp_get_max_threads()*nOpenings*nCols*nrowsPack];
        uint64_t maxTemp1Size = 0;
        uint64_t maxTemp3Size = 0;
        for (uint64_t j = 0; j < dests.size(); ++j) {
            for (uint64_t k = 0; k < dests[j].params.size(); ++k) {
                if (dests[j].params[k].parserParams.nTemp1*nrowsPack > maxTemp1Size) {
                    maxTemp1Size = dests[j].params[k].parserParams.nTemp1*nrowsPack;
                }
                if (dests[j].params[k].parserParams.nTemp3*nrowsPack*FIELD_EXTENSION > maxTemp3Size) {
                    maxTemp3Size = dests[j].params[k].parserParams.nTemp3*nrowsPack*FIELD_EXTENSION;
                }
            }
        }

        Goldilocks::Element *tmp1_ = new Goldilocks::Element[omp_get_max_threads() * maxTemp1Size];
        Goldilocks::Element *tmp3_ = new Goldilocks::Element[omp_get_max_threads() * maxTemp3Size];

    #pragma omp parallel for
        for (uint64_t i = 0; i < domainSize; i+= nrowsPack) {
            Goldilocks::Element* expressions_params[10];
            expressions_params[2] = publics;
            expressions_params[3] = numbers_;
            expressions_params[4] = airValues;
            expressions_params[5] = proofValues;
            expressions_params[7] = airgroupValues;
            expressions_params[8] = challenges;
            expressions_params[9] = evals;

            Goldilocks::Element *bufferT_ = &bufferT__[omp_get_thread_num()*nOpenings*nCols*nrowsPack];

            if(!compilation_time) loadPolynomials(params, parserArgs, dests, bufferT_, i, domainSize, domainExtended);

            Goldilocks::Element **destVals = new Goldilocks::Element*[dests.size()];

            for(uint64_t j = 0; j < dests.size(); ++j) {
                if(i >= dests[j].domainSize) continue;
                destVals[j] = new Goldilocks::Element[dests[j].params.size() * FIELD_EXTENSION* nrowsPack];
                for(uint64_t k = 0; k < dests[j].params.size(); ++k) {
                    uint64_t i_args = 0;

                    if(dests[j].params[k].op == opType::cm || dests[j].params[k].op == opType::const_) {
                        uint64_t openingPointIndex = dests[j].params[k].rowOffsetIndex;
                        uint64_t buffPos = ns*openingPointIndex + dests[j].params[k].stage;
                        uint64_t stagePos = dests[j].params[k].stagePos;
                        copyPolynomial(&destVals[j][k*FIELD_EXTENSION*nrowsPack], dests[j].params[k].inverse,dests[j].params[k].batch,dests[j].params[k].dim, &bufferT_[(nColsStagesAcc[buffPos] + stagePos)*nrowsPack]);
                        continue;
                    } else if(dests[j].params[k].op == opType::number) {
                        for(uint64_t r = 0; r < nrowsPack; ++r) {
                            destVals[j][k*FIELD_EXTENSION*nrowsPack + r] = Goldilocks::fromU64(dests[j].params[k].value);
                        }
                        continue;
                    } else if(dests[j].params[k].op == opType::airvalue) {
                         memcpy(&destVals[j][k*FIELD_EXTENSION*nrowsPack], &airValues[dests[j].params[k].polsMapId*FIELD_EXTENSION*nrowsPack], FIELD_EXTENSION*nrowsPack*sizeof(Goldilocks::Element));
                         continue;
                    }
                    uint8_t* ops = &parserArgs.ops[dests[j].params[k].parserParams.opsOffset];
                    uint16_t* args = &parserArgs.args[dests[j].params[k].parserParams.argsOffset];
                    expressions_params[0] = bufferT_;
                    expressions_params[1] = &tmp1_[omp_get_thread_num()*maxTemp1Size];
                    expressions_params[6] = &tmp3_[omp_get_thread_num()*maxTemp3Size];

                    for (uint64_t kk = 0; kk < dests[j].params[k].parserParams.nOps; ++kk) {
                        switch (ops[kk]) {
                            case 0: {
                                // COPY dim1 to dim1
                                Goldilocks::copy_pack(nrowsPack, &expressions_params[args[i_args]][(nColsStagesAcc[args[i_args + 1]] + args[i_args + 2]) * nrowsPack], &expressions_params[args[i_args + 3]][(nColsStagesAcc[args[i_args + 4]] + args[i_args + 5]) * nrowsPack]);
                                i_args += 7;
                                break;
                            }
                            case 1: {
                                // OPERATION WITH DEST: dim1 - SRC0: dim1 - SRC1: dim1
                                Goldilocks::op_pack(nrowsPack, args[i_args], &expressions_params[args[i_args + 1]][(nColsStagesAcc[args[i_args + 2]] + args[i_args + 3]) * nrowsPack], &expressions_params[args[i_args + 4]][(nColsStagesAcc[args[i_args + 5]] + args[i_args + 6]) * nrowsPack], &expressions_params[args[i_args + 8]][(nColsStagesAcc[args[i_args + 9]] + args[i_args + 10]) * nrowsPack]);
                                i_args += 12;
                                break;
                            }
                            case 2: {
                                // OPERATION WITH DEST: dim3 - SRC0: dim3 - SRC1: dim1
                                Goldilocks3::op_31_pack(nrowsPack, args[i_args], &expressions_params[args[i_args + 1]][(nColsStagesAcc[args[i_args + 2]] + args[i_args + 3]) * nrowsPack], &expressions_params[args[i_args + 4]][(nColsStagesAcc[args[i_args + 5]] + args[i_args + 6]) * nrowsPack], &expressions_params[args[i_args + 8]][(nColsStagesAcc[args[i_args + 9]] + args[i_args + 10]) * nrowsPack]);
                                i_args += 12;
                                break;
                            }
                            case 3: {
                                // OPERATION WITH DEST: dim3 - SRC0: dim3 - SRC1: dim3
                                Goldilocks3::op_pack(nrowsPack, args[i_args], &expressions_params[args[i_args + 1]][(nColsStagesAcc[args[i_args + 2]] + args[i_args + 3]) * nrowsPack], &expressions_params[args[i_args + 4]][(nColsStagesAcc[args[i_args + 5]] + args[i_args + 6]) * nrowsPack], &expressions_params[args[i_args + 8]][(nColsStagesAcc[args[i_args + 9]] + args[i_args + 10]) * nrowsPack]);
                                i_args += 12;
                                break;
                            }
                            case 4: {
                                // COPY dim3 to dim3
                                Goldilocks3::copy_pack(nrowsPack, &expressions_params[args[i_args]][(nColsStagesAcc[args[i_args + 1]] + args[i_args + 2]) * nrowsPack], &expressions_params[args[i_args + 3]][(nColsStagesAcc[args[i_args + 4]] + args[i_args + 5]) * nrowsPack]);
                                i_args += 7;
                                break;
                            }
                            default: {
                                std::cout << " Wrong operation!" << std::endl;
                                exit(1);
                            }
                        }
                    }

                    if (i_args != dests[j].params[k].parserParams.nArgs) std::cout << " " << i_args << " - " << dests[j].params[k].parserParams.nArgs << std::endl;
                    assert(i_args == dests[j].params[k].parserParams.nArgs);

                    if(dests[j].params[k].parserParams.destDim == 1) {
                        copyPolynomial(&destVals[j][k*FIELD_EXTENSION*nrowsPack], dests[j].params[k].inverse,dests[j].params[k].batch, dests[j].params[k].parserParams.destDim, &expressions_params[1][dests[j].params[k].parserParams.destId*nrowsPack]);
                    } else {
                        copyPolynomial(&destVals[j][k*FIELD_EXTENSION*nrowsPack], dests[j].params[k].inverse,dests[j].params[k].batch, dests[j].params[k].parserParams.destDim, &expressions_params[6][dests[j].params[k].parserParams.destId*FIELD_EXTENSION*nrowsPack]);
                    }
                }
                if(dests[j].params.size() == 2) {
                    multiplyPolynomials(dests[j], destVals[j]);
                }
            }
            storePolynomial(dests, destVals, i);

            for(uint64_t j = 0; j < dests.size(); ++j) {
                if(i >= dests[j].domainSize) continue;
                delete[] destVals[j];
            }
            delete[] destVals;
        }
        delete[] numbers_;
        delete[] bufferT__;
        delete[] tmp1_;
        delete[] tmp3_;
    }
};

#endif