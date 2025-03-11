#ifndef EXPRESSIONS_AVX512_HPP
#define EXPRESSIONS_AVX512_HPP
#include "expressions_ctx.hpp"

#ifdef __AVX512__

class ExpressionsAvx512 : public ExpressionsCtx {
public:
    uint64_t nrowsPack = 8;
    uint64_t nCols;
    vector<uint64_t> nColsStages;
    vector<uint64_t> nColsStagesAcc;
    vector<uint64_t> offsetsStages;
    ExpressionsAvx512(SetupCtx& setupCtx) : ExpressionsCtx(setupCtx) {};

    void setBufferTInfo(bool domainExtended, int64_t expId) {
        uint64_t nOpenings = setupCtx.starkInfo.openingPoints.size();
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

    inline void loadPolynomials(StepsParams& params, ParserArgs &parserArgs, std::vector<Dest> &dests, Goldilocks::Element *bufferT, __m512i *bufferT_, uint64_t row, uint64_t domainSize) {
        uint64_t nOpenings = setupCtx.starkInfo.openingPoints.size();
        uint64_t ns = 2 + setupCtx.starkInfo.nStages + setupCtx.starkInfo.customCommits.size();
        bool domainExtended = domainSize == uint64_t(1 << setupCtx.starkInfo.starkStruct.nBitsExt) ? true : false;

        uint64_t extendBits = (setupCtx.starkInfo.starkStruct.nBitsExt - setupCtx.starkInfo.starkStruct.nBits);
        int64_t extend = domainExtended ? (1 << extendBits) : 1;
        uint64_t nextStrides[nOpenings];
        for(uint64_t i = 0; i < nOpenings; ++i) {
            uint64_t opening = setupCtx.starkInfo.openingPoints[i] < 0 ? setupCtx.starkInfo.openingPoints[i] + domainSize : setupCtx.starkInfo.openingPoints[i];
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
                    bufferT[nrowsPack*o + j] = constPols[l * nColsStages[0] + k];
                }
                Goldilocks::load_avx512(bufferT_[nColsStagesAcc[ns*o] + k], &bufferT[nrowsPack*o]);
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
                            bufferT[nrowsPack*o + j] = params.trace[l * nColsStages[stage] + stagePos + d];
                        } else {
                            bufferT[nrowsPack*o + j] = params.aux_trace[offsetsStages[stage] + l * nColsStages[stage] + stagePos + d];
                        }
                    }
                    Goldilocks::load_avx512(bufferT_[nColsStagesAcc[ns*o + stage] + (stagePos + d)], &bufferT[nrowsPack*o]);
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
                            bufferT[nrowsPack*o + j] = params.pCustomCommitsFixed[offsetsStages[stage] + l * nColsStages[stage] + stagePos + d];
                        }
                        Goldilocks::load_avx512(bufferT_[nColsStagesAcc[ns*o + stage] + (stagePos + d)], &bufferT[nrowsPack*o]);
                    }
                }
            }
        }

        if(dests[0].params[0].parserParams.expId == int64_t(setupCtx.starkInfo.cExpId)) {
            for(uint64_t j = 0; j < nrowsPack; ++j) {
                bufferT[j] = setupCtx.proverHelpers.x_2ns[row + j];
            }
            Goldilocks::load_avx512(bufferT_[nColsStagesAcc[ns*nOpenings]], &bufferT[0]);
            for(uint64_t d = 0; d < setupCtx.starkInfo.boundaries.size(); ++d) {
                for(uint64_t j = 0; j < nrowsPack; ++j) {
                    bufferT[j] = setupCtx.proverHelpers.zi[row + j + d*domainSize];
                }
                Goldilocks::load_avx512(bufferT_[nColsStagesAcc[ns*nOpenings] + 1 + d], &bufferT[0]);
            }
        } else if(dests[0].params[0].parserParams.expId == int64_t(setupCtx.starkInfo.friExpId)) {
            for(uint64_t d = 0; d < setupCtx.starkInfo.openingPoints.size(); ++d) {
               for(uint64_t k = 0; k < FIELD_EXTENSION; ++k) {
                    for(uint64_t j = 0; j < nrowsPack; ++j) {
                        bufferT[j] = params.xDivXSub[((row + j)*setupCtx.starkInfo.openingPoints.size() + d)*FIELD_EXTENSION + k];
                    }
                    Goldilocks::load_avx512(bufferT_[nColsStagesAcc[ns*nOpenings] + d*FIELD_EXTENSION + k], &bufferT[0]);
                }
            }
        } else {
            for(uint64_t j = 0; j < nrowsPack; ++j) {
                bufferT[j] = setupCtx.proverHelpers.x_n[row + j];
            }
            Goldilocks::load_avx512(bufferT_[nColsStagesAcc[ns*nOpenings]], &bufferT[0]);
        }
    }

    inline void copyPolynomial(__m512i* destVals, bool inverse, uint64_t dim, __m512i* tmp) {
        if(dim == 1) {
            if(inverse) {
                Goldilocks::Element buff[nrowsPack];
                Goldilocks::store_avx512(buff, tmp[0]);
                Goldilocks::batchInverse(buff, buff, nrowsPack);
                Goldilocks::load_avx512(destVals[0], buff);
            } else {
                Goldilocks::copy_avx512(destVals[0],tmp[0]);
            }
        } else if(dim == FIELD_EXTENSION) {
            if(inverse) {
                Goldilocks::Element buff[FIELD_EXTENSION*nrowsPack];
                Goldilocks::store_avx512( &buff[0], uint64_t(FIELD_EXTENSION), tmp[0]);
                Goldilocks::store_avx512( &buff[1], uint64_t(FIELD_EXTENSION), tmp[1]);
                Goldilocks::store_avx512( &buff[2], uint64_t(FIELD_EXTENSION), tmp[2]);
                Goldilocks3::batchInverse((Goldilocks3::Element *)buff, (Goldilocks3::Element *)buff, nrowsPack);
                Goldilocks::load_avx512(destVals[0], &buff[0], uint64_t(FIELD_EXTENSION));
                Goldilocks::load_avx512(destVals[1], &buff[1], uint64_t(FIELD_EXTENSION));
                Goldilocks::load_avx512(destVals[2], &buff[2], uint64_t(FIELD_EXTENSION));
            } else {
                Goldilocks::copy_avx512(destVals[0], tmp[0]);
                Goldilocks::copy_avx512(destVals[1],tmp[1]);
                Goldilocks::copy_avx512(destVals[2],tmp[2]);
            }
        }
    }

    inline void multiplyPolynomials(Dest &dest, __m512i* destVals) {
        if(dest.dim == 1) {
            Goldilocks::op_avx512(2, destVals[0], destVals[0], destVals[FIELD_EXTENSION]);
        } else {
            __m512i vals3[FIELD_EXTENSION];
            if(dest.params[0].dim == FIELD_EXTENSION && dest.params[1].dim == FIELD_EXTENSION) {
                Goldilocks3::op_avx512(2, (Goldilocks3::Element_avx512 &)vals3, (Goldilocks3::Element_avx512 &)destVals[0], (Goldilocks3::Element_avx512 &)destVals[FIELD_EXTENSION]);
            } else if(dest.params[0].dim == FIELD_EXTENSION && dest.params[1].dim == 1) {
                Goldilocks3::op_31_avx512(2, (Goldilocks3::Element_avx512 &)vals3, (Goldilocks3::Element_avx512 &)destVals[0], destVals[FIELD_EXTENSION]);
            } else {
                Goldilocks3::op_31_avx512(2, (Goldilocks3::Element_avx512 &)vals3, (Goldilocks3::Element_avx512 &)destVals[FIELD_EXTENSION], destVals[0]);
            }
            Goldilocks::copy_avx512(destVals[0], vals3[0]);
            Goldilocks::copy_avx512(destVals[1], vals3[1]);
            Goldilocks::copy_avx512(destVals[2], vals3[2]);
        }
    }

    inline void storePolynomial(std::vector<Dest> dests, __m512i** destVals, uint64_t row) {
        for(uint64_t i = 0; i < dests.size(); ++i) {
            if(row >= dests[i].domainSize) continue;
            if(dests[i].dim == 1) {
                uint64_t offset = dests[i].offset != 0 ? dests[i].offset : 1;
                Goldilocks::store_avx512(&dests[i].dest[row*offset], uint64_t(offset), destVals[i][0]);
            } else {
                uint64_t offset = dests[i].offset != 0 ? dests[i].offset : FIELD_EXTENSION;
                Goldilocks::store_avx512(&dests[i].dest[row*offset], uint64_t(offset), destVals[i][0]);
                Goldilocks::store_avx512(&dests[i].dest[row*offset + 1], uint64_t(offset),destVals[i][1]);
                Goldilocks::store_avx512(&dests[i].dest[row*offset + 2], uint64_t(offset), destVals[i][2]);
            }
        }
    }

    inline void printTmp1(uint64_t row, __m512i tmp) {
        Goldilocks::Element buff[nrowsPack];
        Goldilocks::store_avx512(buff, tmp);
        for(uint64_t i = 0; i < nrowsPack; ++i) {
            cout << "Value at row " << row + i << " is " << Goldilocks::toString(buff[i]) << endl;
        }
    }

    inline void printTmp3(uint64_t row, Goldilocks3::Element_avx512 tmp) {
        Goldilocks::Element buff[FIELD_EXTENSION*nrowsPack];
        Goldilocks::store_avx512(&buff[0], uint64_t(FIELD_EXTENSION), tmp[0]);
        Goldilocks::store_avx512(&buff[1], uint64_t(FIELD_EXTENSION), tmp[1]);
        Goldilocks::store_avx512(&buff[2], uint64_t(FIELD_EXTENSION), tmp[2]);
        for(uint64_t i = 0; i < nrowsPack; ++i) {
            cout << "Value at row " << row + i << " is [" << Goldilocks::toString(buff[FIELD_EXTENSION*i]) << ", " << Goldilocks::toString(buff[FIELD_EXTENSION*i + 1]) << ", " << Goldilocks::toString(buff[FIELD_EXTENSION*i + 2]) << "]" << endl;
        }
    }

    inline void printCommit(uint64_t row, __m512i* bufferT, bool extended) {
        if(extended) {
            Goldilocks::Element buff[FIELD_EXTENSION*nrowsPack];
            Goldilocks::store_avx512(&buff[0], uint64_t(FIELD_EXTENSION), bufferT[0]);
            Goldilocks::store_avx512(&buff[1], uint64_t(FIELD_EXTENSION), bufferT[setupCtx.starkInfo.openingPoints.size()]);
            Goldilocks::store_avx512(&buff[2], uint64_t(FIELD_EXTENSION), bufferT[2*setupCtx.starkInfo.openingPoints.size()]);
            for(uint64_t i = 0; i < nrowsPack; ++i) {
                cout << "Value at row " << row + i << " is [" << Goldilocks::toString(buff[FIELD_EXTENSION*i]) << ", " << Goldilocks::toString(buff[FIELD_EXTENSION*i + 1]) << ", " << Goldilocks::toString(buff[FIELD_EXTENSION*i + 2]) << "]" << endl;
            }
        } else {
            Goldilocks::Element buff[nrowsPack];
            Goldilocks::store_avx512(&buff[0], bufferT[0]);
            for(uint64_t i = 0; i < nrowsPack; ++i) {
                cout << "Value at row " << row + i << " is " << Goldilocks::toString(buff[i]) << endl;
            }
        }
    }

    void calculateExpressions(StepsParams& params, ParserArgs &parserArgs, std::vector<Dest> dests, uint64_t domainSize, bool compilation_time) override {
        uint64_t nOpenings = setupCtx.starkInfo.openingPoints.size();
        uint64_t ns = 2 + setupCtx.starkInfo.nStages + setupCtx.starkInfo.customCommits.size();
        bool domainExtended = domainSize == uint64_t(1 << setupCtx.starkInfo.starkStruct.nBitsExt) ? true : false;

        uint64_t expId = dests[0].params[0].op == opType::tmp ? dests[0].expId : 0;
        setBufferTInfo(domainExtended, expId);

        __m512i *numbers_ = new __m512i[parserArgs.nNumbers];
        for(uint64_t i = 0; i < parserArgs.nNumbers; ++i) {
            numbers_[i] = _mm512_set1_epi64(parserArgs.numbers[i]);
        }

        Goldilocks3::Element_avx512 challenges[setupCtx.starkInfo.challengesMap.size()];
        for(uint64_t i = 0; i < setupCtx.starkInfo.challengesMap.size(); ++i) {
            challenges[i][0] = _mm512_set1_epi64(params.challenges[i * FIELD_EXTENSION].fe);
            challenges[i][1] = _mm512_set1_epi64(params.challenges[i * FIELD_EXTENSION + 1].fe);
            challenges[i][2] = _mm512_set1_epi64(params.challenges[i * FIELD_EXTENSION + 2].fe);
        }

        __m512i publics[setupCtx.starkInfo.nPublics];
        for(uint64_t i = 0; i < setupCtx.starkInfo.nPublics; ++i) {
            publics[i] = _mm512_set1_epi64(params.publicInputs[i].fe);
        }

        uint64_t p = 0;
        Goldilocks3::Element_avx512 proofValues[setupCtx.starkInfo.proofValuesMap.size()];
        for(uint64_t i = 0; i < setupCtx.starkInfo.proofValuesMap.size(); ++i) {
            if(setupCtx.starkInfo.proofValuesMap[i].stage == 1) {
               proofValues[i][0] = _mm512_set1_epi64(params.proofValues[p].fe);
               proofValues[i][1] = _mm512_set1_epi64(0);
               proofValues[i][2] = _mm512_set1_epi64(0);
               p += 1;
            } else {
               proofValues[i][0] = _mm512_set1_epi64(params.proofValues[p].fe);
               proofValues[i][1] = _mm512_set1_epi64(params.proofValues[p + 1].fe);
               proofValues[i][2] = _mm512_set1_epi64(params.proofValues[p + 2].fe);
               p += 3;
            }
        }

        Goldilocks3::Element_avx512 airgroupValues[setupCtx.starkInfo.airgroupValuesMap.size()];
        p = 0;
        for(uint64_t i = 0; i < setupCtx.starkInfo.airgroupValuesMap.size(); ++i) {
            if(setupCtx.starkInfo.airgroupValuesMap[i].stage == 1) {
               airgroupValues[i][0] = _mm512_set1_epi64(params.airgroupValues[p].fe);
               airgroupValues[i][1] = _mm512_set1_epi64(0);
               airgroupValues[i][2] = _mm512_set1_epi64(0);
               p += 1;
            } else {
               airgroupValues[i][0] = _mm512_set1_epi64(params.airgroupValues[p].fe);
               airgroupValues[i][1] = _mm512_set1_epi64(params.airgroupValues[p+ 1].fe);
               airgroupValues[i][2] = _mm512_set1_epi64(params.airgroupValues[p+ 2].fe);
               p += 3;
            }
        }

        Goldilocks3::Element_avx512 airValues[setupCtx.starkInfo.airValuesMap.size()];
        p = 0;
        for(uint64_t i = 0; i < setupCtx.starkInfo.airValuesMap.size(); ++i) {
            if(setupCtx.starkInfo.airValuesMap[i].stage == 1) {
               airValues[i][0] = _mm512_set1_epi64(params.airValues[p].fe);
               airValues[i][1] = _mm512_set1_epi64(0);
               airValues[i][2] = _mm512_set1_epi64(0);
               p += 1;
            } else {
               airValues[i][0] = _mm512_set1_epi64(params.airValues[p].fe);
               airValues[i][1] = _mm512_set1_epi64(params.airValues[p + 1].fe);
               airValues[i][2] = _mm512_set1_epi64(params.airValues[p + 2].fe);
               p += 3;
            }
        }

        Goldilocks3::Element_avx512 evals[setupCtx.starkInfo.evMap.size()];
        for(uint64_t i = 0; i < setupCtx.starkInfo.evMap.size(); ++i) {
            evals[i][0] = _mm512_set1_epi64(params.evals[i * FIELD_EXTENSION].fe);
            evals[i][1] = _mm512_set1_epi64(params.evals[i * FIELD_EXTENSION + 1].fe);
            evals[i][2] = _mm512_set1_epi64(params.evals[i * FIELD_EXTENSION + 2].fe);
        }

        Goldilocks::Element *bufferL = new Goldilocks::Element[omp_get_max_threads()*nOpenings*nrowsPack];
        __m512i* bufferT = new __m512i[omp_get_max_threads()*nOpenings*nCols];

        uint64_t maxTemp1Size = 0;
        uint64_t maxTemp3Size = 0;
        for (uint64_t j = 0; j < dests.size(); ++j) {
            for (uint64_t k = 0; k < dests[j].params.size(); ++k) {
                if (dests[j].params[k].parserParams.nTemp1 > maxTemp1Size) {
                    maxTemp1Size = dests[j].params[k].parserParams.nTemp1;
                }
                if (dests[j].params[k].parserParams.nTemp3 > maxTemp3Size) {
                    maxTemp3Size = dests[j].params[k].parserParams.nTemp3;
                }
            }
        }

        __m512i *tmp1_ = new __m512i[omp_get_max_threads() * maxTemp1Size];
        Goldilocks3::Element_avx512 *tmp3_ = new Goldilocks3::Element_avx512[omp_get_max_threads() * maxTemp3Size];

    #pragma omp parallel for
        for (uint64_t i = 0; i < domainSize; i+= nrowsPack) {
            __m512i* expressions_params[10];
            expressions_params[2] = publics;
            expressions_params[3] = numbers_;
            expressions_params[4] = (__m512i *)airValues;
            expressions_params[5] = (__m512i *)proofValues;
            expressions_params[7] = (__m512i *)airgroupValues;
            expressions_params[8] = (__m512i *)challenges;
            expressions_params[9] = (__m512i *)evals;

            __m512i* bufferT_ = &bufferT[omp_get_thread_num()*nOpenings*nCols];

            loadPolynomials(params, parserArgs, dests, &bufferL[omp_get_thread_num()*nOpenings*nrowsPack], bufferT_, i, domainSize);

            __m512i** destVals = new __m512i*[dests.size()];

            for(uint64_t j = 0; j < dests.size(); ++j) {
                if(i >= dests[j].domainSize) continue;
                destVals[j] = new __m512i[dests[j].params.size() * FIELD_EXTENSION];
                for(uint64_t k = 0; k < dests[j].params.size(); ++k) {
                    uint64_t i_args = 0;

                    if(dests[j].params[k].op == opType::cm || dests[j].params[k].op == opType::const_) {
                        uint64_t openingPointIndex = dests[j].params[k].rowOffsetIndex;
                        uint64_t buffPos = ns*openingPointIndex + dests[j].params[k].stage;
                        uint64_t stagePos = dests[j].params[k].stagePos;
                        copyPolynomial(&destVals[j][k*FIELD_EXTENSION], dests[j].params[k].inverse,dests[j].params[k].dim, &bufferT_[nColsStagesAcc[buffPos] + stagePos]);
                        continue;
                    } else if(dests[j].params[k].op == opType::number) {
                        destVals[j][k*FIELD_EXTENSION] = _mm512_set1_epi64(dests[j].params[k].value);
                        continue;
                    } else if(dests[j].params[k].op == opType::airvalue) {
                        Goldilocks::copy_avx512(destVals[j][k*FIELD_EXTENSION], airValues[dests[j].params[k].polsMapId][0]);
                        Goldilocks::copy_avx512(destVals[j][k*FIELD_EXTENSION + 1], airValues[dests[j].params[k].polsMapId][1]);
                        Goldilocks::copy_avx512(destVals[j][k*FIELD_EXTENSION + 2], airValues[dests[j].params[k].polsMapId][2]);
                        continue;
                    }
                    uint8_t* ops = &parserArgs.ops[dests[j].params[k].parserParams.opsOffset];
                    uint16_t* args = &parserArgs.args[dests[j].params[k].parserParams.argsOffset];
                    expressions_params[0] = bufferT_;
                    expressions_params[1] = &tmp1_[omp_get_thread_num()*maxTemp1Size];
                    expressions_params[6] = (__m512i *)&tmp3_[omp_get_thread_num()*maxTemp3Size];

                    for (uint64_t kk = 0; kk < dests[j].params[k].parserParams.nOps; ++kk) {
                        switch (ops[kk]) {
                            case 0: {
                                // COPY dim1 to dim1
                                Goldilocks::copy_avx512(expressions_params[args[i_args]][(nColsStagesAcc[args[i_args + 1]] + args[i_args + 2])], expressions_params[args[i_args + 3]][(nColsStagesAcc[args[i_args + 4]] + args[i_args + 5])]);
                                i_args += 7;
                                break;
                            }
                            case 1: {
                                // OPERATION WITH DEST: dim1 - SRC0: dim1 - SRC1: dim1
                                Goldilocks::op_avx512(args[i_args], expressions_params[args[i_args + 1]][(nColsStagesAcc[args[i_args + 2]] + args[i_args + 3])], expressions_params[args[i_args + 4]][(nColsStagesAcc[args[i_args + 5]] + args[i_args + 6])], expressions_params[args[i_args + 8]][(nColsStagesAcc[args[i_args + 9]] + args[i_args + 10])]);
                                i_args += 12;
                                break;
                            }
                            case 2: {
                                // OPERATION WITH DEST: dim3 - SRC0: dim3 - SRC1: dim1
                                Goldilocks3::op_31_avx512(args[i_args], (Goldilocks3::Element_avx512 &)expressions_params[args[i_args + 1]][(nColsStagesAcc[args[i_args + 2]] + args[i_args + 3])], (Goldilocks3::Element_avx512 &)expressions_params[args[i_args + 4]][(nColsStagesAcc[args[i_args + 5]] + args[i_args + 6])], expressions_params[args[i_args + 8]][(nColsStagesAcc[args[i_args + 9]] + args[i_args + 10])]);
                                i_args += 12;
                                break;
                            }
                            case 3: {
                                // OPERATION WITH DEST: dim3 - SRC0: dim3 - SRC1: dim3
                                Goldilocks3::op_avx512(args[i_args], (Goldilocks3::Element_avx512 &)expressions_params[args[i_args + 1]][(nColsStagesAcc[args[i_args + 2]] + args[i_args + 3])], (Goldilocks3::Element_avx512 &)expressions_params[args[i_args + 4]][(nColsStagesAcc[args[i_args + 5]] + args[i_args + 6])], (Goldilocks3::Element_avx512 &)expressions_params[args[i_args + 8]][(nColsStagesAcc[args[i_args + 9]] + args[i_args + 10])]);
                                i_args += 12;
                                break;
                            }
                            case 4: {
                                // COPY dim3 to dim3
                                Goldilocks3::copy_avx512((Goldilocks3::Element_avx512 &)expressions_params[args[i_args]][(nColsStagesAcc[args[i_args + 1]] + args[i_args + 2])], (Goldilocks3::Element_avx512 &)expressions_params[args[i_args + 3]][(nColsStagesAcc[args[i_args + 4]] + args[i_args + 5])]);
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
                        copyPolynomial(&destVals[j][k*FIELD_EXTENSION], dests[j].params[k].inverse, dests[j].params[k].parserParams.destDim, &expressions_params[1][dests[j].params[k].parserParams.destId]);
                    } else {
                        copyPolynomial(&destVals[j][k*FIELD_EXTENSION], dests[j].params[k].inverse, dests[j].params[k].parserParams.destDim, &expressions_params[6][dests[j].params[k].parserParams.destId]);
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
        delete[] bufferT;
        delete[] bufferL;
        delete[] tmp1_;
        delete[] tmp3_;
    }
};

#endif
#endif