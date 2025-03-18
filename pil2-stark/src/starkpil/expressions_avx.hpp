#ifndef EXPRESSIONS_AVX_HPP
#define EXPRESSIONS_AVX_HPP
#include "expressions_ctx.hpp"

#ifdef __AVX2__

class ExpressionsAvx : public ExpressionsCtx {
public:
    ExpressionsAvx(SetupCtx& setupCtx, ProverHelpers &proverHelpers) : ExpressionsCtx(setupCtx, proverHelpers) {
        nrowsPack = 4;
    };

    inline __m256i* load(__m256i *value, StepsParams& params, __m256i** expressions_params, uint16_t* args, uint64_t *mapOffsetsExps, uint64_t* mapOffsetsCustomExps, int64_t* nextStridesExps, uint64_t i_args, uint64_t row, uint64_t dim, uint64_t domainSize, bool domainExtended, bool isCyclic) {        
        uint64_t type = args[i_args];
        if (type == 0) {
            if(dim == FIELD_EXTENSION) { exit(-1); }
            Goldilocks::Element *constPols = domainExtended ? &params.pConstPolsExtendedTreeAddress[2] : params.pConstPolsAddress;
            uint64_t stagePos = args[i_args + 1];
            int64_t o = nextStridesExps[args[i_args + 2]];
            uint64_t nCols = mapSectionsN[0];
            if(isCyclic) {
                Goldilocks::Element buff[nrowsPack];
                for(uint64_t j = 0; j < nrowsPack; ++j) {
                    int64_t l = (row + j + o) % domainSize;
                    buff[j] = constPols[stagePos + l * nCols];
                }
                Goldilocks::load_avx(value[0], buff);
            } else {
                Goldilocks::load_avx(value[0], &constPols[stagePos + (row + o) * nCols], nCols);
            }            
            return value;
        } else if (type <= nStages + 1) {
            uint64_t stagePos = args[i_args + 1];
            uint64_t offset = mapOffsetsExps[type];
            uint64_t nCols = mapSectionsN[type];
            int64_t o = nextStridesExps[args[i_args + 2]];
            if(isCyclic) {
                Goldilocks::Element buff[dim * nrowsPack];
                for(uint64_t j = 0; j < nrowsPack; ++j) {
                    uint64_t l = (row + j + o) % domainSize;
                    if(type == 1 && !domainExtended) {
                        buff[j] = params.trace[stagePos + l * nCols];
                    } else {
                        for(uint64_t d = 0; d < dim; ++d) {
                            buff[j + d*nrowsPack] = params.aux_trace[offset + stagePos + l * nCols + d];
                        }
                    }
                }
                for(uint64_t d = 0; d < dim; ++d) {
                    Goldilocks::load_avx(value[d], &buff[d * nrowsPack]);
                }
            } else {
                if(type == 1 && !domainExtended) {
                    Goldilocks::load_avx(value[0], &params.trace[stagePos + (row + o) * nCols], nCols);
                } else {
                    for(uint64_t d = 0; d < dim; ++d) {
                        Goldilocks::load_avx(value[d], &params.aux_trace[offset + stagePos + (row + o) * nCols + d], nCols);
                    }
                }
            }
            return value;
        } else if (type == nStages + 2) {
            uint64_t boundary = args[i_args + 1];
            if(boundary == 0) {
                Goldilocks::Element *x = domainExtended ? &proverHelpers.x[row] : &proverHelpers.x_n[row];
                Goldilocks::load_avx(value[0], x);
            } else {
                Goldilocks::load_avx(value[0], &proverHelpers.zi[(boundary -1)*domainSize + row]);
            }
            return value;
        } else if (type == nStages + 3) {
            if(dim == 1) { exit(-1); }
            uint64_t o = args[i_args + 1];
            Goldilocks::Element buff[nrowsPack * FIELD_EXTENSION];
            __m256i x_avx;
            value[0] = _mm256_set1_epi64x(xis[o * FIELD_EXTENSION].fe);
            value[1] = _mm256_set1_epi64x(xis[o * FIELD_EXTENSION + 1].fe);
            value[2] = _mm256_set1_epi64x(xis[o * FIELD_EXTENSION + 2].fe);
            Goldilocks::load_avx(x_avx, &proverHelpers.x[row]);
            Goldilocks3::op_31_avx(3, (Goldilocks3::Element_avx &)value[0], (Goldilocks3::Element_avx &)value[0], x_avx);
            Goldilocks::store_avx(&buff[0], FIELD_EXTENSION, value[0]);
            Goldilocks::store_avx(&buff[1], FIELD_EXTENSION, value[1]);
            Goldilocks::store_avx(&buff[2], FIELD_EXTENSION, value[2]);
            Goldilocks3::batchInverse((Goldilocks3::Element *)buff, (Goldilocks3::Element *)buff, nrowsPack);
            Goldilocks::load_avx(value[0], &buff[0], FIELD_EXTENSION);
            Goldilocks::load_avx(value[1], &buff[1], FIELD_EXTENSION);
            Goldilocks::load_avx(value[2], &buff[2], FIELD_EXTENSION);
            Goldilocks3::op_31_avx(2, (Goldilocks3::Element_avx &)value[0], (Goldilocks3::Element_avx &)value[0], x_avx);
            return value;
        } else if (type >= nStages + 4 && type < setupCtx.starkInfo.customCommits.size() + nStages + 4) {
            uint64_t index = type - (nStages + 4);
            uint64_t stagePos = args[i_args + 1];
            uint64_t offset = mapOffsetsCustomExps[index];
            uint64_t nCols = mapSectionsNCustomFixed[index];
            int64_t o = nextStridesExps[args[i_args + 2]];
            if(isCyclic) {
                Goldilocks::Element buff[nrowsPack];
                for(uint64_t j = 0; j < nrowsPack; ++j) {
                    uint64_t l = (row + j + o) % domainSize;
                    buff[j] = params.pCustomCommitsFixed[offset + l * nCols + stagePos];
                }
                Goldilocks::load_avx(value[0], buff);
            } else {
                Goldilocks::load_avx(value[0], &params.pCustomCommitsFixed[offset + (row + o) * nCols + stagePos], nCols);
            }
            
            return value;
        } else {
            return &expressions_params[type][args[i_args + 1]];
        }
    }

    inline void copyPolynomial(__m256i* destVals, bool inverse, uint64_t dim, __m256i* tmp) {
        if(dim == 1) {
            if(inverse) {
                Goldilocks::Element buff[nrowsPack];
                Goldilocks::store_avx(buff, tmp[0]);
                Goldilocks::batchInverse(buff, buff, nrowsPack);
                Goldilocks::load_avx(destVals[0], buff);
            } else {
                Goldilocks::copy_avx(destVals[0],tmp[0]);
            }
        } else if(dim == FIELD_EXTENSION) {
            if(inverse) {
                Goldilocks::Element buff[FIELD_EXTENSION*nrowsPack];
                Goldilocks::store_avx( &buff[0], uint64_t(FIELD_EXTENSION), tmp[0]);
                Goldilocks::store_avx( &buff[1], uint64_t(FIELD_EXTENSION), tmp[1]);
                Goldilocks::store_avx( &buff[2], uint64_t(FIELD_EXTENSION), tmp[2]);
                Goldilocks3::batchInverse((Goldilocks3::Element *)buff, (Goldilocks3::Element *)buff, nrowsPack);
                Goldilocks::load_avx(destVals[0], &buff[0], uint64_t(FIELD_EXTENSION));
                Goldilocks::load_avx(destVals[1], &buff[1], uint64_t(FIELD_EXTENSION));
                Goldilocks::load_avx(destVals[2], &buff[2], uint64_t(FIELD_EXTENSION));
            } else {
                Goldilocks::copy_avx(destVals[0],tmp[0]);
                Goldilocks::copy_avx(destVals[1],tmp[1]);
                Goldilocks::copy_avx(destVals[2],tmp[2]);
            }
        }
    }

    inline void multiplyPolynomials(Dest &dest, __m256i* destVals) {
        if(dest.dim == 1) {
            Goldilocks::op_avx(2, destVals[0], destVals[0], destVals[FIELD_EXTENSION]);
        } else {
            __m256i vals3[FIELD_EXTENSION];
            if(dest.params[0].dim == FIELD_EXTENSION && dest.params[1].dim == FIELD_EXTENSION) {
                Goldilocks3::op_avx(2, (Goldilocks3::Element_avx &)vals3, (Goldilocks3::Element_avx &)destVals[0], (Goldilocks3::Element_avx &)destVals[FIELD_EXTENSION]);
            } else if(dest.params[0].dim == FIELD_EXTENSION && dest.params[1].dim == 1) {
                Goldilocks3::op_31_avx(2, (Goldilocks3::Element_avx &)vals3, (Goldilocks3::Element_avx &)destVals[0], destVals[FIELD_EXTENSION]);
            } else {
                Goldilocks3::op_31_avx(2, (Goldilocks3::Element_avx &)vals3, (Goldilocks3::Element_avx &)destVals[FIELD_EXTENSION], destVals[0]);
            }
            Goldilocks::copy_avx(destVals[0], vals3[0]);
            Goldilocks::copy_avx(destVals[1], vals3[1]);
            Goldilocks::copy_avx(destVals[2], vals3[2]);
        }
    }

    inline void storePolynomial(std::vector<Dest> dests, __m256i** destVals, uint64_t row) {
        for(uint64_t i = 0; i < dests.size(); ++i) {
            if(row >= dests[i].domainSize) continue;
            if(dests[i].dim == 1) {
                uint64_t offset = dests[i].offset != 0 ? dests[i].offset : 1;
                Goldilocks::store_avx(&dests[i].dest[row*offset], uint64_t(offset), destVals[i][0]);
            } else {
                uint64_t offset = dests[i].offset != 0 ? dests[i].offset : FIELD_EXTENSION;
                Goldilocks::store_avx(&dests[i].dest[row*offset], uint64_t(offset), destVals[i][0]);
                Goldilocks::store_avx(&dests[i].dest[row*offset + 1], uint64_t(offset),destVals[i][1]);
                Goldilocks::store_avx(&dests[i].dest[row*offset + 2], uint64_t(offset), destVals[i][2]);
            }
        }
    }

    inline void printTmp1(uint64_t row, __m256i tmp) {
        Goldilocks::Element buff[nrowsPack];
        Goldilocks::store_avx(buff, tmp);
        for(uint64_t i = 0; i < nrowsPack; ++i) {
            cout << "Value at row " << row + i << " is " << Goldilocks::toString(buff[i]) << endl;
        }
    }

    inline void printTmp3(uint64_t row, Goldilocks3::Element_avx tmp) {
        Goldilocks::Element buff[FIELD_EXTENSION*nrowsPack];
        Goldilocks::store_avx(&buff[0], uint64_t(FIELD_EXTENSION), tmp[0]);
        Goldilocks::store_avx(&buff[1], uint64_t(FIELD_EXTENSION), tmp[1]);
        Goldilocks::store_avx(&buff[2], uint64_t(FIELD_EXTENSION), tmp[2]);
        for(uint64_t i = 0; i < nrowsPack; ++i) {
            cout << "Value at row " << row + i << " is [" << Goldilocks::toString(buff[FIELD_EXTENSION*i]) << ", " << Goldilocks::toString(buff[FIELD_EXTENSION*i + 1]) << ", " << Goldilocks::toString(buff[FIELD_EXTENSION*i + 2]) << "]" << endl;
        }
    }

    void calculateExpressions(StepsParams& params, ParserArgs &parserArgs, std::vector<Dest> dests, uint64_t domainSize, bool domainExtended, bool compilation_time) override {
        uint64_t *mapOffsetsExps = domainExtended ? mapOffsetsExtended : mapOffsets;
        uint64_t *mapOffsetsCustomExps = domainExtended ? mapOffsetsCustomFixedExtended : mapOffsetsCustomFixed;
        int64_t *nextStridesExps = domainExtended ? nextStridesExtended : nextStrides;
        
        uint64_t k_min = domainExtended 
            ? uint64_t((minRowExtended + nrowsPack - 1) / nrowsPack) * nrowsPack
            : uint64_t((minRow + nrowsPack - 1) / nrowsPack) * nrowsPack;
        uint64_t k_max = domainExtended
            ? uint64_t(maxRowExtended / nrowsPack)*nrowsPack
            : uint64_t(maxRow / nrowsPack)*nrowsPack;
        
        __m256i *numbers_ = new __m256i[parserArgs.nNumbers];
        for(uint64_t i = 0; i < parserArgs.nNumbers; ++i) {
            numbers_[i] = _mm256_set1_epi64x(parserArgs.numbers[i].fe);
        }

        __m256i challenges[nChallenges*FIELD_EXTENSION];
        for(uint64_t i = 0; i < nChallenges; ++i) {
            challenges[FIELD_EXTENSION*i] = _mm256_set1_epi64x(params.challenges[i * FIELD_EXTENSION].fe);
            challenges[FIELD_EXTENSION*i + 1] = _mm256_set1_epi64x(params.challenges[i * FIELD_EXTENSION + 1].fe);
            challenges[FIELD_EXTENSION*i + 2] = _mm256_set1_epi64x(params.challenges[i * FIELD_EXTENSION + 2].fe);
        }

        __m256i publics[nPublics];
        for(uint64_t i = 0; i < nPublics; ++i) {
            publics[i] = _mm256_set1_epi64x(params.publicInputs[i].fe);
        }

        uint64_t p = 0;
        __m256i proofValues[setupCtx.starkInfo.proofValuesMap.size()*FIELD_EXTENSION];
        for(uint64_t i = 0; i < setupCtx.starkInfo.proofValuesMap.size(); ++i) {
            if(setupCtx.starkInfo.proofValuesMap[i].stage == 1) {
               proofValues[p] = _mm256_set1_epi64x(params.proofValues[p].fe);
               p += 1;
            } else {
               proofValues[p] = _mm256_set1_epi64x(params.proofValues[p].fe);
               proofValues[p + 1] = _mm256_set1_epi64x(params.proofValues[p + 1].fe);
               proofValues[p + 2] = _mm256_set1_epi64x(params.proofValues[p + 2].fe);
               p += 3;
            }
        }

        __m256i airgroupValues[setupCtx.starkInfo.airgroupValuesMap.size()*FIELD_EXTENSION];
        p = 0;
        for(uint64_t i = 0; i < setupCtx.starkInfo.airgroupValuesMap.size(); ++i) {
            if(setupCtx.starkInfo.airgroupValuesMap[i].stage == 1) {
               airgroupValues[p] = _mm256_set1_epi64x(params.airgroupValues[p].fe);
               p += 1;
            } else {
               airgroupValues[p] = _mm256_set1_epi64x(params.airgroupValues[p].fe);
               airgroupValues[p + 1] = _mm256_set1_epi64x(params.airgroupValues[p+1].fe);
               airgroupValues[p + 2] = _mm256_set1_epi64x(params.airgroupValues[p+2].fe);
               p += 3;
            }
        }

        __m256i airValues[setupCtx.starkInfo.airValuesMap.size()*FIELD_EXTENSION];
        p = 0;
        for(uint64_t i = 0; i < setupCtx.starkInfo.airValuesMap.size(); ++i) {
            if(setupCtx.starkInfo.airValuesMap[i].stage == 1) {
               airValues[p] = _mm256_set1_epi64x(params.airValues[p].fe);
               p += 1;
            } else {
               airValues[p] = _mm256_set1_epi64x(params.airValues[p].fe);
               airValues[p+1] = _mm256_set1_epi64x(params.airValues[p + 1].fe);
               airValues[p+2] = _mm256_set1_epi64x(params.airValues[p + 2].fe);
               p += 3;
            }
        }

        __m256i evals[nEvals*FIELD_EXTENSION];
        for(uint64_t i = 0; i < nEvals; ++i) {
            evals[FIELD_EXTENSION*i] = _mm256_set1_epi64x(params.evals[i * FIELD_EXTENSION].fe);
            evals[FIELD_EXTENSION*i + 1] = _mm256_set1_epi64x(params.evals[i * FIELD_EXTENSION + 1].fe);
            evals[FIELD_EXTENSION*i + 2] = _mm256_set1_epi64x(params.evals[i * FIELD_EXTENSION + 2].fe);
        }

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
        maxTemp3Size *= FIELD_EXTENSION;

        __m256i *tmp1_ = new __m256i[omp_get_max_threads() * maxTemp1Size];
        __m256i *tmp3_ = new __m256i[omp_get_max_threads() * maxTemp3Size];
        __m256i *values_ = new __m256i[omp_get_max_threads() * 2 * FIELD_EXTENSION];

    #pragma omp parallel for
        for (uint64_t i = 0; i < domainSize; i+= nrowsPack) {
            bool isCyclic = i < k_min || i >= k_max;
            uint64_t expressions_params_size = bufferCommitsSize + 9;
            __m256i* expressions_params[expressions_params_size];
            expressions_params[bufferCommitsSize + 2] = publics;
            expressions_params[bufferCommitsSize + 3] = numbers_;
            expressions_params[bufferCommitsSize + 4] = airValues;
            expressions_params[bufferCommitsSize + 5] = proofValues;
            expressions_params[bufferCommitsSize + 6] = airgroupValues;
            expressions_params[bufferCommitsSize + 7] = challenges;
            expressions_params[bufferCommitsSize + 8] = evals;

            __m256i** destVals = new __m256i*[dests.size()];

            for(uint64_t j = 0; j < dests.size(); ++j) {
                if(i >= dests[j].domainSize) continue;
                destVals[j] = new __m256i[dests[j].params.size() * FIELD_EXTENSION];
                for(uint64_t k = 0; k < dests[j].params.size(); ++k) {
                    uint64_t i_args = 0;

                    if(dests[j].params[k].op == opType::cm || dests[j].params[k].op == opType::const_) {
                        uint64_t openingPointIndex = dests[j].params[k].rowOffsetIndex;
                        uint64_t stagePos = dests[j].params[k].stagePos;
                        int64_t o = nextStridesExps[openingPointIndex];
                        uint64_t nCols = mapSectionsN[0];
                        Goldilocks::Element buff[FIELD_EXTENSION * nrowsPack];
                        if (dests[j].params[k].op == opType::const_) {
                            for(uint64_t r = 0; r < nrowsPack; ++r) {
                                uint64_t l = (i + r + o) % domainSize;
                                buff[r] = params.pConstPolsAddress[l * nCols + stagePos];
                            }
                        } else {
                            uint64_t offset = mapOffsetsExps[dests[j].params[k].stage];
                            uint64_t nCols = mapSectionsN[dests[j].params[k].stage];
                            for(uint64_t r = 0; r < nrowsPack; ++r) {
                                uint64_t l = (i + r + o) % domainSize;
                                if(dests[j].params[k].stage == 1) {
                                    buff[r] = params.trace[l * nCols + stagePos];
                                } else {
                                    for(uint64_t d = 0; d < dests[j].params[k].dim; ++d) {
                                        buff[r + d*nrowsPack] = params.aux_trace[offset + l * nCols + stagePos + d];
                                    }
                                }
                            }
                        }
                        if(dests[j].params[k].inverse) {
                            if(dests[j].params[k].dim == 1) {
                                Goldilocks::batchInverse(buff, buff, nrowsPack);
                            } else {
                                Goldilocks3::batchInverse((Goldilocks3::Element *)buff, (Goldilocks3::Element *)buff, nrowsPack);
                            }
                        }
                        for(uint64_t d = 0; d < dests[j].params[k].dim; ++d) {
                            Goldilocks::load_avx(destVals[j][k*FIELD_EXTENSION + d], &buff[d*nrowsPack]);
                        }
                        continue;
                    } else if(dests[j].params[k].op == opType::number) {
                        destVals[j][k*FIELD_EXTENSION] = _mm256_set1_epi64x(dests[j].params[k].value);
                        continue;
                    } else if(dests[j].params[k].op == opType::airvalue) {
                        Goldilocks::copy_avx(destVals[j][k*FIELD_EXTENSION], airValues[dests[j].params[k].polsMapId*FIELD_EXTENSION]);
                        Goldilocks::copy_avx(destVals[j][k*FIELD_EXTENSION + 1], airValues[dests[j].params[k].polsMapId*FIELD_EXTENSION + 1]);
                        Goldilocks::copy_avx(destVals[j][k*FIELD_EXTENSION + 2], airValues[dests[j].params[k].polsMapId*FIELD_EXTENSION + 2]);
                        continue;
                    }
                    uint8_t* ops = &parserArgs.ops[dests[j].params[k].parserParams.opsOffset];
                    uint16_t* args = &parserArgs.args[dests[j].params[k].parserParams.argsOffset];
                    expressions_params[bufferCommitsSize] = &tmp1_[omp_get_thread_num()*maxTemp1Size];
                    expressions_params[bufferCommitsSize + 1] = &tmp3_[omp_get_thread_num()*maxTemp3Size];

                    for (uint64_t kk = 0; kk < dests[j].params[k].parserParams.nOps; ++kk) {
                        switch (ops[kk]) {
                            case 0: {
                                // COPY dim1 to dim1
                                __m256i* a = load(&values_[omp_get_thread_num()*2*FIELD_EXTENSION], params, expressions_params, args, mapOffsetsExps, mapOffsetsCustomExps, nextStridesExps, i_args + 1, i, 1, domainSize, domainExtended, isCyclic);
                                Goldilocks::copy_avx(expressions_params[bufferCommitsSize][args[i_args]], a[0]);
                                i_args += 4;
                                break;
                            }
                            case 1: {
                                // OPERATION WITH DEST: dim1 - SRC0: dim1 - SRC1: dim1
                                __m256i* a = load(&values_[omp_get_thread_num()*2*FIELD_EXTENSION], params, expressions_params, args, mapOffsetsExps, mapOffsetsCustomExps, nextStridesExps, i_args + 2, i, 1, domainSize, domainExtended, isCyclic);
                                __m256i* b = load(&values_[omp_get_thread_num()*2*FIELD_EXTENSION + FIELD_EXTENSION], params, expressions_params, args, mapOffsetsExps, mapOffsetsCustomExps, nextStridesExps, i_args + 5, i, 1, domainSize, domainExtended, isCyclic);
                                Goldilocks::op_avx(args[i_args], expressions_params[bufferCommitsSize][args[i_args + 1]], a[0], b[0]);
                                i_args += 8;
                                break;
                            }
                            case 2: {
                                // OPERATION WITH DEST: dim3 - SRC0: dim3 - SRC1: dim1   
                                __m256i* a = load(&values_[omp_get_thread_num()*2*FIELD_EXTENSION], params, expressions_params, args, mapOffsetsExps, mapOffsetsCustomExps, nextStridesExps, i_args + 2, i, 3, domainSize, domainExtended, isCyclic);
                                __m256i* b = load(&values_[omp_get_thread_num()*2*FIELD_EXTENSION + FIELD_EXTENSION], params, expressions_params, args, mapOffsetsExps, mapOffsetsCustomExps, nextStridesExps, i_args + 5, i, 1, domainSize, domainExtended, isCyclic);
                                Goldilocks3::op_31_avx(args[i_args], (Goldilocks3::Element_avx &)expressions_params[bufferCommitsSize + 1][args[i_args + 1]], (Goldilocks3::Element_avx &)a[0], b[0]);
                                i_args += 8;
                                break;
                            }
                            case 3: {
                                // OPERATION WITH DEST: dim3 - SRC0: dim3 - SRC1: dim3
                                __m256i* a = load(&values_[omp_get_thread_num()*2*FIELD_EXTENSION], params, expressions_params, args, mapOffsetsExps, mapOffsetsCustomExps, nextStridesExps, i_args + 2, i, 3, domainSize, domainExtended, isCyclic);
                                __m256i* b = load(&values_[omp_get_thread_num()*2*FIELD_EXTENSION + FIELD_EXTENSION], params, expressions_params, args, mapOffsetsExps, mapOffsetsCustomExps, nextStridesExps, i_args + 5, i, 3, domainSize, domainExtended, isCyclic);
                                Goldilocks3::op_avx(args[i_args], (Goldilocks3::Element_avx &)expressions_params[bufferCommitsSize + 1][args[i_args + 1]], (Goldilocks3::Element_avx &)a[0], (Goldilocks3::Element_avx &)b[0]);
                                i_args += 8;
                                break;
                            }
                            case 4: {
                                // COPY dim3 to dim3
                                __m256i* a = load(&values_[omp_get_thread_num()*2*FIELD_EXTENSION], params, expressions_params, args, mapOffsetsExps, mapOffsetsCustomExps, nextStridesExps, i_args + 1, i, 3, domainSize, domainExtended, isCyclic);
                                Goldilocks3::copy_avx((Goldilocks3::Element_avx &)expressions_params[bufferCommitsSize + 1][args[i_args]], (Goldilocks3::Element_avx &)a[0]);
                                i_args += 4;
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
                        copyPolynomial(&destVals[j][k*FIELD_EXTENSION], dests[j].params[k].inverse, dests[j].params[k].parserParams.destDim, &expressions_params[bufferCommitsSize][dests[j].params[k].parserParams.destId]);
                    } else {
                        copyPolynomial(&destVals[j][k*FIELD_EXTENSION], dests[j].params[k].inverse, dests[j].params[k].parserParams.destDim, &expressions_params[bufferCommitsSize + 1][dests[j].params[k].parserParams.destId]);
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
        delete[] values_;
        delete[] tmp1_;
        delete[] tmp3_;
    }
};

#endif
#endif