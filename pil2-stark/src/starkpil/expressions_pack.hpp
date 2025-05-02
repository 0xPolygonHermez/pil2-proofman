#ifndef EXPRESSIONS_PACK_HPP
#define EXPRESSIONS_PACK_HPP
#include "expressions_ctx.hpp"

#define DEBUG 0
#define DEBUG_ROW 0

#define NROWS_PACK 128
class ExpressionsPack : public ExpressionsCtx {
public:
    ExpressionsPack(SetupCtx& setupCtx, ProverHelpers* proverHelpers, uint64_t nrowsPack = NROWS_PACK) : ExpressionsCtx(setupCtx, proverHelpers) {
        nrowsPack_ = std::min(nrowsPack, uint64_t(1 << setupCtx.starkInfo.starkStruct.nBits));
    };

    inline Goldilocks::Element* load(uint64_t nrowsPack, Goldilocks::Element *value, StepsParams& params, Goldilocks::Element** expressions_params, uint16_t* args, uint64_t *mapOffsetsExps, uint64_t* mapOffsetsCustomExps, int64_t* nextStridesExps, uint64_t i_args, uint64_t row, uint64_t dim, uint64_t domainSize, bool domainExtended, bool isCyclic) {
        
#if DEBUG 
        bool print = row == DEBUG_ROW;
#endif
        uint64_t type = args[i_args];

#if DEBUG
        //if(print) printf("Expression debug type: %lu nStages: %lu nCustomCommits: %lu bufferCommitSize: %lu\n", type, setupCtx.starkInfo.nStages, setupCtx.starkInfo.customCommits.size(), bufferCommitsSize);
#endif  
        if (type == 0) {
            if(dim == FIELD_EXTENSION) { exit(-1); }
            Goldilocks::Element *constPols = domainExtended ? &params.pConstPolsExtendedTreeAddress[2] : params.pConstPolsAddress;
            uint64_t stagePos = args[i_args + 1];
            int64_t o = nextStridesExps[args[i_args + 2]];
            uint64_t nCols = mapSectionsN[0];
            if(isCyclic) {
#if DEBUG 
                if(print) printf("Expression debug constPols cyclic\n");
#endif
                for(uint64_t j = 0; j < nrowsPack; ++j) {
                    uint64_t l = (row + j + o) % domainSize;
                    value[j] = constPols[l * setupCtx.starkInfo.nConstants + stagePos];
                }
                return value;
            } else {
#if DEBUG
                if(print) printf("Expression debug constPols\n");
#endif
                uint64_t offsetCol = (row + o) * nCols + stagePos;
                for(uint64_t j = 0; j < nrowsPack; ++j) {
                    value[j] = constPols[offsetCol + j*nCols];
                }
                return value;
            }
        } else if (type <= setupCtx.starkInfo.nStages + 1) {
            uint64_t stagePos = args[i_args + 1];
            uint64_t offset = mapOffsetsExps[type];
            uint64_t nCols = mapSectionsN[type];
            int64_t o = nextStridesExps[args[i_args + 2]];
            if(isCyclic) {

                for(uint64_t j = 0; j < nrowsPack; ++j) {
                    uint64_t l = (row + j + o) % domainSize;
                    if(type == 1 && !domainExtended) {
#if DEBUG
                        if(print && j==0) printf("Expression debug trace cyclic: %lu\n",l * nCols + stagePos);
#endif
                        value[j] = params.trace[l * nCols + stagePos];
                    } else {
#if DEBUG
                        if(print && j==0) printf("Expression debug aux_trace cyclic %lu\n", offset + l * nCols + stagePos);
#endif
                        for(uint64_t d = 0; d < dim; ++d) {
                            value[j + d*nrowsPack] = params.aux_trace[offset + l * nCols + stagePos + d];
                        }
                    }
                }
                return value;
            } else {
                if(type == 1 && !domainExtended) {
#if DEBUG
                    if(print) printf("Expression debug trace\n");
#endif
                    uint64_t offsetCol = (row + o) * nCols + stagePos;
                    for(uint64_t j = 0; j < nrowsPack; ++j) {
                        value[j] = params.trace[offsetCol + j*nCols];
                    }
                    return value;
                } else {
#if DEBUG
                    if(print) printf("Expression debug aux_trace\n");
#endif
                    uint64_t offsetCol = offset + (row + o) * nCols + stagePos;
                    for(uint64_t j = 0; j < nrowsPack; ++j) {
                        for(uint64_t d = 0; d < dim; ++d) {
                            value[j + d*nrowsPack] = params.aux_trace[offsetCol + d + j*nCols];
                        }
                    }
                    return value;
                }
            }
        } else if (type == setupCtx.starkInfo.nStages + 2) {
            uint64_t boundary = args[i_args + 1];
            if(setupCtx.starkInfo.verify) {
                if(boundary == 0) {
                    for(uint64_t j = 0; j < nrowsPack; ++j) {
                        for(uint64_t e = 0; e < FIELD_EXTENSION; ++e) {
                            value[j + e*nrowsPack] = proverHelpers->x_n[e];
                        }
                    }
                } else {
                    for(uint64_t j = 0; j < nrowsPack; ++j) {
                        for(uint64_t e = 0; e < FIELD_EXTENSION; ++e) {
                            value[j + e*nrowsPack] = proverHelpers->zi[(boundary - 1)*FIELD_EXTENSION + e];
                        }
                    }
                }
                return value;
            } else {
                if(boundary == 0) {
#if DEBUG
                if(print) printf("Expression debug x or x_n\n");
#endif
                    Goldilocks::Element *x = domainExtended ? &proverHelpers->x[row] : &proverHelpers->x_n[row];
                    return x;
                } else {
#if DEBUG
                    if(print) printf("Expression debug zi\n");
#endif
                    return &proverHelpers->zi[(boundary - 1)*domainSize  + row];
                }
            }
        } else if (type == setupCtx.starkInfo.nStages + 3) {
#if DEBUG
            if(print) printf("Expression debug xi\n");
#endif
            if(dim == 1) { exit(-1); }
            uint64_t o = args[i_args + 1];
            if(setupCtx.starkInfo.verify) {
                    for(uint64_t k = 0; k < nrowsPack; ++k) {
                    for(uint64_t e = 0; e < FIELD_EXTENSION; ++e) {
                        value[k + e*nrowsPack] = params.xDivXSub[((row + k)*setupCtx.starkInfo.openingPoints.size() + o)*FIELD_EXTENSION + e];
                    }
                }
                return value;
            } else {
                Goldilocks::Element *xdivxsub = &params.aux_trace[mapOffsetFriPol + row*FIELD_EXTENSION];
                Goldilocks3::op_31_pack(nrowsPack, 3, xdivxsub, &xis[o * FIELD_EXTENSION], true, &proverHelpers->x[row], false);
                getInversePolinomial(nrowsPack, xdivxsub, value, true, 3);
                Goldilocks3::op_31_pack(nrowsPack, 2, xdivxsub, xdivxsub, false, &proverHelpers->x[row], false);
                return xdivxsub;
            }
        } else if (type >= setupCtx.starkInfo.nStages + 4 && type < setupCtx.starkInfo.customCommits.size() + setupCtx.starkInfo.nStages + 4) {
            uint64_t index = type - (nStages + 4);
            uint64_t stagePos = args[i_args + 1];
            uint64_t offset = mapOffsetsCustomExps[index];
            uint64_t nCols = mapSectionsNCustomFixed[index];
            int64_t o = nextStridesExps[args[i_args + 2]];
            if(isCyclic) {
#if DEBUG
                if(print) printf("Expression debug customCommits cyclic\n");
#endif
                for(uint64_t j = 0; j < nrowsPack; ++j) {
                    uint64_t l = (row + j + o) % domainSize;
                    value[j] = params.pCustomCommitsFixed[offset + l * nCols + stagePos];
                }
                return value;
            } else {
#if DEBUG
                if(print) printf("Expression debug customCommits\n");
#endif
                uint64_t offsetCol = offset + (row + o) * nCols + stagePos;
                for(uint64_t j = 0; j < nrowsPack; ++j) {
                    value[j] = params.pCustomCommitsFixed[offsetCol + j*nCols];
                }
                return value;
            }
        } else if (type == bufferCommitsSize || type == bufferCommitsSize + 1) {
#if DEBUG
            if(print){ 
                if(type == bufferCommitsSize) printf("Expression debug tmp1\n");
                if(type == bufferCommitsSize + 1) printf("Expression debug tmp3\n");
            }
#endif
            return &expressions_params[type][args[i_args + 1]*nrowsPack];
        } else {
#if DEBUG
            if(print){
                if(type == bufferCommitsSize + 2 ) printf("Expression debug publicInputs\n");
                if(type == bufferCommitsSize + 3 ) printf("Expression debug numbers\n");
                if(type == bufferCommitsSize + 4 ) printf("Expression debug airValues\n");
                if(type == bufferCommitsSize + 5 ) printf("Expression debug proofValues\n");
                if(type == bufferCommitsSize + 6 ) printf("Expression debug airgroupValues\n");
                if(type == bufferCommitsSize + 7 ) printf("Expression debug challenges\n");
                if(type == bufferCommitsSize + 8 ) printf("Expression debug evals\n");
            }

#endif
            return &expressions_params[type][args[i_args + 1]];
        }
    }

    inline void getInversePolinomial(uint64_t nrowsPack, Goldilocks::Element* destVals, Goldilocks::Element* buffHelper, bool batch, uint64_t dim) {
        if(dim == 1) {
            if(batch) {
                Goldilocks::batchInverse(&destVals[0], &destVals[0], nrowsPack);
            } else {
                for(uint64_t i = 0; i < nrowsPack; ++i) {
                    Goldilocks::inv(destVals[i], destVals[i]);
                }
            }
        } else if(dim == FIELD_EXTENSION) {
            Goldilocks::copy_pack(nrowsPack, &buffHelper[0], uint64_t(FIELD_EXTENSION), &destVals[0]);
            Goldilocks::copy_pack(nrowsPack, &buffHelper[1], uint64_t(FIELD_EXTENSION), &destVals[nrowsPack]);
            Goldilocks::copy_pack(nrowsPack, &buffHelper[2], uint64_t(FIELD_EXTENSION), &destVals[2*nrowsPack]);
            if(batch) {
                Goldilocks3::batchInverse((Goldilocks3::Element *)buffHelper, (Goldilocks3::Element *)buffHelper, nrowsPack);
            } else {
                for(uint64_t i = 0; i < nrowsPack; ++i) {
                    Goldilocks3::inv((Goldilocks3::Element &)buffHelper[i*FIELD_EXTENSION], (Goldilocks3::Element &)buffHelper[i*FIELD_EXTENSION]);
                }
            }
            Goldilocks::copy_pack(nrowsPack, &destVals[0], &buffHelper[0], uint64_t(FIELD_EXTENSION));
            Goldilocks::copy_pack(nrowsPack, &destVals[nrowsPack], &buffHelper[1], uint64_t(FIELD_EXTENSION));
            Goldilocks::copy_pack(nrowsPack, &destVals[2*nrowsPack], &buffHelper[2], uint64_t(FIELD_EXTENSION));
        }
    }

    inline void multiplyPolynomials(uint64_t nrowsPack, Dest &dest, Goldilocks::Element* destVals, Goldilocks::Element* buffHelper, bool isConstantA, bool isConstantB) {
        if(dest.dim == 1) {
            Goldilocks::op_pack(nrowsPack, 2, &destVals[0], &destVals[0], isConstantA, &destVals[FIELD_EXTENSION*nrowsPack], isConstantB);
        } else {
            Goldilocks::Element buffHelper[FIELD_EXTENSION*nrowsPack];
            if(dest.params[0].dim == FIELD_EXTENSION && dest.params[1].dim == FIELD_EXTENSION) {
                Goldilocks3::op_pack(nrowsPack, 2, &buffHelper[0], &destVals[0], isConstantA, &destVals[FIELD_EXTENSION*nrowsPack], isConstantB);
            } else if(dest.params[0].dim == FIELD_EXTENSION && dest.params[1].dim == 1) {
                Goldilocks3::op_31_pack(nrowsPack, 2, &buffHelper[0], &destVals[0], isConstantA, &destVals[FIELD_EXTENSION*nrowsPack], isConstantB);
            } else {
                Goldilocks3::op_31_pack(nrowsPack, 2, &buffHelper[0], &destVals[FIELD_EXTENSION*nrowsPack], isConstantB, &destVals[0], isConstantA);
            }
            Goldilocks::copy_pack(nrowsPack, &destVals[0], &buffHelper[0]);
            Goldilocks::copy_pack(nrowsPack, &destVals[nrowsPack], &buffHelper[nrowsPack]);
            Goldilocks::copy_pack(nrowsPack, &destVals[2*nrowsPack], &buffHelper[2*nrowsPack]);
        }
    }

    inline void storePolynomial(uint64_t nrowsPack, Dest &dest, Goldilocks::Element* destVals, uint64_t row, uint64_t isConstant) {
        if(dest.dim == 1) {
            uint64_t offset = dest.offset != 0 ? dest.offset : 1;
            Goldilocks::copy_pack(nrowsPack, &dest.dest[row*offset], uint64_t(offset), &destVals[0], isConstant);
        } else {
            uint64_t offset = dest.offset != 0 ? dest.offset : FIELD_EXTENSION;
            Goldilocks::copy_pack(nrowsPack, &dest.dest[row*offset], uint64_t(offset), &destVals[0], isConstant);
            Goldilocks::copy_pack(nrowsPack, &dest.dest[row*offset + 1], uint64_t(offset), &destVals[nrowsPack], isConstant);
            Goldilocks::copy_pack(nrowsPack, &dest.dest[row*offset + 2], uint64_t(offset), &destVals[2*nrowsPack], isConstant);
        }
    }

    inline void printTmp1(uint64_t nrowsPack, uint64_t row, Goldilocks::Element* tmp, bool isConstant) {
        Goldilocks::Element buff[nrowsPack];
        Goldilocks::copy_pack(nrowsPack, buff, tmp);
        for(uint64_t i = 0; i < nrowsPack; ++i) {
            if(isConstant) {
                cout << "Value at row " << row + i << " is " << Goldilocks::toString(buff[0]) << endl;
            } else {
                cout << "Value at row " << row + i << " is " << Goldilocks::toString(buff[i]) << endl;
            }
        }
    }

    inline void printTmp3(uint64_t nrowsPack, uint64_t row, Goldilocks::Element* tmp, bool isConstant) {
        for(uint64_t i = 0; i < nrowsPack; ++i) {
            if(isConstant) {
                cout << "Value at row " << row + i << " is [" << Goldilocks::toString(tmp[0]) << ", " << Goldilocks::toString(tmp[1]) << ", " << Goldilocks::toString(tmp[2]) << "]" << endl;
            } else {
                cout << "Value at row " << row + i << " is [" << Goldilocks::toString(tmp[i]) << ", " << Goldilocks::toString(tmp[nrowsPack + i]) << ", " << Goldilocks::toString(tmp[2*nrowsPack + i]) << "]" << endl;
            }
        }
    }


    void printArguments(uint64_t nrowsPack, Goldilocks::Element *a, uint32_t dimA, bool constA, Goldilocks::Element *b, uint32_t dimB, bool constB, int i, uint64_t op_type, uint64_t op, uint64_t nOps){
        #if DEBUG
            bool print = i == DEBUG_ROW;
            if(print){
                printf("Expression debug op: %lu of %lu with type %lu\n", op, nOps, op_type);
                if(a!= NULL){
                    for(uint32_t i = 0; i < dimA; i++){
                        Goldilocks::Element val = constA ? a[i] : a[i*nrowsPack];
                        printf("Expression debug a[%d]: %llu (constant %u)\n", i, val.fe % GOLDILOCKS_PRIME, constA);
                    }
                }
                if(b!= NULL){
                    for(uint32_t i = 0; i < dimB; i++){
                        Goldilocks::Element val = constB ? b[i] : b[i*nrowsPack];
                        printf("Expression debug b[%d]: %llu (constant %u)\n", i, val.fe % GOLDILOCKS_PRIME, constB);
                    }
        
                }
            }
        #endif
    }

    void printRes(uint64_t nrowsPack, Goldilocks::Element *res, uint32_t dimRes, int i)
    {
        #if DEBUG
            bool print = i == DEBUG_ROW;
            if(print){
                for(uint32_t i = 0; i < dimRes; i++){
                    printf("Expression debug res[%d]: %llu\n", i, res[i*nrowsPack].fe % GOLDILOCKS_PRIME);
                }
            }
        #endif
    }



    void calculateExpressions(StepsParams& params, Dest &dest, uint64_t domainSize, bool domainExtended, bool compilation_time, bool verify_constraints = false) override {
        uint64_t nrowsPack = std::min(nrowsPack_, domainSize);

        uint64_t *mapOffsetsExps = domainExtended ? mapOffsetsExtended : mapOffsets;
        uint64_t *mapOffsetsCustomExps = domainExtended ? mapOffsetsCustomFixedExtended : mapOffsetsCustomFixed;
        int64_t *nextStridesExps = domainExtended ? nextStridesExtended : nextStrides;

        uint64_t k_min = domainExtended 
            ? uint64_t((minRowExtended + nrowsPack - 1) / nrowsPack) * nrowsPack
            : uint64_t((minRow + nrowsPack - 1) / nrowsPack) * nrowsPack;
        uint64_t k_max = domainExtended
            ? uint64_t(maxRowExtended / nrowsPack)*nrowsPack
            : uint64_t(maxRow / nrowsPack)*nrowsPack;


        ParserArgs parserArgs = verify_constraints ? setupCtx.expressionsBin.expressionsBinArgsConstraints : setupCtx.expressionsBin.expressionsBinArgsExpressions;
        ParserParams parserParams[dest.params.size()];

        uint64_t maxTemp1Size = 0;
        uint64_t maxTemp3Size = 0;

        assert(dest.params.size() == 1 || dest.params.size() == 2);

        for (uint64_t k = 0; k < dest.params.size(); ++k) {
            if(dest.params[k].op != opType::tmp) continue;
            parserParams[k] = verify_constraints 
                ? setupCtx.expressionsBin.constraintsInfoDebug[dest.params[k].expId]
                : setupCtx.expressionsBin.expressionsInfo[dest.params[k].expId];
            if (parserParams[k].nTemp1*nrowsPack > maxTemp1Size) {
                maxTemp1Size = parserParams[k].nTemp1*nrowsPack;
            }
            if (parserParams[k].nTemp3*nrowsPack*FIELD_EXTENSION > maxTemp3Size) {
                maxTemp3Size = parserParams[k].nTemp3*nrowsPack*FIELD_EXTENSION;
            }
        }
        
        Goldilocks::Element *tmp1_ = &params.aux_trace[setupCtx.starkInfo.mapOffsets[std::make_pair("tmp1", false)]];
        Goldilocks::Element *tmp3_ = &params.aux_trace[setupCtx.starkInfo.mapOffsets[std::make_pair("tmp3", false)]];
        Goldilocks::Element *values_ = &params.aux_trace[setupCtx.starkInfo.mapOffsets[std::make_pair("values", false)]];
    #pragma omp parallel for
        for (uint64_t i = 0; i < domainSize; i+= nrowsPack) {
            bool isCyclic = i < k_min || i >= k_max;
            uint64_t expressions_params_size = bufferCommitsSize + 9;
            Goldilocks::Element* expressions_params[expressions_params_size];
            expressions_params[bufferCommitsSize + 2] = params.publicInputs;
            expressions_params[bufferCommitsSize + 3] = parserArgs.numbers;
            expressions_params[bufferCommitsSize + 4] = params.airValues;
            expressions_params[bufferCommitsSize + 5] = params.proofValues;
            expressions_params[bufferCommitsSize + 6] = params.airgroupValues;
            expressions_params[bufferCommitsSize + 7] = params.challenges;
            expressions_params[bufferCommitsSize + 8] = params.evals;

            Goldilocks::Element *values = &values_[omp_get_thread_num()*3*FIELD_EXTENSION*nrowsPack];
            for(uint64_t k = 0; k < dest.params.size(); ++k) {
                uint64_t i_args = 0;
                if(dest.params[k].op == opType::cm || dest.params[k].op == opType::const_) {
                    uint64_t openingPointIndex = dest.params[k].rowOffsetIndex;
                    uint64_t stagePos = dest.params[k].stagePos;
                    int64_t o = nextStridesExps[openingPointIndex];
                    uint64_t nCols = mapSectionsN[0];
                    if (dest.params[k].op == opType::const_) {
                        for(uint64_t r = 0; r < nrowsPack; ++r) {
                            uint64_t l = (i + r + o) % domainSize;
                            values[k*FIELD_EXTENSION*nrowsPack + r] = params.pConstPolsAddress[l * nCols + stagePos];
                        }
                    } else {
                        uint64_t offset = mapOffsetsExps[dest.params[k].stage];
                        uint64_t nCols = mapSectionsN[dest.params[k].stage];
                        for(uint64_t r = 0; r < nrowsPack; ++r) {
                            uint64_t l = (i + r + o) % domainSize;
#if DEBUG
                            if(i== DEBUG_ROW && r==0) printf("Expression debug trace\n");
#endif
                            if(dest.params[k].stage == 1) {
                                values[k*FIELD_EXTENSION*nrowsPack + r] = params.trace[l * nCols + stagePos];
                            } else {
#if DEBUG
                                if(i== DEBUG_ROW && r==0 ) printf("Expression debug aux_trace\n");
#endif                               
                                for(uint64_t d = 0; d < dest.params[k].dim; ++d) {
                                    values[k*FIELD_EXTENSION*nrowsPack + r + d*nrowsPack] = params.aux_trace[offset + l * nCols + stagePos + d];
                                }
                            }
                        }
                    }

                    if(dest.params[k].inverse) {
                        getInversePolinomial(nrowsPack, &values[k*FIELD_EXTENSION*nrowsPack], &values[2*FIELD_EXTENSION*nrowsPack], dest.params[k].batch,dest.params[k].dim);
                    }
                    continue;
                } else if(dest.params[k].op == opType::number) {
#if DEBUG
                    if(i== DEBUG_ROW) printf("Expression debug number\n");
#endif
                    values[k*FIELD_EXTENSION*nrowsPack] = Goldilocks::fromU64(dest.params[k].value);
                    continue;
                } else if(dest.params[k].op == opType::airvalue) {
                    if(dest.params[k].dim == 1) {
                        values[k*FIELD_EXTENSION*nrowsPack] = params.airValues[dest.params[k].polsMapId];
                        continue;
                    } else {
                        values[k*FIELD_EXTENSION*nrowsPack] = params.airValues[dest.params[k].polsMapId];
                        values[k*FIELD_EXTENSION*nrowsPack + nrowsPack] = params.airValues[dest.params[k].polsMapId + 1];
                        values[k*FIELD_EXTENSION*nrowsPack + 2*nrowsPack] = params.airValues[dest.params[k].polsMapId + 2];
                    }
                    continue;
                }
                uint8_t* ops = &parserArgs.ops[parserParams[k].opsOffset];
                uint16_t* args = &parserArgs.args[parserParams[k].argsOffset];
                expressions_params[bufferCommitsSize] = &tmp1_[omp_get_thread_num()*maxTemp1Size];
                expressions_params[bufferCommitsSize + 1] = &tmp3_[omp_get_thread_num()*maxTemp3Size];

                Goldilocks::Element *valueA = &values[FIELD_EXTENSION*nrowsPack];
                Goldilocks::Element *valueB = &values[2*FIELD_EXTENSION*nrowsPack];

                for (uint64_t kk = 0; kk < parserParams[k].nOps; ++kk) {
                    // if(i == 0) cout << kk << "of " << parserParams[k].nOps << " is " << uint64_t(ops[kk]) << endl;
                    switch (ops[kk]) {
                        case 0: {
                            // COPY dim1 to dim1
                            Goldilocks::Element* a = load(nrowsPack, valueA, params, expressions_params, args, mapOffsetsExps, mapOffsetsCustomExps, nextStridesExps, i_args + 1, i, 1, domainSize, domainExtended, isCyclic);
                            bool isConstant = args[i_args + 1] > bufferCommitsSize + 1 ? true : false;
                            Goldilocks::Element* res = kk == parserParams[k].nOps - 1 ? &values[k*FIELD_EXTENSION*nrowsPack] : &expressions_params[bufferCommitsSize][args[i_args] * nrowsPack];
                            printArguments(nrowsPack, a, 1, isConstant,  NULL, 0, true, i, 4, kk, parserParams[k].nOps);
                            Goldilocks::copy_pack(nrowsPack, res, a, isConstant);
                            printRes(nrowsPack, res, 1, i);
                            i_args += 4;
                            break;
                        }
                        case 1: {
                            // OPERATION WITH DEST: dim1 - SRC0: dim1 - SRC1: dim1
                            Goldilocks::Element* a = load(nrowsPack, valueA, params, expressions_params, args, mapOffsetsExps, mapOffsetsCustomExps, nextStridesExps, i_args + 2, i, 1, domainSize, domainExtended, isCyclic);
                            Goldilocks::Element* b = load(nrowsPack, valueB, params, expressions_params, args, mapOffsetsExps, mapOffsetsCustomExps, nextStridesExps, i_args + 5, i, 1, domainSize, domainExtended, isCyclic);
                            bool isConstantA = args[i_args + 2] > bufferCommitsSize + 1 ? true : false;
                            bool isConstantB = args[i_args + 5] > bufferCommitsSize + 1 ? true : false;
                            Goldilocks::Element* res = kk == parserParams[k].nOps - 1 ? &values[k*FIELD_EXTENSION*nrowsPack] : &expressions_params[bufferCommitsSize][args[i_args + 1] * nrowsPack];
                            // if(i == 0) printTmp1(nrowsPack, i, a, isConstantA);
                            // if(i == 0) printTmp1(nrowsPack, i, b, isConstantB);
                            printArguments(nrowsPack, a, 1, isConstantA, b, 1, isConstantB, i, args[i_args], kk, parserParams[k].nOps);
                            Goldilocks::op_pack(nrowsPack, args[i_args], res, a, isConstantA, b, isConstantB);
                            printRes(nrowsPack, res, 1,i);
                            // if(i == 0) printTmp1(nrowsPack, i, res, false);
                            i_args += 8;
                            break;
                        }
                        case 2: {
                            // OPERATION WITH DEST: dim3 - SRC0: dim3 - SRC1: dim1
                            Goldilocks::Element* a = load(nrowsPack, valueA, params, expressions_params, args, mapOffsetsExps, mapOffsetsCustomExps, nextStridesExps, i_args + 2, i, 3, domainSize, domainExtended, isCyclic);
                            Goldilocks::Element* b = load(nrowsPack, valueB, params, expressions_params, args, mapOffsetsExps, mapOffsetsCustomExps, nextStridesExps, i_args + 5, i, 1, domainSize, domainExtended, isCyclic);
                            bool isConstantA = args[i_args + 2] > bufferCommitsSize + 1 ? true : false;
                            bool isConstantB = args[i_args + 5] > bufferCommitsSize + 1 ? true : false;
                            Goldilocks::Element *res = kk == parserParams[k].nOps - 1 ? &values[k*FIELD_EXTENSION*nrowsPack] : &expressions_params[bufferCommitsSize + 1][args[i_args + 1] * nrowsPack];
                            // if(i == 0) printTmp3(nrowsPack, i, a, isConstantA);
                            // if(i == 0) printTmp1(nrowsPack, i, b, isConstantB);
                            printArguments(nrowsPack, a, 3, isConstantA, b, 1, isConstantB, i, args[i_args], kk, parserParams[k].nOps);
                            Goldilocks3::op_31_pack(nrowsPack, args[i_args], res, a, isConstantA, b, isConstantB);
                            printRes(nrowsPack, res, 3, i);
                            // if(i == 0) printTmp3(nrowsPack, i, res, false);
                            i_args += 8;
                            break;
                        }
                        case 3: {
                            // OPERATION WITH DEST: dim3 - SRC0: dim3 - SRC1: dim3
                            Goldilocks::Element* a = load(nrowsPack, valueA, params, expressions_params, args, mapOffsetsExps, mapOffsetsCustomExps, nextStridesExps, i_args + 2, i, 3, domainSize, domainExtended, isCyclic);
                            Goldilocks::Element* b = load(nrowsPack, valueB, params, expressions_params, args, mapOffsetsExps, mapOffsetsCustomExps, nextStridesExps, i_args + 5, i, 3, domainSize, domainExtended, isCyclic);
                            bool isConstantA = args[i_args + 2] > bufferCommitsSize + 1 ? true : false;
                            bool isConstantB = args[i_args + 5] > bufferCommitsSize + 1 ? true : false;
                            Goldilocks::Element *res = kk == parserParams[k].nOps - 1 ? &values[k*FIELD_EXTENSION*nrowsPack] : &expressions_params[bufferCommitsSize + 1][args[i_args + 1] * nrowsPack];
                            // if(i == 0) printTmp3(nrowsPack, i, a, isConstantA);
                            // if(i == 0) printTmp3(nrowsPack, i, b, isConstantB);
                            printArguments(nrowsPack, a, 3, isConstantA, b, 3, isConstantB, i, args[i_args], kk, parserParams[k].nOps);
                            Goldilocks3::op_pack(nrowsPack, args[i_args], res, a, isConstantA, b, isConstantB);
                            printRes(nrowsPack, res, 3, i);
                            // if(i == 0) printTmp3(nrowsPack, i, res, false);
                            i_args += 8;
                            break;
                        }
                        case 4: {
                            // COPY dim3 to dim3
                            Goldilocks::Element* a = load(nrowsPack, valueA, params, expressions_params, args, mapOffsetsExps, mapOffsetsCustomExps, nextStridesExps, i_args + 1, i, 3, domainSize, domainExtended, isCyclic);
                            bool isConstant = args[i_args + 1] > bufferCommitsSize + 1 ? true : false;
                            Goldilocks::Element *res = kk == parserParams[k].nOps - 1 ? &values[k*FIELD_EXTENSION*nrowsPack] : &expressions_params[bufferCommitsSize + 1][args[i_args] * nrowsPack];
                            // if(i == 0) printTmp3(nrowsPack, i, a, isConstant);
                            printArguments(nrowsPack, a, 3, isConstant, NULL, 0, true, i, 4, kk, parserParams[k].nOps);
                            printRes(nrowsPack, res, 3, i);
                            Goldilocks3::copy_pack(nrowsPack, res, a, isConstant);
                            i_args += 4;
                            break;
                        }
                        default: {
                            std::cout << " Wrong operation!" << std::endl;
                            exit(1);
                        }
                    }
                }

                if (i_args != parserParams[k].nArgs) std::cout << " " << i_args << " - " << parserParams[k].nArgs << std::endl;
                assert(i_args == parserParams[k].nArgs);
                
                if(dest.params[k].inverse) {
                    getInversePolinomial(nrowsPack, &values[k*FIELD_EXTENSION*nrowsPack], &values[2*FIELD_EXTENSION*nrowsPack], dest.params[k].batch, parserParams[k].destDim);
                }
                
            }
            bool isConstant = false;

            if(dest.params.size() == 2) {
                bool isConstantA = dest.params[0].op == opType::number || dest.params[0].op == opType::airvalue;
                bool isConstantB = dest.params[1].op == opType::number || dest.params[1].op == opType::airvalue;
                isConstant = isConstantA && isConstantB;
                multiplyPolynomials(nrowsPack, dest, values, &values[2*FIELD_EXTENSION*nrowsPack], isConstantA, isConstantB);
            } else {
                isConstant = dest.params[0].op == opType::number || dest.params[0].op == opType::airvalue;
            }
            
            storePolynomial(nrowsPack, dest, values, i, isConstant);
        }
    }
};

#endif