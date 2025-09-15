#ifndef EXPRESSIONS_PACK_Q_HPP
#define EXPRESSIONS_PACK_Q_HPP
#include "expressions_ctx.hpp"

#define DEBUG 0
#define DEBUG_ROW 0

#define NROWS_PACK 128
class ExpressionsPackQ : public ExpressionsCtx {
public:
    ExpressionsPackQ(SetupCtx& setupCtx, ProverHelpers* proverHelpers, uint64_t nrowsPack = NROWS_PACK) : ExpressionsCtx(setupCtx, proverHelpers) {
        nrowsPack_ = std::min(nrowsPack, uint64_t(1 << setupCtx.starkInfo.starkStruct.nBits));
    };

    inline Goldilocks::Element* load(uint64_t nrowsPack, Goldilocks::Element *value, StepsParams& params, Goldilocks::Element** expressions_params, uint16_t* args, uint64_t *mapOffsetsExps, uint64_t* mapOffsetsCustomExps, int64_t* nextStridesExps, uint64_t i_args, uint64_t row, uint64_t dim, uint64_t domainSize, bool domainExtended, bool isCyclic, bool debug) {
        
#if DEBUG 
        bool print = debug && (DEBUG_ROW >= row && DEBUG_ROW < row + nrowsPack);
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

    
    inline void accumulate(uint64_t nrowsPack, uint32_t dim, Goldilocks::Element* acumulator, Goldilocks::Element * tmp, Goldilocks3::Element& challenge, uint32_t constraint_id, uint32_t nConstraints, bool print_factor = false) {
        Goldilocks3::Element factor = {Goldilocks::one(), Goldilocks::zero(), Goldilocks::zero()};
        for(uint32_t i = 0; i < nConstraints-(constraint_id+1); ++i) {
            Goldilocks3::mul(factor, factor, challenge);
        }
        if(print_factor) {
            std::cout << "Factor for constraint " << constraint_id << " is " << factor[0].fe << "  " << factor[1].fe << " " << factor[2].fe << " " << std::endl;
        }
        if(dim == 1) {
            for (uint64_t i = 0; i < nrowsPack; ++i)
            {
                Goldilocks3::Element aux_tmp;
                Goldilocks3::Element aux_acc={acumulator[i], acumulator[i+nrowsPack], acumulator[i+2*nrowsPack]};
                Goldilocks3::mul(aux_tmp, factor, (Goldilocks::Element &)tmp[i]);
                Goldilocks3::add(aux_acc, aux_acc, aux_tmp);
                acumulator[i] = aux_acc[0];
                acumulator[i+nrowsPack] = aux_acc[1];
                acumulator[i+2*nrowsPack] = aux_acc[2];
            }
            
        } else {
            for (uint64_t i = 0; i < nrowsPack; ++i)
            {
                Goldilocks3::Element aux_tmp={tmp[i], tmp[i+nrowsPack], tmp[i+2*nrowsPack]};
                Goldilocks3::Element aux_acc={acumulator[i], acumulator[i+nrowsPack], acumulator[i+2*nrowsPack]};
                Goldilocks3::mul(aux_tmp, factor, aux_tmp);
                Goldilocks3::add(aux_acc, aux_acc, aux_tmp);
                acumulator[i] = aux_acc[0];
                acumulator[i+nrowsPack] = aux_acc[1];
                acumulator[i+2*nrowsPack] = aux_acc[2];
            }
        }
    }

    inline void ziAndStorePolynomial(uint64_t nrowsPack, Dest &dest, Goldilocks::Element* accumulator, uint64_t row) {
        
        /*for (uint64_t i = 0; i < nrowsPack; ++i)
        {
            Goldilocks3::Element acc_challenge = {accumulator[i], accumulator[i+nrowsPack], accumulator[i+2*nrowsPack]};
            Goldilocks3::mul((Goldilocks3::Element &)dest.dest[(row+i)*FIELD_EXTENSION], (Goldilocks3::Element &)accumulator[i*FIELD_EXTENSION], proverHelpers->zi[row+i]);
        }*/
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


    void printArguments(uint64_t nrowsPack, Goldilocks::Element *a, uint32_t dimA, bool constA, Goldilocks::Element *b, uint32_t dimB, bool constB, int i, uint64_t op_type, uint64_t op, uint64_t nOps, bool debug){
        #if DEBUG
            bool print = debug && (DEBUG_ROW >= i && DEBUG_ROW < i + nrowsPack);
            if(print){
                printf("Expression debug op: %lu of %lu with type %lu\n", op, nOps, op_type);
                if(a != NULL){
                    for(uint32_t j = 0; j < dimA; j++){
                        Goldilocks::Element val = constA ? a[j] : a[j*nrowsPack + DEBUG_ROW % nrowsPack];
                        printf("Expression debug a[%d]: %llu (constant %u)\n", j, val.fe % GOLDILOCKS_PRIME, constA);
                    }
                }
                if(b!= NULL){
                    for(uint32_t j = 0; j < dimB; j++){
                        Goldilocks::Element val = constB ? b[j] : b[j*nrowsPack + DEBUG_ROW % nrowsPack];
                        printf("Expression debug b[%d]: %llu (constant %u)\n", j, val.fe % GOLDILOCKS_PRIME, constB);
                    }
        
                }
            }
        #endif
    }

    void printRes(uint64_t nrowsPack, Goldilocks::Element *res, uint32_t dimRes, int i, bool debug)
    {
        #if DEBUG
            bool print = debug && (DEBUG_ROW >= i && DEBUG_ROW < i + nrowsPack);
            if(print){
                for(uint32_t j = 0; j < dimRes; j++){
                    printf("Expression debug res[%d]: %llu\n", j, res[j*nrowsPack + DEBUG_ROW % nrowsPack].fe % GOLDILOCKS_PRIME);
                }
            }
        #endif
    }

   void calculateExpressions(StepsParams& params, Dest &dest, uint64_t domainSize, bool domainExtended, bool compilation_time, bool verify_constraints = false, bool debug = false) override {

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

        for (uint64_t k = 0; k < dest.params.size(); ++k) {  //aqui anire per tots els params, o simplement un loop segons dimensio no?
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
        
        Goldilocks::Element *tmp1_ = &params.aux_trace[setupCtx.starkInfo.mapOffsets[std::make_pair("tmp1", false)]]; //rick: suficient espai aqui
        Goldilocks::Element *tmp3_ = &params.aux_trace[setupCtx.starkInfo.mapOffsets[std::make_pair("tmp3", false)]]; //rick: suficient espai aqui
        Goldilocks::Element *values_ = &params.aux_trace[setupCtx.starkInfo.mapOffsets[std::make_pair("values", false)]]; //rick: que son aquest values
    //#pragma omp parallel for
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

                memset(values, 0, 3*FIELD_EXTENSION*nrowsPack*sizeof(Goldilocks::Element));
                if(i==0 ){
                    std::cout << "Calculating expression " << k << " of " << dest.params.size() << " with id " << dest.params[k].expId << " with dim " << dest.params[k].dim <<" nops "<< parserParams[k].nOps <<" nargs: "<<parserParams[k].nArgs<<std::endl;
                }
                
                //Goldilocks::Element *accumulator = &values[3*FIELD_EXTENSION*nrowsPack];
                uint64_t i_args = 0;
                uint8_t* ops = &parserArgs.ops[parserParams[k].opsOffset];
                uint16_t* args = &parserArgs.args[parserParams[k].argsOffset];
                expressions_params[bufferCommitsSize] = &tmp1_[omp_get_thread_num()*maxTemp1Size];
                expressions_params[bufferCommitsSize + 1] = &tmp3_[omp_get_thread_num()*maxTemp3Size];

                memset(expressions_params[bufferCommitsSize], 0, maxTemp1Size*sizeof(Goldilocks::Element));
                memset(expressions_params[bufferCommitsSize + 1], 0, maxTemp3Size*sizeof(Goldilocks::Element));
    

                Goldilocks::Element *valueA = &values[FIELD_EXTENSION*nrowsPack];
                Goldilocks::Element *valueB = &values[2*FIELD_EXTENSION*nrowsPack];
                for (uint64_t kk = 0; kk < parserParams[k].nOps; ++kk) {
                    // if(i == 0) cout << kk << "of " << parserParams[k].nOps << " is " << uint64_t(ops[kk]) << endl;
                    switch (ops[kk]) {
                        case 0: {
                            // OPERATION WITH DEST: dim1 - SRC0: dim1 - SRC1: dim1
                            Goldilocks::Element* a = load(nrowsPack, valueA, params, expressions_params, args, mapOffsetsExps, mapOffsetsCustomExps, nextStridesExps, i_args + 2, i, 1, domainSize, domainExtended, isCyclic, debug);
                            Goldilocks::Element* b = load(nrowsPack, valueB, params, expressions_params, args, mapOffsetsExps, mapOffsetsCustomExps, nextStridesExps, i_args + 5, i, 1, domainSize, domainExtended, isCyclic, debug);
                            bool isConstantA = args[i_args + 2] > bufferCommitsSize + 1 ? true : false;
                            bool isConstantB = args[i_args + 5] > bufferCommitsSize + 1 ? true : false;
                            Goldilocks::Element* res = kk == parserParams[k].nOps - 1 ? values : &expressions_params[bufferCommitsSize][args[i_args + 1] * nrowsPack];
                            printArguments(nrowsPack, a, 1, isConstantA, b, 1, isConstantB, i, args[i_args], kk, parserParams[k].nOps, debug);
                            Goldilocks::op_pack(nrowsPack, args[i_args], res, a, isConstantA, b, isConstantB);
                            printRes(nrowsPack, res, 1,i, debug);
                            i_args += 8;
                            break;
                        }
                        case 1: {
                            // OPERATION WITH DEST: dim3 - SRC0: dim3 - SRC1: dim1
                            Goldilocks::Element* a = load(nrowsPack, valueA, params, expressions_params, args, mapOffsetsExps, mapOffsetsCustomExps, nextStridesExps, i_args + 2, i, 3, domainSize, domainExtended, isCyclic, debug);
                            Goldilocks::Element* b = load(nrowsPack, valueB, params, expressions_params, args, mapOffsetsExps, mapOffsetsCustomExps, nextStridesExps, i_args + 5, i, 1, domainSize, domainExtended, isCyclic, debug);
                            bool isConstantA = args[i_args + 2] > bufferCommitsSize + 1 ? true : false;
                            bool isConstantB = args[i_args + 5] > bufferCommitsSize + 1 ? true : false;
                            Goldilocks::Element *res = kk == parserParams[k].nOps - 1 ? values : &expressions_params[bufferCommitsSize + 1][args[i_args + 1] * nrowsPack];
                            printArguments(nrowsPack, a, 3, isConstantA, b, 1, isConstantB, i, args[i_args], kk, parserParams[k].nOps, debug);
                            Goldilocks3::op_31_pack(nrowsPack, args[i_args], res, a, isConstantA, b, isConstantB);
                            printRes(nrowsPack, res, 3, i, debug);
                            i_args += 8;
                            break;
                        }
                        case 2: {
                            // OPERATION WITH DEST: dim3 - SRC0: dim3 - SRC1: dim3
                            Goldilocks::Element* a = load(nrowsPack, valueA, params, expressions_params, args, mapOffsetsExps, mapOffsetsCustomExps, nextStridesExps, i_args + 2, i, 3, domainSize, domainExtended, isCyclic, debug);
                            Goldilocks::Element* b = load(nrowsPack, valueB, params, expressions_params, args, mapOffsetsExps, mapOffsetsCustomExps, nextStridesExps, i_args + 5, i, 3, domainSize, domainExtended, isCyclic, debug);
                            bool isConstantA = args[i_args + 2] > bufferCommitsSize + 1 ? true : false;
                            bool isConstantB = args[i_args + 5] > bufferCommitsSize + 1 ? true : false;
                            Goldilocks::Element *res = kk == parserParams[k].nOps - 1 ? values : &expressions_params[bufferCommitsSize + 1][args[i_args + 1] * nrowsPack];
                            printArguments(nrowsPack, a, 3, isConstantA, b, 3, isConstantB, i, args[i_args], kk, parserParams[k].nOps, debug);
                            Goldilocks3::op_pack(nrowsPack, args[i_args], res, a, isConstantA, b, isConstantB);
                            printRes(nrowsPack, res, 3, i, debug);
                            i_args += 8;
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
                
                if(i==0){
                    //print first 3 values of the constraint
                    std::cout << "First values of the constraint: "<< dest.params[k].expId <<" nops: "<<  parserParams[k].nOps<< std::endl;
                    for(uint64_t j = 0; j < std::min(nrowsPack, uint64_t(3)); ++j) {
                        std::cout << "[" << values[j].fe << ", " << values[j+nrowsPack].fe << ", " << values[j+2*nrowsPack].fe << "]" << std::endl;
                    }
                }
                //accumulate(nrowsPack, dest.params[k].dim, accumulator, values, challenge, k, dest.params.size(), i==0);

            }
            /*if(i==0){
                //print first 3 values of the acumulator
                std::cout << "First values of the acumulator: " << std::endl;
                for(uint64_t j = 0; j < std::min(nrowsPack, uint64_t(3)); ++j) {
                    std::cout << "[" << accumulator[j].fe << ", " << accumulator[j+nrowsPack].fe << ", " << accumulator[j+2*nrowsPack].fe << "]" << std::endl;
                }
            }
            ziAndStorePolynomial(nrowsPack, dest, accumulator, i);*/
        }
    }
};

#endif