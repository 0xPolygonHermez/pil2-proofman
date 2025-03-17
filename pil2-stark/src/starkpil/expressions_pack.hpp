#ifndef EXPRESSIONS_PACK_HPP
#define EXPRESSIONS_PACK_HPP
#include "expressions_ctx.hpp"

class ExpressionsPack : public ExpressionsCtx {
public:
    uint64_t nrowsPack;
    ExpressionsPack(SetupCtx& setupCtx, ProverHelpers& proverHelpers, uint64_t nrowsPack_ = 4) : ExpressionsCtx(setupCtx, proverHelpers), nrowsPack(nrowsPack_) {};

    inline void load(Goldilocks::Element *value, StepsParams& params, Goldilocks::Element** expressions_params, uint16_t* args, uint64_t i_args, uint64_t row, uint64_t dim, uint64_t domainSize, bool domainExtended) {
        int64_t extend = domainExtended ? (1 << (setupCtx.starkInfo.starkStruct.nBitsExt - setupCtx.starkInfo.starkStruct.nBits)) : 1;
        
        uint64_t type = args[i_args];
        if (type == 0) {
            if(dim == FIELD_EXTENSION) { exit(-1); }
            Goldilocks::Element *constPols = domainExtended ? &params.pConstPolsExtendedTreeAddress[2] : params.pConstPolsAddress;
            uint64_t stagePos = args[i_args + 1];
            uint64_t o = setupCtx.starkInfo.verify ? 0 : (setupCtx.starkInfo.openingPoints[args[i_args + 2]] < 0 ? setupCtx.starkInfo.openingPoints[args[i_args + 2]] + domainSize : setupCtx.starkInfo.openingPoints[args[i_args + 2]]) * extend;
            for(uint64_t j = 0; j < nrowsPack; ++j) {
                uint64_t l = (row + j + o) % domainSize;
                value[j] = constPols[l * setupCtx.starkInfo.nConstants + stagePos];
            }
        } else if (type <= setupCtx.starkInfo.nStages + 1) {
            std::string section = "cm" + to_string(type);
            uint64_t stagePos = args[i_args + 1];
            uint64_t offset = setupCtx.starkInfo.mapOffsets[std::make_pair(section, domainExtended)];
            uint64_t nCols = setupCtx.starkInfo.mapSectionsN[section];
            uint64_t o = setupCtx.starkInfo.verify ? 0 : (setupCtx.starkInfo.openingPoints[args[i_args + 2]] < 0 ? setupCtx.starkInfo.openingPoints[args[i_args + 2]] + domainSize : setupCtx.starkInfo.openingPoints[args[i_args + 2]]) * extend;
            for(uint64_t j = 0; j < nrowsPack; ++j) {
                uint64_t l = (row + j + o) % domainSize;
                if(type == 1 && !domainExtended) {
                    value[j] = params.trace[l * nCols + stagePos];
                } else {
                    for(uint64_t d = 0; d < dim; ++d) {
                        value[j + d*nrowsPack] = params.aux_trace[offset + l * nCols + stagePos + d];
                    }
                }
            }
        } else if (type == setupCtx.starkInfo.nStages + 2) {
            uint64_t boundary = args[i_args + 1];
            if(boundary == 0) {
                for(uint64_t j = 0; j < nrowsPack; ++j) {
                    if(setupCtx.starkInfo.verify) {
                        for(uint64_t e = 0; e < FIELD_EXTENSION; ++e) {
                            value[j + e*nrowsPack] = proverHelpers.x_n[e];
                        }
                    } else {
                        value[j] = domainExtended ? proverHelpers.x[row + j] : proverHelpers.x_n[row + j];
                    }
                }
            } else {
                for(uint64_t j = 0; j < nrowsPack; ++j) {
                    if(setupCtx.starkInfo.verify) {
                        for(uint64_t e = 0; e < FIELD_EXTENSION; ++e) {
                            value[j + e*nrowsPack] = proverHelpers.zi[(boundary - 1)*FIELD_EXTENSION + e];
                        }
                    } else {
                        value[j] = proverHelpers.zi[row + j + (boundary - 1)*domainSize];
                    }
                    
                }
            }
        } else if (type == setupCtx.starkInfo.nStages + 3) {
            if(dim == 1) { exit(-1); }
            uint64_t o = args[i_args + 1];
            if(!setupCtx.starkInfo.verify) {
                for (uint64_t k = 0; k < nrowsPack; k++) {    
                    Goldilocks3::sub((Goldilocks3::Element &)(value[k * FIELD_EXTENSION]), proverHelpers.x[row + k], (Goldilocks3::Element &)(xis[o * FIELD_EXTENSION]));
                }

                Goldilocks3::batchInverse((Goldilocks3::Element *)value, (Goldilocks3::Element *)value, nrowsPack);

                for (uint64_t k = 0; k < nrowsPack; k++) {                
                    Goldilocks3::mul((Goldilocks3::Element &)(value[k * FIELD_EXTENSION]), (Goldilocks3::Element &)(value[k * FIELD_EXTENSION]), proverHelpers.x[row + k]);
                }
            } else {
                for(uint64_t k = 0; k < nrowsPack; ++k) {
                    for(uint64_t e = 0; e < FIELD_EXTENSION; ++e) {
                        value[k + e*nrowsPack] = params.xDivXSub[((row + k)*setupCtx.starkInfo.openingPoints.size() + o)*FIELD_EXTENSION + e];
                    }
                }                            
            }
            

        } else if (type >= setupCtx.starkInfo.nStages + 4 && type < setupCtx.starkInfo.customCommits.size() + setupCtx.starkInfo.nStages + 4) {
            uint64_t index = type - (setupCtx.starkInfo.nStages + 4);
            std::string section = setupCtx.starkInfo.customCommits[index].name + "0";
            uint64_t stagePos = args[i_args + 1];
            uint64_t offset = setupCtx.starkInfo.mapOffsets[std::make_pair(section, domainExtended)];
            uint64_t nCols = setupCtx.starkInfo.mapSectionsN[section];
            uint64_t o = setupCtx.starkInfo.verify ? 0 : (setupCtx.starkInfo.openingPoints[args[i_args + 2]] < 0 ? setupCtx.starkInfo.openingPoints[args[i_args + 2]] + domainSize : setupCtx.starkInfo.openingPoints[args[i_args + 2]]) * extend;
            for(uint64_t j = 0; j < nrowsPack; ++j) {
                uint64_t l = (row + j + o) % domainSize;
                value[j] = params.pCustomCommitsFixed[offset + l * nCols + stagePos];
            }
        } else if (type == setupCtx.starkInfo.customCommits.size() + setupCtx.starkInfo.nStages + 4 || type == setupCtx.starkInfo.customCommits.size() + setupCtx.starkInfo.nStages + 4 + 1) {
            memcpy(value, &expressions_params[type][args[i_args + 1]*nrowsPack], dim * nrowsPack * sizeof(Goldilocks::Element));
        } else {
            for(uint64_t k = 0; k < nrowsPack; ++k) {
                for(uint64_t d = 0; d < dim; ++d) {
                    value[k + d*nrowsPack] = expressions_params[type][args[i_args + 1] + d];
                }
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

    void calculateExpressions(StepsParams& params, ParserArgs &parserArgs, std::vector<Dest> dests, uint64_t domainSize, bool compilation_time) override {
        bool domainExtended = !setupCtx.starkInfo.verify && domainSize == uint64_t(1 << setupCtx.starkInfo.starkStruct.nBitsExt) ? true : false;

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
        Goldilocks::Element *values_ = new Goldilocks::Element[omp_get_max_threads() * 2 * FIELD_EXTENSION * nrowsPack];
    #pragma omp parallel for
        for (uint64_t i = 0; i < domainSize; i+= nrowsPack) {
            uint64_t buffer_types_size = 1 + setupCtx.starkInfo.nStages + 3 + setupCtx.starkInfo.customCommits.size();
            uint64_t expressions_params_size = buffer_types_size + 9;
            Goldilocks::Element* expressions_params[expressions_params_size];
            expressions_params[buffer_types_size + 2] = params.publicInputs;
            expressions_params[buffer_types_size + 3] = parserArgs.numbers;
            expressions_params[buffer_types_size + 4] = params.airValues;
            expressions_params[buffer_types_size + 5] = params.proofValues;
            expressions_params[buffer_types_size + 6] = params.airgroupValues;
            expressions_params[buffer_types_size + 7] = params.challenges;
            expressions_params[buffer_types_size + 8] = params.evals;

            Goldilocks::Element **destVals = new Goldilocks::Element*[dests.size()];

            for(uint64_t j = 0; j < dests.size(); ++j) {
                if(i >= dests[j].domainSize) continue;
                destVals[j] = new Goldilocks::Element[dests[j].params.size() * FIELD_EXTENSION* nrowsPack];
                for(uint64_t k = 0; k < dests[j].params.size(); ++k) {
                    uint64_t i_args = 0;

                    if(dests[j].params[k].op == opType::cm || dests[j].params[k].op == opType::const_) {
                        uint64_t openingPointIndex = dests[j].params[k].rowOffsetIndex;
                        uint64_t stagePos = dests[j].params[k].stagePos;
                        uint64_t o = setupCtx.starkInfo.verify ? 0 : (setupCtx.starkInfo.openingPoints[openingPointIndex] < 0 ? setupCtx.starkInfo.openingPoints[openingPointIndex] + domainSize : setupCtx.starkInfo.openingPoints[openingPointIndex]);
                        Goldilocks::Element *values = &values_[omp_get_thread_num()*2*FIELD_EXTENSION*nrowsPack];
                        if (dests[j].params[k].op == opType::const_) {
                            for(uint64_t r = 0; r < nrowsPack; ++r) {
                                uint64_t l = (i + r + o) % domainSize;
                                values[r] = params.pConstPolsAddress[l * setupCtx.starkInfo.nConstants + stagePos];
                            }
                        } else {
                            std::string section = "cm" + to_string(dests[j].params[k].stage);
                            uint64_t offset = setupCtx.starkInfo.mapOffsets[std::make_pair(section, false)];
                            uint64_t nCols = setupCtx.starkInfo.mapSectionsN[section];
                            for(uint64_t r = 0; r < nrowsPack; ++r) {
                                uint64_t l = (i + r + o) % domainSize;
                                if(dests[j].params[k].stage == 1) {
                                    values[r] = params.trace[l * nCols + stagePos];
                                } else {
                                    for(uint64_t d = 0; d < dests[j].params[k].dim; ++d) {
                                        values[r + d*nrowsPack] = params.aux_trace[offset + l * nCols + stagePos + d];
                                    }
                                }
                            }
                        }
                        copyPolynomial(&destVals[j][k*FIELD_EXTENSION*nrowsPack], dests[j].params[k].inverse,dests[j].params[k].batch,dests[j].params[k].dim, values);
                        continue;
                    } else if(dests[j].params[k].op == opType::number) {
                        for(uint64_t r = 0; r < nrowsPack; ++r) {
                            destVals[j][k*FIELD_EXTENSION*nrowsPack + r] = Goldilocks::fromU64(dests[j].params[k].value);
                        }
                        continue;
                    } else if(dests[j].params[k].op == opType::airvalue) {
                        for(uint64_t r = 0; r < nrowsPack; ++r) {
                            destVals[j][k*FIELD_EXTENSION*nrowsPack + r] = params.airValues[dests[j].params[k].polsMapId];
                            destVals[j][k*FIELD_EXTENSION*nrowsPack + r + 1] = params.airValues[dests[j].params[k].polsMapId + 1];
                            destVals[j][k*FIELD_EXTENSION*nrowsPack + r + 2] = params.airValues[dests[j].params[k].polsMapId + 2];
                        }
                        continue;
                    }
                    uint8_t* ops = &parserArgs.ops[dests[j].params[k].parserParams.opsOffset];
                    uint16_t* args = &parserArgs.args[dests[j].params[k].parserParams.argsOffset];
                    expressions_params[buffer_types_size] = &tmp1_[omp_get_thread_num()*maxTemp1Size];
                    expressions_params[buffer_types_size + 1] = &tmp3_[omp_get_thread_num()*maxTemp3Size];

                    Goldilocks::Element *a = &values_[omp_get_thread_num()*2*FIELD_EXTENSION*nrowsPack];
                    Goldilocks::Element *b = &values_[omp_get_thread_num()*2*FIELD_EXTENSION*nrowsPack + nrowsPack*FIELD_EXTENSION];

                    for (uint64_t kk = 0; kk < dests[j].params[k].parserParams.nOps; ++kk) {
                        switch (ops[kk]) {
                            case 0: {
                                // COPY dim1 to dim1
                                load(a, params, expressions_params, args, i_args + 1, i, 1, domainSize, domainExtended);
                                Goldilocks::copy_pack(nrowsPack, &expressions_params[buffer_types_size][args[i_args] * nrowsPack], a);
                                i_args += 4;
                                break;
                            }
                            case 1: {
                                // OPERATION WITH DEST: dim1 - SRC0: dim1 - SRC1: dim1
                                load(a, params, expressions_params, args, i_args + 2, i, 1, domainSize, domainExtended);
                                load(b, params, expressions_params, args, i_args + 5, i, 1, domainSize, domainExtended);
                                Goldilocks::op_pack(nrowsPack, args[i_args], &expressions_params[buffer_types_size][args[i_args + 1] * nrowsPack], a, b);
                                i_args += 8;
                                break;
                            }
                            case 2: {
                                // OPERATION WITH DEST: dim3 - SRC0: dim3 - SRC1: dim1
                                load(a, params, expressions_params, args, i_args + 2, i, 3, domainSize, domainExtended);
                                load(b, params, expressions_params, args, i_args + 5, i, 1, domainSize, domainExtended);
                                Goldilocks3::op_31_pack(nrowsPack, args[i_args], &expressions_params[buffer_types_size + 1][args[i_args + 1] * nrowsPack], a, b);
                                i_args += 8;
                                break;
                            }
                            case 3: {
                                // OPERATION WITH DEST: dim3 - SRC0: dim3 - SRC1: dim3
                                load(a, params, expressions_params, args, i_args + 2, i, 3, domainSize, domainExtended);
                                load(b, params, expressions_params, args, i_args + 5, i, 3, domainSize, domainExtended);
                                Goldilocks3::op_pack(nrowsPack, args[i_args], &expressions_params[buffer_types_size + 1][args[i_args + 1] * nrowsPack], a, b);
                                i_args += 8;
                                break;
                            }
                            case 4: {
                                // COPY dim3 to dim3
                                load(a, params, expressions_params, args, i_args + 1, i, 3, domainSize, domainExtended);
                                Goldilocks3::copy_pack(nrowsPack, &expressions_params[buffer_types_size + 1][args[i_args] * nrowsPack], a);
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
                        copyPolynomial(&destVals[j][k*FIELD_EXTENSION*nrowsPack], dests[j].params[k].inverse,dests[j].params[k].batch, dests[j].params[k].parserParams.destDim, &expressions_params[buffer_types_size][dests[j].params[k].parserParams.destId*nrowsPack]);
                    } else {
                        copyPolynomial(&destVals[j][k*FIELD_EXTENSION*nrowsPack], dests[j].params[k].inverse,dests[j].params[k].batch, dests[j].params[k].parserParams.destDim, &expressions_params[buffer_types_size + 1][dests[j].params[k].parserParams.destId*FIELD_EXTENSION*nrowsPack]);
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
        delete[] values_;
        delete[] tmp1_;
        delete[] tmp3_;
    }
};

#endif