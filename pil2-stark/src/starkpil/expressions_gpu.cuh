#ifndef EXPRESSIONS_GPU_HPP
#define EXPRESSIONS_GPU_HPP
#include "expressions_ctx.hpp"
#include "cuda_utils.cuh"
#include "cuda_utils.hpp"
#include "gl64_t.cuh"


struct DestGPU {
    Goldilocks::Element *dest = nullptr;
    uint32_t destDim = 0;
    uint64_t offset = 0;
    uint64_t dim = 1;
    uint32_t nParams;
    Params* params;
};
struct DeviceArguments {
    uint64_t N;
    uint64_t NExtended;
    uint64_t domainSize;
    uint64_t nrowsPack;
    uint64_t nCols;
    uint64_t nOpenings;
    uint64_t ns;
    bool domainExtended;
    uint64_t* nextStrides;
    uint64_t* nColsStages;
    uint64_t* nColsStagesAcc;
    uint64_t* offsetsStages;
    Goldilocks::Element* constPols;
    uint64_t constPolsSize;
    uint64_t* cmPolsInfo;
    uint64_t cmPolsInfoSize;
    Goldilocks::Element* trace;
    Goldilocks::Element* pols;
    uint32_t expType;
    uint64_t boundSize;
    Goldilocks::Element* zi;
    Goldilocks::Element* x_n;
    Goldilocks::Element* x_2ns;
    Goldilocks::Element* xDivXSub;
    //non polnomial arguments
    uint32_t nChallenges;
    Goldilocks::Element* challenges;
    uint32_t nNumbers;
    Goldilocks::Element* numbers;
    uint32_t nPublics;
    Goldilocks::Element* publics;
    uint32_t nEvals;
    Goldilocks::Element* evals;
    uint32_t nAirgroupValues;
    Goldilocks::Element* airgroupValues;
    uint32_t nAirValues;
    Goldilocks::Element* airValues;
    //Dests
    DestGPU* dests;
    uint32_t nDests;
    // Expressions bin
    uint8_t* ops;
    uint32_t nOpsTotal;
    uint16_t* args;
    uint32_t nArgsTotal;
    //buffer
    uint64_t nBlocks; 
    uint64_t singleBufferSize;
    uint64_t bufferTSize;
    Goldilocks::Element * bufferT_;
    //destVals
    Goldilocks::Element*** destVals;

};
void computeExpressions(DeviceArguments* d_deviceArgs, DeviceArguments* deviceArgs);
void copyPolynomial(DeviceArguments* deviceArgs, Goldilocks::Element* destVals, bool inverse, bool batch, uint64_t dim, Goldilocks::Element* tmp);
void multiplyPolynomials(DeviceArguments* deviceArgs, DestGPU &dest, Goldilocks::Element* destVals);
void storePolynomial(DeviceArguments* deviceArgs, DestGPU* dests, uint32_t nDests,  Goldilocks::Element** destVals, uint64_t row);
__global__ void allocateDestVals(DeviceArguments* d_deviceArgs);
__global__ void setDestsParams(DestGPU* d_dests, Params* d_params, uint64_t iDests);
__global__ void loadPolynomials_(DeviceArguments* d_deviceArgs, uint64_t row);


class ExpressionsGPU : public ExpressionsCtx {
public:
    uint64_t nrowsPack;
    uint64_t nCols;
    vector<uint64_t> nColsStages;
    vector<uint64_t> nColsStagesAcc;
    vector<uint64_t> offsetsStages;
    DeviceArguments deviceArgs;
    DeviceArguments* d_deviceArgs;
    
    ExpressionsGPU(SetupCtx& setupCtx, uint64_t nrowsPack_ = 64) : ExpressionsCtx(setupCtx), nrowsPack(nrowsPack_) {};

    void setBufferTInfo(uint64_t domainSize, StepsParams& params, ParserArgs &parserArgs, std::vector<Dest> &dests) {

        bool domainExtended = domainSize == uint64_t(1 << setupCtx.starkInfo.starkStruct.nBitsExt) ? true : false; 
        uint64_t expId = dests[0].params[0].op == opType::tmp ? dests[0].params[0].parserParams.destDim : 0;

        uint64_t nOpenings = setupCtx.starkInfo.openingPoints.size();
        uint64_t ns = 2 + setupCtx.starkInfo.nStages + setupCtx.starkInfo.customCommits.size();
        offsetsStages.resize(ns*nOpenings + 1);
        nColsStages.resize(ns*nOpenings + 1);
        nColsStagesAcc.resize(ns*nOpenings + 1);
        
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

        // fill device arguments
        deviceArgs.N = 1 << setupCtx.starkInfo.starkStruct.nBits;
        deviceArgs.NExtended = 1 << setupCtx.starkInfo.starkStruct.nBitsExt;
        deviceArgs.domainSize = domainSize;
        deviceArgs.nrowsPack = nrowsPack;
        deviceArgs.nCols = nCols;
        deviceArgs.nOpenings = nOpenings;
        deviceArgs.ns = ns;
        deviceArgs.domainExtended = domainExtended;
        uint32_t extendBits = (setupCtx.starkInfo.starkStruct.nBitsExt - setupCtx.starkInfo.starkStruct.nBits);
        int64_t extend = domainExtended ? (1 << extendBits) : 1;
        deviceArgs.nextStrides = new uint64_t[nOpenings];
        for(uint64_t i = 0; i < nOpenings; ++i) {
            uint64_t opening = setupCtx.starkInfo.openingPoints[i] < 0 ? setupCtx.starkInfo.openingPoints[i] + domainSize : setupCtx.starkInfo.openingPoints[i];
            deviceArgs.nextStrides[i] = opening * extend;
        }
        deviceArgs.nColsStages = new uint64_t[nColsStages.size()];
        for(uint64_t i = 0; i < nColsStages.size(); ++i) {
            deviceArgs.nColsStages[i] = nColsStages[i];
        }
        deviceArgs.nColsStagesAcc = new uint64_t[nColsStagesAcc.size()];
        for(uint64_t i = 0; i < nColsStagesAcc.size(); ++i) {
            deviceArgs.nColsStagesAcc[i] = nColsStagesAcc[i];
        }
        deviceArgs.offsetsStages = new uint64_t[offsetsStages.size()];
        for(uint64_t i = 0; i < offsetsStages.size(); ++i) {
            deviceArgs.offsetsStages[i] = offsetsStages[i];
        }
        deviceArgs.constPols = domainExtended ? &params.pConstPolsExtendedTreeAddress[2] : params.pConstPolsAddress;
        deviceArgs.constPolsSize = setupCtx.starkInfo.nConstants;
        deviceArgs.cmPolsInfoSize = setupCtx.starkInfo.cmPolsMap.size();
        deviceArgs.cmPolsInfo = new uint64_t[deviceArgs.cmPolsInfoSize*3];
        for(uint64_t i = 0; i < deviceArgs.cmPolsInfoSize; ++i) {
            deviceArgs.cmPolsInfo[i*3] = setupCtx.starkInfo.cmPolsMap[i].stage;
            deviceArgs.cmPolsInfo[i*3 + 1] = setupCtx.starkInfo.cmPolsMap[i].stagePos;
            deviceArgs.cmPolsInfo[i*3 + 2] = setupCtx.starkInfo.cmPolsMap[i].dim;
        }
        if(dests[0].params[0].parserParams.expId == int64_t(setupCtx.starkInfo.cExpId)) {
            deviceArgs.expType = 0;
        } else if(dests[0].params[0].parserParams.expId == int64_t(setupCtx.starkInfo.friExpId)) {
            deviceArgs.expType = 1;
        } else {
            deviceArgs.expType = 2;
        }

        deviceArgs.boundSize = setupCtx.starkInfo.boundaries.size();
        deviceArgs.zi = setupCtx.proverHelpers.zi;
        deviceArgs.x_n = setupCtx.proverHelpers.x_n;
        deviceArgs.x_2ns = setupCtx.proverHelpers.x_2ns;
        deviceArgs.xDivXSub = params.xDivXSub;
        deviceArgs.trace = params.trace;
        deviceArgs.pols = params.pols;

        //Dests
        deviceArgs.dests = new DestGPU[dests.size()];
        deviceArgs.nDests = dests.size();
        for(uint64_t i = 0; i < dests.size(); ++i) {
            deviceArgs.dests[i].dest = dests[i].dest;
            deviceArgs.dests[i].destDim = dests[i].destDim;
            deviceArgs.dests[i].offset = dests[i].offset;
            deviceArgs.dests[i].dim = dests[i].dim;
            deviceArgs.dests[i].params = new Params[dests[i].params.size()];
            deviceArgs.dests[i].nParams = dests[i].params.size();
            for(uint64_t j = 0; j < dests[i].params.size(); ++j) {
                deviceArgs.dests[i].params[j] = dests[i].params[j];
            }
        }
        //non polnomial arguments
        deviceArgs.nChallenges = setupCtx.starkInfo.challengesMap.size();
        deviceArgs.challenges = params.challenges;
        deviceArgs.nNumbers = parserArgs.nNumbers;
        deviceArgs.numbers = (Goldilocks::Element*) parserArgs.numbers;
        deviceArgs.nPublics = setupCtx.starkInfo.nPublics;
        deviceArgs.publics = params.publicInputs;
        deviceArgs.nEvals = setupCtx.starkInfo.evMap.size();
        deviceArgs.evals = params.evals;
        deviceArgs.nAirgroupValues = setupCtx.starkInfo.airgroupValuesMap.size();
        deviceArgs.airgroupValues = params.airgroupValues;
        deviceArgs.nAirValues = setupCtx.starkInfo.airValuesMap.size();
        deviceArgs.airValues = params.airValues;
        // Expressions bin
        deviceArgs.ops = parserArgs.ops;
        deviceArgs.nOpsTotal = parserArgs.nOpsTotal;
        deviceArgs.args = parserArgs.args;
        deviceArgs.nArgsTotal = parserArgs.nArgsTotal;
        
    }

    inline void loadPolynomials(Goldilocks::Element *bufferT_, uint64_t row) {

        uint64_t nOpenings = deviceArgs.nOpenings;
        uint64_t ns = deviceArgs.ns;
        bool domainExtended = deviceArgs.domainExtended;
        uint64_t domainSize = deviceArgs.domainSize;
        uint64_t nrowsPack = deviceArgs.nrowsPack;
        Goldilocks::Element *constPols = deviceArgs.constPols;
        uint64_t constPolsSize = deviceArgs.constPolsSize;
        uint64_t* nextStrides = deviceArgs.nextStrides;
        uint64_t* nColsStages = deviceArgs.nColsStages;
        uint64_t* nColsStagesAcc = deviceArgs.nColsStagesAcc;
        uint64_t* offsetsStages = deviceArgs.offsetsStages;
        uint64_t cmPolsInfoSize = deviceArgs.cmPolsInfoSize;
        uint64_t* cmPolsInfo = deviceArgs.cmPolsInfo;
        Goldilocks::Element* trace = deviceArgs.trace;
        Goldilocks::Element* pols = deviceArgs.pols;
        Goldilocks::Element* zi = deviceArgs.zi;
        Goldilocks::Element* x_n = deviceArgs.x_n;
        Goldilocks::Element* x_2ns = deviceArgs.x_2ns;
        Goldilocks::Element* xDivXSub = deviceArgs.xDivXSub;

        for(uint64_t k = 0; k < constPolsSize;  ++k) {
            for(uint64_t o = 0; o < nOpenings; ++o) {
                for(uint64_t j = 0; j < nrowsPack; ++j) {
                    uint64_t l = (row + j + nextStrides[o]) % domainSize;
                    bufferT_[(nColsStagesAcc[ns*o] + k)*nrowsPack + j] = constPols[l * nColsStages[0] + k];
                }
            }
        }

        for(uint64_t k = 0; k < cmPolsInfoSize; ++k) {
            uint64_t stage = cmPolsInfo[k*3];
            uint64_t stagePos = cmPolsInfo[k*3 + 1];
            for(uint64_t d = 0; d < cmPolsInfo[k*3+2]; ++d) {
                for(uint64_t o = 0; o < nOpenings; ++o) {
                    for(uint64_t j = 0; j < nrowsPack; ++j) {
                        uint64_t l = (row + j + nextStrides[o]) % domainSize;
                        if(stage == 1 && !domainExtended) {
                            bufferT_[(nColsStagesAcc[ns*o + stage] + (stagePos + d))*nrowsPack + j] = trace[l * nColsStages[stage] + stagePos + d];
                        } else {
                            bufferT_[(nColsStagesAcc[ns*o + stage] + (stagePos + d))*nrowsPack + j] = pols[offsetsStages[stage] + l * nColsStages[stage] + stagePos + d];
                        }
                    }
                }
            }
        }
        if(deviceArgs.expType == 0) {
            for(uint64_t d = 0; d < deviceArgs.boundSize; ++d) {
                for(uint64_t j = 0; j < nrowsPack; ++j) {
                    bufferT_[(nColsStagesAcc[ns*nOpenings] + d + 1)*nrowsPack + j] = zi[row + j + d*domainSize];
                }
            }
            for(uint64_t j = 0; j < nrowsPack; ++j) {
                bufferT_[(nColsStagesAcc[ns*nOpenings])*nrowsPack + j] = x_2ns[row + j];
            }
        } else if(deviceArgs.expType == 1) {
            for(uint64_t d = 0; d < nOpenings; ++d) {
               for(uint64_t k = 0; k < FIELD_EXTENSION; ++k) {
                    for(uint64_t j = 0; j < nrowsPack; ++j) {
                        bufferT_[(nColsStagesAcc[ns*nOpenings] + d*FIELD_EXTENSION + k)*nrowsPack + j] = xDivXSub[(row + j + d*domainSize)*FIELD_EXTENSION + k];
                    }
                }
            }
        } else {
            for(uint64_t j = 0; j < nrowsPack; ++j) {
                bufferT_[(nColsStagesAcc[ns*nOpenings])*nrowsPack + j] = x_n[row + j];
            }
        }

    }

    inline void multiplyPolynomials(Dest &dest, Goldilocks::Element* destVals) {
        if(dest.dim == 1) {
            Goldilocks::op_pack(nrowsPack, 2, &destVals[0], &destVals[0], false, &destVals[FIELD_EXTENSION*nrowsPack],false); // rick
        } else {
            Goldilocks::Element vals[FIELD_EXTENSION*nrowsPack];
            if(dest.params[0].dim == FIELD_EXTENSION && dest.params[1].dim == FIELD_EXTENSION) {
                Goldilocks3::op_pack(nrowsPack, 2, &vals[0], &destVals[0], false, &destVals[FIELD_EXTENSION*nrowsPack], false);
            } else if(dest.params[0].dim == FIELD_EXTENSION && dest.params[1].dim == 1) {
                Goldilocks3::op_31_pack(nrowsPack, 2, &vals[0], &destVals[0], false, &destVals[FIELD_EXTENSION*nrowsPack], false);
            } else {
                Goldilocks3::op_31_pack(nrowsPack, 2, &vals[0], &destVals[FIELD_EXTENSION*nrowsPack], false, &destVals[0], false);
            } 
            Goldilocks::copy_pack(nrowsPack, &destVals[0], &vals[0], false);
            Goldilocks::copy_pack(nrowsPack, &destVals[nrowsPack], &vals[nrowsPack], false);
            Goldilocks::copy_pack(nrowsPack, &destVals[2*nrowsPack], &vals[2*nrowsPack], false);
        }
    }

    inline void multiplyPolynomials(DestGPU &dest, Goldilocks::Element* destVals) {
        if(dest.dim == 1) {
            Goldilocks::op_pack(nrowsPack, 2, &destVals[0], &destVals[0], false, &destVals[FIELD_EXTENSION*nrowsPack],false); // rick
        } else {
            Goldilocks::Element vals[FIELD_EXTENSION*nrowsPack];
            if(dest.params[0].dim == FIELD_EXTENSION && dest.params[1].dim == FIELD_EXTENSION) {
                Goldilocks3::op_pack(nrowsPack, 2, &vals[0], &destVals[0], false, &destVals[FIELD_EXTENSION*nrowsPack], false);
            } else if(dest.params[0].dim == FIELD_EXTENSION && dest.params[1].dim == 1) {
                Goldilocks3::op_31_pack(nrowsPack, 2, &vals[0], &destVals[0], false, &destVals[FIELD_EXTENSION*nrowsPack], false);
            } else {
                Goldilocks3::op_31_pack(nrowsPack, 2, &vals[0], &destVals[FIELD_EXTENSION*nrowsPack], false, &destVals[0], false);
            } 
            Goldilocks::copy_pack(nrowsPack, &destVals[0], &vals[0], false);
            Goldilocks::copy_pack(nrowsPack, &destVals[nrowsPack], &vals[nrowsPack], false);
            Goldilocks::copy_pack(nrowsPack, &destVals[2*nrowsPack], &vals[2*nrowsPack], false);
        }
    }

    inline void storePolynomial(std::vector<Dest> dests, Goldilocks::Element** destVals, uint64_t row) {
        for(uint64_t i = 0; i < dests.size(); ++i) {
            if(dests[i].dim == 1) {
                uint64_t offset = dests[i].offset != 0 ? dests[i].offset : 1;
                Goldilocks::copy_pack(nrowsPack, &dests[i].dest[row*offset], uint64_t(offset), &destVals[i][0], false);
            } else {
                uint64_t offset = dests[i].offset != 0 ? dests[i].offset : FIELD_EXTENSION;
                Goldilocks::copy_pack(nrowsPack, &dests[i].dest[row*offset], uint64_t(offset), &destVals[i][0], false);
                Goldilocks::copy_pack(nrowsPack, &dests[i].dest[row*offset + 1], uint64_t(offset), &destVals[i][nrowsPack], false);
                Goldilocks::copy_pack(nrowsPack, &dests[i].dest[row*offset + 2], uint64_t(offset), &destVals[i][2*nrowsPack], false);
            }
        }
    }

    inline void storePolynomial(DestGPU* dests, uint32_t nDests,  Goldilocks::Element** destVals, uint64_t row) {
        for(uint64_t i = 0; i < nDests; ++i) {
            if(dests[i].dim == 1) {
                uint64_t offset = dests[i].offset != 0 ? dests[i].offset : 1;
                Goldilocks::copy_pack(nrowsPack, &dests[i].dest[row*offset], uint64_t(offset), &destVals[i][0], false);
            } else {
                uint64_t offset = dests[i].offset != 0 ? dests[i].offset : FIELD_EXTENSION;
                Goldilocks::copy_pack(nrowsPack, &dests[i].dest[row*offset], uint64_t(offset), &destVals[i][0], false);
                Goldilocks::copy_pack(nrowsPack, &dests[i].dest[row*offset + 1], uint64_t(offset), &destVals[i][nrowsPack], false);
                Goldilocks::copy_pack(nrowsPack, &dests[i].dest[row*offset + 2], uint64_t(offset), &destVals[i][2*nrowsPack], false);
            }
        }
    }

    inline void printTmp1(uint64_t row, Goldilocks::Element* tmp) {
        Goldilocks::Element buff[nrowsPack];
        Goldilocks::copy_pack(nrowsPack, buff, tmp, false);
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
            Goldilocks::copy_pack(nrowsPack, &buff[0], uint64_t(FIELD_EXTENSION), &bufferT[0], false);
            Goldilocks::copy_pack(nrowsPack, &buff[1], uint64_t(FIELD_EXTENSION), &bufferT[setupCtx.starkInfo.openingPoints.size()], false);
            Goldilocks::copy_pack(nrowsPack, &buff[2], uint64_t(FIELD_EXTENSION), &bufferT[2*setupCtx.starkInfo.openingPoints.size()], false);
            for(uint64_t i = 0; i < 1; ++i) {
                cout << "Value at row " << row + i << " is [" << Goldilocks::toString(buff[FIELD_EXTENSION*i]) << ", " << Goldilocks::toString(buff[FIELD_EXTENSION*i + 1]) << ", " << Goldilocks::toString(buff[FIELD_EXTENSION*i + 2]) << "]" << endl;
            }
        } else {
            Goldilocks::Element buff[nrowsPack];
            Goldilocks::copy_pack(nrowsPack, &buff[0], &bufferT[0], false);
            for(uint64_t i = 0; i < nrowsPack; ++i) {
                cout << "Value at row " << row + i << " is " << Goldilocks::toString(buff[i]) << endl;
            }
        }
    }

    void calculateExpressions(StepsParams& params, ParserArgs &parserArgs, std::vector<Dest> dests, uint64_t domainSize) override {

        uint64_t nOpenings = setupCtx.starkInfo.openingPoints.size();
        uint64_t ns = 2 + setupCtx.starkInfo.nStages + setupCtx.starkInfo.customCommits.size();
        setBufferTInfo(domainSize, params, parserArgs, dests);

        Goldilocks::Element* challenges = params.challenges;
        Goldilocks::Element* numbers =(Goldilocks::Element*) parserArgs.numbers;
        Goldilocks::Element* publics = params.publicInputs;
        Goldilocks::Element* evals = params.evals;
        Goldilocks::Element* airgroupValues = params.airgroupValues;
        Goldilocks::Element* airValues = params.airValues;
        
    #pragma omp parallel for
        for (uint64_t i = 0; i < domainSize; i+= nrowsPack) {
            Goldilocks::Element bufferT_[nOpenings*nCols*nrowsPack];
            loadPolynomials(bufferT_, i); 

            Goldilocks::Element **destVals = new Goldilocks::Element*[dests.size()];

            for(uint64_t j = 0; j < dests.size(); ++j) {
                destVals[j] = new Goldilocks::Element[dests[j].params.size() * FIELD_EXTENSION* nrowsPack];
                for(uint64_t k = 0; k < dests[j].params.size(); ++k) {
                    uint64_t i_args = 0;

                    if(dests[j].params[k].op == opType::cm || dests[j].params[k].op == opType::const_) {
                        uint64_t openingPointIndex = dests[j].params[k].rowOffsetIndex;
                        uint64_t buffPos = ns*openingPointIndex + dests[j].params[k].stage;
                        uint64_t stagePos = dests[j].params[k].stagePos;
                        copyPolynomial(&deviceArgs,&destVals[j][k*FIELD_EXTENSION*nrowsPack], dests[j].params[k].inverse, dests[j].params[k].batch, dests[j].params[k].dim, &bufferT_[(nColsStagesAcc[buffPos] + stagePos)*nrowsPack]);
                        continue;
                    } else if(dests[j].params[k].op == opType::number) {
                        uint64_t val = dests[j].params[k].inverse ? Goldilocks::inv(Goldilocks::fromU64(dests[j].params[k].value)).fe : dests[j].params[k].value;
                        for(uint64_t r = 0; r < nrowsPack; ++r) {
                            destVals[j][k*FIELD_EXTENSION*nrowsPack + r] = Goldilocks::fromU64(val);
                        }
                        continue;
                    }

                    uint8_t* ops = &parserArgs.ops[dests[j].params[k].parserParams.opsOffset];
                    uint16_t* args = &parserArgs.args[dests[j].params[k].parserParams.argsOffset];
                    Goldilocks::Element tmp1[dests[j].params[k].parserParams.nTemp1*nrowsPack];
                    Goldilocks::Element tmp3[dests[j].params[k].parserParams.nTemp3*nrowsPack*FIELD_EXTENSION];

                    for (uint64_t kk = 0; kk < dests[j].params[k].parserParams.nOps; ++kk) {
                        switch (ops[kk]) {
                            case 0: {
                                // COPY commit1 to tmp1
                                Goldilocks::copy_pack(nrowsPack, &tmp1[args[i_args] * nrowsPack], &bufferT_[(nColsStagesAcc[args[i_args + 1]] + args[i_args + 2]) * nrowsPack], false);
                                i_args += 3;
                                break;
                            }
                            case 1: {
                                // OPERATION WITH DEST: tmp1 - SRC0: commit1 - SRC1: commit1
                                Goldilocks::op_pack(nrowsPack, args[i_args], &tmp1[args[i_args + 1] * nrowsPack], &bufferT_[(nColsStagesAcc[args[i_args + 2]] + args[i_args + 3]) * nrowsPack], false, &bufferT_[(nColsStagesAcc[args[i_args + 4]] + args[i_args + 5]) * nrowsPack], false);
                                i_args += 6;
                                break;
                            }
                            case 2: {
                                // OPERATION WITH DEST: tmp1 - SRC0: commit1 - SRC1: tmp1
                                Goldilocks::op_pack(nrowsPack, args[i_args], &tmp1[args[i_args + 1] * nrowsPack], &bufferT_[(nColsStagesAcc[args[i_args + 2]] + args[i_args + 3]) * nrowsPack], false, &tmp1[args[i_args + 4] * nrowsPack],false);
                                i_args += 5;
                                break;
                            }
                            case 3: {
                                // OPERATION WITH DEST: tmp1 - SRC0: commit1 - SRC1: public
                                Goldilocks::op_pack(nrowsPack, args[i_args], &tmp1[args[i_args + 1] * nrowsPack], &bufferT_[(nColsStagesAcc[args[i_args + 2]] + args[i_args + 3]) * nrowsPack], false, &publics[args[i_args + 4]], true);
                                i_args += 5;
                                break;
                            }
                            case 4: {
                                // OPERATION WITH DEST: tmp1 - SRC0: commit1 - SRC1: number
                                Goldilocks::op_pack(nrowsPack, args[i_args], &tmp1[args[i_args + 1] * nrowsPack], &bufferT_[(nColsStagesAcc[args[i_args + 2]] + args[i_args + 3]) * nrowsPack], false, &numbers[args[i_args + 4]], true);
                                i_args += 5;
                                break;
                            }
                            case 5: {
                                // OPERATION WITH DEST: tmp1 - SRC0: commit1 - SRC1: airvalue1
                                Goldilocks::op_pack(nrowsPack, args[i_args], &tmp1[args[i_args + 1] * nrowsPack], &bufferT_[(nColsStagesAcc[args[i_args + 2]] + args[i_args + 3]) * nrowsPack], false, &airValues[args[i_args + 4]*FIELD_EXTENSION], true);
                                i_args += 5;
                                break;
                            }
                            case 6: {
                                // COPY tmp1 to tmp1
                                Goldilocks::copy_pack(nrowsPack, &tmp1[args[i_args] * nrowsPack], &tmp1[args[i_args + 1] * nrowsPack], false);
                                i_args += 2;
                                break;
                            }
                            case 7: {
                                // OPERATION WITH DEST: tmp1 - SRC0: tmp1 - SRC1: tmp1
                                Goldilocks::op_pack(nrowsPack, args[i_args], &tmp1[args[i_args + 1] * nrowsPack], &tmp1[args[i_args + 2] * nrowsPack], false, &tmp1[args[i_args + 3] * nrowsPack], false);
                                i_args += 4;
                                break;
                            }
                            case 8: {
                                // OPERATION WITH DEST: tmp1 - SRC0: tmp1 - SRC1: public
                                Goldilocks::op_pack(nrowsPack, args[i_args], &tmp1[args[i_args + 1] * nrowsPack], &tmp1[args[i_args + 2] * nrowsPack], false, &publics[args[i_args + 3]], true);
                                i_args += 4;
                                break;
                            }
                            case 9: {
                                // OPERATION WITH DEST: tmp1 - SRC0: tmp1 - SRC1: number
                                Goldilocks::op_pack(nrowsPack, args[i_args], &tmp1[args[i_args + 1] * nrowsPack], &tmp1[args[i_args + 2] * nrowsPack], false, &numbers[args[i_args + 3]], true);
                                i_args += 4;
                                break;
                            }
                            case 10: {
                                // OPERATION WITH DEST: tmp1 - SRC0: tmp1 - SRC1: airvalue1
                                Goldilocks::op_pack(nrowsPack, args[i_args], &tmp1[args[i_args + 1] * nrowsPack], &tmp1[args[i_args + 2] * nrowsPack], false, &airValues[args[i_args + 3]*FIELD_EXTENSION], true);
                                i_args += 4;
                                break;
                            }
                            case 11: {
                                // COPY public to tmp1
                                Goldilocks::copy_pack(nrowsPack, &tmp1[args[i_args] * nrowsPack], &publics[args[i_args + 1]], true);
                                i_args += 2;
                                break;
                            }
                            case 12: {
                                // OPERATION WITH DEST: tmp1 - SRC0: public - SRC1: public
                                Goldilocks::op_pack(nrowsPack, args[i_args], &tmp1[args[i_args + 1] * nrowsPack], &publics[args[i_args + 2]], true, &publics[args[i_args + 3]],true);
                                i_args += 4;
                                break;
                            }
                            case 13: {
                                // OPERATION WITH DEST: tmp1 - SRC0: public - SRC1: number
                                Goldilocks::op_pack(nrowsPack, args[i_args], &tmp1[args[i_args + 1] * nrowsPack], &publics[args[i_args + 2]], true, &numbers[args[i_args + 3]], true);
                                i_args += 4;
                                break;
                            }
                            case 14: {
                                // OPERATION WITH DEST: tmp1 - SRC0: public - SRC1: airvalue1
                                Goldilocks::op_pack(nrowsPack, args[i_args], &tmp1[args[i_args + 1] * nrowsPack], &publics[args[i_args + 2]], true, &airValues[args[i_args + 3]*FIELD_EXTENSION], true);
                                i_args += 4;
                                break;
                            }
                            case 15: {
                                // COPY number to tmp1
                                Goldilocks::copy_pack(nrowsPack, &tmp1[args[i_args] * nrowsPack], false, &numbers[args[i_args + 1]], true);
                                i_args += 2;
                                break;
                            }
                            case 16: {
                                // OPERATION WITH DEST: tmp1 - SRC0: number - SRC1: number
                                Goldilocks::op_pack(nrowsPack, args[i_args], &tmp1[args[i_args + 1] * nrowsPack], &numbers[args[i_args + 2]], true, &numbers[args[i_args + 3]], true);
                                i_args += 4;
                                break;
                            }
                            case 17: {
                                // OPERATION WITH DEST: tmp1 - SRC0: number - SRC1: airvalue1
                                Goldilocks::op_pack(nrowsPack, args[i_args], &tmp1[args[i_args + 1] * nrowsPack], &numbers[args[i_args + 2]], true,  &airValues[args[i_args + 3]*FIELD_EXTENSION], true);
                                i_args += 4;
                                break;
                            }
                            case 18: {
                                // COPY airvalue1 to tmp1
                                Goldilocks::copy_pack(nrowsPack, &tmp1[args[i_args] * nrowsPack], &airValues[args[i_args + 1]*FIELD_EXTENSION], true);
                                i_args += 2;
                                break;
                            }
                            case 19: {
                                // OPERATION WITH DEST: tmp1 - SRC0: airvalue1 - SRC1: airvalue1
                                Goldilocks::op_pack(nrowsPack, args[i_args], &tmp1[args[i_args + 1] * nrowsPack], &airValues[args[i_args + 2]*FIELD_EXTENSION], true, &airValues[args[i_args + 3]*FIELD_EXTENSION], true);
                                i_args += 4;
                                break;
                            }
                            case 20: {
                                // OPERATION WITH DEST: tmp3 - SRC0: commit3 - SRC1: commit1
                                Goldilocks3::op_31_pack(nrowsPack, args[i_args], &tmp3[args[i_args + 1] * nrowsPack * FIELD_EXTENSION], &bufferT_[(nColsStagesAcc[args[i_args + 2]] + args[i_args + 3]) * nrowsPack], false, &bufferT_[(nColsStagesAcc[args[i_args + 4]] + args[i_args + 5]) * nrowsPack], false);
                                i_args += 6;
                                break;
                            }
                            case 21: {
                                // OPERATION WITH DEST: tmp3 - SRC0: commit3 - SRC1: tmp1
                                Goldilocks3::op_31_pack(nrowsPack, args[i_args], &tmp3[args[i_args + 1] * nrowsPack * FIELD_EXTENSION], &bufferT_[(nColsStagesAcc[args[i_args + 2]] + args[i_args + 3]) * nrowsPack], false, &tmp1[args[i_args + 4] * nrowsPack], false);
                                i_args += 5;
                                break;
                            }
                            case 22: {
                                // OPERATION WITH DEST: tmp3 - SRC0: commit3 - SRC1: public
                                Goldilocks3::op_31_pack(nrowsPack, args[i_args], &tmp3[args[i_args + 1] * nrowsPack * FIELD_EXTENSION], &bufferT_[(nColsStagesAcc[args[i_args + 2]] + args[i_args + 3]) * nrowsPack], false, &publics[args[i_args + 4]], true);
                                i_args += 5;
                                break;
                            }
                            case 23: {
                                // OPERATION WITH DEST: tmp3 - SRC0: commit3 - SRC1: number
                                Goldilocks3::op_31_pack(nrowsPack, args[i_args], &tmp3[args[i_args + 1] * nrowsPack * FIELD_EXTENSION], &bufferT_[(nColsStagesAcc[args[i_args + 2]] + args[i_args + 3]) * nrowsPack], false, &numbers[args[i_args + 4]], true);
                                i_args += 5;
                                break;
                            }
                            case 24: {
                                // OPERATION WITH DEST: tmp3 - SRC0: commit3 - SRC1: airvalue1
                                Goldilocks3::op_31_pack(nrowsPack, args[i_args], &tmp3[args[i_args + 1] * nrowsPack * FIELD_EXTENSION], &bufferT_[(nColsStagesAcc[args[i_args + 2]] + args[i_args + 3]) * nrowsPack], false, &airValues[args[i_args + 4]*FIELD_EXTENSION], true);
                                i_args += 5;
                                break;
                            }
                            case 25: {
                                // OPERATION WITH DEST: tmp3 - SRC0: tmp3 - SRC1: commit1
                                Goldilocks3::op_31_pack(nrowsPack, args[i_args], &tmp3[args[i_args + 1] * nrowsPack * FIELD_EXTENSION], &tmp3[args[i_args + 2] * nrowsPack * FIELD_EXTENSION], false, &bufferT_[(nColsStagesAcc[args[i_args + 3]] + args[i_args + 4]) * nrowsPack],false);
                                i_args += 5;
                                break;
                            }
                            case 26: {
                                // OPERATION WITH DEST: tmp3 - SRC0: tmp3 - SRC1: tmp1
                                Goldilocks3::op_31_pack(nrowsPack, args[i_args], &tmp3[args[i_args + 1] * nrowsPack * FIELD_EXTENSION], &tmp3[args[i_args + 2] * nrowsPack * FIELD_EXTENSION],false, &tmp1[args[i_args + 3] * nrowsPack], false);
                                i_args += 4;
                                break;
                            }
                            case 27: {
                                // OPERATION WITH DEST: tmp3 - SRC0: tmp3 - SRC1: public
                                Goldilocks3::op_31_pack(nrowsPack, args[i_args], &tmp3[args[i_args + 1] * nrowsPack * FIELD_EXTENSION], &tmp3[args[i_args + 2] * nrowsPack * FIELD_EXTENSION],false, &publics[args[i_args + 3]],true);
                                i_args += 4;
                                break;
                            }
                            case 28: {
                                // OPERATION WITH DEST: tmp3 - SRC0: tmp3 - SRC1: number
                                Goldilocks3::op_31_pack(nrowsPack, args[i_args], &tmp3[args[i_args + 1] * nrowsPack * FIELD_EXTENSION], &tmp3[args[i_args + 2] * nrowsPack * FIELD_EXTENSION], false, &numbers[args[i_args + 3]],true);
                                i_args += 4;
                                break;
                            }
                            case 29: {
                                // OPERATION WITH DEST: tmp3 - SRC0: tmp3 - SRC1: airvalue1
                                Goldilocks3::op_31_pack(nrowsPack, args[i_args], &tmp3[args[i_args + 1] * nrowsPack * FIELD_EXTENSION], &tmp3[args[i_args + 2] * nrowsPack * FIELD_EXTENSION],false, &airValues[args[i_args + 3]*FIELD_EXTENSION],true);
                                i_args += 4;
                                break;
                            }
                            case 30: {
                                // OPERATION WITH DEST: tmp3 - SRC0: challenge - SRC1: commit1
                                Goldilocks3::op_31_pack(nrowsPack, args[i_args], &tmp3[args[i_args + 1] * nrowsPack * FIELD_EXTENSION], &challenges[args[i_args + 2]*FIELD_EXTENSION], true, &bufferT_[(nColsStagesAcc[args[i_args + 3]] + args[i_args + 4]) * nrowsPack],false);
                                i_args += 5;
                                break;
                            }
                            case 31: {
                                // OPERATION WITH DEST: tmp3 - SRC0: challenge - SRC1: tmp1
                                Goldilocks3::op_31_pack(nrowsPack, args[i_args], &tmp3[args[i_args + 1] * nrowsPack * FIELD_EXTENSION], &challenges[args[i_args + 2]*FIELD_EXTENSION], true, &tmp1[args[i_args + 3] * nrowsPack],false);
                                i_args += 4;
                                break;
                            }
                            case 32: {
                                // OPERATION WITH DEST: tmp3 - SRC0: challenge - SRC1: public
                                Goldilocks3::op_31_pack(nrowsPack, args[i_args], &tmp3[args[i_args + 1] * nrowsPack * FIELD_EXTENSION], &challenges[args[i_args + 2]*FIELD_EXTENSION], true,  &publics[args[i_args + 3]],true);
                                i_args += 4;
                                break;
                            }
                            case 33: {
                                // OPERATION WITH DEST: tmp3 - SRC0: challenge - SRC1: number
                                Goldilocks3::op_31_pack(nrowsPack, args[i_args], &tmp3[args[i_args + 1] * nrowsPack * FIELD_EXTENSION], &challenges[args[i_args + 2]*FIELD_EXTENSION], true, &numbers[args[i_args + 3]], true);
                                i_args += 4;
                                break;
                            }
                            case 34: {
                                // OPERATION WITH DEST: tmp3 - SRC0: challenge - SRC1: airvalue1
                                Goldilocks3::op_31_pack(nrowsPack, args[i_args], &tmp3[args[i_args + 1] * nrowsPack * FIELD_EXTENSION], &challenges[args[i_args + 2]*FIELD_EXTENSION], true, &airValues[args[i_args + 3]*FIELD_EXTENSION], true);
                                i_args += 4;
                                break;
                            }
                            case 35: {
                                // OPERATION WITH DEST: tmp3 - SRC0: airgroupvalue - SRC1: commit1
                                Goldilocks3::op_31_pack(nrowsPack, args[i_args], &tmp3[args[i_args + 1] * nrowsPack * FIELD_EXTENSION], &airgroupValues[args[i_args + 2]*FIELD_EXTENSION], true, &bufferT_[(nColsStagesAcc[args[i_args + 3]] + args[i_args + 4]) * nrowsPack], false);
                                i_args += 5;
                                break;
                            }
                            case 36: {
                                // OPERATION WITH DEST: tmp3 - SRC0: airgroupvalue - SRC1: tmp1
                                Goldilocks3::op_31_pack(nrowsPack, args[i_args], &tmp3[args[i_args + 1] * nrowsPack * FIELD_EXTENSION], &airgroupValues[args[i_args + 2]*FIELD_EXTENSION], true, &tmp1[args[i_args + 3] * nrowsPack], false);
                                i_args += 4;
                                break;
                            }
                            case 37: {
                                // OPERATION WITH DEST: tmp3 - SRC0: airgroupvalue - SRC1: public
                                Goldilocks3::op_31_pack(nrowsPack, args[i_args], &tmp3[args[i_args + 1] * nrowsPack * FIELD_EXTENSION], &airgroupValues[args[i_args + 2]*FIELD_EXTENSION], true, &publics[args[i_args + 3]],true);
                                i_args += 4;
                                break;
                            }
                            case 38: {
                                // OPERATION WITH DEST: tmp3 - SRC0: airgroupvalue - SRC1: number
                                Goldilocks3::op_31_pack(nrowsPack, args[i_args], &tmp3[args[i_args + 1] * nrowsPack * FIELD_EXTENSION], &airgroupValues[args[i_args + 2]*FIELD_EXTENSION], true, &numbers[args[i_args + 3]], true);
                                i_args += 4;
                                break;
                            }
                            case 39: {
                                // OPERATION WITH DEST: tmp3 - SRC0: airgroupvalue - SRC1: airvalue1
                                Goldilocks3::op_31_pack(nrowsPack, args[i_args], &tmp3[args[i_args + 1] * nrowsPack * FIELD_EXTENSION], &airgroupValues[args[i_args + 2]*FIELD_EXTENSION], true, &airValues[args[i_args + 3]*FIELD_EXTENSION], true);
                                i_args += 4;
                                break;
                            }
                            case 40: {
                                // OPERATION WITH DEST: tmp3 - SRC0: airvalue3 - SRC1: commit1
                                Goldilocks3::op_31_pack(nrowsPack, args[i_args], &tmp3[args[i_args + 1] * nrowsPack * FIELD_EXTENSION], &airValues[args[i_args + 2]*FIELD_EXTENSION], true, &bufferT_[(nColsStagesAcc[args[i_args + 3]] + args[i_args + 4]) * nrowsPack], false);
                                i_args += 5;
                                break;
                            }
                            case 41: {
                                // OPERATION WITH DEST: tmp3 - SRC0: airvalue3 - SRC1: tmp1
                                Goldilocks3::op_31_pack(nrowsPack, args[i_args], &tmp3[args[i_args + 1] * nrowsPack * FIELD_EXTENSION], &airValues[args[i_args + 2]*FIELD_EXTENSION], true, &tmp1[args[i_args + 3] * nrowsPack], false);
                                i_args += 4;
                                break;
                            }
                            case 42: {
                                // OPERATION WITH DEST: tmp3 - SRC0: airvalue3 - SRC1: public
                                Goldilocks3::op_31_pack(nrowsPack, args[i_args], &tmp3[args[i_args + 1] * nrowsPack * FIELD_EXTENSION], &airValues[args[i_args + 2]*FIELD_EXTENSION], true, &publics[args[i_args + 3]], true);
                                i_args += 4;
                                break;
                            }
                            case 43: {
                                // OPERATION WITH DEST: tmp3 - SRC0: airvalue3 - SRC1: number
                                Goldilocks3::op_31_pack(nrowsPack, args[i_args], &tmp3[args[i_args + 1] * nrowsPack * FIELD_EXTENSION], &airValues[args[i_args + 2]*FIELD_EXTENSION], true, &numbers[args[i_args + 3]], true);
                                i_args += 4;
                                break;
                            }
                            case 44: {
                                // OPERATION WITH DEST: tmp3 - SRC0: airvalue3 - SRC1: airvalue1
                                Goldilocks3::op_31_pack(nrowsPack, args[i_args], &tmp3[args[i_args + 1] * nrowsPack * FIELD_EXTENSION], &airValues[args[i_args + 2]*FIELD_EXTENSION], true, &airValues[args[i_args + 3]*FIELD_EXTENSION], true);
                                i_args += 4;
                                break;
                            }
                            case 45: {
                                // COPY commit3 to tmp3
                                Goldilocks3::copy_pack(nrowsPack, &tmp3[args[i_args] * nrowsPack * FIELD_EXTENSION], &bufferT_[(nColsStagesAcc[args[i_args + 1]] + args[i_args + 2]) * nrowsPack], false);
                                i_args += 3;
                                break;
                            }
                            case 46: {
                                // OPERATION WITH DEST: tmp3 - SRC0: commit3 - SRC1: commit3
                                Goldilocks3::op_pack(nrowsPack, args[i_args], &tmp3[args[i_args + 1] * nrowsPack * FIELD_EXTENSION], &bufferT_[(nColsStagesAcc[args[i_args + 2]] + args[i_args + 3]) * nrowsPack], false, &bufferT_[(nColsStagesAcc[args[i_args + 4]] + args[i_args + 5]) * nrowsPack], false);
                                i_args += 6;
                                break;
                            }
                            case 47: {
                                // OPERATION WITH DEST: tmp3 - SRC0: commit3 - SRC1: tmp3
                                Goldilocks3::op_pack(nrowsPack, args[i_args], &tmp3[args[i_args + 1] * nrowsPack * FIELD_EXTENSION], &bufferT_[(nColsStagesAcc[args[i_args + 2]] + args[i_args + 3]) * nrowsPack], false, &tmp3[args[i_args + 4] * nrowsPack * FIELD_EXTENSION],false);
                                i_args += 5;
                                break;
                            }
                            case 48: {
                                // OPERATION WITH DEST: tmp3 - SRC0: commit3 - SRC1: challenge
                                Goldilocks3::op_pack(nrowsPack, args[i_args], &tmp3[args[i_args + 1] * nrowsPack * FIELD_EXTENSION], &bufferT_[(nColsStagesAcc[args[i_args + 2]] + args[i_args + 3]) * nrowsPack], false, &challenges[args[i_args + 4]*FIELD_EXTENSION], true);
                                i_args += 5;
                                break;
                            }
                            case 49: {
                                // OPERATION WITH DEST: tmp3 - SRC0: commit3 - SRC1: airgroupvalue
                                Goldilocks3::op_pack(nrowsPack, args[i_args], &tmp3[args[i_args + 1] * nrowsPack * FIELD_EXTENSION], &bufferT_[(nColsStagesAcc[args[i_args + 2]] + args[i_args + 3]) * nrowsPack], false, &airgroupValues[args[i_args + 4]*FIELD_EXTENSION], true);
                                i_args += 5;
                                break;
                            }
                            case 50: {
                                // OPERATION WITH DEST: tmp3 - SRC0: commit3 - SRC1: airvalue3
                                Goldilocks3::op_pack(nrowsPack, args[i_args], &tmp3[args[i_args + 1] * nrowsPack * FIELD_EXTENSION], &bufferT_[(nColsStagesAcc[args[i_args + 2]] + args[i_args + 3]) * nrowsPack], false, &airValues[args[i_args + 4]*FIELD_EXTENSION], true);
                                i_args += 5;
                                break;
                            }
                            case 51: {
                                // COPY tmp3 to tmp3
                                Goldilocks3::copy_pack(nrowsPack, &tmp3[args[i_args] * nrowsPack * FIELD_EXTENSION], &tmp3[args[i_args + 1] * nrowsPack * FIELD_EXTENSION], false);
                                i_args += 2;
                                break;
                            }
                            case 52: {
                                // OPERATION WITH DEST: tmp3 - SRC0: tmp3 - SRC1: tmp3
                                Goldilocks3::op_pack(nrowsPack, args[i_args], &tmp3[args[i_args + 1] * nrowsPack * FIELD_EXTENSION], &tmp3[args[i_args + 2] * nrowsPack * FIELD_EXTENSION], false, &tmp3[args[i_args + 3] * nrowsPack * FIELD_EXTENSION], false);
                                i_args += 4;
                                break;
                            }
                            case 53: {
                                // OPERATION WITH DEST: tmp3 - SRC0: tmp3 - SRC1: challenge
                                Goldilocks3::op_pack(nrowsPack, args[i_args], &tmp3[args[i_args + 1] * nrowsPack * FIELD_EXTENSION], &tmp3[args[i_args + 2] * nrowsPack * FIELD_EXTENSION], false, &challenges[args[i_args + 3]*FIELD_EXTENSION], true);
                                i_args += 4;
                                break;
                            }
                            case 54: {
                                // OPERATION WITH DEST: tmp3 - SRC0: tmp3 - SRC1: airgroupvalue
                                Goldilocks3::op_pack(nrowsPack, args[i_args], &tmp3[args[i_args + 1] * nrowsPack * FIELD_EXTENSION], &tmp3[args[i_args + 2] * nrowsPack * FIELD_EXTENSION], false, &airgroupValues[args[i_args + 3]*FIELD_EXTENSION], true);
                                i_args += 4;
                                break;
                            }
                            case 55: {
                                // OPERATION WITH DEST: tmp3 - SRC0: tmp3 - SRC1: airvalue3
                                Goldilocks3::op_pack(nrowsPack, args[i_args], &tmp3[args[i_args + 1] * nrowsPack * FIELD_EXTENSION], &tmp3[args[i_args + 2] * nrowsPack * FIELD_EXTENSION], false, &airValues[args[i_args + 3]*FIELD_EXTENSION],true);
                                i_args += 4;
                                break;
                            }
                            case 56: {
                                // OPERATION WITH DEST: tmp3 - SRC0: challenge - SRC1: challenge
                                Goldilocks3::op_pack(nrowsPack, args[i_args], &tmp3[args[i_args + 1] * nrowsPack * FIELD_EXTENSION], &challenges[args[i_args + 2]*FIELD_EXTENSION], true, &challenges[args[i_args + 3]*FIELD_EXTENSION], true);
                                i_args += 4;
                                break;
                            }
                            case 57: {
                                // OPERATION WITH DEST: tmp3 - SRC0: challenge - SRC1: airgroupvalue
                                Goldilocks3::op_pack(nrowsPack, args[i_args], &tmp3[args[i_args + 1] * nrowsPack * FIELD_EXTENSION], &challenges[args[i_args + 2]*FIELD_EXTENSION], true, &airgroupValues[args[i_args + 3]*FIELD_EXTENSION], true);
                                i_args += 4;
                                break;
                            }
                            case 58: {
                                // OPERATION WITH DEST: tmp3 - SRC0: challenge - SRC1: airvalue3
                                Goldilocks3::op_pack(nrowsPack, args[i_args], &tmp3[args[i_args + 1] * nrowsPack * FIELD_EXTENSION], &challenges[args[i_args + 2]*FIELD_EXTENSION], true, &airValues[args[i_args + 3]*FIELD_EXTENSION], true);
                                i_args += 4;
                                break;
                            }
                            case 59: {
                                // OPERATION WITH DEST: tmp3 - SRC0: airgroupvalue - SRC1: airgroupvalue
                                Goldilocks3::op_pack(nrowsPack, args[i_args], &tmp3[args[i_args + 1] * nrowsPack * FIELD_EXTENSION], &airgroupValues[args[i_args + 2]*FIELD_EXTENSION], true, &airgroupValues[args[i_args + 3]*FIELD_EXTENSION], true);
                                i_args += 4;
                                break;
                            }
                            case 60: {
                                // OPERATION WITH DEST: tmp3 - SRC0: airgroupvalue - SRC1: airvalue3
                                Goldilocks3::op_pack(nrowsPack, args[i_args], &tmp3[args[i_args + 1] * nrowsPack * FIELD_EXTENSION], &airgroupValues[args[i_args + 2]*FIELD_EXTENSION], true, &airValues[args[i_args + 3]*FIELD_EXTENSION*nrowsPack], LDBL_TRUE_MIN);
                                i_args += 4;
                                break;
                            }
                            case 61: {
                                // OPERATION WITH DEST: tmp3 - SRC0: airvalue3 - SRC1: airvalue3
                                Goldilocks3::op_pack(nrowsPack, args[i_args], &tmp3[args[i_args + 1] * nrowsPack * FIELD_EXTENSION], &airValues[args[i_args + 2]*FIELD_EXTENSION], true, &airValues[args[i_args + 3]*FIELD_EXTENSION], true);
                                i_args += 4;
                                break;
                            }
                            case 62: {
                                // COPY eval to tmp3
                                Goldilocks3::copy_pack(nrowsPack, &tmp3[args[i_args] * nrowsPack * FIELD_EXTENSION], &evals[args[i_args + 1]*FIELD_EXTENSION], true);
                                i_args += 2;
                                break;
                            }
                            case 63: {
                                // OPERATION WITH DEST: tmp3 - SRC0: challenge - SRC1: eval
                                Goldilocks3::op_pack(nrowsPack, args[i_args], &tmp3[args[i_args + 1] * nrowsPack * FIELD_EXTENSION], &challenges[args[i_args + 2]*FIELD_EXTENSION], true, &evals[args[i_args + 3]*FIELD_EXTENSION], true);
                                i_args += 4;
                                break;
                            }
                            case 64: {
                                // OPERATION WITH DEST: tmp3 - SRC0: tmp3 - SRC1: eval
                                Goldilocks3::op_pack(nrowsPack, args[i_args], &tmp3[args[i_args + 1] * nrowsPack * FIELD_EXTENSION], &tmp3[args[i_args + 2] * nrowsPack * FIELD_EXTENSION], false, &evals[args[i_args + 3]*FIELD_EXTENSION], true);
                                i_args += 4;
                                break;
                            }
                            case 65: {
                                // OPERATION WITH DEST: tmp3 - SRC0: eval - SRC1: commit1
                                Goldilocks3::op_31_pack(nrowsPack, args[i_args], &tmp3[args[i_args + 1] * nrowsPack * FIELD_EXTENSION], &evals[args[i_args + 2]*FIELD_EXTENSION], true, &bufferT_[(nColsStagesAcc[args[i_args + 3]] + args[i_args + 4]) * nrowsPack], false);
                                i_args += 5;
                                break;
                            }
                            case 66: {
                                // OPERATION WITH DEST: tmp3 - SRC0: commit3 - SRC1: eval
                                Goldilocks3::op_pack(nrowsPack, args[i_args], &tmp3[args[i_args + 1] * nrowsPack * FIELD_EXTENSION], &bufferT_[(nColsStagesAcc[args[i_args + 2]] + args[i_args + 3]) * nrowsPack], false, &evals[args[i_args + 4]*FIELD_EXTENSION], true);
                                i_args += 5;
                                break;
                            }
                            case 67: {
                                // OPERATION WITH DEST: tmp3 - SRC0: eval - SRC1: eval
                                Goldilocks3::op_pack(nrowsPack, args[i_args], &tmp3[args[i_args + 1] * nrowsPack * FIELD_EXTENSION], &evals[args[i_args + 2]*FIELD_EXTENSION], true, &evals[args[i_args + 3]*FIELD_EXTENSION], true);
                                i_args += 4;
                                break;
                            }
                            case 68: {
                                // OPERATION WITH DEST: tmp3 - SRC0: eval - SRC1: public
                                Goldilocks3::op_31_pack(nrowsPack, args[i_args], &tmp3[args[i_args + 1] * nrowsPack * FIELD_EXTENSION], &evals[args[i_args + 2]*FIELD_EXTENSION], true, &publics[args[i_args + 3]], true);
                                i_args += 4;
                                break;
                            }
                            case 69: {
                                // OPERATION WITH DEST: tmp3 - SRC0: eval - SRC1: number
                                Goldilocks3::op_31_pack(nrowsPack, args[i_args], &tmp3[args[i_args + 1] * nrowsPack * FIELD_EXTENSION], &evals[args[i_args + 2]*FIELD_EXTENSION], true, &numbers[args[i_args + 3]], true);
                                i_args += 4;
                                break;
                            }
                            case 70: {
                                // OPERATION WITH DEST: tmp3 - SRC0: airgroupvalue - SRC1: eval
                                Goldilocks3::op_pack(nrowsPack, args[i_args], &tmp3[args[i_args + 1] * nrowsPack * FIELD_EXTENSION], &airgroupValues[args[i_args + 2]*FIELD_EXTENSION], true, &evals[args[i_args + 3]*FIELD_EXTENSION], true);
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
                        copyPolynomial(&deviceArgs, &destVals[j][k*FIELD_EXTENSION*nrowsPack], dests[j].params[k].inverse, dests[j].params[k].batch, dests[j].params[k].parserParams.destDim, &tmp1[dests[j].params[k].parserParams.destId*nrowsPack]);
                    } else {
                        copyPolynomial(&deviceArgs, &destVals[j][k*FIELD_EXTENSION*nrowsPack], dests[j].params[k].inverse, dests[j].params[k].batch, dests[j].params[k].parserParams.destDim, &tmp3[dests[j].params[k].parserParams.destId*FIELD_EXTENSION*nrowsPack]);
                    }
                }

                if(dests[j].params.size() == 2) {
                    multiplyPolynomials(dests[j], destVals[j]);
                }
            }
            storePolynomial(dests, destVals, i);

            for(uint64_t j = 0; j < dests.size(); ++j) {
                delete destVals[j];
            }
            delete[] destVals;
        }    
    }

    void calculateExpressions_gpu(StepsParams& params, StepsParams& params_gpu, ParserArgs &parserArgs, std::vector<Dest> dests, uint64_t domainSize) {

        setBufferTInfo(domainSize, params, parserArgs, dests);

        double time = omp_get_wtime();
        loadDeviceArguments(params_gpu);
        time = omp_get_wtime() - time;
        std::cout << "rick cudaMalloc expressions time: " << time << std::endl;
        computeExpressions(d_deviceArgs, &deviceArgs);
    
    }

    void loadDeviceArguments(StepsParams& params_gpu) {
        // Allocate memory for the struct on the device
        cudaMalloc(&d_deviceArgs, sizeof(DeviceArguments));

        // Allocate memory for the pointers within the struct and copy data
        uint64_t* d_nextStrides;
        cudaMalloc(&d_nextStrides, deviceArgs.nOpenings * sizeof(uint64_t));
        cudaMemcpy(d_nextStrides, deviceArgs.nextStrides, deviceArgs.nOpenings * sizeof(uint64_t), cudaMemcpyHostToDevice);

        uint64_t* d_nColsStages;
        cudaMalloc(&d_nColsStages, nColsStages.size() * sizeof(uint64_t));
        cudaMemcpy(d_nColsStages, deviceArgs.nColsStages, nColsStages.size() * sizeof(uint64_t), cudaMemcpyHostToDevice);

        uint64_t* d_nColsStagesAcc;
        cudaMalloc(&d_nColsStagesAcc, nColsStagesAcc.size() * sizeof(uint64_t));
        cudaMemcpy(d_nColsStagesAcc, deviceArgs.nColsStagesAcc, nColsStagesAcc.size() * sizeof(uint64_t), cudaMemcpyHostToDevice);

        uint64_t* d_offsetsStages;
        cudaMalloc(&d_offsetsStages, offsetsStages.size() * sizeof(uint64_t));
        cudaMemcpy(d_offsetsStages, deviceArgs.offsetsStages, offsetsStages.size() * sizeof(uint64_t), cudaMemcpyHostToDevice);

        uint64_t* d_cmPolsInfo;
        cudaMalloc(&d_cmPolsInfo, 3 * deviceArgs.cmPolsInfoSize * sizeof(uint64_t));
        cudaMemcpy(d_cmPolsInfo, deviceArgs.cmPolsInfo, 3 * deviceArgs.cmPolsInfoSize * sizeof(uint64_t), cudaMemcpyHostToDevice);

        Goldilocks::Element* d_zi;
        cudaMalloc(&d_zi, deviceArgs.boundSize * deviceArgs.N * sizeof(Goldilocks::Element));
        cudaMemcpy(d_zi, deviceArgs.zi, deviceArgs.boundSize * deviceArgs.N * sizeof(Goldilocks::Element), cudaMemcpyHostToDevice);

        Goldilocks::Element* d_x_n;
        cudaMalloc(&d_x_n, deviceArgs.N * sizeof(Goldilocks::Element));
        cudaMemcpy(d_x_n, deviceArgs.x_n, deviceArgs.N * sizeof(Goldilocks::Element), cudaMemcpyHostToDevice);

        Goldilocks::Element* d_x_2ns;
        cudaMalloc(&d_x_2ns, deviceArgs.NExtended * sizeof(Goldilocks::Element));
        cudaMemcpy(d_x_2ns, deviceArgs.x_2ns, deviceArgs.NExtended * sizeof(Goldilocks::Element), cudaMemcpyHostToDevice);

        // bufferT_
        deviceArgs.nBlocks = 1;
        deviceArgs.singleBufferSize = deviceArgs.nOpenings * deviceArgs.nCols * deviceArgs.nrowsPack;
        deviceArgs.bufferTSize = deviceArgs.nBlocks * (deviceArgs.singleBufferSize+ 4); // this for GL elements are used just to separate local buffers and avoid collinsions into the same cache line
        Goldilocks::Element* d_bufferT_;
        cudaMalloc(&d_bufferT_, deviceArgs.bufferTSize * sizeof(Goldilocks::Element)); 
        deviceArgs.bufferT_ = d_bufferT_;

        //Dests
        DestGPU* d_dests; //rick: com alliberare aquest punter?
        cudaMalloc(&d_dests, deviceArgs.nDests * sizeof(DestGPU));
        cudaMemcpy(d_dests, deviceArgs.dests, deviceArgs.nDests * sizeof(DestGPU), cudaMemcpyHostToDevice);
        Params** d_params = new Params*[deviceArgs.nDests];
        for(uint32_t i = 0; i < deviceArgs.nDests; ++i) {
            cudaMalloc(&d_params[i], deviceArgs.dests[i].nParams * sizeof(Params));
            cudaMemcpy(d_params[i], deviceArgs.dests[i].params, deviceArgs.dests[i].nParams * sizeof(Params), cudaMemcpyHostToDevice);
        }

        // destVals
        deviceArgs.destVals = new Goldilocks::Element**[deviceArgs.nDests];
        for(int i=0; i<deviceArgs.nBlocks; i++) {
            deviceArgs.destVals[i] = new Goldilocks::Element*[deviceArgs.nDests];
            for(uint64_t j = 0; j < deviceArgs.nDests; ++j) {
                deviceArgs.destVals[i][j] = new Goldilocks::Element[deviceArgs.dests[j].nParams * FIELD_EXTENSION* nrowsPack]; 
            }
        }
        Goldilocks::Element*** d_destVals;
        cudaMalloc(&d_destVals, deviceArgs.nDests * sizeof(Goldilocks::Element**));

        // non-polynomial arguments
        Goldilocks::Element* d_challenges;
        cudaMalloc(&d_challenges, deviceArgs.nChallenges * sizeof(Goldilocks::Element));
        cudaMemcpy(d_challenges, deviceArgs.challenges, deviceArgs.nChallenges * sizeof(Goldilocks::Element), cudaMemcpyHostToDevice);

        Goldilocks::Element* d_numbers;
        cudaMalloc(&d_numbers, deviceArgs.nNumbers * sizeof(Goldilocks::Element));
        cudaMemcpy(d_numbers, deviceArgs.numbers, deviceArgs.nNumbers * sizeof(Goldilocks::Element), cudaMemcpyHostToDevice);

        Goldilocks::Element* d_publics;
        cudaMalloc(&d_publics, deviceArgs.nPublics * sizeof(Goldilocks::Element));
        cudaMemcpy(d_publics, deviceArgs.publics, deviceArgs.nPublics * sizeof(Goldilocks::Element), cudaMemcpyHostToDevice);

        Goldilocks::Element* d_evals;
        cudaMalloc(&d_evals, deviceArgs.nEvals * sizeof(Goldilocks::Element));
        cudaMemcpy(d_evals, deviceArgs.evals, deviceArgs.nEvals * sizeof(Goldilocks::Element), cudaMemcpyHostToDevice);

        Goldilocks::Element* d_airgroupValues;
        cudaMalloc(&d_airgroupValues, deviceArgs.nAirgroupValues * sizeof(Goldilocks::Element));
        cudaMemcpy(d_airgroupValues, deviceArgs.airgroupValues, deviceArgs.nAirgroupValues * sizeof(Goldilocks::Element), cudaMemcpyHostToDevice);

        Goldilocks::Element* d_airValues;
        cudaMalloc(&d_airValues, deviceArgs.nAirValues * sizeof(Goldilocks::Element));
        cudaMemcpy(d_airValues, deviceArgs.airValues, deviceArgs.nAirValues * sizeof(Goldilocks::Element), cudaMemcpyHostToDevice);

        // binary expressions
        uint8_t* d_ops;
        cudaMalloc(&d_ops, deviceArgs.nOpsTotal * sizeof(uint8_t));
        cudaMemcpy(d_ops, deviceArgs.ops, deviceArgs.nOpsTotal * sizeof(uint8_t), cudaMemcpyHostToDevice);

        uint16_t* d_args;
        cudaMalloc(&d_args, deviceArgs.nArgsTotal * sizeof(uint64_t));
        cudaMemcpy(d_args, deviceArgs.args, deviceArgs.nArgsTotal * sizeof(uint64_t), cudaMemcpyHostToDevice);


        // Update the device struct with device pointers
        DeviceArguments h_deviceArgs = deviceArgs;
        h_deviceArgs.nextStrides = d_nextStrides;
        h_deviceArgs.nColsStages = d_nColsStages;
        h_deviceArgs.nColsStagesAcc = d_nColsStagesAcc;
        h_deviceArgs.offsetsStages = d_offsetsStages;
        h_deviceArgs.cmPolsInfo = d_cmPolsInfo;
        h_deviceArgs.zi = d_zi;
        h_deviceArgs.x_n = d_x_n;
        h_deviceArgs.x_2ns = d_x_2ns;
        h_deviceArgs.constPols = h_deviceArgs.domainExtended ? params_gpu.pConstPolsExtendedTreeAddress : params_gpu.pConstPolsAddress;
        h_deviceArgs.trace = params_gpu.trace;
        h_deviceArgs.pols = params_gpu.pols;
        h_deviceArgs.xDivXSub = params_gpu.xDivXSub;
        h_deviceArgs.dests = d_dests;
        for(uint32_t iDest = 0; iDest < deviceArgs.nDests; ++iDest) {
            setDestsParams<<<1,1>>>(d_dests, d_params[iDest], iDest);
        }
        h_deviceArgs.challenges = d_challenges;
        h_deviceArgs.numbers = d_numbers;
        h_deviceArgs.publics = d_publics;
        h_deviceArgs.evals = d_evals;
        h_deviceArgs.airgroupValues = d_airgroupValues;
        h_deviceArgs.airValues = d_airValues;
        h_deviceArgs.ops = d_ops;
        h_deviceArgs.args = d_args;
        h_deviceArgs.destVals = d_destVals;
        
        // Copy the updated struct to the device
        cudaMemcpy(d_deviceArgs, &h_deviceArgs, sizeof(DeviceArguments), cudaMemcpyHostToDevice);
        allocateDestVals<<<1,1>>>(d_deviceArgs);

    }

    
};
    void copyPolynomial(DeviceArguments* deviceArgs, Goldilocks::Element* destVals, bool inverse, bool batch, uint64_t dim, Goldilocks::Element* tmp) {
        if(dim == 1) {
            if(inverse) {
                if(batch) {
                    Goldilocks::batchInverse(&destVals[0], &tmp[0], deviceArgs->nrowsPack);
                } else {
                    for(uint64_t i = 0; i < deviceArgs->nrowsPack; ++i) {
                        Goldilocks::inv(destVals[i], tmp[i]);
                    }
                }
            } else {
                Goldilocks::copy_pack(deviceArgs->nrowsPack, &destVals[0], &tmp[0], false);
            }
        } else if(dim == FIELD_EXTENSION) {
            if(inverse) {
                Goldilocks::Element buff[FIELD_EXTENSION*deviceArgs->nrowsPack];
                Goldilocks::copy_pack(deviceArgs->nrowsPack, &buff[0], uint64_t(FIELD_EXTENSION), &tmp[0], false);
                Goldilocks::copy_pack(deviceArgs->nrowsPack, &buff[1], uint64_t(FIELD_EXTENSION), &tmp[deviceArgs->nrowsPack], false);
                Goldilocks::copy_pack(deviceArgs->nrowsPack, &buff[2], uint64_t(FIELD_EXTENSION), &tmp[2*deviceArgs->nrowsPack], false);
                if(batch) {
                    Goldilocks3::batchInverse((Goldilocks3::Element *)buff, (Goldilocks3::Element *)buff, deviceArgs->nrowsPack);
                } else {
                    for(uint64_t i = 0; i < deviceArgs->nrowsPack; ++i) {
                        Goldilocks3::inv((Goldilocks3::Element &)buff[i*FIELD_EXTENSION], (Goldilocks3::Element &)buff[i*FIELD_EXTENSION]);
                    }
                }
                Goldilocks::copy_pack(deviceArgs->nrowsPack, &destVals[0], &buff[0], uint64_t(FIELD_EXTENSION));
                Goldilocks::copy_pack(deviceArgs->nrowsPack, &destVals[deviceArgs->nrowsPack], &buff[1], uint64_t(FIELD_EXTENSION));
                Goldilocks::copy_pack(deviceArgs->nrowsPack, &destVals[2*deviceArgs->nrowsPack], &buff[2], uint64_t(FIELD_EXTENSION));
            } else {
                Goldilocks::copy_pack(deviceArgs->nrowsPack, &destVals[0], &tmp[0], false);
                Goldilocks::copy_pack(deviceArgs->nrowsPack, &destVals[deviceArgs->nrowsPack], &tmp[deviceArgs->nrowsPack], false);
                Goldilocks::copy_pack(deviceArgs->nrowsPack, &destVals[2*deviceArgs->nrowsPack], &tmp[2*deviceArgs->nrowsPack], false);
            }
        }
    }

    void multiplyPolynomials(DeviceArguments* deviceArgs, DestGPU &dest, Goldilocks::Element* destVals) {
        if(dest.dim == 1) {
            Goldilocks::op_pack(deviceArgs->nrowsPack, 2, &destVals[0], &destVals[0], false, &destVals[FIELD_EXTENSION*deviceArgs->nrowsPack],false); // rick
        } else {
            Goldilocks::Element vals[FIELD_EXTENSION*deviceArgs->nrowsPack];
            if(dest.params[0].dim == FIELD_EXTENSION && dest.params[1].dim == FIELD_EXTENSION) {
                Goldilocks3::op_pack(deviceArgs->nrowsPack, 2, &vals[0], &destVals[0], false, &destVals[FIELD_EXTENSION*deviceArgs->nrowsPack], false);
            } else if(dest.params[0].dim == FIELD_EXTENSION && dest.params[1].dim == 1) {
                Goldilocks3::op_31_pack(deviceArgs->nrowsPack, 2, &vals[0], &destVals[0], false, &destVals[FIELD_EXTENSION*deviceArgs->nrowsPack], false);
            } else {
                Goldilocks3::op_31_pack(deviceArgs->nrowsPack, 2, &vals[0], &destVals[FIELD_EXTENSION*deviceArgs->nrowsPack], false, &destVals[0], false);
            } 
            Goldilocks::copy_pack(deviceArgs->nrowsPack, &destVals[0], &vals[0], false);
            Goldilocks::copy_pack(deviceArgs->nrowsPack, &destVals[deviceArgs->nrowsPack], &vals[deviceArgs->nrowsPack], false);
            Goldilocks::copy_pack(deviceArgs->nrowsPack, &destVals[2*deviceArgs->nrowsPack], &vals[2*deviceArgs->nrowsPack], false);
        }
    } 

    void storePolynomial(DeviceArguments* deviceArgs, DestGPU* dests, uint32_t nDests,  Goldilocks::Element** destVals, uint64_t row) {
        for(uint64_t i = 0; i < nDests; ++i) {
            if(dests[i].dim == 1) {
                uint64_t offset = dests[i].offset != 0 ? dests[i].offset : 1;
                Goldilocks::copy_pack(deviceArgs->nrowsPack, &dests[i].dest[row*offset], uint64_t(offset), &destVals[i][0], false);
            } else {
                uint64_t offset = dests[i].offset != 0 ? dests[i].offset : FIELD_EXTENSION;
                Goldilocks::copy_pack(deviceArgs->nrowsPack, &dests[i].dest[row*offset], uint64_t(offset), &destVals[i][0], false);
                Goldilocks::copy_pack(deviceArgs->nrowsPack, &dests[i].dest[row*offset + 1], uint64_t(offset), &destVals[i][deviceArgs->nrowsPack], false);
                Goldilocks::copy_pack(deviceArgs->nrowsPack, &dests[i].dest[row*offset + 2], uint64_t(offset), &destVals[i][2*deviceArgs->nrowsPack], false);
            }
        }
    }
    
    void computeExpressions(DeviceArguments* d_deviceArgs, DeviceArguments* deviceArgs) {

        Goldilocks::Element* challenges = deviceArgs->challenges;
        Goldilocks::Element* numbers = deviceArgs->numbers;
        Goldilocks::Element* publics = deviceArgs->publics;
        Goldilocks::Element* evals = deviceArgs->evals;
        Goldilocks::Element* airgroupValues = deviceArgs->airgroupValues;
        Goldilocks::Element* airValues = deviceArgs->airValues;
        Goldilocks::Element** destVals = deviceArgs->destVals[0];
        uint64_t* nColsStagesAcc = deviceArgs->nColsStagesAcc;
        uint64_t domainSize = deviceArgs->domainSize;
        uint64_t nrowsPack = deviceArgs->nrowsPack;
        uint64_t nOpenings = deviceArgs->nOpenings;
        uint64_t nCols = deviceArgs->nCols;
        DestGPU* dests = deviceArgs->dests;
        uint32_t nDests = deviceArgs->nDests;
        

        //#pragma omp parallel for
        for (uint64_t i = 0; i < domainSize; i+= nrowsPack) {

            Goldilocks::Element bufferT_[nOpenings*nCols*nrowsPack];
            CHECKCUDAERR(cudaMemset(deviceArgs->bufferT_, 0, deviceArgs->singleBufferSize * sizeof(Goldilocks::Element)));
            loadPolynomials_<<<1, nrowsPack >>>(d_deviceArgs, i);
            CHECKCUDAERR(cudaMemcpy(bufferT_, deviceArgs->bufferT_, deviceArgs->singleBufferSize * sizeof(Goldilocks::Element), cudaMemcpyDeviceToHost));

            for(uint64_t j = 0; j < nDests; ++j) {
                for(uint64_t k = 0; k < dests[j].nParams; ++k) {
                    uint64_t i_args = 0;

                    if(dests[j].params[k].op == opType::cm || dests[j].params[k].op == opType::const_) {
                        uint64_t openingPointIndex = dests[j].params[k].rowOffsetIndex;
                        uint64_t buffPos = deviceArgs->ns*openingPointIndex + dests[j].params[k].stage;
                        uint64_t stagePos = dests[j].params[k].stagePos;
                        copyPolynomial(deviceArgs, &destVals[j][k*FIELD_EXTENSION*nrowsPack], dests[j].params[k].inverse, dests[j].params[k].batch, dests[j].params[k].dim, &bufferT_[(nColsStagesAcc[buffPos] + stagePos)*nrowsPack]);
                        continue;
                    } else if(dests[j].params[k].op == opType::number) {
                        uint64_t val = dests[j].params[k].inverse ? Goldilocks::inv(Goldilocks::fromU64(dests[j].params[k].value)).fe : dests[j].params[k].value;
                        for(uint64_t r = 0; r < nrowsPack; ++r) {
                            destVals[j][k*FIELD_EXTENSION*nrowsPack + r] = Goldilocks::fromU64(val);
                        }
                        continue;
                    }

                    uint8_t* ops = &deviceArgs->ops[dests[j].params[k].parserParams.opsOffset]; //tenir a deviceARgs
                    uint16_t* args = &deviceArgs->args[dests[j].params[k].parserParams.argsOffset]; //tenir a device Args
                    Goldilocks::Element tmp1[dests[j].params[k].parserParams.nTemp1*nrowsPack]; 
                    Goldilocks::Element tmp3[dests[j].params[k].parserParams.nTemp3*nrowsPack*FIELD_EXTENSION]; 

                    for (uint64_t kk = 0; kk < dests[j].params[k].parserParams.nOps; ++kk) {
                        switch (ops[kk]) {
                            case 0: {
                                // COPY commit1 to tmp1
                                Goldilocks::copy_pack(nrowsPack, &tmp1[args[i_args] * nrowsPack], &bufferT_[(nColsStagesAcc[args[i_args + 1]] + args[i_args + 2]) * nrowsPack], false);
                                i_args += 3;
                                break;
                            }
                            case 1: {
                                // OPERATION WITH DEST: tmp1 - SRC0: commit1 - SRC1: commit1
                                Goldilocks::op_pack(nrowsPack, args[i_args], &tmp1[args[i_args + 1] * nrowsPack], &bufferT_[(nColsStagesAcc[args[i_args + 2]] + args[i_args + 3]) * nrowsPack], false, &bufferT_[(nColsStagesAcc[args[i_args + 4]] + args[i_args + 5]) * nrowsPack], false);
                                i_args += 6;
                                break;
                            }
                            case 2: {
                                // OPERATION WITH DEST: tmp1 - SRC0: commit1 - SRC1: tmp1
                                Goldilocks::op_pack(nrowsPack, args[i_args], &tmp1[args[i_args + 1] * nrowsPack], &bufferT_[(nColsStagesAcc[args[i_args + 2]] + args[i_args + 3]) * nrowsPack], false, &tmp1[args[i_args + 4] * nrowsPack],false);
                                i_args += 5;
                                break;
                            }
                            case 3: {
                                // OPERATION WITH DEST: tmp1 - SRC0: commit1 - SRC1: public
                                Goldilocks::op_pack(nrowsPack, args[i_args], &tmp1[args[i_args + 1] * nrowsPack], &bufferT_[(nColsStagesAcc[args[i_args + 2]] + args[i_args + 3]) * nrowsPack], false, &publics[args[i_args + 4]], true);
                                i_args += 5;
                                break;
                            }
                            case 4: {
                                // OPERATION WITH DEST: tmp1 - SRC0: commit1 - SRC1: number
                                Goldilocks::op_pack(nrowsPack, args[i_args], &tmp1[args[i_args + 1] * nrowsPack], &bufferT_[(nColsStagesAcc[args[i_args + 2]] + args[i_args + 3]) * nrowsPack], false, &numbers[args[i_args + 4]], true);
                                i_args += 5;
                                break;
                            }
                            case 5: {
                                // OPERATION WITH DEST: tmp1 - SRC0: commit1 - SRC1: airvalue1
                                Goldilocks::op_pack(nrowsPack, args[i_args], &tmp1[args[i_args + 1] * nrowsPack], &bufferT_[(nColsStagesAcc[args[i_args + 2]] + args[i_args + 3]) * nrowsPack], false, &airValues[args[i_args + 4]*FIELD_EXTENSION], true);
                                i_args += 5;
                                break;
                            }
                            case 6: {
                                // COPY tmp1 to tmp1
                                Goldilocks::copy_pack(nrowsPack, &tmp1[args[i_args] * nrowsPack], &tmp1[args[i_args + 1] * nrowsPack], false);
                                i_args += 2;
                                break;
                            }
                            case 7: {
                                // OPERATION WITH DEST: tmp1 - SRC0: tmp1 - SRC1: tmp1
                                Goldilocks::op_pack(nrowsPack, args[i_args], &tmp1[args[i_args + 1] * nrowsPack], &tmp1[args[i_args + 2] * nrowsPack], false, &tmp1[args[i_args + 3] * nrowsPack], false);
                                i_args += 4;
                                break;
                            }
                            case 8: {
                                // OPERATION WITH DEST: tmp1 - SRC0: tmp1 - SRC1: public
                                Goldilocks::op_pack(nrowsPack, args[i_args], &tmp1[args[i_args + 1] * nrowsPack], &tmp1[args[i_args + 2] * nrowsPack], false, &publics[args[i_args + 3]], true);
                                i_args += 4;
                                break;
                            }
                            case 9: {
                                // OPERATION WITH DEST: tmp1 - SRC0: tmp1 - SRC1: number
                                Goldilocks::op_pack(nrowsPack, args[i_args], &tmp1[args[i_args + 1] * nrowsPack], &tmp1[args[i_args + 2] * nrowsPack], false, &numbers[args[i_args + 3]], true);
                                i_args += 4;
                                break;
                            }
                            case 10: {
                                // OPERATION WITH DEST: tmp1 - SRC0: tmp1 - SRC1: airvalue1
                                Goldilocks::op_pack(nrowsPack, args[i_args], &tmp1[args[i_args + 1] * nrowsPack], &tmp1[args[i_args + 2] * nrowsPack], false, &airValues[args[i_args + 3]*FIELD_EXTENSION], true);
                                i_args += 4;
                                break;
                            }
                            case 11: {
                                // COPY public to tmp1
                                Goldilocks::copy_pack(nrowsPack, &tmp1[args[i_args] * nrowsPack], &publics[args[i_args + 1]], true);
                                i_args += 2;
                                break;
                            }
                            case 12: {
                                // OPERATION WITH DEST: tmp1 - SRC0: public - SRC1: public
                                Goldilocks::op_pack(nrowsPack, args[i_args], &tmp1[args[i_args + 1] * nrowsPack], &publics[args[i_args + 2]], true, &publics[args[i_args + 3]],true);
                                i_args += 4;
                                break;
                            }
                            case 13: {
                                // OPERATION WITH DEST: tmp1 - SRC0: public - SRC1: number
                                Goldilocks::op_pack(nrowsPack, args[i_args], &tmp1[args[i_args + 1] * nrowsPack], &publics[args[i_args + 2]], true, &numbers[args[i_args + 3]], true);
                                i_args += 4;
                                break;
                            }
                            case 14: {
                                // OPERATION WITH DEST: tmp1 - SRC0: public - SRC1: airvalue1
                                Goldilocks::op_pack(nrowsPack, args[i_args], &tmp1[args[i_args + 1] * nrowsPack], &publics[args[i_args + 2]], true, &airValues[args[i_args + 3]*FIELD_EXTENSION], true);
                                i_args += 4;
                                break;
                            }
                            case 15: {
                                // COPY number to tmp1
                                Goldilocks::copy_pack(nrowsPack, &tmp1[args[i_args] * nrowsPack], false, &numbers[args[i_args + 1]], true);
                                i_args += 2;
                                break;
                            }
                            case 16: {
                                // OPERATION WITH DEST: tmp1 - SRC0: number - SRC1: number
                                Goldilocks::op_pack(nrowsPack, args[i_args], &tmp1[args[i_args + 1] * nrowsPack], &numbers[args[i_args + 2]], true, &numbers[args[i_args + 3]], true);
                                i_args += 4;
                                break;
                            }
                            case 17: {
                                // OPERATION WITH DEST: tmp1 - SRC0: number - SRC1: airvalue1
                                Goldilocks::op_pack(nrowsPack, args[i_args], &tmp1[args[i_args + 1] * nrowsPack], &numbers[args[i_args + 2]], true,  &airValues[args[i_args + 3]*FIELD_EXTENSION], true);
                                i_args += 4;
                                break;
                            }
                            case 18: {
                                // COPY airvalue1 to tmp1
                                Goldilocks::copy_pack(nrowsPack, &tmp1[args[i_args] * nrowsPack], &airValues[args[i_args + 1]*FIELD_EXTENSION], true);
                                i_args += 2;
                                break;
                            }
                            case 19: {
                                // OPERATION WITH DEST: tmp1 - SRC0: airvalue1 - SRC1: airvalue1
                                Goldilocks::op_pack(nrowsPack, args[i_args], &tmp1[args[i_args + 1] * nrowsPack], &airValues[args[i_args + 2]*FIELD_EXTENSION], true, &airValues[args[i_args + 3]*FIELD_EXTENSION], true);
                                i_args += 4;
                                break;
                            }
                            case 20: {
                                // OPERATION WITH DEST: tmp3 - SRC0: commit3 - SRC1: commit1
                                Goldilocks3::op_31_pack(nrowsPack, args[i_args], &tmp3[args[i_args + 1] * nrowsPack * FIELD_EXTENSION], &bufferT_[(nColsStagesAcc[args[i_args + 2]] + args[i_args + 3]) * nrowsPack], false, &bufferT_[(nColsStagesAcc[args[i_args + 4]] + args[i_args + 5]) * nrowsPack], false);
                                i_args += 6;
                                break;
                            }
                            case 21: {
                                // OPERATION WITH DEST: tmp3 - SRC0: commit3 - SRC1: tmp1
                                Goldilocks3::op_31_pack(nrowsPack, args[i_args], &tmp3[args[i_args + 1] * nrowsPack * FIELD_EXTENSION], &bufferT_[(nColsStagesAcc[args[i_args + 2]] + args[i_args + 3]) * nrowsPack], false, &tmp1[args[i_args + 4] * nrowsPack], false);
                                i_args += 5;
                                break;
                            }
                            case 22: {
                                // OPERATION WITH DEST: tmp3 - SRC0: commit3 - SRC1: public
                                Goldilocks3::op_31_pack(nrowsPack, args[i_args], &tmp3[args[i_args + 1] * nrowsPack * FIELD_EXTENSION], &bufferT_[(nColsStagesAcc[args[i_args + 2]] + args[i_args + 3]) * nrowsPack], false, &publics[args[i_args + 4]], true);
                                i_args += 5;
                                break;
                            }
                            case 23: {
                                // OPERATION WITH DEST: tmp3 - SRC0: commit3 - SRC1: number
                                Goldilocks3::op_31_pack(nrowsPack, args[i_args], &tmp3[args[i_args + 1] * nrowsPack * FIELD_EXTENSION], &bufferT_[(nColsStagesAcc[args[i_args + 2]] + args[i_args + 3]) * nrowsPack], false, &numbers[args[i_args + 4]], true);
                                i_args += 5;
                                break;
                            }
                            case 24: {
                                // OPERATION WITH DEST: tmp3 - SRC0: commit3 - SRC1: airvalue1
                                Goldilocks3::op_31_pack(nrowsPack, args[i_args], &tmp3[args[i_args + 1] * nrowsPack * FIELD_EXTENSION], &bufferT_[(nColsStagesAcc[args[i_args + 2]] + args[i_args + 3]) * nrowsPack], false, &airValues[args[i_args + 4]*FIELD_EXTENSION], true);
                                i_args += 5;
                                break;
                            }
                            case 25: {
                                // OPERATION WITH DEST: tmp3 - SRC0: tmp3 - SRC1: commit1
                                Goldilocks3::op_31_pack(nrowsPack, args[i_args], &tmp3[args[i_args + 1] * nrowsPack * FIELD_EXTENSION], &tmp3[args[i_args + 2] * nrowsPack * FIELD_EXTENSION], false, &bufferT_[(nColsStagesAcc[args[i_args + 3]] + args[i_args + 4]) * nrowsPack],false);
                                i_args += 5;
                                break;
                            }
                            case 26: {
                                // OPERATION WITH DEST: tmp3 - SRC0: tmp3 - SRC1: tmp1
                                Goldilocks3::op_31_pack(nrowsPack, args[i_args], &tmp3[args[i_args + 1] * nrowsPack * FIELD_EXTENSION], &tmp3[args[i_args + 2] * nrowsPack * FIELD_EXTENSION],false, &tmp1[args[i_args + 3] * nrowsPack], false);
                                i_args += 4;
                                break;
                            }
                            case 27: {
                                // OPERATION WITH DEST: tmp3 - SRC0: tmp3 - SRC1: public
                                Goldilocks3::op_31_pack(nrowsPack, args[i_args], &tmp3[args[i_args + 1] * nrowsPack * FIELD_EXTENSION], &tmp3[args[i_args + 2] * nrowsPack * FIELD_EXTENSION],false, &publics[args[i_args + 3]],true);
                                i_args += 4;
                                break;
                            }
                            case 28: {
                                // OPERATION WITH DEST: tmp3 - SRC0: tmp3 - SRC1: number
                                Goldilocks3::op_31_pack(nrowsPack, args[i_args], &tmp3[args[i_args + 1] * nrowsPack * FIELD_EXTENSION], &tmp3[args[i_args + 2] * nrowsPack * FIELD_EXTENSION], false, &numbers[args[i_args + 3]],true);
                                i_args += 4;
                                break;
                            }
                            case 29: {
                                // OPERATION WITH DEST: tmp3 - SRC0: tmp3 - SRC1: airvalue1
                                Goldilocks3::op_31_pack(nrowsPack, args[i_args], &tmp3[args[i_args + 1] * nrowsPack * FIELD_EXTENSION], &tmp3[args[i_args + 2] * nrowsPack * FIELD_EXTENSION],false, &airValues[args[i_args + 3]*FIELD_EXTENSION],true);
                                i_args += 4;
                                break;
                            }
                            case 30: {
                                // OPERATION WITH DEST: tmp3 - SRC0: challenge - SRC1: commit1
                                Goldilocks3::op_31_pack(nrowsPack, args[i_args], &tmp3[args[i_args + 1] * nrowsPack * FIELD_EXTENSION], &challenges[args[i_args + 2]*FIELD_EXTENSION], true, &bufferT_[(nColsStagesAcc[args[i_args + 3]] + args[i_args + 4]) * nrowsPack],false);
                                i_args += 5;
                                break;
                            }
                            case 31: {
                                // OPERATION WITH DEST: tmp3 - SRC0: challenge - SRC1: tmp1
                                Goldilocks3::op_31_pack(nrowsPack, args[i_args], &tmp3[args[i_args + 1] * nrowsPack * FIELD_EXTENSION], &challenges[args[i_args + 2]*FIELD_EXTENSION], true, &tmp1[args[i_args + 3] * nrowsPack],false);
                                i_args += 4;
                                break;
                            }
                            case 32: {
                                // OPERATION WITH DEST: tmp3 - SRC0: challenge - SRC1: public
                                Goldilocks3::op_31_pack(nrowsPack, args[i_args], &tmp3[args[i_args + 1] * nrowsPack * FIELD_EXTENSION], &challenges[args[i_args + 2]*FIELD_EXTENSION], true,  &publics[args[i_args + 3]],true);
                                i_args += 4;
                                break;
                            }
                            case 33: {
                                // OPERATION WITH DEST: tmp3 - SRC0: challenge - SRC1: number
                                Goldilocks3::op_31_pack(nrowsPack, args[i_args], &tmp3[args[i_args + 1] * nrowsPack * FIELD_EXTENSION], &challenges[args[i_args + 2]*FIELD_EXTENSION], true, &numbers[args[i_args + 3]], true);
                                i_args += 4;
                                break;
                            }
                            case 34: {
                                // OPERATION WITH DEST: tmp3 - SRC0: challenge - SRC1: airvalue1
                                Goldilocks3::op_31_pack(nrowsPack, args[i_args], &tmp3[args[i_args + 1] * nrowsPack * FIELD_EXTENSION], &challenges[args[i_args + 2]*FIELD_EXTENSION], true, &airValues[args[i_args + 3]*FIELD_EXTENSION], true);
                                i_args += 4;
                                break;
                            }
                            case 35: {
                                // OPERATION WITH DEST: tmp3 - SRC0: airgroupvalue - SRC1: commit1
                                Goldilocks3::op_31_pack(nrowsPack, args[i_args], &tmp3[args[i_args + 1] * nrowsPack * FIELD_EXTENSION], &airgroupValues[args[i_args + 2]*FIELD_EXTENSION], true, &bufferT_[(nColsStagesAcc[args[i_args + 3]] + args[i_args + 4]) * nrowsPack], false);
                                i_args += 5;
                                break;
                            }
                            case 36: {
                                // OPERATION WITH DEST: tmp3 - SRC0: airgroupvalue - SRC1: tmp1
                                Goldilocks3::op_31_pack(nrowsPack, args[i_args], &tmp3[args[i_args + 1] * nrowsPack * FIELD_EXTENSION], &airgroupValues[args[i_args + 2]*FIELD_EXTENSION], true, &tmp1[args[i_args + 3] * nrowsPack], false);
                                i_args += 4;
                                break;
                            }
                            case 37: {
                                // OPERATION WITH DEST: tmp3 - SRC0: airgroupvalue - SRC1: public
                                Goldilocks3::op_31_pack(nrowsPack, args[i_args], &tmp3[args[i_args + 1] * nrowsPack * FIELD_EXTENSION], &airgroupValues[args[i_args + 2]*FIELD_EXTENSION], true, &publics[args[i_args + 3]],true);
                                i_args += 4;
                                break;
                            }
                            case 38: {
                                // OPERATION WITH DEST: tmp3 - SRC0: airgroupvalue - SRC1: number
                                Goldilocks3::op_31_pack(nrowsPack, args[i_args], &tmp3[args[i_args + 1] * nrowsPack * FIELD_EXTENSION], &airgroupValues[args[i_args + 2]*FIELD_EXTENSION], true, &numbers[args[i_args + 3]], true);
                                i_args += 4;
                                break;
                            }
                            case 39: {
                                // OPERATION WITH DEST: tmp3 - SRC0: airgroupvalue - SRC1: airvalue1
                                Goldilocks3::op_31_pack(nrowsPack, args[i_args], &tmp3[args[i_args + 1] * nrowsPack * FIELD_EXTENSION], &airgroupValues[args[i_args + 2]*FIELD_EXTENSION], true, &airValues[args[i_args + 3]*FIELD_EXTENSION], true);
                                i_args += 4;
                                break;
                            }
                            case 40: {
                                // OPERATION WITH DEST: tmp3 - SRC0: airvalue3 - SRC1: commit1
                                Goldilocks3::op_31_pack(nrowsPack, args[i_args], &tmp3[args[i_args + 1] * nrowsPack * FIELD_EXTENSION], &airValues[args[i_args + 2]*FIELD_EXTENSION], true, &bufferT_[(nColsStagesAcc[args[i_args + 3]] + args[i_args + 4]) * nrowsPack], false);
                                i_args += 5;
                                break;
                            }
                            case 41: {
                                // OPERATION WITH DEST: tmp3 - SRC0: airvalue3 - SRC1: tmp1
                                Goldilocks3::op_31_pack(nrowsPack, args[i_args], &tmp3[args[i_args + 1] * nrowsPack * FIELD_EXTENSION], &airValues[args[i_args + 2]*FIELD_EXTENSION], true, &tmp1[args[i_args + 3] * nrowsPack], false);
                                i_args += 4;
                                break;
                            }
                            case 42: {
                                // OPERATION WITH DEST: tmp3 - SRC0: airvalue3 - SRC1: public
                                Goldilocks3::op_31_pack(nrowsPack, args[i_args], &tmp3[args[i_args + 1] * nrowsPack * FIELD_EXTENSION], &airValues[args[i_args + 2]*FIELD_EXTENSION], true, &publics[args[i_args + 3]], true);
                                i_args += 4;
                                break;
                            }
                            case 43: {
                                // OPERATION WITH DEST: tmp3 - SRC0: airvalue3 - SRC1: number
                                Goldilocks3::op_31_pack(nrowsPack, args[i_args], &tmp3[args[i_args + 1] * nrowsPack * FIELD_EXTENSION], &airValues[args[i_args + 2]*FIELD_EXTENSION], true, &numbers[args[i_args + 3]], true);
                                i_args += 4;
                                break;
                            }
                            case 44: {
                                // OPERATION WITH DEST: tmp3 - SRC0: airvalue3 - SRC1: airvalue1
                                Goldilocks3::op_31_pack(nrowsPack, args[i_args], &tmp3[args[i_args + 1] * nrowsPack * FIELD_EXTENSION], &airValues[args[i_args + 2]*FIELD_EXTENSION], true, &airValues[args[i_args + 3]*FIELD_EXTENSION], true);
                                i_args += 4;
                                break;
                            }
                            case 45: {
                                // COPY commit3 to tmp3
                                Goldilocks3::copy_pack(nrowsPack, &tmp3[args[i_args] * nrowsPack * FIELD_EXTENSION], &bufferT_[(nColsStagesAcc[args[i_args + 1]] + args[i_args + 2]) * nrowsPack], false);
                                i_args += 3;
                                break;
                            }
                            case 46: {
                                // OPERATION WITH DEST: tmp3 - SRC0: commit3 - SRC1: commit3
                                Goldilocks3::op_pack(nrowsPack, args[i_args], &tmp3[args[i_args + 1] * nrowsPack * FIELD_EXTENSION], &bufferT_[(nColsStagesAcc[args[i_args + 2]] + args[i_args + 3]) * nrowsPack], false, &bufferT_[(nColsStagesAcc[args[i_args + 4]] + args[i_args + 5]) * nrowsPack], false);
                                i_args += 6;
                                break;
                            }
                            case 47: {
                                // OPERATION WITH DEST: tmp3 - SRC0: commit3 - SRC1: tmp3
                                Goldilocks3::op_pack(nrowsPack, args[i_args], &tmp3[args[i_args + 1] * nrowsPack * FIELD_EXTENSION], &bufferT_[(nColsStagesAcc[args[i_args + 2]] + args[i_args + 3]) * nrowsPack], false, &tmp3[args[i_args + 4] * nrowsPack * FIELD_EXTENSION],false);
                                i_args += 5;
                                break;
                            }
                            case 48: {
                                // OPERATION WITH DEST: tmp3 - SRC0: commit3 - SRC1: challenge
                                Goldilocks3::op_pack(nrowsPack, args[i_args], &tmp3[args[i_args + 1] * nrowsPack * FIELD_EXTENSION], &bufferT_[(nColsStagesAcc[args[i_args + 2]] + args[i_args + 3]) * nrowsPack], false, &challenges[args[i_args + 4]*FIELD_EXTENSION], true);
                                i_args += 5;
                                break;
                            }
                            case 49: {
                                // OPERATION WITH DEST: tmp3 - SRC0: commit3 - SRC1: airgroupvalue
                                Goldilocks3::op_pack(nrowsPack, args[i_args], &tmp3[args[i_args + 1] * nrowsPack * FIELD_EXTENSION], &bufferT_[(nColsStagesAcc[args[i_args + 2]] + args[i_args + 3]) * nrowsPack], false, &airgroupValues[args[i_args + 4]*FIELD_EXTENSION], true);
                                i_args += 5;
                                break;
                            }
                            case 50: {
                                // OPERATION WITH DEST: tmp3 - SRC0: commit3 - SRC1: airvalue3
                                Goldilocks3::op_pack(nrowsPack, args[i_args], &tmp3[args[i_args + 1] * nrowsPack * FIELD_EXTENSION], &bufferT_[(nColsStagesAcc[args[i_args + 2]] + args[i_args + 3]) * nrowsPack], false, &airValues[args[i_args + 4]*FIELD_EXTENSION], true);
                                i_args += 5;
                                break;
                            }
                            case 51: {
                                // COPY tmp3 to tmp3
                                Goldilocks3::copy_pack(nrowsPack, &tmp3[args[i_args] * nrowsPack * FIELD_EXTENSION], &tmp3[args[i_args + 1] * nrowsPack * FIELD_EXTENSION], false);
                                i_args += 2;
                                break;
                            }
                            case 52: {
                                // OPERATION WITH DEST: tmp3 - SRC0: tmp3 - SRC1: tmp3
                                Goldilocks3::op_pack(nrowsPack, args[i_args], &tmp3[args[i_args + 1] * nrowsPack * FIELD_EXTENSION], &tmp3[args[i_args + 2] * nrowsPack * FIELD_EXTENSION], false, &tmp3[args[i_args + 3] * nrowsPack * FIELD_EXTENSION], false);
                                i_args += 4;
                                break;
                            }
                            case 53: {
                                // OPERATION WITH DEST: tmp3 - SRC0: tmp3 - SRC1: challenge
                                Goldilocks3::op_pack(nrowsPack, args[i_args], &tmp3[args[i_args + 1] * nrowsPack * FIELD_EXTENSION], &tmp3[args[i_args + 2] * nrowsPack * FIELD_EXTENSION], false, &challenges[args[i_args + 3]*FIELD_EXTENSION], true);
                                i_args += 4;
                                break;
                            }
                            case 54: {
                                // OPERATION WITH DEST: tmp3 - SRC0: tmp3 - SRC1: airgroupvalue
                                Goldilocks3::op_pack(nrowsPack, args[i_args], &tmp3[args[i_args + 1] * nrowsPack * FIELD_EXTENSION], &tmp3[args[i_args + 2] * nrowsPack * FIELD_EXTENSION], false, &airgroupValues[args[i_args + 3]*FIELD_EXTENSION], true);
                                i_args += 4;
                                break;
                            }
                            case 55: {
                                // OPERATION WITH DEST: tmp3 - SRC0: tmp3 - SRC1: airvalue3
                                Goldilocks3::op_pack(nrowsPack, args[i_args], &tmp3[args[i_args + 1] * nrowsPack * FIELD_EXTENSION], &tmp3[args[i_args + 2] * nrowsPack * FIELD_EXTENSION], false, &airValues[args[i_args + 3]*FIELD_EXTENSION],true);
                                i_args += 4;
                                break;
                            }
                            case 56: {
                                // OPERATION WITH DEST: tmp3 - SRC0: challenge - SRC1: challenge
                                Goldilocks3::op_pack(nrowsPack, args[i_args], &tmp3[args[i_args + 1] * nrowsPack * FIELD_EXTENSION], &challenges[args[i_args + 2]*FIELD_EXTENSION], true, &challenges[args[i_args + 3]*FIELD_EXTENSION], true);
                                i_args += 4;
                                break;
                            }
                            case 57: {
                                // OPERATION WITH DEST: tmp3 - SRC0: challenge - SRC1: airgroupvalue
                                Goldilocks3::op_pack(nrowsPack, args[i_args], &tmp3[args[i_args + 1] * nrowsPack * FIELD_EXTENSION], &challenges[args[i_args + 2]*FIELD_EXTENSION], true, &airgroupValues[args[i_args + 3]*FIELD_EXTENSION], true);
                                i_args += 4;
                                break;
                            }
                            case 58: {
                                // OPERATION WITH DEST: tmp3 - SRC0: challenge - SRC1: airvalue3
                                Goldilocks3::op_pack(nrowsPack, args[i_args], &tmp3[args[i_args + 1] * nrowsPack * FIELD_EXTENSION], &challenges[args[i_args + 2]*FIELD_EXTENSION], true, &airValues[args[i_args + 3]*FIELD_EXTENSION], true);
                                i_args += 4;
                                break;
                            }
                            case 59: {
                                // OPERATION WITH DEST: tmp3 - SRC0: airgroupvalue - SRC1: airgroupvalue
                                Goldilocks3::op_pack(nrowsPack, args[i_args], &tmp3[args[i_args + 1] * nrowsPack * FIELD_EXTENSION], &airgroupValues[args[i_args + 2]*FIELD_EXTENSION], true, &airgroupValues[args[i_args + 3]*FIELD_EXTENSION], true);
                                i_args += 4;
                                break;
                            }
                            case 60: {
                                // OPERATION WITH DEST: tmp3 - SRC0: airgroupvalue - SRC1: airvalue3
                                Goldilocks3::op_pack(nrowsPack, args[i_args], &tmp3[args[i_args + 1] * nrowsPack * FIELD_EXTENSION], &airgroupValues[args[i_args + 2]*FIELD_EXTENSION], true, &airValues[args[i_args + 3]*FIELD_EXTENSION*nrowsPack], LDBL_TRUE_MIN);
                                i_args += 4;
                                break;
                            }
                            case 61: {
                                // OPERATION WITH DEST: tmp3 - SRC0: airvalue3 - SRC1: airvalue3
                                Goldilocks3::op_pack(nrowsPack, args[i_args], &tmp3[args[i_args + 1] * nrowsPack * FIELD_EXTENSION], &airValues[args[i_args + 2]*FIELD_EXTENSION], true, &airValues[args[i_args + 3]*FIELD_EXTENSION], true);
                                i_args += 4;
                                break;
                            }
                            case 62: {
                                // COPY eval to tmp3
                                Goldilocks3::copy_pack(nrowsPack, &tmp3[args[i_args] * nrowsPack * FIELD_EXTENSION], &evals[args[i_args + 1]*FIELD_EXTENSION], true);
                                i_args += 2;
                                break;
                            }
                            case 63: {
                                // OPERATION WITH DEST: tmp3 - SRC0: challenge - SRC1: eval
                                Goldilocks3::op_pack(nrowsPack, args[i_args], &tmp3[args[i_args + 1] * nrowsPack * FIELD_EXTENSION], &challenges[args[i_args + 2]*FIELD_EXTENSION], true, &evals[args[i_args + 3]*FIELD_EXTENSION], true);
                                i_args += 4;
                                break;
                            }
                            case 64: {
                                // OPERATION WITH DEST: tmp3 - SRC0: tmp3 - SRC1: eval
                                Goldilocks3::op_pack(nrowsPack, args[i_args], &tmp3[args[i_args + 1] * nrowsPack * FIELD_EXTENSION], &tmp3[args[i_args + 2] * nrowsPack * FIELD_EXTENSION], false, &evals[args[i_args + 3]*FIELD_EXTENSION], true);
                                i_args += 4;
                                break;
                            }
                            case 65: {
                                // OPERATION WITH DEST: tmp3 - SRC0: eval - SRC1: commit1
                                Goldilocks3::op_31_pack(nrowsPack, args[i_args], &tmp3[args[i_args + 1] * nrowsPack * FIELD_EXTENSION], &evals[args[i_args + 2]*FIELD_EXTENSION], true, &bufferT_[(nColsStagesAcc[args[i_args + 3]] + args[i_args + 4]) * nrowsPack], false);
                                i_args += 5;
                                break;
                            }
                            case 66: {
                                // OPERATION WITH DEST: tmp3 - SRC0: commit3 - SRC1: eval
                                Goldilocks3::op_pack(nrowsPack, args[i_args], &tmp3[args[i_args + 1] * nrowsPack * FIELD_EXTENSION], &bufferT_[(nColsStagesAcc[args[i_args + 2]] + args[i_args + 3]) * nrowsPack], false, &evals[args[i_args + 4]*FIELD_EXTENSION], true);
                                i_args += 5;
                                break;
                            }
                            case 67: {
                                // OPERATION WITH DEST: tmp3 - SRC0: eval - SRC1: eval
                                Goldilocks3::op_pack(nrowsPack, args[i_args], &tmp3[args[i_args + 1] * nrowsPack * FIELD_EXTENSION], &evals[args[i_args + 2]*FIELD_EXTENSION], true, &evals[args[i_args + 3]*FIELD_EXTENSION], true);
                                i_args += 4;
                                break;
                            }
                            case 68: {
                                // OPERATION WITH DEST: tmp3 - SRC0: eval - SRC1: public
                                Goldilocks3::op_31_pack(nrowsPack, args[i_args], &tmp3[args[i_args + 1] * nrowsPack * FIELD_EXTENSION], &evals[args[i_args + 2]*FIELD_EXTENSION], true, &publics[args[i_args + 3]], true);
                                i_args += 4;
                                break;
                            }
                            case 69: {
                                // OPERATION WITH DEST: tmp3 - SRC0: eval - SRC1: number
                                Goldilocks3::op_31_pack(nrowsPack, args[i_args], &tmp3[args[i_args + 1] * nrowsPack * FIELD_EXTENSION], &evals[args[i_args + 2]*FIELD_EXTENSION], true, &numbers[args[i_args + 3]], true);
                                i_args += 4;
                                break;
                            }
                            case 70: {
                                // OPERATION WITH DEST: tmp3 - SRC0: airgroupvalue - SRC1: eval
                                Goldilocks3::op_pack(nrowsPack, args[i_args], &tmp3[args[i_args + 1] * nrowsPack * FIELD_EXTENSION], &airgroupValues[args[i_args + 2]*FIELD_EXTENSION], true, &evals[args[i_args + 3]*FIELD_EXTENSION], true);
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

                    //rick: em baixo temp1 i temp3

                    if(dests[j].params[k].parserParams.destDim == 1) {
                        copyPolynomial(deviceArgs, &destVals[j][k*FIELD_EXTENSION*nrowsPack], dests[j].params[k].inverse, dests[j].params[k].batch, dests[j].params[k].parserParams.destDim, &tmp1[dests[j].params[k].parserParams.destId*nrowsPack]);
                    } else {
                        copyPolynomial(deviceArgs, &destVals[j][k*FIELD_EXTENSION*nrowsPack], dests[j].params[k].inverse, dests[j].params[k].batch, dests[j].params[k].parserParams.destDim, &tmp3[dests[j].params[k].parserParams.destId*FIELD_EXTENSION*nrowsPack]);
                    }
                }

                if(dests[j].nParams == 2) {
                    multiplyPolynomials(deviceArgs, dests[j], destVals[j]);
                }
            }
            storePolynomial(deviceArgs,dests, nDests, destVals, i);
        }

    }
    __global__ void setDestsParams(DestGPU* d_dests, Params* d_params, uint64_t iDest) {
        d_dests[iDest].params = d_params;
    }
    __global__ void allocateDestVals(DeviceArguments* d_deviceArgs) {
        for(int i=0; i<d_deviceArgs->nBlocks; i++) {
            d_deviceArgs->destVals[i] = new Goldilocks::Element*[d_deviceArgs->nDests];
            for(uint64_t j = 0; j < d_deviceArgs->nDests; ++j) {
                d_deviceArgs->destVals[i][j] = new Goldilocks::Element[d_deviceArgs->dests[j].nParams * FIELD_EXTENSION* d_deviceArgs->nrowsPack]; 
            }
        }
    }
    __global__ void loadPolynomials_(DeviceArguments* d_deviceArgs, uint64_t row){

        uint64_t row_loc = threadIdx.x;
        uint64_t nOpenings = d_deviceArgs->nOpenings;
        uint64_t ns = d_deviceArgs->ns;
        bool domainExtended = d_deviceArgs->domainExtended;
        uint64_t domainSize = d_deviceArgs->domainSize;
        uint64_t nrowsPack = d_deviceArgs->nrowsPack;
        Goldilocks::Element *constPols = domainExtended ? &d_deviceArgs->constPols[2] : d_deviceArgs->constPols;
        uint64_t constPolsSize = d_deviceArgs->constPolsSize;
        uint64_t* nextStrides = d_deviceArgs->nextStrides;
        uint64_t* nColsStages = d_deviceArgs->nColsStages;
        uint64_t* nColsStagesAcc = d_deviceArgs->nColsStagesAcc;
        uint64_t* offsetsStages = d_deviceArgs->offsetsStages;
        uint64_t cmPolsInfoSize = d_deviceArgs->cmPolsInfoSize;
        uint64_t* cmPolsInfo = d_deviceArgs->cmPolsInfo;
        Goldilocks::Element* trace = d_deviceArgs->trace;
        Goldilocks::Element* pols = d_deviceArgs->pols;
        Goldilocks::Element* zi = d_deviceArgs->zi;
        Goldilocks::Element* x_n = d_deviceArgs->x_n;
        Goldilocks::Element* x_2ns = d_deviceArgs->x_2ns;
        Goldilocks::Element* xDivXSub = d_deviceArgs->xDivXSub;
        Goldilocks::Element* d_bufferT_ = d_deviceArgs->bufferT_;

        for(uint64_t k = 0; k < constPolsSize;  ++k) {
            for(uint64_t o = 0; o < nOpenings; ++o) {
                uint64_t l = (row + row_loc + nextStrides[o]) % domainSize;
                d_bufferT_[(nColsStagesAcc[ns*o] + k)*nrowsPack + row_loc] = constPols[l * nColsStages[0] + k];
            }
        }
        
        for(uint64_t k = 0; k < cmPolsInfoSize; ++k) {
            uint64_t stage = cmPolsInfo[k*3];
            uint64_t stagePos = cmPolsInfo[k*3 + 1];
            for(uint64_t d = 0; d < cmPolsInfo[k*3+2]; ++d) {
                for(uint64_t o = 0; o < nOpenings; ++o) {
                        uint64_t l = (row + row_loc + nextStrides[o]) % domainSize;
                        if(stage == 1 && !domainExtended) {
                            d_bufferT_[(nColsStagesAcc[ns*o + stage] + (stagePos + d))*nrowsPack + row_loc] = trace[l * nColsStages[stage] + stagePos + d];
                        } else {
                            d_bufferT_[(nColsStagesAcc[ns*o + stage] + (stagePos + d))*nrowsPack + row_loc] = pols[offsetsStages[stage] + l * nColsStages[stage] + stagePos + d];
                        }
                }
            }
        }
        
        if(d_deviceArgs->expType == 0) {
            for(uint64_t d = 0; d < d_deviceArgs->boundSize; ++d) {
                d_bufferT_[(nColsStagesAcc[ns*nOpenings] + d + 1)*nrowsPack + row_loc] = zi[row + row_loc + d*domainSize];
                
            }
            d_bufferT_[(nColsStagesAcc[ns*nOpenings])*nrowsPack + row_loc] = x_2ns[row + row_loc];
            
        } else if(d_deviceArgs->expType == 1) {
            for(uint64_t d = 0; d < nOpenings; ++d) {
               for(uint64_t k = 0; k < FIELD_EXTENSION; ++k) {
                    d_bufferT_[(nColsStagesAcc[ns*nOpenings] + d*FIELD_EXTENSION + k)*nrowsPack + row_loc] = xDivXSub[(row + row_loc + d*domainSize)*FIELD_EXTENSION + k];
                }
            }
        } else {
            d_bufferT_[(nColsStagesAcc[ns*nOpenings])*nrowsPack + row_loc] = x_n[row + row_loc];
        }
    }

#endif