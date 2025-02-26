#ifndef EXPRESSIONS_GPU_HPP
#define EXPRESSIONS_GPU_HPP
#include "expressions_ctx.hpp"
#include "cuda_utils.cuh"
#include "cuda_utils.hpp"
#include "gl64_t.cuh"
#include "goldilocks_cubic_extension.cuh"
#include <omp.h>

// #define _ROW_DEBUG_ 0

struct ParserParamsGPU
{
    uint32_t stage;
    uint32_t expId;
    uint32_t nTemp1;
    uint32_t nTemp3;
    uint32_t nOps;
    uint32_t opsOffset;
    uint32_t nArgs;
    uint32_t argsOffset;
    uint32_t constPolsOffset;
    uint32_t cmPolsOffset;
    uint32_t challengesOffset;
    uint32_t publicsOffset;
    uint32_t airgroupValuesOffset;
    uint32_t airValuesOffset;
    uint32_t firstRow;
    uint32_t lastRow;
    uint32_t destDim;
    uint32_t destId;
    bool imPol;
};
struct ParamsGPU
{
    ParserParamsGPU parserParams;
    uint64_t dim;
    uint64_t stage;
    uint64_t stagePos;
    uint64_t polsMapId;
    uint64_t rowOffsetIndex;
    bool inverse = false;
    bool batch = true;
    opType op;
    uint64_t value;
};
struct DestGPU
{
    Goldilocks::Element *dest = nullptr; // rick:this will disapear
    Goldilocks::Element *dest_gpu = nullptr;
    uint64_t offset = 0;
    uint64_t dim = 1;
    uint32_t nParams;
    ParamsGPU *params;
};
struct DeviceArguments
{
    uint64_t N;
    uint64_t NExtended;
    uint64_t domainSize;
    uint64_t nrowsPack;
    uint64_t nCols;
    uint64_t nOpenings;
    uint64_t ns;
    bool domainExtended;
    uint64_t *nextStrides;
    uint64_t *nColsStages;
    uint64_t *nColsStagesAcc;
    uint64_t *offsetsStages;
    Goldilocks::Element *constPols;
    uint64_t constPolsSize;
    uint64_t *cmPolsInfo;
    uint64_t cmPolsInfoSize;
    Goldilocks::Element *trace;
    Goldilocks::Element *aux_trace;
    uint32_t expType;
    uint64_t boundSize;
    Goldilocks::Element *zi;
    Goldilocks::Element *x_n;
    Goldilocks::Element *x_2ns;
    Goldilocks::Element *xDivXSub;
    // non polnomial arguments
    uint32_t nChallenges;
    Goldilocks::Element *challenges;
    uint32_t nNumbers;
    Goldilocks::Element *numbers;
    uint32_t nPublics;
    Goldilocks::Element *publics;
    uint32_t nEvals;
    Goldilocks::Element *evals;
    uint32_t nAirgroupValues;
    Goldilocks::Element *airgroupValues;
    uint32_t nAirValues;
    Goldilocks::Element *airValues;
    // Dests
    DestGPU *dests;
    uint32_t nDests;
    // Expressions bin
    uint8_t *ops;
    uint32_t nOpsTotal;
    uint16_t *args;
    uint32_t nArgsTotal;
    // buffer
    uint64_t nBlocks;
    uint64_t bufferSize;
    Goldilocks::Element *bufferT_;
    // destVals
    uint64_t destValsSize;
    Goldilocks::Element *destVals;
    // tmps
    uint32_t tmp1Size;
    Goldilocks::Element **tmp1;
    uint32_t tmp3Size;
    Goldilocks::Element **tmp3;
};
void computeExpressions(DeviceArguments *d_deviceArgs, DeviceArguments *deviceArgs);
void copyPolynomial(DeviceArguments *deviceArgs, Goldilocks::Element *destVals, bool inverse, bool batch, uint64_t dim, Goldilocks::Element *tmp);
void multiplyPolynomials(DeviceArguments *deviceArgs, DestGPU &dest, Goldilocks::Element *destVals);
void storePolynomial(DeviceArguments *deviceArgs, DestGPU *dests, uint32_t nDests, Goldilocks::Element *destVals, uint64_t row);
__device__ void storeOnePolynomial__(DeviceArguments *d_deviceArgs, gl64_t *destVals, uint64_t row, uint32_t idest);
__device__ void storePolynomial__(DeviceArguments *d_deviceArgs, gl64_t *destVals, uint64_t row);
__global__ void copyPolynomial_(DeviceArguments *d_deviceArgs, Goldilocks::Element *d_destVals, bool inverse, uint64_t dim, Goldilocks::Element *d_tmp);
__device__ void copyPolynomial__(DeviceArguments *d_deviceArgs, gl64_t *d_destVals, bool inverse, uint64_t dim, gl64_t *d_tmp);
__global__ void loadPolynomials_(DeviceArguments *d_deviceArgs, uint64_t row, uint32_t iBlock);
__device__ void loadPolynomials__(DeviceArguments *d_deviceArgs, uint64_t row, uint32_t iBlock);
__device__ void multiplyPolynomials__(DeviceArguments *deviceArgs, DestGPU &dest, gl64_t *destVals);
__global__ void computeExpressions_(DeviceArguments *d_deviceArgs);
__global__ void freeDeviceArguments_(DeviceArguments *d_deviceArgs);

class ExpressionsGPU : public ExpressionsCtx
{
public:
    uint64_t nrowsPack;
    uint32_t nBlocks;
    uint64_t nCols;
    uint32_t nParamsMax;
    uint32_t nTemp1Max;
    uint32_t nTemp3Max;
    vector<uint64_t> nColsStages;
    vector<uint64_t> nColsStagesAcc;
    vector<uint64_t> offsetsStages;
    DeviceArguments deviceArgs;
    DeviceArguments *d_deviceArgs;
    DeviceArguments h_deviceArgs;
    DestGPU *d_dests;

    ExpressionsGPU(SetupCtx &setupCtx, uint32_t nParamsMax, uint32_t nTemp1Max, uint32_t nTemp3Max, uint64_t nrowsPack_ = 64, uint32_t nBlocks_ = 256) : ExpressionsCtx(setupCtx), nParamsMax(nParamsMax), nTemp1Max(nTemp1Max), nTemp3Max(nTemp3Max), nrowsPack(nrowsPack_), nBlocks(nBlocks_)
    {
        uint64_t nOpenings = setupCtx.starkInfo.openingPoints.size();
        uint64_t ns = 2 + setupCtx.starkInfo.nStages + setupCtx.starkInfo.customCommits.size();
        nColsStages.resize(ns * nOpenings + 1);
        nColsStagesAcc.resize(ns * nOpenings + 1);

        for (uint64_t o = 0; o < nOpenings; ++o)
        {
            for (uint64_t stage = 0; stage < ns; ++stage)
            {
                if (stage == 0)
                {
                    nColsStages[ns * o] = setupCtx.starkInfo.mapSectionsN["const"];
                    nColsStagesAcc[ns * o] = o == 0 ? 0 : nColsStagesAcc[ns * o + stage - 1] + nColsStages[stage - 1];
                }
                else if (stage < 2 + setupCtx.starkInfo.nStages)
                {
                    std::string section = "cm" + to_string(stage);
                    nColsStages[ns * o + stage] = setupCtx.starkInfo.mapSectionsN[section];
                    nColsStagesAcc[ns * o + stage] = nColsStagesAcc[ns * o + stage - 1] + nColsStages[stage - 1];
                }
                else
                {
                    uint64_t index = stage - setupCtx.starkInfo.nStages - 2;
                    std::string section = setupCtx.starkInfo.customCommits[index].name + "0";
                    nColsStages[ns * o + stage] = setupCtx.starkInfo.mapSectionsN[section];
                    nColsStagesAcc[ns * o + stage] = nColsStagesAcc[ns * o + stage - 1] + nColsStages[stage - 1];
                }
            }
        }
        nColsStagesAcc[ns * nOpenings] = nColsStagesAcc[ns * nOpenings - 1] + nColsStages[ns * nOpenings - 1];

        uint64_t Nexteded = 1 << setupCtx.starkInfo.starkStruct.nBitsExt;
        uint64_t N = 1 << setupCtx.starkInfo.starkStruct.nBits;
        cudaMalloc(&h_deviceArgs.nextStrides, nOpenings * sizeof(uint64_t));
        cudaMalloc(&h_deviceArgs.nColsStages, (ns * nOpenings + 1) * sizeof(uint64_t));
        cudaMalloc(&h_deviceArgs.nColsStagesAcc, (ns * nOpenings + 1) * sizeof(uint64_t));
        cudaMalloc(&h_deviceArgs.offsetsStages, (ns * nOpenings + 1) * sizeof(uint64_t));
        cudaMalloc(&h_deviceArgs.cmPolsInfo, setupCtx.starkInfo.cmPolsMap.size() * 3 * sizeof(uint64_t));
        cudaMalloc(&h_deviceArgs.zi, setupCtx.starkInfo.boundaries.size() * Nexteded * sizeof(Goldilocks::Element)); // rick: pillar calculada abans
        cudaMalloc(&h_deviceArgs.x_n, N * sizeof(Goldilocks::Element));
        cudaMalloc(&h_deviceArgs.x_2ns, Nexteded * sizeof(Goldilocks::Element));
        cudaMalloc(&h_deviceArgs.challenges, setupCtx.starkInfo.challengesMap.size() * FIELD_EXTENSION * sizeof(Goldilocks::Element));
        uint64_t nNumbers = setupCtx.expressionsBin.expressionsBinArgsExpressions.nNumbers;
        cudaMalloc(&h_deviceArgs.numbers, nNumbers * sizeof(Goldilocks::Element));
        cudaMalloc(&h_deviceArgs.publics, setupCtx.starkInfo.nPublics * sizeof(Goldilocks::Element));
        cudaMalloc(&h_deviceArgs.evals, setupCtx.starkInfo.evMap.size() * FIELD_EXTENSION * sizeof(Goldilocks::Element));
        cudaMalloc(&h_deviceArgs.airgroupValues, setupCtx.starkInfo.airgroupValuesMap.size() * FIELD_EXTENSION * sizeof(Goldilocks::Element));
        cudaMalloc(&h_deviceArgs.airValues, setupCtx.starkInfo.airValuesMap.size() * FIELD_EXTENSION * sizeof(Goldilocks::Element));
        cudaMalloc(&h_deviceArgs.ops, setupCtx.expressionsBin.expressionsBinArgsExpressions.nOpsTotal * sizeof(uint8_t));
        cudaMalloc(&h_deviceArgs.args, setupCtx.expressionsBin.expressionsBinArgsExpressions.nArgsTotal * sizeof(uint16_t));

        uint64_t nCols1 = nColsStagesAcc[ns * nOpenings] + setupCtx.starkInfo.boundaries.size() + 1;
        uint64_t nCols2 = nColsStagesAcc[ns * nOpenings] + nOpenings * FIELD_EXTENSION;
        uint64_t nCols3 = nColsStagesAcc[ns * nOpenings] + 1;
        uint64_t nColsMax = max(nCols1, max(nCols2, nCols3));

        // bufferT_
        deviceArgs.bufferSize = nOpenings * nrowsPack * nColsMax; // this must be moved from here
        cudaMalloc(&h_deviceArgs.bufferT_, nBlocks * deviceArgs.bufferSize * sizeof(Goldilocks::Element));
        std::cout << "Total memory in expressions buffers [Gb]: " << (1.0 * nBlocks * deviceArgs.bufferSize * sizeof(Goldilocks::Element)) / (1024.0 * 1024.0 * 1024.0) << std::endl;

        // destVals
        deviceArgs.destValsSize = nParamsMax * FIELD_EXTENSION * nrowsPack;
        cudaMalloc(&h_deviceArgs.destVals, nBlocks * deviceArgs.destValsSize * sizeof(Goldilocks::Element));
        std::cout << "Total memory in expressions destVals [Gb]: " << (1.0 * nBlocks * deviceArgs.destValsSize * sizeof(Goldilocks::Element)) / (1024.0 * 1024.0 * 1024.0) << std::endl;

        // tmps
        deviceArgs.tmp1Size = nTemp1Max * nrowsPack;
        deviceArgs.tmp3Size = nTemp3Max * FIELD_EXTENSION * nrowsPack;
        cudaMalloc(&h_deviceArgs.tmp1, nBlocks * deviceArgs.tmp1Size * sizeof(Goldilocks::Element *));
        cudaMalloc(&h_deviceArgs.tmp3, nBlocks * deviceArgs.tmp3Size * sizeof(Goldilocks::Element *));
        std::cout << "Total memory in expressions tmp1 [Gb]: " << (1.0 * nBlocks * deviceArgs.tmp1Size * sizeof(Goldilocks::Element)) / (1024.0 * 1024.0 * 1024.0) << std::endl;
        std::cout << "Total memory in expressions tmp3 [Gb]: " << (1.0 * nBlocks * deviceArgs.tmp3Size * sizeof(Goldilocks::Element)) / (1024.0 * 1024.0 * 1024.0) << std::endl;
    };

    ~ExpressionsGPU()
    {
        cudaFree(h_deviceArgs.nextStrides);
        cudaFree(h_deviceArgs.nColsStages);
        cudaFree(h_deviceArgs.nColsStagesAcc);
        cudaFree(h_deviceArgs.offsetsStages);
        cudaFree(h_deviceArgs.cmPolsInfo);
        cudaFree(h_deviceArgs.zi);
        cudaFree(h_deviceArgs.x_n);
        cudaFree(h_deviceArgs.x_2ns);
        cudaFree(h_deviceArgs.challenges);
        cudaFree(h_deviceArgs.numbers);
        cudaFree(h_deviceArgs.publics);
        cudaFree(h_deviceArgs.evals);
        cudaFree(h_deviceArgs.airgroupValues);
        cudaFree(h_deviceArgs.airValues);
        cudaFree(h_deviceArgs.ops);
        cudaFree(h_deviceArgs.args);
        cudaFree(h_deviceArgs.bufferT_);
        cudaFree(h_deviceArgs.destVals);
        cudaFree(h_deviceArgs.tmp1);
        cudaFree(h_deviceArgs.tmp3);
    }

    void setBufferTInfo(uint64_t domainSize, StepsParams &params, ParserArgs &parserArgs, std::vector<Dest> &dests)
    {

        bool domainExtended = domainSize == uint64_t(1 << setupCtx.starkInfo.starkStruct.nBitsExt) ? true : false;
        uint64_t expId = dests[0].params[0].op == opType::tmp ? dests[0].params[0].parserParams.destDim : 0;

        uint64_t nOpenings = setupCtx.starkInfo.openingPoints.size();
        uint64_t ns = 2 + setupCtx.starkInfo.nStages + setupCtx.starkInfo.customCommits.size();
        offsetsStages.resize(ns * nOpenings + 1);
        nColsStages.clear();
        nColsStages.resize(ns * nOpenings + 1);
        nColsStagesAcc.clear();
        nColsStagesAcc.resize(ns * nOpenings + 1);

        for (uint64_t o = 0; o < nOpenings; ++o)
        {
            for (uint64_t stage = 0; stage < ns; ++stage)
            {
                if (stage == 0)
                {
                    offsetsStages[ns * o] = 0;
                    nColsStages[ns * o] = setupCtx.starkInfo.mapSectionsN["const"];
                    nColsStagesAcc[ns * o] = o == 0 ? 0 : nColsStagesAcc[ns * o + stage - 1] + nColsStages[stage - 1];
                }
                else if (stage < 2 + setupCtx.starkInfo.nStages)
                {
                    std::string section = "cm" + to_string(stage);
                    offsetsStages[ns * o + stage] = setupCtx.starkInfo.mapOffsets[std::make_pair(section, domainExtended)];
                    nColsStages[ns * o + stage] = setupCtx.starkInfo.mapSectionsN[section];
                    nColsStagesAcc[ns * o + stage] = nColsStagesAcc[ns * o + stage - 1] + nColsStages[stage - 1];
                }
                else
                {
                    uint64_t index = stage - setupCtx.starkInfo.nStages - 2;
                    std::string section = setupCtx.starkInfo.customCommits[index].name + "0";
                    offsetsStages[ns * o + stage] = setupCtx.starkInfo.mapOffsets[std::make_pair(section, domainExtended)];
                    nColsStages[ns * o + stage] = setupCtx.starkInfo.mapSectionsN[section];
                    nColsStagesAcc[ns * o + stage] = nColsStagesAcc[ns * o + stage - 1] + nColsStages[stage - 1];
                }
            }
        }

        nColsStagesAcc[ns * nOpenings] = nColsStagesAcc[ns * nOpenings - 1] + nColsStages[ns * nOpenings - 1];
        if (expId == int64_t(setupCtx.starkInfo.cExpId))
        {
            nCols = nColsStagesAcc[ns * nOpenings] + setupCtx.starkInfo.boundaries.size() + 1;
        }
        else if (expId == int64_t(setupCtx.starkInfo.friExpId))
        {
            nCols = nColsStagesAcc[ns * nOpenings] + nOpenings * FIELD_EXTENSION;
        }
        else
        {
            nCols = nColsStagesAcc[ns * nOpenings] + 1;
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
        for (uint64_t i = 0; i < nOpenings; ++i)
        {
            uint64_t opening = setupCtx.starkInfo.openingPoints[i] < 0 ? setupCtx.starkInfo.openingPoints[i] + domainSize : setupCtx.starkInfo.openingPoints[i];
            deviceArgs.nextStrides[i] = opening * extend;
        }
        deviceArgs.nColsStages = new uint64_t[nColsStages.size()];
        for (uint64_t i = 0; i < nColsStages.size(); ++i)
        {
            deviceArgs.nColsStages[i] = nColsStages[i];
        }
        deviceArgs.nColsStagesAcc = new uint64_t[nColsStagesAcc.size()];
        for (uint64_t i = 0; i < nColsStagesAcc.size(); ++i)
        {
            deviceArgs.nColsStagesAcc[i] = nColsStagesAcc[i];
        }
        deviceArgs.offsetsStages = new uint64_t[offsetsStages.size()];
        for (uint64_t i = 0; i < offsetsStages.size(); ++i)
        {
            deviceArgs.offsetsStages[i] = offsetsStages[i];
        }
        deviceArgs.constPols = domainExtended ? &params.pConstPolsExtendedTreeAddress[2] : params.pConstPolsAddress;
        deviceArgs.constPolsSize = setupCtx.starkInfo.nConstants;
        deviceArgs.cmPolsInfoSize = setupCtx.starkInfo.cmPolsMap.size();
        deviceArgs.cmPolsInfo = new uint64_t[deviceArgs.cmPolsInfoSize * 3];
        for (uint64_t i = 0; i < deviceArgs.cmPolsInfoSize; ++i)
        {
            deviceArgs.cmPolsInfo[i * 3] = setupCtx.starkInfo.cmPolsMap[i].stage;
            deviceArgs.cmPolsInfo[i * 3 + 1] = setupCtx.starkInfo.cmPolsMap[i].stagePos;
            deviceArgs.cmPolsInfo[i * 3 + 2] = setupCtx.starkInfo.cmPolsMap[i].dim;
        }
        if (dests[0].params[0].parserParams.expId == int64_t(setupCtx.starkInfo.cExpId))
        {
            deviceArgs.expType = 0;
        }
        else if (dests[0].params[0].parserParams.expId == int64_t(setupCtx.starkInfo.friExpId))
        {
            deviceArgs.expType = 1;
        }
        else
        {
            deviceArgs.expType = 2;
        }

        deviceArgs.boundSize = setupCtx.starkInfo.boundaries.size();
        deviceArgs.zi = setupCtx.proverHelpers.zi;
        deviceArgs.x_n = setupCtx.proverHelpers.x_n;
        deviceArgs.x_2ns = setupCtx.proverHelpers.x_2ns;
        deviceArgs.xDivXSub = params.xDivXSub;
        deviceArgs.trace = params.trace;
        deviceArgs.aux_trace = params.aux_trace;

        // Dests
        deviceArgs.dests = new DestGPU[dests.size()];
        deviceArgs.nDests = dests.size();
        for (uint64_t i = 0; i < dests.size(); ++i)
        {
            deviceArgs.dests[i].dest = dests[i].dest;
            deviceArgs.dests[i].dest_gpu = dests[i].dest_gpu;
            deviceArgs.dests[i].offset = dests[i].offset;
            deviceArgs.dests[i].dim = dests[i].dim;
            deviceArgs.dests[i].nParams = dests[i].params.size();
            assert(deviceArgs.dests[i].nParams <= nParamsMax);
            deviceArgs.dests[i].params = new ParamsGPU[dests[i].params.size()];

            for (uint64_t j = 0; j < deviceArgs.dests[i].nParams; ++j)
            {
                deviceArgs.dests[i].params[j].dim = dests[i].params[j].dim;
                deviceArgs.dests[i].params[j].stage = dests[i].params[j].stage;
                deviceArgs.dests[i].params[j].stagePos = dests[i].params[j].stagePos;
                deviceArgs.dests[i].params[j].polsMapId = dests[i].params[j].polsMapId;
                deviceArgs.dests[i].params[j].rowOffsetIndex = dests[i].params[j].rowOffsetIndex;
                deviceArgs.dests[i].params[j].inverse = dests[i].params[j].inverse;
                deviceArgs.dests[i].params[j].batch = dests[i].params[j].batch;
                deviceArgs.dests[i].params[j].op = dests[i].params[j].op;
                deviceArgs.dests[i].params[j].value = dests[i].params[j].value;
                deviceArgs.dests[i].params[j].parserParams.stage = dests[i].params[j].parserParams.stage;
                deviceArgs.dests[i].params[j].parserParams.expId = dests[i].params[j].parserParams.expId;
                deviceArgs.dests[i].params[j].parserParams.nTemp1 = dests[i].params[j].parserParams.nTemp1;
                assert(deviceArgs.dests[i].params[j].parserParams.nTemp1 < nTemp1Max);
                deviceArgs.dests[i].params[j].parserParams.nTemp3 = dests[i].params[j].parserParams.nTemp3;
                assert(deviceArgs.dests[i].params[j].parserParams.nTemp3 < nTemp3Max);
                deviceArgs.dests[i].params[j].parserParams.nOps = dests[i].params[j].parserParams.nOps;
                deviceArgs.dests[i].params[j].parserParams.opsOffset = dests[i].params[j].parserParams.opsOffset;
                deviceArgs.dests[i].params[j].parserParams.nArgs = dests[i].params[j].parserParams.nArgs;
                deviceArgs.dests[i].params[j].parserParams.argsOffset = dests[i].params[j].parserParams.argsOffset;
                deviceArgs.dests[i].params[j].parserParams.constPolsOffset = dests[i].params[j].parserParams.constPolsOffset;
                deviceArgs.dests[i].params[j].parserParams.cmPolsOffset = dests[i].params[j].parserParams.cmPolsOffset;
                deviceArgs.dests[i].params[j].parserParams.challengesOffset = dests[i].params[j].parserParams.challengesOffset;
                deviceArgs.dests[i].params[j].parserParams.publicsOffset = dests[i].params[j].parserParams.publicsOffset;
                deviceArgs.dests[i].params[j].parserParams.airgroupValuesOffset = dests[i].params[j].parserParams.airgroupValuesOffset;
                deviceArgs.dests[i].params[j].parserParams.airValuesOffset = dests[i].params[j].parserParams.airValuesOffset;
                deviceArgs.dests[i].params[j].parserParams.firstRow = dests[i].params[j].parserParams.firstRow;
                deviceArgs.dests[i].params[j].parserParams.lastRow = dests[i].params[j].parserParams.lastRow;
                deviceArgs.dests[i].params[j].parserParams.destDim = dests[i].params[j].parserParams.destDim;
                deviceArgs.dests[i].params[j].parserParams.destId = dests[i].params[j].parserParams.destId;
                deviceArgs.dests[i].params[j].parserParams.imPol = dests[i].params[j].parserParams.imPol;
            }
        }
        // non polnomial arguments
        deviceArgs.nChallenges = setupCtx.starkInfo.challengesMap.size();
        deviceArgs.challenges = params.challenges;
        deviceArgs.nNumbers = parserArgs.nNumbers;
        deviceArgs.numbers = (Goldilocks::Element *)parserArgs.numbers;
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

        // bufferT_
        deviceArgs.nBlocks = nBlocks;
    }

    inline void loadPolynomials(Goldilocks::Element *bufferT_, uint64_t row)
    {

        uint64_t nOpenings = deviceArgs.nOpenings;
        uint64_t ns = deviceArgs.ns;
        bool domainExtended = deviceArgs.domainExtended;
        uint64_t domainSize = deviceArgs.domainSize;
        uint64_t nrowsPack = deviceArgs.nrowsPack;
        Goldilocks::Element *constPols = deviceArgs.constPols;
        uint64_t constPolsSize = deviceArgs.constPolsSize;
        uint64_t *nextStrides = deviceArgs.nextStrides;
        uint64_t *nColsStages = deviceArgs.nColsStages;
        uint64_t *nColsStagesAcc = deviceArgs.nColsStagesAcc;
        uint64_t *offsetsStages = deviceArgs.offsetsStages;
        uint64_t cmPolsInfoSize = deviceArgs.cmPolsInfoSize;
        uint64_t *cmPolsInfo = deviceArgs.cmPolsInfo;
        Goldilocks::Element *trace = deviceArgs.trace;
        Goldilocks::Element *aux_trace = deviceArgs.aux_trace;
        Goldilocks::Element *zi = deviceArgs.zi;
        Goldilocks::Element *x_n = deviceArgs.x_n;
        Goldilocks::Element *x_2ns = deviceArgs.x_2ns;
        Goldilocks::Element *xDivXSub = deviceArgs.xDivXSub;

        for (uint64_t k = 0; k < constPolsSize; ++k)
        {
            for (uint64_t o = 0; o < nOpenings; ++o)
            {
                for (uint64_t j = 0; j < nrowsPack; ++j)
                {
                    uint64_t l = (row + j + nextStrides[o]) % domainSize;
                    bufferT_[(nColsStagesAcc[ns * o] + k) * nrowsPack + j] = constPols[l * nColsStages[0] + k];
                }
            }
        }

        for (uint64_t k = 0; k < cmPolsInfoSize; ++k)
        {
            uint64_t stage = cmPolsInfo[k * 3];
            uint64_t stagePos = cmPolsInfo[k * 3 + 1];
            for (uint64_t d = 0; d < cmPolsInfo[k * 3 + 2]; ++d)
            {
                for (uint64_t o = 0; o < nOpenings; ++o)
                {
                    for (uint64_t j = 0; j < nrowsPack; ++j)
                    {
                        uint64_t l = (row + j + nextStrides[o]) % domainSize;
                        if (stage == 1 && !domainExtended)
                        {
                            bufferT_[(nColsStagesAcc[ns * o + stage] + (stagePos + d)) * nrowsPack + j] = trace[l * nColsStages[stage] + stagePos + d];
                        }
                        else
                        {
                            bufferT_[(nColsStagesAcc[ns * o + stage] + (stagePos + d)) * nrowsPack + j] = aux_trace[offsetsStages[stage] + l * nColsStages[stage] + stagePos + d];
                        }
                    }
                }
            }
        }
        if (deviceArgs.expType == 0)
        {
            for (uint64_t d = 0; d < deviceArgs.boundSize; ++d)
            {
                for (uint64_t j = 0; j < nrowsPack; ++j)
                {
                    bufferT_[(nColsStagesAcc[ns * nOpenings] + d + 1) * nrowsPack + j] = zi[row + j + d * domainSize];
                }
            }
            for (uint64_t j = 0; j < nrowsPack; ++j)
            {
                bufferT_[(nColsStagesAcc[ns * nOpenings]) * nrowsPack + j] = x_2ns[row + j];
            }
        }
        else if (deviceArgs.expType == 1)
        {
            for (uint64_t d = 0; d < nOpenings; ++d)
            {
                for (uint64_t k = 0; k < FIELD_EXTENSION; ++k)
                {
                    for (uint64_t j = 0; j < nrowsPack; ++j)
                    {
                        bufferT_[(nColsStagesAcc[ns * nOpenings] + d * FIELD_EXTENSION + k) * nrowsPack + j] = xDivXSub[(row + j + d * domainSize) * FIELD_EXTENSION + k];
                    }
                }
            }
        }
        else
        {
            for (uint64_t j = 0; j < nrowsPack; ++j)
            {
                bufferT_[(nColsStagesAcc[ns * nOpenings]) * nrowsPack + j] = x_n[row + j];
            }
        }
    }

    inline void printTmp1(uint64_t row, Goldilocks::Element *tmp)
    {
        Goldilocks::Element buff[nrowsPack];
        Goldilocks::copy_pack(nrowsPack, buff, tmp, false);
        for (uint64_t i = 0; i < nrowsPack; ++i)
        {
            cout << "Value at row " << row + i << " is " << Goldilocks::toString(buff[i]) << endl;
        }
    }

    inline void printTmp3(uint64_t row, Goldilocks::Element *tmp)
    {
        for (uint64_t i = 0; i < nrowsPack; ++i)
        {
            cout << "Value at row " << row + i << " is [" << Goldilocks::toString(tmp[i]) << ", " << Goldilocks::toString(tmp[nrowsPack + i]) << ", " << Goldilocks::toString(tmp[2 * nrowsPack + i]) << "]" << endl;
        }
    }

    void calculateExpressions_gpu(StepsParams &params, StepsParams &params_gpu, ParserArgs &parserArgs, std::vector<Dest> dests, uint64_t domainSize)
    {

        CHECKCUDAERR(cudaDeviceSynchronize());
        double time = omp_get_wtime();
        setBufferTInfo(domainSize, params, parserArgs, dests);
        CHECKCUDAERR(cudaDeviceSynchronize());
        time = omp_get_wtime() - time;
        std::cout << "goal2_ setBufferTInfo time: " << time << std::endl;

        CHECKCUDAERR(cudaDeviceSynchronize());
        time = omp_get_wtime();
        loadDeviceArguments(params_gpu);
        CHECKCUDAERR(cudaDeviceSynchronize());
        time = omp_get_wtime() - time;
        std::cout << "goal2_ cudaMalloc expressions time: " << time << std::endl;

        CHECKCUDAERR(cudaDeviceSynchronize());
        time = omp_get_wtime();
        dim3 nBlocks = deviceArgs.nBlocks;
        dim3 nThreads = deviceArgs.nrowsPack;
        std::cout << "goal2_ nBlocks: " << nBlocks.x << std::endl;
        computeExpressions_<<<nBlocks, nThreads>>>(d_deviceArgs);
        CHECKCUDAERR(cudaDeviceSynchronize());
        time = omp_get_wtime() - time;
        std::cout << "goal2_ de computeExpressions: " << time << std::endl;

        CHECKCUDAERR(cudaDeviceSynchronize());
        time = omp_get_wtime();
        for (uint32_t i = 0; i < deviceArgs.nDests; ++i)
        {
            cudaMemcpy(dests[i].dest, deviceArgs.dests[i].dest_gpu, domainSize * FIELD_EXTENSION * sizeof(Goldilocks::Element), cudaMemcpyDeviceToHost);
        }
        CHECKCUDAERR(cudaDeviceSynchronize());
        time = omp_get_wtime() - time;
        std::cout << "goal2_ de cudaMemcpy dests time: " << time << std::endl;

        CHECKCUDAERR(cudaDeviceSynchronize());
        time = omp_get_wtime();
        freeDeviceArguments();
        CHECKCUDAERR(cudaDeviceSynchronize());
        time = omp_get_wtime() - time;
        std::cout << "goal2_ freeDeviceArguments time: " << time << std::endl;
    }
    void calculateExpressions_gpu2(StepsParams &params, StepsParams &params_gpu, ParserArgs &parserArgs, std::vector<Dest> dests, uint64_t domainSize)
    {

        CHECKCUDAERR(cudaDeviceSynchronize());
        double time = omp_get_wtime();
        setBufferTInfo(domainSize, params, parserArgs, dests);
        CHECKCUDAERR(cudaDeviceSynchronize());
        time = omp_get_wtime() - time;
        std::cout << "goal2 setBufferTInfo time: " << time << std::endl;

        CHECKCUDAERR(cudaDeviceSynchronize());
        time = omp_get_wtime();
        loadDeviceArguments(params_gpu);
        CHECKCUDAERR(cudaDeviceSynchronize());
        time = omp_get_wtime() - time;
        std::cout << "goal2 cudaMalloc expressions time: " << time << std::endl;

        CHECKCUDAERR(cudaDeviceSynchronize());
        time = omp_get_wtime();
        dim3 nBlocks = deviceArgs.nBlocks;
        dim3 nThreads = deviceArgs.nrowsPack;
        computeExpressions_<<<nBlocks, nThreads>>>(d_deviceArgs);
        CHECKCUDAERR(cudaDeviceSynchronize());
        time = omp_get_wtime() - time;
        std::cout << "goal2 nBlocks aqui: " << nBlocks.x << std::endl;
        std::cout << "goal2 despres de computeExpressions 2: " << time << std::endl;

        CHECKCUDAERR(cudaDeviceSynchronize());
        time = omp_get_wtime();
        freeDeviceArguments();
        CHECKCUDAERR(cudaDeviceSynchronize());
        time = omp_get_wtime() - time;
        std::cout << "goal2 freeDeviceArguments time: " << time << std::endl;
    }

    void loadDeviceArguments(StepsParams &params_gpu)
    {
        double time = omp_get_wtime();
        cudaMemcpy(h_deviceArgs.nextStrides, deviceArgs.nextStrides, deviceArgs.nOpenings * sizeof(uint64_t), cudaMemcpyHostToDevice);
        cudaMemcpy(h_deviceArgs.nColsStages, deviceArgs.nColsStages, nColsStages.size() * sizeof(uint64_t), cudaMemcpyHostToDevice);
        cudaMemcpy(h_deviceArgs.nColsStagesAcc, deviceArgs.nColsStagesAcc, nColsStagesAcc.size() * sizeof(uint64_t), cudaMemcpyHostToDevice);
        cudaMemcpy(h_deviceArgs.offsetsStages, deviceArgs.offsetsStages, offsetsStages.size() * sizeof(uint64_t), cudaMemcpyHostToDevice);
        cudaMemcpy(h_deviceArgs.cmPolsInfo, deviceArgs.cmPolsInfo, 3 * deviceArgs.cmPolsInfoSize * sizeof(uint64_t), cudaMemcpyHostToDevice);
        cudaMemcpy(h_deviceArgs.zi, deviceArgs.zi, deviceArgs.boundSize * deviceArgs.NExtended * sizeof(Goldilocks::Element), cudaMemcpyHostToDevice); // cal copiar cada cop?
        cudaMemcpy(h_deviceArgs.x_n, deviceArgs.x_n, deviceArgs.N * sizeof(Goldilocks::Element), cudaMemcpyHostToDevice);                              // cal cada cop? no es pot transportar?
        cudaMemcpy(h_deviceArgs.x_2ns, deviceArgs.x_2ns, deviceArgs.NExtended * sizeof(Goldilocks::Element), cudaMemcpyHostToDevice);                  // cal cada cop? no es pot transportar?
        cudaMemcpy(h_deviceArgs.challenges, deviceArgs.challenges, deviceArgs.nChallenges * FIELD_EXTENSION * sizeof(Goldilocks::Element), cudaMemcpyHostToDevice);
        cudaMemcpy(h_deviceArgs.numbers, deviceArgs.numbers, deviceArgs.nNumbers * sizeof(Goldilocks::Element), cudaMemcpyHostToDevice);
        cudaMemcpy(h_deviceArgs.publics, deviceArgs.publics, deviceArgs.nPublics * sizeof(Goldilocks::Element), cudaMemcpyHostToDevice);
        cudaMemcpy(h_deviceArgs.evals, deviceArgs.evals, deviceArgs.nEvals * FIELD_EXTENSION * sizeof(Goldilocks::Element), cudaMemcpyHostToDevice);
        cudaMemcpy(h_deviceArgs.airgroupValues, deviceArgs.airgroupValues, deviceArgs.nAirgroupValues * FIELD_EXTENSION * sizeof(Goldilocks::Element), cudaMemcpyHostToDevice);
        cudaMemcpy(h_deviceArgs.airValues, deviceArgs.airValues, deviceArgs.nAirValues * FIELD_EXTENSION * sizeof(Goldilocks::Element), cudaMemcpyHostToDevice);
        cudaMemcpy(h_deviceArgs.ops, deviceArgs.ops, deviceArgs.nOpsTotal * sizeof(uint8_t), cudaMemcpyHostToDevice);
        cudaMemcpy(h_deviceArgs.args, deviceArgs.args, deviceArgs.nArgsTotal * sizeof(uint16_t), cudaMemcpyHostToDevice);

        time = omp_get_wtime() - time;
        std::cout << "goal2 cudaMalloc cudaMemcpy host to device time: " << time << std::endl;

        // Dests
        time = omp_get_wtime();
        d_dests = new DestGPU[deviceArgs.nDests];
        for (int i = 0; i < deviceArgs.nDests; ++i)
        {
            d_dests[i].dest = deviceArgs.dests[i].dest;
            d_dests[i].dest_gpu = deviceArgs.dests[i].dest_gpu;
            d_dests[i].offset = deviceArgs.dests[i].offset;
            d_dests[i].dim = deviceArgs.dests[i].dim;
            d_dests[i].nParams = deviceArgs.dests[i].nParams;
            cudaMalloc(&d_dests[i].params, d_dests[i].nParams * sizeof(ParamsGPU));
            cudaMemcpy(d_dests[i].params, deviceArgs.dests[i].params, d_dests[i].nParams * sizeof(ParamsGPU), cudaMemcpyHostToDevice);
        }
        DestGPU *d_dests_;
        cudaMalloc(&d_dests_, deviceArgs.nDests * sizeof(DestGPU));
        cudaMemcpy(d_dests_, d_dests, deviceArgs.nDests * sizeof(DestGPU), cudaMemcpyHostToDevice);

        time = omp_get_wtime() - time;
        std::cout << "goal2 cudaMalloc dests: " << time << std::endl;

        // Update the device struct with device pointers
        h_deviceArgs.N = deviceArgs.N;
        h_deviceArgs.NExtended = deviceArgs.NExtended;
        h_deviceArgs.domainSize = deviceArgs.domainSize;
        h_deviceArgs.nrowsPack = deviceArgs.nrowsPack;
        h_deviceArgs.nCols = deviceArgs.nCols;
        h_deviceArgs.nOpenings = deviceArgs.nOpenings;
        h_deviceArgs.ns = deviceArgs.ns;
        h_deviceArgs.domainExtended = deviceArgs.domainExtended;
        h_deviceArgs.constPolsSize = deviceArgs.constPolsSize;
        h_deviceArgs.cmPolsInfoSize = deviceArgs.cmPolsInfoSize;
        h_deviceArgs.expType = deviceArgs.expType;
        h_deviceArgs.boundSize = deviceArgs.boundSize;
        h_deviceArgs.nChallenges = deviceArgs.nChallenges;
        h_deviceArgs.nNumbers = deviceArgs.nNumbers;
        h_deviceArgs.nPublics = deviceArgs.nPublics;
        h_deviceArgs.nEvals = deviceArgs.nEvals;
        h_deviceArgs.nAirgroupValues = deviceArgs.nAirgroupValues;
        h_deviceArgs.nAirValues = deviceArgs.nAirValues;
        h_deviceArgs.nDests = deviceArgs.nDests;
        h_deviceArgs.nOpsTotal = deviceArgs.nOpsTotal;
        h_deviceArgs.nArgsTotal = deviceArgs.nArgsTotal;
        h_deviceArgs.nBlocks = deviceArgs.nBlocks;
        h_deviceArgs.bufferSize = deviceArgs.bufferSize;
        h_deviceArgs.destValsSize = deviceArgs.destValsSize;
        h_deviceArgs.tmp1Size = deviceArgs.tmp1Size;
        h_deviceArgs.tmp3Size = deviceArgs.tmp3Size;

        h_deviceArgs.constPols = h_deviceArgs.domainExtended ? params_gpu.pConstPolsExtendedTreeAddress : params_gpu.pConstPolsAddress;
        h_deviceArgs.trace = params_gpu.trace;
        h_deviceArgs.aux_trace = params_gpu.aux_trace;
        h_deviceArgs.xDivXSub = params_gpu.xDivXSub;

        h_deviceArgs.dests = d_dests_;

        // Allocate memory for the struct on the device
        cudaMalloc(&d_deviceArgs, sizeof(DeviceArguments));
        cudaMemcpy(d_deviceArgs, &h_deviceArgs, sizeof(DeviceArguments), cudaMemcpyHostToDevice);
    }
    void freeDeviceArguments()
    {
        for (int i = 0; i < deviceArgs.nDests; ++i)
        {
            cudaFree(d_dests[i].params);
        }
        cudaFree(h_deviceArgs.dests);
        cudaFree(d_deviceArgs);
    }
};

__device__ void storeOnePolynomial__(DeviceArguments *d_deviceArgs, gl64_t *destVals, uint64_t row, uint32_t idest)
{

    if (d_deviceArgs->dests[idest].dim == 1)
    {
        uint64_t offset = d_deviceArgs->dests[idest].offset != 0 ? d_deviceArgs->dests[idest].offset : 1;
        gl64_t::copy_gpu((gl64_t *)&d_deviceArgs->dests[idest].dest_gpu[row * offset], uint64_t(offset), &destVals[0], false);
    }
    else
    {
        uint64_t offset = d_deviceArgs->dests[idest].offset != 0 ? d_deviceArgs->dests[idest].offset : FIELD_EXTENSION;
        gl64_t::copy_gpu((gl64_t *)&d_deviceArgs->dests[idest].dest_gpu[row * offset], uint64_t(offset), &destVals[0], false);
        gl64_t::copy_gpu((gl64_t *)&d_deviceArgs->dests[idest].dest_gpu[row * offset + 1], uint64_t(offset), &destVals[d_deviceArgs->nrowsPack], false);
        gl64_t::copy_gpu((gl64_t *)&d_deviceArgs->dests[idest].dest_gpu[row * offset + 2], uint64_t(offset), &destVals[2 * d_deviceArgs->nrowsPack], false);
    }
}

__global__ __launch_bounds__(128, 4) void copyPolynomial_(DeviceArguments *d_deviceArgs, Goldilocks::Element *d_destVals, bool inverse, uint64_t dim, Goldilocks::Element *d_tmp)
{
    copyPolynomial__(d_deviceArgs, (gl64_t *)d_destVals, inverse, dim, (gl64_t *)d_tmp);
}
__device__ void copyPolynomial__(DeviceArguments *d_deviceArgs, gl64_t *destVals, bool inverse, uint64_t dim, gl64_t *temp)
{
    int idx = threadIdx.x;
    if (dim == 1)
    {
        if (inverse)
        {
            destVals[idx] = temp[idx].reciprocal();
        }
        else
        {
            destVals[idx] = temp[idx];
        }
    }
    else if (dim == FIELD_EXTENSION)
    {
        if (inverse)
        {
            Goldilocks3GPU::Element aux;
            aux[0] = temp[idx];
            aux[1] = temp[d_deviceArgs->nrowsPack + idx];
            aux[2] = temp[2 * d_deviceArgs->nrowsPack + idx];
            Goldilocks3GPU::inv(aux, aux);
            destVals[idx] = aux[0];
            destVals[d_deviceArgs->nrowsPack + idx] = aux[1];
            destVals[2 * d_deviceArgs->nrowsPack + idx] = aux[2];
        }
        else
        {
            destVals[idx] = temp[idx];
            destVals[d_deviceArgs->nrowsPack + idx] = temp[d_deviceArgs->nrowsPack + idx];
            destVals[2 * d_deviceArgs->nrowsPack + idx] = temp[2 * d_deviceArgs->nrowsPack + idx];
        }
    }
}
__global__ __launch_bounds__(128, 4) void loadPolynomials_(DeviceArguments *d_deviceArgs, uint64_t row, uint32_t iBlock)
{

    loadPolynomials__(d_deviceArgs, row, iBlock);
}

__device__ void loadPolynomials__(DeviceArguments *d_deviceArgs, uint64_t row, uint32_t iBlock)
{

    uint64_t row_loc = threadIdx.x;
    uint64_t nOpenings = d_deviceArgs->nOpenings;
    uint64_t ns = d_deviceArgs->ns;
    bool domainExtended = d_deviceArgs->domainExtended;
    uint64_t domainSize = d_deviceArgs->domainSize;
    uint64_t nrowsPack = d_deviceArgs->nrowsPack;
    Goldilocks::Element *constPols = domainExtended ? &d_deviceArgs->constPols[2] : d_deviceArgs->constPols;
    uint64_t constPolsSize = d_deviceArgs->constPolsSize;
    uint64_t *nextStrides = d_deviceArgs->nextStrides;
    uint64_t *nColsStages = d_deviceArgs->nColsStages;
    uint64_t *nColsStagesAcc = d_deviceArgs->nColsStagesAcc;
    uint64_t *offsetsStages = d_deviceArgs->offsetsStages;
    uint64_t cmPolsInfoSize = d_deviceArgs->cmPolsInfoSize;
    uint64_t *cmPolsInfo = d_deviceArgs->cmPolsInfo;
    Goldilocks::Element *trace = d_deviceArgs->trace;
    Goldilocks::Element *aux_trace = d_deviceArgs->aux_trace;
    Goldilocks::Element *zi = d_deviceArgs->zi;
    Goldilocks::Element *x_n = d_deviceArgs->x_n;
    Goldilocks::Element *x_2ns = d_deviceArgs->x_2ns;
    Goldilocks::Element *xDivXSub = d_deviceArgs->xDivXSub;
    Goldilocks::Element *d_bufferT_ = &d_deviceArgs->bufferT_[iBlock * d_deviceArgs->bufferSize];

    for (uint64_t k = 0; k < constPolsSize; ++k)
    {
        for (uint64_t o = 0; o < nOpenings; ++o)
        {
            uint64_t l = (row + row_loc + nextStrides[o]) % domainSize;
            d_bufferT_[(nColsStagesAcc[ns * o] + k) * nrowsPack + row_loc] = constPols[l * nColsStages[0] + k];
        }
    }

    for (uint64_t k = 0; k < cmPolsInfoSize; ++k)
    {
        uint64_t stage = cmPolsInfo[k * 3];
        uint64_t stagePos = cmPolsInfo[k * 3 + 1];
        for (uint64_t d = 0; d < cmPolsInfo[k * 3 + 2]; ++d)
        {
            for (uint64_t o = 0; o < nOpenings; ++o)
            {
                uint64_t l = (row + row_loc + nextStrides[o]) % domainSize;
                if (stage == 1 && !domainExtended)
                {
                    d_bufferT_[(nColsStagesAcc[ns * o + stage] + (stagePos + d)) * nrowsPack + row_loc] = trace[l * nColsStages[stage] + stagePos + d];
                }
                else
                {
                    d_bufferT_[(nColsStagesAcc[ns * o + stage] + (stagePos + d)) * nrowsPack + row_loc] = aux_trace[offsetsStages[stage] + l * nColsStages[stage] + stagePos + d];
                }
            }
        }
    }

    if (d_deviceArgs->expType == 0)
    {
        for (uint64_t d = 0; d < d_deviceArgs->boundSize; ++d)
        {
            d_bufferT_[(nColsStagesAcc[ns * nOpenings] + d + 1) * nrowsPack + row_loc] = zi[row + row_loc + d * domainSize];
        }
        d_bufferT_[(nColsStagesAcc[ns * nOpenings]) * nrowsPack + row_loc] = x_2ns[row + row_loc];
    }
    else if (d_deviceArgs->expType == 1)
    {
        for (uint64_t d = 0; d < nOpenings; ++d)
        {
            for (uint64_t k = 0; k < FIELD_EXTENSION; ++k)
            {
                d_bufferT_[(nColsStagesAcc[ns * nOpenings] + d * FIELD_EXTENSION + k) * nrowsPack + row_loc] = xDivXSub[(row + row_loc + d * domainSize) * FIELD_EXTENSION + k];
            }
        }
    }
    else
    {
        d_bufferT_[(nColsStagesAcc[ns * nOpenings]) * nrowsPack + row_loc] = x_n[row + row_loc];
    }
}

__device__ void multiplyPolynomials__(DeviceArguments *deviceArgs, DestGPU &dest, gl64_t *destVals)
{
    if (dest.dim == 1)
    {
        gl64_t::op_gpu(2, &destVals[0], &destVals[0], false, &destVals[FIELD_EXTENSION * deviceArgs->nrowsPack], false);
    }
    else
    {
        assert(blockDim.x <= 256);
        __shared__ gl64_t vals[FIELD_EXTENSION * 256]; // rick: corregir
        if (dest.params[0].dim == FIELD_EXTENSION && dest.params[1].dim == FIELD_EXTENSION)
        {
            Goldilocks3GPU::op_gpu(2, &vals[0], &destVals[0], false, &destVals[FIELD_EXTENSION * deviceArgs->nrowsPack], false);
        }
        else if (dest.params[0].dim == FIELD_EXTENSION && dest.params[1].dim == 1)
        {
            Goldilocks3GPU::op_31_gpu(2, &vals[0], &destVals[0], false, &destVals[FIELD_EXTENSION * deviceArgs->nrowsPack], false);
        }
        else
        {
            Goldilocks3GPU::op_31_gpu(2, &vals[0], &destVals[FIELD_EXTENSION * deviceArgs->nrowsPack], false, &destVals[0], false);
        }
        gl64_t::copy_gpu(&destVals[0], &vals[0], false);
        gl64_t::copy_gpu(&destVals[deviceArgs->nrowsPack], &vals[deviceArgs->nrowsPack], false);
        gl64_t::copy_gpu(&destVals[2 * deviceArgs->nrowsPack], &vals[2 * deviceArgs->nrowsPack], false);
    }
}
__global__ __launch_bounds__(128, 4) void computeExpressions_(DeviceArguments *d_deviceArgs)
{

    int chunk_idx = blockIdx.x;
    int pack_idx = threadIdx.x;
    uint32_t iBlock = blockIdx.x;

    gl64_t *challenges = (gl64_t *)d_deviceArgs->challenges;
    gl64_t *numbers = (gl64_t *)d_deviceArgs->numbers;
    gl64_t *publics = (gl64_t *)d_deviceArgs->publics;
    gl64_t *evals = (gl64_t *)d_deviceArgs->evals;
    gl64_t *airgroupValues = (gl64_t *)d_deviceArgs->airgroupValues;
    gl64_t *airValues = (gl64_t *)d_deviceArgs->airValues;
    uint64_t *nColsStagesAcc = d_deviceArgs->nColsStagesAcc;
    uint64_t domainSize = d_deviceArgs->domainSize;
    uint64_t nrowsPack = d_deviceArgs->nrowsPack;
    assert(nrowsPack == blockDim.x);
    DestGPU *dests = d_deviceArgs->dests;
    uint32_t nDests = d_deviceArgs->nDests;
    uint64_t nchunks = domainSize / nrowsPack;
    gl64_t *destVals = (gl64_t *)(&d_deviceArgs->destVals[iBlock * d_deviceArgs->destValsSize]);
    gl64_t *bufferT_ = (gl64_t *)(&d_deviceArgs->bufferT_[iBlock * d_deviceArgs->bufferSize]);
    gl64_t *tmp1 = (gl64_t *)(&d_deviceArgs->tmp1[iBlock * d_deviceArgs->tmp1Size]);
    gl64_t *tmp3 = (gl64_t *)(&d_deviceArgs->tmp3[iBlock * d_deviceArgs->tmp3Size]);

    gl64_t * expressions_params[10];
    expressions_params[0] = bufferT_;
    expressions_params[1] = tmp1;
    expressions_params[2] = publics;
    expressions_params[3] = numbers;
    expressions_params[4] = airValues;
    expressions_params[5] = NULL;//proofValues;
    expressions_params[6] = tmp3;
    expressions_params[7] = airgroupValues;
    expressions_params[8] = challenges;
    expressions_params[9] = evals;
    //uint64_t debug_line = 150528;
    //uint64_t debug_thread = 0;

    while (chunk_idx < nchunks)
    {
        uint64_t i = chunk_idx * nrowsPack;
        loadPolynomials__(d_deviceArgs, i, iBlock);
        for (uint64_t j = 0; j < nDests; ++j)
        {
            for (uint64_t k = 0; k < dests[j].nParams; ++k)
            {
                uint64_t i_args = 0;

                if (dests[j].params[k].op == opType::cm || dests[j].params[k].op == opType::const_)
                {
                    uint64_t openingPointIndex = dests[j].params[k].rowOffsetIndex;
                    uint64_t buffPos = d_deviceArgs->ns * openingPointIndex + dests[j].params[k].stage;
                    uint64_t stagePos = dests[j].params[k].stagePos;
                    copyPolynomial__(d_deviceArgs, &destVals[k * FIELD_EXTENSION * nrowsPack], dests[j].params[k].inverse, dests[j].params[k].dim, &bufferT_[(nColsStagesAcc[buffPos] + stagePos) * nrowsPack]);
                    continue;
                }
                else if (dests[j].params[k].op == opType::number)
                {
                    gl64_t val(dests[j].params[k].value);
                    if (dests[j].params[k].inverse)
                        val = val.reciprocal();
                    destVals[k * FIELD_EXTENSION * nrowsPack + pack_idx] = val;
                    continue;
                }
                uint8_t *ops = &d_deviceArgs->ops[dests[j].params[k].parserParams.opsOffset];
                uint16_t *args = &d_deviceArgs->args[dests[j].params[k].parserParams.argsOffset];
                //bool print = (3286 == dests[j].params[k].parserParams.nOps);

            
                for (uint64_t kk = 0; kk < dests[j].params[k].parserParams.nOps; ++kk) {
                    switch (ops[kk]) {
                        case 0: {
                            // COPY dim1 to dim1
                            gl64_t::copy_gpu( &expressions_params[args[i_args]][(nColsStagesAcc[args[i_args + 1]] + args[i_args + 2]) * nrowsPack], &expressions_params[args[i_args + 3]][(nColsStagesAcc[args[i_args + 4]] + args[i_args + 5]) * (1 - args[i_args + 6])  * nrowsPack + args[i_args + 6]* args[i_args + 5]],args[i_args + 6] );
                            /*if(  i==debug_line && threadIdx.x == debug_thread && print){
                                //result
                                printf("Case 0\n");
                                printf("Op %lu of %d\n", kk, dests[j].params[k].parserParams.nOps);
                                printf("Arguments %lu\n", expressions_params[args[i_args + 3]][(nColsStagesAcc[args[i_args + 4]] + args[i_args + 5]) * nrowsPack].get_val());
                                printf("Result: %lu\n", expressions_params[args[i_args]][(nColsStagesAcc[args[i_args + 1]] + args[i_args + 2]) * nrowsPack].get_val());
                            }*/
                            i_args += 7;
                            break;
                        }
                        case 1: {
                            // OPERATION WITH DEST: dim1 - SRC0: dim1 - SRC1: dim1
                            /*if( i==debug_line && threadIdx.x == debug_thread && print){
                                //result                                
                                printf("Arguments: %lu %d %lu %d\n", 
                                    expressions_params[args[i_args + 4]][(nColsStagesAcc[args[i_args + 5]] + args[i_args + 6]) * (1 - args[i_args + 7])  * nrowsPack + args[i_args + 7]* args[i_args + 6]].get_val(), 
                                    args[i_args + 7], 
                                    expressions_params[args[i_args + 8]][(nColsStagesAcc[args[i_args + 9]] + args[i_args + 10]) * (1 - args[i_args + 11]) * nrowsPack + args[i_args + 11] * args[i_args + 10]].get_val(), 
                                    args[i_args + 11]);
                                printf("domainSize: %lu\n", domainSize);
                            }*/
                            gl64_t::op_gpu( args[i_args], &expressions_params[args[i_args + 1]][(nColsStagesAcc[args[i_args + 2]] + args[i_args + 3]) * nrowsPack ], &expressions_params[args[i_args + 4]][(nColsStagesAcc[args[i_args + 5]] + args[i_args + 6]) * (1 - args[i_args + 7])  * nrowsPack + args[i_args + 7]* args[i_args + 6]],args[i_args + 7] , &expressions_params[args[i_args + 8]][(nColsStagesAcc[args[i_args + 9]] + args[i_args + 10]) * (1 - args[i_args + 11]) * nrowsPack + args[i_args + 11] * args[i_args + 10]], args[i_args + 11]);
                            /*if( i==debug_line && threadIdx.x == debug_thread && print){
                                //result
                                printf("Case 1\n");
                                printf("Op %lu of %d\n", kk, dests[j].params[k].parserParams.nOps);
                                printf("Buffer: %d %d %d \n", args[i_args + 1], args[i_args + 4], args[i_args + 8]);
                                printf("Arguments: %lu %d %lu %d\n", 
                                    expressions_params[args[i_args + 4]][(nColsStagesAcc[args[i_args + 5]] + args[i_args + 6]) * (1 - args[i_args + 7])  * nrowsPack + args[i_args + 7]* args[i_args + 6]].get_val(), 
                                    args[i_args + 7], 
                                    expressions_params[args[i_args + 8]][(nColsStagesAcc[args[i_args + 9]] + args[i_args + 10]) * (1 - args[i_args + 11]) * nrowsPack + args[i_args + 11] * args[i_args + 10]].get_val(), 
                                    args[i_args + 11]);
                                printf("Result: %lu\n", expressions_params[args[i_args + 1]][(nColsStagesAcc[args[i_args + 2]] + args[i_args + 3]) * nrowsPack].get_val());
                                printf("args: %d %d %d %d %d %d %d %d %d %d %d %d\n", args[i_args], args[i_args + 1], args[i_args + 2], args[i_args + 3], args[i_args + 4], args[i_args + 5], args[i_args + 6], args[i_args + 7], args[i_args + 8], args[i_args + 9], args[i_args + 10], args[i_args + 11]);
                            }*/
                            i_args += 12;
                            break;
                        }
                        case 2: {
                            // OPERATION WITH DEST: dim3 - SRC0: dim3 - SRC1: dim1
                            Goldilocks3GPU::op_31_gpu( args[i_args], &expressions_params[args[i_args + 1]][(nColsStagesAcc[args[i_args + 2]] + args[i_args + 3]) * nrowsPack ], &expressions_params[args[i_args + 4]][(nColsStagesAcc[args[i_args + 5]] + args[i_args + 6]) * (1 - args[i_args + 7])  * nrowsPack + args[i_args + 7]* args[i_args + 6]],args[i_args + 7] , &expressions_params[args[i_args + 8]][(nColsStagesAcc[args[i_args + 9]] + args[i_args + 10]) * (1 - args[i_args + 11]) * nrowsPack + args[i_args + 11] * args[i_args + 10]], args[i_args + 11]);
                            /*if(  i==debug_line && threadIdx.x == debug_thread && print){
                                //result
                                printf("Case 2\n");
                                printf("Op %lu of %d\n", kk, dests[j].params[k].parserParams.nOps);
                                printf("Buffer: %d %d %d \n", args[i_args + 1], args[i_args + 4], args[i_args + 8]);
                                printf("Arguments: %lu %lu %lu, %d, %lu %lu %lu,  %d\n", 
                                    expressions_params[args[i_args + 4]][(nColsStagesAcc[args[i_args + 5]] + args[i_args + 6]) * (1 - args[i_args + 7]) * nrowsPack + args[i_args + 7]* args[i_args + 6]].get_val(), 
                                    expressions_params[args[i_args + 4]][(nColsStagesAcc[args[i_args + 5]] + args[i_args + 6]) * (1 - args[i_args + 7]) * nrowsPack+ args[i_args + 7]* args[i_args + 6] +1].get_val(),
                                    expressions_params[args[i_args + 4]][(nColsStagesAcc[args[i_args + 5]] + args[i_args + 6]) * (1 - args[i_args + 7]) * nrowsPack+ args[i_args + 7]* args[i_args + 6] +2].get_val(),
                                    args[i_args + 7], 
                                    expressions_params[args[i_args + 8]][(nColsStagesAcc[args[i_args + 9]] + args[i_args + 10]) * (1 - args[i_args + 11]) * nrowsPack + args[i_args + 11] * args[i_args + 10]].get_val(),
                                    expressions_params[args[i_args + 8]][(nColsStagesAcc[args[i_args + 9]] + args[i_args + 10]) * (1 - args[i_args + 11]) * nrowsPack+ args[i_args + 11] * args[i_args + 10]+1].get_val(),
                                    expressions_params[args[i_args + 8]][(nColsStagesAcc[args[i_args + 9]] + args[i_args + 10]) * (1 - args[i_args + 11]) * nrowsPack+ args[i_args + 11] * args[i_args + 10]+2].get_val(), 
                                    args[i_args + 11]);
                                printf("Result: %lu %lu %lu\n", 
                                    expressions_params[args[i_args + 1]][(nColsStagesAcc[args[i_args + 2]] + args[i_args + 3]) * nrowsPack].get_val(), 
                                    expressions_params[args[i_args + 1]][(nColsStagesAcc[args[i_args + 2]] + args[i_args + 3]) * nrowsPack + 1].get_val(),
                                    expressions_params[args[i_args + 1]][(nColsStagesAcc[args[i_args + 2]] + args[i_args + 3]) * nrowsPack + 2].get_val());
                            }*/
                            i_args += 12;
                            break;
                        }
                        case 3: {
                            // OPERATION WITH DEST: dim3 - SRC0: dim3 - SRC1: dim3
                            Goldilocks3GPU::op_gpu( args[i_args], &expressions_params[args[i_args + 1]][(nColsStagesAcc[args[i_args + 2]] + args[i_args + 3]) * nrowsPack ], &expressions_params[args[i_args + 4]][(nColsStagesAcc[args[i_args + 5]] + args[i_args + 6]) * (1 - args[i_args + 7])  * nrowsPack + args[i_args + 7]* args[i_args + 6]],args[i_args + 7] , &expressions_params[args[i_args + 8]][(nColsStagesAcc[args[i_args + 9]] + args[i_args + 10]) * (1 - args[i_args + 11]) * nrowsPack + args[i_args + 11] * args[i_args + 10]], args[i_args + 11]);
                            /*if(  i==debug_line && threadIdx.x == debug_thread && print){
                                //result
                                printf("Case 3\n");
                                printf("Op %lu of %d\n", kk, dests[j].params[k].parserParams.nOps);
                                printf("Result: %lu %lu %lu\n", 
                                    expressions_params[args[i_args + 1]][(nColsStagesAcc[args[i_args + 2]] + args[i_args + 3]) * nrowsPack].get_val(), 
                                    expressions_params[args[i_args + 1]][(nColsStagesAcc[args[i_args + 2]] + args[i_args + 3]) * nrowsPack + 1].get_val(),
                                    expressions_params[args[i_args + 1]][(nColsStagesAcc[args[i_args + 2]] + args[i_args + 3]) * nrowsPack + 2].get_val());
                            }*/
                            i_args += 12;
                            break;
                        }
                        case 4: {
                            // COPY dim3 to dim3
                            Goldilocks3GPU::copy_gpu( &expressions_params[args[i_args]][(nColsStagesAcc[args[i_args + 1]] + args[i_args + 2]) * nrowsPack], &expressions_params[args[i_args + 3]][(nColsStagesAcc[args[i_args + 4]] + args[i_args + 5]) * (1 - args[i_args + 6])  * nrowsPack + args[i_args + 6]* args[i_args + 5]],args[i_args + 6] );
                            /*if(  i==debug_line && threadIdx.x == debug_thread && print){
                                //result
                                printf("Case 4\n");
                                printf("Op %lu of %d\n", kk, dests[j].params[k].parserParams.nOps);
                                printf("Result: %lu %lu %lu\n", 
                                    expressions_params[args[i_args]][(nColsStagesAcc[args[i_args + 1]] + args[i_args + 2]) * nrowsPack].get_val(), 
                                    expressions_params[args[i_args]][(nColsStagesAcc[args[i_args + 1]] + args[i_args + 2]) * nrowsPack + 1].get_val(),
                                    expressions_params[args[i_args]][(nColsStagesAcc[args[i_args + 1]] + args[i_args + 2]) * nrowsPack + 2].get_val());
                            }*/
                            i_args += 7;
                            break;
                        }
                        default: {
                            printf(" Wrong operation!\n");
                            assert(0);
                        }
                    }
                }

                if (dests[j].params[k].parserParams.destDim == 1)
                {
                    copyPolynomial__(d_deviceArgs, &destVals[k * FIELD_EXTENSION * nrowsPack], dests[j].params[k].inverse, dests[j].params[k].parserParams.destDim, &tmp1[dests[j].params[k].parserParams.destId * nrowsPack]);
                }
                else
                {
                    copyPolynomial__(d_deviceArgs, &destVals[k * FIELD_EXTENSION * nrowsPack], dests[j].params[k].inverse, dests[j].params[k].parserParams.destDim, &tmp3[dests[j].params[k].parserParams.destId * FIELD_EXTENSION * nrowsPack]);
                }
            }

            if (dests[j].nParams == 2)
            {
                multiplyPolynomials__(d_deviceArgs, dests[j], destVals);
            }
            storeOnePolynomial__(d_deviceArgs, destVals, i, j);
        }
        chunk_idx += gridDim.x;
    }
}

#endif