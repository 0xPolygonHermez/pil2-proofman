#include "expressions_gpu.cuh"
#include "cuda_utils.cuh"
#include "cuda_utils.hpp"
#include "gl64_t.cuh"
#include "goldilocks_cubic_extension.cuh"

ExpressionsGPU::ExpressionsGPU(SetupCtx &setupCtx, uint32_t nParamsMax, uint32_t nTemp1Max, uint32_t nTemp3Max, uint64_t nrowsPack_, uint32_t nBlocks_) : ExpressionsCtx(setupCtx), nParamsMax(nParamsMax), nTemp1Max(nTemp1Max), nTemp3Max(nTemp3Max), nrowsPack(nrowsPack_), nBlocks(nBlocks_)
{
    uint64_t nOpenings = setupCtx.starkInfo.openingPoints.size();
    uint64_t ns = 2 + setupCtx.starkInfo.nStages + setupCtx.starkInfo.customCommits.size();
    vector<uint64_t> nColsStages(ns * nOpenings + 1, 0);
    vector<uint64_t> nColsStagesAcc(ns * nOpenings + 1, 0);

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
    cudaMalloc(&h_deviceArgs.proofValues, setupCtx.starkInfo.proofValuesMap.size() * FIELD_EXTENSION * sizeof(Goldilocks::Element));
    cudaMalloc(&h_deviceArgs.ops, setupCtx.expressionsBin.expressionsBinArgsExpressions.nOpsTotal * sizeof(uint8_t));
    cudaMalloc(&h_deviceArgs.args, setupCtx.expressionsBin.expressionsBinArgsExpressions.nArgsTotal * sizeof(uint16_t));

    uint64_t nCols1 = nColsStagesAcc[ns * nOpenings] + setupCtx.starkInfo.boundaries.size() + 1;
    uint64_t nCols2 = nColsStagesAcc[ns * nOpenings] + nOpenings * FIELD_EXTENSION;
    uint64_t nCols3 = nColsStagesAcc[ns * nOpenings] + 1;
    uint64_t nColsMax = max(nCols1, max(nCols2, nCols3));

    // bufferT_
    h_deviceArgs.bufferSize = nOpenings * nrowsPack * nColsMax; // this must be moved from here

    while(nBlocks * h_deviceArgs.bufferSize * sizeof(Goldilocks::Element) > 1024 * 1024 * 1024 * 16){
        nBlocks = nBlocks / 2;
        nrowsPack = nrowsPack / 2;
        h_deviceArgs.bufferSize = nOpenings * nrowsPack * nColsMax;
    }
    cudaMalloc(&h_deviceArgs.bufferT_, nBlocks * h_deviceArgs.bufferSize * sizeof(Goldilocks::Element));
    std::cout << "Total memory in expressions buffers [Gb]: " << (1.0 * nBlocks * h_deviceArgs.bufferSize * sizeof(Goldilocks::Element)) / (1024.0 * 1024.0 * 1024.0) << std::endl;

    // destVals
    h_deviceArgs.destValsSize = nParamsMax * FIELD_EXTENSION * nrowsPack;
    cudaMalloc(&h_deviceArgs.destVals, nBlocks * h_deviceArgs.destValsSize * sizeof(Goldilocks::Element));
    std::cout << "Total memory in expressions destVals [Gb]: " << (1.0 * nBlocks * h_deviceArgs.destValsSize * sizeof(Goldilocks::Element)) / (1024.0 * 1024.0 * 1024.0) << std::endl;

    // tmps
    h_deviceArgs.tmp1Size = nTemp1Max * nrowsPack;
    h_deviceArgs.tmp3Size = nTemp3Max * FIELD_EXTENSION * nrowsPack;
    cudaMalloc(&h_deviceArgs.tmp1, nBlocks * h_deviceArgs.tmp1Size * sizeof(Goldilocks::Element));
    cudaMalloc(&h_deviceArgs.tmp3, nBlocks * h_deviceArgs.tmp3Size * sizeof(Goldilocks::Element));
    std::cout << "Total memory in expressions tmp1 [Gb]: " << (1.0 * nBlocks * h_deviceArgs.tmp1Size * sizeof(Goldilocks::Element)) / (1024.0 * 1024.0 * 1024.0) << std::endl;
    std::cout << "Total memory in expressions tmp3 [Gb]: " << (1.0 * nBlocks * h_deviceArgs.tmp3Size * sizeof(Goldilocks::Element)) / (1024.0 * 1024.0 * 1024.0) << std::endl;

    // constant deviceArgs
    cudaMemcpy(h_deviceArgs.nColsStages, nColsStages.data(), nColsStages.size() * sizeof(uint64_t), cudaMemcpyHostToDevice);
    cudaMemcpy(h_deviceArgs.nColsStagesAcc, nColsStagesAcc.data(), nColsStagesAcc.size() * sizeof(uint64_t), cudaMemcpyHostToDevice);
    h_deviceArgs.N = 1 << setupCtx.starkInfo.starkStruct.nBits;
    h_deviceArgs.NExtended = 1 << setupCtx.starkInfo.starkStruct.nBitsExt;
    h_deviceArgs.nrowsPack = nrowsPack;
    h_deviceArgs.nOpenings = nOpenings;
    h_deviceArgs.ns = ns;

    // Size fixed customCommit
    if (setupCtx.starkInfo.mapTotalNCustomCommitsFixed > 0)
    {
        h_deviceArgs.customCommitsCols = setupCtx.starkInfo.customCommits[0].stageWidths[0];
    }
    else
    {
        h_deviceArgs.customCommitsCols = 0;
    }
};

ExpressionsGPU::~ExpressionsGPU()
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
    cudaFree(h_deviceArgs.proofValues);
    cudaFree(h_deviceArgs.ops);
    cudaFree(h_deviceArgs.args);
    cudaFree(h_deviceArgs.bufferT_);
    cudaFree(h_deviceArgs.destVals);
    cudaFree(h_deviceArgs.tmp1);
    cudaFree(h_deviceArgs.tmp3);
}

void ExpressionsGPU::setBufferTInfo(uint64_t domainSize, StepsParams &params, StepsParams &params_gpu, ParserArgs &parserArgs, std::vector<Dest> &dests)
{

    bool domainExtended = domainSize == uint64_t(1 << setupCtx.starkInfo.starkStruct.nBitsExt) ? true : false;
    uint64_t expId = dests[0].params[0].op == opType::tmp ? dests[0].params[0].parserParams.destDim : 0;
    uint64_t nOpenings = h_deviceArgs.nOpenings;
    uint64_t ns = h_deviceArgs.ns;
    vector<uint64_t> nColsStages(ns * nOpenings + 1, 0);
    vector<uint64_t> nColsStagesAcc(ns * nOpenings + 1, 0);
    vector<uint64_t> offsetsStages(ns * nOpenings + 1, 0);

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
    h_deviceArgs.domainSize = domainSize;
    h_deviceArgs.nCols = nCols;
    h_deviceArgs.domainExtended = domainExtended;

    uint32_t extendBits = (setupCtx.starkInfo.starkStruct.nBitsExt - setupCtx.starkInfo.starkStruct.nBits);
    int64_t extend = domainExtended ? (1 << extendBits) : 1;
    uint64_t *nextStrides = new uint64_t[nOpenings];
    for (uint64_t i = 0; i < nOpenings; ++i)
    {
        uint64_t opening = setupCtx.starkInfo.openingPoints[i] < 0 ? setupCtx.starkInfo.openingPoints[i] + domainSize : setupCtx.starkInfo.openingPoints[i];
        nextStrides[i] = opening * extend;
    }
    cudaMemcpy(h_deviceArgs.nextStrides, nextStrides, h_deviceArgs.nOpenings * sizeof(uint64_t), cudaMemcpyHostToDevice);
    delete[] nextStrides;
    CHECKCUDAERR(cudaGetLastError());


    cudaMemcpy(h_deviceArgs.offsetsStages, offsetsStages.data(), offsetsStages.size() * sizeof(uint64_t), cudaMemcpyHostToDevice);
    CHECKCUDAERR(cudaGetLastError());

    h_deviceArgs.constPolsSize = setupCtx.starkInfo.nConstants;
    h_deviceArgs.cmPolsInfoSize = setupCtx.starkInfo.cmPolsMap.size();
    uint64_t *cmPolsInfo = new uint64_t[h_deviceArgs.cmPolsInfoSize * 3];
    for (uint64_t i = 0; i < h_deviceArgs.cmPolsInfoSize; ++i)
    {
        cmPolsInfo[i * 3] = setupCtx.starkInfo.cmPolsMap[i].stage;
        cmPolsInfo[i * 3 + 1] = setupCtx.starkInfo.cmPolsMap[i].stagePos;
        cmPolsInfo[i * 3 + 2] = setupCtx.starkInfo.cmPolsMap[i].dim;
    }
    cudaMemcpy(h_deviceArgs.cmPolsInfo, cmPolsInfo, h_deviceArgs.cmPolsInfoSize * 3 * sizeof(uint64_t), cudaMemcpyHostToDevice);
    CHECKCUDAERR(cudaGetLastError());

    free(cmPolsInfo);
    if (dests[0].params[0].parserParams.expId == int64_t(setupCtx.starkInfo.cExpId))
    {
        h_deviceArgs.expType = 0;
    }
    else if (dests[0].params[0].parserParams.expId == int64_t(setupCtx.starkInfo.friExpId))
    {
        h_deviceArgs.expType = 1;
    }
    else
    {

        h_deviceArgs.expType = 2;
    }

    h_deviceArgs.boundSize = setupCtx.starkInfo.boundaries.size();

    // Dests
    DestGPU *dests_aux = new DestGPU[dests.size()];
    h_deviceArgs.nDests = dests.size();
    for (uint64_t i = 0; i < dests.size(); ++i)
    {
        dests_aux[i].dest_gpu = dests[i].dest_gpu;
        dests_aux[i].offset = dests[i].offset;
        dests_aux[i].dim = dests[i].dim;
        dests_aux[i].nParams = dests[i].params.size();
        dests_aux[i].domainSize = dests[i].domainSize;
        assert(dests_aux[i].nParams <= nParamsMax);
        dests_aux[i].params = new ParamsGPU[dests[i].params.size()]; // rick

        for (uint64_t j = 0; j < dests_aux[i].nParams; ++j)
        {
            dests_aux[i].params[j].dim = dests[i].params[j].dim;
            dests_aux[i].params[j].stage = dests[i].params[j].stage;
            dests_aux[i].params[j].stagePos = dests[i].params[j].stagePos;
            dests_aux[i].params[j].polsMapId = dests[i].params[j].polsMapId;
            dests_aux[i].params[j].rowOffsetIndex = dests[i].params[j].rowOffsetIndex;
            dests_aux[i].params[j].inverse = dests[i].params[j].inverse;
            dests_aux[i].params[j].batch = dests[i].params[j].batch;
            dests_aux[i].params[j].op = dests[i].params[j].op;
            dests_aux[i].params[j].value = dests[i].params[j].value;
            dests_aux[i].params[j].parserParams.stage = dests[i].params[j].parserParams.stage;
            dests_aux[i].params[j].parserParams.expId = dests[i].params[j].parserParams.expId;
            dests_aux[i].params[j].parserParams.nTemp1 = dests[i].params[j].parserParams.nTemp1;
            assert(dests_aux[i].params[j].parserParams.nTemp1 < nTemp1Max);
            dests_aux[i].params[j].parserParams.nTemp3 = dests[i].params[j].parserParams.nTemp3;
            assert(dests_aux[i].params[j].parserParams.nTemp3 < nTemp3Max);
            dests_aux[i].params[j].parserParams.nOps = dests[i].params[j].parserParams.nOps;
            dests_aux[i].params[j].parserParams.opsOffset = dests[i].params[j].parserParams.opsOffset;
            dests_aux[i].params[j].parserParams.nArgs = dests[i].params[j].parserParams.nArgs;
            dests_aux[i].params[j].parserParams.argsOffset = dests[i].params[j].parserParams.argsOffset;
            dests_aux[i].params[j].parserParams.constPolsOffset = dests[i].params[j].parserParams.constPolsOffset;
            dests_aux[i].params[j].parserParams.cmPolsOffset = dests[i].params[j].parserParams.cmPolsOffset;
            dests_aux[i].params[j].parserParams.challengesOffset = dests[i].params[j].parserParams.challengesOffset;
            dests_aux[i].params[j].parserParams.publicsOffset = dests[i].params[j].parserParams.publicsOffset;
            dests_aux[i].params[j].parserParams.airgroupValuesOffset = dests[i].params[j].parserParams.airgroupValuesOffset;
            dests_aux[i].params[j].parserParams.airValuesOffset = dests[i].params[j].parserParams.airValuesOffset;
            dests_aux[i].params[j].parserParams.firstRow = dests[i].params[j].parserParams.firstRow;
            dests_aux[i].params[j].parserParams.lastRow = dests[i].params[j].parserParams.lastRow;
            dests_aux[i].params[j].parserParams.destDim = dests[i].params[j].parserParams.destDim;
            dests_aux[i].params[j].parserParams.destId = dests[i].params[j].parserParams.destId;
            dests_aux[i].params[j].parserParams.imPol = dests[i].params[j].parserParams.imPol;
        }
    }

    // Dests
    DestGPU *d_dests = new DestGPU[h_deviceArgs.nDests];
    for (int i = 0; i < h_deviceArgs.nDests; ++i)
    {
        d_dests[i].dest_gpu = dests_aux[i].dest_gpu;
        d_dests[i].offset = dests_aux[i].offset;
        d_dests[i].dim = dests_aux[i].dim;
        d_dests[i].nParams = dests_aux[i].nParams;
        d_dests[i].domainSize = dests_aux[i].domainSize;
        cudaMalloc(&d_dests[i].params, d_dests[i].nParams * sizeof(ParamsGPU));
        if (d_dests[i].nParams > 0)
            dest_params.push_back(d_dests[i].params);
        cudaMemcpy(d_dests[i].params, dests_aux[i].params, d_dests[i].nParams * sizeof(ParamsGPU), cudaMemcpyHostToDevice);
        CHECKCUDAERR(cudaGetLastError());


    }
    for (int i = 0; i < dests.size(); i++)
    {
        delete[] dests_aux[i].params;
    }
    delete[] dests_aux;
    DestGPU *d_dests_;
    cudaMalloc(&d_dests_, h_deviceArgs.nDests * sizeof(DestGPU));
    cudaMemcpy(d_dests_, d_dests, h_deviceArgs.nDests * sizeof(DestGPU), cudaMemcpyHostToDevice);
    CHECKCUDAERR(cudaGetLastError());

    delete[] d_dests;
    h_deviceArgs.dests = d_dests_;

    // non polnomial arguments
    h_deviceArgs.nChallenges = setupCtx.starkInfo.challengesMap.size();
    h_deviceArgs.nNumbers = parserArgs.nNumbers;
    h_deviceArgs.nPublics = setupCtx.starkInfo.nPublics;
    h_deviceArgs.nEvals = setupCtx.starkInfo.evMap.size();
    h_deviceArgs.nAirgroupValues = setupCtx.starkInfo.airgroupValuesMap.size();
    h_deviceArgs.nAirValues = setupCtx.starkInfo.airValuesMap.size();
    h_deviceArgs.nProofValues = setupCtx.starkInfo.proofValuesMap.size();
    // Expressions bin
    h_deviceArgs.nOpsTotal = parserArgs.nOpsTotal;
    h_deviceArgs.nArgsTotal = parserArgs.nArgsTotal;

    // bufferT_
    h_deviceArgs.nBlocks = nBlocks;

    cudaMemcpy(h_deviceArgs.cmPolsInfo, h_deviceArgs.cmPolsInfo, 3 * h_deviceArgs.cmPolsInfoSize * sizeof(uint64_t), cudaMemcpyHostToDevice);
    CHECKCUDAERR(cudaGetLastError());

    cudaMemcpy(h_deviceArgs.zi, setupCtx.proverHelpers.zi, h_deviceArgs.boundSize * h_deviceArgs.NExtended * sizeof(Goldilocks::Element), cudaMemcpyHostToDevice); // cal copiar cada cop?
    CHECKCUDAERR(cudaGetLastError());

    cudaMemcpy(h_deviceArgs.x_n, setupCtx.proverHelpers.x_n, h_deviceArgs.N * sizeof(Goldilocks::Element), cudaMemcpyHostToDevice);                                // cal cada cop? no es pot transportar?
    CHECKCUDAERR(cudaGetLastError());

    cudaMemcpy(h_deviceArgs.x_2ns, setupCtx.proverHelpers.x_2ns, h_deviceArgs.NExtended * sizeof(Goldilocks::Element), cudaMemcpyHostToDevice);                    // cal cada cop? no es pot transportar?
    CHECKCUDAERR(cudaGetLastError());

    cudaMemcpy(h_deviceArgs.challenges, params.challenges, h_deviceArgs.nChallenges * FIELD_EXTENSION * sizeof(Goldilocks::Element), cudaMemcpyHostToDevice);
    CHECKCUDAERR(cudaGetLastError());

    cudaMemcpy(h_deviceArgs.numbers, (Goldilocks::Element *)parserArgs.numbers, h_deviceArgs.nNumbers * sizeof(Goldilocks::Element), cudaMemcpyHostToDevice);
    CHECKCUDAERR(cudaGetLastError());

    cudaMemcpy(h_deviceArgs.publics, params.publicInputs, h_deviceArgs.nPublics * sizeof(Goldilocks::Element), cudaMemcpyHostToDevice);
    CHECKCUDAERR(cudaGetLastError());

    cudaMemcpy(h_deviceArgs.evals, params.evals, h_deviceArgs.nEvals * FIELD_EXTENSION * sizeof(Goldilocks::Element), cudaMemcpyHostToDevice);
    CHECKCUDAERR(cudaGetLastError());


    Goldilocks::Element *airgroupValues_aux = new Goldilocks::Element[h_deviceArgs.nAirgroupValues * FIELD_EXTENSION];
    uint64_t p = 0;
    for (uint64_t i = 0; i < setupCtx.starkInfo.airgroupValuesMap.size(); ++i)
    {

        if (setupCtx.starkInfo.airgroupValuesMap[i].stage == 1)
        {
            airgroupValues_aux[i * FIELD_EXTENSION] = params.airgroupValues[p];
            airgroupValues_aux[i * FIELD_EXTENSION + 1] = Goldilocks::zero();
            airgroupValues_aux[i * FIELD_EXTENSION + 2] = Goldilocks::zero();
        }
        else
        {
            airgroupValues_aux[i * FIELD_EXTENSION] = params.airgroupValues[p];
            airgroupValues_aux[i * FIELD_EXTENSION + 1] = params.airgroupValues[p + 1];
            airgroupValues_aux[i * FIELD_EXTENSION + 2] = params.airgroupValues[p + 2];
        }

        if (setupCtx.starkInfo.airgroupValuesMap[i].stage == 1)
        {
            p += 1;
        }
        else
        {
            p += 3;
        }
    }
    cudaMemcpy(h_deviceArgs.airgroupValues, airgroupValues_aux, h_deviceArgs.nAirgroupValues * FIELD_EXTENSION * sizeof(Goldilocks::Element), cudaMemcpyHostToDevice);
    CHECKCUDAERR(cudaGetLastError());
    delete[] airgroupValues_aux;

    Goldilocks::Element *airValues_aux = new Goldilocks::Element[h_deviceArgs.nAirValues * FIELD_EXTENSION];
    p = 0;
    for (uint64_t i = 0; i < setupCtx.starkInfo.airValuesMap.size(); ++i)
    {

        if (setupCtx.starkInfo.airValuesMap[i].stage == 1)
        {
            airValues_aux[i * FIELD_EXTENSION] = params.airValues[p];
            airValues_aux[i * FIELD_EXTENSION + 1] = Goldilocks::zero();
            airValues_aux[i * FIELD_EXTENSION + 2] = Goldilocks::zero();
        }
        else
        {
            airValues_aux[i * FIELD_EXTENSION] = params.airValues[p];
            airValues_aux[i * FIELD_EXTENSION + 1] = params.airValues[p + 1];
            airValues_aux[i * FIELD_EXTENSION + 2] = params.airValues[p + 2];
        }

        if (setupCtx.starkInfo.airValuesMap[i].stage == 1)
        {
            p += 1;
        }
        else
        {
            p += 3;
        }
    }
    cudaMemcpy(h_deviceArgs.airValues, airValues_aux, h_deviceArgs.nAirValues * FIELD_EXTENSION * sizeof(Goldilocks::Element), cudaMemcpyHostToDevice);
    CHECKCUDAERR(cudaGetLastError());
    delete[] airValues_aux;
    Goldilocks::Element *proofValues_aux = new Goldilocks::Element[h_deviceArgs.nProofValues * FIELD_EXTENSION];
    p = 0;
    for (uint64_t i = 0; i < setupCtx.starkInfo.proofValuesMap.size(); ++i)
    {

        if (setupCtx.starkInfo.proofValuesMap[i].stage == 1)
        {
            proofValues_aux[i * FIELD_EXTENSION] = params.proofValues[p];
            proofValues_aux[i * FIELD_EXTENSION + 1] = Goldilocks::zero();
            proofValues_aux[i * FIELD_EXTENSION + 2] = Goldilocks::zero();
        }
        else
        {
            proofValues_aux[i * FIELD_EXTENSION] = params.proofValues[p];
            proofValues_aux[i * FIELD_EXTENSION + 1] = params.proofValues[p + 1];
            proofValues_aux[i * FIELD_EXTENSION + 2] = params.proofValues[p + 2];
        }

        if (setupCtx.starkInfo.proofValuesMap[i].stage == 1)
        {
            p += 1;
        }
        else
        {
            p += 3;
        }
    }

    cudaMemcpy(h_deviceArgs.proofValues, proofValues_aux, h_deviceArgs.nProofValues * FIELD_EXTENSION * sizeof(Goldilocks::Element), cudaMemcpyHostToDevice);
    CHECKCUDAERR(cudaGetLastError());
    delete[] proofValues_aux;
    cudaMemcpy(h_deviceArgs.ops, parserArgs.ops, h_deviceArgs.nOpsTotal * sizeof(uint8_t), cudaMemcpyHostToDevice);
    CHECKCUDAERR(cudaGetLastError());
    cudaMemcpy(h_deviceArgs.args, parserArgs.args, h_deviceArgs.nArgsTotal * sizeof(uint16_t), cudaMemcpyHostToDevice);
    CHECKCUDAERR(cudaGetLastError());

    h_deviceArgs.constPols = h_deviceArgs.domainExtended ? params_gpu.pConstPolsExtendedTreeAddress : params_gpu.pConstPolsAddress;
    h_deviceArgs.trace = params_gpu.trace;
    h_deviceArgs.aux_trace = params_gpu.aux_trace;
    h_deviceArgs.xDivXSub = params_gpu.xDivXSub;
    h_deviceArgs.customCommits = params_gpu.pCustomCommitsFixed;

    // Allocate memory for the struct on the device
    cudaMalloc(&d_deviceArgs, sizeof(DeviceArguments));
    cudaMemcpy(d_deviceArgs, &h_deviceArgs, sizeof(DeviceArguments), cudaMemcpyHostToDevice);
    CHECKCUDAERR(cudaGetLastError());
}

void ExpressionsGPU::calculateExpressions_gpu(StepsParams &params, StepsParams &params_gpu, ParserArgs &parserArgs, std::vector<Dest> dests, uint64_t domainSize)
{

    CHECKCUDAERR(cudaDeviceSynchronize());
    double time = omp_get_wtime();
    setBufferTInfo(domainSize, params, params_gpu, parserArgs, dests);
    CHECKCUDAERR(cudaGetLastError());
    CHECKCUDAERR(cudaDeviceSynchronize());
    time = omp_get_wtime() - time;
    std::cout << "goal2_ setBufferTInfo time: " << time << std::endl;

    CHECKCUDAERR(cudaDeviceSynchronize());
    time = omp_get_wtime();
    dim3 nBlocks = h_deviceArgs.nBlocks;
    dim3 nThreads = h_deviceArgs.nrowsPack;
    std::cout << "goal2_ nBlocks: " << nBlocks.x << std::endl;
    //computeExpressions_explore_<<<nBlocks, nThreads>>>(d_deviceArgs);
    computeExpressions_<<<nBlocks, nThreads>>>(d_deviceArgs);
    CHECKCUDAERR(cudaGetLastError());
    CHECKCUDAERR(cudaDeviceSynchronize());
    time = omp_get_wtime() - time;
    std::cout << "goal2_ de computeExpressions: " << time << std::endl;

    CHECKCUDAERR(cudaDeviceSynchronize());
    time = omp_get_wtime();
    std::cout << " nDests: " << h_deviceArgs.nDests << std::endl;
    for (uint32_t i = 0; i < h_deviceArgs.nDests; ++i)
    {
        if (dests[i].dest != NULL)
        {
            std::cout << " copy size: " << domainSize * FIELD_EXTENSION * sizeof(Goldilocks::Element) << std::endl;
            std::cout << " domainSize: " << dests[i].domainSize << std::endl;    
            Goldilocks::Element *dest = new Goldilocks::Element[dests[i].domainSize * FIELD_EXTENSION];
            cudaMemcpy(dest, dests[i].dest_gpu, dests[i].domainSize * FIELD_EXTENSION * sizeof(Goldilocks::Element), cudaMemcpyDeviceToHost);
            memcpy(dests[i].dest, dest, dests[i].domainSize * FIELD_EXTENSION * sizeof(Goldilocks::Element));
        }
    }
    CHECKCUDAERR(cudaGetLastError());
    CHECKCUDAERR(cudaDeviceSynchronize());
    time = omp_get_wtime() - time;
    std::cout << "goal2_ de cudaMemcpy dests time: " << time << std::endl;

    CHECKCUDAERR(cudaDeviceSynchronize());
    time = omp_get_wtime();
    freeDeviceArguments();
    CHECKCUDAERR(cudaGetLastError());
    CHECKCUDAERR(cudaDeviceSynchronize());
    time = omp_get_wtime() - time;
    std::cout << "goal2_ freeDeviceArguments time: " << time << std::endl;
}

void ExpressionsGPU::freeDeviceArguments()
{
    for (std::vector<ParamsGPU *>::iterator it = dest_params.begin(); it != dest_params.end(); ++it)
    {
        cudaFree(*it);
    }
    dest_params.clear();
    cudaFree(h_deviceArgs.dests);
    cudaFree(d_deviceArgs);
}

__device__ __noinline__ void storeOnePolynomial__(DeviceArguments *d_deviceArgs, gl64_t *destVals, uint64_t row, uint32_t idest)
{
    if (row+blockIdx.x >= d_deviceArgs->dests[idest].domainSize){ //rick
        return;
    }
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

__device__ __noinline__ void copyPolynomial__(DeviceArguments *d_deviceArgs, gl64_t *destVals, bool inverse, uint64_t dim, gl64_t *temp)
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

__device__ __noinline__ void loadPolynomials__(DeviceArguments *d_deviceArgs, uint64_t row, uint32_t iBlock)
{

    bool domainExtended = d_deviceArgs->domainExtended;
    uint64_t *nextStrides = d_deviceArgs->nextStrides;
    uint64_t *nColsStages = d_deviceArgs->nColsStages;
    uint64_t *nColsStagesAcc = d_deviceArgs->nColsStagesAcc;
    Goldilocks::Element *d_bufferT_ = &d_deviceArgs->bufferT_[iBlock * d_deviceArgs->bufferSize];

#pragma unroll 1
    for (uint64_t k = 0; k < d_deviceArgs->constPolsSize; ++k)
    {
        Goldilocks::Element *constPols = domainExtended ? &d_deviceArgs->constPols[2] : d_deviceArgs->constPols;
        for (uint64_t o = 0; o < d_deviceArgs->nOpenings; ++o)
        {
            uint64_t l = (row + threadIdx.x + nextStrides[o]) % d_deviceArgs->domainSize;
            d_bufferT_[(nColsStagesAcc[d_deviceArgs->ns * o] + k) * d_deviceArgs->nrowsPack + threadIdx.x] = constPols[l * nColsStages[0] + k];
        }
    }

#pragma unroll 1
    for (uint64_t k = 0; k < d_deviceArgs->cmPolsInfoSize; ++k)
    {
        uint64_t *cmPolsInfo = d_deviceArgs->cmPolsInfo;
        uint64_t stage = cmPolsInfo[k * 3];
        uint64_t stagePos = cmPolsInfo[k * 3 + 1];
        for (uint64_t d = 0; d < cmPolsInfo[k * 3 + 2]; ++d)
        {
            for (uint64_t o = 0; o < d_deviceArgs->nOpenings; ++o)
            {
                uint64_t l = (row + threadIdx.x + nextStrides[o]) % d_deviceArgs->domainSize;
                if (stage == 1 && !d_deviceArgs->domainExtended)
                {
                    d_bufferT_[(nColsStagesAcc[d_deviceArgs->ns * o + stage] + (stagePos + d)) * d_deviceArgs->nrowsPack + threadIdx.x] = d_deviceArgs->trace[l * nColsStages[stage] + stagePos + d];
                }
                else
                {
                    uint64_t *offsetsStages = d_deviceArgs->offsetsStages;
                    d_bufferT_[(nColsStagesAcc[d_deviceArgs->ns * o + stage] + (stagePos + d)) * d_deviceArgs->nrowsPack + threadIdx.x] = d_deviceArgs->aux_trace[offsetsStages[stage] + l * nColsStages[stage] + stagePos + d];
                }
            }
        }
    }

#pragma unroll 1
    for (uint64_t k = 0; k < d_deviceArgs->customCommitsCols; ++k)
    {
        uint64_t stage = d_deviceArgs->ns - 1; // rick: harcoded one single custom commit
        for (uint64_t o = 0; o < d_deviceArgs->nOpenings; ++o)
        {
            uint64_t l = (row + threadIdx.x + nextStrides[o]) % d_deviceArgs->domainSize;
            uint64_t offset = 0;
            if (d_deviceArgs->domainExtended)
            {
                offset = d_deviceArgs->N * d_deviceArgs->customCommitsCols;
            }
            d_bufferT_[(nColsStagesAcc[d_deviceArgs->ns * o + stage] + k) * d_deviceArgs->nrowsPack + threadIdx.x] = d_deviceArgs->customCommits[offset + l * nColsStages[stage] + k];
        }
    }

    if (d_deviceArgs->expType == 0)
    {
#pragma unroll 1
        for (uint64_t d = 0; d < d_deviceArgs->boundSize; ++d)
        {
            d_bufferT_[(nColsStagesAcc[d_deviceArgs->ns * d_deviceArgs->nOpenings] + d + 1) * d_deviceArgs->nrowsPack + threadIdx.x] = d_deviceArgs->zi[row + threadIdx.x + d * d_deviceArgs->domainSize];
        }
        d_bufferT_[(nColsStagesAcc[d_deviceArgs->ns * d_deviceArgs->nOpenings]) * d_deviceArgs->nrowsPack + threadIdx.x] = d_deviceArgs->x_2ns[row + threadIdx.x];
    }
    else if (d_deviceArgs->expType == 1)
    {
#pragma unroll 1
        for (uint64_t d = 0; d < d_deviceArgs->nOpenings; ++d)
        {
            for (uint64_t k = 0; k < FIELD_EXTENSION; ++k)
            {
                d_bufferT_[(nColsStagesAcc[d_deviceArgs->ns * d_deviceArgs->nOpenings] + d * FIELD_EXTENSION + k) * d_deviceArgs->nrowsPack + threadIdx.x] = d_deviceArgs->xDivXSub[(row + threadIdx.x + d * d_deviceArgs->domainSize) * FIELD_EXTENSION + k];
            }
        }
    }
    else
    {
        d_bufferT_[(nColsStagesAcc[d_deviceArgs->ns * d_deviceArgs->nOpenings]) * d_deviceArgs->nrowsPack + threadIdx.x] = d_deviceArgs->x_n[row + threadIdx.x];
    }
}

__device__ __noinline__ void multiplyPolynomials__(DeviceArguments *deviceArgs, DestGPU &dest, gl64_t *destVals)
{
    if (dest.dim == 1)
    {
        gl64_t::op_gpu(2, &destVals[0], &destVals[0], false, &destVals[FIELD_EXTENSION * deviceArgs->nrowsPack], false);
    }
    else
    {
        assert(blockDim.x <= 128);
        __shared__ gl64_t vals[FIELD_EXTENSION * 128]; // rick: corregir
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

__global__ __launch_bounds__(128) void computeExpressions_explore_(DeviceArguments *d_deviceArgs)
{

    int chunk_idx = blockIdx.x;
    assert(d_deviceArgs->nrowsPack == blockDim.x);
    uint64_t nchunks = d_deviceArgs->domainSize / blockDim.x;


    __shared__ gl64_t *expressions_params[10];

    if (threadIdx.x == 0)
    {
        expressions_params[0] = (gl64_t *)(&d_deviceArgs->bufferT_[blockIdx.x * d_deviceArgs->bufferSize]);
        expressions_params[1] = (gl64_t *)(&d_deviceArgs->tmp1[blockIdx.x * d_deviceArgs->tmp1Size]);
        expressions_params[2] = (gl64_t *)d_deviceArgs->publics;
        expressions_params[3] = (gl64_t *)d_deviceArgs->numbers;
        expressions_params[4] = (gl64_t *)d_deviceArgs->airValues;
        expressions_params[5] = (gl64_t *)d_deviceArgs->proofValues;
        expressions_params[6] = (gl64_t *)(&d_deviceArgs->tmp3[blockIdx.x * d_deviceArgs->tmp3Size]);
        expressions_params[7] = (gl64_t *)d_deviceArgs->airgroupValues;
        expressions_params[8] = (gl64_t *)d_deviceArgs->challenges;
        expressions_params[9] = (gl64_t *)d_deviceArgs->evals;
    }
    __syncthreads();

    while (chunk_idx < nchunks)
    {
        uint64_t i = chunk_idx * blockDim.x;
        bool print = false ; //threadIdx.x == 0 && blockIdx.x == 0 && i==0;
        loadPolynomials__(d_deviceArgs, i, blockIdx.x);

        if(print) printf(" Ndests: %d\n", d_deviceArgs->nDests);

#pragma unroll 1
        for (uint64_t j = 0; j < d_deviceArgs->nDests; ++j)
        {
            if(print) printf(" dest: %lu\n", j);

            for (uint64_t k = 0; k < d_deviceArgs->dests[j].nParams; ++k)
            {
                if(print) printf(" param: %lu of %d\n", k, d_deviceArgs->dests[j].nParams);

                gl64_t *destVals = (gl64_t *)(&d_deviceArgs->destVals[blockIdx.x * d_deviceArgs->destValsSize]);
                uint64_t *nColsStagesAcc = d_deviceArgs->nColsStagesAcc;


                if(print) printf(" nOps: %d\n", d_deviceArgs->dests[j].params[k].parserParams.nOps);
                if(print) printf(" offset: %d\n", d_deviceArgs->dests[j].params[k].parserParams.opsOffset);
                if(print) printf(" nArgs: %d\n", d_deviceArgs->dests[j].params[k].parserParams.nArgs);
                if(print) printf(" argsOffset: %d\n", d_deviceArgs->dests[j].params[k].parserParams.argsOffset);
                if(print) printf(" nOpsTotal: %d\n", d_deviceArgs->nOpsTotal);
                for (uint64_t kk = 0; kk < d_deviceArgs->dests[j].params[k].parserParams.nOps; ++kk)
                {
                    assert(d_deviceArgs->dests[j].params[k].parserParams.opsOffset + kk < d_deviceArgs->nOpsTotal);
                }
            }
        }       
        chunk_idx += gridDim.x;
    }
}
__global__ __launch_bounds__(128) void computeExpressions_(DeviceArguments *d_deviceArgs)
{

    int chunk_idx = blockIdx.x;
    assert(d_deviceArgs->nrowsPack == blockDim.x);
    uint64_t nchunks = d_deviceArgs->domainSize / blockDim.x;


    __shared__ gl64_t *expressions_params[10];

    if (threadIdx.x == 0)
    {
        expressions_params[0] = (gl64_t *)(&d_deviceArgs->bufferT_[blockIdx.x * d_deviceArgs->bufferSize]);
        expressions_params[1] = (gl64_t *)(&d_deviceArgs->tmp1[blockIdx.x * d_deviceArgs->tmp1Size]);
        expressions_params[2] = (gl64_t *)d_deviceArgs->publics;
        expressions_params[3] = (gl64_t *)d_deviceArgs->numbers;
        expressions_params[4] = (gl64_t *)d_deviceArgs->airValues;
        expressions_params[5] = (gl64_t *)d_deviceArgs->proofValues;
        expressions_params[6] = (gl64_t *)(&d_deviceArgs->tmp3[blockIdx.x * d_deviceArgs->tmp3Size]);
        expressions_params[7] = (gl64_t *)d_deviceArgs->airgroupValues;
        expressions_params[8] = (gl64_t *)d_deviceArgs->challenges;
        expressions_params[9] = (gl64_t *)d_deviceArgs->evals;
    }
    __syncthreads();

    while (chunk_idx < nchunks)
    {
        uint64_t i = chunk_idx * blockDim.x;
        bool print = false; //threadIdx.x == 0 && blockIdx.x == 0 && i==0;
        loadPolynomials__(d_deviceArgs, i, blockIdx.x);

        if(print) printf(" Ndests: %d\n", d_deviceArgs->nDests);

#pragma unroll 1
        for (uint64_t j = 0; j < d_deviceArgs->nDests; ++j)
        {
            if(print) printf(" dest: %lu\n", j);

            for (uint64_t k = 0; k < d_deviceArgs->dests[j].nParams; ++k)
            {
                if(print) printf(" param: %lu of %d\n", k, d_deviceArgs->dests[j].nParams);

                gl64_t *destVals = (gl64_t *)(&d_deviceArgs->destVals[blockIdx.x * d_deviceArgs->destValsSize]);
                uint64_t *nColsStagesAcc = d_deviceArgs->nColsStagesAcc;

                if (d_deviceArgs->dests[j].params[k].op == opType::cm || d_deviceArgs->dests[j].params[k].op == opType::const_)
                {
                    uint64_t openingPointIndex = d_deviceArgs->dests[j].params[k].rowOffsetIndex;
                    uint64_t buffPos = d_deviceArgs->ns * openingPointIndex + d_deviceArgs->dests[j].params[k].stage;
                    uint64_t stagePos = d_deviceArgs->dests[j].params[k].stagePos;
                    copyPolynomial__(d_deviceArgs, &destVals[k * FIELD_EXTENSION * blockDim.x], d_deviceArgs->dests[j].params[k].inverse, d_deviceArgs->dests[j].params[k].dim, &expressions_params[0][(nColsStagesAcc[buffPos] + stagePos) * blockDim.x]);
                    continue;
                }
                else if (d_deviceArgs->dests[j].params[k].op == opType::number)
                {
                    gl64_t val(d_deviceArgs->dests[j].params[k].value);
                    if (d_deviceArgs->dests[j].params[k].inverse)
                        val = val.reciprocal();
                    destVals[k * FIELD_EXTENSION * blockDim.x + threadIdx.x] = val;
                    continue;
                }

                if(print) printf(" nOps: %d\n", d_deviceArgs->dests[j].params[k].parserParams.nOps);
                if(print) printf(" offset: %d\n", d_deviceArgs->dests[j].params[k].parserParams.opsOffset);
                uint8_t *ops = &d_deviceArgs->ops[d_deviceArgs->dests[j].params[k].parserParams.opsOffset];
                uint16_t *args = &d_deviceArgs->args[d_deviceArgs->dests[j].params[k].parserParams.argsOffset];

                uint64_t i_args = 0;
                for (uint64_t kk = 0; kk < d_deviceArgs->dests[j].params[k].parserParams.nOps; ++kk)
                {
                    switch (ops[kk])
                    {
                    case 0:
                    {
                        // COPY dim1 to dim1
                        if (print)
                        {
                            // printf("Abans args: %d %d %d %d %d %d %d %d %d %d %d %d\n", args[i_args], args[i_args + 1], args[i_args + 2], args[i_args + 3], args[i_args + 4], args[i_args + 5], args[i_args + 6], args[i_args + 7], args[i_args + 8], args[i_args + 9], args[i_args + 10], args[i_args + 11]);
                        }
                        gl64_t::copy_gpu(&expressions_params[args[i_args]][(nColsStagesAcc[args[i_args + 1]] + args[i_args + 2]) * blockDim.x], &expressions_params[args[i_args + 3]][(nColsStagesAcc[args[i_args + 4]] + args[i_args + 5]) * (1 - args[i_args + 6]) * blockDim.x + args[i_args + 6] * args[i_args + 5]], args[i_args + 6]);
                        if (print)
                        {
                            // result
                            printf("Case 0\n");
                            printf("Op %lu of %d\n", kk, d_deviceArgs->dests[j].params[k].parserParams.nOps);
                            // printf("Arguments %lu\n", expressions_params[args[i_args + 3]][(nColsStagesAcc[args[i_args + 4]] + args[i_args + 5]) * d_deviceArgs->nrowsPack].get_val());
                            printf("Result: %lu\n", expressions_params[args[i_args]][(nColsStagesAcc[args[i_args + 1]] + args[i_args + 2]) * d_deviceArgs->nrowsPack].get_val());
                            // printf("args %d %d %d %d %d %d\n", args[i_args], args[i_args + 1], args[i_args + 2], args[i_args + 3], args[i_args + 4], args[i_args + 5]);
                        }

                        i_args += 7;
                        break;
                    }
                    case 1:
                    {
                        // OPERATION WITH DEST: dim1 - SRC0: dim1 - SRC1: dim1
                        gl64_t::op_gpu(args[i_args], &expressions_params[args[i_args + 1]][(nColsStagesAcc[args[i_args + 2]] + args[i_args + 3]) * blockDim.x], &expressions_params[args[i_args + 4]][(nColsStagesAcc[args[i_args + 5]] + args[i_args + 6]) * (1 - args[i_args + 7]) * blockDim.x + args[i_args + 7] * args[i_args + 6]], args[i_args + 7], &expressions_params[args[i_args + 8]][(nColsStagesAcc[args[i_args + 9]] + args[i_args + 10]) * (1 - args[i_args + 11]) * blockDim.x + args[i_args + 11] * args[i_args + 10]], args[i_args + 11]);
                        if (print)
                        {
                            // result
                            printf("Case 1\n");
                            printf("Op %lu of %d\n", kk, d_deviceArgs->dests[j].params[k].parserParams.nOps);
                            // printf("Buffer: %d %d %d \n", args[i_args + 1], args[i_args + 4], args[i_args + 8]);
                            printf("Result: %lu\n", expressions_params[args[i_args + 1]][(nColsStagesAcc[args[i_args + 2]] + args[i_args + 3]) * d_deviceArgs->nrowsPack].get_val());
                            // printf("args: %d %d %d %d %d %d %d %d %d %d %d %d\n", args[i_args], args[i_args + 1], args[i_args + 2], args[i_args + 3], args[i_args + 4], args[i_args + 5], args[i_args + 6], args[i_args + 7], args[i_args + 8], args[i_args + 9], args[i_args + 10], args[i_args + 11]);
                        }
                        i_args += 12;
                        break;
                    }
                    case 2:
                    {
                        // OPERATION WITH DEST: dim3 - SRC0: dim3 - SRC1: dim1
                        /*if( print){
                            printf("Abans args: %d %d %d %d %d %d %d %d %d %d %d %d\n", args[i_args], args[i_args + 1], args[i_args + 2], args[i_args + 3], args[i_args + 4], args[i_args + 5], args[i_args + 6], args[i_args + 7], args[i_args + 8], args[i_args + 9], args[i_args + 10], args[i_args + 11]);
                        }*/
                        Goldilocks3GPU::op_31_gpu(args[i_args], &expressions_params[args[i_args + 1]][(nColsStagesAcc[args[i_args + 2]] + args[i_args + 3]) * blockDim.x], &expressions_params[args[i_args + 4]][(nColsStagesAcc[args[i_args + 5]] + args[i_args + 6]) * (1 - args[i_args + 7]) * blockDim.x + args[i_args + 7] * args[i_args + 6]], args[i_args + 7], &expressions_params[args[i_args + 8]][(nColsStagesAcc[args[i_args + 9]] + args[i_args + 10]) * (1 - args[i_args + 11]) * blockDim.x + args[i_args + 11] * args[i_args + 10]], args[i_args + 11]);
                        if (print)
                        {
                            // result
                            printf("Case 2\n");
                            printf("Op %lu of %d\n", kk, d_deviceArgs->dests[j].params[k].parserParams.nOps);
                            // printf("Buffer: %d %d %d \n", args[i_args + 1], args[i_args + 4], args[i_args + 8]);
                            printf("Result: %lu %lu %lu\n",
                                   expressions_params[args[i_args + 1]][(nColsStagesAcc[args[i_args + 2]] + args[i_args + 3]) * d_deviceArgs->nrowsPack].get_val(),
                                   expressions_params[args[i_args + 1]][(nColsStagesAcc[args[i_args + 2]] + args[i_args + 3]) * d_deviceArgs->nrowsPack + 1].get_val(),
                                   expressions_params[args[i_args + 1]][(nColsStagesAcc[args[i_args + 2]] + args[i_args + 3]) * d_deviceArgs->nrowsPack + 2].get_val());
                        }
                        i_args += 12;
                        break;
                    }
                    case 3:
                    {
                        /*if( print){
                            printf("Abans args: %d %d %d %d %d %d %d %d %d %d %d %d\n", args[i_args], args[i_args + 1], args[i_args + 2], args[i_args + 3], args[i_args + 4], args[i_args + 5], args[i_args + 6], args[i_args + 7], args[i_args + 8], args[i_args + 9], args[i_args + 10], args[i_args + 11]);
                        }*/
                        // OPERATION WITH DEST: dim3 - SRC0: dim3 - SRC1: dim3
                        Goldilocks3GPU::op_gpu(args[i_args], &expressions_params[args[i_args + 1]][(nColsStagesAcc[args[i_args + 2]] + args[i_args + 3]) * blockDim.x], &expressions_params[args[i_args + 4]][(nColsStagesAcc[args[i_args + 5]] + args[i_args + 6]) * (1 - args[i_args + 7]) * blockDim.x + args[i_args + 7] * args[i_args + 6]], args[i_args + 7], &expressions_params[args[i_args + 8]][(nColsStagesAcc[args[i_args + 9]] + args[i_args + 10]) * (1 - args[i_args + 11]) * blockDim.x + args[i_args + 11] * args[i_args + 10]], args[i_args + 11]);
                        if (print)
                        {
                            // result
                            printf("Case 3\n");
                            printf("Op %lu of %d\n", kk, d_deviceArgs->dests[j].params[k].parserParams.nOps);
                            printf("Result: %lu %lu %lu\n",
                                   expressions_params[args[i_args + 1]][(nColsStagesAcc[args[i_args + 2]] + args[i_args + 3]) * d_deviceArgs->nrowsPack].get_val(),
                                   expressions_params[args[i_args + 1]][(nColsStagesAcc[args[i_args + 2]] + args[i_args + 3]) * d_deviceArgs->nrowsPack + 1].get_val(),
                                   expressions_params[args[i_args + 1]][(nColsStagesAcc[args[i_args + 2]] + args[i_args + 3]) * d_deviceArgs->nrowsPack + 2].get_val());
                        }
                        i_args += 12;
                        break;
                    }
                    case 4:
                    {
                        /*if( print){
                            printf("Abans args: %d %d %d %d %d %d %d %d %d %d %d %d\n", args[i_args], args[i_args + 1], args[i_args + 2], args[i_args + 3], args[i_args + 4], args[i_args + 5], args[i_args + 6], args[i_args + 7], args[i_args + 8], args[i_args + 9], args[i_args + 10], args[i_args + 11]);
                        }*/
                        // COPY dim3 to dim3
                        Goldilocks3GPU::copy_gpu(&expressions_params[args[i_args]][(nColsStagesAcc[args[i_args + 1]] + args[i_args + 2]) * blockDim.x], &expressions_params[args[i_args + 3]][(nColsStagesAcc[args[i_args + 4]] + args[i_args + 5]) * (1 - args[i_args + 6]) * blockDim.x + args[i_args + 6] * args[i_args + 5]], args[i_args + 6]);
                        if (print)
                        {
                            // result
                            printf("Case 4\n");
                            printf("Op %lu of %d\n", kk, d_deviceArgs->dests[j].params[k].parserParams.nOps);
                            printf("Result: %lu %lu %lu\n",
                                   expressions_params[args[i_args]][(nColsStagesAcc[args[i_args + 1]] + args[i_args + 2]) * d_deviceArgs->nrowsPack].get_val(),
                                   expressions_params[args[i_args]][(nColsStagesAcc[args[i_args + 1]] + args[i_args + 2]) * d_deviceArgs->nrowsPack + 1].get_val(),
                                   expressions_params[args[i_args]][(nColsStagesAcc[args[i_args + 1]] + args[i_args + 2]) * d_deviceArgs->nrowsPack + 2].get_val());
                        }
                        i_args += 7;
                        break;
                    }
                    default:
                    {
                        printf(" Wrong operation!\n");
                        assert(0);
                    }
                    }
                }

                if (d_deviceArgs->dests[j].params[k].parserParams.destDim == 1)
                {
                    copyPolynomial__(d_deviceArgs, &destVals[k * FIELD_EXTENSION * blockDim.x], d_deviceArgs->dests[j].params[k].inverse, d_deviceArgs->dests[j].params[k].parserParams.destDim, &expressions_params[1][d_deviceArgs->dests[j].params[k].parserParams.destId * blockDim.x]);
                }
                else
                {
                    copyPolynomial__(d_deviceArgs, &destVals[k * FIELD_EXTENSION * blockDim.x], d_deviceArgs->dests[j].params[k].inverse, d_deviceArgs->dests[j].params[k].parserParams.destDim, &expressions_params[6][d_deviceArgs->dests[j].params[k].parserParams.destId * FIELD_EXTENSION * blockDim.x]);
                }
            }

            gl64_t *destVals = (gl64_t *)(&d_deviceArgs->destVals[blockIdx.x * d_deviceArgs->destValsSize]);
            if (d_deviceArgs->dests[j].nParams == 2)
            {
                multiplyPolynomials__(d_deviceArgs, d_deviceArgs->dests[j], destVals);
            }
            storeOnePolynomial__(d_deviceArgs, destVals, i, j);
        }
        chunk_idx += gridDim.x;
    }
}
