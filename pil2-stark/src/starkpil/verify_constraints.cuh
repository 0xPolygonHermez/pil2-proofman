#include "verify_constraints.hpp"

void verifyConstraintsGPU(SetupCtx& setupCtx, StepsParams &params, ConstraintInfo *constraintsInfo, DeviceCommitBuffers *d_buffers) {
    
    uint64_t N = (1 << setupCtx.starkInfo.starkStruct.nBits);

    Goldilocks::Element* pBuffer = &params.aux_trace[setupCtx.starkInfo.mapOffsets[std::make_pair("q", true)]];

    ExpressionsGPU expressionsGPU(setupCtx, setupCtx.starkInfo.nrowsPack, setupCtx.starkInfo.maxNBlocks);

    uint64_t customCommitsSize = setupCtx.starkInfo.mapTotalNCustomCommitsFixed;
    uint64_t constPolsSize = setupCtx.starkInfo.nConstants * N;
    uint64_t traceSize = setupCtx.starkInfo.mapSectionsN["cm1"] * N;
    uint64_t auxTraceSize = setupCtx.starkInfo.mapTotalN;
    uint64_t nPublicsSize = setupCtx.starkInfo.nPublics;
    uint64_t proofValuesSize = setupCtx.starkInfo.proofValuesSize;
    uint64_t airvaluesSize = setupCtx.starkInfo.airValuesSize;
    uint64_t airgroupValuesSize = setupCtx.starkInfo.airgroupValuesSize;
    uint64_t challengesSize = 2 * FIELD_EXTENSION;

    gl64_t *d_trace;
    CHECKCUDAERR(cudaMalloc((void**)&d_trace, traceSize * sizeof(Goldilocks::Element)));
    CHECKCUDAERR(cudaMemcpy(d_trace, params.trace, traceSize * sizeof(Goldilocks::Element), cudaMemcpyHostToDevice));

    gl64_t *d_aux_trace;
    CHECKCUDAERR(cudaMalloc((void**)&d_aux_trace, auxTraceSize * sizeof(Goldilocks::Element)));
    CHECKCUDAERR(cudaMemcpy(d_aux_trace, params.aux_trace, auxTraceSize * sizeof(Goldilocks::Element), cudaMemcpyHostToDevice));
    
    gl64_t *d_publicInputs;
    CHECKCUDAERR(cudaMalloc((void**)&d_publicInputs, nPublicsSize * sizeof(Goldilocks::Element)));
    CHECKCUDAERR(cudaMemcpy(d_publicInputs, params.publicInputs, nPublicsSize * sizeof(Goldilocks::Element), cudaMemcpyHostToDevice));

    gl64_t *d_proofValues;
    CHECKCUDAERR(cudaMalloc((void**)&d_proofValues, proofValuesSize * sizeof(Goldilocks::Element)));
    CHECKCUDAERR(cudaMemcpy(d_proofValues, params.proofValues, proofValuesSize * sizeof(Goldilocks::Element), cudaMemcpyHostToDevice));

    gl64_t *d_airValues;
    CHECKCUDAERR(cudaMalloc((void**)&d_airValues, airvaluesSize * sizeof(Goldilocks::Element)));
    CHECKCUDAERR(cudaMemcpy(d_airValues, params.airValues, airvaluesSize * sizeof(Goldilocks::Element), cudaMemcpyHostToDevice));

    gl64_t *d_airgroupValues;
    CHECKCUDAERR(cudaMalloc((void**)&d_airgroupValues, airgroupValuesSize * sizeof(Goldilocks::Element)));
    CHECKCUDAERR(cudaMemcpy(d_airgroupValues, params.airgroupValues, airgroupValuesSize * sizeof(Goldilocks::Element), cudaMemcpyHostToDevice));

    gl64_t *d_challenges;
    CHECKCUDAERR(cudaMalloc((void**)&d_challenges, challengesSize * sizeof(Goldilocks::Element)));
    CHECKCUDAERR(cudaMemcpy(d_challenges, params.challenges, challengesSize * sizeof(Goldilocks::Element), cudaMemcpyHostToDevice));
    
    gl64_t *d_const_pols;
    CHECKCUDAERR(cudaMalloc((void**)&d_const_pols, constPolsSize * sizeof(Goldilocks::Element)));
    CHECKCUDAERR(cudaMemcpy(d_const_pols, params.pConstPolsAddress, constPolsSize * sizeof(Goldilocks::Element), cudaMemcpyHostToDevice));

    gl64_t *d_customCommits;
    CHECKCUDAERR(cudaMalloc((void**)&d_customCommits, customCommitsSize * sizeof(Goldilocks::Element)));
    CHECKCUDAERR(cudaMemcpy(d_customCommits, params.pCustomCommitsFixed, customCommitsSize * sizeof(Goldilocks::Element), cudaMemcpyHostToDevice));

    StepsParams h_params = {
        trace : (Goldilocks::Element *)d_trace,
        aux_trace : (Goldilocks::Element *)d_aux_trace,
        publicInputs : (Goldilocks::Element *)d_publicInputs,
        proofValues : (Goldilocks::Element *)d_proofValues,
        challenges : (Goldilocks::Element *)d_challenges,
        airgroupValues : (Goldilocks::Element *)d_airgroupValues,
        airValues : (Goldilocks::Element *)d_airValues,
        evals : nullptr,
        xDivXSub : nullptr,
        pConstPolsAddress: (Goldilocks::Element *)d_const_pols,
        pConstPolsExtendedTreeAddress: nullptr,
        pCustomCommitsFixed: (Goldilocks::Element *)d_customCommits,
    };

    StepsParams *d_params;
    CHECKCUDAERR(cudaMalloc((void**)&d_params, sizeof(StepsParams)));
    CHECKCUDAERR(cudaMemcpy(d_params, &h_params, sizeof(StepsParams), cudaMemcpyHostToDevice));
    
    for (uint64_t i = 0; i < setupCtx.expressionsBin.constraintsInfoDebug.size(); i++) {
        constraintsInfo[i].id = i;
        constraintsInfo[i].stage = setupCtx.expressionsBin.constraintsInfoDebug[i].stage;
        constraintsInfo[i].imPol = setupCtx.expressionsBin.constraintsInfoDebug[i].imPol;

        Goldilocks::Element *pinned_exps_params = d_buffers->streamsData[0].pinned_buffer_exps_params;
        Goldilocks::Element *pinned_exps_args = d_buffers->streamsData[0].pinned_buffer_exps_args;
        DestParamsGPU *d_destParams = d_buffers->streamsData[0].d_destParams;
        ExpsArguments *d_expsArgs = d_buffers->streamsData[0].d_expsArgs;

        cudaStream_t stream = d_buffers->streamsData[0].stream;
        TimerGPU &timer = d_buffers->streamsData[0].timer;

        if(!constraintsInfo[i].skip) {
            Goldilocks::Element *pBufferGPU = (Goldilocks::Element *)(h_params.aux_trace + setupCtx.starkInfo.mapOffsets[std::make_pair("q", true)]);
            Dest constraintDest(NULL, N, 0);
            constraintDest.addParams(i, setupCtx.expressionsBin.constraintsInfoDebug[i].destDim);
            constraintDest.dest_gpu = pBufferGPU;
            expressionsGPU.calculateExpressions_gpu(d_params, constraintDest, N, false, d_expsArgs, d_destParams, pinned_exps_params, pinned_exps_args, i, timer, stream, false, true);
            CHECKCUDAERR(cudaMemcpy(pBuffer, pBufferGPU, N * setupCtx.expressionsBin.constraintsInfoDebug[i].destDim * sizeof(Goldilocks::Element), cudaMemcpyDeviceToHost));
            verifyConstraint(setupCtx, pBuffer, i, constraintsInfo[i]);
        }
    }    
}