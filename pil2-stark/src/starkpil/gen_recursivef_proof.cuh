#include "bn128.cuh"
#include "starks.hpp"
#include "transcriptBN128.cuh"
#include "starks_gpu_bn128.cuh"
#include "cuda_utils.cuh"
#include "gl64_tooling.cuh"
#include "expressions_gpu.cuh"
#include "starks_gpu.cuh"
#include "hints.cuh"
#include "gpu_timer.cuh"
#include <iomanip>

/* Todo:
    - netejar genproof tota inicialitzacio inecessaria del starks
 */


void calculateWitnessSTD_BN128_gpu(SetupCtx& setupCtx, StepsParams& h_params, StepsParams *d_params, bool prod, ExpressionsGPU *expressionsCtxGPU, ExpsArguments *d_expsArgs, DestParamsGPU *d_destParams, Goldilocks::Element *pinned_exps_params, Goldilocks::Element *pinned_exps_args, uint64_t& countId, TimerGPU &timer, cudaStream_t stream) {

    std::string name = prod ? "gprod_col" : "gsum_col";
    if(setupCtx.expressionsBin.getNumberHintIdsByName(name) == 0) return;
    uint64_t hint[1];
    setupCtx.expressionsBin.getHintIdsByName(hint, name);

    uint64_t nImHints = setupCtx.expressionsBin.getNumberHintIdsByName("im_col");
    uint64_t nImHintsAirVals = setupCtx.expressionsBin.getNumberHintIdsByName("im_airval");
    uint64_t nImTotalHints = nImHints + nImHintsAirVals;
    if(nImTotalHints > 0) {
        uint64_t imHints[nImHints + nImHintsAirVals];
        setupCtx.expressionsBin.getHintIdsByName(imHints, "im_col");
        setupCtx.expressionsBin.getHintIdsByName(&imHints[nImHints], "im_airval");
        std::string hintFieldDest[nImTotalHints];
        std::string hintField1[nImTotalHints];
        std::string hintField2[nImTotalHints];
        HintFieldOptions hintOptions1[nImTotalHints];
        HintFieldOptions hintOptions2[nImTotalHints];
        for(uint64_t i = 0; i < nImTotalHints; i++) {
            hintFieldDest[i] = "reference";
            hintField1[i] = "numerator";
            hintField2[i] = "denominator";
            HintFieldOptions options1;
            HintFieldOptions options2;
            options2.inverse = true;
            hintOptions1[i] = options1;
            hintOptions2[i] = options2;
        }

        multiplyHintFieldsGPU(setupCtx, h_params, d_params, nImTotalHints, imHints, hintFieldDest, hintField1, hintField2, hintOptions1, hintOptions2, expressionsCtxGPU, d_expsArgs, d_destParams, pinned_exps_params, pinned_exps_args, countId, timer, stream);
    }

    HintFieldOptions options1;
    HintFieldOptions options2;
    options2.inverse = true;

    std::string hintFieldNameAirgroupVal = setupCtx.starkInfo.airgroupValuesMap.size() > 0 ? "result" : "";

    accMulHintFieldsGPU(setupCtx, h_params, d_params, hint[0], "reference", hintFieldNameAirgroupVal, "numerator_air", "denominator_air",options1, options2, !prod,expressionsCtxGPU, d_expsArgs, d_destParams, pinned_exps_params, pinned_exps_args, countId, timer, stream);
    updateAirgroupValueGPU(setupCtx, h_params, d_params, hint[0], hintFieldNameAirgroupVal, "numerator_direct", "denominator_direct", options1, options2, !prod, expressionsCtxGPU, d_expsArgs, d_destParams, pinned_exps_params, pinned_exps_args, countId, timer, stream);
}

void *genRecursiveProofBN128_gpu(SetupCtx& setupCtx, uint64_t airgroupId, uint64_t airId, uint64_t instanceId, Goldilocks::Element *d_aux_trace, Goldilocks::Element *d_constTree, std::string proofFile, cudaStream_t stream) {

    //2) hem falta AirInstanceInfo
    //2) do I need constTreePath?
    
    TimerGPU timer;
    cudaStreamSynchronize(stream);

    TimerStartGPU(timer, STARK_GPU_PROOF);
    TimerStartGPU(timer, STARK_STEP_0);

    uint64_t countId = 0;

    StepsParams *params_pinned;
    CHECKCUDAERR(cudaMallocHost((void **)&params_pinned, sizeof(StepsParams)));
    //Goldilocks::Element *proof_buffer_pinned = d_buffers->streamsData[stream_id].pinned_buffer_proof;
    Goldilocks::Element *pinned_exps_params;
    uint64_t maxExps = 20000; // TODO: CALCULATE IT PROPERLY!
    CHECKCUDAERR(cudaMallocHost((void **)&pinned_exps_params, maxExps * 2 * sizeof(DestParamsGPU)));
    Goldilocks::Element *pinned_exps_args;
    CHECKCUDAERR(cudaMallocHost((void **)&pinned_exps_args, maxExps * sizeof(ExpsArguments)));
    TranscriptBN128_GPU d_transcript(setupCtx.starkInfo.starkStruct.merkleTreeArity, setupCtx.starkInfo.starkStruct.merkleTreeCustom, stream);
    TranscriptBN128_GPU d_transcript_helper(setupCtx.starkInfo.starkStruct.merkleTreeArity, setupCtx.starkInfo.starkStruct.merkleTreeCustom, stream);

    StepsParams *d_params;
    CHECKCUDAERR(cudaMalloc((void **)&d_params, sizeof(StepsParams)));
    ExpsArguments *d_expsArgs;
    CHECKCUDAERR(cudaMalloc((void **)&d_expsArgs, maxExps * sizeof(ExpsArguments)));
    DestParamsGPU *d_destParams;
    CHECKCUDAERR(cudaMalloc((void **)&d_destParams, maxExps * 2 * sizeof(DestParamsGPU)));

    uint64_t NExtended = 1 << setupCtx.starkInfo.starkStruct.nBitsExt;

    Goldilocks::Element *pCustomCommitsFixed = (Goldilocks::Element *)d_aux_trace + setupCtx.starkInfo.mapOffsets[std::make_pair("custom_fixed", false)];
    
    Starks<RawFr::Element> starks(setupCtx, d_constTree, pCustomCommitsFixed, false, false);
    uint64_t nFieldElements = 1;
    
    uint64_t offsetCm1 = setupCtx.starkInfo.mapOffsets[std::make_pair("cm1", false)];
    uint64_t offsetPublicInputs = setupCtx.starkInfo.mapOffsets[std::make_pair("publics", false)];
    uint64_t offsetAirgroupValues = setupCtx.starkInfo.mapOffsets[std::make_pair("airgroupvalues", false)];
    uint64_t offsetAirValues = setupCtx.starkInfo.mapOffsets[std::make_pair("airvalues", false)];
    uint64_t offsetProofValues = setupCtx.starkInfo.mapOffsets[std::make_pair("proofvalues", false)];
    uint64_t offsetEvals = setupCtx.starkInfo.mapOffsets[std::make_pair("evals", false)];
    uint64_t offsetChallenges = setupCtx.starkInfo.mapOffsets[std::make_pair("challenges", false)];
    uint64_t offsetXDivXSub = setupCtx.starkInfo.mapOffsets[std::make_pair("xdivxsub", false)];
    uint64_t offsetFriQueries = setupCtx.starkInfo.mapOffsets[std::make_pair("fri_queries", false)];
    uint64_t offsetChallenge = setupCtx.starkInfo.mapOffsets[std::make_pair("challenge", false)];
    uint64_t offsetNonce = setupCtx.starkInfo.mapOffsets[std::make_pair("nonce", false)];
    uint64_t offsetNonceBlocks = setupCtx.starkInfo.mapOffsets[std::make_pair("nonce_blocks", false)];
    uint64_t offsetInputHashNonce = setupCtx.starkInfo.mapOffsets[std::make_pair("input_hash_nonce", false)];
    uint64_t offsetProofQueries = setupCtx.starkInfo.mapOffsets[std::make_pair("proof_queries", false)];
    uint64_t offsetConstPols = setupCtx.starkInfo.mapOffsets[std::make_pair("const", false)];
    uint64_t offsetQ = setupCtx.starkInfo.mapOffsets[std::make_pair("q", true)];


    StepsParams h_params = {
        trace : (Goldilocks::Element *)d_aux_trace + offsetCm1,
        aux_trace : (Goldilocks::Element *)d_aux_trace,
        publicInputs : (Goldilocks::Element *)d_aux_trace + offsetPublicInputs,
        proofValues : (Goldilocks::Element *)d_aux_trace + offsetProofValues,
        challenges : (Goldilocks::Element *)d_aux_trace + offsetChallenges,
        airgroupValues : (Goldilocks::Element *)d_aux_trace + offsetAirgroupValues,
        airValues : (Goldilocks::Element *)d_aux_trace + offsetAirValues,
        evals : (Goldilocks::Element *)d_aux_trace + offsetEvals,
        xDivXSub : (Goldilocks::Element *)d_aux_trace + offsetXDivXSub,
        pConstPolsAddress: (Goldilocks::Element *)d_aux_trace + offsetConstPols,
        pConstPolsExtendedTreeAddress:d_constTree,
        pCustomCommitsFixed: pCustomCommitsFixed,
    };

    memcpy(params_pinned, &h_params, sizeof(StepsParams));
    
    CHECKCUDAERR(cudaMemcpyAsync(d_params, params_pinned, sizeof(StepsParams), cudaMemcpyHostToDevice, stream));
        
    Goldilocks::Element *d_nonce = (Goldilocks::Element *)d_aux_trace + offsetNonce;
    Goldilocks::Element *d_nonceBlocks = (Goldilocks::Element *)d_aux_trace + offsetNonceBlocks;
    uint64_t *friQueries_gpu = (uint64_t *)d_aux_trace + offsetFriQueries;

    gl64_t *d_queries_buff = (gl64_t *)d_aux_trace + offsetProofQueries;
    uint64_t nTrees = setupCtx.starkInfo.nStages + setupCtx.starkInfo.customCommits.size() + 2;
    uint64_t nTreesFRI = setupCtx.starkInfo.starkStruct.steps.size() - 1;

    ExpressionsGPU *expressions_gpu = new ExpressionsGPU(setupCtx, setupCtx.starkInfo.nrowsPack, setupCtx.starkInfo.maxNBlocks);

    //=============================
    
    // the challenges buffer can allways be used safely (todo: double check)
    PoseidonBN128GPU::FrElement * d_hash_gpu = (PoseidonBN128GPU::FrElement *) h_params.challenges;

    d_transcript.reset(stream);
    d_transcript.put(((PoseidonBN128GPU::FrElement *) starks.treesGL[setupCtx.starkInfo.nStages + 1]->get_nodes_ptr()) + starks.treesGL[setupCtx.starkInfo.nStages + 1]->numNodes - 1, 1, stream);
    if (setupCtx.starkInfo.nPublics > 0)
    {
        if (!setupCtx.starkInfo.starkStruct.hashCommits)
        {
            d_transcript.put(h_params.publicInputs, setupCtx.starkInfo.nPublics, stream);
        }
        else
        {
            calculateHashBN128_gpu(&d_transcript_helper, d_hash_gpu, setupCtx, h_params.publicInputs, setupCtx.starkInfo.nPublics, stream);
            d_transcript.put(d_hash_gpu, nFieldElements, stream);
        }
    }
    TimerStopGPU(timer, STARK_STEP_0);
    
    TimerStartGPU(timer, STARK_STEP_1);
    for (uint64_t i = 0; i < setupCtx.starkInfo.challengesMap.size(); i++)
    {
        if (setupCtx.starkInfo.challengesMap[i].stage == 1)
        {
            d_transcript.getField((uint64_t *)&h_params.challenges[i * FIELD_EXTENSION], stream);
        }
    }
    TimerStopGPU(timer, STARK_STEP_1);

    TimerStartGPU(timer, STARK_COMMIT_STAGE_1);
    commitStage_bn128_gpu(1, setupCtx, starks.treesGL, h_params.trace, d_aux_trace, &d_transcript, timer, stream);     
    TimerStopGPU(timer, STARK_COMMIT_STAGE_1);

    TimerStartGPU(timer, STARK_CALCULATE_WITNESS_STD);
    for (uint64_t i = 0; i < setupCtx.starkInfo.challengesMap.size(); i++)
    {
        if (setupCtx.starkInfo.challengesMap[i].stage == 2)
        {
            d_transcript.getField((uint64_t *)&h_params.challenges[i * FIELD_EXTENSION], stream);
        }
    }
    calculateWitnessSTD_BN128_gpu(setupCtx, h_params, d_params, true, expressions_gpu, d_expsArgs, d_destParams, pinned_exps_params, pinned_exps_args, countId, timer, stream);
    calculateWitnessSTD_BN128_gpu(setupCtx, h_params, d_params, false, expressions_gpu, d_expsArgs, d_destParams, pinned_exps_params, pinned_exps_args, countId, timer, stream);
    TimerStopGPU(timer, STARK_CALCULATE_WITNESS_STD);

    TimerStartGPU(timer, CALCULATE_IM_POLS);
    calculateImPolsExpressions(setupCtx, expressions_gpu, h_params, d_params, 2, d_expsArgs, d_destParams, pinned_exps_params, pinned_exps_args, countId, timer, stream);
    TimerStopGPU(timer, CALCULATE_IM_POLS);

    TimerStartGPU(timer, STARK_COMMIT_STAGE_2);
    commitStage_bn128_gpu(2, setupCtx, starks.treesGL, h_params.trace, d_aux_trace, &d_transcript, timer, stream);     
    TimerStopGPU(timer, STARK_COMMIT_STAGE_2);

    TimerStartGPU(timer, STARK_STEP_Q);
    for (uint64_t i = 0; i < setupCtx.starkInfo.challengesMap.size(); i++)
    {
        if(setupCtx.starkInfo.challengesMap[i].stage == 3) {
            d_transcript.getField((uint64_t *)&h_params.challenges[i * FIELD_EXTENSION], stream);
        }
    }
    uint64_t zi_offset = setupCtx.starkInfo.mapOffsets[std::make_pair("zi", true)];
    computeZerofier(h_params.aux_trace + zi_offset, setupCtx.starkInfo.starkStruct.nBits, setupCtx.starkInfo.starkStruct.nBitsExt, stream);
    TimerStartGPU(timer, STARK_QUOTIENT_POLYNOMIAL);
    calculateExpressionQ(setupCtx,expressions_gpu, d_params, h_params.aux_trace + offsetQ, d_expsArgs, d_destParams, pinned_exps_params, pinned_exps_args, countId, timer, stream);
    TimerStopGPU(timer, STARK_QUOTIENT_POLYNOMIAL);
    commitStage_bn128_gpu(3, setupCtx, starks.treesGL, h_params.trace, d_aux_trace, &d_transcript, timer, stream); 
    TimerStopGPU(timer, STARK_STEP_Q);

    TimerStartGPU(timer, STARK_STEP_EVALS);
    
    uint64_t xiChallengeIndex = 0;
    for (uint64_t i = 0; i < setupCtx.starkInfo.challengesMap.size(); i++)
    {
        if(setupCtx.starkInfo.challengesMap[i].stage == 4) {
            if(setupCtx.starkInfo.challengesMap[i].stageId == 0) xiChallengeIndex = i;
            d_transcript.getField((uint64_t *)&h_params.challenges[i * FIELD_EXTENSION], stream);
        }
    }
    TimerStopGPU(timer, STARK_STEP_EVALS);

    // Print GPU challenges (Stage 1-4) for comparison with CPU
    {
        cudaStreamSynchronize(stream);
        uint64_t numChallenges = setupCtx.starkInfo.challengesMap.size();
        std::vector<Goldilocks::Element> h_challenges(numChallenges * FIELD_EXTENSION);
        cudaMemcpy(h_challenges.data(), h_params.challenges, numChallenges * FIELD_EXTENSION * sizeof(Goldilocks::Element), cudaMemcpyDeviceToHost);
        
        std::cout << "=== GPU Challenges (Stage 1-3) ===" << std::endl;
        for (uint64_t i = 0; i < numChallenges; i++) {
            if (setupCtx.starkInfo.challengesMap[i].stage <= 4) {
                std::cout << "Challenge[" << i << "]: ";
                for (uint64_t j = 0; j < FIELD_EXTENSION; j++) {
                    std::cout << Goldilocks::toU64(h_challenges[i * FIELD_EXTENSION + j]);
                    if (j < FIELD_EXTENSION - 1) std::cout << ", ";
                }
                std::cout << std::endl;
            }
        }
        std::cout << "==================================" << std::endl;
    }

    TimerStopGPU(timer,STARK_GPU_PROOF);

    // free allocated pinned memory
    cudaFreeHost(params_pinned);
    cudaFreeHost(pinned_exps_params);
    cudaFreeHost(pinned_exps_args);

    //free deice memory allocated
    cudaFree(d_params);
    cudaFree(d_expsArgs);
    cudaFree(d_destParams);





    // Free stark trees
    /*for (uint64_t i = 0; i < setupCtx.starkInfo.nStages + setupCtx.starkInfo.customCommits.size() + 2; i++)
    {
       cudaFree(starks.treesGL[i]->get_nodes_ptr());
    }*/
    return nullptr;
}
