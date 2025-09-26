#ifndef GEN_PROOF_CUH
#define GEN_PROOF_CUH

#include "starks.hpp"
#include "cuda_utils.cuh"
#include "gl64_tooling.cuh"
#include "expressions_gpu.cuh"
#include "starks_gpu.cuh"
#include "hints.cuh"
#include "gpu_timer.cuh"
#include <iomanip>

// TOTO list: //rick
// carregar-me els d_trees
// _inplace not good name


void calculateWitnessSTD_gpu(SetupCtx& setupCtx, StepsParams& h_params, StepsParams *d_params, bool prod, ExpressionsGPU *expressionsCtxGPU, ExpsArguments *d_expsArgs, DestParamsGPU *d_destParams, Goldilocks::Element *pinned_exps_params, Goldilocks::Element *pinned_exps_args, uint64_t& countId, TimerGPU &timer, cudaStream_t stream) {

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

void genProof_gpu(SetupCtx& setupCtx, gl64_t *d_aux_trace, gl64_t *d_const_pols, gl64_t *d_const_tree, char *constTreePath, uint32_t stream_id, uint64_t instance_id, DeviceCommitBuffers *d_buffers, AirInstanceInfo *air_instance_info, bool skipRecalculation, TimerGPU &timer, cudaStream_t stream, bool recursive = false) {
    TimerStartGPU(timer, STARK_GPU_PROOF);
    TimerStartGPU(timer, STARK_STEP_0);

    uint64_t countId = 0;

    StepsParams *params_pinned = d_buffers->streamsData[stream_id].pinned_params;
    Goldilocks::Element *proof_buffer_pinned = d_buffers->streamsData[stream_id].pinned_buffer_proof;
    Goldilocks::Element *pinned_exps_params = d_buffers->streamsData[stream_id].pinned_buffer_exps_params;
    Goldilocks::Element *pinned_exps_params_q = d_buffers->streamsData[stream_id].pinned_buffer_exps_params_q;
    Goldilocks::Element *pinned_exps_args = d_buffers->streamsData[stream_id].pinned_buffer_exps_args;
    TranscriptGL_GPU *d_transcript = d_buffers->streamsData[stream_id].transcript;
    TranscriptGL_GPU *d_transcript_helper = d_buffers->streamsData[stream_id].transcript_helper;
    StepsParams *d_params =  d_buffers->streamsData[stream_id].params;
    ExpsArguments *d_expsArgs = d_buffers->streamsData[stream_id].d_expsArgs;
    DestParamsGPU *d_destParams = d_buffers->streamsData[stream_id].d_destParams;
    Goldilocks::Element *d_challengePowers = d_buffers->streamsData[stream_id].d_challengePowers;

    uint64_t N = 1 << setupCtx.starkInfo.starkStruct.nBits;
    uint64_t NExtended = 1 << setupCtx.starkInfo.starkStruct.nBitsExt;

    Goldilocks::Element *pConstPolsAddress = (Goldilocks::Element *)d_const_pols;
    Goldilocks::Element *pConstPolsExtendedTreeAddress = (Goldilocks::Element *)d_const_tree;
    Goldilocks::Element *pCustomCommitsFixed = (Goldilocks::Element *)d_aux_trace + setupCtx.starkInfo.mapOffsets[std::make_pair("custom_fixed", false)];
    
    Starks<Goldilocks::Element> starks(setupCtx, nullptr, nullptr, false);
    starks.treesGL[setupCtx.starkInfo.nStages + 1]->setSource(&pConstPolsExtendedTreeAddress[2]);
    starks.treesGL[setupCtx.starkInfo.nStages + 1]->setNodes(&pConstPolsExtendedTreeAddress[2 + setupCtx.starkInfo.nConstants * NExtended]);
    for(uint64_t i = 0; i < setupCtx.starkInfo.customCommits.size(); i++) {
        uint64_t nCols = setupCtx.starkInfo.mapSectionsN[setupCtx.starkInfo.customCommits[i].name + "0"];
            starks.treesGL[setupCtx.starkInfo.nStages + 2 + i]->setSource(&pCustomCommitsFixed[N * nCols]);
            starks.treesGL[setupCtx.starkInfo.nStages + 2 + i]->setNodes(&pCustomCommitsFixed[(N + NExtended) * nCols]);
    }

    uint64_t nFieldElements = setupCtx.starkInfo.starkStruct.verificationHashType == std::string("BN128") ? 1 : HASH_SIZE;
    
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
    uint64_t offsetProofQueries = setupCtx.starkInfo.mapOffsets[std::make_pair("proof_queries", false)];


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
        pConstPolsAddress,
        pConstPolsExtendedTreeAddress,
        pCustomCommitsFixed,
    };

    memcpy(params_pinned, &h_params, sizeof(StepsParams));
    
    CHECKCUDAERR(cudaMemcpyAsync(d_params, params_pinned, sizeof(StepsParams), cudaMemcpyHostToDevice, stream));
    
    Goldilocks::Element *d_challenge = (Goldilocks::Element *)d_aux_trace + offsetChallenge;
           
    uint64_t *friQueries_gpu = (uint64_t *)d_aux_trace + offsetFriQueries;

    gl64_t *d_queries_buff = (gl64_t *)d_aux_trace + offsetProofQueries;
    uint64_t nTrees = setupCtx.starkInfo.nStages + setupCtx.starkInfo.customCommits.size() + 2;
    uint64_t nTreesFRI = setupCtx.starkInfo.starkStruct.steps.size() - 1;

    d_transcript->reset(stream);
    if (recursive) {
        d_transcript->put(starks.treesGL[setupCtx.starkInfo.nStages+1]->get_nodes_ptr() + starks.treesGL[setupCtx.starkInfo.nStages + 1]->numNodes - HASH_SIZE, HASH_SIZE, stream);
        if (setupCtx.starkInfo.nPublics > 0)
        {
            if (!setupCtx.starkInfo.starkStruct.hashCommits)
            {
                d_transcript->put(h_params.publicInputs, setupCtx.starkInfo.nPublics, stream);
            }
            else
            {
                calculateHash(d_transcript_helper, h_params.challenges, setupCtx, h_params.publicInputs, setupCtx.starkInfo.nPublics, stream);
                d_transcript->put(h_params.challenges, HASH_SIZE, stream);
            }
        }
    } else {
       d_transcript->put(d_challenge, FIELD_EXTENSION, stream); 
    }
    
    TimerStopGPU(timer, STARK_STEP_0);
    
    TimerStartGPU(timer, STARK_COMMIT_STAGE_1);
    if (recursive) {
        commitStage_inplace(1, setupCtx, starks.treesGL, (gl64_t*) h_params.trace, (gl64_t*)h_params.aux_trace, d_transcript, false, timer, stream);
    } else {
        commitStage_inplace(1, setupCtx, starks.treesGL, (gl64_t*) h_params.trace, (gl64_t*)h_params.aux_trace, nullptr, skipRecalculation, timer, stream);
    }
    TimerStopGPU(timer, STARK_COMMIT_STAGE_1);

    TimerStartGPU(timer, STARK_CALCULATE_WITNESS_STD);
    for (uint64_t i = 0; i < setupCtx.starkInfo.challengesMap.size(); i++) {
        if(setupCtx.starkInfo.challengesMap[i].stage == 2) {
            d_transcript->getField((uint64_t *)&h_params.challenges[i * FIELD_EXTENSION], stream);
        }
    }
    calculateWitnessSTD_gpu(setupCtx, h_params, d_params, true, air_instance_info->expressions_gpu, d_expsArgs, d_destParams, pinned_exps_params, pinned_exps_args, countId, timer, stream);
    calculateWitnessSTD_gpu(setupCtx, h_params, d_params, false, air_instance_info->expressions_gpu, d_expsArgs, d_destParams, pinned_exps_params, pinned_exps_args, countId, timer, stream);

    TimerStopGPU(timer, STARK_CALCULATE_WITNESS_STD);

    TimerStartGPU(timer, CALCULATE_IM_POLS);
    calculateImPolsExpressions(setupCtx, air_instance_info->expressions_gpu, h_params, d_params, 2, d_expsArgs, d_destParams, pinned_exps_params, pinned_exps_args, countId, timer, stream);
    TimerStopGPU(timer, CALCULATE_IM_POLS);
    
    TimerStartGPU(timer, STARK_COMMIT_STAGE_2);
    commitStage_inplace(2, setupCtx, starks.treesGL, (gl64_t*)h_params.trace, (gl64_t*)h_params.aux_trace, d_transcript, false, timer, stream);

    uint64_t a = 0;
    for(uint64_t i = 0; i < setupCtx.starkInfo.airValuesMap.size(); i++) {
        if(setupCtx.starkInfo.airValuesMap[i].stage == 1) a++;
        if(setupCtx.starkInfo.airValuesMap[i].stage == 2) {
            d_transcript->put(&h_params.airValues[a], FIELD_EXTENSION, stream);
            a += 3;
        }
    }
    TimerStopGPU(timer, STARK_COMMIT_STAGE_2);
    TimerStartGPU(timer, STARK_STEP_Q);
    for (uint64_t i = 0; i < setupCtx.starkInfo.challengesMap.size(); i++)
    {
        if(setupCtx.starkInfo.challengesMap[i].stage == setupCtx.starkInfo.nStages + 1) {
            d_transcript->getField((uint64_t *)&h_params.challenges[i * FIELD_EXTENSION], stream);
        }
    }
    uint64_t zi_offset = setupCtx.starkInfo.mapOffsets[std::make_pair("zi", true)];
    computeZerofier(h_params.aux_trace + zi_offset, setupCtx.starkInfo.starkStruct.nBits, setupCtx.starkInfo.starkStruct.nBitsExt, stream);

    if (setupCtx.starkInfo.calculateFixedExtended) {
        extendAndMerkelizeFixed(setupCtx, pConstPolsAddress, pConstPolsExtendedTreeAddress, timer, stream);
    }

    TimerStartGPU(timer, STARK_QUOTIENT_POLYNOMIAL);
    if (recursive) {
        //todo_q: fix max_n_constraints for pinned bufers
        calculateExpression(setupCtx, air_instance_info->expressions_gpu, d_params, (Goldilocks::Element *)(h_params.aux_trace + setupCtx.starkInfo.mapOffsets[std::make_pair("q", true)]), setupCtx.starkInfo.cExpId, false, d_expsArgs, d_destParams, pinned_exps_params, pinned_exps_args, countId, timer, stream);
     } else{
        calculateExpressionQ(setupCtx, air_instance_info->expressions_gpu_q, d_params, (Goldilocks::Element *)(h_params.aux_trace + setupCtx.starkInfo.mapOffsets[std::make_pair("q", true)]), setupCtx.starkInfo.cExpId, d_expsArgs, d_destParams, pinned_exps_params_q, pinned_exps_args, d_challengePowers, countId, timer, stream);
    }
    TimerStopGPU(timer, STARK_QUOTIENT_POLYNOMIAL);
    commitStage_inplace(setupCtx.starkInfo.nStages + 1, setupCtx, starks.treesGL, (gl64_t *)h_params.trace, (gl64_t *)h_params.aux_trace, d_transcript, false, timer, stream);
    TimerStopGPU(timer, STARK_STEP_Q);
    TimerStartGPU(timer, STARK_STEP_EVALS);
    
    uint64_t xiChallengeIndex = 0;
    for (uint64_t i = 0; i < setupCtx.starkInfo.challengesMap.size(); i++)
    {
        if(setupCtx.starkInfo.challengesMap[i].stage == setupCtx.starkInfo.nStages + 2) {
            if(setupCtx.starkInfo.challengesMap[i].stageId == 0) xiChallengeIndex = i;
            d_transcript->getField((uint64_t *)&h_params.challenges[i * FIELD_EXTENSION], stream);
        }
    }

    Goldilocks::Element *d_xiChallenge = &h_params.challenges[xiChallengeIndex * FIELD_EXTENSION];
    gl64_t * d_LEv = (gl64_t *) h_params.aux_trace +setupCtx.starkInfo.mapOffsets[std::make_pair("lev", false)];

    CHECKCUDAERR(cudaMemsetAsync(h_params.evals, 0, setupCtx.starkInfo.evMap.size() * FIELD_EXTENSION * sizeof(Goldilocks::Element), stream));
    uint64_t count = 0;
    for(uint64_t i = 0; i < setupCtx.starkInfo.openingPoints.size(); i += 4) {
        std::vector<int64_t> openingPoints;
        for(uint64_t j = 0; j < 4; ++j) {
            if(i + j < setupCtx.starkInfo.openingPoints.size()) {
                openingPoints.push_back(setupCtx.starkInfo.openingPoints[i + j]);
            }
        }
        uint64_t offset_helper = setupCtx.starkInfo.mapOffsets[std::make_pair("extra_helper_fft_lev", false)];
        computeLEv_inplace(d_xiChallenge, setupCtx.starkInfo.starkStruct.nBits, openingPoints.size(), &air_instance_info->opening_points[i], d_aux_trace, offset_helper, d_LEv, timer, stream);
        evmap_inplace(setupCtx, h_params, count++, openingPoints.size(), openingPoints.data(), air_instance_info, (Goldilocks::Element*)d_LEv, offset_helper, timer, stream);
    }
    
    if(!setupCtx.starkInfo.starkStruct.hashCommits) {
        d_transcript->put(h_params.evals, setupCtx.starkInfo.evMap.size() * FIELD_EXTENSION, stream);
    } else {
        calculateHash(d_transcript_helper, d_challenge, setupCtx, h_params.evals, setupCtx.starkInfo.evMap.size() * FIELD_EXTENSION, stream);
        d_transcript->put(d_challenge, HASH_SIZE, stream);
    }

    // Challenges for FRI polynomial
    for (uint64_t i = 0; i < setupCtx.starkInfo.challengesMap.size(); i++)
    {
        if(setupCtx.starkInfo.challengesMap[i].stage == setupCtx.starkInfo.nStages + 3) {
            d_transcript->getField((uint64_t *)&h_params.challenges[i * FIELD_EXTENSION], stream);
        }
    }
    TimerStopGPU(timer, STARK_STEP_EVALS);
    //--------------------------------
    // 6. Compute FRI
    //--------------------------------
    TimerStartGPU(timer, STARK_STEP_FRI);
    calculateXis_inplace(setupCtx, h_params, air_instance_info->opening_points, d_xiChallenge, stream);    
    uint64_t x_offset = setupCtx.starkInfo.mapOffsets[std::make_pair("x", true)];
    dim3 threads(256);
    dim3 blocks((NExtended + threads.x - 1) / threads.x);
    computeX_kernel<<<blocks, threads, 0, stream>>>((gl64_t *)h_params.aux_trace + x_offset, NExtended, Goldilocks::shift(), Goldilocks::w(setupCtx.starkInfo.starkStruct.nBitsExt));
    TimerStartGPU(timer, STARK_FRI_POLYNOMIAL);
    // calculateExpression(setupCtx, air_instance_info->expressions_gpu, d_params, (Goldilocks::Element *)(h_params.aux_trace + setupCtx.starkInfo.mapOffsets[std::make_pair("f", true)]), setupCtx.starkInfo.friExpId, false, d_expsArgs, d_destParams, pinned_exps_params, pinned_exps_args, countId, timer, stream, air_instance_info->airId == 0);
    TimerStartCategoryGPU(timer, EXPRESSIONS);
    calculateFRIExpression(setupCtx, h_params, air_instance_info, stream);
    TimerStopCategoryGPU(timer, EXPRESSIONS);
    TimerStopGPU(timer, STARK_FRI_POLYNOMIAL);
    for(uint64_t step = 0; step < setupCtx.starkInfo.starkStruct.steps.size() - 1; ++step) { 
        Goldilocks::Element *src = h_params.aux_trace + setupCtx.starkInfo.mapOffsets[std::make_pair("fri_" + to_string(step + 1), true)];
        starks.treesFRI[step]->setSource(src);

        if(setupCtx.starkInfo.starkStruct.verificationHashType == "GL") {
            Goldilocks::Element *pBuffNodesGL = h_params.aux_trace + setupCtx.starkInfo.mapOffsets[std::make_pair("mt_fri_" + to_string(step + 1), true)];
            starks.treesFRI[step]->setNodes(pBuffNodesGL);
        }
    }
    uint64_t friPol_offset = setupCtx.starkInfo.mapOffsets[std::make_pair("f", true)];
    uint64_t offset_helper = setupCtx.starkInfo.mapOffsets[std::make_pair("buff_helper", false)];
    gl64_t *d_friPol = (gl64_t *)(h_params.aux_trace + friPol_offset);
    
    uint64_t nBitsExt =  setupCtx.starkInfo.starkStruct.steps[0].nBits;

    for (uint64_t step = 0; step < setupCtx.starkInfo.starkStruct.steps.size(); step++)
    {
        uint64_t currentBits = setupCtx.starkInfo.starkStruct.steps[step].nBits;
        uint64_t prevBits = step == 0 ? currentBits : setupCtx.starkInfo.starkStruct.steps[step - 1].nBits;
        fold_inplace(step, friPol_offset, offset_helper, d_challenge, nBitsExt, prevBits, currentBits, d_aux_trace, timer, stream);
        if (step < setupCtx.starkInfo.starkStruct.steps.size() - 1)
        {
            merkelizeFRI_inplace(setupCtx, h_params, step, d_friPol, starks.treesFRI[step], currentBits, setupCtx.starkInfo.starkStruct.steps[step + 1].nBits, d_transcript, timer, stream);
        }
        else
        {
            if(!setupCtx.starkInfo.starkStruct.hashCommits) {
                d_transcript->put((Goldilocks::Element *)d_friPol, (1 << setupCtx.starkInfo.starkStruct.steps[step].nBits) * FIELD_EXTENSION, stream);
            } else {
                calculateHash(d_transcript_helper, d_challenge, setupCtx, (Goldilocks::Element *)d_friPol, (1 << setupCtx.starkInfo.starkStruct.steps[step].nBits) * FIELD_EXTENSION, stream);
                d_transcript->put(d_challenge, HASH_SIZE, stream);
            }
        }
        d_transcript->getField((uint64_t *)d_challenge, stream);
    }   
    TimerStartCategoryGPU(timer, FRI);
    d_transcript_helper->reset(stream);
    d_transcript_helper->put(d_challenge, FIELD_EXTENSION, stream);
    d_transcript_helper->getPermutations(friQueries_gpu, setupCtx.starkInfo.starkStruct.nQueries, setupCtx.starkInfo.starkStruct.steps[0].nBits, stream);

    proveQueries_inplace(setupCtx, d_queries_buff, friQueries_gpu, setupCtx.starkInfo.starkStruct.nQueries, starks.treesGL, nTrees, d_aux_trace, d_const_tree, setupCtx.starkInfo.nStages, stream);
    for(uint64_t step = 0; step < setupCtx.starkInfo.starkStruct.steps.size() - 1; ++step) {
        proveFRIQueries_inplace(setupCtx, &d_queries_buff[(nTrees + step) * setupCtx.starkInfo.starkStruct.nQueries * setupCtx.starkInfo.maxProofBuffSize], step + 1, setupCtx.starkInfo.starkStruct.steps[step + 1].nBits, friQueries_gpu, setupCtx.starkInfo.starkStruct.nQueries, starks.treesFRI[step], stream);
    }
    TimerStopCategoryGPU(timer, FRI);
    TimerStopGPU(timer, STARK_STEP_FRI);

    setProof(setupCtx, (Goldilocks::Element *)d_aux_trace, proof_buffer_pinned, stream);

    TimerStopGPU(timer, STARK_GPU_PROOF);
}

#endif