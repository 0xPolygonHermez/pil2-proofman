#ifndef GEN_RECURSIVE_PROOF_GPU_HPP
#define GEN_RECURSIVE_PROOF_GPU_HPP

#include "starks.hpp"
#include "proof2zkinStark.hpp"
#include "cuda_utils.cuh"
#include "gl64_t.cuh"
#include "expressions_gpu.cuh"
#include "starks_gpu.cuh"
#include <iomanip>

// TOTO list: 
// fer que lo dls params vagi igual
// evitar copies inecetssaries
// fer que lo dels arbres vagi igual (primer arreglar els de gen_proof)

template <typename ElementType>
void genRecursiveProof_gpu(SetupCtx &setupCtx, gl64_t *d_trace, gl64_t *d_aux_trace, gl64_t *d_const_pols, gl64_t *d_const_tree, uint32_t stream_id, DeviceCommitBuffers *d_buffers, AirInstanceInfo *air_instance_info, TimerGPU &timer, cudaStream_t stream)
{
    TimerStartGPU(timer, STARK_GPU_PROOF);
    TimerStartGPU(timer, STARK_STEP_0);
    
    StepsParams *params_pinned = d_buffers->streamsData[stream_id].pinned_params;
    Goldilocks::Element *proof_buffer_pinned = d_buffers->streamsData[stream_id].pinned_buffer_proof;
    TranscriptGL_GPU *d_transcript = d_buffers->streamsData[stream_id].transcript;
    TranscriptGL_GPU *d_transcript_helper = d_buffers->streamsData[stream_id].transcript_helper;
    StepsParams *d_params =  d_buffers->streamsData[stream_id].params;
    ExpsArguments *d_expsArgs = d_buffers->streamsData[stream_id].d_expsArgs;
    DestParamsGPU *d_destParams = d_buffers->streamsData[stream_id].d_destParams;

    uint64_t N = 1 << setupCtx.starkInfo.starkStruct.nBits;
    uint64_t NExtended = 1 << setupCtx.starkInfo.starkStruct.nBitsExt;

    Goldilocks::Element *pConstPolsExtendedTreeAddress = (Goldilocks::Element *)d_const_tree;

    Starks<Goldilocks::Element> starks(setupCtx, nullptr, nullptr, false);
    starks.treesGL[setupCtx.starkInfo.nStages + 1]->setSource(&pConstPolsExtendedTreeAddress[2]);
    starks.treesGL[setupCtx.starkInfo.nStages + 1]->setNodes(&pConstPolsExtendedTreeAddress[2 + setupCtx.starkInfo.nConstants * NExtended]);

    uint64_t offsetPublicInputs = setupCtx.starkInfo.mapOffsets[std::make_pair("publics", false)];
    uint64_t offsetEvals = setupCtx.starkInfo.mapOffsets[std::make_pair("evals", false)];
    uint64_t offsetChallenges = setupCtx.starkInfo.mapOffsets[std::make_pair("challenges", false)];
    uint64_t offsetXDivXSub = setupCtx.starkInfo.mapOffsets[std::make_pair("xdivxsub", false)];
    uint64_t offsetFriQueries = setupCtx.starkInfo.mapOffsets[std::make_pair("fri_queries", false)];
    uint64_t offsetChallenge = setupCtx.starkInfo.mapOffsets[std::make_pair("challenge", false)];
    uint64_t offsetProofQueries = setupCtx.starkInfo.mapOffsets[std::make_pair("proof_queries", false)];

    StepsParams h_params = {
        trace : (Goldilocks::Element *)d_trace,
        aux_trace : (Goldilocks::Element *)d_aux_trace,
        publicInputs : (Goldilocks::Element *)d_aux_trace + offsetPublicInputs,
        proofValues : nullptr,
        challenges : (Goldilocks::Element *)d_aux_trace + offsetChallenges,
        airgroupValues : nullptr,
        airValues : nullptr,
        evals : (Goldilocks::Element *)d_aux_trace + offsetEvals,
        xDivXSub : (Goldilocks::Element *)d_aux_trace + offsetXDivXSub,
        pConstPolsAddress : (Goldilocks::Element *)d_const_pols,
        pConstPolsExtendedTreeAddress,
        pCustomCommitsFixed : nullptr,
    };
    
    *params_pinned = h_params;
    
    CHECKCUDAERR(cudaMemcpyAsync(d_params, params_pinned, sizeof(StepsParams), cudaMemcpyHostToDevice, stream));

    uint64_t *friQueries_gpu = (uint64_t *)d_aux_trace + offsetFriQueries;

    Goldilocks::Element *challenge_gpu = (Goldilocks::Element *)d_aux_trace + offsetChallenge;

    gl64_t *d_queries_buff = (gl64_t *)d_aux_trace + offsetProofQueries;
    uint64_t nTrees = setupCtx.starkInfo.nStages + setupCtx.starkInfo.customCommits.size() + 2;
    uint64_t nTreesFRI = setupCtx.starkInfo.starkStruct.steps.size() - 1;

    //--------------------------------
    // 0.- Add const root and publics to transcript
    //--------------------------------
    d_transcript->reset(stream);
    d_transcript->put(starks.treesGL[setupCtx.starkInfo.nStages+1]->get_nodes_ptr() + starks.treesGL[setupCtx.starkInfo.nStages + 1]->numNodes - HASH_SIZE, HASH_SIZE, stream);
    if (setupCtx.starkInfo.nPublics > 0)
    {
        if (!setupCtx.starkInfo.starkStruct.hashCommits)
        {
            d_transcript->put(h_params.publicInputs, setupCtx.starkInfo.nPublics, stream);
        }
        else
        {
            calculateHash(d_transcript_helper, challenge_gpu, setupCtx, h_params.publicInputs, setupCtx.starkInfo.nPublics, stream);
            d_transcript->put(challenge_gpu, HASH_SIZE, stream);
        }
    }
    for (uint64_t i = 0; i < setupCtx.starkInfo.challengesMap.size(); i++)
    {
        if (setupCtx.starkInfo.challengesMap[i].stage == 1)
        {
            d_transcript->getField((uint64_t *)&h_params.challenges[i * FIELD_EXTENSION], stream);
        }
    }
    TimerStopGPU(timer, STARK_STEP_0);
    TimerStartGPU(timer, STARK_COMMIT_STAGE_1);
    commitStage_inplace(1, setupCtx, starks.treesGL, d_trace, d_aux_trace, d_transcript, false, timer, stream);
    for (uint64_t i = 0; i < setupCtx.starkInfo.challengesMap.size(); i++)
    {
        if (setupCtx.starkInfo.challengesMap[i].stage == 2)
        {
            d_transcript->getField((uint64_t *)&h_params.challenges[i * FIELD_EXTENSION], stream);
        }
    }
    TimerStopGPU(timer, STARK_COMMIT_STAGE_1);


    TimerStartGPU(timer, STARK_CALCULATE_GPROD);

    uint64_t gprodFieldId = setupCtx.expressionsBin.hints[0].fields[0].values[0].id;
    uint64_t numFieldId = setupCtx.expressionsBin.hints[0].fields[1].values[0].id;
    uint64_t denFieldId = setupCtx.expressionsBin.hints[0].fields[2].values[0].id;

    uint64_t offsetAuxTrace = setupCtx.starkInfo.mapOffsets[std::make_pair("q", true)];
    Goldilocks::Element* vals_gpu = h_params.aux_trace + offsetAuxTrace;
    Goldilocks::Element* vals_gpu_shifted = vals_gpu + FIELD_EXTENSION;
    Goldilocks::Element* helpers = h_params.aux_trace + offsetAuxTrace + FIELD_EXTENSION*N + FIELD_EXTENSION;  

    Dest destStruct(nullptr, N);
    destStruct.dest_gpu = vals_gpu_shifted;
    destStruct.addParams(numFieldId, setupCtx.expressionsBin.expressionsInfo[numFieldId].destDim);
    destStruct.addParams(denFieldId, setupCtx.expressionsBin.expressionsInfo[denFieldId].destDim, true);
    uint64_t xn_offset = setupCtx.starkInfo.mapOffsets[std::make_pair("x_n", false)];
    dim3 threads_(256);
    dim3 blocks_((N + threads_.x - 1) / threads_.x);
    computeX_kernel<<<blocks_, threads_, 0, stream>>>((gl64_t *)h_params.aux_trace + xn_offset, N, Goldilocks::one(), Goldilocks::w(setupCtx.starkInfo.starkStruct.nBits));
    air_instance_info->expressions_gpu->calculateExpressions_gpu(d_params, destStruct, uint64_t(1 << setupCtx.starkInfo.starkStruct.nBits), false, d_expsArgs, d_destParams, timer, stream);

    accOperationGPU((gl64_t *)vals_gpu_shifted, N, false, FIELD_EXTENSION, (gl64_t *)helpers, stream);
   
    setProdIdentity3<<<1,1, 0, stream>>>((gl64_t *)vals_gpu);
    setPolynomialGPU(setupCtx, h_params.aux_trace, vals_gpu, gprodFieldId, stream);

    TimerStopGPU(timer,STARK_CALCULATE_GPROD);

    TimerStartGPU(timer, CALCULATE_IM_POLS);
    calculateImPolsExpressions(setupCtx, air_instance_info->expressions_gpu, h_params, d_params, 2, d_expsArgs, d_destParams, timer, stream);
    TimerStopGPU(timer, CALCULATE_IM_POLS);


    TimerStartGPU(timer, STARK_COMMIT_STAGE_2);
    commitStage_inplace(2, setupCtx, starks.treesGL, (gl64_t*)h_params.trace, (gl64_t*)h_params.aux_trace, d_transcript, false, timer, stream);
    for (uint64_t i = 0; i < setupCtx.starkInfo.challengesMap.size(); i++)
    {
        if (setupCtx.starkInfo.challengesMap[i].stage == setupCtx.starkInfo.nStages + 1)
        {
            d_transcript->getField((uint64_t *)&h_params.challenges[i * FIELD_EXTENSION], stream);        
        }
    }
    TimerStopGPU(timer, STARK_COMMIT_STAGE_2);
    TimerStartGPU(timer, STARK_STEP_Q);
    
    uint64_t zi_offset = setupCtx.starkInfo.mapOffsets[std::make_pair("zi", true)];
    uint64_t x_offset = setupCtx.starkInfo.mapOffsets[std::make_pair("x", true)];
    computeZerofier(h_params.aux_trace + zi_offset, setupCtx.starkInfo.starkStruct.nBits, setupCtx.starkInfo.starkStruct.nBitsExt, stream);
    dim3 threads_x(256);
    dim3 blocks_x((NExtended + threads_x.x - 1) / threads_x.x);
    computeX_kernel<<<blocks_x, threads_x, 0, stream>>>((gl64_t *)h_params.aux_trace + x_offset, NExtended, Goldilocks::shift(), Goldilocks::w(setupCtx.starkInfo.starkStruct.nBitsExt));            
    calculateExpression(setupCtx, air_instance_info->expressions_gpu, d_params, (Goldilocks::Element *)(h_params.aux_trace + setupCtx.starkInfo.mapOffsets[std::make_pair("q", true)]), setupCtx.starkInfo.cExpId, false, d_expsArgs, d_destParams, timer, stream);

    commitStage_inplace(setupCtx.starkInfo.nStages + 1, setupCtx, starks.treesGL, nullptr, d_aux_trace, d_transcript, false, timer, stream);
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
    gl64_t * d_LEv = (gl64_t *)  h_params.aux_trace +setupCtx.starkInfo.mapOffsets[std::make_pair("lev", false)];;

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
        evmap_inplace(setupCtx, h_params, count++, openingPoints.size(), openingPoints.data(), air_instance_info, (Goldilocks::Element*)d_LEv, timer, stream);
    }

    if(!setupCtx.starkInfo.starkStruct.hashCommits) {
        d_transcript->put(h_params.evals, setupCtx.starkInfo.evMap.size() * FIELD_EXTENSION, stream);
    } else {
        calculateHash(d_transcript_helper, challenge_gpu, setupCtx, h_params.evals, setupCtx.starkInfo.evMap.size() * FIELD_EXTENSION, stream);
        d_transcript->put(challenge_gpu, HASH_SIZE, stream);
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

    computeX_kernel<<<blocks_x, threads_x, 0, stream>>>((gl64_t *)h_params.aux_trace + x_offset, NExtended, Goldilocks::shift(), Goldilocks::w(setupCtx.starkInfo.starkStruct.nBitsExt));
    calculateExpression(setupCtx, air_instance_info->expressions_gpu, d_params, (Goldilocks::Element *)(h_params.aux_trace + setupCtx.starkInfo.mapOffsets[std::make_pair("f", true)]), setupCtx.starkInfo.friExpId, false, d_expsArgs, d_destParams, timer, stream);
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
        fold_inplace(step, friPol_offset, offset_helper, challenge_gpu, nBitsExt, prevBits, currentBits, d_aux_trace, timer, stream);

        if (step < setupCtx.starkInfo.starkStruct.steps.size() - 1)
        {
            merkelizeFRI_inplace(setupCtx, h_params, step, d_friPol, starks.treesFRI[step], currentBits, setupCtx.starkInfo.starkStruct.steps[step + 1].nBits, d_transcript, timer, stream);
        }
        else
        {
            if(!setupCtx.starkInfo.starkStruct.hashCommits) {
                d_transcript->put((Goldilocks::Element *)d_friPol, (1 << setupCtx.starkInfo.starkStruct.steps[step].nBits) * FIELD_EXTENSION, stream);
            } else {
                calculateHash(d_transcript_helper, challenge_gpu, setupCtx, (Goldilocks::Element *)d_friPol, (1 << setupCtx.starkInfo.starkStruct.steps[step].nBits) * FIELD_EXTENSION, stream);
                d_transcript->put(challenge_gpu, HASH_SIZE, stream);
            }
        }
        d_transcript->getField((uint64_t *)challenge_gpu, stream);
    }
   
    TimerStartCategoryGPU(timer, FRI);
    d_transcript_helper->reset(stream);
    d_transcript_helper->put(challenge_gpu, FIELD_EXTENSION, stream);
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
