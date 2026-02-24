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
#include <thread>



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

void *genRecursiveProofBN128_gpu(SetupCtx& setupCtx, uint64_t airgroupId, uint64_t airId, uint64_t instanceId, Goldilocks::Element *d_aux_trace, Goldilocks::Element *h_publicInputs, std::string proofFile, DeviceRecursiveFBuffers *d_buffers) {

    Goldilocks::Element *d_constTree = (Goldilocks::Element *)d_buffers->d_const_tree;
    cudaStream_t stream = d_buffers->stream;
    TimerGPU &timer = d_buffers->timer;
    
    TimerStartGPU(timer, STARK_GPU_PROOF);
    TimerStartGPU(timer, STARK_STEP_0);

    AirInstanceInfo *air_instance_info = new AirInstanceInfo(airgroupId, airId, &setupCtx, nullptr, nullptr);

    uint64_t countId = 0;

    StepsParams *params_pinned = d_buffers->params_pinned;
    Goldilocks::Element *pinned_exps_params = d_buffers->pinned_exps_params;
    Goldilocks::Element *pinned_exps_args = d_buffers->pinned_exps_args;
    TranscriptBN128_GPU d_transcript(setupCtx.starkInfo.starkStruct.merkleTreeArity, setupCtx.starkInfo.starkStruct.merkleTreeCustom, stream);
    TranscriptBN128_GPU d_transcript_helper(setupCtx.starkInfo.starkStruct.merkleTreeArity, setupCtx.starkInfo.starkStruct.merkleTreeCustom, stream);

    StepsParams *d_params = d_buffers->d_params;
    ExpsArguments *d_expsArgs = d_buffers->d_expsArgs;
    DestParamsGPU *d_destParams = d_buffers->d_destParams;

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
    uint64_t offsetFRI = setupCtx.starkInfo.mapOffsets[std::make_pair("f", true)];
    uint64_t offsetHelper = setupCtx.starkInfo.mapOffsets[std::make_pair("buff_helper", false)];


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

    uint64_t nTrees = setupCtx.starkInfo.nStages + setupCtx.starkInfo.customCommits.size() + 2;
    uint64_t nTreesFRI = setupCtx.starkInfo.starkStruct.steps.size() - 1;

    gl64_t *d_queries_buff = (gl64_t *)d_aux_trace + offsetProofQueries;
    
    // Use input_hash_nonce buffer as temporary storage for BN128 hash output
    PoseidonBN128GPU::FrElement * d_hash_gpu = (PoseidonBN128GPU::FrElement *)((Goldilocks::Element *)d_aux_trace + offsetInputHashNonce);

    d_transcript.reset(stream);

    // Use verification key from d_buffers instead of const tree root
    d_transcript.put((PoseidonBN128GPU::FrElement *)d_buffers->d_verkey, 1, stream, &timer);
    if (setupCtx.starkInfo.nPublics > 0)
    {
        if (!setupCtx.starkInfo.starkStruct.hashCommits)
        {
            d_transcript.put(h_params.publicInputs, setupCtx.starkInfo.nPublics, stream, &timer);
        }
        else
        {
            calculateHashBN128_gpu(&d_transcript_helper, d_hash_gpu, setupCtx, h_params.publicInputs, setupCtx.starkInfo.nPublics, stream);
            d_transcript.put(d_hash_gpu, nFieldElements, stream, &timer);
        }
    }
    TimerStopGPU(timer, STARK_STEP_0);
    
    TimerStartGPU(timer, STARK_COMMIT_STAGE_1);
    for (uint64_t i = 0; i < setupCtx.starkInfo.challengesMap.size(); i++)
    {
        if (setupCtx.starkInfo.challengesMap[i].stage == 1)
        {
            d_transcript.getField((uint64_t *)&h_params.challenges[i * FIELD_EXTENSION], stream);
        }
    }
    commitStage_bn128_gpu(1, setupCtx, starks.treesGL, h_params.trace, d_aux_trace, &d_transcript, timer, stream);     
    TimerStopGPU(timer, STARK_COMMIT_STAGE_1);

    TimerStartGPU(timer, STARK_CALCULATE_WITNESS_STD);
    for (uint64_t i = 0; i < setupCtx.starkInfo.challengesMap.size(); i++)
    {
        if (setupCtx.starkInfo.challengesMap[i].stage == setupCtx.starkInfo.nStages )
        {
            d_transcript.getField((uint64_t *)&h_params.challenges[i * FIELD_EXTENSION], stream);
        }
    }
    calculateWitnessSTD_BN128_gpu(setupCtx, h_params, d_params, true, air_instance_info->expressions_gpu, d_expsArgs, d_destParams, pinned_exps_params, pinned_exps_args, countId, timer, stream);
    calculateWitnessSTD_BN128_gpu(setupCtx, h_params, d_params, false, air_instance_info->expressions_gpu, d_expsArgs, d_destParams, pinned_exps_params, pinned_exps_args, countId, timer, stream);
    TimerStopGPU(timer, STARK_CALCULATE_WITNESS_STD);

    TimerStartGPU(timer, CALCULATE_IM_POLS);
    calculateImPolsExpressions(setupCtx, air_instance_info->expressions_gpu, h_params, d_params, 2, d_expsArgs, d_destParams, pinned_exps_params, pinned_exps_args, countId, timer, stream);
    TimerStopGPU(timer, CALCULATE_IM_POLS);

    TimerStartGPU(timer, STARK_COMMIT_STAGE_2);
    commitStage_bn128_gpu(2, setupCtx, starks.treesGL, h_params.trace, d_aux_trace, &d_transcript, timer, stream);     
    TimerStopGPU(timer, STARK_COMMIT_STAGE_2);

    TimerStartCategoryGPU(timer, LOAD_TREE_WAIT);
        // Wait for background thread to finish const tree copy (CPU blocks here until copy is done)
        while (!d_buffers->const_tree_loaded.load(std::memory_order_acquire)) {
            std::this_thread::yield();
        }
    TimerStopCategoryGPU(timer, LOAD_TREE_WAIT)

    TimerStartGPU(timer, STARK_STEP_Q);
    for (uint64_t i = 0; i < setupCtx.starkInfo.challengesMap.size(); i++)
    {
        if(setupCtx.starkInfo.challengesMap[i].stage == setupCtx.starkInfo.nStages + 1 ) {
            d_transcript.getField((uint64_t *)&h_params.challenges[i * FIELD_EXTENSION], stream);
        }
    }
    uint64_t zi_offset = setupCtx.starkInfo.mapOffsets[std::make_pair("zi", true)];
    computeZerofier(h_params.aux_trace + zi_offset, setupCtx.starkInfo.starkStruct.nBits, setupCtx.starkInfo.starkStruct.nBitsExt, stream);
    TimerStartGPU(timer, STARK_QUOTIENT_POLYNOMIAL);
    calculateExpressionQ(setupCtx,air_instance_info->expressions_gpu, d_params, h_params.aux_trace + offsetQ, d_expsArgs, d_destParams, pinned_exps_params, pinned_exps_args, countId, timer, stream);
    TimerStopGPU(timer, STARK_QUOTIENT_POLYNOMIAL);
    commitStage_bn128_gpu(3, setupCtx, starks.treesGL, h_params.trace, d_aux_trace, &d_transcript, timer, stream); 
    TimerStopGPU(timer, STARK_STEP_Q);

    TimerStartGPU(timer, STARK_STEP_EVALS);
    TimerStartCategoryGPU(timer, EVALS);

    uint64_t xiChallengeIndex = 0;
    for (uint64_t i = 0; i < setupCtx.starkInfo.challengesMap.size(); i++)
    {
        if(setupCtx.starkInfo.challengesMap[i].stage == setupCtx.starkInfo.nStages + 2) {
            if(setupCtx.starkInfo.challengesMap[i].stageId == 0) xiChallengeIndex = i;
            d_transcript.getField((uint64_t *)&h_params.challenges[i * FIELD_EXTENSION], stream);
        }
    }
    Goldilocks::Element *d_xiChallenge = &h_params.challenges[xiChallengeIndex * FIELD_EXTENSION];
    gl64_t * d_LEv = (gl64_t *) h_params.aux_trace +setupCtx.starkInfo.mapOffsets[std::make_pair("lev", false)];

    CHECKCUDAERR(cudaMemsetAsync(h_params.evals, 0, setupCtx.starkInfo.evMap.size() * FIELD_EXTENSION * sizeof(Goldilocks::Element), stream));
    uint64_t count = 0;
    TimerStopCategoryGPU(timer, EVALS);
    for(uint64_t i = 0; i < setupCtx.starkInfo.openingPoints.size(); i += 4) {
        std::vector<int64_t> openingPoints;
        for(uint64_t j = 0; j < 4; ++j) {
            if(i + j < setupCtx.starkInfo.openingPoints.size()) {
                openingPoints.push_back(setupCtx.starkInfo.openingPoints[i + j]);
            }
        }
        uint64_t offset_helper = setupCtx.starkInfo.mapOffsets[std::make_pair("extra_helper_fft_lev", false)];
        computeLEv_inplace(d_xiChallenge, setupCtx.starkInfo.starkStruct.nBits, openingPoints.size(), &air_instance_info->opening_points[i], (gl64_t*) d_aux_trace, offset_helper, d_LEv, timer, stream);
        evmap_inplace(setupCtx, h_params, count++, openingPoints.size(), openingPoints.data(), air_instance_info, (Goldilocks::Element*)d_LEv, offset_helper, timer, stream);
    }
    if(!setupCtx.starkInfo.starkStruct.hashCommits) {
        d_transcript.put(h_params.evals, setupCtx.starkInfo.evMap.size() * FIELD_EXTENSION, stream, &timer);
    } else {
        calculateHashBN128_gpu(&d_transcript_helper, d_hash_gpu, setupCtx, h_params.evals, setupCtx.starkInfo.evMap.size() * FIELD_EXTENSION, stream);
        d_transcript.put(d_hash_gpu, nFieldElements, stream, &timer);
    }
    
    TimerStopGPU(timer, STARK_STEP_EVALS);
    //--------------------------------
    // 6. Compute FRI
    //--------------------------------
    TimerStartGPU(timer, STARK_STEP_FRI);
    TimerStartCategoryGPU(timer, FRI);
    // Challenges for FRI polynomial
    for (uint64_t i = 0; i < setupCtx.starkInfo.challengesMap.size(); i++)
    {
        if(setupCtx.starkInfo.challengesMap[i].stage == setupCtx.starkInfo.nStages + 3) {
            d_transcript.getField((uint64_t *)&h_params.challenges[i * FIELD_EXTENSION], stream);
        }
    }
    calculateXis_inplace(setupCtx, h_params, air_instance_info->opening_points, d_xiChallenge, stream);    
    uint64_t x_offset = setupCtx.starkInfo.mapOffsets[std::make_pair("x", true)];
    dim3 threads(256);
    dim3 blocks((NExtended + threads.x - 1) / threads.x);
    computeX_kernel<<<blocks, threads, 0, stream>>>((gl64_t *)h_params.aux_trace + x_offset, NExtended, Goldilocks::shift(), Goldilocks::w(setupCtx.starkInfo.starkStruct.nBitsExt));
    TimerStopCategoryGPU(timer, FRI);
    TimerStartGPU(timer, STARK_FRI_POLYNOMIAL);
    TimerStartCategoryGPU(timer, EXPRESSIONS);
    calculateFRIExpression(setupCtx, h_params, air_instance_info, stream);
    TimerStopCategoryGPU(timer, EXPRESSIONS);

    TimerStopGPU(timer, STARK_FRI_POLYNOMIAL);    
    Goldilocks::Element *d_friPol = (Goldilocks::Element *)(h_params.aux_trace + offsetFRI);    
    uint64_t nBits =  setupCtx.starkInfo.starkStruct.steps[0].nBits;

    uint64_t nStepsFRI = setupCtx.starkInfo.starkStruct.steps.size();
    for (uint64_t step = 0; step < nStepsFRI; step++)
    {
        uint64_t currentBits = setupCtx.starkInfo.starkStruct.steps[step].nBits;
        if (step > 0) {
            uint64_t prevBits = setupCtx.starkInfo.starkStruct.steps[step - 1].nBits;
            fold_inplace(step, offsetFRI, offsetHelper, h_params.challenges, nBits, prevBits, currentBits, (gl64_t *)d_aux_trace, timer, stream);
        }
        if (step < nStepsFRI - 1)
        {
            merkelizeFRI_bn128_gpu(setupCtx, h_params, step, d_friPol, starks.treesFRI[step], currentBits, setupCtx.starkInfo.starkStruct.steps[step + 1].nBits, &d_transcript, timer, stream);
        }
        else
        {
            if(!setupCtx.starkInfo.starkStruct.hashCommits) {
                d_transcript.put((Goldilocks::Element *)d_friPol, (1 << setupCtx.starkInfo.starkStruct.steps[step].nBits) * FIELD_EXTENSION, stream, &timer);
            } else {
                calculateHashBN128_gpu(&d_transcript_helper, d_hash_gpu, setupCtx, (Goldilocks::Element *)d_friPol, (1 << setupCtx.starkInfo.starkStruct.steps[step].nBits) * FIELD_EXTENSION, stream);
                d_transcript.put(d_hash_gpu, nFieldElements, stream, &timer);
            }
        }
        d_transcript.getField((uint64_t *)h_params.challenges, stream);
    }

    TimerStartCategoryGPU(timer, GRINDING);
    PoseidonBN128GPU::FrElement *d_grinding_state = (PoseidonBN128GPU::FrElement *)(d_aux_trace + offsetInputHashNonce);
    convertGLToBN128ScalarField(d_grinding_state, (const uint64_t *)h_params.challenges, FIELD_EXTENSION, stream);
    CHECKCUDAERR(cudaGetLastError());
    
    PoseidonBN128GPU::grinding((uint64_t *)d_nonce, (uint64_t *)d_nonceBlocks, d_grinding_state, setupCtx.starkInfo.starkStruct.powBits, stream);
    CHECKCUDAERR(cudaGetLastError());
    TimerStopCategoryGPU(timer, GRINDING);
    TimerStartCategoryGPU(timer, FRI);

    d_transcript_helper.reset(stream);
    d_transcript_helper.put(h_params.challenges, FIELD_EXTENSION, stream);
    d_transcript_helper.put(d_nonce, 1, stream);
    d_transcript_helper.getPermutations(friQueries_gpu, setupCtx.starkInfo.starkStruct.nQueries, setupCtx.starkInfo.starkStruct.steps[0].nBits, stream);

    proveQueries_bn128_gpu(setupCtx, d_queries_buff, friQueries_gpu, setupCtx.starkInfo.starkStruct.nQueries, starks.treesGL, nTrees, (gl64_t*)d_aux_trace, setupCtx.starkInfo.nStages, stream);

    for(uint64_t step = 0; step < setupCtx.starkInfo.starkStruct.steps.size() - 1; ++step) {
        proveFRIQueries_bn128_gpu(setupCtx, &d_queries_buff[(nTrees + step) * setupCtx.starkInfo.starkStruct.nQueries * setupCtx.starkInfo.maxProofBuffSize], step + 1, setupCtx.starkInfo.starkStruct.steps[step + 1].nBits, friQueries_gpu, setupCtx.starkInfo.starkStruct.nQueries, starks.treesFRI[step], stream);
    }
    TimerStopCategoryGPU(timer, FRI);

    TimerStopGPU(timer, STARK_STEP_FRI);

    //--------------------------------
    // 7. Generate proof file
    //--------------------------------
    TimerStartGPU(timer, STARK_SAVE_PROOF);
    FRIProof<RawFr::Element> proof(setupCtx.starkInfo, airgroupId, airId, instanceId);    
    
    setProof_bn128_gpu(
        starks,
        proof,
        d_aux_trace,
        stream
    );

    nlohmann::json zkin = proof.proof.proof2json();
    zkin["publics"] = nlohmann::json::array();
    for(uint64_t i = 0; i < setupCtx.starkInfo.nPublics; ++i) {
        zkin["publics"][i] = Goldilocks::toString(h_publicInputs[i]);
    }

    if(!proofFile.empty()) {
        json2file(zkin, proofFile);
    }
    TimerStopGPU(timer, STARK_SAVE_PROOF);
    TimerStartGPU(timer, STARK_CLEANUP);

    // Free stark trees (skip constants tree at nStages+1 - passed in from outside)
    for (uint64_t i = 0; i < setupCtx.starkInfo.nStages + setupCtx.starkInfo.customCommits.size() + 2; i++)
    {
        if (i != setupCtx.starkInfo.nStages + 1) {
            cudaFree(starks.treesGL[i]->get_nodes_ptr());
        }
    }
    // free FRI trees
    for (uint64_t i = 0; i < nTreesFRI; i++)
    {
        cudaFree(starks.treesFRI[i]->get_nodes_ptr());
    }

    // Free AirInstanceInfo (and all its GPU allocations)
    delete air_instance_info;

    TimerStopGPU(timer, STARK_CLEANUP);
    TimerStopGPU(timer,STARK_GPU_PROOF);
    TimerSyncAndLogAllGPU(timer, instanceId, airgroupId, airId);
    TimerSyncCategoriesGPU(timer);
    TimerLogCategoryContributionsGPU(timer, STARK_GPU_PROOF);
    return (void *) new nlohmann::json(zkin);
}
