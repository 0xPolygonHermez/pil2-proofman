#ifndef GEN_PROOF_CUH
#define GEN_PROOF_CUH

#include "starks.hpp"
#include "cuda_utils.cuh"
#include "gl64_t.cuh"
#include "expressions_gpu.cuh"
#include "starks_gpu.cuh"
#include "gpu_timer.hpp"
#include <iomanip>

// TOTO list: //rick
// carregar-me els d_trees
// _inplace not good name
#define PRINT_TIME_SUMMARY 1


void calculateWitnessSTD_gpu(SetupCtx& setupCtx, StepsParams& h_params, StepsParams& d_params, bool prod, ExpressionsGPU *expressionsCtxGPU, cudaStream_t stream = 0) {

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

        multiplyHintFieldsGPU(setupCtx, h_params, d_params, nImTotalHints, imHints, hintFieldDest, hintField1, hintField2, hintOptions1, hintOptions2, expressionsCtxGPU, stream);
    }

    HintFieldOptions options1;
    HintFieldOptions options2;
    options2.inverse = true;
    accMulHintFieldsGPU(setupCtx, h_params, d_params, hint[0], "reference", "result", "numerator_air", "denominator_air",options1, options2, !prod,expressionsCtxGPU, stream);
    updateAirgroupValueGPU(setupCtx, h_params, d_params, hint[0], "result", "numerator_direct", "denominator_direct", options1, options2, !prod, expressionsCtxGPU, stream);
}

void genProof_gpu(SetupCtx& setupCtx, gl64_t *d_aux_trace, TimerGPU &timer, cudaStream_t stream = 0) {
    
    TimerStart(GEN_PROOF_GPU);
    TimerStartGPU(timer, STARK_GPU_PROOF);
    TimerStartGPU(timer, STARK_INITIALIZATION);

    uint64_t N = 1 << setupCtx.starkInfo.starkStruct.nBits;
    uint64_t NExtended = 1 << setupCtx.starkInfo.starkStruct.nBitsExt;

    uint64_t offsetConstTree = setupCtx.starkInfo.mapOffsets[std::make_pair("const", true)];

    Goldilocks::Element *pConstPolsExtendedTreeAddress = (Goldilocks::Element *)d_aux_trace + offsetConstTree;
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
    
    ExpressionsGPU expressionsCtx(setupCtx, setupCtx.starkInfo.nrowsPack, setupCtx.starkInfo.maxNBlocks);

    uint64_t offsetCm1 = setupCtx.starkInfo.mapOffsets[std::make_pair("cm1", false)];
    uint64_t offsetConstPols = setupCtx.starkInfo.mapOffsets[std::make_pair("const", false)];
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
        pConstPolsAddress : (Goldilocks::Element *)d_aux_trace + offsetConstPols,
        pConstPolsExtendedTreeAddress,
        pCustomCommitsFixed,
    };
    
    StepsParams* d_params;
    CHECKCUDAERR(cudaMalloc(&d_params, sizeof(StepsParams)));
    CHECKCUDAERR(cudaMemcpy(d_params, &h_params, sizeof(StepsParams), cudaMemcpyHostToDevice));
    
    int64_t *d_openingPoints;
    cudaMalloc(&d_openingPoints, setupCtx.starkInfo.openingPoints.size() * sizeof(int64_t));
    cudaMemcpy(d_openingPoints, setupCtx.starkInfo.openingPoints.data(), setupCtx.starkInfo.openingPoints.size() * sizeof(int64_t), cudaMemcpyHostToDevice);


    Goldilocks::Element *d_challenge = (Goldilocks::Element *)d_aux_trace + offsetChallenge;
           
    uint64_t *friQueries_gpu = (uint64_t *)d_aux_trace + offsetFriQueries;
    
    TranscriptGL_GPU d_transcript(setupCtx.starkInfo.starkStruct.merkleTreeArity, setupCtx.starkInfo.starkStruct.merkleTreeCustom, stream);
    TranscriptGL_GPU d_transcript_helper(setupCtx.starkInfo.starkStruct.merkleTreeArity, setupCtx.starkInfo.starkStruct.merkleTreeCustom, stream);

    prepare_evmap(setupCtx, d_aux_trace);

    gl64_t *d_queries_buff = (gl64_t *)d_aux_trace + offsetProofQueries;
    uint64_t nTrees = setupCtx.starkInfo.nStages + setupCtx.starkInfo.customCommits.size() + 2;
    uint64_t nTreesFRI = setupCtx.starkInfo.starkStruct.steps.size() - 1;

    TimerStopGPU(timer, STARK_INITIALIZATION);
    TimerStartGPU(timer, STARK_STEP_0);
    d_transcript.put(d_challenge, FIELD_EXTENSION, stream);
    for (uint64_t i = 0; i < setupCtx.starkInfo.challengesMap.size(); i++) {
        if(setupCtx.starkInfo.challengesMap[i].stage == 2) {
            d_transcript.getField((uint64_t *)&h_params.challenges[i * FIELD_EXTENSION], stream);
        }
    }
    TimerStopGPU(timer, STARK_STEP_0);
    
    TimerStartGPU(timer, STARK_CALCULATE_WITNESS_STD);
    calculateWitnessSTD_gpu(setupCtx, h_params, *d_params, true, &expressionsCtx, stream);
    calculateWitnessSTD_gpu(setupCtx, h_params, *d_params, false, &expressionsCtx, stream);

    TimerStopGPU(timer, STARK_CALCULATE_WITNESS_STD);

    TimerStartGPU(timer, CALCULATE_IM_POLS);
    calculateImPolsExpressions(setupCtx, expressionsCtx, h_params, *d_params, 2, stream);
    TimerStopGPU(timer, CALCULATE_IM_POLS);

    TimerStartGPU(timer, STARK_COMMIT_STAGE_1);
    commitStage_inplace(1, setupCtx, starks.treesGL, (gl64_t*) h_params.trace, (gl64_t*)h_params.aux_trace, nullptr, timer, stream);
    TimerStopGPU(timer, STARK_COMMIT_STAGE_1);
    
    TimerStartGPU(timer, STARK_COMMIT_STAGE_2);
    commitStage_inplace(2, setupCtx, starks.treesGL, (gl64_t*)h_params.trace, (gl64_t*)h_params.aux_trace, &d_transcript, timer, stream);

    uint64_t a = 0;
    for(uint64_t i = 0; i < setupCtx.starkInfo.airValuesMap.size(); i++) {
        if(setupCtx.starkInfo.airValuesMap[i].stage == 1) a++;
        if(setupCtx.starkInfo.airValuesMap[i].stage == 2) {
            d_transcript.put(&h_params.airValues[a], FIELD_EXTENSION, stream);
            a += 3;
        }
    }
    TimerStopGPU(timer, STARK_COMMIT_STAGE_2);
    TimerStartGPU(timer, STARK_STEP_Q);
    TimerStartGPU(timer, STARK_STEP_Q_EXPRESSIONS);
    for (uint64_t i = 0; i < setupCtx.starkInfo.challengesMap.size(); i++)
    {
        if(setupCtx.starkInfo.challengesMap[i].stage == setupCtx.starkInfo.nStages + 1) {
            d_transcript.getField((uint64_t *)&h_params.challenges[i * FIELD_EXTENSION], stream);
        }
    }
    uint64_t zi_offset = setupCtx.starkInfo.mapOffsets[std::make_pair("zi", true)];
    computeZerofier(h_params.aux_trace + zi_offset, setupCtx.starkInfo.starkStruct.nBits, setupCtx.starkInfo.starkStruct.nBitsExt, stream);
    calculateExpression(setupCtx, expressionsCtx, d_params, (Goldilocks::Element *)(h_params.aux_trace + setupCtx.starkInfo.mapOffsets[std::make_pair("q", true)]), setupCtx.starkInfo.cExpId, false, stream);
    TimerStopGPU(timer, STARK_STEP_Q_EXPRESSIONS);
    TimerStartGPU(timer, STARK_STEP_Q_COMMIT);
    commitStage_inplace(setupCtx.starkInfo.nStages + 1, setupCtx, starks.treesGL, (gl64_t *)h_params.trace, (gl64_t *)h_params.aux_trace, &d_transcript, timer);

    TimerStopGPU(timer, STARK_STEP_Q_COMMIT);
    TimerStopGPU(timer, STARK_STEP_Q);
    TimerStartGPU(timer, STARK_STEP_EVALS);

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
    for(uint64_t i = 0; i < setupCtx.starkInfo.openingPoints.size(); i += 4) {
        std::vector<int64_t> openingPoints;
        for(uint64_t j = 0; j < 4; ++j) {
            if(i + j < setupCtx.starkInfo.openingPoints.size()) {
                openingPoints.push_back(setupCtx.starkInfo.openingPoints[i + j]);
            }
        }
        uint64_t offset_helper = setupCtx.starkInfo.mapOffsets[std::make_pair("extra_helper_fft_lev", false)];
        computeLEv_inplace(d_xiChallenge, setupCtx.starkInfo.starkStruct.nBits, openingPoints.size(), &d_openingPoints[i], d_aux_trace, offset_helper, d_LEv, timer, stream);
        evmap_inplace(setupCtx, h_params, count++, openingPoints.size(), openingPoints.data(), (Goldilocks::Element*)d_LEv, timer, stream);
    }

    if(!setupCtx.starkInfo.starkStruct.hashCommits) {
        d_transcript.put(h_params.evals, setupCtx.starkInfo.evMap.size() * FIELD_EXTENSION, stream);
    } else {
        calculateHash(&d_transcript_helper, d_challenge, setupCtx, h_params.evals, setupCtx.starkInfo.evMap.size() * FIELD_EXTENSION, stream);
        d_transcript.put(d_challenge, HASH_SIZE, stream);
    }

    // Challenges for FRI polynomial
    for (uint64_t i = 0; i < setupCtx.starkInfo.challengesMap.size(); i++)
    {
        if(setupCtx.starkInfo.challengesMap[i].stage == setupCtx.starkInfo.nStages + 3) {
            d_transcript.getField((uint64_t *)&h_params.challenges[i * FIELD_EXTENSION], stream);
        }
    }
    TimerStopGPU(timer, STARK_STEP_EVALS);
    //--------------------------------
    // 6. Compute FRI
    //--------------------------------
    TimerStartGPU(timer, STARK_STEP_FRI);
    TimerStartGPU(timer, STARK_STEP_FRI_XIS);

    calculateXis_inplace(setupCtx, h_params, d_openingPoints, d_xiChallenge, stream);    
    TimerStopGPU(timer, STARK_STEP_FRI_XIS);

    TimerStartGPU(timer, STARK_STEP_FRI_POLYNOMIAL);
    uint64_t x_offset = setupCtx.starkInfo.mapOffsets[std::make_pair("x", true)];
    dim3 threads(256);
    dim3 blocks((NExtended + threads.x - 1) / threads.x);
    computeX_kernel<<<blocks, threads, 0, stream>>>((gl64_t *)h_params.aux_trace + x_offset, NExtended, Goldilocks::shift(), Goldilocks::w(setupCtx.starkInfo.starkStruct.nBitsExt));
    calculateExpression(setupCtx, expressionsCtx, d_params, (Goldilocks::Element *)(h_params.aux_trace + setupCtx.starkInfo.mapOffsets[std::make_pair("f", true)]), setupCtx.starkInfo.friExpId, false, stream);
    for(uint64_t step = 0; step < setupCtx.starkInfo.starkStruct.steps.size() - 1; ++step) { 
        Goldilocks::Element *src = h_params.aux_trace + setupCtx.starkInfo.mapOffsets[std::make_pair("fri_" + to_string(step + 1), true)];
        starks.treesFRI[step]->setSource(src);

        if(setupCtx.starkInfo.starkStruct.verificationHashType == "GL") {
            Goldilocks::Element *pBuffNodesGL = h_params.aux_trace + setupCtx.starkInfo.mapOffsets[std::make_pair("mt_fri_" + to_string(step + 1), true)];
            starks.treesFRI[step]->setNodes(pBuffNodesGL);
        }
    }
    TimerStopGPU(timer, STARK_STEP_FRI_POLYNOMIAL);
    TimerStartGPU(timer, STARK_STEP_FRI_FOLDING);

    uint64_t friPol_offset = setupCtx.starkInfo.mapOffsets[std::make_pair("f", true)];
    uint64_t offset_helper = setupCtx.starkInfo.mapOffsets[std::make_pair("buff_helper", false)];
    gl64_t *d_friPol = (gl64_t *)(h_params.aux_trace + friPol_offset);
    
    uint64_t nBitsExt =  setupCtx.starkInfo.starkStruct.steps[0].nBits;

    for (uint64_t step = 0; step < setupCtx.starkInfo.starkStruct.steps.size(); step++)
    {
        uint64_t currentBits = setupCtx.starkInfo.starkStruct.steps[step].nBits;
        uint64_t prevBits = step == 0 ? currentBits : setupCtx.starkInfo.starkStruct.steps[step - 1].nBits;
        fold_inplace(step, friPol_offset, offset_helper, d_challenge, nBitsExt, prevBits, currentBits, d_aux_trace, stream);

        if (step < setupCtx.starkInfo.starkStruct.steps.size() - 1)
        {
            merkelizeFRI_inplace(setupCtx, h_params, step, d_friPol, starks.treesFRI[step], currentBits, setupCtx.starkInfo.starkStruct.steps[step + 1].nBits, &d_transcript, timer, stream);
        }
        else
        {
            if(!setupCtx.starkInfo.starkStruct.hashCommits) {
                d_transcript.put((Goldilocks::Element *)d_friPol, (1 << setupCtx.starkInfo.starkStruct.steps[step].nBits) * FIELD_EXTENSION, stream);
            } else {
                calculateHash(&d_transcript_helper, d_challenge, setupCtx, (Goldilocks::Element *)d_friPol, (1 << setupCtx.starkInfo.starkStruct.steps[step].nBits) * FIELD_EXTENSION, stream);
                d_transcript.put(d_challenge, HASH_SIZE, stream);
            }
        }
        d_transcript.getField((uint64_t *)d_challenge, stream);
    }
    TimerStopGPU(timer, STARK_STEP_FRI_FOLDING);
   
    TimerStartGPU(timer, STARK_STEP_FRI_QUERIES);
    d_transcript_helper.reset(stream);
    d_transcript_helper.put(d_challenge, FIELD_EXTENSION, stream);
    d_transcript_helper.getPermutations(friQueries_gpu, setupCtx.starkInfo.starkStruct.nQueries, setupCtx.starkInfo.starkStruct.steps[0].nBits, stream);

    proveQueries_inplace(setupCtx, d_queries_buff, friQueries_gpu, setupCtx.starkInfo.starkStruct.nQueries, starks.treesGL, nTrees, d_aux_trace, d_aux_trace + offsetConstTree, setupCtx.starkInfo.nStages, stream);
    for(uint64_t step = 0; step < setupCtx.starkInfo.starkStruct.steps.size() - 1; ++step) {
        proveFRIQueries_inplace(setupCtx, &d_queries_buff[(nTrees + step) * setupCtx.starkInfo.starkStruct.nQueries * setupCtx.starkInfo.maxProofBuffSize], step + 1, setupCtx.starkInfo.starkStruct.steps[step + 1].nBits, friQueries_gpu, setupCtx.starkInfo.starkStruct.nQueries, starks.treesFRI[step], stream);
    }
    TimerStopGPU(timer, STARK_STEP_FRI_QUERIES);
    TimerStopGPU(timer, STARK_STEP_FRI);
    TimerStopGPU(timer, STARK_GPU_PROOF);

    CHECKCUDAERR(cudaFree(d_params));
    CHECKCUDAERR(cudaFree(d_openingPoints));

    cudaStreamSynchronize(stream);
    
    TimerSyncAndLogAllGPU(timer); 

    TimerSyncCategoriesGPU(timer);
    
    TimerStopAndLog(GEN_PROOF_GPU);

#if PRINT_TIME_SUMMARY

    double time_total = TimerGetElapsedGPU(timer, STARK_GPU_PROOF);
    double ntt_time = TimerGetElapsedCategoryGPU(timer, NTT);
    double merkletree_time = TimerGetElapsedCategoryGPU(timer, MERKLE_TREE);
    std::ostringstream oss;

    zklog.trace("    TIMES SUMMARY: ");

    double expressions_time = TimerGetElapsedGPU(timer, CALCULATE_IM_POLS) + TimerGetElapsedGPU(timer, STARK_STEP_Q_EXPRESSIONS) + TimerGetElapsedGPU(timer, STARK_CALCULATE_WITNESS_STD) + TimerGetElapsedGPU(timer, STARK_STEP_FRI_POLYNOMIAL);
    oss << std::fixed << std::setprecision(2) << expressions_time << "s (" << (expressions_time / time_total) * 100 << "%)";
    zklog.trace("        EXPRESSIONS:  " + oss.str());
    oss.str("");
    oss.clear();

    double commit_time = TimerGetElapsedGPU(timer, STARK_COMMIT_STAGE_1) + TimerGetElapsedGPU(timer, STARK_COMMIT_STAGE_2) + TimerGetElapsedGPU(timer, STARK_STEP_Q_COMMIT);
    oss << std::fixed << std::setprecision(2) << commit_time << "s (" << (commit_time / time_total) * 100 << "%)";
    zklog.trace("        COMMIT:       " + oss.str());
    oss.str("");
    oss.clear();

    double evmap_time = TimerGetElapsedGPU(timer, STARK_STEP_EVALS);
    double evmap_time_no_ntt = TimerGetElapsedCategoryGPU(timer, EVALS);
    cout << evmap_time_no_ntt << endl;
    oss << std::fixed << std::setprecision(2) << evmap_time << "s (" << (evmap_time / time_total) * 100 << "%)";
    zklog.trace("        EVALUATIONS:  " + oss.str());
    oss.str("");
    oss.clear();

    double fri_time = TimerGetElapsedGPU(timer, STARK_STEP_FRI) - TimerGetElapsedGPU(timer, STARK_STEP_FRI_POLYNOMIAL);
    oss << std::fixed << std::setprecision(2) << fri_time << "s (" << (fri_time / time_total) * 100 << "%)";
    zklog.trace("        FRI:          " + oss.str());
    oss.str("");
    oss.clear();

    double others_time = TimerGetElapsedGPU(timer, STARK_INITIALIZATION) + TimerGetElapsedGPU(timer, STARK_STEP_0);
    oss << std::fixed << std::setprecision(2) << others_time << "s (" << (others_time / time_total) * 100 << "%)";
    zklog.trace("        OTHERS:       " + oss.str());
    oss.str("");
    oss.clear();

    if (others_time + commit_time + evmap_time + expressions_time + fri_time >= time_total ||
        others_time + commit_time + evmap_time + expressions_time + fri_time <= 0.99 * time_total) {
        oss << std::fixed << std::setprecision(2) << (others_time + commit_time + evmap_time + expressions_time + fri_time);
        std::string calculated_time = oss.str();
        oss.str("");
        oss.clear();
        oss << std::fixed << std::setprecision(2) << time_total;
        std::string total_time = oss.str();
        zklog.error("    TIME SUMMARY ERROR: " + calculated_time + " != " + total_time);
    }
    zklog.trace("    KERNELS CONTRIBUTIONS: ");
    
    oss << std::fixed << std::setprecision(2) << ntt_time << "s (" << (ntt_time / time_total) * 100 << "%)";
    zklog.trace("        NTT:          " + oss.str());
    oss.str("");
    oss.clear();

    oss << std::fixed << std::setprecision(2) << merkletree_time << "s (" << (merkletree_time / time_total) * 100 << "%)";
    zklog.trace("        MERKLE:       " + oss.str());
    oss.str("");
    oss.clear();

#endif
}

void getProof_gpu(SetupCtx& setupCtx, uint64_t airgroupId, uint64_t airId, uint64_t instanceId, uint64_t *proofBuffer, std::string proofFile, gl64_t *d_aux_trace) {
    TimerStart(STARK_POSTPROCESS);
    writeProof(setupCtx, (Goldilocks::Element *)d_aux_trace, proofBuffer, airgroupId, airId, instanceId, proofFile);
    TimerStopAndLog(STARK_POSTPROCESS);
}

#endif