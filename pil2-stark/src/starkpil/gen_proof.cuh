#ifndef GEN_PROOF_CUH
#define GEN_PROOF_CUH

#include "starks.hpp"
#include "hints.hpp"
#include "cuda_utils.cuh"
#include "gl64_t.cuh"
#include "expressions_gpu.cuh"
#include "starks_gpu.cuh"
#include <iomanip>

// TOTO list: //rick
// carregar-me els d_trees
// _inplace not good name
#define PRINT_TIME_SUMMARY 1


void calculateWitnessSTD_gpu(SetupCtx& setupCtx, StepsParams& h_params, ExpressionsCtx &expressionsCtx, bool prod, ExpressionsGPU *expressionsCtxGPU, double *time_expressions) {

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

        multiplyHintFieldsGPU(setupCtx, h_params, expressionsCtx, nImTotalHints, imHints, hintFieldDest, hintField1, hintField2, hintOptions1, hintOptions2, expressionsCtxGPU, time_expressions);
    }

    HintFieldOptions options1;
    HintFieldOptions options2;
    options2.inverse = true;
    accMulHintFieldsGPU(setupCtx, h_params, expressionsCtx, hint[0], "reference", "result", "numerator_air", "denominator_air",options1, options2, !prod,expressionsCtxGPU, time_expressions);
    updateAirgroupValueGPU(setupCtx, h_params, hint[0], "result", "numerator_direct", "denominator_direct", options1, options2, !prod, expressionsCtxGPU, time_expressions);
}

void genProof_gpu(SetupCtx& setupCtx, uint64_t airgroupId, uint64_t airId, uint64_t instanceId, Goldilocks::Element *globalChallenge, uint64_t *proofBuffer, std::string proofFile, DeviceCommitBuffers *d_buffers) {
    
    TimerStart(STARK_GPU_PROOF);
    TimerStart(STARK_INITIALIZATION);

    uint64_t N = 1 << setupCtx.starkInfo.starkStruct.nBits;
    uint64_t NExtended = 1 << setupCtx.starkInfo.starkStruct.nBitsExt;

    double totalNTTTime = 0;
    double totalMerkleTime = 0;
    double nttTime = 0;
    double merkleTime = 0;

    TimerStart(PROVER_HELPERS);
    ProverHelpers proverHelpers(setupCtx.starkInfo, false);
    TimerStopAndLog(PROVER_HELPERS);
    
    uint64_t offsetConstTree = setupCtx.starkInfo.mapOffsets[std::make_pair("const", true)];

    Goldilocks::Element *pConstPolsExtendedTreeAddress = (Goldilocks::Element *)d_buffers->d_aux_trace + offsetConstTree;
    Goldilocks::Element *pCustomCommitsFixed = (Goldilocks::Element *)d_buffers->d_aux_trace + setupCtx.starkInfo.mapOffsets[std::make_pair("custom_fixed", false)];

    Starks<Goldilocks::Element> starks(setupCtx, nullptr, nullptr, false);
    starks.treesGL[setupCtx.starkInfo.nStages + 1]->setSource(&pConstPolsExtendedTreeAddress[2]);
    starks.treesGL[setupCtx.starkInfo.nStages + 1]->setNodes(&pConstPolsExtendedTreeAddress[2 + setupCtx.starkInfo.nConstants * NExtended]);
    for(uint64_t i = 0; i < setupCtx.starkInfo.customCommits.size(); i++) {
        uint64_t nCols = setupCtx.starkInfo.mapSectionsN[setupCtx.starkInfo.customCommits[i].name + "0"];
            starks.treesGL[setupCtx.starkInfo.nStages + 2 + i]->setSource(&pCustomCommitsFixed[N * nCols]);
            starks.treesGL[setupCtx.starkInfo.nStages + 2 + i]->setNodes(&pCustomCommitsFixed[(N + NExtended) * nCols]);
    }

    uint64_t nFieldElements = setupCtx.starkInfo.starkStruct.verificationHashType == std::string("BN128") ? 1 : HASH_SIZE;
    CHECKCUDAERR(cudaGetLastError());    
    
    ExpressionsPack expressionsCtx_(setupCtx, nullptr); //rick: get rid of this
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
        trace : (Goldilocks::Element *)d_buffers->d_aux_trace + offsetCm1,
        aux_trace : (Goldilocks::Element *)d_buffers->d_aux_trace,
        publicInputs : (Goldilocks::Element *)d_buffers->d_aux_trace + offsetPublicInputs,
        proofValues : (Goldilocks::Element *)d_buffers->d_aux_trace + offsetProofValues,
        challenges : (Goldilocks::Element *)d_buffers->d_aux_trace + offsetChallenges,
        airgroupValues : (Goldilocks::Element *)d_buffers->d_aux_trace + offsetAirgroupValues,
        airValues : (Goldilocks::Element *)d_buffers->d_aux_trace + offsetAirValues,
        evals : (Goldilocks::Element *)d_buffers->d_aux_trace + offsetEvals,
        xDivXSub : (Goldilocks::Element *)d_buffers->d_aux_trace + offsetXDivXSub,
        pConstPolsAddress : (Goldilocks::Element *)d_buffers->d_aux_trace + offsetConstPols,
        pConstPolsExtendedTreeAddress,
        pCustomCommitsFixed,
    };
    
    Goldilocks::Element *d_challenge = (Goldilocks::Element *)d_buffers->d_aux_trace + offsetChallenge;
    CHECKCUDAERR(cudaMemcpy(d_challenge, globalChallenge, FIELD_EXTENSION * sizeof(Goldilocks::Element), cudaMemcpyHostToDevice));
           
    uint64_t *friQueries_gpu = (uint64_t *)d_buffers->d_aux_trace + offsetFriQueries;
    
    TranscriptGL_GPU d_transcript(setupCtx.starkInfo.starkStruct.merkleTreeArity, setupCtx.starkInfo.starkStruct.merkleTreeCustom);

    gl64_t *d_queries_buff = (gl64_t *)d_buffers->d_aux_trace + offsetProofQueries;
    uint64_t nTrees = setupCtx.starkInfo.nStages + setupCtx.starkInfo.customCommits.size() + 2;
    uint64_t nTreesFRI = setupCtx.starkInfo.starkStruct.steps.size() - 1;

    TimerStopAndLog(STARK_INITIALIZATION);
    TimerStart(STARK_STEP_0);
    d_transcript.put(d_challenge, FIELD_EXTENSION);
    for (uint64_t i = 0; i < setupCtx.starkInfo.challengesMap.size(); i++) {
        if(setupCtx.starkInfo.challengesMap[i].stage == 2) {
            d_transcript.getField((uint64_t *)&h_params.challenges[i * FIELD_EXTENSION]);
        }
    }
    TimerStopAndLog(STARK_STEP_0);
    
    TimerStart(STARK_CALCULATE_WITNESS_STD);
    double time_expressions = 0;
    calculateWitnessSTD_gpu(setupCtx, h_params, expressionsCtx_, true, &expressionsCtx, &time_expressions);
    calculateWitnessSTD_gpu(setupCtx, h_params, expressionsCtx_, false, &expressionsCtx, &time_expressions);

    TimerStopAndLog(STARK_CALCULATE_WITNESS_STD);

    TimerStart(CALCULATE_IM_POLS);
    calculateImPolsExpressions(setupCtx, expressionsCtx, h_params, 2);
    TimerStopAndLog(CALCULATE_IM_POLS);

    TimerStart(STARK_COMMIT_STAGE_1);
    commitStage_inplace(1, setupCtx, starks.treesGL, (gl64_t*) h_params.trace, (gl64_t*)h_params.aux_trace, d_buffers, nullptr, &nttTime, &merkleTime);
    totalNTTTime += nttTime;
    totalMerkleTime += merkleTime;
    TimerStopAndLog(STARK_COMMIT_STAGE_1);
    
    TimerStart(STARK_COMMIT_STAGE_2);
    commitStage_inplace(2, setupCtx, starks.treesGL, (gl64_t*)h_params.trace, (gl64_t*)h_params.aux_trace, d_buffers, &d_transcript, &nttTime, &merkleTime);
    totalNTTTime += nttTime;
    totalMerkleTime += merkleTime;

    uint64_t a = 0;
    for(uint64_t i = 0; i < setupCtx.starkInfo.airValuesMap.size(); i++) {
        if(setupCtx.starkInfo.airValuesMap[i].stage == 1) a++;
        if(setupCtx.starkInfo.airValuesMap[i].stage == 2) {
            d_transcript.put(&h_params.airValues[a], FIELD_EXTENSION);
            a += 3;
        }
    }
    TimerStopAndLog(STARK_COMMIT_STAGE_2);
    TimerStart(STARK_STEP_Q);
    TimerStart(STARK_STEP_Q_EXPRESSIONS);
    for (uint64_t i = 0; i < setupCtx.starkInfo.challengesMap.size(); i++)
    {
        if(setupCtx.starkInfo.challengesMap[i].stage == setupCtx.starkInfo.nStages + 1) {
            d_transcript.getField((uint64_t *)&h_params.challenges[i * FIELD_EXTENSION]);
        }
    }
    uint64_t zi_offset = setupCtx.starkInfo.mapOffsets[std::make_pair("zi", true)];
    CHECKCUDAERR(cudaMemcpy(h_params.aux_trace + zi_offset, proverHelpers.zi, setupCtx.starkInfo.boundaries.size() * NExtended * sizeof(Goldilocks::Element), cudaMemcpyHostToDevice));            
    calculateExpression(setupCtx, expressionsCtx, &h_params, (Goldilocks::Element *)(h_params.aux_trace + setupCtx.starkInfo.mapOffsets[std::make_pair("q", true)]), setupCtx.starkInfo.cExpId);
    TimerStopAndLog(STARK_STEP_Q_EXPRESSIONS);
    TimerStart(STARK_STEP_Q_COMMIT);
    commitStage_inplace(setupCtx.starkInfo.nStages + 1, setupCtx, starks.treesGL, (gl64_t *)h_params.trace, (gl64_t *)h_params.aux_trace, d_buffers, &d_transcript, &nttTime, &merkleTime);
    totalNTTTime += nttTime;
    totalMerkleTime += merkleTime;

    TimerStopAndLog(STARK_STEP_Q_COMMIT);
    TimerStopAndLog(STARK_STEP_Q);
    TimerStart(STARK_STEP_EVALS);

    uint64_t xiChallengeIndex = 0;
    for (uint64_t i = 0; i < setupCtx.starkInfo.challengesMap.size(); i++)
    {
        if(setupCtx.starkInfo.challengesMap[i].stage == setupCtx.starkInfo.nStages + 2) {
            if(setupCtx.starkInfo.challengesMap[i].stageId == 0) xiChallengeIndex = i;
            d_transcript.getField((uint64_t *)&h_params.challenges[i * FIELD_EXTENSION]);
        }
    }

    Goldilocks::Element *d_xiChallenge = &h_params.challenges[xiChallengeIndex * FIELD_EXTENSION];
    gl64_t * d_LEv = (gl64_t *) h_params.aux_trace +setupCtx.starkInfo.mapOffsets[std::make_pair("lev", false)];

    TimerStart(STARK_STEP_EVALS_EVMAP);
    CHECKCUDAERR(cudaMemset(h_params.evals, 0, setupCtx.starkInfo.evMap.size() * FIELD_EXTENSION * sizeof(Goldilocks::Element)));
    for(uint64_t i = 0; i < setupCtx.starkInfo.openingPoints.size(); i += 4) {
        std::vector<int64_t> openingPoints;
        for(uint64_t j = 0; j < 4; ++j) {
            if(i + j < setupCtx.starkInfo.openingPoints.size()) {
                openingPoints.push_back(setupCtx.starkInfo.openingPoints[i + j]);
            }
        }
        uint64_t offset_helper = setupCtx.starkInfo.mapOffsets[std::make_pair("extra_helper_fft_lev", false)];
        computeLEv_inplace(d_xiChallenge, setupCtx.starkInfo.starkStruct.nBits, openingPoints.size(), openingPoints.data(), d_buffers, offset_helper, d_LEv, &nttTime);
        totalNTTTime += nttTime;
        evmap_inplace(setupCtx, h_params, d_buffers, openingPoints.size(), openingPoints.data(), (Goldilocks::Element*)d_LEv);
    }
    TimerStopAndLog(STARK_STEP_EVALS_EVMAP);

    if(!setupCtx.starkInfo.starkStruct.hashCommits) {
        d_transcript.put(h_params.evals, setupCtx.starkInfo.evMap.size() * FIELD_EXTENSION);
    } else {
        calculateHash(d_challenge, setupCtx, h_params.evals, setupCtx.starkInfo.evMap.size() * FIELD_EXTENSION);
        d_transcript.put(d_challenge, HASH_SIZE);
    }

    // Challenges for FRI polynomial
    for (uint64_t i = 0; i < setupCtx.starkInfo.challengesMap.size(); i++)
    {
        if(setupCtx.starkInfo.challengesMap[i].stage == setupCtx.starkInfo.nStages + 3) {
            d_transcript.getField((uint64_t *)&h_params.challenges[i * FIELD_EXTENSION]);
        }
    }
    TimerStopAndLog(STARK_STEP_EVALS);

    //--------------------------------
    // 6. Compute FRI
    //--------------------------------
    TimerStart(STARK_STEP_FRI);
    TimerStart(STARK_STEP_FRI_XIS);

    calculateXis_inplace(setupCtx, h_params, d_xiChallenge);    
    TimerStopAndLog(STARK_STEP_FRI_XIS);

    TimerStart(STARK_STEP_FRI_POLYNOMIAL);
    uint64_t x_offset = setupCtx.starkInfo.mapOffsets[std::make_pair("x", true)];
    dim3 threads(256);
    dim3 blocks((NExtended + threads.x - 1) / threads.x);
    computeX_kernel<<<blocks, threads>>>((gl64_t *)h_params.aux_trace + x_offset, NExtended, Goldilocks::shift(), Goldilocks::w(setupCtx.starkInfo.starkStruct.nBitsExt));
    calculateExpression(setupCtx, expressionsCtx, &h_params, (Goldilocks::Element *)(h_params.aux_trace + setupCtx.starkInfo.mapOffsets[std::make_pair("f", true)]), setupCtx.starkInfo.friExpId);
    for(uint64_t step = 0; step < setupCtx.starkInfo.starkStruct.steps.size() - 1; ++step) { 
        Goldilocks::Element *src = h_params.aux_trace + setupCtx.starkInfo.mapOffsets[std::make_pair("fri_" + to_string(step + 1), true)];
        starks.treesFRI[step]->setSource(src);

        if(setupCtx.starkInfo.starkStruct.verificationHashType == "GL") {
            Goldilocks::Element *pBuffNodesGL = h_params.aux_trace + setupCtx.starkInfo.mapOffsets[std::make_pair("mt_fri_" + to_string(step + 1), true)];
            starks.treesFRI[step]->setNodes(pBuffNodesGL);
        }
    }
    TimerStopAndLog(STARK_STEP_FRI_POLYNOMIAL);
    TimerStart(STARK_STEP_FRI_FOLDING);

    uint64_t friPol_offset = setupCtx.starkInfo.mapOffsets[std::make_pair("f", true)];
    uint64_t offset_helper = setupCtx.starkInfo.mapOffsets[std::make_pair("buff_helper", false)];
    gl64_t *d_friPol = (gl64_t *)(h_params.aux_trace + friPol_offset);
    
    uint64_t nBitsExt =  setupCtx.starkInfo.starkStruct.steps[0].nBits;

    for (uint64_t step = 0; step < setupCtx.starkInfo.starkStruct.steps.size(); step++)
    {
        uint64_t currentBits = setupCtx.starkInfo.starkStruct.steps[step].nBits;
        uint64_t prevBits = step == 0 ? currentBits : setupCtx.starkInfo.starkStruct.steps[step - 1].nBits;
        fold_inplace(step, friPol_offset, offset_helper, d_challenge, nBitsExt, prevBits, currentBits, d_buffers);

        if (step < setupCtx.starkInfo.starkStruct.steps.size() - 1)
        {
            merkelizeFRI_inplace(setupCtx, h_params, step, d_friPol, starks.treesFRI[step], currentBits, setupCtx.starkInfo.starkStruct.steps[step + 1].nBits, &d_transcript, &merkleTime);
            totalMerkleTime += merkleTime;
        }
        else
        {
            if(!setupCtx.starkInfo.starkStruct.hashCommits) {
                d_transcript.put((Goldilocks::Element *)d_friPol, (1 << setupCtx.starkInfo.starkStruct.steps[step].nBits) * FIELD_EXTENSION);
            } else {
                calculateHash(d_challenge, setupCtx, (Goldilocks::Element *)d_friPol, (1 << setupCtx.starkInfo.starkStruct.steps[step].nBits) * FIELD_EXTENSION);
                d_transcript.put(d_challenge, HASH_SIZE);
            }
        }
        d_transcript.getField((uint64_t *)d_challenge);
    }
    TimerStopAndLog(STARK_STEP_FRI_FOLDING);
   
    TimerStart(STARK_STEP_FRI_QUERIES);
    TranscriptGL_GPU d_transcriptPermutation(setupCtx.starkInfo.starkStruct.merkleTreeArity, setupCtx.starkInfo.starkStruct.merkleTreeCustom);
    d_transcriptPermutation.put(d_challenge, FIELD_EXTENSION);
    d_transcriptPermutation.getPermutations(friQueries_gpu, setupCtx.starkInfo.starkStruct.nQueries, setupCtx.starkInfo.starkStruct.steps[0].nBits);

    proveQueries_inplace(setupCtx, d_queries_buff, friQueries_gpu, setupCtx.starkInfo.starkStruct.nQueries, starks.treesGL, nTrees, d_buffers, d_buffers->d_aux_trace + offsetConstTree, setupCtx.starkInfo.nStages);
    for(uint64_t step = 0; step < setupCtx.starkInfo.starkStruct.steps.size() - 1; ++step) {
        proveFRIQueries_inplace(setupCtx, &d_queries_buff[(nTrees + step) * setupCtx.starkInfo.starkStruct.nQueries * setupCtx.starkInfo.maxProofBuffSize], step + 1, setupCtx.starkInfo.starkStruct.steps[step + 1].nBits, friQueries_gpu, setupCtx.starkInfo.starkStruct.nQueries, starks.treesFRI[step]);
    }
    TimerStopAndLog(STARK_STEP_FRI_QUERIES);
    TimerStopAndLog(STARK_STEP_FRI);
    TimerStart(STARK_POSTPROCESS);
    writeProof(setupCtx, h_params.aux_trace, proofBuffer, airgroupId, airId, instanceId, proofFile);
    TimerStopAndLog(STARK_POSTPROCESS);
    TimerStopAndLog(STARK_GPU_PROOF);

#if PRINT_TIME_SUMMARY

    double time_total = TimerGetElapsed(STARK_GPU_PROOF);

    std::ostringstream oss;

    zklog.trace("    TIMES SUMMARY: ");

    double expressions_time = time_expressions + TimerGetElapsed(CALCULATE_IM_POLS) + TimerGetElapsed(STARK_STEP_Q_EXPRESSIONS) + TimerGetElapsed(STARK_STEP_FRI_POLYNOMIAL);
    oss << std::fixed << std::setprecision(2) << expressions_time << "s (" << (expressions_time / time_total) * 100 << "%)";
    zklog.trace("        EXPRESSIONS:  " + oss.str());
    oss.str("");
    oss.clear();

    double commit_time = TimerGetElapsed(STARK_COMMIT_STAGE_1) + TimerGetElapsed(STARK_COMMIT_STAGE_2) + TimerGetElapsed(STARK_STEP_Q_COMMIT);
    oss << std::fixed << std::setprecision(2) << commit_time << "s (" << (commit_time / time_total) * 100 << "%)";
    zklog.trace("        COMMIT:       " + oss.str());
    oss.str("");
    oss.clear();

    double evmap_time = TimerGetElapsed(STARK_STEP_EVALS);
    oss << std::fixed << std::setprecision(2) << evmap_time << "s (" << (evmap_time / time_total) * 100 << "%)";
    zklog.trace("        EVALUATIONS:  " + oss.str());
    oss.str("");
    oss.clear();

    double fri_time = TimerGetElapsed(STARK_STEP_FRI) - TimerGetElapsed(STARK_STEP_FRI_POLYNOMIAL);
    oss << std::fixed << std::setprecision(2) << fri_time << "s (" << (fri_time / time_total) * 100 << "%)";
    zklog.trace("        FRI:          " + oss.str());
    oss.str("");
    oss.clear();

    double others_time = TimerGetElapsed(STARK_INITIALIZATION) + TimerGetElapsed(STARK_STEP_0) + TimerGetElapsed(STARK_POSTPROCESS) + TimerGetElapsed(STARK_CALCULATE_WITNESS_STD) - time_expressions;
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
    
    oss << std::fixed << std::setprecision(2) << totalNTTTime << "s (" << (totalNTTTime / time_total) * 100 << "%)";
    zklog.trace("        NTT:          " + oss.str());
    oss.str("");
    oss.clear();

    oss << std::fixed << std::setprecision(2) << totalMerkleTime << "s (" << (totalMerkleTime / time_total) * 100 << "%)";
    zklog.trace("        MERKLE:       " + oss.str());
    oss.str("");
    oss.clear();

#endif
}

#endif