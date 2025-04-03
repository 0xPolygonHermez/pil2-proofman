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
// eliminar els params
// carregar-me els d_trees
// _inplace not good name
#define PRINT_TIME_SUMMARY 1


void calculateWitnessSTD_gpu(SetupCtx& setupCtx, StepsParams& params, ExpressionsCtx &expressionsCtx, bool prod, ExpressionsGPU *expressionsCtxGPU, StepsParams* d_params) {

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

        multiplyHintFields(setupCtx, params, expressionsCtx, nImTotalHints, imHints, hintFieldDest, hintField1, hintField2, hintOptions1, hintOptions2, expressionsCtxGPU, d_params);
    }

    HintFieldOptions options1;
    HintFieldOptions options2;
    options2.inverse = true;
    accMulHintFields(setupCtx, params, expressionsCtx, hint[0], "reference", "result", "numerator_air", "denominator_air",options1, options2, !prod,expressionsCtxGPU, d_params);
    updateAirgroupValue(setupCtx, params, hint[0], "result", "numerator_direct", "denominator_direct", options1, options2, !prod, expressionsCtxGPU, d_params);
}

void genProof_gpu(SetupCtx& setupCtx, uint64_t airgroupId, uint64_t airId, uint64_t instanceId, StepsParams& params, Goldilocks::Element *globalChallenge, uint64_t *proofBuffer, std::string proofFile, DeviceCommitBuffers *d_buffers) {
    
    TimerStart(STARK_GPU_PROOF);
    TimerStart(STARK_INITIALIZATION);

    double totalNTTTime = 0;
    double totalMerkleTime = 0;
    double nttTime = 0;
    double merkleTime = 0;

    ProverHelpers proverHelpers(setupCtx.starkInfo, false);

    uint64_t offsetConstTree = setupCtx.starkInfo.mapOffsets[std::make_pair("const", true)];

    FRIProof<Goldilocks::Element> proof(setupCtx.starkInfo, airgroupId, airId, instanceId);
    Starks<Goldilocks::Element> starks(setupCtx, proverHelpers, &params.aux_trace[offsetConstTree], params.pCustomCommitsFixed, false); 

    uint64_t nFieldElements = setupCtx.starkInfo.starkStruct.verificationHashType == std::string("BN128") ? 1 : HASH_SIZE;
    CHECKCUDAERR(cudaGetLastError());    
    
    ExpressionsPack expressionsCtx_(setupCtx, proverHelpers); //rick: get rid of this
    ExpressionsGPU expressionsCtx(setupCtx, proverHelpers, 128, 2048);

    uint64_t offsetCm1 = setupCtx.starkInfo.mapOffsets[std::make_pair("cm1", false)];
    uint64_t offsetConstPols = setupCtx.starkInfo.mapOffsets[std::make_pair("const", false)];

    uint64_t NExtended = 1 << setupCtx.starkInfo.starkStruct.nBitsExt;

    StepsParams h_params = {
        trace : (Goldilocks::Element *)d_buffers->d_aux_trace + offsetCm1,
        aux_trace : (Goldilocks::Element *)d_buffers->d_aux_trace,
        publicInputs : nullptr,
        proofValues : nullptr,
        challenges : nullptr,
        airgroupValues : nullptr,
        airValues : nullptr,
        evals : nullptr,
        xDivXSub : nullptr, //rick: canviar nom per xi_s
        pConstPolsAddress : (Goldilocks::Element *)d_buffers->d_aux_trace + offsetConstPols,
        pConstPolsExtendedTreeAddress : (Goldilocks::Element *)d_buffers->d_aux_trace + offsetConstTree,
        pCustomCommitsFixed : (Goldilocks::Element *)d_buffers->d_aux_trace + setupCtx.starkInfo.mapOffsets[std::make_pair("custom_fixed", false)],
    };
    
    // Allocate memory and copy data
    CHECKCUDAERR(cudaMalloc(&h_params.publicInputs, setupCtx.starkInfo.nPublics * sizeof(Goldilocks::Element)));
    CHECKCUDAERR(cudaMemcpy(h_params.publicInputs, params.publicInputs, setupCtx.starkInfo.nPublics * sizeof(Goldilocks::Element), cudaMemcpyHostToDevice));
    CHECKCUDAERR(cudaMalloc(&h_params.evals, setupCtx.starkInfo.evMap.size() * FIELD_EXTENSION * sizeof(Goldilocks::Element)));
    CHECKCUDAERR(cudaMalloc(&h_params.challenges, setupCtx.starkInfo.challengesMap.size() * FIELD_EXTENSION * sizeof(Goldilocks::Element)));
    CHECKCUDAERR(cudaMemcpy(h_params.challenges, params.challenges, setupCtx.starkInfo.challengesMap.size() * FIELD_EXTENSION * sizeof(Goldilocks::Element), cudaMemcpyHostToDevice));
    CHECKCUDAERR(cudaMalloc(&h_params.xDivXSub, FIELD_EXTENSION * setupCtx.starkInfo.openingPoints.size() * sizeof(Goldilocks::Element)));

    if (setupCtx.starkInfo.mapTotalNCustomCommitsFixed > 0) {
        CHECKCUDAERR(cudaMemcpy(h_params.pCustomCommitsFixed, params.pCustomCommitsFixed, setupCtx.starkInfo.mapTotalNCustomCommitsFixed * sizeof(Goldilocks::Element), cudaMemcpyHostToDevice));
    }
    if (setupCtx.starkInfo.proofValuesSize > 0) {
        CHECKCUDAERR(cudaMalloc(&h_params.proofValues, setupCtx.starkInfo.proofValuesSize * sizeof(Goldilocks::Element)));
        CHECKCUDAERR(cudaMemcpy(h_params.proofValues, params.proofValues, setupCtx.starkInfo.proofValuesSize * sizeof(Goldilocks::Element), cudaMemcpyHostToDevice));
    }
    if (setupCtx.starkInfo.airgroupValuesSize > 0) {
        CHECKCUDAERR(cudaMalloc(&h_params.airgroupValues, setupCtx.starkInfo.airgroupValuesSize * sizeof(Goldilocks::Element)));
        CHECKCUDAERR(cudaMemcpy(h_params.airgroupValues, params.airgroupValues, setupCtx.starkInfo.airgroupValuesSize * sizeof(Goldilocks::Element), cudaMemcpyHostToDevice));
    }
    if (setupCtx.starkInfo.airValuesSize > 0) {
        CHECKCUDAERR(cudaMalloc(&h_params.airValues, setupCtx.starkInfo.airValuesSize * sizeof(Goldilocks::Element)));
        CHECKCUDAERR(cudaMemcpy(h_params.airValues, params.airValues, setupCtx.starkInfo.airValuesSize * sizeof(Goldilocks::Element), cudaMemcpyHostToDevice));
    }
   
    StepsParams* d_params;
    CHECKCUDAERR(cudaMalloc(&d_params, sizeof(StepsParams)));
    CHECKCUDAERR(cudaMemcpy(d_params, &h_params, sizeof(StepsParams), cudaMemcpyHostToDevice));
    
    TranscriptGL transcript(setupCtx.starkInfo.starkStruct.merkleTreeArity, setupCtx.starkInfo.starkStruct.merkleTreeCustom);
    TimerStopAndLog(STARK_INITIALIZATION);
    TimerStart(STARK_STEP_0);
    for (uint64_t i = 0; i < setupCtx.starkInfo.customCommits.size(); i++) {
        if(setupCtx.starkInfo.customCommits[i].stageWidths[0] != 0) {
            uint64_t pos = setupCtx.starkInfo.nStages + 2 + i;
            starks.treesGL[pos]->getRoot(&proof.proof.roots[pos - 1][0]);
        }
    }
    starks.addTranscript(transcript, globalChallenge, FIELD_EXTENSION);
    uint64_t min_challenge = setupCtx.starkInfo.challengesMap.size();
    uint64_t max_challenge = 0;
    for (uint64_t i = 0; i < setupCtx.starkInfo.challengesMap.size(); i++) {
        if(setupCtx.starkInfo.challengesMap[i].stage == 2) {
            min_challenge = std::min(min_challenge, i);
            max_challenge = std::max(max_challenge, i);
            starks.getChallenge(transcript, params.challenges[i * FIELD_EXTENSION]);
        }
    }
    CHECKCUDAERR(cudaMemcpy(h_params.challenges + min_challenge * FIELD_EXTENSION, params.challenges + min_challenge * FIELD_EXTENSION, (max_challenge - min_challenge + 1) * FIELD_EXTENSION * sizeof(Goldilocks::Element), cudaMemcpyHostToDevice));
    TimerStopAndLog(STARK_STEP_0);
    
    TimerStart(STARK_CALCULATE_WITNESS_STD);
        
    calculateWitnessSTD_gpu(setupCtx, h_params, expressionsCtx_, true, &expressionsCtx, d_params);
    calculateWitnessSTD_gpu(setupCtx, h_params, expressionsCtx_, false, &expressionsCtx, d_params);

    TimerStopAndLog(STARK_CALCULATE_WITNESS_STD);


    TimerStart(CALCULATE_IM_POLS);
    calculateImPolsExpressions(setupCtx, expressionsCtx, h_params, d_params, 2);
    TimerStopAndLog(CALCULATE_IM_POLS);

    TimerStart(STARK_COMMIT_STAGE_1);
    starks.commitStage_inplace(1, (gl64_t*) h_params.trace, (gl64_t*)h_params.aux_trace, d_buffers, &nttTime, &merkleTime);
    totalNTTTime += nttTime;
    totalMerkleTime += merkleTime;
    offloadCommit(1, starks.treesGL, (gl64_t*)h_params.aux_trace, proof, setupCtx);
    TimerStopAndLog(STARK_COMMIT_STAGE_1);
    
    TimerStart(STARK_COMMIT_STAGE_2);
    starks.commitStage_inplace(2, (gl64_t*)h_params.trace, (gl64_t*)h_params.aux_trace, d_buffers, &nttTime, &merkleTime);
    totalNTTTime += nttTime;
    totalMerkleTime += merkleTime;
    offloadCommit(2, starks.treesGL, (gl64_t*)h_params.aux_trace, proof, setupCtx);

    CHECKCUDAERR(cudaMemcpy(params.airValues, h_params.airValues, setupCtx.starkInfo.airValuesSize * sizeof(Goldilocks::Element), cudaMemcpyDeviceToHost));

    starks.addTranscript(transcript, &proof.proof.roots[1][0], HASH_SIZE);
        
    uint64_t a = 0;
    for(uint64_t i = 0; i < setupCtx.starkInfo.airValuesMap.size(); i++) {
        if(setupCtx.starkInfo.airValuesMap[i].stage == 1) a++;
        if(setupCtx.starkInfo.airValuesMap[i].stage == 2) {
            starks.addTranscript(transcript, &params.airValues[a], FIELD_EXTENSION);
            a += 3;
        }
    }
    TimerStopAndLog(STARK_COMMIT_STAGE_2);
    TimerStart(STARK_STEP_Q);
    TimerStart(STARK_STEP_Q_EXPRESSIONS);
    min_challenge = setupCtx.starkInfo.challengesMap.size();
    max_challenge = 0;
    for (uint64_t i = 0; i < setupCtx.starkInfo.challengesMap.size(); i++)
    {
        if(setupCtx.starkInfo.challengesMap[i].stage == setupCtx.starkInfo.nStages + 1) {
            min_challenge = std::min(min_challenge, i);
            max_challenge = std::max(max_challenge, i);
            starks.getChallenge(transcript, params.challenges[i * FIELD_EXTENSION]);
        }
    }
    CHECKCUDAERR(cudaMemcpy(h_params.challenges + min_challenge * FIELD_EXTENSION, params.challenges + min_challenge * FIELD_EXTENSION, (max_challenge - min_challenge + 1) * FIELD_EXTENSION * sizeof(Goldilocks::Element), cudaMemcpyHostToDevice));
    uint64_t zi_offset = setupCtx.starkInfo.mapOffsets[std::make_pair("zi", true)];
    CHECKCUDAERR(cudaMemcpy(h_params.aux_trace + zi_offset, proverHelpers.zi, setupCtx.starkInfo.boundaries.size() * NExtended * sizeof(Goldilocks::Element), cudaMemcpyHostToDevice));            
    calculateExpression(setupCtx, expressionsCtx, d_params, (Goldilocks::Element *)(h_params.aux_trace + setupCtx.starkInfo.mapOffsets[std::make_pair("q", true)]), setupCtx.starkInfo.cExpId);
    TimerStopAndLog(STARK_STEP_Q_EXPRESSIONS);
    TimerStart(STARK_STEP_Q_COMMIT);
    starks.commitStage_inplace(setupCtx.starkInfo.nStages + 1, (gl64_t *)h_params.trace, (gl64_t *)h_params.aux_trace, d_buffers, &nttTime, &merkleTime);
    totalNTTTime += nttTime;
    totalMerkleTime += merkleTime;
    offloadCommit(setupCtx.starkInfo.nStages + 1, starks.treesGL, (gl64_t *)h_params.aux_trace, proof, setupCtx);
    starks.addTranscript(transcript, &proof.proof.roots[setupCtx.starkInfo.nStages][0], HASH_SIZE);

    TimerStopAndLog(STARK_STEP_Q_COMMIT);
    TimerStopAndLog(STARK_STEP_Q);
    TimerStart(STARK_STEP_EVALS);
    TimerStart(STARK_STEP_EVALS_CALCULATE_LEV);

    min_challenge = setupCtx.starkInfo.challengesMap.size();
    max_challenge = 0;
    uint64_t xiChallengeIndex = 0;
    for (uint64_t i = 0; i < setupCtx.starkInfo.challengesMap.size(); i++)
    {
        if(setupCtx.starkInfo.challengesMap[i].stage == setupCtx.starkInfo.nStages + 2) {
            if(setupCtx.starkInfo.challengesMap[i].stageId == 0) xiChallengeIndex = i;
            min_challenge = std::min(min_challenge, i);
            max_challenge = std::max(max_challenge, i);
            starks.getChallenge(transcript, params.challenges[i * FIELD_EXTENSION]);
        }
    }
    CHECKCUDAERR(cudaMemcpy(h_params.challenges + min_challenge * FIELD_EXTENSION, params.challenges + min_challenge * FIELD_EXTENSION, (max_challenge - min_challenge + 1) * FIELD_EXTENSION * sizeof(Goldilocks::Element), cudaMemcpyHostToDevice));

    Goldilocks::Element *xiChallenge = &params.challenges[xiChallengeIndex * FIELD_EXTENSION];
   gl64_t * d_LEv = (gl64_t *) h_params.aux_trace +setupCtx.starkInfo.mapOffsets[std::make_pair("lev", false)];

    
    computeLEv_inplace(xiChallenge, setupCtx.starkInfo.starkStruct.nBits, setupCtx.starkInfo.openingPoints.size(), setupCtx.starkInfo.openingPoints.data(), d_buffers, setupCtx.starkInfo.mapOffsets[std::make_pair("extra_helper_fft_lev", false)], d_LEv, &nttTime);
    totalNTTTime += nttTime;
    TimerStopAndLog(STARK_STEP_EVALS_CALCULATE_LEV);
    TimerStart(STARK_STEP_EVALS_EVMAP);
    
    evmap_inplace(params.evals, h_params, proof, &starks, d_buffers, (Goldilocks::Element*)d_LEv);

    if(!setupCtx.starkInfo.starkStruct.hashCommits) {
        starks.addTranscriptGL(transcript, params.evals, setupCtx.starkInfo.evMap.size() * FIELD_EXTENSION);
    } else {
        Goldilocks::Element hash[HASH_SIZE];
        starks.calculateHash(hash, params.evals, setupCtx.starkInfo.evMap.size() * FIELD_EXTENSION);
        starks.addTranscript(transcript, hash, HASH_SIZE);
    }

    // Challenges for FRI polynomial
    min_challenge = setupCtx.starkInfo.challengesMap.size();
    max_challenge = 0;
    for (uint64_t i = 0; i < setupCtx.starkInfo.challengesMap.size(); i++)
    {
        if(setupCtx.starkInfo.challengesMap[i].stage == setupCtx.starkInfo.nStages + 3) {
            min_challenge = std::min(min_challenge, i);
            max_challenge = std::max(max_challenge, i);
            starks.getChallenge(transcript, params.challenges[i * FIELD_EXTENSION]);
        }
    }
    CHECKCUDAERR(cudaMemcpy(h_params.challenges + min_challenge * FIELD_EXTENSION, params.challenges + min_challenge * FIELD_EXTENSION, (max_challenge - min_challenge + 1) * FIELD_EXTENSION * sizeof(Goldilocks::Element), cudaMemcpyHostToDevice));
    TimerStopAndLog(STARK_STEP_EVALS_EVMAP);
    TimerStopAndLog(STARK_STEP_EVALS);

    //--------------------------------
    // 6. Compute FRI
    //--------------------------------
    TimerStart(STARK_STEP_FRI);
    TimerStart(STARK_STEP_FRI_XIS);

    calculateXis_inplace(setupCtx, h_params, xiChallenge);    
    TimerStopAndLog(STARK_STEP_FRI_XIS);

    TimerStart(STARK_STEP_FRI_POLYNOMIAL);
    uint64_t x_offset = setupCtx.starkInfo.mapOffsets[std::make_pair("x", true)];
    CHECKCUDAERR(cudaMemcpy(h_params.aux_trace + x_offset, proverHelpers.x, NExtended * sizeof(Goldilocks::Element), cudaMemcpyHostToDevice));            
    calculateExpression(setupCtx, expressionsCtx, d_params, (Goldilocks::Element *)(h_params.aux_trace + setupCtx.starkInfo.mapOffsets[std::make_pair("f", true)]), setupCtx.starkInfo.friExpId);

    TimerStopAndLog(STARK_STEP_FRI_POLYNOMIAL);
    TimerStart(STARK_STEP_FRI_FOLDING);

    Goldilocks::Element challenge[FIELD_EXTENSION];
    uint64_t friPol_offset = setupCtx.starkInfo.mapOffsets[std::make_pair("f", true)];
    uint64_t offset_helper = setupCtx.starkInfo.mapOffsets[std::make_pair("buff_helper", false)];
    gl64_t *d_friPol = (gl64_t *)(h_params.aux_trace + friPol_offset);
    
    uint64_t nBitsExt =  setupCtx.starkInfo.starkStruct.steps[0].nBits;
    Goldilocks::Element *foldedFRIPol = new Goldilocks::Element[(1 << setupCtx.starkInfo.starkStruct.steps[setupCtx.starkInfo.starkStruct.steps.size() - 1].nBits) * FIELD_EXTENSION];

    for (uint64_t step = 0; step < setupCtx.starkInfo.starkStruct.steps.size(); step++)
    {
        uint64_t currentBits = setupCtx.starkInfo.starkStruct.steps[step].nBits;
        uint64_t prevBits = step == 0 ? currentBits : setupCtx.starkInfo.starkStruct.steps[step - 1].nBits;
        fold_inplace(step, friPol_offset, offset_helper, challenge, nBitsExt, prevBits, currentBits, d_buffers);

        if (step < setupCtx.starkInfo.starkStruct.steps.size() - 1)
        {
            merkelizeFRI_inplace(setupCtx, h_params, step, proof, d_friPol, starks.treesFRI[step], currentBits, setupCtx.starkInfo.starkStruct.steps[step + 1].nBits, &merkleTime);
            totalMerkleTime += merkleTime;
            starks.addTranscript(transcript, &proof.proof.fri.treesFRI[step].root[0], HASH_SIZE);
        }
        else
        {
            CHECKCUDAERR(cudaMemcpy(foldedFRIPol, d_friPol, (1 << setupCtx.starkInfo.starkStruct.steps[step].nBits) * FIELD_EXTENSION * sizeof(Goldilocks::Element), cudaMemcpyDeviceToHost));
            if(!setupCtx.starkInfo.starkStruct.hashCommits) {
                starks.addTranscriptGL(transcript, foldedFRIPol, (1 << setupCtx.starkInfo.starkStruct.steps[step].nBits) * FIELD_EXTENSION);
            } else {
                Goldilocks::Element hash[HASH_SIZE];
                starks.calculateHash(hash, foldedFRIPol, (1 << setupCtx.starkInfo.starkStruct.steps[step].nBits) * FIELD_EXTENSION);
                starks.addTranscript(transcript, hash, HASH_SIZE);
            } 
            
        }
        starks.getChallenge(transcript, *challenge);
    }
    TimerStopAndLog(STARK_STEP_FRI_FOLDING);
   
    TimerStart(STARK_STEP_FRI_QUERIES);

    uint64_t friQueries[setupCtx.starkInfo.starkStruct.nQueries];

    TranscriptGL transcriptPermutation(setupCtx.starkInfo.starkStruct.merkleTreeArity, setupCtx.starkInfo.starkStruct.merkleTreeCustom);
    starks.addTranscriptGL(transcriptPermutation, challenge, FIELD_EXTENSION);
    transcriptPermutation.getPermutations(friQueries, setupCtx.starkInfo.starkStruct.nQueries, setupCtx.starkInfo.starkStruct.steps[0].nBits);

    gl64_t *d_constTree = d_buffers->d_aux_trace + offsetConstTree;

    uint64_t nTrees = setupCtx.starkInfo.nStages + setupCtx.starkInfo.customCommits.size() + 2;
    proveQueries_inplace(setupCtx, friQueries, setupCtx.starkInfo.starkStruct.nQueries, proof, starks.treesGL, nTrees, d_buffers, d_constTree, setupCtx.starkInfo.nStages, h_params);
    
    for(uint64_t step = 1; step < setupCtx.starkInfo.starkStruct.steps.size(); ++step) {

        FRI<Goldilocks::Element>::proveFRIQueries(friQueries, setupCtx.starkInfo.starkStruct.nQueries, step, setupCtx.starkInfo.starkStruct.steps[step].nBits, proof, starks.treesFRI[step - 1]);
    }

    FRI<Goldilocks::Element>::setFinalPol(proof, foldedFRIPol, setupCtx.starkInfo.starkStruct.steps[setupCtx.starkInfo.starkStruct.steps.size() - 1].nBits);
    TimerStopAndLog(STARK_STEP_FRI_QUERIES);
    TimerStopAndLog(STARK_STEP_FRI);
    TimerStart(STARK_POSTPROCESS);

    // offload airgroup values 
    CHECKCUDAERR(cudaMemcpy(params.airgroupValues, h_params.airgroupValues, setupCtx.starkInfo.airgroupValuesSize * sizeof(Goldilocks::Element), cudaMemcpyDeviceToHost));
    CHECKCUDAERR(cudaMemcpy(params.airValues, h_params.airValues, setupCtx.starkInfo.airValuesSize * sizeof(Goldilocks::Element), cudaMemcpyDeviceToHost));

    proof.proof.setAirgroupValues(params.airgroupValues); 
    proof.proof.setAirValues(params.airValues);
    proof.proof.proof2pointer(proofBuffer);

    if(!proofFile.empty()) {
        json2file(pointer2json(proofBuffer, setupCtx.starkInfo), proofFile);
    }

    // Free memory
    CHECKCUDAERR(cudaFree(h_params.evals));
    CHECKCUDAERR(cudaFree(h_params.xDivXSub));
    CHECKCUDAERR(cudaFree(h_params.challenges));
    if(h_params.proofValues != nullptr) CHECKCUDAERR(cudaFree(h_params.proofValues));
    if(h_params.airgroupValues != nullptr) CHECKCUDAERR(cudaFree(h_params.airgroupValues));
    if(h_params.airValues != nullptr) CHECKCUDAERR(cudaFree(h_params.airValues));
    delete[] foldedFRIPol;
    CHECKCUDAERR(cudaFree(d_params));
    TimerStopAndLog(STARK_POSTPROCESS);
    TimerStopAndLog(STARK_GPU_PROOF);

#if PRINT_TIME_SUMMARY

    double time_total = TimerGetElapsed(STARK_GPU_PROOF);

    std::ostringstream oss;

    zklog.trace("    TIMES SUMMARY: ");

    double expressions_time = TimerGetElapsed(STARK_CALCULATE_WITNESS_STD) + TimerGetElapsed(CALCULATE_IM_POLS) + TimerGetElapsed(STARK_STEP_Q_EXPRESSIONS) + TimerGetElapsed(STARK_STEP_FRI_POLYNOMIAL);
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

    double others_time = TimerGetElapsed(STARK_INITIALIZATION) + TimerGetElapsed(STARK_STEP_0) + TimerGetElapsed(STARK_POSTPROCESS);
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