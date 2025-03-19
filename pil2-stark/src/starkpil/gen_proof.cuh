#ifndef GEN_PROOF_CUH
#define GEN_PROOF_CUH

#include "starks.hpp"
#include "hints.hpp"
#include "cuda_utils.cuh"
#include "gl64_t.cuh"
#include "expressions_gpu.cuh"
#include "starks_gpu.cuh"

// TOTO list: //rick
// no usar pBuffHelper
// eliminar els params
// carregar-me els d_trees
// _inplace not good name

void offloadCommit_(uint64_t step, MerkleTreeGL **treesGL, gl64_t *d_aux_trace, uint64_t *d_tree, FRIProof<Goldilocks::Element> &proof, SetupCtx &setupCtx, StepsParams& params)
{

    CHECKCUDAERR(cudaDeviceSynchronize());
    double time = omp_get_wtime();

    uint64_t NExtended = 1 << setupCtx.starkInfo.starkStruct.nBitsExt;
    std::string section = "cm" + to_string(step);
    uint64_t offset = setupCtx.starkInfo.mapOffsets[make_pair(section, true)];
    Goldilocks::Element *pBuffExtended = &params.aux_trace[offset];
    treesGL[step - 1]->setSource(pBuffExtended);
    treesGL[step - 1]->souceTraceOffset = offset;
    uint64_t mtOffset = setupCtx.starkInfo.mapOffsets[make_pair("mt" + to_string(step), true)];
    Goldilocks::Element *pBuffNodesGL = &params.aux_trace[mtOffset];
    treesGL[step - 1]->setNodes(pBuffNodesGL);

    //gl64_t *d_pBuffNodesGL = d_aux_trace + mtOffset;
    uint64_t tree_size = treesGL[step - 1]->getNumNodes(NExtended);
    uint32_t nFielsElements = treesGL[step - 1]->getMerkleTreeNFieldElements();
    CHECKCUDAERR(cudaMemcpy(&proof.proof.roots[step - 1][0], &d_tree[tree_size - nFielsElements], nFielsElements * sizeof(uint64_t), cudaMemcpyDeviceToHost));
}

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
    updateAirgroupValue(setupCtx, params, hint[0], "result", "numerator_direct", "denominator_direct", options1, options2, !prod);
}

void genProof_gpu(SetupCtx& setupCtx, uint64_t airgroupId, uint64_t airId, uint64_t instanceId, StepsParams& params, Goldilocks::Element *globalChallenge, Goldilocks::Element* pBuffHelper, uint64_t *proofBuffer, std::string proofFile, DeviceCommitBuffers *d_buffers) {
    
    TimerStart(STARK_PROOF);
    CHECKCUDAERR(cudaDeviceSynchronize());
    double time0 = omp_get_wtime();

    //uint64_t N = 1 << setupCtx.starkInfo.starkStruct.nBits;
    uint64_t NExtended = 1 << setupCtx.starkInfo.starkStruct.nBitsExt;

    ProverHelpers proverHelpers(setupCtx.starkInfo, false);

    FRIProof<Goldilocks::Element> proof(setupCtx.starkInfo, airgroupId, airId, instanceId);
    Starks<Goldilocks::Element> starks(setupCtx, proverHelpers, params.pConstPolsExtendedTreeAddress, params.pCustomCommitsFixed, false); //initialze starks
    uint64_t nFieldElements = setupCtx.starkInfo.starkStruct.verificationHashType == std::string("BN128") ? 1 : HASH_SIZE;
    CHECKCUDAERR(cudaGetLastError());
    // GPU tree-nodes
    GPUTree *d_trees = new GPUTree[setupCtx.starkInfo.nStages + 2];
    for (uint64_t i = 0; i < setupCtx.starkInfo.nStages + 1; i++)
    {
        d_trees[i].nFieldElements = nFieldElements;
    }
    
    ExpressionsPack expressionsCtx_(setupCtx, proverHelpers); //rick: get rid of this
    ExpressionsGPU expressionsCtx(setupCtx, proverHelpers, 2, 1176, 465, 128, 2048); //maxNparams, maxNTemp1, maxNTemp3 //rick
    CHECKCUDAERR(cudaGetLastError());
    StepsParams d_params = {
        trace : (Goldilocks::Element *)d_buffers->d_trace,
        aux_trace : (Goldilocks::Element *)d_buffers->d_aux_trace,
        publicInputs : (Goldilocks::Element *)d_buffers->d_publicInputs,
        proofValues : nullptr,
        challenges : nullptr,
        airgroupValues : nullptr,
        airValues : nullptr,
        evals : nullptr,
        xDivXSub : nullptr,
        pConstPolsAddress : (Goldilocks::Element *)d_buffers->d_constPols,
        pConstPolsExtendedTreeAddress : (Goldilocks::Element *)d_buffers->d_constTree,
        pCustomCommitsFixed : nullptr,
    };

    CHECKCUDAERR(cudaMalloc(&d_params.evals, setupCtx.starkInfo.evMap.size() * FIELD_EXTENSION * sizeof(Goldilocks::Element)));
    CHECKCUDAERR(cudaMalloc(&d_params.xDivXSub, NExtended *  setupCtx.starkInfo.openingPoints.size() * FIELD_EXTENSION * sizeof(Goldilocks::Element)));
    uint64_t customCommitSize = setupCtx.starkInfo.mapTotalNCustomCommitsFixed;
    CHECKCUDAERR(cudaMalloc(&d_params.pCustomCommitsFixed,customCommitSize * sizeof(Goldilocks::Element)));
    CHECKCUDAERR(cudaMemcpy(d_params.pCustomCommitsFixed, params.pCustomCommitsFixed, customCommitSize * sizeof(Goldilocks::Element), cudaMemcpyHostToDevice));
    CHECKCUDAERR(cudaGetLastError());

    TranscriptGL transcript(setupCtx.starkInfo.starkStruct.merkleTreeArity, setupCtx.starkInfo.starkStruct.merkleTreeCustom);

    TimerStart(STARK_STEP_0);
    for (uint64_t i = 0; i < setupCtx.starkInfo.customCommits.size(); i++) {
        if(setupCtx.starkInfo.customCommits[i].stageWidths[0] != 0) {
            uint64_t pos = setupCtx.starkInfo.nStages + 2 + i;
            starks.treesGL[pos]->getRoot(&proof.proof.roots[pos - 1][0]);
        }
    }
    TimerStopAndLog(STARK_STEP_0);

    TimerStart(STARK_STEP_1);
    starks.commitStage_inplace(1, (gl64_t*) d_params.trace, (gl64_t*)d_params.aux_trace, (uint64_t **)(&d_trees[0].nodes), d_buffers);
    offloadCommit_(1, starks.treesGL, (gl64_t*)d_params.aux_trace, (uint64_t *)d_trees[0].nodes, proof, setupCtx, params);
    TimerStopAndLog(STARK_STEP_1);

    starks.addTranscript(transcript, globalChallenge, FIELD_EXTENSION);

    TimerStart(STARK_STEP_2);
    for (uint64_t i = 0; i < setupCtx.starkInfo.challengesMap.size(); i++) {
        if(setupCtx.starkInfo.challengesMap[i].stage == 2) {
            starks.getChallenge(transcript, params.challenges[i * FIELD_EXTENSION]);
        }
    }

    calculateWitnessSTD_gpu(setupCtx, params, expressionsCtx_, true, &expressionsCtx, &d_params);
    calculateWitnessSTD_gpu(setupCtx, params, expressionsCtx_, false, &expressionsCtx, &d_params);

    TimerStart(CALCULATE_IM_POLS);
    calculateImPolsExpressions(setupCtx, expressionsCtx, d_buffers, params, d_params, 2);
    TimerStopAndLog(CALCULATE_IM_POLS);

    
    TimerStart(STARK_COMMIT_STAGE_2);
    starks.commitStage_inplace(2, (gl64_t*)d_params.trace, (gl64_t*)d_params.aux_trace, (uint64_t **)(&d_trees[1].nodes), d_buffers);
    offloadCommit_(2, starks.treesGL, (gl64_t*)d_params.aux_trace, (uint64_t *)d_trees[1].nodes, proof, setupCtx, params);
    TimerStopAndLog(STARK_COMMIT_STAGE_2);

    starks.addTranscript(transcript, &proof.proof.roots[1][0], HASH_SIZE);

    uint64_t a = 0;
    for(uint64_t i = 0; i < setupCtx.starkInfo.airValuesMap.size(); i++) {
        if(setupCtx.starkInfo.airValuesMap[i].stage == 1) a++;
        if(setupCtx.starkInfo.airValuesMap[i].stage == 2) {
            starks.addTranscript(transcript, &params.airValues[a], FIELD_EXTENSION);
            a += 3;
        }
    }
    TimerStopAndLog(STARK_STEP_2);

    TimerStart(STARK_STEP_Q);
    for (uint64_t i = 0; i < setupCtx.starkInfo.challengesMap.size(); i++)
    {
        if(setupCtx.starkInfo.challengesMap[i].stage == setupCtx.starkInfo.nStages + 1) {
            starks.getChallenge(transcript, params.challenges[i * FIELD_EXTENSION]);
        }
    }
    calculateExpression(setupCtx, expressionsCtx, params, d_params, (Goldilocks::Element *)(d_params.aux_trace + setupCtx.starkInfo.mapOffsets[std::make_pair("q", true)]), setupCtx.starkInfo.cExpId);

    TimerStart(STARK_COMMIT_QUOTIENT_POLYNOMIAL);
    starks.commitStage_inplace(setupCtx.starkInfo.nStages + 1, (gl64_t *)d_params.trace, (gl64_t *)d_params.aux_trace, (uint64_t **)(&d_trees[setupCtx.starkInfo.nStages].nodes), d_buffers);
    offloadCommit_(setupCtx.starkInfo.nStages + 1, starks.treesGL, (gl64_t *)d_params.aux_trace, (uint64_t *)d_trees[setupCtx.starkInfo.nStages].nodes, proof, setupCtx, params);
    TimerStopAndLog(STARK_COMMIT_QUOTIENT_POLYNOMIAL);

    starks.addTranscript(transcript, &proof.proof.roots[setupCtx.starkInfo.nStages][0], HASH_SIZE);
    TimerStopAndLog(STARK_STEP_Q);

    TimerStart(STARK_STEP_EVALS);

    uint64_t xiChallengeIndex = 0;
    for (uint64_t i = 0; i < setupCtx.starkInfo.challengesMap.size(); i++)
    {
        if(setupCtx.starkInfo.challengesMap[i].stage == setupCtx.starkInfo.nStages + 2) {
            if(setupCtx.starkInfo.challengesMap[i].stageId == 0) xiChallengeIndex = i;
            starks.getChallenge(transcript, params.challenges[i * FIELD_EXTENSION]);
        }
    }

    Goldilocks::Element *xiChallenge = &params.challenges[xiChallengeIndex * FIELD_EXTENSION];
    gl64_t * d_LEv = (gl64_t *) d_params.xDivXSub;

    computeLEv_inplace(xiChallenge, 0, setupCtx.starkInfo.starkStruct.nBits, setupCtx.starkInfo.openingPoints.size(), setupCtx.starkInfo.openingPoints.data(), d_buffers, d_LEv);
    evmap_inplace(params.evals, d_params, 0, proof, &starks, d_buffers, (Goldilocks::Element*)d_LEv);
    
    

    if(!setupCtx.starkInfo.starkStruct.hashCommits) {
        starks.addTranscriptGL(transcript, params.evals, setupCtx.starkInfo.evMap.size() * FIELD_EXTENSION);
    } else {
        Goldilocks::Element hash[HASH_SIZE];
        starks.calculateHash(hash, params.evals, setupCtx.starkInfo.evMap.size() * FIELD_EXTENSION);
        starks.addTranscript(transcript, hash, HASH_SIZE);
    }

    // Challenges for FRI polynomial
    for (uint64_t i = 0; i < setupCtx.starkInfo.challengesMap.size(); i++)
    {
        if(setupCtx.starkInfo.challengesMap[i].stage == setupCtx.starkInfo.nStages + 3) {
            starks.getChallenge(transcript, params.challenges[i * FIELD_EXTENSION]);
        }
    }

    TimerStopAndLog(STARK_STEP_EVALS);

    //--------------------------------
    // 6. Compute FRI
    //--------------------------------
    TimerStart(STARK_STEP_FRI);

    TimerStart(COMPUTE_FRI_POLYNOMIAL);

    params.xDivXSub = &pBuffHelper[0];
    calculateXDivXSub_inplace(0, xiChallenge, setupCtx, d_buffers, (gl64_t*) d_params.xDivXSub);
    
    // FRI expressions
    calculateExpression(setupCtx, expressionsCtx, params, d_params, (Goldilocks::Element *)(d_params.aux_trace + setupCtx.starkInfo.mapOffsets[std::make_pair("f", true)]), setupCtx.starkInfo.friExpId);
    
    TimerStopAndLog(COMPUTE_FRI_POLYNOMIAL);

    Goldilocks::Element challenge[FIELD_EXTENSION];
    uint64_t friPol_offset = setupCtx.starkInfo.mapOffsets[std::make_pair("f", true)];
    gl64_t *d_friPol = (gl64_t *)(d_params.aux_trace + friPol_offset);
    
    TimerStart(STARK_FRI_FOLDING);
    uint64_t nBitsExt =  setupCtx.starkInfo.starkStruct.steps[0].nBits;
    Goldilocks::Element *foldedFRIPol = new Goldilocks::Element[(1 << setupCtx.starkInfo.starkStruct.steps[setupCtx.starkInfo.starkStruct.steps.size() - 1].nBits) * FIELD_EXTENSION];

    for (uint64_t step = 0; step < setupCtx.starkInfo.starkStruct.steps.size(); step++)
    {   
        uint64_t currentBits = setupCtx.starkInfo.starkStruct.steps[step].nBits;
        uint64_t prevBits = step == 0 ? currentBits : setupCtx.starkInfo.starkStruct.steps[step - 1].nBits;
        fold_inplace(step, friPol_offset, challenge, nBitsExt, prevBits, currentBits, d_buffers);

        if (step < setupCtx.starkInfo.starkStruct.steps.size() - 1)
        {
            merkelizeFRI_inplace(setupCtx, params, d_params, step, proof, d_friPol, starks.treesFRI[step], currentBits, setupCtx.starkInfo.starkStruct.steps[step + 1].nBits, false);
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
    TimerStopAndLog(STARK_FRI_FOLDING);
   
    TimerStart(STARK_FRI_QUERIES);

    uint64_t friQueries[setupCtx.starkInfo.starkStruct.nQueries];

    TranscriptGL transcriptPermutation(setupCtx.starkInfo.starkStruct.merkleTreeArity, setupCtx.starkInfo.starkStruct.merkleTreeCustom);
    starks.addTranscriptGL(transcriptPermutation, challenge, FIELD_EXTENSION);
    transcriptPermutation.getPermutations(friQueries, setupCtx.starkInfo.starkStruct.nQueries, setupCtx.starkInfo.starkStruct.steps[0].nBits);

    uint64_t nTrees = setupCtx.starkInfo.nStages + setupCtx.starkInfo.customCommits.size() + 2;
    proveQueries_inplace(setupCtx, friQueries, setupCtx.starkInfo.starkStruct.nQueries, proof, starks.treesGL, d_trees, nTrees, d_buffers, setupCtx.starkInfo.nStages, d_params);
    
    for(uint64_t step = 1; step < setupCtx.starkInfo.starkStruct.steps.size(); ++step) {

        FRI<Goldilocks::Element>::proveFRIQueries(friQueries, setupCtx.starkInfo.starkStruct.nQueries, step, setupCtx.starkInfo.starkStruct.steps[step].nBits, proof, starks.treesFRI[step - 1]);
        delete starks.treesFRI[step - 1]->source;
        delete starks.treesFRI[step - 1]->nodes;
    }

    FRI<Goldilocks::Element>::setFinalPol(proof, foldedFRIPol, setupCtx.starkInfo.starkStruct.steps[setupCtx.starkInfo.starkStruct.steps.size() - 1].nBits);
    TimerStopAndLog(STARK_FRI_QUERIES);

    TimerStopAndLog(STARK_STEP_FRI);

    proof.proof.setAirgroupValues(params.airgroupValues); 
    proof.proof.setAirValues(params.airValues);
    proof.proof.proof2pointer(proofBuffer);

    if(!proofFile.empty()) {
        json2file(pointer2json(proofBuffer, setupCtx.starkInfo), proofFile);
    }

    TimerStopAndLog(STARK_PROOF);    


    ///rick!! falta el free dels trees
    cudaFree(d_trees[0].nodes);
    cudaFree(d_trees[1].nodes);
    cudaFree(d_trees[2].nodes);
    cudaFree(d_params.xDivXSub);
    if(d_params.pCustomCommitsFixed != nullptr)
        cudaFree(d_params.pCustomCommitsFixed);
}

#endif