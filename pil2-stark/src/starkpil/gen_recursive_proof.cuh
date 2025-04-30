#ifndef GEN_RECURSIVE_PROOF_GPU_HPP
#define GEN_RECURSIVE_PROOF_GPU_HPP

#include "starks.hpp"
#include "proof2zkinStark.hpp"
#include "cuda_utils.cuh"
#include "gl64_t.cuh"
#include "expressions_gpu.cuh"
#include "starks_gpu.cuh"
#include <iomanip>

#define PRINT_TIME_SUMMARY 1

// TOTO list: 
// fer que lo dls params vagi igual
// evitar copies inecetssaries
// fer que lo dels arbres vagi igual (primer arreglar els de gen_proof)

template <typename ElementType>
void genRecursiveProof_gpu(SetupCtx &setupCtx, json &globalInfo, uint64_t airgroupId, uint64_t airId, uint64_t instanceId, Goldilocks::Element *trace, Goldilocks::Element *pConstPols, Goldilocks::Element *pConstTree, Goldilocks::Element *publicInputs, uint64_t *proofBuffer, std::string proofFile, DeviceCommitBuffers *d_buffers, bool vadcop)
{

    TimerStart(STARK_GPU_PROOF);
    TimerStart(STARK_INITIALIZATION);
    
    double totalNTTTime = 0;
    double totalMerkleTime = 0;
    double nttTime = 0;
    double merkleTime = 0;

    uint64_t N = 1 << setupCtx.starkInfo.starkStruct.nBits;
    uint64_t NExtended = 1 << setupCtx.starkInfo.starkStruct.nBitsExt;

    ProverHelpers proverHelpers(setupCtx.starkInfo, true);

    FRIProof<Goldilocks::Element> proof(setupCtx.starkInfo, airgroupId, airId, instanceId);

    using TranscriptType = std::conditional_t<std::is_same<ElementType, Goldilocks::Element>::value, TranscriptGL, TranscriptBN128>;

    Starks<Goldilocks::Element> starks(setupCtx, proverHelpers, pConstTree, nullptr, false);

    ExpressionsGPU expressionsCtx(setupCtx, proverHelpers, setupCtx.starkInfo.nrowsPack, setupCtx.starkInfo.maxNBlocks); 

    StepsParams h_params = {
        trace : (Goldilocks::Element *)d_buffers->d_trace,
        aux_trace : (Goldilocks::Element *)d_buffers->d_aux_trace,
        publicInputs : nullptr,
        proofValues : nullptr,
        challenges : nullptr,
        airgroupValues : nullptr,
        airValues : nullptr,
        evals : nullptr,
        xDivXSub : nullptr, //rick: canviar nom per xi_s
        pConstPolsAddress : (Goldilocks::Element *)d_buffers->d_constPols,
        pConstPolsExtendedTreeAddress : (Goldilocks::Element *)d_buffers->d_constTree,
        pCustomCommitsFixed : nullptr,
    };
    
    // Allocate memory and copy data
    Goldilocks::Element *evals = new Goldilocks::Element[setupCtx.starkInfo.evMap.size() * FIELD_EXTENSION];
    Goldilocks::Element *challenges = new Goldilocks::Element[setupCtx.starkInfo.challengesMap.size() * FIELD_EXTENSION];
    CHECKCUDAERR(cudaMalloc(&h_params.publicInputs, setupCtx.starkInfo.nPublics * sizeof(Goldilocks::Element)));
    CHECKCUDAERR(cudaMalloc(&h_params.evals, setupCtx.starkInfo.evMap.size() * FIELD_EXTENSION * sizeof(Goldilocks::Element)));
    CHECKCUDAERR(cudaMalloc(&h_params.challenges, setupCtx.starkInfo.challengesMap.size() * FIELD_EXTENSION * sizeof(Goldilocks::Element)));
    CHECKCUDAERR(cudaMalloc(&h_params.xDivXSub, FIELD_EXTENSION * setupCtx.starkInfo.openingPoints.size() * sizeof(Goldilocks::Element)));
    CHECKCUDAERR(cudaMemcpy(h_params.publicInputs, publicInputs, setupCtx.starkInfo.nPublics * sizeof(Goldilocks::Element), cudaMemcpyHostToDevice));

    StepsParams* d_params;
    CHECKCUDAERR(cudaMalloc(&d_params, sizeof(StepsParams)));
    CHECKCUDAERR(cudaMemcpy(d_params, &h_params, sizeof(StepsParams), cudaMemcpyHostToDevice));

    TranscriptType transcript(setupCtx.starkInfo.starkStruct.merkleTreeArity, setupCtx.starkInfo.starkStruct.merkleTreeCustom);    
    TranscriptGL_GPU d_transcript(setupCtx.starkInfo.starkStruct.merkleTreeArity, setupCtx.starkInfo.starkStruct.merkleTreeCustom);
    TimerStopAndLog(STARK_INITIALIZATION);

    //--------------------------------
    // 0.- Add const root and publics to transcript
    //--------------------------------
    TimerStart(STARK_STEP_0);
    ElementType verkey[HASH_SIZE];
    starks.treesGL[setupCtx.starkInfo.nStages + 1]->getRoot(verkey);
    starks.addTranscript(transcript, &verkey[0], HASH_SIZE);
    if (setupCtx.starkInfo.nPublics > 0)
    {
        if (!setupCtx.starkInfo.starkStruct.hashCommits)
        {
            starks.addTranscriptGL(transcript, &publicInputs[0], setupCtx.starkInfo.nPublics);
        }
        else
        {
            ElementType hash[HASH_SIZE];
            starks.calculateHash(hash, &publicInputs[0], setupCtx.starkInfo.nPublics);
            starks.addTranscript(transcript, hash, HASH_SIZE);
        }
    }
    uint64_t min_challenge = setupCtx.starkInfo.challengesMap.size();
    uint64_t max_challenge = 0;
    for (uint64_t i = 0; i < setupCtx.starkInfo.challengesMap.size(); i++)
    {
        if (setupCtx.starkInfo.challengesMap[i].stage == 1)
        {
            min_challenge = std::min(min_challenge, i);
            max_challenge = std::max(max_challenge, i);
            starks.getChallenge(transcript, challenges[i * FIELD_EXTENSION]);
        }
    }
    if(min_challenge == setupCtx.starkInfo.challengesMap.size()) min_challenge =0;
    CHECKCUDAERR(cudaMemcpy(h_params.challenges + min_challenge * FIELD_EXTENSION, challenges + min_challenge * FIELD_EXTENSION, (max_challenge - min_challenge + 1) * FIELD_EXTENSION * sizeof(Goldilocks::Element), cudaMemcpyHostToDevice));
    TimerStopAndLog(STARK_STEP_0);    

    TimerStart(STARK_COMMIT_STAGE_1);
    commitStage_inplace(1, setupCtx, starks.treesGL, d_buffers->d_trace, d_buffers->d_aux_trace, d_buffers, &d_transcript, &nttTime, &merkleTime);
    totalNTTTime += nttTime;
    totalMerkleTime += merkleTime;
    offloadCommit(1, starks.treesGL, (gl64_t*)h_params.aux_trace, proof, setupCtx);
    starks.addTranscript(transcript, &proof.proof.roots[0][0], HASH_SIZE);
    min_challenge = setupCtx.starkInfo.challengesMap.size();
    max_challenge = 0;
    for (uint64_t i = 0; i < setupCtx.starkInfo.challengesMap.size(); i++)
    {
        if (setupCtx.starkInfo.challengesMap[i].stage == 2)
        {
            min_challenge = std::min(min_challenge, i);
            max_challenge = std::max(max_challenge, i);
            starks.getChallenge(transcript, challenges[i * FIELD_EXTENSION]);
        }
    }
    CHECKCUDAERR(cudaMemcpy(h_params.challenges + min_challenge * FIELD_EXTENSION, challenges + min_challenge * FIELD_EXTENSION, (max_challenge - min_challenge + 1) * FIELD_EXTENSION * sizeof(Goldilocks::Element), cudaMemcpyHostToDevice));
    TimerStopAndLog(STARK_COMMIT_STAGE_1);


    TimerStart(STARK_CALCULATE_GPROD);
    

    Goldilocks::Element *res = new Goldilocks::Element[N * FIELD_EXTENSION];
    Goldilocks::Element *gprod = new Goldilocks::Element[N * FIELD_EXTENSION];
    uint64_t gprodFieldId = setupCtx.expressionsBin.hints[0].fields[0].values[0].id;
    uint64_t numFieldId = setupCtx.expressionsBin.hints[0].fields[1].values[0].id;
    uint64_t denFieldId = setupCtx.expressionsBin.hints[0].fields[2].values[0].id;

    Dest destStruct(res, N);
    CHECKCUDAERR(cudaMalloc(&destStruct.dest_gpu, N * FIELD_EXTENSION * sizeof(Goldilocks::Element)));
    destStruct.addParams(numFieldId, setupCtx.expressionsBin.expressionsInfo[numFieldId].destDim);
    destStruct.addParams(denFieldId, setupCtx.expressionsBin.expressionsInfo[denFieldId].destDim, true);
    uint64_t xn_offset = setupCtx.starkInfo.mapOffsets[std::make_pair("x_n", false)];
    CHECKCUDAERR(cudaMemcpy(h_params.aux_trace + xn_offset, proverHelpers.x_n, N * sizeof(Goldilocks::Element), cudaMemcpyHostToDevice));            

    expressionsCtx.calculateExpressions_gpu(d_params, destStruct, uint64_t(1 << setupCtx.starkInfo.starkStruct.nBits), false);

    CHECKCUDAERR(cudaFree(destStruct.dest_gpu));

    Goldilocks3::copy((Goldilocks3::Element *)&gprod[0], &Goldilocks3::one());
    for(uint64_t i = 1; i < N; ++i) {
        Goldilocks3::mul((Goldilocks3::Element *)&gprod[i * FIELD_EXTENSION], (Goldilocks3::Element *)&gprod[(i - 1) * FIELD_EXTENSION], (Goldilocks3::Element *)&res[(i - 1) * FIELD_EXTENSION]);
    }

    Goldilocks::Element *d_grod;
    CHECKCUDAERR(cudaMalloc(&d_grod, N * FIELD_EXTENSION * sizeof(Goldilocks::Element)));
    CHECKCUDAERR(cudaMemcpy(d_grod, gprod, N * FIELD_EXTENSION * sizeof(Goldilocks::Element), cudaMemcpyHostToDevice));

    uint64_t offset = setupCtx.starkInfo.getTraceOffset("cm", setupCtx.starkInfo.cmPolsMap[gprodFieldId], false);
    uint64_t nCols = setupCtx.starkInfo.getTraceNColsSection("cm", setupCtx.starkInfo.cmPolsMap[gprodFieldId], false);

    dim3 nThreads(256);
    dim3 nBlocks((N + nThreads.x - 1) / nThreads.x);
    insertTracePol<<<nBlocks, nThreads>>>((Goldilocks::Element *)d_buffers->d_aux_trace, offset, nCols, d_grod, FIELD_EXTENSION, N);

    delete res;
    delete gprod;
    CHECKCUDAERR(cudaFree(d_grod));

   TimerStopAndLog(STARK_CALCULATE_GPROD);

    TimerStart(CALCULATE_IM_POLS);
    calculateImPolsExpressions(setupCtx, expressionsCtx, h_params, d_params, 2);
    TimerStopAndLog(CALCULATE_IM_POLS);


    TimerStart(STARK_COMMIT_STAGE_2);
    commitStage_inplace(2, setupCtx, starks.treesGL, (gl64_t*)h_params.trace, (gl64_t*)h_params.aux_trace, d_buffers, &d_transcript, &nttTime, &merkleTime);
    totalNTTTime += nttTime;
    totalMerkleTime += merkleTime;
    offloadCommit(2, starks.treesGL, (gl64_t*)h_params.aux_trace, proof, setupCtx);
    starks.addTranscript(transcript, &proof.proof.roots[1][0], HASH_SIZE);
    min_challenge = setupCtx.starkInfo.challengesMap.size();
    max_challenge = 0;
    for (uint64_t i = 0; i < setupCtx.starkInfo.challengesMap.size(); i++)
    {
        if (setupCtx.starkInfo.challengesMap[i].stage == setupCtx.starkInfo.nStages + 1)
        {
            min_challenge = std::min(min_challenge, i);
            max_challenge = std::max(max_challenge, i);
            starks.getChallenge(transcript, challenges[i * FIELD_EXTENSION]);
        }
    }
    CHECKCUDAERR(cudaMemcpy(h_params.challenges + min_challenge * FIELD_EXTENSION, challenges + min_challenge * FIELD_EXTENSION, (max_challenge - min_challenge + 1) * FIELD_EXTENSION * sizeof(Goldilocks::Element), cudaMemcpyHostToDevice));
    TimerStopAndLog(STARK_COMMIT_STAGE_2);
    

    TimerStart(STARK_STEP_Q);
    
    TimerStart(STARK_STEP_Q_EXPRESSIONS);
    uint64_t zi_offset = setupCtx.starkInfo.mapOffsets[std::make_pair("zi", true)];
    uint64_t x_offset = setupCtx.starkInfo.mapOffsets[std::make_pair("x", true)];
    CHECKCUDAERR(cudaMemcpy(h_params.aux_trace + zi_offset, proverHelpers.zi, setupCtx.starkInfo.boundaries.size() * NExtended * sizeof(Goldilocks::Element), cudaMemcpyHostToDevice));            
    CHECKCUDAERR(cudaMemcpy(h_params.aux_trace + x_offset, proverHelpers.x, NExtended * sizeof(Goldilocks::Element), cudaMemcpyHostToDevice));            
    calculateExpression(setupCtx, expressionsCtx, d_params, (Goldilocks::Element *)(h_params.aux_trace + setupCtx.starkInfo.mapOffsets[std::make_pair("q", true)]), setupCtx.starkInfo.cExpId);
    TimerStopAndLog(STARK_STEP_Q_EXPRESSIONS);

    TimerStart(STARK_STEP_Q_COMMIT);

    commitStage_inplace(setupCtx.starkInfo.nStages + 1, setupCtx, starks.treesGL, nullptr, d_buffers->d_aux_trace, d_buffers, &d_transcript, &nttTime, &merkleTime);
    totalNTTTime += nttTime;
    totalMerkleTime += merkleTime;
    offloadCommit(setupCtx.starkInfo.nStages + 1, starks.treesGL, (gl64_t *)h_params.aux_trace, proof, setupCtx);
    starks.addTranscript(transcript, &proof.proof.roots[setupCtx.starkInfo.nStages][0], HASH_SIZE);

    TimerStopAndLog(STARK_STEP_Q_COMMIT);
    TimerStopAndLog(STARK_STEP_Q);

    TimerStart(STARK_STEP_EVALS);
    uint64_t xiChallengeIndex = 0;
    min_challenge = setupCtx.starkInfo.challengesMap.size();
    max_challenge = 0;
    for (uint64_t i = 0; i < setupCtx.starkInfo.challengesMap.size(); i++)
    {
        if (setupCtx.starkInfo.challengesMap[i].stage == setupCtx.starkInfo.nStages + 2)
        {
            if (setupCtx.starkInfo.challengesMap[i].stageId == 0)
                xiChallengeIndex = i;
            min_challenge = std::min(min_challenge, i);
            max_challenge = std::max(max_challenge, i);
            starks.getChallenge(transcript, challenges[i * FIELD_EXTENSION]);
        }
    }
    CHECKCUDAERR(cudaMemcpy(h_params.challenges + min_challenge * FIELD_EXTENSION, challenges + min_challenge * FIELD_EXTENSION, (max_challenge - min_challenge + 1) * FIELD_EXTENSION * sizeof(Goldilocks::Element), cudaMemcpyHostToDevice));

    Goldilocks::Element *xiChallenge = &challenges[xiChallengeIndex * FIELD_EXTENSION];
    gl64_t * d_LEv = (gl64_t *)  h_params.aux_trace +setupCtx.starkInfo.mapOffsets[std::make_pair("lev", false)];;

    TimerStart(STARK_STEP_EVMAP);
    for(uint64_t i = 0; i < setupCtx.starkInfo.openingPoints.size(); i += 4) {
        std::vector<int64_t> openingPoints;
        for(uint64_t j = 0; j < 4; ++j) {
            if(i + j < setupCtx.starkInfo.openingPoints.size()) {
                openingPoints.push_back(setupCtx.starkInfo.openingPoints[i + j]);
            }
        }
        uint64_t offset_helper = setupCtx.starkInfo.mapOffsets[std::make_pair("extra_helper_fft_lev", false)];
        computeLEv_inplace(xiChallenge, setupCtx.starkInfo.starkStruct.nBits, openingPoints.size(), openingPoints.data(), d_buffers, offset_helper, d_LEv, &nttTime);
        totalNTTTime += nttTime;
        evmap_inplace(h_params, &starks, d_buffers, openingPoints.size(), openingPoints.data(), (Goldilocks::Element*)d_LEv);
    }
    TimerStopAndLog(STARK_STEP_EVMAP);

    if (!setupCtx.starkInfo.starkStruct.hashCommits)
    {
        starks.addTranscriptGL(transcript, evals, setupCtx.starkInfo.evMap.size() * FIELD_EXTENSION);
    }
    else
    {
        ElementType hash[HASH_SIZE];
        starks.calculateHash(hash, evals, setupCtx.starkInfo.evMap.size() * FIELD_EXTENSION);
        starks.addTranscript(transcript, hash, HASH_SIZE);
    }

    // Challenges for FRI polynomial
    min_challenge = setupCtx.starkInfo.challengesMap.size();
    max_challenge = 0;
    for (uint64_t i = 0; i < setupCtx.starkInfo.challengesMap.size(); i++)
    {
        if (setupCtx.starkInfo.challengesMap[i].stage == setupCtx.starkInfo.nStages + 3)
        {
            min_challenge = std::min(min_challenge, i);
            max_challenge = std::max(max_challenge, i);
            starks.getChallenge(transcript, challenges[i * FIELD_EXTENSION]);
        }
    }
    CHECKCUDAERR(cudaMemcpy(h_params.challenges + min_challenge * FIELD_EXTENSION, challenges + min_challenge * FIELD_EXTENSION, (max_challenge - min_challenge + 1) * FIELD_EXTENSION * sizeof(Goldilocks::Element), cudaMemcpyHostToDevice));
    TimerStopAndLog(STARK_STEP_EVALS);

    //--------------------------------
    // 6. Compute FRI
    //--------------------------------
    TimerStart(STARK_STEP_FRI);
    TimerStart(STARK_STEP_FRI_XIS);
    calculateXis_inplace(setupCtx, h_params, xiChallenge);    
    TimerStopAndLog(STARK_STEP_FRI_XIS);

    TimerStart(STARK_STEP_FRI_POLYNOMIAL);
    calculateExpression(setupCtx, expressionsCtx, d_params, (Goldilocks::Element *)(h_params.aux_trace + setupCtx.starkInfo.mapOffsets[std::make_pair("f", true)]), setupCtx.starkInfo.friExpId);
    for(uint64_t step = 0; step < setupCtx.starkInfo.starkStruct.steps.size() - 1; ++step) { 
        Goldilocks::Element *src = h_params.aux_trace + setupCtx.starkInfo.mapOffsets[std::make_pair("fri_" + to_string(step + 1), true)];
        starks.treesFRI[step]->setSource(src);

        if(setupCtx.starkInfo.starkStruct.verificationHashType == "GL") {
            Goldilocks::Element *pBuffNodesGL = h_params.aux_trace + setupCtx.starkInfo.mapOffsets[std::make_pair("mt_fri_" + to_string(step + 1), true)];
            ElementType *pBuffNodes = (ElementType *)pBuffNodesGL;
            starks.treesFRI[step]->setNodes(pBuffNodes);
        }
    }
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
            merkelizeFRI_inplace(setupCtx, h_params, step, d_friPol, starks.treesFRI[step], currentBits, setupCtx.starkInfo.starkStruct.steps[step + 1].nBits, &d_transcript, &merkleTime);
            totalMerkleTime += merkleTime;
            offloadCommitFRI(step, starks.treesFRI[step], proof, setupCtx);
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

    gl64_t *d_constTree = d_buffers->d_constTree;

    uint64_t nTrees = setupCtx.starkInfo.nStages + setupCtx.starkInfo.customCommits.size() + 2;
    // TODO: WRONG, d_queries_buff not declared
    proveQueries_inplace(setupCtx, d_constTree, friQueries, setupCtx.starkInfo.starkStruct.nQueries, proof, starks.treesGL, nTrees, d_buffers, d_constTree, setupCtx.starkInfo.nStages, h_params);
    
    for(uint64_t step = 1; step < setupCtx.starkInfo.starkStruct.steps.size(); ++step) {
        // TODO: WRONG, d_queries_buff not declared
        proveFRIQueries_inplace(setupCtx, d_constTree, step, setupCtx.starkInfo.starkStruct.steps[step].nBits, friQueries, setupCtx.starkInfo.starkStruct.nQueries, proof, starks.treesFRI[step - 1]);
    }


    FRI<Goldilocks::Element>::setFinalPol(proof, foldedFRIPol, setupCtx.starkInfo.starkStruct.steps[setupCtx.starkInfo.starkStruct.steps.size() - 1].nBits);
    TimerStopAndLog(STARK_STEP_FRI_QUERIES);

    TimerStopAndLog(STARK_STEP_FRI);
    TimerStart(STARK_POSTPROCESS);

    proof.proof.proof2pointer(proofBuffer);
    if(!proofFile.empty()) {
        json2file(pointer2json(proofBuffer, setupCtx.starkInfo), proofFile);
    }

    // Free memory
    if(h_params.pCustomCommitsFixed != nullptr)
        cudaFree(h_params.pCustomCommitsFixed);
    cudaFree(h_params.evals);
    cudaFree(h_params.xDivXSub);
    cudaFree(h_params.challenges);
    cudaFree(d_params);  
    delete[] foldedFRIPol;
    delete challenges;
    delete evals;
    TimerStopAndLog(STARK_POSTPROCESS);
    TimerStopAndLog(STARK_GPU_PROOF);

    #if PRINT_TIME_SUMMARY

    double time_total = TimerGetElapsed(STARK_GPU_PROOF);

    std::ostringstream oss;

    zklog.trace("    TIMES SUMMARY: ");

    double expressions_time = TimerGetElapsed(STARK_CALCULATE_GPROD) + TimerGetElapsed(CALCULATE_IM_POLS) + TimerGetElapsed(STARK_STEP_Q_EXPRESSIONS) + TimerGetElapsed(STARK_STEP_FRI_POLYNOMIAL);;
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
