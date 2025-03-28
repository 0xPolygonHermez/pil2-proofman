#ifndef GEN_RECURSIVE_PROOF_GPU_HPP
#define GEN_RECURSIVE_PROOF_GPU_HPP

#include "starks.hpp"
#include "proof2zkinStark.hpp"
#include "cuda_utils.cuh"
#include "gl64_t.cuh"
#include "expressions_gpu.cuh"
#include "starks_gpu.cuh"

// TOTO list: 
// fer que lo dls params vagi igual
// evitar copies inecetssaries
// fer que lo dels arbres vagi igual (primer arreglar els de gen_proof)

template <typename ElementType>
void genRecursiveProof_gpu(SetupCtx &setupCtx, json &globalInfo, uint64_t airgroupId, uint64_t airId, uint64_t instanceId, Goldilocks::Element *trace, Goldilocks::Element *pConstPols, Goldilocks::Element *pConstTree, Goldilocks::Element *publicInputs, uint64_t *proofBuffer, std::string proofFile, DeviceCommitBuffers *d_buffers, bool vadcop)
{

    TimerStart(STARK_PROOF);
    CHECKCUDAERR(cudaDeviceSynchronize());
    double time0 = omp_get_wtime();
    //double time_prev = time0;

    uint64_t N = 1 << setupCtx.starkInfo.starkStruct.nBits;

    ProverHelpers proverHelpers(setupCtx.starkInfo, true);

    FRIProof<Goldilocks::Element> proof(setupCtx.starkInfo, airgroupId, airId, instanceId);

    using TranscriptType = std::conditional_t<std::is_same<ElementType, Goldilocks::Element>::value, TranscriptGL, TranscriptBN128>;

    Starks<Goldilocks::Element> starks(setupCtx, proverHelpers, pConstTree, nullptr, false);

    ExpressionsGPU expressionsCtx(setupCtx, proverHelpers, 128, 2048); 

    StepsParams h_params = {
        trace : (Goldilocks::Element *)d_buffers->d_trace,
        aux_trace : (Goldilocks::Element *)d_buffers->d_aux_trace,
        publicInputs : (Goldilocks::Element *)d_buffers->d_publicInputs,
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
    CHECKCUDAERR(cudaMemcpy(h_params.publicInputs, publicInputs, setupCtx.starkInfo.nPublics * sizeof(Goldilocks::Element), cudaMemcpyHostToDevice));
    CHECKCUDAERR(cudaMalloc(&h_params.evals, setupCtx.starkInfo.evMap.size() * FIELD_EXTENSION * sizeof(Goldilocks::Element)));
    CHECKCUDAERR(cudaMalloc(&h_params.challenges, setupCtx.starkInfo.challengesMap.size() * FIELD_EXTENSION * sizeof(Goldilocks::Element)));
    CHECKCUDAERR(cudaMalloc(&h_params.xDivXSub, FIELD_EXTENSION * setupCtx.starkInfo.openingPoints.size() * sizeof(Goldilocks::Element)));

    StepsParams* d_params;
    CHECKCUDAERR(cudaMalloc(&d_params, sizeof(StepsParams)));
    CHECKCUDAERR(cudaMemcpy(d_params, &h_params, sizeof(StepsParams), cudaMemcpyHostToDevice));

    /*CHECKCUDAERR(cudaDeviceSynchronize());
    double time = omp_get_wtime();
    std::cout << "Rick fins PUNT1 (pre-process) " << time - time0 << " " << time - time_prev << std::endl;
    time_prev = omp_get_wtime();*/

    Goldilocks::Element *d_pBuffHelper = h_params.aux_trace +setupCtx.starkInfo.mapOffsets[std::make_pair("buff_helper_fft", false)];

    TranscriptType transcript(setupCtx.starkInfo.starkStruct.merkleTreeArity, setupCtx.starkInfo.starkStruct.merkleTreeCustom);
    
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
    TimerStopAndLog(STARK_STEP_0);
    /*CHECKCUDAERR(cudaDeviceSynchronize());
    time = omp_get_wtime();
    std::cout << "Rick fins PUNT2 (step0) " << time - time0 << " " << time - time_prev << std::endl;
    time_prev = omp_get_wtime();*/

    TimerStart(STARK_STEP_1);
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

    TimerStart(STARK_COMMIT_STAGE_1);
    starks.commitStage_inplace(1, d_buffers->d_trace, d_buffers->d_aux_trace, d_buffers);
    TimerStopAndLog(STARK_COMMIT_STAGE_1);

    CHECKCUDAERR(cudaDeviceSynchronize());
    /*time = omp_get_wtime();
    std::cout << "Rick fins PUNT3 (step1) " << time - time0 << " " << time - time_prev << std::endl;
    time_prev = omp_get_wtime();*/

    offloadCommit(1, starks.treesGL, (gl64_t*)h_params.aux_trace, proof, setupCtx);
    starks.addTranscript(transcript, &proof.proof.roots[0][0], HASH_SIZE);
    TimerStopAndLog(STARK_STEP_1);

    /*CHECKCUDAERR(cudaDeviceSynchronize());
    time = omp_get_wtime();
    std::cout << "Rick fins PUNT4 (offload) " << time - time0 << " " << time - time_prev << std::endl;
    time_prev = time;*/

    TimerStart(STARK_STEP_2);
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

    Goldilocks::Element *res = new Goldilocks::Element[N * FIELD_EXTENSION];
    Goldilocks::Element *gprod = new Goldilocks::Element[N * FIELD_EXTENSION];
   
    uint64_t gprodFieldId = setupCtx.expressionsBin.hints[0].fields[0].values[0].id;
    uint64_t numFieldId = setupCtx.expressionsBin.hints[0].fields[1].values[0].id;
    uint64_t denFieldId = setupCtx.expressionsBin.hints[0].fields[2].values[0].id;

    Dest destStruct(res, N);
    cudaMalloc(&destStruct.dest_gpu, N * FIELD_EXTENSION * sizeof(Goldilocks::Element));
    destStruct.addParams(numFieldId, setupCtx.expressionsBin.expressionsInfo[numFieldId].destDim);
    destStruct.addParams(denFieldId, setupCtx.expressionsBin.expressionsInfo[denFieldId].destDim, true);

    /*CHECKCUDAERR(cudaDeviceSynchronize());
    time = omp_get_wtime();
    std::cout << "Rick fins PUNT5 (pre-expressions) " << time - time0 << " " << time - time_prev << std::endl;
    time_prev = time;*/

    expressionsCtx.calculateExpressions_gpu(d_params, destStruct, uint64_t(1 << setupCtx.starkInfo.starkStruct.nBits), false);

    CHECKCUDAERR(cudaDeviceSynchronize());
    CHECKCUDAERR(cudaFree(destStruct.dest_gpu));

    /*CHECKCUDAERR(cudaDeviceSynchronize());
    time = omp_get_wtime();
    std::cout << "Rick fins PUNT6 (expressions) " << time - time0 << " " << time - time_prev << std::endl;
    time_prev = time;*/

    Goldilocks3::copy((Goldilocks3::Element *)&gprod[0], &Goldilocks3::one());
    for(uint64_t i = 1; i < N; ++i) {
        Goldilocks3::mul((Goldilocks3::Element *)&gprod[i * FIELD_EXTENSION], (Goldilocks3::Element *)&gprod[(i - 1) * FIELD_EXTENSION], (Goldilocks3::Element *)&res[(i - 1) * FIELD_EXTENSION]);
    }


    Goldilocks::Element *d_grod;
    cudaMalloc(&d_grod, N * FIELD_EXTENSION * sizeof(Goldilocks::Element));
    cudaMemcpy(d_grod, gprod, N * FIELD_EXTENSION * sizeof(Goldilocks::Element), cudaMemcpyHostToDevice);

    uint64_t offset = setupCtx.starkInfo.getTraceOffset("cm", setupCtx.starkInfo.cmPolsMap[gprodFieldId], false);
    uint64_t nCols = setupCtx.starkInfo.getTraceNColsSection("cm", setupCtx.starkInfo.cmPolsMap[gprodFieldId], false);

    /*CHECKCUDAERR(cudaDeviceSynchronize());
    time = omp_get_wtime();
    std::cout << "Rick fins PUNT7 (gprod) " << time - time0 << " " << time - time_prev << std::endl;
    time_prev = time;*/

    dim3 nThreads(256);
    dim3 nBlocks((N + nThreads.x - 1) / nThreads.x);
    insertTracePol<<<nBlocks, nThreads>>>((Goldilocks::Element *)d_buffers->d_aux_trace, offset, nCols, d_grod, FIELD_EXTENSION, N);

    delete res;
    delete gprod;
    CHECKCUDAERR(cudaDeviceSynchronize());
    CHECKCUDAERR(cudaFree(d_grod));

    /*CHECKCUDAERR(cudaDeviceSynchronize());
    time = omp_get_wtime();
    std::cout << "Rick fins PUNT8 (upload) " << time - time0 << " " << time - time_prev << std::endl;
    time_prev = time;*/

    TimerStart(CALCULATE_IM_POLS);
    calculateImPolsExpressions(setupCtx, expressionsCtx, h_params, d_params, 2);
    TimerStopAndLog(CALCULATE_IM_POLS);

    /*CHECKCUDAERR(cudaDeviceSynchronize());
    time = omp_get_wtime();
    std::cout << "Rick fins PUNT10 (expressions im pols) " << time - time0 << " " << time - time_prev << std::endl;
    time_prev = time;*/

    TimerStart(STARK_COMMIT_STAGE_2);
    starks.commitStage_inplace(2, (gl64_t*)h_params.trace, (gl64_t*)h_params.aux_trace, d_buffers);

    /*CHECKCUDAERR(cudaDeviceSynchronize());
    time = omp_get_wtime();
    std::cout << "Rick fins PUNT11 (commit) " << time - time0 << " " << time - time_prev << std::endl;
    time_prev = time;*/

    offloadCommit(2, starks.treesGL, (gl64_t*)h_params.aux_trace, proof, setupCtx);

    TimerStopAndLog(STARK_COMMIT_STAGE_2);
    starks.addTranscript(transcript, &proof.proof.roots[1][0], HASH_SIZE);
    TimerStopAndLog(STARK_STEP_2);
    /*CHECKCUDAERR(cudaDeviceSynchronize());
    time = omp_get_wtime();
    std::cout << "Rick fins PUNT12 (offload) " << time - time0 << " " << time - time_prev << std::endl;
    time_prev = time;*/

    TimerStart(STARK_STEP_Q);
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

    TimerStart(STARK_CALCULATE_QUOTIENT_POLYNOMIAL);
    calculateExpression(setupCtx, expressionsCtx, d_params, (Goldilocks::Element *)(h_params.aux_trace + setupCtx.starkInfo.mapOffsets[std::make_pair("q", true)]), setupCtx.starkInfo.cExpId);
    TimerStopAndLog(STARK_CALCULATE_QUOTIENT_POLYNOMIAL);

    TimerStart(STARK_COMMIT_QUOTIENT_POLYNOMIAL);
    starks.commitStage_inplace(setupCtx.starkInfo.nStages + 1, nullptr, d_buffers->d_aux_trace, d_buffers);

    /*CHECKCUDAERR(cudaDeviceSynchronize());
    time = omp_get_wtime();
    std::cout << "Rick fins PUNT15 (Q commit) " << time - time0 << " " << time - time_prev << std::endl;
    time_prev = time;*/

    offloadCommit(setupCtx.starkInfo.nStages + 1, starks.treesGL, (gl64_t *)h_params.aux_trace, proof, setupCtx);

    TimerStopAndLog(STARK_COMMIT_QUOTIENT_POLYNOMIAL);
    starks.addTranscript(transcript, &proof.proof.roots[setupCtx.starkInfo.nStages][0], HASH_SIZE);
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

    /*CHECKCUDAERR(cudaDeviceSynchronize());
    time = omp_get_wtime();
    std::cout << "Rick fins PUNT16 (Q offload) " << time - time0 << " " << time - time_prev << std::endl;
    time_prev = time;*/

    computeLEv_inplace(xiChallenge, setupCtx.starkInfo.starkStruct.nBits, setupCtx.starkInfo.openingPoints.size(), setupCtx.starkInfo.openingPoints.data(), d_buffers, setupCtx.starkInfo.mapOffsets[std::make_pair("buff_helper_fft_lev", false)], d_LEv);

    /*CHECKCUDAERR(cudaDeviceSynchronize());
    time = omp_get_wtime();
    std::cout << "Rick fins PUNT17 (LEv) " << time - time0 << " " << time - time_prev << std::endl;
    time_prev = time;*/

    evmap_inplace(evals, h_params, proof, &starks, d_buffers, (Goldilocks::Element*)d_LEv);
    /*CHECKCUDAERR(cudaDeviceSynchronize());
    time = omp_get_wtime();
    std::cout << "Rick fins PUNT18 (Evmap) " << time - time0 << " " << time - time_prev << std::endl;
    time_prev = time;*/

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

    TimerStart(COMPUTE_FRI_POLYNOMIAL);

    calculateXis_inplace(setupCtx, h_params, xiChallenge);    
    calculateExpression(setupCtx, expressionsCtx, d_params, (Goldilocks::Element *)(h_params.aux_trace + setupCtx.starkInfo.mapOffsets[std::make_pair("f", true)]), setupCtx.starkInfo.friExpId);

    /*CHECKCUDAERR(cudaDeviceSynchronize());
    time = omp_get_wtime();
    std::cout << "Rick fins PUNT22 (expressions FRI) " << time - time0 << " " << time - time_prev << std::endl;
    time_prev = time;*/

    TimerStopAndLog(COMPUTE_FRI_POLYNOMIAL);

    Goldilocks::Element challenge[FIELD_EXTENSION];
    uint64_t friPol_offset = setupCtx.starkInfo.mapOffsets[std::make_pair("f", true)];
    gl64_t *d_friPol = (gl64_t *)(h_params.aux_trace + friPol_offset);
    
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
            merkelizeFRI_inplace(setupCtx, h_params, step, proof, d_friPol, starks.treesFRI[step], currentBits, setupCtx.starkInfo.starkStruct.steps[step + 1].nBits);
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
    proveQueries_inplace(setupCtx, friQueries, setupCtx.starkInfo.starkStruct.nQueries, proof, starks.treesGL, nTrees, d_buffers, setupCtx.starkInfo.nStages, h_params);
    
    for(uint64_t step = 1; step < setupCtx.starkInfo.starkStruct.steps.size(); ++step) {

        FRI<Goldilocks::Element>::proveFRIQueries(friQueries, setupCtx.starkInfo.starkStruct.nQueries, step, setupCtx.starkInfo.starkStruct.steps[step].nBits, proof, starks.treesFRI[step - 1]);
    }

    FRI<Goldilocks::Element>::setFinalPol(proof, foldedFRIPol, setupCtx.starkInfo.starkStruct.steps[setupCtx.starkInfo.starkStruct.steps.size() - 1].nBits);
    TimerStopAndLog(STARK_FRI_QUERIES);

    TimerStopAndLog(STARK_STEP_FRI);

    /*CHECKCUDAERR(cudaDeviceSynchronize());
    time = omp_get_wtime();
    std::cout << "Rick fins PUNT23 (FRI) " << time - time0 << " " << time - time_prev << std::endl;
    time_prev = time;*/

    
    TimerStopAndLog(STARK_PROOF);

    proof.proof.proof2pointer(proofBuffer);
    if(!proofFile.empty()) {
        json2file(pointer2json(proofBuffer, setupCtx.starkInfo), proofFile);
    }

    // Free memory
    CHECKCUDAERR(cudaDeviceSynchronize());
    if(h_params.pCustomCommitsFixed != nullptr)
        cudaFree(h_params.pCustomCommitsFixed);
    cudaFree(h_params.evals);
    cudaFree(h_params.xDivXSub);
    cudaFree(h_params.challenges);
    cudaFree(d_params);
    delete[] foldedFRIPol;
    delete challenges;
    delete evals;
}
#endif
