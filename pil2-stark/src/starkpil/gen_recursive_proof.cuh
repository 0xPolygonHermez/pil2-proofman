#ifndef GEN_RECURSIVE_PROOF_GPU_HPP
#define GEN_RECURSIVE_PROOF_GPU_HPP

#include "starks.hpp"
#include "proof2zkinStark.hpp"
#include "cuda_utils.cuh"
#include "gl64_t.cuh"
#include "expressions_gpu.cuh"
#include "starks_gpu.cuh"

void offloadCommit(uint64_t step, MerkleTreeGL **treesGL, gl64_t *d_aux_trace, uint64_t *d_tree, FRIProof<Goldilocks::Element> &proof, SetupCtx &setupCtx)
{

    double time = omp_get_wtime();
    uint64_t NExtended = 1 << setupCtx.starkInfo.starkStruct.nBitsExt;
    uint64_t tree_size = treesGL[step - 1]->getNumNodes(NExtended);
    std::string section = "cm" + to_string(step);
    uint64_t offset = setupCtx.starkInfo.mapOffsets[make_pair(section, true)];
    // treesGL[step - 1]->setSource(trace + offset);
    treesGL[step - 1]->souceTraceOffset = offset;
    time = omp_get_wtime() - time;
    std::cout << "offloadPart1: " << time << std::endl;

    uint32_t nFielsElements = treesGL[step - 1]->getMerkleTreeNFieldElements();
    // sync the GPU
    CHECKCUDAERR(cudaDeviceSynchronize());
    time = omp_get_wtime();
    CHECKCUDAERR(cudaMemcpy(&proof.proof.roots[step - 1][0], &d_tree[tree_size - nFielsElements], nFielsElements * sizeof(uint64_t), cudaMemcpyDeviceToHost));
    time = omp_get_wtime() - time;
    std::cout << "offloadPart3: " << time << std::endl;
}

template <typename ElementType>
void genRecursiveProof_gpu(SetupCtx &setupCtx, json &globalInfo, uint64_t airgroupId, uint64_t airId, uint64_t instanceId, Goldilocks::Element *trace, Goldilocks::Element *pConstPols, Goldilocks::Element *pConstTree, Goldilocks::Element *publicInputs, uint64_t *proofBuffer, std::string proofFile, DeviceCommitBuffers *d_buffers, bool vadcop)
{

#if 0

    TimerStart(STARK_PROOF);
    CHECKCUDAERR(cudaDeviceSynchronize());
    double time0 = omp_get_wtime();
    double time_prev = time0;

    CHECKCUDAERR(cudaMemset(d_buffers->d_aux_trace, 0, setupCtx.starkInfo.mapTotalN * sizeof(Goldilocks::Element)));
    uint64_t N = 1 << setupCtx.starkInfo.starkStruct.nBits;

    ProverHelpers proverHelpers(setupCtx.starkInfo, false);

    FRIProof<Goldilocks::Element> proof(setupCtx.starkInfo, airgroupId, airId, instanceId);

    using TranscriptType = std::conditional_t<std::is_same<ElementType, Goldilocks::Element>::value, TranscriptGL, TranscriptBN128>;

    Starks<Goldilocks::Element> starks(setupCtx, proverHelpers, pConstTree, nullptr, true); //rick: initialze starks


    uint64_t nFieldElements = setupCtx.starkInfo.starkStruct.verificationHashType == std::string("BN128") ? 1 : HASH_SIZE;
    
    // GPU tree-nodes
    GPUTree *d_trees = new GPUTree[setupCtx.starkInfo.nStages + 2];
    for (uint64_t i = 0; i < setupCtx.starkInfo.nStages + 1; i++)
    {
        d_trees[i].nFieldElements = nFieldElements;
    }

    ExpressionsGPU expressionsCtx(setupCtx, proverHelpers, 2, 1176, 465, 128, 2048); //maxNparams, maxNTemp1, maxNTemp3

    TranscriptType transcript(setupCtx.starkInfo.starkStruct.merkleTreeArity, setupCtx.starkInfo.starkStruct.merkleTreeCustom);

    Goldilocks::Element *evals = new Goldilocks::Element[setupCtx.starkInfo.evMap.size() * FIELD_EXTENSION];
    Goldilocks::Element *challenges = new Goldilocks::Element[setupCtx.starkInfo.challengesMap.size() * FIELD_EXTENSION];
    Goldilocks::Element *airgroupValues = nullptr;

    Goldilocks::Element *d_evals;
    CHECKCUDAERR(cudaMalloc(&d_evals, setupCtx.starkInfo.evMap.size() * FIELD_EXTENSION * sizeof(Goldilocks::Element)));

    StepsParams params = {
        trace : trace,
        aux_trace : nullptr,
        publicInputs : publicInputs,
        challenges : challenges,
        airgroupValues : nullptr,
        evals : evals,
        xDivXSub : nullptr,
        pConstPolsAddress : pConstPols,
        pConstPolsExtendedTreeAddress : pConstTree,
    };

    StepsParams d_params = {
        trace : (Goldilocks::Element *)d_buffers->d_trace,
        aux_trace : (Goldilocks::Element *)d_buffers->d_aux_trace,
        publicInputs : (Goldilocks::Element *)d_buffers->d_publicInputs,
        challenges : nullptr,
        airgroupValues : nullptr,
        evals : d_evals,
        xDivXSub : nullptr,
        pConstPolsAddress : (Goldilocks::Element *)d_buffers->d_constPols,
        pConstPolsExtendedTreeAddress : (Goldilocks::Element *)d_buffers->d_constTree,
    };

    CHECKCUDAERR(cudaDeviceSynchronize());
    double time = omp_get_wtime();
    std::cout << "Rick fins PUNT1 (pre-process) " << time - time0 << " " << time - time_prev << std::endl;
    time_prev = omp_get_wtime();

    //--------------------------------
    // 0.- Add const root and publics to transcript
    //--------------------------------
    TimerStart(STARK_STEP_0);
    ElementType verkey[nFieldElements];
    starks.treesGL[setupCtx.starkInfo.nStages + 1]->getRoot(verkey);
    starks.addTranscript(transcript, &verkey[0], nFieldElements);
    if (setupCtx.starkInfo.nPublics > 0)
    {
        if (!setupCtx.starkInfo.starkStruct.hashCommits)
        {
            starks.addTranscriptGL(transcript, &publicInputs[0], setupCtx.starkInfo.nPublics);
        }
        else
        {
            ElementType hash[nFieldElements];
            starks.calculateHash(hash, &publicInputs[0], setupCtx.starkInfo.nPublics);
            starks.addTranscript(transcript, hash, nFieldElements);
        }
    }
    TimerStopAndLog(STARK_STEP_0);
    CHECKCUDAERR(cudaDeviceSynchronize());
    time = omp_get_wtime();
    std::cout << "Rick fins PUNT2 (step0) " << time - time0 << " " << time - time_prev << std::endl;
    time_prev = omp_get_wtime();

    TimerStart(STARK_STEP_1);
    for (uint64_t i = 0; i < setupCtx.starkInfo.challengesMap.size(); i++)
    {
        if (setupCtx.starkInfo.challengesMap[i].stage == 1)
        {
            starks.getChallenge(transcript, challenges[i * FIELD_EXTENSION]);
        }
    }

    TimerStart(STARK_COMMIT_STAGE_1);
    starks.commitStage_inplace(1, d_buffers->d_trace, d_buffers->d_aux_trace, (uint64_t **)(&d_trees[0].nodes), d_buffers);
    TimerStopAndLog(STARK_COMMIT_STAGE_1);

    CHECKCUDAERR(cudaDeviceSynchronize());
    time = omp_get_wtime();
    std::cout << "Rick fins PUNT3 (step1) " << time - time0 << " " << time - time_prev << std::endl;
    time_prev = omp_get_wtime();

    offloadCommit(1, starks.treesGL, d_buffers->d_aux_trace, (uint64_t *)d_trees[0].nodes, proof, setupCtx);
    starks.addTranscript(transcript, &proof.proof.roots[0][0], nFieldElements);
    TimerStopAndLog(STARK_STEP_1);

    CHECKCUDAERR(cudaDeviceSynchronize());
    time = omp_get_wtime();
    std::cout << "Rick fins PUNT4 (offload) " << time - time0 << " " << time - time_prev << std::endl;
    time_prev = time;

    TimerStart(STARK_STEP_2);
    for (uint64_t i = 0; i < setupCtx.starkInfo.challengesMap.size(); i++)
    {
        if (setupCtx.starkInfo.challengesMap[i].stage == 2)
        {
            starks.getChallenge(transcript, challenges[i * FIELD_EXTENSION]);
        }
    }

    Goldilocks::Element *res = new Goldilocks::Element[N * FIELD_EXTENSION];
    Goldilocks::Element *gprod = new Goldilocks::Element[N * FIELD_EXTENSION];
   
    uint64_t gprodFieldId = setupCtx.expressionsBin.hints[0].fields[0].values[0].id;
    uint64_t numFieldId = setupCtx.expressionsBin.hints[0].fields[1].values[0].id;
    uint64_t denFieldId = setupCtx.expressionsBin.hints[0].fields[2].values[0].id;

    Dest destStruct(res, N);
    cudaMalloc(&destStruct.dest_gpu, N * FIELD_EXTENSION * sizeof(Goldilocks::Element));
    destStruct.addParams(setupCtx.expressionsBin.expressionsInfo[numFieldId]);
    destStruct.addParams(setupCtx.expressionsBin.expressionsInfo[denFieldId], true);
    std::vector<Dest> dests = {destStruct};

    CHECKCUDAERR(cudaDeviceSynchronize());
    time = omp_get_wtime();
    std::cout << "Rick fins PUNT5 (pre-expressions) " << time - time0 << " " << time - time_prev << std::endl;
    time_prev = time;

    expressionsCtx.calculateExpressions_gpu(params, d_params, setupCtx.expressionsBin.expressionsBinArgsExpressions, dests, uint64_t(1 << setupCtx.starkInfo.starkStruct.nBits));

    cudaFree(destStruct.dest_gpu);

    CHECKCUDAERR(cudaDeviceSynchronize());
    time = omp_get_wtime();
    std::cout << "Rick fins PUNT6 (expressions) " << time - time0 << " " << time - time_prev << std::endl;
    time_prev = time;

    Goldilocks3::copy((Goldilocks3::Element *)&gprod[0], &Goldilocks3::one());
    for(uint64_t i = 1; i < N; ++i) {
        Goldilocks3::mul((Goldilocks3::Element *)&gprod[i * FIELD_EXTENSION], (Goldilocks3::Element *)&gprod[(i - 1) * FIELD_EXTENSION], (Goldilocks3::Element *)&res[(i - 1) * FIELD_EXTENSION]);
    }


    Goldilocks::Element *d_grod;
    cudaMalloc(&d_grod, N * FIELD_EXTENSION * sizeof(Goldilocks::Element));
    cudaMemcpy(d_grod, gprod, N * FIELD_EXTENSION * sizeof(Goldilocks::Element), cudaMemcpyHostToDevice);

    uint64_t offset = setupCtx.starkInfo.getTraceOffset("cm", setupCtx.starkInfo.cmPolsMap[gprodFieldId], false);
    uint64_t nCols = setupCtx.starkInfo.getTraceNColsSection("cm", setupCtx.starkInfo.cmPolsMap[gprodFieldId], false);

    CHECKCUDAERR(cudaDeviceSynchronize());
    time = omp_get_wtime();
    std::cout << "Rick fins PUNT7 (gprod) " << time - time0 << " " << time - time_prev << std::endl;
    time_prev = time;

    dim3 nThreads(256);
    dim3 nBlocks((N + nThreads.x - 1) / nThreads.x);
    insertTracePol<<<nBlocks, nThreads>>>((Goldilocks::Element *)d_buffers->d_aux_trace, offset, nCols, d_grod, FIELD_EXTENSION, N);

    delete res;
    delete gprod;
    cudaFree(d_grod);

    CHECKCUDAERR(cudaDeviceSynchronize());
    time = omp_get_wtime();
    std::cout << "Rick fins PUNT8 (upload) " << time - time0 << " " << time - time_prev << std::endl;
    time_prev = time;

    TimerStart(CALCULATE_IM_POLS);
    calculateImPolsExpressions(setupCtx, expressionsCtx, d_buffers, params, d_params, 2);
    TimerStopAndLog(CALCULATE_IM_POLS);

    CHECKCUDAERR(cudaDeviceSynchronize());
    time = omp_get_wtime();
    std::cout << "Rick fins PUNT10 (expressions im pols) " << time - time0 << " " << time - time_prev << std::endl;
    time_prev = time;

    TimerStart(STARK_COMMIT_STAGE_2);
    starks.commitStage_inplace(2, d_buffers->d_trace, d_buffers->d_aux_trace, (uint64_t **)(&d_trees[1].nodes), d_buffers);

    CHECKCUDAERR(cudaDeviceSynchronize());
    time = omp_get_wtime();
    std::cout << "Rick fins PUNT11 (commit) " << time - time0 << " " << time - time_prev << std::endl;
    time_prev = time;

    offloadCommit(2, starks.treesGL, d_buffers->d_aux_trace, (uint64_t *)d_trees[1].nodes, proof, setupCtx);

    TimerStopAndLog(STARK_COMMIT_STAGE_2);
    starks.addTranscript(transcript, &proof.proof.roots[1][0], nFieldElements);
    TimerStopAndLog(STARK_STEP_2);
    CHECKCUDAERR(cudaDeviceSynchronize());
    time = omp_get_wtime();
    std::cout << "Rick fins PUNT12 (offload) " << time - time0 << " " << time - time_prev << std::endl;
    time_prev = time;

    TimerStart(STARK_STEP_Q);

    for (uint64_t i = 0; i < setupCtx.starkInfo.challengesMap.size(); i++)
    {
        if (setupCtx.starkInfo.challengesMap[i].stage == setupCtx.starkInfo.nStages + 1)
        {
            starks.getChallenge(transcript, challenges[i * FIELD_EXTENSION]);
        }
    }

    calculateExpression(setupCtx, expressionsCtx, params, d_params, (Goldilocks::Element *)(d_params.aux_trace + setupCtx.starkInfo.mapOffsets[std::make_pair("q", true)]), setupCtx.starkInfo.cExpId);

    TimerStart(STARK_COMMIT_QUOTIENT_POLYNOMIAL);
    starks.commitStage_inplace(setupCtx.starkInfo.nStages + 1, nullptr, d_buffers->d_aux_trace, (uint64_t **)(&d_trees[setupCtx.starkInfo.nStages].nodes), d_buffers);

    CHECKCUDAERR(cudaDeviceSynchronize());
    time = omp_get_wtime();
    std::cout << "Rick fins PUNT15 (Q commit) " << time - time0 << " " << time - time_prev << std::endl;
    time_prev = time;

    offloadCommit(setupCtx.starkInfo.nStages + 1, starks.treesGL, d_buffers->d_aux_trace, (uint64_t *)d_trees[setupCtx.starkInfo.nStages].nodes, proof, setupCtx);

    TimerStopAndLog(STARK_COMMIT_QUOTIENT_POLYNOMIAL);
    starks.addTranscript(transcript, &proof.proof.roots[setupCtx.starkInfo.nStages][0], nFieldElements);
    TimerStopAndLog(STARK_STEP_Q);

    TimerStart(STARK_STEP_EVALS);

    uint64_t xiChallengeIndex = 0;
    for (uint64_t i = 0; i < setupCtx.starkInfo.challengesMap.size(); i++)
    {
        if (setupCtx.starkInfo.challengesMap[i].stage == setupCtx.starkInfo.nStages + 2)
        {
            if (setupCtx.starkInfo.challengesMap[i].stageId == 0)
                xiChallengeIndex = i;
            starks.getChallenge(transcript, challenges[i * FIELD_EXTENSION]);
        }
    }

    Goldilocks::Element *xiChallenge = &challenges[xiChallengeIndex * FIELD_EXTENSION];
    uint64_t LEv_offset = setupCtx.starkInfo.mapOffsets[std::make_pair("LEv", true)];

    CHECKCUDAERR(cudaDeviceSynchronize());
    time = omp_get_wtime();
    std::cout << "Rick fins PUNT16 (Q offload) " << time - time0 << " " << time - time_prev << std::endl;
    time_prev = time;

    computeLEv_inplace(xiChallenge, setupCtx.starkInfo.mapOffsets[make_pair("LEv", true)], setupCtx.starkInfo.starkStruct.nBits, setupCtx.starkInfo.openingPoints.size(), setupCtx.starkInfo.openingPoints.data(), d_buffers);

    CHECKCUDAERR(cudaDeviceSynchronize());
    time = omp_get_wtime();
    std::cout << "Rick fins PUNT17 (LEv) " << time - time0 << " " << time - time_prev << std::endl;
    time_prev = time;

    evmap_inplace(evals, d_params, LEv_offset, proof, &starks, d_buffers);

    CHECKCUDAERR(cudaDeviceSynchronize());
    time = omp_get_wtime();
    std::cout << "Rick fins PUNT18 (Evmap) " << time - time0 << " " << time - time_prev << std::endl;
    time_prev = time;

    if (!setupCtx.starkInfo.starkStruct.hashCommits)
    {
        starks.addTranscriptGL(transcript, evals, setupCtx.starkInfo.evMap.size() * FIELD_EXTENSION);
    }
    else
    {
        ElementType hash[nFieldElements];
        starks.calculateHash(hash, evals, setupCtx.starkInfo.evMap.size() * FIELD_EXTENSION);
        starks.addTranscript(transcript, hash, nFieldElements);
    }

    // Challenges for FRI polynomial
    for (uint64_t i = 0; i < setupCtx.starkInfo.challengesMap.size(); i++)
    {
        if (setupCtx.starkInfo.challengesMap[i].stage == setupCtx.starkInfo.nStages + 3)
        {
            starks.getChallenge(transcript, challenges[i * FIELD_EXTENSION]);
        }
    }

    TimerStopAndLog(STARK_STEP_EVALS);

    //--------------------------------
    // 6. Compute FRI
    //--------------------------------
    TimerStart(STARK_STEP_FRI);

    TimerStart(COMPUTE_FRI_POLYNOMIAL);
    uint64_t xDivXSub_offset = setupCtx.starkInfo.mapOffsets[std::make_pair("xDivXSubXi", true)];
    d_params.xDivXSub = (Goldilocks::Element *)(d_buffers->d_aux_trace + xDivXSub_offset);

    CHECKCUDAERR(cudaDeviceSynchronize());
    time = omp_get_wtime();
    std::cout << "Rick fins PUNT19 (transition) " << time - time0 << " " << time - time_prev << std::endl;
    time_prev = time;

    calculateXDivXSub_inplace(xDivXSub_offset, xiChallenge, setupCtx, d_buffers);

    CHECKCUDAERR(cudaDeviceSynchronize());
    time = omp_get_wtime();
    std::cout << "Rick fins PUNT20 (xDivxSub) " << time - time0 << " " << time - time_prev << std::endl;
    time_prev = time;

    // FRI expressions
    calculateExpression(setupCtx, expressionsCtx, params, d_params, (Goldilocks::Element *)(d_params.aux_trace + setupCtx.starkInfo.mapOffsets[std::make_pair("f", true)]), setupCtx.starkInfo.friExpId);

    CHECKCUDAERR(cudaDeviceSynchronize());
    time = omp_get_wtime();
    std::cout << "Rick fins PUNT22 (expressions FRI) " << time - time0 << " " << time - time_prev << std::endl;
    time_prev = time;

    TimerStopAndLog(COMPUTE_FRI_POLYNOMIAL);
    Goldilocks::Element challenge[FIELD_EXTENSION];
    uint64_t friPol_offset = setupCtx.starkInfo.mapOffsets[std::make_pair("f", true)];
    gl64_t *d_friPol = (gl64_t *)(d_buffers->d_aux_trace + friPol_offset);

    TimerStart(STARK_FRI_FOLDING);
    uint64_t nBitsExt = setupCtx.starkInfo.starkStruct.steps[0].nBits;
    Goldilocks::Element *foldedFRIPol = new Goldilocks::Element[(1 << setupCtx.starkInfo.starkStruct.steps[setupCtx.starkInfo.starkStruct.steps.size() - 1].nBits) * FIELD_EXTENSION];
    for (uint64_t step = 0; step < setupCtx.starkInfo.starkStruct.steps.size(); step++)
    {
        uint64_t currentBits = setupCtx.starkInfo.starkStruct.steps[step].nBits;
        uint64_t prevBits = step == 0 ? currentBits : setupCtx.starkInfo.starkStruct.steps[step - 1].nBits;
        fold_inplace(step, friPol_offset, challenge, nBitsExt, prevBits, currentBits, d_buffers);

        if (step < setupCtx.starkInfo.starkStruct.steps.size() - 1)
        {
            merkelizeFRI_inplace(setupCtx, params, d_params, step, proof, d_friPol, starks.treesFRI[step], currentBits, setupCtx.starkInfo.starkStruct.steps[step + 1].nBits, true);
            starks.addTranscript(transcript, &proof.proof.fri.treesFRI[step].root[0], nFieldElements);
        }
        else
        {
            CHECKCUDAERR(cudaMemcpy(foldedFRIPol, d_friPol, (1 << setupCtx.starkInfo.starkStruct.steps[step].nBits) * FIELD_EXTENSION * sizeof(Goldilocks::Element), cudaMemcpyDeviceToHost));
            if (!setupCtx.starkInfo.starkStruct.hashCommits)
            {
                starks.addTranscriptGL(transcript, foldedFRIPol, (1 << setupCtx.starkInfo.starkStruct.steps[step].nBits) * FIELD_EXTENSION);
            }
            else
            {
                ElementType hash[nFieldElements];
                starks.calculateHash(hash, foldedFRIPol, (1 << setupCtx.starkInfo.starkStruct.steps[step].nBits) * FIELD_EXTENSION);
                starks.addTranscript(transcript, hash, nFieldElements);
            }
        }
        starks.getChallenge(transcript, *challenge);
    }
    TimerStopAndLog(STARK_FRI_FOLDING);

    TimerStart(STARK_FRI_QUERIES);
    uint64_t friQueries[setupCtx.starkInfo.starkStruct.nQueries];
    TranscriptType transcriptPermutation(setupCtx.starkInfo.starkStruct.merkleTreeArity, setupCtx.starkInfo.starkStruct.merkleTreeCustom);
    starks.addTranscriptGL(transcriptPermutation, challenge, FIELD_EXTENSION);
    transcriptPermutation.getPermutations(friQueries, setupCtx.starkInfo.starkStruct.nQueries, setupCtx.starkInfo.starkStruct.steps[0].nBits);

    uint64_t nTrees = setupCtx.starkInfo.nStages + setupCtx.starkInfo.customCommits.size() + 2;
    proveQueries_inplace(setupCtx, friQueries, setupCtx.starkInfo.starkStruct.nQueries, proof, starks.treesGL, d_trees, nTrees, d_buffers, setupCtx.starkInfo.nStages, d_params);

    /*proveFRIQueries_inplace(friQueries, setupCtx.starkInfo.starkStruct.nQueries, setupCtx, proof, starks.treesFRI, d_buffers);*/ // Not run in the GPU at this point

    for (uint64_t step = 1; step < setupCtx.starkInfo.starkStruct.steps.size(); ++step)
    {
        FRI<Goldilocks::Element>::proveFRIQueries(friQueries, setupCtx.starkInfo.starkStruct.nQueries, step, setupCtx.starkInfo.starkStruct.steps[step].nBits, proof, starks.treesFRI[step - 1]);
    }

    FRI<ElementType>::setFinalPol(proof, foldedFRIPol, setupCtx.starkInfo.starkStruct.steps[setupCtx.starkInfo.starkStruct.steps.size() - 1].nBits);
    TimerStopAndLog(STARK_FRI_QUERIES);

    TimerStopAndLog(STARK_STEP_FRI);

    CHECKCUDAERR(cudaDeviceSynchronize());
    time = omp_get_wtime();
    std::cout << "Rick fins PUNT23 (FRI) " << time - time0 << " " << time - time_prev << std::endl;
    time_prev = time;

    delete challenges;
    delete evals;
    delete airgroupValues;
    delete foldedFRIPol;
    TimerStopAndLog(STARK_PROOF);

    proof.proof.proof2pointer(proofBuffer);
    if(!proofFile.empty()) {
        json2file(pointer2json(proofBuffer, setupCtx.starkInfo), proofFile);
    }

    cudaFree(d_evals);
    for (uint64_t i = 0; i < setupCtx.starkInfo.nStages + 1; i++)
    {
       cudaFree(d_trees[i].nodes);
    }
#endif
}
#endif
