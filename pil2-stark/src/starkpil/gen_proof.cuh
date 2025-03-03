#ifndef GEN_PROOF_CUH
#define GEN_PROOF_CUH

#include "starks.hpp"
#include "hints.hpp"
#include "cuda_utils.cuh"
#include "gl64_t.cuh"
#include "expressions_gpu.cuh"

void offloadCommit_(uint64_t step, MerkleTreeGL **treesGL, gl64_t *d_aux_trace, uint64_t *d_tree, FRIProof<Goldilocks::Element> &proof, SetupCtx &setupCtx, StepsParams& params)
{

    CHECKCUDAERR(cudaDeviceSynchronize());
    double time = omp_get_wtime();

    uint64_t N = 1 << setupCtx.starkInfo.starkStruct.nBits;
    uint64_t NExtended = 1 << setupCtx.starkInfo.starkStruct.nBitsExt;
   
    std::string section = "cm" + to_string(step);
    uint64_t nCols = setupCtx.starkInfo.mapSectionsN["cm" + to_string(step)];

    Goldilocks::Element *pBuff = step == 1 ? params.trace : &params.aux_trace[setupCtx.starkInfo.mapOffsets[make_pair(section, false)]];
    Goldilocks::Element *pBuffExtended = &params.aux_trace[setupCtx.starkInfo.mapOffsets[make_pair(section, true)]];

    treesGL[step - 1]->setSource(pBuffExtended);
   
    Goldilocks::Element *pBuffNodesGL = &params.aux_trace[setupCtx.starkInfo.mapOffsets[make_pair("mt" + to_string(step), true)]];
    treesGL[step - 1]->setNodes(pBuffNodesGL);

    uint64_t tree_size = treesGL[step - 1]->getNumNodes(NExtended);
    uint32_t nFielsElements = treesGL[step - 1]->getMerkleTreeNFieldElements();

    CHECKCUDAERR(cudaMemcpy(pBuffNodesGL, &d_tree[0], tree_size * sizeof(uint64_t), cudaMemcpyDeviceToHost));
    CHECKCUDAERR(cudaMemcpy(&proof.proof.roots[step - 1][0], &d_tree[tree_size - nFielsElements], nFielsElements * sizeof(uint64_t), cudaMemcpyDeviceToHost));

    /*Goldilocks::Element root[4];
    Goldilocks::Element nodes[4];
    CHECKCUDAERR(cudaMemcpy(&root, &d_tree[tree_size - nFielsElements], 4 * sizeof(uint64_t), cudaMemcpyDeviceToHost));
    CHECKCUDAERR(cudaMemcpy(&nodes, &d_tree[0], 4 * sizeof(uint64_t), cudaMemcpyDeviceToHost));
    time = omp_get_wtime() - time;
    std::cout << "nodes[0] " <<nodes[0].fe << std::endl;
    std::cout << "nodes[1] " <<nodes[1].fe << std::endl;
    std::cout << "nodes[2] " <<nodes[2].fe << std::endl;
    std::cout << "nodes[3] " <<nodes[3].fe << std::endl;
    std::cout <<" root gpu " << root[0].fe << std::endl;
    std::cout <<" root gpu " << root[1].fe << std::endl;
    std::cout <<" root gpu " << root[2].fe << std::endl;
    std::cout <<" root gpu " << root[3].fe << std::endl;*/
    time = omp_get_wtime() - time;
    std::cout << "offload time " << time << std::endl;
}

void calculateWitnessSTD_gpu(SetupCtx& setupCtx, StepsParams& params, Goldilocks::Element *pBuffHelper, bool prod) {
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

        multiplyHintFields(setupCtx, params, nImTotalHints, imHints, hintFieldDest, hintField1, hintField2, hintOptions1, hintOptions2);
        
    }

    HintFieldOptions options1;
    HintFieldOptions options2;
    options2.inverse = true;
    accMulHintFields(setupCtx, params, pBuffHelper, hint[0], "reference", "result", "numerator_air", "denominator_air",options1, options2, !prod);
    updateAirgroupValue(setupCtx, params, hint[0], "result", "numerator_direct", "denominator_direct", options1, options2, !prod);
}

void genProof_gpu(SetupCtx& setupCtx, uint64_t airgroupId, uint64_t airId, uint64_t instanceId, StepsParams& params, Goldilocks::Element *globalChallenge, Goldilocks::Element* pBuffHelper, uint64_t *proofBuffer, std::string proofFile, DeviceCommitBuffers *d_buffers) {
    
    TimerStart(STARK_PROOF);
    CHECKCUDAERR(cudaDeviceSynchronize());
    double time0 = omp_get_wtime();
    double time_prev = time0;

    CHECKCUDAERR(cudaMemset(d_buffers->d_aux_trace, 0, setupCtx.starkInfo.mapTotalN * sizeof(Goldilocks::Element)));
    uint64_t N = 1 << setupCtx.starkInfo.starkStruct.nBits;
    uint64_t NExtended = 1 << setupCtx.starkInfo.starkStruct.nBitsExt;


    FRIProof<Goldilocks::Element> proof(setupCtx.starkInfo, airgroupId, airId, instanceId);
    
    Starks<Goldilocks::Element> starks(setupCtx, params.pConstPolsExtendedTreeAddress, params.pCustomCommitsFixed, false);

    // GPU tree-nodes
    GPUTree *d_trees = new GPUTree[setupCtx.starkInfo.nStages + 2];
    /*for (uint64_t i = 0; i < setupCtx.starkInfo.nStages + 1; i++)
    {
        std::string section = "cm" + to_string(i + 1);
        uint64_t nCols = setupCtx.starkInfo.mapSectionsN[section];
        d_trees[i].nFieldElements = 4;
        // uint64_t numNodes = NExtended * d_trees[i].nFieldElements + (NExtended - 1) * d_trees[i].nFieldElements;
        // CHECKCUDAERR(cudaMalloc(&d_trees[i].nodes, numNodes * sizeof(gl64_t)));
    }*/
    
#ifdef __AVX512__
    ExpressionsAvx512 expressionsCtx(setupCtx);
#elif defined(__AVX2__)
    ExpressionsAvx expressionsCtx(setupCtx);
#else
    ExpressionsPack expressionsCtx(setupCtx);
#endif

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
    //CHECKCUDAERR(cudaMemcpy(params.trace, d_buffers->d_trace, N * setupCtx.starkInfo.mapSectionsN["cm1"] * sizeof(Goldilocks::Element), cudaMemcpyDeviceToHost));
    //rick// starks.commitStage(1, params.trace, params.aux_trace, proof, pBuffHelper);
    starks.commitStage_inplace(1, d_buffers->d_trace, d_buffers->d_aux_trace, (uint64_t **)(&d_trees[0].nodes), d_buffers);
    /*gl64_t *dst = d_buffers->d_aux_trace;
    uint64_t offset_dst = setupCtx.starkInfo.mapOffsets[make_pair("cm1", true)];
    uint64_t ncols = setupCtx.starkInfo.mapSectionsN["cm1"];
    std::cout << "Columnes constants " << ncols << std::endl;
    // compare the CPU and gpu results
    Goldilocks::Element *gpu_copy = new Goldilocks::Element[NExtended * ncols];
    CHECKCUDAERR(cudaMemcpy(gpu_copy, dst + offset_dst, NExtended * ncols * sizeof(Goldilocks::Element), cudaMemcpyDeviceToHost));
    Goldilocks::Element *cpu_copy = params.aux_trace + offset_dst;
    for (uint64_t i = 0; i < NExtended * ncols; i++)
    {
        if (cpu_copy[i].fe != gpu_copy[i].fe)
        {
            std::cout << "Error at " << i << " " << cpu_copy[i].fe << " " << gpu_copy[i].fe << std::endl;
            exit(0);
        }
    }*/
    //free(gpu_copy);
    //offload all the trace
    CHECKCUDAERR(cudaMemcpy(params.aux_trace, d_buffers->d_aux_trace, setupCtx.starkInfo.mapTotalN * sizeof(Goldilocks::Element), cudaMemcpyDeviceToHost));
    offloadCommit_(1, starks.treesGL, d_buffers->d_aux_trace, (uint64_t *)d_trees[0].nodes, proof, setupCtx, params);

    /*std::cout << "nodes[0] " <<starks.treesGL[0]->nodes[0].fe << std::endl;
    std::cout << "nodes[1] " <<starks.treesGL[0]->nodes[1].fe << std::endl;
    std::cout << "nodes[2] " <<starks.treesGL[0]->nodes[2].fe << std::endl;
    std::cout << "nodes[3] " <<starks.treesGL[0]->nodes[3].fe << std::endl;
    std::cout << "root[0] " << proof.proof.roots[0][0].fe << std::endl;
    std::cout << "root[1] " << proof.proof.roots[0][1].fe << std::endl;
    std::cout << "root[2] " << proof.proof.roots[0][2].fe << std::endl;
    std::cout << "root[3] " << proof.proof.roots[0][3].fe << std::endl;*/
    TimerStopAndLog(STARK_STEP_1);

    starks.addTranscript(transcript, globalChallenge, FIELD_EXTENSION);

    TimerStart(STARK_STEP_2);
    for (uint64_t i = 0; i < setupCtx.starkInfo.challengesMap.size(); i++) {
        if(setupCtx.starkInfo.challengesMap[i].stage == 2) {
            starks.getChallenge(transcript, params.challenges[i * FIELD_EXTENSION]);
        }
    }

    calculateWitnessSTD_gpu(setupCtx, params, pBuffHelper, true);
    calculateWitnessSTD_gpu(setupCtx, params, pBuffHelper, false);

    
    TimerStart(CALCULATE_IM_POLS);
    starks.calculateImPolsExpressions(2, params);
    TimerStopAndLog(CALCULATE_IM_POLS);

    

    
    TimerStart(STARK_COMMIT_STAGE_2);
    CHECKCUDAERR(cudaMemcpy(d_buffers->d_aux_trace, params.aux_trace, setupCtx.starkInfo.mapTotalN * sizeof(Goldilocks::Element), cudaMemcpyHostToDevice));
    starks.commitStage_inplace(2, d_buffers->d_trace, d_buffers->d_aux_trace, (uint64_t **)(&d_trees[1].nodes), d_buffers);
    CHECKCUDAERR(cudaMemcpy(params.aux_trace, d_buffers->d_aux_trace, setupCtx.starkInfo.mapTotalN * sizeof(Goldilocks::Element), cudaMemcpyDeviceToHost)); 
    offloadCommit_(2, starks.treesGL, d_buffers->d_aux_trace, (uint64_t *)d_trees[1].nodes, proof, setupCtx, params);
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

    // TODO: ADD PROOF VALUES ???

   
    TimerStopAndLog(STARK_STEP_2);

    TimerStart(STARK_STEP_Q);

    for (uint64_t i = 0; i < setupCtx.starkInfo.challengesMap.size(); i++)
    {
        if(setupCtx.starkInfo.challengesMap[i].stage == setupCtx.starkInfo.nStages + 1) {
            starks.getChallenge(transcript, params.challenges[i * FIELD_EXTENSION]);
        }
    }
    
    expressionsCtx.calculateExpression(params, &params.aux_trace[setupCtx.starkInfo.mapOffsets[std::make_pair("q", true)]], setupCtx.starkInfo.cExpId);

    TimerStart(STARK_COMMIT_QUOTIENT_POLYNOMIAL);
    CHECKCUDAERR(cudaMemcpy(d_buffers->d_aux_trace, params.aux_trace, setupCtx.starkInfo.mapTotalN * sizeof(Goldilocks::Element), cudaMemcpyHostToDevice));
    starks.commitStage_inplace(setupCtx.starkInfo.nStages + 1, d_buffers->d_trace, d_buffers->d_aux_trace, (uint64_t **)(&d_trees[setupCtx.starkInfo.nStages].nodes), d_buffers);
    CHECKCUDAERR(cudaMemcpy(params.aux_trace, d_buffers->d_aux_trace, setupCtx.starkInfo.mapTotalN * sizeof(Goldilocks::Element), cudaMemcpyDeviceToHost)); 
    offloadCommit_(setupCtx.starkInfo.nStages + 1, starks.treesGL, d_buffers->d_aux_trace, (uint64_t *)d_trees[setupCtx.starkInfo.nStages].nodes, proof, setupCtx, params);
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
    Goldilocks::Element* LEv = &pBuffHelper[0];

    starks.computeLEv(xiChallenge, LEv);
    starks.computeEvals(params ,LEv, proof);

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
    starks.calculateXDivXSub(xiChallenge, params.xDivXSub);
    starks.calculateFRIPolynomial(params);
    TimerStopAndLog(COMPUTE_FRI_POLYNOMIAL);

    Goldilocks::Element challenge[FIELD_EXTENSION];
    Goldilocks::Element *friPol = &params.aux_trace[setupCtx.starkInfo.mapOffsets[std::make_pair("f", true)]];
    
    TimerStart(STARK_FRI_FOLDING);
    uint64_t nBitsExt =  setupCtx.starkInfo.starkStruct.steps[0].nBits;
    for (uint64_t step = 0; step < setupCtx.starkInfo.starkStruct.steps.size(); step++)
    {   
        uint64_t currentBits = setupCtx.starkInfo.starkStruct.steps[step].nBits;
        uint64_t prevBits = step == 0 ? currentBits : setupCtx.starkInfo.starkStruct.steps[step - 1].nBits;
        FRI<Goldilocks::Element>::fold(step, friPol, challenge, nBitsExt, prevBits, currentBits);
        if (step < setupCtx.starkInfo.starkStruct.steps.size() - 1)
        {
            FRI<Goldilocks::Element>::merkelize(step, proof, friPol, starks.treesFRI[step], currentBits, setupCtx.starkInfo.starkStruct.steps[step + 1].nBits);
            starks.addTranscript(transcript, &proof.proof.fri.treesFRI[step].root[0], HASH_SIZE);
        }
        else
        {
            if(!setupCtx.starkInfo.starkStruct.hashCommits) {
                starks.addTranscriptGL(transcript, friPol, (1 << setupCtx.starkInfo.starkStruct.steps[step].nBits) * FIELD_EXTENSION);
            } else {
                Goldilocks::Element hash[HASH_SIZE];
                starks.calculateHash(hash, friPol, (1 << setupCtx.starkInfo.starkStruct.steps[step].nBits) * FIELD_EXTENSION);
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
    FRI<Goldilocks::Element>::proveQueries(friQueries, setupCtx.starkInfo.starkStruct.nQueries, proof, starks.treesGL, nTrees);

    for(uint64_t step = 1; step < setupCtx.starkInfo.starkStruct.steps.size(); ++step) {

        FRI<Goldilocks::Element>::proveFRIQueries(friQueries, setupCtx.starkInfo.starkStruct.nQueries, step, setupCtx.starkInfo.starkStruct.steps[step].nBits, proof, starks.treesFRI[step - 1]);
    }

    FRI<Goldilocks::Element>::setFinalPol(proof, friPol, setupCtx.starkInfo.starkStruct.steps[setupCtx.starkInfo.starkStruct.steps.size() - 1].nBits);
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
}

#endif