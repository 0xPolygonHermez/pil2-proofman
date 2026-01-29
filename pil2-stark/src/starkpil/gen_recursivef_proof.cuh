#include "bn128.cuh"
#include "starks.hpp"
#include "transcriptBN128.cuh"
#include "starks_gpu_bn128.cuh"

void calculateWitnessSTD_BN128_gpu(SetupCtx& setupCtx, StepsParams& params, ExpressionsCtx &expressionsCtx, bool prod) {
    std::string name = prod ? "gprod_col" : "gsum_col";
    if(setupCtx.expressionsBin.getNumberHintIdsByName(name) == 0) return;
    uint64_t hint[1];
    setupCtx.expressionsBin.getHintIdsByName(hint, name);

    uint64_t nImHints = setupCtx.expressionsBin.getNumberHintIdsByName("im_col");
    uint64_t nImHintsAirVals = setupCtx.expressionsBin.getNumberHintIdsByName("im_airval");
    uint64_t nImTotalHints = nImHints + nImHintsAirVals;
    if(nImTotalHints > 0) {
        std::vector<uint64_t> imHints(nImHints + nImHintsAirVals);
        setupCtx.expressionsBin.getHintIdsByName(imHints.data(), "im_col");
        setupCtx.expressionsBin.getHintIdsByName(&imHints[nImHints], "im_airval");
        std::vector<std::string> hintFieldDest(nImTotalHints);
        std::vector<std::string> hintField1(nImTotalHints);
        std::vector<std::string> hintField2(nImTotalHints);
        std::vector<HintFieldOptions> hintOptions1(nImTotalHints);
        std::vector<HintFieldOptions> hintOptions2(nImTotalHints);
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

        multiplyHintFields(setupCtx, params, expressionsCtx, nImTotalHints, imHints.data(), hintFieldDest.data(), hintField1.data(), hintField2.data(), hintOptions1.data(), hintOptions2.data());
        
    }

    HintFieldOptions options1;
    HintFieldOptions options2;
    options2.inverse = true;

    std::string hintFieldNameAirgroupVal = setupCtx.starkInfo.airgroupValuesMap.size() > 0 ? "result" : "";

    accMulHintFields(setupCtx, params, expressionsCtx, hint[0], "reference", hintFieldNameAirgroupVal, "numerator_air", "denominator_air", options1, options2, !prod);
    updateAirgroupValue(setupCtx, params, hint[0], hintFieldNameAirgroupVal, "numerator_direct", "denominator_direct", options1, options2, !prod);
}

void *genRecursiveProofBN128_gpu(SetupCtx& setupCtx, uint64_t airgroupId, uint64_t airId, uint64_t instanceId, Goldilocks::Element *d_witness, Goldilocks::Element *d_aux_trace, Goldilocks::Element *d_constPols, Goldilocks::Element *d_constTree, Goldilocks::Element *d_publicInputs, Goldilocks::Element *proofBuffer, std::string proofFile, cudaStream_t stream) {
    cudaStreamSynchronize(stream);
    TimerStart(STARK_PROOF);
    TimerGPU timer;
    Starks<RawFr::Element> starks(setupCtx, d_constTree, nullptr, false, false);
    uint64_t nFieldElements = 1;
    
    TranscriptBN128_GPU d_transcript(setupCtx.starkInfo.starkStruct.merkleTreeArity, setupCtx.starkInfo.starkStruct.merkleTreeCustom, stream);
    TranscriptBN128_GPU d_transcript_helper(setupCtx.starkInfo.starkStruct.merkleTreeArity, setupCtx.starkInfo.starkStruct.merkleTreeCustom, stream);

    uint64_t hashOffset = 0;
    PoseidonBN128GPU::FrElement * d_hash_gpu = (PoseidonBN128GPU::FrElement *)&proofBuffer[hashOffset];
    uint64_t challengeOffset = hashOffset + nFieldElements*4;
    Goldilocks::Element* d_challenges = &proofBuffer[challengeOffset];


    TimerStart(STARK_STEP_0);
    d_transcript.reset(stream);
    
    d_transcript.put(((PoseidonBN128GPU::FrElement *) starks.treesGL[setupCtx.starkInfo.nStages + 1]->get_nodes_ptr()) + starks.treesGL[setupCtx.starkInfo.nStages + 1]->numNodes - 1, 1, stream);
    if (setupCtx.starkInfo.nPublics > 0)
    {
        if (!setupCtx.starkInfo.starkStruct.hashCommits)
        {
            d_transcript.put(d_publicInputs, setupCtx.starkInfo.nPublics, stream);
        }
        else
        {
            calculateHashBN128_gpu(&d_transcript_helper, d_hash_gpu, setupCtx, d_publicInputs, setupCtx.starkInfo.nPublics, stream);
            d_transcript.put(d_hash_gpu, nFieldElements, stream);
        }
    }

    TimerStopAndLog(STARK_STEP_0);
    
    TimerStartGPU(timer, STARK_STEP_1);
    for (uint64_t i = 0; i < setupCtx.starkInfo.challengesMap.size(); i++)
    {
        if (setupCtx.starkInfo.challengesMap[i].stage == 1)
        {
            d_transcript.getField((uint64_t *)&d_challenges[i * FIELD_EXTENSION], stream);
        }
    }
    TimerStopGPU(timer, STARK_STEP_1);

    TimerStartGPU(timer, STARK_COMMIT_STAGE_1);
    commitStage_bn128_gpu(1, setupCtx, starks.treesGL, d_witness, d_aux_trace, &d_transcript, timer, stream);     
    TimerStopGPU(timer, STARK_COMMIT_STAGE_1);
    
    // Debug: print Stage 1 root
    {
        cudaStreamSynchronize(stream);
        uint64_t NExtended = 1 << setupCtx.starkInfo.starkStruct.nBitsExt;
        uint64_t tree_size = starks.treesGL[0]->getNumNodes(NExtended);
        PoseidonBN128GPU::FrElement h_root;
        PoseidonBN128GPU::FrElement* d_root_ptr = ((PoseidonBN128GPU::FrElement *) starks.treesGL[0]->get_nodes_ptr()) + tree_size - 1;
        cudaMemcpy(&h_root, d_root_ptr, sizeof(PoseidonBN128GPU::FrElement), cudaMemcpyDeviceToHost);
        std::cout << "=== GPU Stage 1 Root ===" << std::endl;
        std::cout << "Root: ";
        uint64_t* limbs = (uint64_t*)&h_root;
        for (int i = 0; i < 4; i++) {
            std::cout << limbs[i];
            if (i < 3) std::cout << ", ";
        }
        std::cout << std::endl;
        std::cout << "========================" << std::endl;
    }

    // NOTE: Don't call getState() here for debugging - it modifies transcript state by flushing pending elements!

    TimerStartGPU(timer, STARK_STEP_2);
    for (uint64_t i = 0; i < setupCtx.starkInfo.challengesMap.size(); i++)
    {
        if (setupCtx.starkInfo.challengesMap[i].stage == 2)
        {
            d_transcript.getField((uint64_t *)&d_challenges[i * FIELD_EXTENSION], stream);
        }
    }
    TimerStopGPU(timer, STARK_STEP_2);
    cudaStreamSynchronize(stream);
    
    // Offload and print all the challenges for debugging
    {
        uint64_t numChallenges = setupCtx.starkInfo.challengesMap.size();
        std::vector<Goldilocks::Element> h_challenges(numChallenges * FIELD_EXTENSION);
        cudaMemcpy(h_challenges.data(), d_challenges, numChallenges * FIELD_EXTENSION * sizeof(Goldilocks::Element), cudaMemcpyDeviceToHost);
        
        std::cout << "=== GPU Challenges (Stage 2) ===" << std::endl;
        for (uint64_t i = 0; i < numChallenges; i++) {
            if (setupCtx.starkInfo.challengesMap[i].stage == 2) {
                std::cout << "Challenge[" << i << "]: ";
                for (uint64_t j = 0; j < FIELD_EXTENSION; j++) {
                    std::cout << Goldilocks::toU64(h_challenges[i * FIELD_EXTENSION + j]);
                    if (j < FIELD_EXTENSION - 1) std::cout << ", ";
                }
                std::cout << std::endl;
            }
        }
        std::cout << "================================" << std::endl;
    }



    cudaStreamSynchronize(stream);
    TimerStopAndLog(STARK_PROOF);





    // Free stark trees
    /*for (uint64_t i = 0; i < setupCtx.starkInfo.nStages + setupCtx.starkInfo.customCommits.size() + 2; i++)
    {
       cudaFree(starks.treesGL[i]->get_nodes_ptr());
    }*/
    return nullptr;
}
