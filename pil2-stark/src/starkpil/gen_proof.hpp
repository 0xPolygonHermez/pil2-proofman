#include "starks.hpp"

void calculateWitnessSTD(SetupCtx& setupCtx, StepsParams& params, Goldilocks::Element *pBuffHelper, bool prod) {
    std::string name = prod ? "gprod_col" : "gsum_col";
    if(setupCtx.expressionsBin.getNumberHintIdsByName(name) == 0) return;
    uint64_t hint[1];
    setupCtx.expressionsBin.getHintIdsByName(hint, name);

    uint64_t nImHints = setupCtx.expressionsBin.getNumberHintIdsByName("im_col");
    if(nImHints > 0) {
        uint64_t imHints[nImHints];
        setupCtx.expressionsBin.getHintIdsByName(imHints, "im_col");
        for(uint64_t i = 0; i < nImHints; i++) {
            HintFieldOptions options1;
            HintFieldOptions options2;
            options2.inverse = true;
            multiplyHintFields(setupCtx, params, imHints[i], "reference", "numerator", "denominator", options1, options2);
        }
    }

    HintFieldOptions options1;
    HintFieldOptions options2;
    options2.inverse = true;
    accMulHintFields(setupCtx, params, pBuffHelper, hint[0], "reference", "result", "numerator_air", "denominator_air",options1, options2, !prod);
    updateAirgroupValue(setupCtx, params, hint[0], "result", "numerator_direct", "denominator_direct", options1, options2, !prod);
}

void *genProof(SetupCtx& setupCtx, uint64_t airgroupId, uint64_t airId, uint64_t instanceId, StepsParams& params, Goldilocks::Element *globalChallenge, Goldilocks::Element* pBuffHelper, std::string proofFile) {
    TimerStart(STARK_PROOF);

    FRIProof<Goldilocks::Element> proof(setupCtx.starkInfo, airgroupId, airId, instanceId);
    
    Starks<Goldilocks::Element> starks(setupCtx, params.pConstPolsExtendedTreeAddress, params.pCustomCommitsFixed);
    
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
    starks.commitStage(1, params.trace, params.aux_trace, proof, pBuffHelper);
    TimerStopAndLog(STARK_STEP_1);

    starks.addTranscript(transcript, globalChallenge, FIELD_EXTENSION);

    TimerStart(STARK_STEP_2);
    for (uint64_t i = 0; i < setupCtx.starkInfo.challengesMap.size(); i++) {
        if(setupCtx.starkInfo.challengesMap[i].stage == 2) {
            starks.getChallenge(transcript, params.challenges[i * FIELD_EXTENSION]);
        }
    }

    calculateWitnessSTD(setupCtx, params, pBuffHelper, true);
    calculateWitnessSTD(setupCtx, params, pBuffHelper, false);

    
    TimerStart(CALCULATE_IM_POLS);
    starks.calculateImPolsExpressions(2, params);
    TimerStopAndLog(CALCULATE_IM_POLS);

    TimerStart(STARK_COMMIT_STAGE_2);
    starks.commitStage(2, nullptr, params.aux_trace, proof, pBuffHelper);
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

    proof.proof.setAirgroupValues(params.airgroupValues);
    proof.proof.setAirValues(params.airValues);
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
    starks.commitStage(setupCtx.starkInfo.nStages + 1, nullptr, params.aux_trace, proof);
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

    nlohmann::json zkin = proof.proof.proof2json();

    for (uint64_t i = 0; i < setupCtx.starkInfo.nPublics; i++)
    {
        zkin["publics"][i] = Goldilocks::toString(params.publicInputs[i]);
    }

    uint64_t p = 0;
    for (uint64_t i = 0; i < setupCtx.starkInfo.proofValuesMap.size(); i++)
    {
        if(setupCtx.starkInfo.proofValuesMap[i].stage == 1) {
            zkin["proofvalues"][i][0] = Goldilocks::toString(params.proofValues[p++]);
            zkin["proofvalues"][i][1] = "0";
            zkin["proofvalues"][i][2] = "0";
        } else {
            zkin["proofvalues"][i][0] = Goldilocks::toString(params.proofValues[p++]);
            zkin["proofvalues"][i][1] = Goldilocks::toString(params.proofValues[p++]);
            zkin["proofvalues"][i][2] = Goldilocks::toString(params.proofValues[p++]);
        }
    }

    for(uint64_t k = 0; k < FIELD_EXTENSION; ++k) {
        zkin["globalChallenge"][k] = Goldilocks::toString(globalChallenge[k]);
    }

    TimerStopAndLog(STARK_PROOF);

    if(!proofFile.empty()) {
        json2file(zkin, proofFile);
    }
    
    return (void *) new nlohmann::json(zkin);
}
