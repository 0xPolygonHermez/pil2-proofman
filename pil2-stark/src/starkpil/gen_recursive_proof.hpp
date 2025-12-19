#include "starks.hpp"

void calculateWitnessSTD_BN128(SetupCtx& setupCtx, StepsParams& params, ExpressionsCtx &expressionsCtx, bool prod) {
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

        multiplyHintFields(setupCtx, params, expressionsCtx, nImTotalHints, imHints, hintFieldDest, hintField1, hintField2, hintOptions1, hintOptions2);
        
    }

    HintFieldOptions options1;
    HintFieldOptions options2;
    options2.inverse = true;

    std::string hintFieldNameAirgroupVal = setupCtx.starkInfo.airgroupValuesMap.size() > 0 ? "result" : "";

    accMulHintFields(setupCtx, params, expressionsCtx, hint[0], "reference", hintFieldNameAirgroupVal, "numerator_air", "denominator_air", options1, options2, !prod);
    updateAirgroupValue(setupCtx, params, hint[0], hintFieldNameAirgroupVal, "numerator_direct", "denominator_direct", options1, options2, !prod);
}

void *genRecursiveProofBN128(SetupCtx& setupCtx, json& globalInfo, uint64_t airgroupId, uint64_t airId, uint64_t instanceId, Goldilocks::Element *witness, Goldilocks::Element *aux_trace, Goldilocks::Element *pConstPols, Goldilocks::Element *pConstTree, Goldilocks::Element *publicInputs, uint64_t *proofBuffer, std::string proofFile) {
    TimerStart(STARK_PROOF);

    NTT_Goldilocks ntt(1 << setupCtx.starkInfo.starkStruct.nBits);
    NTT_Goldilocks nttExtended(1 << setupCtx.starkInfo.starkStruct.nBitsExt);

    ProverHelpers proverHelpers(setupCtx.starkInfo, true);

    FRIProof<RawFrP::Element> proof(setupCtx.starkInfo, airgroupId, airId, instanceId);
    
    Starks<RawFrP::Element> starks(setupCtx, pConstTree);
    
    ExpressionsPack expressionsCtx(setupCtx, &proverHelpers);

    uint64_t nFieldElements = 1;

    TranscriptBN128 transcript(setupCtx.starkInfo.starkStruct.merkleTreeArity, setupCtx.starkInfo.starkStruct.merkleTreeCustom);

    Goldilocks::Element evals[setupCtx.starkInfo.evMap.size() * FIELD_EXTENSION];
    Goldilocks::Element challenges[setupCtx.starkInfo.challengesMap.size() * FIELD_EXTENSION];
    
    StepsParams params = {
        .trace = witness,
        .aux_trace = aux_trace,
        .publicInputs = publicInputs,
        .proofValues = nullptr,
        .challenges = challenges,
        .airgroupValues = nullptr,
        .evals = evals,
        .xDivXSub = nullptr,
        .pConstPolsAddress = pConstPols,
        .pConstPolsExtendedTreeAddress = pConstTree,
        .pCustomCommitsFixed = nullptr,
    };

    //--------------------------------
    // 0.- Add const root and publics to transcript
    //--------------------------------

    TimerStart(STARK_STEP_0);
    RawFrP::Element verkey[nFieldElements];
    starks.treesGL[setupCtx.starkInfo.nStages + 1]->getRoot(verkey);
    starks.addTranscript(transcript, &verkey[0], nFieldElements);
    if(setupCtx.starkInfo.nPublics > 0) {
        if(!setupCtx.starkInfo.starkStruct.hashCommits) {
            starks.addTranscriptGL(transcript, &publicInputs[0], setupCtx.starkInfo.nPublics);
        } else {
            RawFrP::Element hash[nFieldElements];
            starks.calculateHash(hash, &publicInputs[0], setupCtx.starkInfo.nPublics);
            starks.addTranscript(transcript, hash, nFieldElements);
        }
    }

    TimerStopAndLog(STARK_STEP_0);

    TimerStart(STARK_STEP_1);
    for (uint64_t i = 0; i < setupCtx.starkInfo.challengesMap.size(); i++) {
        if(setupCtx.starkInfo.challengesMap[i].stage == 1) {
            starks.getChallenge(transcript, challenges[i * FIELD_EXTENSION]);
        }
    }
    TimerStart(STARK_COMMIT_STAGE_1);
    starks.commitStage(1, params.trace, params.aux_trace, proof, ntt);
    TimerStopAndLog(STARK_COMMIT_STAGE_1);
    starks.addTranscript(transcript, &proof.proof.roots[0][0], nFieldElements);
    
    TimerStopAndLog(STARK_STEP_1);

    TimerStart(STARK_STEP_2);
    for (uint64_t i = 0; i < setupCtx.starkInfo.challengesMap.size(); i++) {
        if(setupCtx.starkInfo.challengesMap[i].stage == 2) {
            starks.getChallenge(transcript, challenges[i * FIELD_EXTENSION]);
        }
    }

    TimerStart(STARK_CALCULATE_WITNESS_STD);
    calculateWitnessSTD_BN128(setupCtx, params, expressionsCtx, true);
    calculateWitnessSTD_BN128(setupCtx, params, expressionsCtx, false);
    TimerStopAndLog(STARK_CALCULATE_WITNESS_STD);
    
    TimerStart(CALCULATE_IM_POLS);
    starks.calculateImPolsExpressions(2, params, expressionsCtx);
    TimerStopAndLog(CALCULATE_IM_POLS);

    TimerStart(STARK_COMMIT_STAGE_2);
    starks.commitStage(2, nullptr, params.aux_trace, proof, ntt);
    TimerStopAndLog(STARK_COMMIT_STAGE_2);
    starks.addTranscript(transcript, &proof.proof.roots[1][0], nFieldElements);

    TimerStopAndLog(STARK_STEP_2);

    TimerStart(STARK_STEP_Q);

    for (uint64_t i = 0; i < setupCtx.starkInfo.challengesMap.size(); i++)
    {
        if(setupCtx.starkInfo.challengesMap[i].stage == setupCtx.starkInfo.nStages + 1) {
            starks.getChallenge(transcript, challenges[i * FIELD_EXTENSION]);
        }
    }
    
    TimerStart(STARK_CALCULATE_QUOTIENT_POLYNOMIAL);
    starks.calculateQuotientPolynomial(params, expressionsCtx);
    TimerStopAndLog(STARK_CALCULATE_QUOTIENT_POLYNOMIAL);
    
    TimerStart(STARK_COMMIT_QUOTIENT_POLYNOMIAL);
    starks.commitStage(setupCtx.starkInfo.nStages + 1, nullptr, params.aux_trace, proof, nttExtended);
    TimerStopAndLog(STARK_COMMIT_QUOTIENT_POLYNOMIAL);
    starks.addTranscript(transcript, &proof.proof.roots[setupCtx.starkInfo.nStages][0], nFieldElements);
    TimerStopAndLog(STARK_STEP_Q);

    TimerStart(STARK_STEP_EVALS);

    uint64_t xiChallengeIndex = 0;
    for (uint64_t i = 0; i < setupCtx.starkInfo.challengesMap.size(); i++)
    {
        if(setupCtx.starkInfo.challengesMap[i].stage == setupCtx.starkInfo.nStages + 2) {
            if(setupCtx.starkInfo.challengesMap[i].stageId == 0) xiChallengeIndex = i;
            starks.getChallenge(transcript, challenges[i * FIELD_EXTENSION]);
        }
    }

    Goldilocks::Element *xiChallenge = &challenges[xiChallengeIndex * FIELD_EXTENSION];
    Goldilocks::Element* LEv = &params.aux_trace[setupCtx.starkInfo.mapOffsets[make_pair("lev", false)]];

    for(uint64_t i = 0; i < setupCtx.starkInfo.openingPoints.size(); i+= 4) {
        std::vector<int64_t> openingPoints;
        for(uint64_t j = 0; j < 4; ++j) {
            if(i + j < setupCtx.starkInfo.openingPoints.size()) {
                openingPoints.push_back(setupCtx.starkInfo.openingPoints[i + j]);
            }
        }
        starks.computeLEv(xiChallenge, LEv, openingPoints, ntt);
        starks.computeEvals(params ,LEv, proof, openingPoints);
    }

    if(!setupCtx.starkInfo.starkStruct.hashCommits) {
        starks.addTranscriptGL(transcript, evals, setupCtx.starkInfo.evMap.size() * FIELD_EXTENSION);
    } else {
        RawFrP::Element hash[nFieldElements];
        starks.calculateHash(hash, evals, setupCtx.starkInfo.evMap.size() * FIELD_EXTENSION);
        starks.addTranscript(transcript, hash, nFieldElements);
    }

    // Challenges for FRI polynomial
    for (uint64_t i = 0; i < setupCtx.starkInfo.challengesMap.size(); i++)
    {
        if(setupCtx.starkInfo.challengesMap[i].stage == setupCtx.starkInfo.nStages + 3) {
            starks.getChallenge(transcript, challenges[i * FIELD_EXTENSION]);
        }
    }

    TimerStopAndLog(STARK_STEP_EVALS);

    //--------------------------------
    // 6. Compute FRI
    //--------------------------------
    TimerStart(STARK_STEP_FRI);

    TimerStart(COMPUTE_FRI_POLYNOMIAL);
    starks.calculateFRIPolynomial(params, expressionsCtx);
    TimerStopAndLog(COMPUTE_FRI_POLYNOMIAL);

    Goldilocks::Element challenge[FIELD_EXTENSION];
    Goldilocks::Element *friPol = &params.aux_trace[setupCtx.starkInfo.mapOffsets[std::make_pair("f", true)]];
    
    TimerStart(STARK_FRI_FOLDING);
    uint64_t nBitsExt =  setupCtx.starkInfo.starkStruct.steps[0].nBits;
    for (uint64_t step = 0; step < setupCtx.starkInfo.starkStruct.steps.size(); step++)
    {   
        uint64_t currentBits = setupCtx.starkInfo.starkStruct.steps[step].nBits;
        uint64_t prevBits = step == 0 ? currentBits : setupCtx.starkInfo.starkStruct.steps[step - 1].nBits;
        FRI<RawFrP::Element>::fold(step, friPol, challenge, nBitsExt, prevBits, currentBits);
        if (step < setupCtx.starkInfo.starkStruct.steps.size() - 1)
        {
            FRI<RawFrP::Element>::merkelize(step, proof, friPol, starks.treesFRI[step], currentBits, setupCtx.starkInfo.starkStruct.steps[step + 1].nBits);
            starks.addTranscript(transcript, &proof.proof.fri.treesFRI[step].root[0], nFieldElements);
        }
        else
        {
            if(!setupCtx.starkInfo.starkStruct.hashCommits) {
                starks.addTranscriptGL(transcript, friPol, (1 << setupCtx.starkInfo.starkStruct.steps[step].nBits) * FIELD_EXTENSION);
            } else {
                RawFrP::Element hash[nFieldElements];
                starks.calculateHash(hash, friPol, (1 << setupCtx.starkInfo.starkStruct.steps[step].nBits) * FIELD_EXTENSION);
                starks.addTranscript(transcript, hash, nFieldElements);
            } 
            
        }
        starks.getChallenge(transcript, *challenge);
    }
    TimerStopAndLog(STARK_FRI_FOLDING);
    TimerStart(STARK_FRI_QUERIES);

    uint64_t friQueries[setupCtx.starkInfo.starkStruct.nQueries];

    TranscriptBN128 transcriptPermutation(setupCtx.starkInfo.starkStruct.merkleTreeArity, setupCtx.starkInfo.starkStruct.merkleTreeCustom);
    starks.addTranscriptGL(transcriptPermutation, challenge, FIELD_EXTENSION);
    transcriptPermutation.getPermutations(friQueries, setupCtx.starkInfo.starkStruct.nQueries, setupCtx.starkInfo.starkStruct.steps[0].nBits);

    uint64_t nTrees = setupCtx.starkInfo.nStages + setupCtx.starkInfo.customCommits.size() + 2;
    FRI<RawFrP::Element>::proveQueries(friQueries, setupCtx.starkInfo.starkStruct.nQueries, proof, starks.treesGL, nTrees);
    for(uint64_t step = 1; step < setupCtx.starkInfo.starkStruct.steps.size(); ++step) {
        FRI<RawFrP::Element>::proveFRIQueries(friQueries, setupCtx.starkInfo.starkStruct.nQueries, step, setupCtx.starkInfo.starkStruct.steps[step].nBits, proof, starks.treesFRI[step - 1]);
    }

    FRI<RawFrP::Element>::setFinalPol(proof, friPol, setupCtx.starkInfo.starkStruct.steps[setupCtx.starkInfo.starkStruct.steps.size() - 1].nBits);
    TimerStopAndLog(STARK_FRI_QUERIES);

    TimerStopAndLog(STARK_STEP_FRI);

    proof.proof.setEvals(params.evals);
    nlohmann::json zkin = proof.proof.proof2json();
    zkin["publics"] = json::array();
    for(uint64_t i = 0; i < uint64_t(globalInfo["nPublics"]); ++i) {
        zkin["publics"][i] = Goldilocks::toString(publicInputs[i]);
    }

    if(!proofFile.empty()) {
        json2file(zkin, proofFile);
    }

    return (void *) new nlohmann::json(zkin);
    

    TimerStopAndLog(STARK_PROOF);

    
    return nullptr;
}
