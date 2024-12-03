#ifndef GEN_RECURSIVE_PROOF_GPU_HPP
#define GEN_RECURSIVE_PROOF_GPU_HPP

#include "starks.hpp"
#include "proof2zkinStark.hpp"
#include "cuda_utils.cuh"
#include "gl64_t.cuh"


gl64_t * genDeviceTrace(Goldilocks::Element *pAddress, SetupCtx& setupCtx){

    uint64_t total_n = setupCtx.starkInfo.mapTotalN;
    gl64_t *d_pAddress;
    CHECKCUDAERR(cudaMalloc(&d_pAddress, total_n * sizeof(Goldilocks::Element)));
    CHECKCUDAERR(cudaMemset(d_pAddress, 0, total_n * sizeof(Goldilocks::Element)));
    //get_map_offsets_c(p_stark_info, "cm1", false);
    uint64_t offset = setupCtx.starkInfo.mapOffsets[std::make_pair("cm1", false)];
    uint64_t n = setupCtx.starkInfo.starkStruct.steps[0].nBits;
    uint64_t N = 1 << n;
    uint64_t n_cols = setupCtx.starkInfo.mapSectionsN["cm1"];
    CHECKCUDAERR(cudaMemcpy(d_pAddress+offset, pAddress+offset, (N*n_cols) * sizeof(Goldilocks::Element), cudaMemcpyHostToDevice));
    return d_pAddress;
}

void freeDeviceTrace(gl64_t *d_pAddress){
    CHECKCUDAERR(cudaFree(d_pAddress));
}

template <typename ElementType>
void *genRecursiveProof_gpu(SetupCtx& setupCtx, json& globalInfo, uint64_t airgroupId, Goldilocks::Element *pAddress, gl64_t *d_pAddress, Goldilocks::Element *pConstPols, Goldilocks::Element *pConstTree, Goldilocks::Element *publicInputs, std::string proofFile) { // sprint: afegir l'adreça de la gpu

    TimerStart(STARK_PROOF);
    //sprint: copiar  witness i setejar la resta a zero
    

    TimerStart(SOLAPE1);

    FRIProof<Goldilocks::Element> proof(setupCtx.starkInfo);
    //sprint: alocatar aixo a la gpu

    using TranscriptType = std::conditional_t<std::is_same<ElementType, Goldilocks::Element>::value, TranscriptGL, TranscriptBN128>;
    
    Starks<ElementType> starks(setupCtx, pConstTree);
    //rick: aqui ja puc llençar el commit 1 no?


    ExpressionsPack expressionsCtx(setupCtx);

    uint64_t nFieldElements = setupCtx.starkInfo.starkStruct.verificationHashType == std::string("BN128") ? 1 : HASH_SIZE;

    TranscriptType transcript(setupCtx.starkInfo.starkStruct.merkleTreeArity, setupCtx.starkInfo.starkStruct.merkleTreeCustom);

    Goldilocks::Element* evals = new Goldilocks::Element[setupCtx.starkInfo.evMap.size() * FIELD_EXTENSION];
    Goldilocks::Element* challenges = new Goldilocks::Element[setupCtx.starkInfo.challengesMap.size() * FIELD_EXTENSION];
    Goldilocks::Element* airgroupValues = nullptr;    

    StepsParams params = {
        pols : pAddress,
        publicInputs : publicInputs,
        challenges : challenges,
        airgroupValues : airgroupValues,
        evals : evals,
        xDivXSub : nullptr,
        pConstPolsAddress: pConstPols,
        pConstPolsExtendedTreeAddress: pConstTree,
    };

    //--------------------------------
    // 0.- Add const root and publics to transcript
    //--------------------------------

    TimerStart(STARK_STEP_0);
    ElementType verkey[nFieldElements];
    starks.treesGL[setupCtx.starkInfo.nStages + 1]->getRoot(verkey);
    starks.addTranscript(transcript, &verkey[0], nFieldElements);
    if(setupCtx.starkInfo.nPublics > 0) {
        if(!setupCtx.starkInfo.starkStruct.hashCommits) {
            starks.addTranscriptGL(transcript, &publicInputs[0], setupCtx.starkInfo.nPublics);
        } else {
            ElementType hash[nFieldElements];
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
    TimerStopAndLog(SOLAPE1);

    TimerStart(STARK_COMMIT_STAGE_1);
    starks.commitStage_inplace(1, pAddress, d_pAddress, proof); // srpint: correr sobre el buffer
    TimerStopAndLog(STARK_COMMIT_STAGE_1);
    //sprint: baixar tot el que necessito per aplicar expressions
    //rick: aqui em sincronitzo amb el llençament del commit 1
    TimerStart(SOLAPE2);

    starks.addTranscript(transcript, &proof.proof.roots[0][0], nFieldElements);
    
    TimerStopAndLog(STARK_STEP_1);

    TimerStart(STARK_STEP_2);
    for (uint64_t i = 0; i < setupCtx.starkInfo.challengesMap.size(); i++) {
        if(setupCtx.starkInfo.challengesMap[i].stage == 2) {
            starks.getChallenge(transcript, challenges[i * FIELD_EXTENSION]);
        }
    }

    //rick: tot aixo tambe ho puc posar sota del commit 1 no
    uint64_t N = 1 << setupCtx.starkInfo.starkStruct.nBits;
    Goldilocks::Element *num = new Goldilocks::Element[N*FIELD_EXTENSION];
    Goldilocks::Element *den = new Goldilocks::Element[N*FIELD_EXTENSION];
    Goldilocks::Element *gprod = new Goldilocks::Element[N*FIELD_EXTENSION];

    uint64_t gprodFieldId = setupCtx.expressionsBin.hints[0].fields[0].values[0].id;
    uint64_t numFieldId = setupCtx.expressionsBin.hints[0].fields[1].values[0].id;
    uint64_t denFieldId = setupCtx.expressionsBin.hints[0].fields[2].values[0].id;

    Dest numStruct(num);
    numStruct.addParams(setupCtx.expressionsBin.expressionsInfo[numFieldId]);
    Dest denStruct(den);
    denStruct.addParams(setupCtx.expressionsBin.expressionsInfo[denFieldId], true);
    std::vector<Dest> dests = {numStruct, denStruct};
    //rick: fins aqui

    //rick: tot aixo va a la GPU
    expressionsCtx.calculateExpressions(params, setupCtx.expressionsBin.expressionsBinArgsExpressions, dests, uint64_t(1 << setupCtx.starkInfo.starkStruct.nBits));
    

    Goldilocks3::copy((Goldilocks3::Element *)&gprod[0], &Goldilocks3::one());
    for(uint64_t i = 1; i < N; ++i) {
        Goldilocks::Element res[3];
        Goldilocks3::mul((Goldilocks3::Element *)res, (Goldilocks3::Element *)&num[(i - 1) * FIELD_EXTENSION], (Goldilocks3::Element *)&den[(i - 1) * FIELD_EXTENSION]);
        Goldilocks3::mul((Goldilocks3::Element *)&gprod[i * FIELD_EXTENSION], (Goldilocks3::Element *)&gprod[(i - 1) * FIELD_EXTENSION], (Goldilocks3::Element *)res);
    }
    //rick fins aqui

    //rick: aqui que fas, copiea a la traça?
    Polinomial gprodTransposedPol;
    setupCtx.starkInfo.getPolynomial(gprodTransposedPol, pAddress, "cm", setupCtx.starkInfo.cmPolsMap[gprodFieldId], false);
#pragma omp parallel for
    for(uint64_t j = 0; j < N; ++j) {
        std::memcpy(gprodTransposedPol[j], &gprod[j*FIELD_EXTENSION], FIELD_EXTENSION * sizeof(Goldilocks::Element));
    }
    //rick
    
    delete num;
    delete den;
    delete gprod;

    TimerStart(CALCULATE_IM_POLS);
    starks.calculateImPolsExpressions(2, params); // a la GPU
    TimerStopAndLog(CALCULATE_IM_POLS);
    TimerStopAndLog(SOLAPE2);


    TimerStart(STARK_COMMIT_STAGE_2);
    starks.commitStage(2, pAddress, proof);
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
    
    expressionsCtx.calculateExpression(params, &params.pols[setupCtx.starkInfo.mapOffsets[std::make_pair("q", true)]], setupCtx.starkInfo.cExpId);

    TimerStart(STARK_COMMIT_QUOTIENT_POLYNOMIAL);
    starks.commitStage(setupCtx.starkInfo.nStages + 1, pAddress, proof);
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
    Goldilocks::Element* LEv = &pAddress[setupCtx.starkInfo.mapOffsets[make_pair("LEv", true)]];

    starks.computeLEv(xiChallenge, LEv);
    starks.computeEvals(params ,LEv, proof);

    if(!setupCtx.starkInfo.starkStruct.hashCommits) {
        starks.addTranscriptGL(transcript, evals, setupCtx.starkInfo.evMap.size() * FIELD_EXTENSION);
    } else {
        ElementType hash[nFieldElements];
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
    params.xDivXSub = &pAddress[setupCtx.starkInfo.mapOffsets[std::make_pair("xDivXSubXi", true)]];
    starks.calculateXDivXSub(xiChallenge, params.xDivXSub);
    starks.calculateFRIPolynomial(params);
    TimerStopAndLog(COMPUTE_FRI_POLYNOMIAL);

    Goldilocks::Element challenge[FIELD_EXTENSION];
    Goldilocks::Element *friPol = &pAddress[setupCtx.starkInfo.mapOffsets[std::make_pair("f", true)]];
    
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
            starks.addTranscript(transcript, &proof.proof.fri.treesFRI[step].root[0], nFieldElements);
        }
        else
        {
            if(!setupCtx.starkInfo.starkStruct.hashCommits) {
                starks.addTranscriptGL(transcript, friPol, (1 << setupCtx.starkInfo.starkStruct.steps[step].nBits) * FIELD_EXTENSION);
            } else {
                ElementType hash[nFieldElements];
                starks.calculateHash(hash, friPol, (1 << setupCtx.starkInfo.starkStruct.steps[step].nBits) * FIELD_EXTENSION);
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
    FRI<Goldilocks::Element>::proveQueries(friQueries, setupCtx.starkInfo.starkStruct.nQueries, proof, starks.treesGL, nTrees);
    for(uint64_t step = 1; step < setupCtx.starkInfo.starkStruct.steps.size(); ++step) {
        FRI<Goldilocks::Element>::proveFRIQueries(friQueries, setupCtx.starkInfo.starkStruct.nQueries, step, setupCtx.starkInfo.starkStruct.steps[step].nBits, proof, starks.treesFRI[step - 1]);
    }

    FRI<ElementType>::setFinalPol(proof, friPol, setupCtx.starkInfo.starkStruct.steps[setupCtx.starkInfo.starkStruct.steps.size() - 1].nBits);
    TimerStopAndLog(STARK_FRI_QUERIES);

    TimerStopAndLog(STARK_STEP_FRI);

    delete challenges;
    delete evals;
    delete airgroupValues;
    
    nlohmann::json jProof = proof.proof.proof2json();
    nlohmann::json zkin = proof2zkinStark(jProof, setupCtx.starkInfo);

    if(!proofFile.empty()) {
        json2file(jProof, proofFile);
    }

    TimerStopAndLog(STARK_PROOF);

    zkin = publics2zkin(zkin, publicInputs, globalInfo, airgroupId);

    return (void *) new nlohmann::json(zkin);
}
#endif
