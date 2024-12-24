#ifndef GEN_RECURSIVE_PROOF_GPU_HPP
#define GEN_RECURSIVE_PROOF_GPU_HPP

#include "starks.hpp"
#include "proof2zkinStark.hpp"
#include "cuda_utils.cuh"
#include "gl64_t.cuh"
#include "expressions_gpu.cuh"

__global__ void insertTracePol(Goldilocks::Element* d_trace, uint64_t offset, uint64_t stride, Goldilocks::Element* d_pol, uint64_t dim, uint64_t N){
    uint64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < N){
        if(dim == 1) d_trace[offset + idx * stride] = d_pol[idx];
        else{
            d_trace[offset + idx * stride] = d_pol[idx * dim ];
            d_trace[offset + idx * stride + 1] = d_pol[idx * dim + 1];
            d_trace[offset + idx * stride + 2] = d_pol[idx * dim + 2];
        }
    }
}

void offloadCommit(uint64_t step, MerkleTreeGL** treesGL, Goldilocks::Element *trace, gl64_t *d_trace, uint64_t* d_tree, FRIProof<Goldilocks::Element> &proof, SetupCtx& setupCtx){

    uint64_t ncols = setupCtx.starkInfo.mapSectionsN["cm" + to_string(step)];
    uint64_t NExtended = 1 << setupCtx.starkInfo.starkStruct.nBitsExt;
    uint64_t size = NExtended * ncols * sizeof(Goldilocks::Element);
    uint64_t tree_size = treesGL[step - 1]->getNumNodes(NExtended) * sizeof(uint64_t);
    std::string section = "cm" + to_string(step);  
    uint64_t offset = setupCtx.starkInfo.mapOffsets[make_pair(section, true)];
    treesGL[step - 1]->setSource(trace + offset);
    if(ncols > 0){
        CHECKCUDAERR(cudaMemcpy(trace + offset, d_trace + offset, size, cudaMemcpyDeviceToHost));
        CHECKCUDAERR(cudaMemcpy(treesGL[step - 1]->get_nodes_ptr(), d_tree, tree_size, cudaMemcpyDeviceToHost));
    }else{
        treesGL[step - 1]->merkelize();
    }
    treesGL[step - 1]->getRoot(&proof.proof.roots[step - 1][0]);
}

template <typename ElementType>
void *genRecursiveProof_gpu(SetupCtx& setupCtx, json& globalInfo, uint64_t airgroupId, Goldilocks::Element *witness, Goldilocks::Element *pConstPols, Goldilocks::Element *pConstTree, Goldilocks::Element *publicInputs, std::string proofFile, DeviceCommitBuffers* d_buffers) { 

    TimerStart(STARK_PROOF);

    Goldilocks::Element *trace = new Goldilocks::Element[setupCtx.starkInfo.mapTotalN];
    CHECKCUDAERR(cudaMemset(d_buffers->d_trace, 0, setupCtx.starkInfo.mapTotalN * sizeof(Goldilocks::Element)));
    Goldilocks::Element* trace_debug = new Goldilocks::Element[setupCtx.starkInfo.mapTotalN*2];

    FRIProof<Goldilocks::Element> proof(setupCtx.starkInfo);

    using TranscriptType = std::conditional_t<std::is_same<ElementType, Goldilocks::Element>::value, TranscriptGL, TranscriptBN128>;
    
    Starks<ElementType> starks(setupCtx, pConstTree);

    ExpressionsGPU expressionsCtx(setupCtx);

    uint64_t nFieldElements = setupCtx.starkInfo.starkStruct.verificationHashType == std::string("BN128") ? 1 : HASH_SIZE;

    TranscriptType transcript(setupCtx.starkInfo.starkStruct.merkleTreeArity, setupCtx.starkInfo.starkStruct.merkleTreeCustom);

    Goldilocks::Element* evals = new Goldilocks::Element[setupCtx.starkInfo.evMap.size() * FIELD_EXTENSION];
    Goldilocks::Element* challenges = new Goldilocks::Element[setupCtx.starkInfo.challengesMap.size() * FIELD_EXTENSION];
    Goldilocks::Element* airgroupValues = nullptr;    

    StepsParams params = {
        trace: witness,
        pols : trace,
        publicInputs : publicInputs,
        challenges : challenges,
        airgroupValues : nullptr,
        evals : evals,
        xDivXSub : nullptr,
        pConstPolsAddress: pConstPols,
        pConstPolsExtendedTreeAddress: pConstTree,
    };

    StepsParams d_params = {
        trace: (Goldilocks::Element* ) d_buffers->d_witness,
        pols : (Goldilocks::Element* ) d_buffers->d_trace,
        publicInputs : (Goldilocks::Element* ) d_buffers->d_publicInputs,
        challenges : nullptr,
        airgroupValues : nullptr,
        evals : nullptr,
        xDivXSub : nullptr,
        pConstPolsAddress: (Goldilocks::Element* ) d_buffers->d_constPols,
        pConstPolsExtendedTreeAddress: (Goldilocks::Element* ) d_buffers->d_constTree,
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

    //Allocate d_tree
    uint64_t** d_tree = new uint64_t*[1];
    TimerStart(STARK_COMMIT_STAGE_1);
    starks.commitStage_inplace(1, d_buffers->d_witness, d_buffers->d_trace, d_tree, d_buffers);
    TimerStopAndLog(STARK_COMMIT_STAGE_1);

    offloadCommit(1, starks.treesGL, trace, d_buffers->d_trace, *d_tree, proof, setupCtx);

    ////////////
    // rick: check if the traces are equal at this point
    std::cout <<" CHECK TRACE 1" <<std::endl;
    CHECKCUDAERR(cudaMemcpy(trace_debug, d_buffers->d_trace, setupCtx.starkInfo.mapTotalN * sizeof(Goldilocks::Element), cudaMemcpyDeviceToHost));
    for(uint64_t i = 0; i < setupCtx.starkInfo.mapTotalN; ++i) {
        if(trace_debug[i].fe != trace[i].fe) {
            std::cout << "Error in trace" << std::endl;
            exit(1);
        }
    }

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

    Dest numStruct(num, false);
    cudaMalloc(&numStruct.dest_gpu, N*FIELD_EXTENSION*sizeof(Goldilocks::Element));
    numStruct.addParams(setupCtx.expressionsBin.expressionsInfo[numFieldId]);
    
    Dest denStruct(den, false);
    cudaMalloc(&denStruct.dest_gpu, N*FIELD_EXTENSION*sizeof(Goldilocks::Element));
    denStruct.addParams(setupCtx.expressionsBin.expressionsInfo[denFieldId], true);
    std::vector<Dest> dests = {numStruct, denStruct};

    double time = omp_get_wtime();
    expressionsCtx.calculateExpressions_gpu(params, d_params, setupCtx.expressionsBin.expressionsBinArgsExpressions, dests, uint64_t(1 << setupCtx.starkInfo.starkStruct.nBits));

    

    time = omp_get_wtime() - time;
    std::cout << "rick calculateExpressions time: " << time << std::endl;
    

    Goldilocks3::copy((Goldilocks3::Element *)&gprod[0], &Goldilocks3::one());
    for(uint64_t i = 1; i < N; ++i) {
        Goldilocks::Element res[3];
        Goldilocks3::mul((Goldilocks3::Element *)res, (Goldilocks3::Element *)&num[(i - 1) * FIELD_EXTENSION], (Goldilocks3::Element *)&den[(i - 1) * FIELD_EXTENSION]);
        Goldilocks3::mul((Goldilocks3::Element *)&gprod[i * FIELD_EXTENSION], (Goldilocks3::Element *)&gprod[(i - 1) * FIELD_EXTENSION], (Goldilocks3::Element *)res);
    }

    Goldilocks::Element *d_grod;
    cudaMalloc(&d_grod, N*FIELD_EXTENSION*sizeof(Goldilocks::Element));
    cudaMemcpy(d_grod, gprod, N*FIELD_EXTENSION*sizeof(Goldilocks::Element), cudaMemcpyHostToDevice);

    Polinomial gprodTransposedPol;
    setupCtx.starkInfo.getPolynomial(gprodTransposedPol, trace, "cm", setupCtx.starkInfo.cmPolsMap[gprodFieldId], false);

    uint64_t offset = setupCtx.starkInfo.getTraceOffset("cm", setupCtx.starkInfo.cmPolsMap[gprodFieldId], false);
    uint64_t nCols = setupCtx.starkInfo.getTraceNColsSection("cm", setupCtx.starkInfo.cmPolsMap[gprodFieldId], false);

    #pragma omp parallel for
    for(uint64_t j = 0; j < N; ++j) {
        std::memcpy(gprodTransposedPol[j], &gprod[j*FIELD_EXTENSION], FIELD_EXTENSION * sizeof(Goldilocks::Element));
    }
    dim3 nThreads(256);
    dim3 nBlocks((N + nThreads.x - 1) / nThreads.x);
    insertTracePol<<<nBlocks, nThreads>>>((Goldilocks::Element*) d_buffers->d_trace, offset, nCols, d_grod, FIELD_EXTENSION, N);
    
    delete num;
    delete den;
    delete gprod;

    ////////////
    // rick: check if the traces are equal at this point
    std::cout <<" CHECK TRACE 2" <<std::endl;
    CHECKCUDAERR(cudaMemcpy(trace_debug, d_buffers->d_trace, setupCtx.starkInfo.mapTotalN * sizeof(Goldilocks::Element), cudaMemcpyDeviceToHost));
    for(uint64_t i = 0; i < setupCtx.starkInfo.mapTotalN; ++i) {
        if(trace_debug[i].fe != trace[i].fe) {
            std::cout << "Error in trace" << std::endl;
            exit(1);
        }
    }

    time = omp_get_wtime();
    TimerStart(CALCULATE_IM_POLS);

    std::vector<Dest> dests2;
    for(uint64_t i = 0; i < setupCtx.starkInfo.cmPolsMap.size(); i++) {
        if(setupCtx.starkInfo.cmPolsMap[i].imPol && setupCtx.starkInfo.cmPolsMap[i].stage == 2) {
            Goldilocks::Element* pols = setupCtx.starkInfo.cmPolsMap[i].stage == 1 ? params.trace : params.pols;
            uint64_t offset_ = setupCtx.starkInfo.mapOffsets[std::make_pair("cm" + to_string(2), false)] + setupCtx.starkInfo.cmPolsMap[i].stagePos;
            Dest destStruct(&pols[offset_], setupCtx.starkInfo.mapSectionsN["cm" + to_string(2)]);
            destStruct.addParams(setupCtx.expressionsBin.expressionsInfo[setupCtx.starkInfo.cmPolsMap[i].expId], false);
            destStruct.dest_gpu = (Goldilocks::Element *)(d_buffers->d_trace + offset_);     
            dests2.push_back(destStruct);
        }
    }

    expressionsCtx.calculateExpressions(params, setupCtx.expressionsBin.expressionsBinArgsExpressions, dests2, uint64_t(1 << setupCtx.starkInfo.starkStruct.nBits));
    expressionsCtx.calculateExpressions_gpu2(params, d_params, setupCtx.expressionsBin.expressionsBinArgsExpressions, dests2, uint64_t(1 << setupCtx.starkInfo.starkStruct.nBits));

     TimerStopAndLog(CALCULATE_IM_POLS);
    time = omp_get_wtime() - time;
    std::cout << "rick calculateImPolsExpressions time: " << time << std::endl;
        
    cudaMemcpy(d_buffers->d_trace, trace, setupCtx.starkInfo.mapTotalN * sizeof(Goldilocks::Element), cudaMemcpyHostToDevice);
    ////////////
    // rick: check if the traces are equal at this point
    std::cout <<" CHECK TRACE 3" <<std::endl;
    CHECKCUDAERR(cudaMemcpy(trace_debug, d_buffers->d_trace, setupCtx.starkInfo.mapTotalN * sizeof(Goldilocks::Element), cudaMemcpyDeviceToHost));
    for(uint64_t i = 0; i < setupCtx.starkInfo.mapTotalN; ++i) {
        if(trace_debug[i].fe != trace[i].fe) {
            std::cout << "Error in trace" << std::endl;
            exit(1);
        }
    }
    TimerStart(STARK_COMMIT_STAGE_2);
    starks.commitStage_inplace(2, d_buffers->d_witness, d_buffers->d_trace, d_tree, d_buffers);
    offloadCommit(2, starks.treesGL, trace, d_buffers->d_trace, *d_tree, proof, setupCtx);
    
    ////////////
    // rick: check if the traces are equal at this point
    std::cout <<" CHECK TRACE 4" <<std::endl;
    CHECKCUDAERR(cudaMemcpy(trace_debug, d_buffers->d_trace, setupCtx.starkInfo.mapTotalN * sizeof(Goldilocks::Element), cudaMemcpyDeviceToHost));
    for(uint64_t i = 0; i < setupCtx.starkInfo.mapTotalN; ++i) {
        if(trace_debug[i].fe != trace[i].fe) {
            std::cout << "Error in trace" << std::endl;
            exit(1);
        }
    }
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
    time = omp_get_wtime();
    //expressionsCtx.calculateExpression(params, &params.pols[setupCtx.starkInfo.mapOffsets[std::make_pair("q", true)]], setupCtx.starkInfo.cExpId);
    uint64_t domainSize;
    uint64_t expressionId = setupCtx.starkInfo.cExpId;
    if(expressionId == setupCtx.starkInfo.cExpId || expressionId == setupCtx.starkInfo.friExpId) {
        setupCtx.expressionsBin.expressionsInfo[expressionId].destDim = 3;
        domainSize = 1 << setupCtx.starkInfo.starkStruct.nBitsExt;
    } else {
        domainSize = 1 << setupCtx.starkInfo.starkStruct.nBits;
    }
    Dest destStruct(&params.pols[setupCtx.starkInfo.mapOffsets[std::make_pair("q", true)]]);
    destStruct.addParams(setupCtx.expressionsBin.expressionsInfo[expressionId], false);
    destStruct.dest_gpu = (Goldilocks::Element *)(d_buffers->d_trace + setupCtx.starkInfo.mapOffsets[std::make_pair("q", true)]);     
    std::vector<Dest> dests3 = {destStruct};
    expressionsCtx.calculateExpressions(params, setupCtx.expressionsBin.expressionsBinArgsExpressions, dests3, domainSize);
    expressionsCtx.calculateExpressions_gpu2(params, d_params, setupCtx.expressionsBin.expressionsBinArgsExpressions, dests3, domainSize);
    time = omp_get_wtime() - time;
    std::cout << "rick calculateExpression time: " << time << std::endl;
    ////////////
    // rick: check if the traces are equal at this point
    std::cout <<" CHECK TRACE 5" <<std::endl;
    CHECKCUDAERR(cudaMemcpy(trace_debug, d_buffers->d_trace, setupCtx.starkInfo.mapTotalN * sizeof(Goldilocks::Element), cudaMemcpyDeviceToHost));
    for(uint64_t i = 0; i < setupCtx.starkInfo.mapTotalN; ++i) {
        if(trace_debug[i].fe != trace[i].fe) {
            std::cout << "Error in trace" << std::endl;
            exit(1);
        }
    }
    /*uint64_t offset_ = setupCtx.starkInfo.mapOffsets[std::make_pair("q", true)];
    for(uint64_t i = 0; i < domainSize*FIELD_EXTENSION; ++i) {
        if(trace_debug[offset_ + i].fe != trace[offset_ + i].fe) {
            std::cout << "Error in trace, row: " << i << " " << trace_debug[offset_ + i].fe << " " << trace[offset_ + i].fe << std::endl;
            exit(0);
        }        
    }*/
    TimerStart(STARK_COMMIT_QUOTIENT_POLYNOMIAL);
    starks.commitStage(setupCtx.starkInfo.nStages + 1, nullptr, params.pols, proof);
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
    Goldilocks::Element* LEv = &trace[setupCtx.starkInfo.mapOffsets[make_pair("LEv", true)]];

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
    params.xDivXSub = &trace[setupCtx.starkInfo.mapOffsets[std::make_pair("xDivXSubXi", true)]];
    starks.calculateXDivXSub(xiChallenge, params.xDivXSub);
    starks.calculateFRIPolynomial(params);
    TimerStopAndLog(COMPUTE_FRI_POLYNOMIAL);

    Goldilocks::Element challenge[FIELD_EXTENSION];
    Goldilocks::Element *friPol = &trace[setupCtx.starkInfo.mapOffsets[std::make_pair("f", true)]];
    
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

    delete trace;
    return (void *) new nlohmann::json(zkin);
}
#endif
