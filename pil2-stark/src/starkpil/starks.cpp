#include "definitions.hpp"
#include "starks.hpp"
#include "zklog.hpp"
#include "exit_process.hpp"

template <typename ElementType>
void Starks<ElementType>::extendAndMerkelizeCustomCommit(uint64_t commitId, uint64_t step, Goldilocks::Element *buffer, FRIProof<ElementType> &proof, Goldilocks::Element *pBuffHelper)
{   
    uint64_t N = 1 << setupCtx.starkInfo.starkStruct.nBits;
    uint64_t NExtended = 1 << setupCtx.starkInfo.starkStruct.nBitsExt;

    std::string section = setupCtx.starkInfo.customCommits[commitId].name + to_string(step);
    uint64_t nCols = setupCtx.starkInfo.mapSectionsN[section];
    Goldilocks::Element *pBuff = &buffer[setupCtx.starkInfo.mapOffsets[make_pair(section, false)]];
    Goldilocks::Element *pBuffExtended = &buffer[setupCtx.starkInfo.mapOffsets[make_pair(section, true)]];

    if(pBuffHelper != nullptr) {
        ntt.extendPol(pBuffExtended, pBuff, NExtended, N, nCols, pBuffHelper);
    } else {
        ntt.extendPol(pBuffExtended, pBuff, NExtended, N, nCols);
    }
    
    uint64_t pos = setupCtx.starkInfo.nStages + 2 + commitId;
    treesGL[pos]->setSource(pBuffExtended);
    if(setupCtx.starkInfo.starkStruct.verificationHashType == "GL") {
        Goldilocks::Element *pBuffNodesGL = &buffer[(N + NExtended) * nCols];
        ElementType *pBuffNodes = (ElementType *)pBuffNodesGL;
        treesGL[pos]->setNodes(pBuffNodes);
    }
    treesGL[pos]->merkelize();
    treesGL[pos]->getRoot(&proof.proof.roots[pos - 1][0]);
}

template <typename ElementType>
void Starks<ElementType>::extendAndMerkelize(uint64_t step, Goldilocks::Element *trace, Goldilocks::Element *aux_trace, FRIProof<ElementType> &proof, Goldilocks::Element *pBuffHelper)
{   
    uint64_t N = 1 << setupCtx.starkInfo.starkStruct.nBits;
    uint64_t NExtended = 1 << setupCtx.starkInfo.starkStruct.nBitsExt;

    std::string section = "cm" + to_string(step);  
    uint64_t nCols = setupCtx.starkInfo.mapSectionsN["cm" + to_string(step)];
    
    Goldilocks::Element *pBuff = step == 1 ? trace : &aux_trace[setupCtx.starkInfo.mapOffsets[make_pair(section, false)]];
    Goldilocks::Element *pBuffExtended = &aux_trace[setupCtx.starkInfo.mapOffsets[make_pair(section, true)]];
    

    if(pBuffHelper != nullptr) {
        ntt.extendPol(pBuffExtended, pBuff, NExtended, N, nCols, pBuffHelper);
    } else {
        ntt.extendPol(pBuffExtended, pBuff, NExtended, N, nCols);
    }
    
    treesGL[step - 1]->setSource(pBuffExtended);
    if(setupCtx.starkInfo.starkStruct.verificationHashType == "GL") {
        Goldilocks::Element *pBuffNodesGL = &aux_trace[setupCtx.starkInfo.mapOffsets[make_pair("mt" + to_string(step), true)]];
        ElementType *pBuffNodes = (ElementType *)pBuffNodesGL;
        treesGL[step - 1]->setNodes(pBuffNodes);
    }
    treesGL[step - 1]->merkelize();
    treesGL[step - 1]->getRoot(&proof.proof.roots[step - 1][0]);
}
template <typename ElementType>
void Starks<ElementType>::extendAndMerkelize_inplace(uint64_t step, gl64_t *d_trace, gl64_t *d_aux_trace, DeviceCommitBuffers *d_buffers)
{
#ifdef __USE_CUDA__

    uint64_t N = 1 << setupCtx.starkInfo.starkStruct.nBits;
    uint64_t NExtended = 1 << setupCtx.starkInfo.starkStruct.nBitsExt;
    std::string section = "cm" + to_string(step);
    uint64_t nCols = setupCtx.starkInfo.mapSectionsN[section];

    gl64_t *src = step == 1 ? d_trace : d_aux_trace;
    uint64_t offset_src = step == 1 ? 0 : setupCtx.starkInfo.mapOffsets[make_pair(section, false)];
    gl64_t *dst = d_aux_trace;
    uint64_t offset_dst = setupCtx.starkInfo.mapOffsets[make_pair(section, true)];
    Goldilocks::Element * dstGL = (Goldilocks::Element*) (d_aux_trace);

    treesGL[step - 1]->setSource(dstGL + offset_dst);
    Goldilocks::Element *pNodes = dstGL + setupCtx.starkInfo.mapOffsets[make_pair("mt" + to_string(step), true)];
    treesGL[step - 1]->setNodes((ElementType*) pNodes);
    
    NTT_Goldilocks ntt(N);

    if (nCols > 0)
    {

        ntt.LDE_MerkleTree_GPU_inplace(pNodes, dst, offset_dst, src, offset_src, N, NExtended, nCols, d_buffers, setupCtx.starkInfo.mapOffsets[std::make_pair("buff_helper_fft", false)]);
    }
#endif
}

template <typename ElementType>
void Starks<ElementType>::commitStage(uint64_t step, Goldilocks::Element *trace, Goldilocks::Element *aux_trace, FRIProof<ElementType> &proof, Goldilocks::Element* pBuffHelper)
{  

    if (step <= setupCtx.starkInfo.nStages)
    {
        extendAndMerkelize(step, trace, aux_trace, proof, pBuffHelper);
    }
    else
    {
        computeQ(step, aux_trace, proof, pBuffHelper);
    }
}

template <typename ElementType>
void Starks<ElementType>::commitStage_inplace(uint64_t step, gl64_t *d_trace, gl64_t *d_aux_trace, DeviceCommitBuffers *d_buffers)
{
    if (step <= setupCtx.starkInfo.nStages)
    {
        extendAndMerkelize_inplace(step, d_trace, d_aux_trace, d_buffers);
    }
    else
    {
        computeQ_inplace(step, d_aux_trace, d_buffers);
    }
}

template <typename ElementType>
void Starks<ElementType>::computeQ(uint64_t step, Goldilocks::Element *buffer, FRIProof<ElementType> &proof, Goldilocks::Element *pBuffHelper)
{
    uint64_t N = 1 << setupCtx.starkInfo.starkStruct.nBits;
    uint64_t NExtended = 1 << setupCtx.starkInfo.starkStruct.nBitsExt;

    std::string section = "cm" + to_string(setupCtx.starkInfo.nStages + 1);
    uint64_t nCols = setupCtx.starkInfo.mapSectionsN["cm" + to_string(setupCtx.starkInfo.nStages + 1)];
    Goldilocks::Element *cmQ = &buffer[setupCtx.starkInfo.mapOffsets[make_pair(section, true)]];
    
    if(pBuffHelper != nullptr) {
        nttExtended.INTT(&buffer[setupCtx.starkInfo.mapOffsets[std::make_pair("q", true)]], &buffer[setupCtx.starkInfo.mapOffsets[std::make_pair("q", true)]], NExtended, setupCtx.starkInfo.qDim, pBuffHelper);
    } else {
        nttExtended.INTT(&buffer[setupCtx.starkInfo.mapOffsets[std::make_pair("q", true)]], &buffer[setupCtx.starkInfo.mapOffsets[std::make_pair("q", true)]], NExtended, setupCtx.starkInfo.qDim);
    }

    Goldilocks::Element S[setupCtx.starkInfo.qDeg];
    Goldilocks::Element shiftIn = Goldilocks::exp(Goldilocks::inv(Goldilocks::shift()), N);
    S[0] = Goldilocks::one();
    for(uint64_t i = 1; i < setupCtx.starkInfo.qDeg; i++) {
        S[i] = Goldilocks::mul(S[i - 1], shiftIn);
    }

#pragma omp parallel for collapse(2)
    for (uint64_t p = 0; p < setupCtx.starkInfo.qDeg; p++)
    {   
        for(uint64_t i = 0; i < N; i++)
        { 
            Goldilocks3::mul((Goldilocks3::Element &)cmQ[(i * setupCtx.starkInfo.qDeg + p) * FIELD_EXTENSION], (Goldilocks3::Element &)buffer[setupCtx.starkInfo.mapOffsets[std::make_pair("q", true)] + (p * N + i) * FIELD_EXTENSION], S[p]);
        }
    }

#pragma omp parallel for
    for(uint64_t i = 0; i < (NExtended - N) * setupCtx.starkInfo.qDeg * setupCtx.starkInfo.qDim; ++i) {
        cmQ[N * setupCtx.starkInfo.qDeg * setupCtx.starkInfo.qDim + i] = Goldilocks::zero();
    }
    
    if(pBuffHelper != nullptr) {
        nttExtended.NTT(cmQ, cmQ, NExtended, nCols, pBuffHelper);
    } else {
        nttExtended.NTT(cmQ, cmQ, NExtended, nCols);
    }

    treesGL[step - 1]->setSource(&buffer[setupCtx.starkInfo.mapOffsets[std::make_pair("cm" + to_string(step), true)]]);
    if(setupCtx.starkInfo.starkStruct.verificationHashType == "GL") {
        Goldilocks::Element *pBuffNodesGL = &buffer[setupCtx.starkInfo.mapOffsets[std::make_pair("mt" + to_string(step), true)]];
        ElementType *pBuffNodes = (ElementType *)pBuffNodesGL;
        treesGL[step - 1]->setNodes(pBuffNodes);
    }
    
    treesGL[step - 1]->merkelize();
    treesGL[step - 1]->getRoot(&proof.proof.roots[step - 1][0]);
    
}

template <typename ElementType>
void Starks<ElementType>::computeQ_inplace(uint64_t step, gl64_t *d_aux_trace, DeviceCommitBuffers *d_buffers)
{
#ifdef __USE_CUDA__

    uint64_t N = 1 << setupCtx.starkInfo.starkStruct.nBits;
    uint64_t NExtended = 1 << setupCtx.starkInfo.starkStruct.nBitsExt;
    std::string section = "cm" + to_string(step);
    uint64_t nCols = setupCtx.starkInfo.mapSectionsN[section];

    uint64_t offset_cmQ = setupCtx.starkInfo.mapOffsets[std::make_pair(section, true)];
    uint64_t offset_q = setupCtx.starkInfo.mapOffsets[std::make_pair("q", true)];
    uint64_t qDeg = setupCtx.starkInfo.qDeg;
    uint64_t qDim = setupCtx.starkInfo.qDim;

    Goldilocks::Element S[setupCtx.starkInfo.qDeg];
    Goldilocks::Element shiftIn = Goldilocks::exp(Goldilocks::inv(Goldilocks::shift()), N);
    S[0] = Goldilocks::one();
    for(uint64_t i = 1; i < setupCtx.starkInfo.qDeg; i++) {
        S[i] = Goldilocks::mul(S[i - 1], shiftIn);
    }
    Goldilocks::Element* d_aux_traceGL = (Goldilocks::Element*) d_aux_trace;

    treesGL[step - 1]->setSource( d_aux_traceGL + offset_q);
    Goldilocks::Element *pNodes = d_aux_traceGL + setupCtx.starkInfo.mapOffsets[make_pair("mt" + to_string(step), true)];
    treesGL[step - 1]->setNodes(( ElementType*)pNodes);

    if (nCols > 0)
    {
        nttExtended.computeQ_inplace(pNodes, offset_cmQ, offset_q, qDeg, qDim, S, N, NExtended, nCols, d_buffers, setupCtx.starkInfo.mapOffsets[std::make_pair("buff_helper_fft", false)]);
    }
#endif
}

template <typename ElementType>
void Starks<ElementType>::computeLEv(Goldilocks::Element *xiChallenge, Goldilocks::Element *LEv) {
    uint64_t N = 1 << setupCtx.starkInfo.starkStruct.nBits;
        
    Goldilocks::Element xis[setupCtx.starkInfo.openingPoints.size() * FIELD_EXTENSION];
    Goldilocks::Element xisShifted[setupCtx.starkInfo.openingPoints.size() * FIELD_EXTENSION];
    
    Goldilocks::Element shift_inv = Goldilocks::inv(Goldilocks::shift());
        for (uint64_t i = 0; i < setupCtx.starkInfo.openingPoints.size(); ++i)
    {
        uint64_t openingAbs = setupCtx.starkInfo.openingPoints[i] < 0 ? -setupCtx.starkInfo.openingPoints[i] : setupCtx.starkInfo.openingPoints[i];
        Goldilocks::Element w = Goldilocks::pow(Goldilocks::w(setupCtx.starkInfo.starkStruct.nBits), openingAbs);

        if (setupCtx.starkInfo.openingPoints[i] < 0)
        {
            w = Goldilocks::inv(w);
        }

        Goldilocks3::mul((Goldilocks3::Element &)(xis[i * FIELD_EXTENSION]), (Goldilocks3::Element &)xiChallenge[0], w);
        Goldilocks3::mul((Goldilocks3::Element &)(xisShifted[i * FIELD_EXTENSION]), (Goldilocks3::Element &)(xis[i * FIELD_EXTENSION]), shift_inv);
    }

    #pragma omp parallel for collapse(2)
    for (uint64_t k = 0; k < N; k+=4096)
    {
        for (uint64_t i = 0; i < setupCtx.starkInfo.openingPoints.size(); ++i)
        {
            Goldilocks3::pow((Goldilocks3::Element &)(LEv[(k*setupCtx.starkInfo.openingPoints.size() + i)*FIELD_EXTENSION]), (Goldilocks3::Element &)(xisShifted[i * FIELD_EXTENSION]), k);
            for(uint64_t j = k+1; j < std::min(k + 4096, N); ++j) {
                uint64_t curr = (j*setupCtx.starkInfo.openingPoints.size() + i)*FIELD_EXTENSION;
                uint64_t prev = ((j-1)*setupCtx.starkInfo.openingPoints.size() + i)*FIELD_EXTENSION;
                Goldilocks3::mul((Goldilocks3::Element &)(LEv[curr]), (Goldilocks3::Element &)(LEv[prev]), (Goldilocks3::Element &)(xisShifted[i * FIELD_EXTENSION]));
            }
        }
    }

    ntt.INTT(&LEv[0], &LEv[0], N, FIELD_EXTENSION * setupCtx.starkInfo.openingPoints.size());
}


template <typename ElementType>
void Starks<ElementType>::computeEvals(StepsParams &params, Goldilocks::Element *LEv, FRIProof<ElementType> &proof)
{
    evmap(params, LEv);
}

template <typename ElementType>
void Starks<ElementType>::evmap(StepsParams& params, Goldilocks::Element *LEv)
{
    uint64_t extendBits = setupCtx.starkInfo.starkStruct.nBitsExt - setupCtx.starkInfo.starkStruct.nBits;
    u_int64_t size_eval = setupCtx.starkInfo.evMap.size();

    uint64_t N = 1 << setupCtx.starkInfo.starkStruct.nBits;

    int num_threads = omp_get_max_threads();
    int size_thread = size_eval * FIELD_EXTENSION;
    Goldilocks::Element *evals_acc = &params.aux_trace[setupCtx.starkInfo.mapOffsets[std::make_pair("evals", true)]];
    memset(&evals_acc[0], 0, num_threads * size_thread * sizeof(Goldilocks::Element));
    
    uint64_t dims[size_eval];
    uint64_t strides[size_eval];
    Goldilocks::Element *pointers[size_eval];

    for (uint64_t i = 0; i < size_eval; i++)
    {
        EvMap ev = setupCtx.starkInfo.evMap[i];
        string type = ev.type == EvMap::eType::cm ? "cm" : ev.type == EvMap::eType::custom ? "custom" : "fixed";
        Goldilocks::Element *pAddress = type == "cm" ? params.aux_trace : type == "custom" 
            ? params.pCustomCommitsFixed
            : &params.pConstPolsExtendedTreeAddress[2];
        PolMap polInfo = type == "cm" ? setupCtx.starkInfo.cmPolsMap[ev.id] : type == "custom" ? setupCtx.starkInfo.customCommitsMap[ev.commitId][ev.id] : setupCtx.starkInfo.constPolsMap[ev.id];
        dims[i] = polInfo.dim;
        std::string stage = type == "cm" ? "cm" + to_string(polInfo.stage) : type == "custom" ? setupCtx.starkInfo.customCommits[polInfo.commitId].name + "0" : "const";
        uint64_t nCols = setupCtx.starkInfo.mapSectionsN[stage];
        uint64_t offset = setupCtx.starkInfo.mapOffsets[std::make_pair(stage, true)] + polInfo.stagePos;
        pointers[i] = &pAddress[offset];
        strides[i] = nCols;
    }

#pragma omp parallel
    {
        int thread_idx = omp_get_thread_num();
        Goldilocks::Element *evals_acc_thread = &evals_acc[thread_idx * size_thread];
#pragma omp for
        for (uint64_t k = 0; k < N; k++)
        {
            Goldilocks3::Element LEv_[setupCtx.starkInfo.openingPoints.size()];
            for(uint64_t o = 0; o < setupCtx.starkInfo.openingPoints.size(); o++) {
                uint64_t pos = (o + k*setupCtx.starkInfo.openingPoints.size()) * FIELD_EXTENSION;
                LEv_[o][0] = LEv[pos];
                LEv_[o][1] = LEv[pos + 1];
                LEv_[o][2] = LEv[pos + 2];
            }
            uint64_t row = (k << extendBits);
            for (uint64_t i = 0; i < size_eval; i++)
            {
                EvMap ev = setupCtx.starkInfo.evMap[i];
                Goldilocks3::Element res;
                if (dims[i] == 1) {
                    Goldilocks3::mul(res, LEv_[ev.openingPos], pointers[i][row*strides[i]]);
                } else {
                    Goldilocks3::mul(res, LEv_[ev.openingPos], (Goldilocks3::Element &)(pointers[i][row*strides[i]]));
                }
                Goldilocks3::add((Goldilocks3::Element &)(evals_acc_thread[i * FIELD_EXTENSION]), (Goldilocks3::Element &)(evals_acc_thread[i * FIELD_EXTENSION]), res);
            }
        }
#pragma omp for
        for (uint64_t i = 0; i < size_eval; ++i)
        {
            Goldilocks3::Element sum = { Goldilocks::zero(), Goldilocks::zero(), Goldilocks::zero() };
            for (int k = 0; k < num_threads; ++k)
            {
                Goldilocks3::add(sum, sum, (Goldilocks3::Element &)(evals_acc[k * size_thread + i * FIELD_EXTENSION]));
            }
            std::memcpy((Goldilocks3::Element &)(params.evals[i * FIELD_EXTENSION]), sum, FIELD_EXTENSION * sizeof(Goldilocks::Element));
        }
    }
}

template <typename ElementType>
void Starks<ElementType>::getChallenge(TranscriptType &transcript, Goldilocks::Element &challenge)
{
    transcript.getField((uint64_t *)&challenge);
}

template <typename ElementType>
void Starks<ElementType>::calculateHash(ElementType* hash, Goldilocks::Element* buffer, uint64_t nElements) {
    TranscriptType transcriptHash(setupCtx.starkInfo.starkStruct.merkleTreeArity, setupCtx.starkInfo.starkStruct.merkleTreeCustom);
    transcriptHash.put(buffer, nElements);
    transcriptHash.getState(hash);
};

template <typename ElementType>
void Starks<ElementType>::addTranscriptGL(TranscriptType &transcript, Goldilocks::Element *buffer, uint64_t nElements)
{
    transcript.put(buffer, nElements);
};

template <typename ElementType>
void Starks<ElementType>::addTranscript(TranscriptType &transcript, ElementType *buffer, uint64_t nElements)
{
    transcript.put(buffer, nElements);
};

template <typename ElementType>
void Starks<ElementType>::ffi_treesGL_get_root(uint64_t index, ElementType *dst)
{
    treesGL[index]->getRoot(dst);
}

template <typename ElementType>
void Starks<ElementType>::calculateImPolsExpressions(uint64_t step, StepsParams &params, ExpressionsCtx &expressionsCtx) {
    uint64_t domainSize = (1 << setupCtx.starkInfo.starkStruct.nBits);
    std::vector<Dest> dests;
    for(uint64_t i = 0; i < setupCtx.starkInfo.cmPolsMap.size(); i++) {
        if(setupCtx.starkInfo.cmPolsMap[i].imPol && setupCtx.starkInfo.cmPolsMap[i].stage == step) {
            Goldilocks::Element* pAddress = setupCtx.starkInfo.cmPolsMap[i].stage == 1 ? params.trace : params.aux_trace;
            Dest destStruct(&pAddress[setupCtx.starkInfo.mapOffsets[std::make_pair("cm" + to_string(step), false)] + setupCtx.starkInfo.cmPolsMap[i].stagePos], domainSize, setupCtx.starkInfo.mapSectionsN["cm" + to_string(step)]);
            destStruct.addParams(setupCtx.starkInfo.cmPolsMap[i].expId, setupCtx.starkInfo.cmPolsMap[i].dim, false);
            
            expressionsCtx.calculateExpressions(params, destStruct, domainSize, false, false);
        }
    }
}

template <typename ElementType>
void Starks<ElementType>::calculateQuotientPolynomial(StepsParams &params, ExpressionsCtx &expressionsCtx) {
    expressionsCtx.calculateExpression(params, &params.aux_trace[setupCtx.starkInfo.mapOffsets[std::make_pair("q", true)]], setupCtx.starkInfo.cExpId);
}

template <typename ElementType>
void Starks<ElementType>::calculateFRIPolynomial(StepsParams &params, ExpressionsCtx &expressionsCtx) {
uint64_t xiChallengeIndex = 0;
    for (uint64_t i = 0; i < setupCtx.starkInfo.challengesMap.size(); i++)
    {
        if(setupCtx.starkInfo.challengesMap[i].stage == setupCtx.starkInfo.nStages + 2) {
            if(setupCtx.starkInfo.challengesMap[i].stageId == 0) xiChallengeIndex = i;
        }
    }

    Goldilocks::Element *xiChallenge = &params.challenges[xiChallengeIndex * FIELD_EXTENSION];
    
    Goldilocks::Element xis[setupCtx.starkInfo.openingPoints.size() * FIELD_EXTENSION];
    for (uint64_t i = 0; i < setupCtx.starkInfo.openingPoints.size(); ++i)
    {
        uint64_t openingAbs = setupCtx.starkInfo.openingPoints[i] < 0 ? -setupCtx.starkInfo.openingPoints[i] : setupCtx.starkInfo.openingPoints[i];
        Goldilocks::Element w = Goldilocks::pow(Goldilocks::w(setupCtx.starkInfo.starkStruct.nBits), openingAbs);

        if (setupCtx.starkInfo.openingPoints[i] < 0) w = Goldilocks::inv(w);

        Goldilocks3::mul((Goldilocks3::Element &)(xis[i * FIELD_EXTENSION]), (Goldilocks3::Element &)xiChallenge[0], w);
    }

    expressionsCtx.setXi(xis);

    expressionsCtx.calculateExpression(params, &params.aux_trace[setupCtx.starkInfo.mapOffsets[std::make_pair("f", true)]], setupCtx.starkInfo.friExpId);

    for(uint64_t step = 0; step < setupCtx.starkInfo.starkStruct.steps.size() - 1; ++step) { 
        Goldilocks::Element *src = &params.aux_trace[setupCtx.starkInfo.mapOffsets[std::make_pair("fri_" + to_string(step + 1), true)]];
        treesFRI[step]->setSource(src);

        if(setupCtx.starkInfo.starkStruct.verificationHashType == "GL") {
            Goldilocks::Element *pBuffNodesGL = &params.aux_trace[setupCtx.starkInfo.mapOffsets[std::make_pair("mt_fri_" + to_string(step + 1), true)]];
            ElementType *pBuffNodes = (ElementType *)pBuffNodesGL;
            treesFRI[step]->setNodes(pBuffNodes);
        }
    }
}