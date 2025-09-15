#ifndef STARKS_HPP
#define STARKS_HPP

#include <algorithm>
#include <cmath>
#include "utils.hpp"
#include "timer.hpp"
#include "const_pols.hpp"
#include "proof_stark.hpp"
#include "fri.hpp"
#include "transcriptGL.hpp"
#include "steps.hpp"
#include "zklog.hpp"
#include "merkleTreeBN128.hpp"
#include "transcriptBN128.hpp"
#include "exit_process.hpp"
#include "expressions_bin.hpp"
#include "expressions_pack.hpp"
#include "expressions_pack_q.hpp"
#include "hints.hpp"

class gl64_t;
struct DeviceCommitBuffers;

template <typename ElementType>
class Starks
{
public:
    SetupCtx& setupCtx;
    using TranscriptType = std::conditional_t<std::is_same<ElementType, Goldilocks::Element>::value, TranscriptGL, TranscriptBN128>;
    using MerkleTreeType = std::conditional_t<std::is_same<ElementType, Goldilocks::Element>::value, MerkleTreeGL, MerkleTreeBN128>;

    MerkleTreeType **treesGL;
    MerkleTreeType **treesFRI;

public:
    Starks(SetupCtx& setupCtx_,Goldilocks::Element *pConstPolsExtendedTreeAddress, Goldilocks::Element *pConstPolsCustomCommitsTree = nullptr, bool initializeTrees = false) : setupCtx(setupCtx_)                    
    {

        uint64_t N = 1 << setupCtx.starkInfo.starkStruct.nBits;
        uint64_t NExtended = 1 << setupCtx.starkInfo.starkStruct.nBitsExt;

        bool allocateNodes = setupCtx.starkInfo.starkStruct.verificationHashType == "GL" ? false : true;
        treesGL = new MerkleTreeType*[setupCtx.starkInfo.nStages + setupCtx.starkInfo.customCommits.size() + 2];
        if (pConstPolsExtendedTreeAddress != nullptr) {
            treesGL[setupCtx.starkInfo.nStages + 1] = new MerkleTreeType(setupCtx.starkInfo.starkStruct.merkleTreeArity, setupCtx.starkInfo.starkStruct.merkleTreeCustom, pConstPolsExtendedTreeAddress);
        } else {
            treesGL[setupCtx.starkInfo.nStages + 1] = new MerkleTreeType(setupCtx.starkInfo.starkStruct.merkleTreeArity, setupCtx.starkInfo.starkStruct.merkleTreeCustom, NExtended, setupCtx.starkInfo.nConstants, initializeTrees, allocateNodes || initializeTrees);
        }
        for (uint64_t i = 0; i < setupCtx.starkInfo.nStages + 1; i++)
        {
            std::string section = "cm" + to_string(i + 1);
            uint64_t nCols = setupCtx.starkInfo.mapSectionsN[section];
            treesGL[i] = new MerkleTreeType(setupCtx.starkInfo.starkStruct.merkleTreeArity, setupCtx.starkInfo.starkStruct.merkleTreeCustom, NExtended, nCols, initializeTrees, allocateNodes || initializeTrees);
        }

        for(uint64_t i = 0; i < setupCtx.starkInfo.customCommits.size(); i++) {
            uint64_t nCols = setupCtx.starkInfo.mapSectionsN[setupCtx.starkInfo.customCommits[i].name + "0"];
            treesGL[setupCtx.starkInfo.nStages + 2 + i] = new MerkleTreeType(setupCtx.starkInfo.starkStruct.merkleTreeArity, setupCtx.starkInfo.starkStruct.merkleTreeCustom, NExtended, nCols);
            if (pConstPolsCustomCommitsTree != nullptr) {        
                treesGL[setupCtx.starkInfo.nStages + 2 + i]->setSource(&pConstPolsCustomCommitsTree[N * nCols]);
                ElementType *nodes = (ElementType *)&pConstPolsCustomCommitsTree[(N + NExtended) * nCols];
                treesGL[setupCtx.starkInfo.nStages + 2 + i]->setNodes(nodes);
            }
        }

        treesFRI = new MerkleTreeType*[setupCtx.starkInfo.starkStruct.steps.size() - 1];
        for(uint64_t step = 0; step < setupCtx.starkInfo.starkStruct.steps.size() - 1; ++step) {
            uint64_t nGroups = 1 << setupCtx.starkInfo.starkStruct.steps[step + 1].nBits;
            uint64_t groupSize = (1 << setupCtx.starkInfo.starkStruct.steps[step].nBits) / nGroups;

            treesFRI[step] = new MerkleTreeType(setupCtx.starkInfo.starkStruct.merkleTreeArity, setupCtx.starkInfo.starkStruct.merkleTreeCustom, nGroups, groupSize * FIELD_EXTENSION, initializeTrees, allocateNodes || initializeTrees);
        }
    };
    ~Starks()
    {
        for (uint i = 0; i < setupCtx.starkInfo.nStages + setupCtx.starkInfo.customCommits.size() + 2; i++)
        {
            delete treesGL[i];
        }
        delete[] treesGL;

        for (uint64_t i = 0; i < setupCtx.starkInfo.starkStruct.steps.size() - 1; i++)
        {
            delete treesFRI[i];
        }
        delete[] treesFRI;
    };
    
    void extendAndMerkelizeCustomCommit(uint64_t commitId, uint64_t step, Goldilocks::Element *buffer, FRIProof<ElementType> &proof, Goldilocks::Element *pBuffHelper, NTT_Goldilocks &ntt);
    void extendAndMerkelize(uint64_t step, Goldilocks::Element *trace, Goldilocks::Element *buffer, FRIProof<ElementType> &proof, NTT_Goldilocks &ntt, Goldilocks::Element* pBuffHelper = nullptr);

    void commitStage(uint64_t step, Goldilocks::Element *trace, Goldilocks::Element *buffer, FRIProof<ElementType> &proof, NTT_Goldilocks &ntt, Goldilocks::Element* pBuffHelper = nullptr);
    void computeQ(uint64_t step, Goldilocks::Element *buffer, FRIProof<ElementType> &proof, NTT_Goldilocks &nttExtended, Goldilocks::Element* pBuffHelper = nullptr);
    
    void calculateImPolsExpressions(uint64_t step, StepsParams& params, ExpressionsCtx& expressionsCtx);
    void calculateQuotientPolynomialOut(StepsParams& params, ExpressionsCtx& expressionsCtx);
    void calculateQuotientPolynomialIn(StepsParams& params, ExpressionsCtx& expressionsCtx);
    void calculateQuotientPolynomial(StepsParams& params, ExpressionsCtx& expressionsCtx);
    void calculateFRIPolynomial(StepsParams& params, ExpressionsCtx& expressionsCtx);

    void computeLEv(Goldilocks::Element *xiChallenge, Goldilocks::Element *LEv, std::vector<int64_t> &openingPoints, NTT_Goldilocks &ntt);
    void computeEvals(StepsParams &params, Goldilocks::Element *LEv, FRIProof<ElementType> &proof, std::vector<int64_t> &openingPoints);

    void calculateHash(ElementType* hash, Goldilocks::Element* buffer, uint64_t nElements);

    void addTranscriptGL(TranscriptType &transcript, Goldilocks::Element* buffer, uint64_t nElements);
    void addTranscript(TranscriptType &transcript, ElementType* buffer, uint64_t nElements);
    void getChallenge(TranscriptType &transcript, Goldilocks::Element& challenge);

    // Following function are created to be used by the ffi interface
    void ffi_treesGL_get_root(uint64_t index, ElementType *dst);

    void evmap(StepsParams& params, Goldilocks::Element *LEv, std::vector<int64_t> &openingPoints);
};

template <typename ElementType>
void Starks<ElementType>::extendAndMerkelizeCustomCommit(uint64_t commitId, uint64_t step, Goldilocks::Element *buffer, FRIProof<ElementType> &proof, Goldilocks::Element *pBuffHelper, NTT_Goldilocks &ntt)
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
void Starks<ElementType>::extendAndMerkelize(uint64_t step, Goldilocks::Element *trace, Goldilocks::Element *aux_trace, FRIProof<ElementType> &proof, NTT_Goldilocks &ntt, Goldilocks::Element *pBuffHelper)
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
void Starks<ElementType>::commitStage(uint64_t step, Goldilocks::Element *trace, Goldilocks::Element *aux_trace, FRIProof<ElementType> &proof,  NTT_Goldilocks &ntt, Goldilocks::Element* pBuffHelper)
{  

    if (step <= setupCtx.starkInfo.nStages)
    {
        extendAndMerkelize(step, trace, aux_trace, proof, ntt, pBuffHelper);
    }
    else
    {
        computeQ(step, aux_trace, proof, ntt, pBuffHelper);
    }
}

template <typename ElementType>
void Starks<ElementType>::computeQ(uint64_t step, Goldilocks::Element *buffer, FRIProof<ElementType> &proof, NTT_Goldilocks &nttExtended, Goldilocks::Element *pBuffHelper)
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
void Starks<ElementType>::computeLEv(Goldilocks::Element *xiChallenge, Goldilocks::Element *LEv, std::vector<int64_t> &openingPoints, NTT_Goldilocks &ntt) {
    uint64_t N = 1 << setupCtx.starkInfo.starkStruct.nBits;
        
    Goldilocks::Element xis[openingPoints.size() * FIELD_EXTENSION];
    Goldilocks::Element xisShifted[openingPoints.size() * FIELD_EXTENSION];
    
    Goldilocks::Element shift_inv = Goldilocks::inv(Goldilocks::shift());
        for (uint64_t i = 0; i < openingPoints.size(); ++i)
    {
        uint64_t openingAbs = openingPoints[i] < 0 ? -openingPoints[i] : openingPoints[i];
        Goldilocks::Element w = Goldilocks::pow(Goldilocks::w(setupCtx.starkInfo.starkStruct.nBits), openingAbs);

        if (openingPoints[i] < 0)
        {
            w = Goldilocks::inv(w);
        }

        Goldilocks3::mul((Goldilocks3::Element &)(xis[i * FIELD_EXTENSION]), (Goldilocks3::Element &)xiChallenge[0], w);
        Goldilocks3::mul((Goldilocks3::Element &)(xisShifted[i * FIELD_EXTENSION]), (Goldilocks3::Element &)(xis[i * FIELD_EXTENSION]), shift_inv);
    }

    #pragma omp parallel for collapse(2)
    for (uint64_t k = 0; k < N; k+=4096)
    {
        for (uint64_t i = 0; i < openingPoints.size(); ++i)
        {
            Goldilocks3::pow((Goldilocks3::Element &)(LEv[(k*openingPoints.size() + i)*FIELD_EXTENSION]), (Goldilocks3::Element &)(xisShifted[i * FIELD_EXTENSION]), k);
            for(uint64_t j = k+1; j < std::min(k + 4096, N); ++j) {
                uint64_t curr = (j*openingPoints.size() + i)*FIELD_EXTENSION;
                uint64_t prev = ((j-1)*openingPoints.size() + i)*FIELD_EXTENSION;
                Goldilocks3::mul((Goldilocks3::Element &)(LEv[curr]), (Goldilocks3::Element &)(LEv[prev]), (Goldilocks3::Element &)(xisShifted[i * FIELD_EXTENSION]));
            }
        }
    }

    ntt.INTT(&LEv[0], &LEv[0], N, FIELD_EXTENSION * openingPoints.size());
}


template <typename ElementType>
void Starks<ElementType>::computeEvals(StepsParams &params, Goldilocks::Element *LEv, FRIProof<ElementType> &proof, std::vector<int64_t> &openingPoints)
{
    evmap(params, LEv, openingPoints);
}

template <typename ElementType>
void Starks<ElementType>::evmap(StepsParams& params, Goldilocks::Element *LEv, std::vector<int64_t> &openingPoints)
{
    uint64_t extendBits = setupCtx.starkInfo.starkStruct.nBitsExt - setupCtx.starkInfo.starkStruct.nBits;
    u_int64_t size_eval = setupCtx.starkInfo.evMap.size();

    uint64_t N = 1 << setupCtx.starkInfo.starkStruct.nBits;
    
    uint64_t dims[size_eval];
    uint64_t strides[size_eval];
    uint64_t openingPos[size_eval];
    Goldilocks::Element *pointers[size_eval];
    std::vector<uint64_t> evalsToCalculate;
    uint64_t nEvals = 0;
    for (uint64_t i = 0; i < size_eval; i++)
    {
        EvMap ev = setupCtx.starkInfo.evMap[i];
        auto it = std::find(openingPoints.begin(), openingPoints.end(), ev.prime);
        bool containsPrime = (it != openingPoints.end());
        if(!containsPrime) continue;
        string type = ev.type == EvMap::eType::cm ? "cm" : ev.type == EvMap::eType::custom ? "custom" : "fixed";
        Goldilocks::Element *pAddress = type == "cm" ? params.aux_trace : type == "custom"
            ? params.pCustomCommitsFixed
            : &params.pConstPolsExtendedTreeAddress[2];
        PolMap polInfo = type == "cm" ? setupCtx.starkInfo.cmPolsMap[ev.id] : type == "custom" ? setupCtx.starkInfo.customCommitsMap[ev.commitId][ev.id] : setupCtx.starkInfo.constPolsMap[ev.id];
        dims[nEvals] = polInfo.dim;
        std::string stage = type == "cm" ? "cm" + to_string(polInfo.stage) : type == "custom" ? setupCtx.starkInfo.customCommits[polInfo.commitId].name + "0" : "const";
        uint64_t nCols = setupCtx.starkInfo.mapSectionsN[stage];
        uint64_t offset = setupCtx.starkInfo.mapOffsets[std::make_pair(stage, true)] + polInfo.stagePos;
        pointers[nEvals] = &pAddress[offset];
        strides[nEvals] = nCols;
        openingPos[nEvals] = std::distance(openingPoints.begin(), it);
        evalsToCalculate.push_back(i);
        nEvals++;
    }

    int num_threads = omp_get_max_threads();
    int size_thread = nEvals * FIELD_EXTENSION;
    Goldilocks::Element *evals_acc = &params.aux_trace[setupCtx.starkInfo.mapOffsets[std::make_pair("evals", true)]];
    memset(&evals_acc[0], 0, omp_get_max_threads() * size_eval * FIELD_EXTENSION * sizeof(Goldilocks::Element));

#pragma omp parallel
    {
        int thread_idx = omp_get_thread_num();
        Goldilocks::Element *evals_acc_thread = &evals_acc[thread_idx * size_thread];
#pragma omp for
        for (uint64_t k = 0; k < N; k++)
        {
            Goldilocks3::Element LEv_[openingPoints.size()];
            for(uint64_t o = 0; o < openingPoints.size(); o++) {
                uint64_t pos = (o + k*openingPoints.size()) * FIELD_EXTENSION;
                LEv_[o][0] = LEv[pos];
                LEv_[o][1] = LEv[pos + 1];
                LEv_[o][2] = LEv[pos + 2];
            }
            uint64_t row = (k << extendBits);
            for (uint64_t i = 0; i < nEvals; i++)
            {
                Goldilocks3::Element res;
                if (dims[i] == 1) {
                    Goldilocks3::mul(res, LEv_[openingPos[i]], pointers[i][row*strides[i]]);
                } else {
                    Goldilocks3::mul(res, LEv_[openingPos[i]], (Goldilocks3::Element &)(pointers[i][row*strides[i]]));
                }
                Goldilocks3::add((Goldilocks3::Element &)(evals_acc_thread[i * FIELD_EXTENSION]), (Goldilocks3::Element &)(evals_acc_thread[i * FIELD_EXTENSION]), res);
            }
        }
#pragma omp for
        for (uint64_t i = 0; i < nEvals; ++i)
        {
            Goldilocks3::Element sum = { Goldilocks::zero(), Goldilocks::zero(), Goldilocks::zero() };
            for (int k = 0; k < num_threads; ++k)
            {
                Goldilocks3::add(sum, sum, (Goldilocks3::Element &)(evals_acc[k * size_thread + i * FIELD_EXTENSION]));
            }
            std::memcpy((Goldilocks3::Element &)(params.evals[evalsToCalculate[i] * FIELD_EXTENSION]), sum, FIELD_EXTENSION * sizeof(Goldilocks::Element));
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
void Starks<ElementType>::calculateQuotientPolynomialOut(StepsParams &params, ExpressionsCtx &expressionsCtx) {

    uint64_t NExtended = 1 << setupCtx.starkInfo.starkStruct.nBitsExt;

    Goldilocks::Element *tmpQ3;
    Goldilocks::Element *tmpQ1;
    tmpQ3 = new Goldilocks::Element[3 * NExtended];
    tmpQ1 = new Goldilocks::Element[NExtended];

    Goldilocks::Element *Q = &params.aux_trace[setupCtx.starkInfo.mapOffsets[std::make_pair("q", true)]];
    memset(Q, 0, 3 * NExtended * sizeof(Goldilocks::Element));

    Goldilocks::Element *zi = expressionsCtx.proverHelpers->zi;

    Goldilocks3::Element zi_challenge = {params.challenges[2*FIELD_EXTENSION], params.challenges[2*FIELD_EXTENSION + 1], params.challenges[2*FIELD_EXTENSION + 2]};
    Goldilocks3::Element acc_challenge ={Goldilocks::one(), Goldilocks::zero(), Goldilocks::zero()};

    //for (uint64_t i = 0; i < setupCtx.expressionsBin.constraintsInfoDebug.size(); i++) {
    for (int i = setupCtx.expressionsBin.constraintsInfoDebug.size()-1; i >=0; i--) {
        memset(tmpQ3, 0, 3 * NExtended * sizeof(Goldilocks::Element));
        memset(tmpQ1, 0, NExtended * sizeof(Goldilocks::Element));
        uint64_t destDim = setupCtx.expressionsBin.constraintsInfoDebug[i].destDim;

        //calculate constraint
        Goldilocks::Element* opQ = destDim == 1 ? tmpQ1 : tmpQ3;
        Dest constraintDest(opQ, NExtended, 0, i);
        constraintDest.addParams(i, destDim);
        expressionsCtx.calculateExpressions(params, constraintDest, NExtended, true, false, true);
        // add contribution
        for(uint64_t j = 0; j <  NExtended; j++) {
            //tmQ*=challenge
            if (destDim == 1) {
                Goldilocks3::mul((Goldilocks3::Element &)(tmpQ3[j * FIELD_EXTENSION]), (Goldilocks3::Element &)acc_challenge, tmpQ1[j]);
            } else {
                Goldilocks3::mul((Goldilocks3::Element &)(tmpQ3[j * FIELD_EXTENSION]), (Goldilocks3::Element &)acc_challenge, (Goldilocks3::Element &)(tmpQ3[j * FIELD_EXTENSION]));
            }
            
            //Q+=tmpQ3
            Goldilocks3::add((Goldilocks3::Element &)(Q[j * FIELD_EXTENSION]), (Goldilocks3::Element &)(Q[j * FIELD_EXTENSION]), (Goldilocks3::Element &)(tmpQ3[j * FIELD_EXTENSION]));
        }
        //challenge*=initial_challenge
        Goldilocks3::mul(acc_challenge, acc_challenge, zi_challenge); 
    }

    for(uint64_t i = 0; i <  NExtended; i++) {
        //Q *= invZ
        Goldilocks3::mul((Goldilocks3::Element &)(Q[i * FIELD_EXTENSION]), (Goldilocks3::Element &)(Q[i * FIELD_EXTENSION]), zi[i]);
    }
    delete[] tmpQ3;
    delete[] tmpQ1;
}

template <typename ElementType>
void Starks<ElementType>::calculateQuotientPolynomial(StepsParams &params, ExpressionsCtx &expressionsCtx) {
    expressionsCtx.calculateExpression(params, &params.aux_trace[setupCtx.starkInfo.mapOffsets[std::make_pair("q", true)]], setupCtx.starkInfo.cExpId, false, false, true);
}
template <typename ElementType>
void Starks<ElementType>::calculateQuotientPolynomialIn(StepsParams &params, ExpressionsCtx &expressionsCtx) {

    uint64_t domainSize = 1 << setupCtx.starkInfo.starkStruct.nBitsExt;
    bool domainExtended = true;
    Goldilocks::Element *dest = &params.aux_trace[setupCtx.starkInfo.mapOffsets[std::make_pair("q", domainExtended)]];

    Dest destStruct(dest, domainSize, 0,-1);
    for (int i = setupCtx.expressionsBin.constraintsInfoDebug.size()-1; i >=0; i--) {
        destStruct.addParams(i, setupCtx.expressionsBin.constraintsInfoDebug[i].destDim);
    }
    expressionsCtx.calculateExpressions(params, destStruct, domainSize, domainExtended, false, true);
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


#endif // STARKS_H
