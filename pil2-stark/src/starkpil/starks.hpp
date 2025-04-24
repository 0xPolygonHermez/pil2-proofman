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

class gl64_t;
struct DeviceCommitBuffers;

template <typename ElementType>
class Starks
{
public:
    SetupCtx& setupCtx;
    ProverHelpers& proverHelpers;
    using TranscriptType = std::conditional_t<std::is_same<ElementType, Goldilocks::Element>::value, TranscriptGL, TranscriptBN128>;
    using MerkleTreeType = std::conditional_t<std::is_same<ElementType, Goldilocks::Element>::value, MerkleTreeGL, MerkleTreeBN128>;

    MerkleTreeType **treesGL;
    MerkleTreeType **treesFRI;

    NTT_Goldilocks ntt;
    NTT_Goldilocks nttExtended;

public:
    Starks(SetupCtx& setupCtx_, ProverHelpers& proverHelpers_, Goldilocks::Element *pConstPolsExtendedTreeAddress, Goldilocks::Element *pConstPolsCustomCommitsTree = nullptr, bool initializeTrees = false) : setupCtx(setupCtx_), proverHelpers(proverHelpers_), ntt(1 << setupCtx.starkInfo.starkStruct.nBits), nttExtended(1 << setupCtx.starkInfo.starkStruct.nBitsExt)                     
    {

        uint64_t N = 1 << setupCtx.starkInfo.starkStruct.nBits;
        uint64_t NExtended = 1 << setupCtx.starkInfo.starkStruct.nBitsExt;

        bool allocateNodes = setupCtx.starkInfo.starkStruct.verificationHashType == "GL" ? false : true;
        treesGL = new MerkleTreeType*[setupCtx.starkInfo.nStages + setupCtx.starkInfo.customCommits.size() + 2];
        if (pConstPolsExtendedTreeAddress != nullptr) treesGL[setupCtx.starkInfo.nStages + 1] = new MerkleTreeType(setupCtx.starkInfo.starkStruct.merkleTreeArity, setupCtx.starkInfo.starkStruct.merkleTreeCustom, pConstPolsExtendedTreeAddress);
        for (uint64_t i = 0; i < setupCtx.starkInfo.nStages + 1; i++)
        {
            std::string section = "cm" + to_string(i + 1);
            uint64_t nCols = setupCtx.starkInfo.mapSectionsN[section];
            treesGL[i] = new MerkleTreeType(setupCtx.starkInfo.starkStruct.merkleTreeArity, setupCtx.starkInfo.starkStruct.merkleTreeCustom, NExtended, nCols, initializeTrees, allocateNodes || initializeTrees);
        }

        

        for(uint64_t i = 0; i < setupCtx.starkInfo.customCommits.size(); i++) {
            uint64_t nCols = setupCtx.starkInfo.mapSectionsN[setupCtx.starkInfo.customCommits[i].name + "0"];
            treesGL[setupCtx.starkInfo.nStages + 2 + i] = new MerkleTreeType(setupCtx.starkInfo.starkStruct.merkleTreeArity, setupCtx.starkInfo.starkStruct.merkleTreeCustom, NExtended, nCols);
            treesGL[setupCtx.starkInfo.nStages + 2 + i]->setSource(&pConstPolsCustomCommitsTree[N * nCols]);
            ElementType *nodes = (ElementType *)&pConstPolsCustomCommitsTree[(N + NExtended) * nCols];
            treesGL[setupCtx.starkInfo.nStages + 2 + i]->setNodes(nodes);
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
    
    void extendAndMerkelizeCustomCommit(uint64_t commitId, uint64_t step, Goldilocks::Element *buffer, FRIProof<ElementType> &proof, Goldilocks::Element *pBuffHelper);
    void extendAndMerkelize(uint64_t step, Goldilocks::Element *trace, Goldilocks::Element *buffer, FRIProof<ElementType> &proof, Goldilocks::Element* pBuffHelper = nullptr);
    void extendAndMerkelize_inplace(uint64_t step, gl64_t *d_witness, gl64_t* d_aux_trace, double *nttTime = nullptr, double *merkleTime=nullptr);

    void commitStage(uint64_t step, Goldilocks::Element *trace, Goldilocks::Element *buffer, FRIProof<ElementType> &proof, Goldilocks::Element* pBuffHelper = nullptr);
    void commitStage_inplace(uint64_t step, gl64_t *d_witness, gl64_t *d_trace, double *nttTime=nullptr, double *merkleTime=nullptr);
    void computeQ(uint64_t step, Goldilocks::Element *buffer, FRIProof<ElementType> &proof, Goldilocks::Element* pBuffHelper = nullptr);
    void computeQ_inplace(uint64_t step, gl64_t *d_aux_trace, double *nttTime=nullptr, double *merkleTime=nullptr);
    
    void calculateImPolsExpressions(uint64_t step, StepsParams& params, ExpressionsCtx& expressionsCtx);
    void calculateQuotientPolynomial(StepsParams& params, ExpressionsCtx& expressionsCtx);
    void calculateFRIPolynomial(StepsParams& params, ExpressionsCtx& expressionsCtx);

    void computeLEv(Goldilocks::Element *xiChallenge, Goldilocks::Element *LEv, std::vector<int64_t> &openingPoints);
    void computeEvals(StepsParams &params, Goldilocks::Element *LEv, FRIProof<ElementType> &proof, std::vector<int64_t> &openingPoints);

    void calculateHash(ElementType* hash, Goldilocks::Element* buffer, uint64_t nElements);

    void addTranscriptGL(TranscriptType &transcript, Goldilocks::Element* buffer, uint64_t nElements);
    void addTranscript(TranscriptType &transcript, ElementType* buffer, uint64_t nElements);
    void getChallenge(TranscriptType &transcript, Goldilocks::Element& challenge);

    // Following function are created to be used by the ffi interface
    void ffi_treesGL_get_root(uint64_t index, ElementType *dst);

    void evmap(StepsParams& params, Goldilocks::Element *LEv, std::vector<int64_t> &openingPoints);
};

template class Starks<Goldilocks::Element>;
template class Starks<RawFr::Element>;

#endif // STARKS_H
