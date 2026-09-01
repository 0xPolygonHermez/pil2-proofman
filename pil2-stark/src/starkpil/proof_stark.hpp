#ifndef PROOF
#define PROOF

#include "goldilocks_base_field.hpp"
#include "goldilocks_cubic_extension.hpp"
#include "poseidon2_goldilocks.hpp"
#include "stark_info.hpp"
#include "fr.hpp"
#include <vector>
#include "nlohmann/json.hpp"

using json = nlohmann::json;

template <typename ElementType>
std::string toString(const ElementType& element);

template <typename ElementType>
uint64_t toU64(const ElementType& element);

template<>
inline uint64_t toU64(const Goldilocks::Element& element) {
    return Goldilocks::toU64(element);
}

template<>
inline uint64_t toU64(const RawFr::Element& element) {
    throw std::runtime_error("Error: Cannot convert RawFr::Element to U64.");
}

template<>
inline std::string toString(const Goldilocks::Element& element) {
    return Goldilocks::toString(element);
}

template<>
inline std::string toString(const RawFr::Element& element) {
    return RawFr::field.toString(element, 10);
}

template <typename ElementType>
class MerkleProof
{
public:
    std::vector<std::vector<Goldilocks::Element>> v;
    std::vector<std::vector<ElementType>> mp;

    MerkleProof(uint64_t nLinears, uint64_t elementsTree, uint64_t numSiblings, void *pointer) : v(nLinears, std::vector<Goldilocks::Element>(1, Goldilocks::zero())), mp(elementsTree, std::vector<ElementType>(numSiblings))
    {
        for (uint64_t i = 0; i < nLinears; i++)
        {
            std::memcpy(&v[i][0], &((Goldilocks::Element *)pointer)[i], sizeof(Goldilocks::Element));
        }
        ElementType *mpCursor = (ElementType *)&((Goldilocks::Element *)pointer)[nLinears];
        for (uint64_t j = 0; j < elementsTree; j++)
        {
            std::memcpy(&mp[j][0], &mpCursor[j * numSiblings], numSiblings * sizeof(ElementType));
        }
    }
    MerkleProof(uint64_t nLinears, uint64_t elementsTree, uint64_t numSiblings, void *pointer, uint64_t offsetTree) : v(nLinears, std::vector<Goldilocks::Element>(1, Goldilocks::zero())), mp(elementsTree, std::vector<ElementType>(numSiblings))
    {
        for (uint64_t i = 0; i < nLinears; i++)
        {
            std::memcpy(&v[i][0], &((Goldilocks::Element *)pointer)[i], sizeof(Goldilocks::Element));
        }
        ElementType *mpCursor = (ElementType *)&((Goldilocks::Element *)pointer)[offsetTree];
        for (uint64_t j = 0; j < elementsTree; j++)
        {
            std::memcpy(&mp[j][0], &mpCursor[j * numSiblings], numSiblings * sizeof(ElementType));
        }
    }
};

template <typename ElementType>
class ProofTree
{
public:
    std::vector<ElementType> root;
    std::vector<ElementType> last_levels;
    std::vector<std::vector<MerkleProof<ElementType>>> polQueries;

    uint64_t nFieldElements;
    uint64_t arity;
    uint64_t last_level;

    ProofTree(uint64_t nFieldElements_, uint64_t nQueries, uint64_t arity_, uint64_t lastLevel_) : root(nFieldElements_), last_levels(lastLevel_ == 0 ? 0 : nFieldElements_ * std::pow(arity_, lastLevel_)), polQueries(nQueries), nFieldElements(nFieldElements_), arity(arity_), last_level(lastLevel_) {}

    void setRoot(ElementType *_root)
    {
        std::memcpy(&root[0], &_root[0], nFieldElements * sizeof(ElementType));
    };

    void setLastLevels(ElementType *_last_level) 
    {
        if (last_level == 0) return;
        std::memcpy(&last_levels[0], &_last_level[0], nFieldElements * std::pow(arity, last_level) * sizeof(ElementType));
    }
};

template <typename ElementType>
class Fri
{
public:
    ProofTree<ElementType> trees;
    std::vector<ProofTree<ElementType>> treesFRI;
    std::vector<std::vector<Goldilocks::Element>> pol;
   

    // `trees` holds the openings of the stage, constant and custom trees at the queried rows of the
    // extended domain. Those are the same for every low-degree test so this member is shared; only 
    // `treesFRI` and `pol` are FRI's own.
    Fri(StarkInfo &starkInfo) :  trees((starkInfo.starkStruct.verificationHashType == "GL") ? HASH_SIZE : 1, starkInfo.starkStruct.nQueries, starkInfo.starkStruct.merkleTreeArity, starkInfo.starkStruct.lastLevelVerification),
                                 treesFRI(),
                                 pol() {
        if (starkInfo.starkStruct.lowDegreeTest != LowDegreeTestKind::FRI) return;

        pol.assign(1 << starkInfo.starkStruct.logDomainSizes[starkInfo.starkStruct.logDomainSizes.size() - 1], std::vector<Goldilocks::Element>(FIELD_EXTENSION, Goldilocks::zero()));

        uint64_t nQueries = starkInfo.starkStruct.nQueries;
        uint64_t nFieldElements = (starkInfo.starkStruct.verificationHashType == "GL") ? HASH_SIZE : 1;
       
        for (size_t i = 0; i < starkInfo.starkStruct.logDomainSizes.size() - 1; i++)
        {
            treesFRI.emplace_back(nFieldElements, nQueries, starkInfo.starkStruct.merkleTreeArity, starkInfo.starkStruct.lastLevelVerification);
        }
    }

    void setPol(Goldilocks::Element *pPol, uint64_t degree)
    {
        for (uint64_t i = 0; i < degree; i++)
        {
            std::memcpy(&pol[i][0], &pPol[i * FIELD_EXTENSION], FIELD_EXTENSION * sizeof(Goldilocks::Element));
        }
    }
};

// The STIR section of a proof (Arnon–Chiesa–Fenzi–Yogev, Construction 5.2), the counterpart of
// `Fri` above. The stage/constant/custom openings live in `Fri::trees`, shared by both tests.
//
//   trees[i]   commitment to the oracle of f_i, i = 0..M−1: T_0 commits f_0 and T_i commits g_i,
//              each in k_i-cosets. Its polQueries are the t_i openings made by iteration i+1.
//   betas[i−1] β_{i,1..s}, the out-of-domain answers of iteration i = 1..M−1.
//   nonces[i−1] grinding nonce of iteration i's query message, i = 1..M.
//   finalPol   p, the final polynomial in the clear, as its d_M coefficients (FRI sends its final
//              polynomial as evaluations instead, so this is a shorter and degree-bounded form).
template <typename ElementType>
class StirProof
{
public:
    std::vector<ProofTree<ElementType>> trees;
    std::vector<std::vector<Goldilocks::Element>> betas;
    std::vector<uint64_t> nonces;
    std::vector<Goldilocks::Element> finalPol;
    // Coefficients of Âns_i for the quotient rounds i = 1..M−1, zero-padded to s + t_{i−1}
    // (duplicate shift queries shrink |G_i|, so the true degree can be lower). Pure hints for
    // the recursion circuit, which constrains them itself; the native verifier recomputes Âns
    // and ignores these.
    std::vector<std::vector<Goldilocks::Element>> ansCoeffs;

    StirProof() {}

    StirProof(StarkInfo &starkInfo) {
        if (starkInfo.starkStruct.lowDegreeTest != LowDegreeTestKind::STIR) return;
        const StirStruct &stir = starkInfo.starkStruct.stir;
        init(stir.numQueries, stir.numOodSamples, stir.logDegrees[stir.numIterations()],
             starkInfo.starkStruct.merkleTreeArity, starkInfo.starkStruct.lastLevelVerification,
             (starkInfo.starkStruct.verificationHashType == "GL") ? HASH_SIZE : 1);
    }

    // Sized from the schedule alone, for callers that have no StarkInfo (tests).
    void init(const std::vector<uint64_t> &numQueries, uint64_t numOodSamples, uint64_t logFinalDegree, uint64_t arity, uint64_t lastLevelVerification, uint64_t nFieldElements) {
        uint64_t M = numQueries.size();
        trees.clear();
        for (uint64_t i = 0; i < M; i++)
        {
            trees.emplace_back(nFieldElements, numQueries[i], arity, lastLevelVerification);
        }
        betas.assign(M - 1, std::vector<Goldilocks::Element>(numOodSamples * FIELD_EXTENSION, Goldilocks::zero()));
        ansCoeffs.clear();
        for (uint64_t i = 1; i < M; i++)
        {
            ansCoeffs.emplace_back((numOodSamples + numQueries[i - 1]) * FIELD_EXTENSION, Goldilocks::zero());
        }
        nonces.assign(M, 0);
        finalPol.assign((uint64_t(1) << logFinalDegree) * FIELD_EXTENSION, Goldilocks::zero());
    }
};

template <typename ElementType>
class Proofs
{
public:
    StarkInfo &starkInfo;
    uint64_t nStages;
    uint64_t nCustomCommits;
    uint64_t nFieldElements;
    uint64_t lastLevelVerification;
    ElementType **roots;
    ElementType **last_levels;
    Fri<ElementType> fri;
    StirProof<ElementType> stir;
    std::vector<std::vector<Goldilocks::Element>> evals;
    std::vector<std::vector<Goldilocks::Element>> airgroupValues;
    std::vector<std::vector<Goldilocks::Element>> airValues;
    std::vector<std::string> customCommits;
    uint64_t nonce;
    Proofs(StarkInfo &starkInfo_) :
        starkInfo(starkInfo_),
        fri(starkInfo_),
        stir(starkInfo_),
        evals(starkInfo_.evMap.size(), std::vector<Goldilocks::Element>(FIELD_EXTENSION, Goldilocks::zero())),
        airgroupValues(starkInfo_.airgroupValuesMap.size(), std::vector<Goldilocks::Element>(FIELD_EXTENSION, Goldilocks::zero())),
        airValues(starkInfo_.airValuesMap.size(), std::vector<Goldilocks::Element>(FIELD_EXTENSION, Goldilocks::zero())),
        customCommits(starkInfo_.customCommits.size())
        {
            nStages = starkInfo_.nStages + 1;
            nCustomCommits = starkInfo_.customCommits.size();
            roots = new ElementType*[nStages + nCustomCommits];
            last_levels = new ElementType*[1 + nStages + nCustomCommits];
            lastLevelVerification = starkInfo_.starkStruct.lastLevelVerification;
            nFieldElements = starkInfo_.starkStruct.verificationHashType == "GL" ? HASH_SIZE : 1;

            for(uint64_t i = 0; i < nStages + nCustomCommits; i++)
            {
                roots[i] = new ElementType[nFieldElements];
            }

            if (lastLevelVerification > 0) {
                size_t num_nodes = std::pow(starkInfo_.starkStruct.merkleTreeArity, lastLevelVerification);

                for(uint64_t i = 0; i < 1 + nStages + nCustomCommits; i++)
                {
                    last_levels[i] = new ElementType[nFieldElements * num_nodes];
                }
            }

            for(uint64_t i = 0; i < nCustomCommits; ++i) {
                customCommits[i] = starkInfo.customCommits[i].name;    
            }
        };

    ~Proofs() {
        for (uint64_t i = 0; i < nStages + nCustomCommits; ++i) {
            delete[] roots[i];
        }

        if (lastLevelVerification > 0) {
            for (uint64_t i = 0; i < 1 + nStages + nCustomCommits; ++i) {
                delete[] last_levels[i];
            }
        }

        delete[] roots;
        delete[] last_levels;
    }

    void setEvals(Goldilocks::Element *_evals)
    {
        for (uint64_t i = 0; i < evals.size(); i++)
        {
            std::memcpy(&evals[i][0], &_evals[i * evals[i].size()], evals[i].size() * sizeof(Goldilocks::Element));
        }
    }

    void setAirgroupValues(Goldilocks::Element *_airgroupValues) {
        uint64_t p = 0;
        for (uint64_t i = 0; i < starkInfo.airgroupValuesMap.size(); i++)
        {
            if(starkInfo.airgroupValuesMap[i].stage == 1) {
                airgroupValues[i][0] = _airgroupValues[p++];
                airgroupValues[i][1] = Goldilocks::zero();
                airgroupValues[i][2] = Goldilocks::zero();
            } else {
                std::memcpy(&airgroupValues[i][0], &_airgroupValues[p], FIELD_EXTENSION * sizeof(Goldilocks::Element));
                p += 3;
            }
        }
    }

    void setAirValues(Goldilocks::Element *_airValues) {
        uint64_t p = 0;
        for (uint64_t i = 0; i < starkInfo.airValuesMap.size(); i++)
        {
            if(starkInfo.airValuesMap[i].stage == 1) {
                airValues[i][0] = _airValues[p++];
                airValues[i][1] = Goldilocks::zero();
                airValues[i][2] = Goldilocks::zero();
            } else {
                std::memcpy(&airValues[i][0], &_airValues[p], FIELD_EXTENSION * sizeof(Goldilocks::Element));
                p += 3;
            }
        }
    }
    
    void setNonce(uint64_t _nonce) {
        nonce = _nonce;
    }

    uint64_t *proof2pointer(uint64_t *pointer) {
        uint64_t p = 0;

        for(uint64_t i = 0; i < starkInfo.airgroupValuesMap.size(); i++) {
            for (uint64_t k = 0; k < FIELD_EXTENSION; k++)
            {
                pointer[p++] = Goldilocks::toU64(airgroupValues[i][k]);
            }
        }


        for(uint64_t i = 0; i < starkInfo.airValuesMap.size(); i++) {
            for (uint64_t k = 0; k < FIELD_EXTENSION; k++)
            {
                pointer[p++] = Goldilocks::toU64(airValues[i][k]);
            }
        }

        for(uint64_t i = 0; i < starkInfo.nStages + 1; i++) {
            for (uint64_t k = 0; k < nFieldElements; k++)
            {
                pointer[p++] = toU64(roots[i][k]);
            }
        }

        for(uint64_t i = 0; i < starkInfo.evMap.size(); i++) {
            for (uint64_t k = 0; k < FIELD_EXTENSION; k++)
            {
                pointer[p++] = Goldilocks::toU64(evals[i][k]);
            }
        }

        uint64_t nSiblings = merkleProofLevels(starkInfo.starkStruct.nBitsExt, starkInfo.starkStruct.merkleTreeArity, starkInfo.starkStruct.lastLevelVerification, starkInfo.starkStruct.verificationHashType == std::string("BN128"));
        uint64_t nSiblingsPerLevel = (starkInfo.starkStruct.merkleTreeArity - 1) * nFieldElements;

        for (uint64_t i = 0; i < starkInfo.starkStruct.nQueries; i++) {
            for(uint64_t l = 0; l < starkInfo.nConstants; l++) {
                pointer[p++] = Goldilocks::toU64(fri.trees.polQueries[i][starkInfo.nStages + 1].v[l][0]);
            }
        }

        for (uint64_t i = 0; i < starkInfo.starkStruct.nQueries; i++) {
            for(uint64_t l = 0; l < nSiblings; ++l) {
                for(uint64_t k = 0; k < nSiblingsPerLevel; ++k) {
                    pointer[p++] = toU64(fri.trees.polQueries[i][starkInfo.nStages + 1].mp[l][k]);
                }
            }
        }

        if (starkInfo.starkStruct.lastLevelVerification != 0) {
            for (uint64_t k = 0; k < std::pow(starkInfo.starkStruct.merkleTreeArity, starkInfo.starkStruct.lastLevelVerification) * nFieldElements; k++)
            {
                pointer[p++] = toU64(last_levels[starkInfo.nStages + 1][k]);
            }
        }

        for(uint64_t c = 0; c < starkInfo.customCommits.size(); ++c) {
            for (uint64_t i = 0; i < starkInfo.starkStruct.nQueries; i++) {
                for(uint64_t l = 0; l < starkInfo.mapSectionsN[starkInfo.customCommits[c].name + "0"]; l++) {
                    pointer[p++] = Goldilocks::toU64(fri.trees.polQueries[i][starkInfo.nStages + 2 + c].v[l][0]);
                }
            }
            for (uint64_t i = 0; i < starkInfo.starkStruct.nQueries; i++) {
                for(uint64_t l = 0; l < nSiblings; ++l) {
                    for(uint64_t k = 0; k < nSiblingsPerLevel; ++k) {
                        pointer[p++] = toU64(fri.trees.polQueries[i][starkInfo.nStages + 2 + c].mp[l][k]);
                    }
                }
            }

            if (starkInfo.starkStruct.lastLevelVerification != 0) {
                for (uint64_t k = 0; k < std::pow(starkInfo.starkStruct.merkleTreeArity, starkInfo.starkStruct.lastLevelVerification) * nFieldElements; k++)
                {
                    pointer[p++] = toU64(last_levels[starkInfo.nStages + 2 + c][k]);
                }
            }
        }
        
        for (uint64_t s = 0; s < starkInfo.nStages + 1; ++s) {
            uint64_t stage = s + 1;
            for (uint64_t i = 0; i < starkInfo.starkStruct.nQueries; i++) {
                for(uint64_t l = 0; l < starkInfo.mapSectionsN["cm" + to_string(stage)]; l++) {
                    pointer[p++] = Goldilocks::toU64(fri.trees.polQueries[i][s].v[l][0]);
                }
            }

            for (uint64_t i = 0; i < starkInfo.starkStruct.nQueries; i++) {
                for(uint64_t l = 0; l < nSiblings; ++l) {
                    for(uint64_t k = 0; k < nSiblingsPerLevel; ++k) {
                        pointer[p++] = toU64(fri.trees.polQueries[i][s].mp[l][k]);
                    }
                }
            }

            if (starkInfo.starkStruct.lastLevelVerification != 0) {
                for (uint64_t k = 0; k < std::pow(starkInfo.starkStruct.merkleTreeArity, starkInfo.starkStruct.lastLevelVerification) * nFieldElements; k++)
                {
                    pointer[p++] = toU64(last_levels[s][k]);
                }
            }
        }
        

        if (starkInfo.starkStruct.lowDegreeTest == LowDegreeTestKind::STIR) {
            const StirStruct &stirStruct = starkInfo.starkStruct.stir;
            uint64_t M = stirStruct.numIterations();
            uint64_t numNodesLevel = starkInfo.starkStruct.lastLevelVerification == 0 ? 0 : std::pow(starkInfo.starkStruct.merkleTreeArity, starkInfo.starkStruct.lastLevelVerification);

            // Roots of T_0..T_{M−1}: unlike FRI, the first one is a commitment of its own (to f_0)
            // rather than a re-commitment of an already rooted oracle.
            for(uint64_t i = 0; i < M; ++i) {
                for(uint64_t k = 0; k < nFieldElements; k++) {
                    pointer[p++] = toU64(stir.trees[i].root[k]);
                }
            }

            // Per iteration: the opened cosets, their Merkle paths, and the published last level.
            for(uint64_t i = 0; i < M; ++i) {
                uint64_t k = uint64_t(1) << stirStruct.foldingFactors[i];
                uint64_t logLeaves = stirStruct.logDomainSizes[i] - stirStruct.foldingFactors[i];
                uint64_t nSiblingsStir = merkleProofLevels(logLeaves, starkInfo.starkStruct.merkleTreeArity, starkInfo.starkStruct.lastLevelVerification, starkInfo.starkStruct.verificationHashType == std::string("BN128"));

                for (uint64_t q = 0; q < stirStruct.numQueries[i]; q++) {
                    for(uint64_t l = 0; l < k * FIELD_EXTENSION; l++) {
                        pointer[p++] = Goldilocks::toU64(stir.trees[i].polQueries[q][0].v[l][0]);
                    }
                }
                for (uint64_t q = 0; q < stirStruct.numQueries[i]; q++) {
                    for(uint64_t l = 0; l < nSiblingsStir; ++l) {
                        for(uint64_t c = 0; c < nSiblingsPerLevel; ++c) {
                            pointer[p++] = toU64(stir.trees[i].polQueries[q][0].mp[l][c]);
                        }
                    }
                }
                for(uint64_t l = 0; l < numNodesLevel * nFieldElements; l++) {
                    pointer[p++] = toU64(stir.trees[i].last_levels[l]);
                }
            }

            // Out-of-domain answers β_{i,·}, i = 1..M−1.
            for(uint64_t i = 0; i + 1 < M; ++i) {
                for(uint64_t l = 0; l < stir.betas[i].size(); l++) {
                    pointer[p++] = Goldilocks::toU64(stir.betas[i][l]);
                }
            }

            // p in the clear, as d_M coefficients.
            for(uint64_t l = 0; l < stir.finalPol.size(); l++) {
                pointer[p++] = Goldilocks::toU64(stir.finalPol[l]);
            }

            // One grinding nonce per query message.
            for(uint64_t i = 0; i < M; ++i) {
                pointer[p++] = stir.nonces[i];
            }

            // Âns coefficient hints for the recursion circuit (zero-padded, see StirProof).
            for(uint64_t i = 0; i + 1 < M; ++i) {
                for(uint64_t l = 0; l < stir.ansCoeffs[i].size(); l++) {
                    pointer[p++] = Goldilocks::toU64(stir.ansCoeffs[i][l]);
                }
            }

            return pointer;
        }

        for(uint64_t step = 1; step < starkInfo.starkStruct.logDomainSizes.size(); ++step) {
             for(uint64_t i = 0; i < nFieldElements; i++) {
                pointer[p++] = toU64(fri.treesFRI[step - 1].root[i]);
            }
        }
        
        for(uint64_t step = 1; step < starkInfo.starkStruct.logDomainSizes.size(); ++step) {
            for (uint64_t i = 0; i < starkInfo.starkStruct.nQueries; i++) {
                for(uint64_t l = 0; l < uint64_t(1 << (starkInfo.starkStruct.logDomainSizes[step - 1] - starkInfo.starkStruct.logDomainSizes[step])) * FIELD_EXTENSION; l++) {
                    pointer[p++] = Goldilocks::toU64(fri.treesFRI[step - 1].polQueries[i][0].v[l][0]);
                }
            }

            for (uint64_t i = 0; i < starkInfo.starkStruct.nQueries; i++) {
                uint64_t nSiblings = merkleProofLevels(starkInfo.starkStruct.logDomainSizes[step], starkInfo.starkStruct.merkleTreeArity, starkInfo.starkStruct.lastLevelVerification, starkInfo.starkStruct.verificationHashType == std::string("BN128"));
                uint64_t nSiblingsPerLevel = (starkInfo.starkStruct.merkleTreeArity - 1) * nFieldElements;
                for(uint64_t l = 0; l < nSiblings; ++l) {
                    for(uint64_t k = 0; k < nSiblingsPerLevel; ++k) {
                        pointer[p++] = toU64(fri.treesFRI[step - 1].polQueries[i][0].mp[l][k]);
                    }
                }
            }

            if (starkInfo.starkStruct.lastLevelVerification != 0) {
                for(uint64_t i = 0; i < std::pow(starkInfo.starkStruct.merkleTreeArity, starkInfo.starkStruct.lastLevelVerification) * nFieldElements; i++) {
                    pointer[p++] = toU64(fri.treesFRI[step - 1].last_levels[i]);
                }
            }
        }

        for (uint64_t i = 0; i < uint64_t (1 << (starkInfo.starkStruct.logDomainSizes[starkInfo.starkStruct.logDomainSizes.size() - 1])); i++)
        {
            for(uint64_t l = 0; l < FIELD_EXTENSION; l++) {
                pointer[p++] = Goldilocks::toU64(fri.pol[i][l]);
            }
        }

        pointer[p++] = nonce;

        return pointer;
    }

    json proof2json()
    {
        json j = json::object();
        
        for(uint64_t i = 0; i < nStages; i++) {
            if(nFieldElements == 1) {
                j["root" + to_string(i + 1)] = toString(roots[i][0]);
            } else {
                j["root" + to_string(i + 1)] = json::array();
                for (uint k = 0; k < nFieldElements; k++)
                {
                    j["root" + to_string(i + 1)][k] = toString(roots[i][k]);
                }
            }
        }

        j["evals"] = json::array();
        for (uint i = 0; i < evals.size(); i++)
        {
            j["evals"][i] = json::array();
            for (uint k = 0; k < FIELD_EXTENSION; k++)
            {
                j["evals"][i][k] = Goldilocks::toString(evals[i][k]);
            }
        }

        if(airgroupValues.size() > 0) {
            j["airgroupvalues"] = json::array();
            for (uint i = 0; i < airgroupValues.size(); i++)
            {
                j["airgroupvalues"][i] = json::array();
                for (uint k = 0; k < FIELD_EXTENSION; k++)
                {
                    j["airgroupvalues"][i][k] = Goldilocks::toString(airgroupValues[i][k]);
                }
            }
        }

        if(airValues.size() > 0) {
            j["airvalues"] = json::array();
            for (uint i = 0; i < airValues.size(); i++)
            {
                j["airvalues"][i] = json::array();
                for (uint k = 0; k < airValues[i].size(); k++)
                {
                    j["airvalues"][i][k] = Goldilocks::toString(airValues[i][k]);
                }
            }
        }

        
        j["s0_valsC"] = json::array();
        j["s0_siblingsC"] = json::array();

        for(uint64_t i = 0; i < starkInfo.nStages + 1; ++i) {
            uint64_t stage = i + 1;
            j["s0_siblings" + to_string(stage)] = json::array();
            j["s0_vals" + to_string(stage)] = json::array();
        }

        for(uint64_t i = 0; i < starkInfo.customCommits.size(); ++i) {
            j["s0_siblings_" + starkInfo.customCommits[i].name + "_0"] = json::array();
            j["s0_vals_" + starkInfo.customCommits[i].name + "_0"] = json::array();
        }

        for (uint64_t i = 0; i < starkInfo.starkStruct.nQueries; i++) {
            uint64_t nSiblings = merkleProofLevels(starkInfo.starkStruct.nBitsExt, starkInfo.starkStruct.merkleTreeArity, starkInfo.starkStruct.lastLevelVerification, starkInfo.starkStruct.verificationHashType == std::string("BN128"));
            uint64_t nSiblingsPerLevel = starkInfo.starkStruct.verificationHashType == std::string("BN128") ? starkInfo.starkStruct.merkleTreeArity : (starkInfo.starkStruct.merkleTreeArity - 1) * nFieldElements;

            j["s0_valsC"][i] = json::array();
            j["s0_siblingsC"][i] = json::array();
            for(uint64_t l = 0; l < starkInfo.nConstants; l++) {
                j["s0_valsC"][i][l] = Goldilocks::toString(fri.trees.polQueries[i][starkInfo.nStages + 1].v[l][0]);
            }
            for(uint64_t l = 0; l < nSiblings; ++l) {
                for(uint64_t k = 0; k < nSiblingsPerLevel; ++k) {
                    j["s0_siblingsC"][i][l][k] = toString(fri.trees.polQueries[i][starkInfo.nStages + 1].mp[l][k]);
                }
            }

            for (uint64_t s = 0; s < nStages; ++s) {
                uint64_t stage = s + 1;
                j["s0_vals" + to_string(stage)][i] = json::array();
                for(uint64_t l = 0; l < starkInfo.mapSectionsN["cm" + to_string(stage)]; l++) {
                    j["s0_vals" + to_string(stage)][i][l] = Goldilocks::toString(fri.trees.polQueries[i][s].v[l][0]);
                }

                j["s0_siblings" + to_string(stage)][i] = json::array();
                for(uint64_t l = 0; l < nSiblings; ++l) {
                    for(uint64_t k = 0; k < nSiblingsPerLevel; ++k) {
                        j["s0_siblings" + to_string(stage)][i][l][k] = toString(fri.trees.polQueries[i][s].mp[l][k]);
                    }
                }
            }

            for(uint64_t c = 0; c < starkInfo.customCommits.size(); ++c) {
                j["s0_siblings_" + starkInfo.customCommits[c].name + "_0"][i] = json::array();
                j["s0_vals_" + starkInfo.customCommits[c].name + "_0"][i] = json::array();

                for(uint64_t l = 0; l < starkInfo.mapSectionsN[starkInfo.customCommits[c].name + "0"]; l++) {
                    j["s0_vals_" + starkInfo.customCommits[c].name + "_0"][i][l] = Goldilocks::toString(fri.trees.polQueries[i][starkInfo.nStages + 2 + c].v[l][0]);
                }
                for(uint64_t l = 0; l < nSiblings; ++l) {
                    for(uint64_t k = 0; k < nSiblingsPerLevel; ++k) {
                        j["s0_siblings_" + starkInfo.customCommits[c].name + "_0"][i][l][k] = toString(fri.trees.polQueries[i][starkInfo.nStages + 2 + c].mp[l][k]);
                    }
                }
            }
        }

        if (starkInfo.starkStruct.lastLevelVerification != 0) {
            uint64_t nNodesLastLevel = std::pow(starkInfo.starkStruct.merkleTreeArity, starkInfo.starkStruct.lastLevelVerification);
            auto emitLastLevels = [&](const std::string &name, ElementType *levels) {
                j[name] = json::array();
                for (uint64_t k = 0; k < nNodesLastLevel; k++) {
                    if (nFieldElements == 1) {
                        j[name][k] = toString(levels[k]);
                    } else {
                        j[name][k] = json::array();
                        for (uint64_t l = 0; l < nFieldElements; l++) {
                            j[name][k][l] = toString(levels[k * nFieldElements + l]);
                        }
                    }
                }
            };
            for (uint64_t s = 0; s < nStages; ++s) {
                emitLastLevels("s0_last_levels" + to_string(s + 1), last_levels[s]);
            }
            emitLastLevels("s0_last_levelsC", last_levels[starkInfo.nStages + 1]);
            for (uint64_t c = 0; c < starkInfo.customCommits.size(); ++c) {
                emitLastLevels("s0_last_levels_" + starkInfo.customCommits[c].name + "_0", last_levels[starkInfo.nStages + 2 + c]);
            }
            for (uint64_t step = 1; step < starkInfo.starkStruct.logDomainSizes.size(); ++step) {
                emitLastLevels("s" + std::to_string(step) + "_last_levels", fri.treesFRI[step - 1].last_levels.data());
            }
        }

        if (starkInfo.starkStruct.lowDegreeTest == LowDegreeTestKind::STIR) {
            const StirStruct &stirStruct = starkInfo.starkStruct.stir;
            uint64_t M = stirStruct.numIterations();

            // One object per committed oracle: "s{i+1}_root" / "_vals" / "_siblings" /
            // "_last_levels" — the same 1-based naming as FRI's trees (s1 commits the
            // DEEP polynomial in both tests),
            // named apart from FRI's "s{step}_*" so a reader cannot confuse the two layouts.
            for(uint64_t i = 0; i < M; ++i) {
                std::string prefix = "s" + std::to_string(i + 1);
                if(nFieldElements == 1) {
                    j[prefix + "_root"] = toString(stir.trees[i].root[0]);
                } else {
                    j[prefix + "_root"] = json::array();
                    for(uint64_t l = 0; l < nFieldElements; l++) {
                        j[prefix + "_root"][l] = toString(stir.trees[i].root[l]);
                    }
                }

                uint64_t k = uint64_t(1) << stirStruct.foldingFactors[i];
                uint64_t logLeaves = stirStruct.logDomainSizes[i] - stirStruct.foldingFactors[i];
                uint64_t nSiblingsStir = merkleProofLevels(logLeaves, starkInfo.starkStruct.merkleTreeArity, starkInfo.starkStruct.lastLevelVerification, starkInfo.starkStruct.verificationHashType == std::string("BN128"));
                uint64_t nSiblingsPerLevelStir = starkInfo.starkStruct.verificationHashType == std::string("BN128") ? starkInfo.starkStruct.merkleTreeArity : (starkInfo.starkStruct.merkleTreeArity - 1) * nFieldElements;

                j[prefix + "_vals"] = json::array();
                j[prefix + "_siblings"] = json::array();
                for(uint64_t q = 0; q < stirStruct.numQueries[i]; q++) {
                    j[prefix + "_vals"][q] = json::array();
                    j[prefix + "_siblings"][q] = json::array();
                    for(uint64_t l = 0; l < k * FIELD_EXTENSION; l++) {
                        j[prefix + "_vals"][q][l] = Goldilocks::toString(stir.trees[i].polQueries[q][0].v[l][0]);
                    }
                    for(uint64_t l = 0; l < nSiblingsStir; ++l) {
                        for(uint64_t c = 0; c < nSiblingsPerLevelStir; ++c) {
                            j[prefix + "_siblings"][q][l][c] = toString(stir.trees[i].polQueries[q][0].mp[l][c]);
                        }
                    }
                }

                if(starkInfo.starkStruct.lastLevelVerification != 0) {
                    j[prefix + "_last_levels"] = json::array();
                    uint64_t numNodesLevel = stir.trees[i].last_levels.size() / nFieldElements;
                    for(uint64_t l = 0; l < numNodesLevel; ++l) {
                        j[prefix + "_last_levels"][l] = json::array();
                        for(uint64_t c = 0; c < nFieldElements; ++c) {
                            j[prefix + "_last_levels"][l][c] = toString(stir.trees[i].last_levels[l * nFieldElements + c]);
                        }
                    }
                }
            }

            j["betas"] = json::array();
            for(uint64_t i = 0; i + 1 < M; ++i) {
                j["betas"][i] = json::array();
                for(uint64_t l = 0; l < stir.betas[i].size(); l++) {
                    j["betas"][i][l] = Goldilocks::toString(stir.betas[i][l]);
                }
            }

            // p as coefficients, so its degree bound is structural rather than checked.
            j["finalPol"] = json::array();
            for(uint64_t l = 0; l < stir.finalPol.size(); l++) {
                j["finalPol"][l] = Goldilocks::toString(stir.finalPol[l]);
            }

            j["nonces"] = json::array();
            for(uint64_t i = 0; i < M; ++i) {
                j["nonces"][i] = std::to_string(stir.nonces[i]);
            }

            j["ansCoeffs"] = json::array();
            for(uint64_t i = 0; i + 1 < M; ++i) {
                j["ansCoeffs"][i] = json::array();
                for(uint64_t l = 0; l < stir.ansCoeffs[i].size(); l++) {
                    j["ansCoeffs"][i][l] = Goldilocks::toString(stir.ansCoeffs[i][l]);
                }
            }

            return j;
        }

        for(uint64_t step = 1; step < starkInfo.starkStruct.logDomainSizes.size(); ++step) {
            if(nFieldElements == 1) {
                j["s" + std::to_string(step) + "_root"] = toString(fri.treesFRI[step - 1].root[0]);
            } else {
                j["s" + std::to_string(step) + "_root"] = json::array();
                for(uint64_t i = 0; i < nFieldElements; i++) {
                    j["s" + std::to_string(step) + "_root"][i] = toString(fri.treesFRI[step - 1].root[i]);
                }
                j["s" + std::to_string(step) + "_vals"] = json::array();
                j["s" + std::to_string(step) + "_siblings"] = json::array();
            }
        }

        for(uint64_t i = 0; i < starkInfo.starkStruct.nQueries; i++) {
            for(uint64_t step = 1; step < starkInfo.starkStruct.logDomainSizes.size(); ++step) {
                j["s" + std::to_string(step) + "_vals"][i] = json::array();
                j["s" + std::to_string(step) + "_siblings"][i] = json::array();

                for(uint64_t l = 0; l < uint64_t(1 << (starkInfo.starkStruct.logDomainSizes[step - 1] - starkInfo.starkStruct.logDomainSizes[step])) * FIELD_EXTENSION; l++) {
                    j["s" + std::to_string(step) + "_vals"][i][l] = Goldilocks::toString(fri.treesFRI[step - 1].polQueries[i][0].v[l][0]);
                }

                uint64_t nSiblings = merkleProofLevels(starkInfo.starkStruct.logDomainSizes[step], starkInfo.starkStruct.merkleTreeArity, starkInfo.starkStruct.lastLevelVerification, starkInfo.starkStruct.verificationHashType == std::string("BN128"));
                uint64_t nSiblingsPerLevel = starkInfo.starkStruct.verificationHashType == std::string("BN128") ? starkInfo.starkStruct.merkleTreeArity : (starkInfo.starkStruct.merkleTreeArity - 1) * nFieldElements;

                for(uint64_t l = 0; l < nSiblings; ++l) {
                    for(uint64_t k = 0; k < nSiblingsPerLevel; ++k) {
                        j["s" + std::to_string(step) + "_siblings"][i][l][k] = toString(fri.treesFRI[step - 1].polQueries[i][0].mp[l][k]);
                    }
                }
            }
        }
        

        j["finalPol"] = json::array();
        for (uint64_t i = 0; i < uint64_t (1 << (starkInfo.starkStruct.logDomainSizes[starkInfo.starkStruct.logDomainSizes.size() - 1])); i++)
        {
            j["finalPol"][i] = json::array();
            for(uint64_t l = 0; l < FIELD_EXTENSION; l++) {
                j["finalPol"][i][l] = Goldilocks::toString(fri.pol[i][l]);
            }
        }

        j["nonce"] = std::to_string(nonce);
        
        return j;
    }
};

template <typename ElementType>
class FRIProof
{
public:
    Proofs<ElementType> proof;
    std::vector<ElementType> publics;
    
    uint64_t airgroupId;
    uint64_t airId;
    uint64_t instanceId;

    FRIProof(StarkInfo &starkInfo, uint64_t _airgroupId, uint64_t _airId, uint64_t _instanceId) : 
        proof(starkInfo), 
        publics(starkInfo.nPublics),
        airgroupId(_airgroupId),
        airId(_airId),
        instanceId(_instanceId) {};
};


#endif