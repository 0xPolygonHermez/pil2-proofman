#ifndef MERKLETREEBN128
#define MERKLETREEBN128

#include <math.h>
#include "fr.hpp"
#include "goldilocks_base_field.hpp"
#include "goldilocks_cubic_extension.hpp"
#include "poseidon_bn128.hpp"
#include "zklog.hpp"

class MerkleTreeBN128
{
private:
    
    Goldilocks::Element getElement(uint64_t idx, uint64_t subIdx);
    void genMerkleProof(RawFr::Element *proof, uint64_t idx, uint64_t offset, uint64_t n);
    void calculateRootFromProof(RawFr::Element *value, std::vector<std::vector<RawFr::Element>> &mp, uint64_t &idx, uint64_t offset);

public:
    MerkleTreeBN128(){};
    MerkleTreeBN128(uint64_t arity, uint64_t last_level_verification, bool custom, Goldilocks::Element *tree, uint64_t height, uint64_t width);
    MerkleTreeBN128(uint64_t arity, uint64_t last_level_verification, bool custom, uint64_t _height, uint64_t _width, bool allocateSource = false, bool allocateNodes = false);
    ~MerkleTreeBN128();

    uint64_t numNodes;
    uint64_t height;
    uint64_t width;

    Goldilocks::Element *source;
    RawFr::Element *nodes;

    bool isSourceAllocated = false;
    bool isNodesAllocated = false;

    uint64_t arity;
    uint64_t last_level_verification = 0;
    bool custom;
    uint64_t nFieldElements = 1;

    uint64_t getNumSiblings();
    uint64_t getMerkleTreeWidth();
    uint64_t getMerkleProofSize();
    uint64_t getMerkleProofLength();

    uint64_t getNumNodes(uint64_t height);
    void getRoot(RawFr::Element *root);
    void getLevel(RawFr::Element *level);
    void setSource(Goldilocks::Element *source);
    void setNodes(RawFr::Element *nodes);

    void getGroupProof(RawFr::Element *proof, uint64_t idx);
    
    void merkelize();
    void* get_nodes_ptr() {
        return nodes;
    }

    bool verifyGroupProof(RawFr::Element* root, RawFr::Element* level, std::vector<std::vector<RawFr::Element>> &mp, uint64_t idx, std::vector<Goldilocks::Element> &v);

    void writeFile(std::string constTreeFile);

    bool static verifyMerkleRoot(RawFr::Element *root, RawFr::Element *level, uint64_t height, uint64_t lastLevelVerification, uint64_t arity, uint64_t nFieldElements) {
        uint64_t numNodesLevel = height;
        while (numNodesLevel > std::pow(arity, lastLevelVerification)) {
            numNodesLevel = (numNodesLevel + arity - 1) / arity;
        }

        // Hash the level up to the root, replicating PoseidonBN128::merkletree
        // padding: parents = ceil(n / arity), missing children are zeros, and
        // each parent = hash([0 capacity, child_0 .. child_{arity-1}]).
        PoseidonBN128 p;
        std::vector<RawFr::Element> nodes(level, level + numNodesLevel);
        uint64_t n = numNodesLevel;
        while (n > 1) {
            uint64_t nextN = (n - 1) / arity + 1;
            nodes.resize(nextN * arity);
            for (uint64_t i = n; i < nextN * arity; i++) {
                nodes[i] = RawFr::field.zero();
            }
            for (uint64_t j = 0; j < nextN; j++) {
                std::vector<RawFr::Element> elements(arity + 1);
                elements[0] = RawFr::field.zero();
                for (uint64_t a = 0; a < arity; a++) {
                    elements[1 + a] = nodes[j * arity + a];
                }
                p.hash(elements, &nodes[j]);
            }
            n = nextN;
        }

        return RawFr::field.eq(root[0], nodes[0]);
    }
};
#endif