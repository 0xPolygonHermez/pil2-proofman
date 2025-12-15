#ifndef MERKLETREEGL
#define MERKLETREEGL

#include "goldilocks_base_field.hpp"
#include "poseidon2_goldilocks.hpp"
#include "zklog.hpp"
#include <math.h>

class MerkleTreeGL
{
private:
    Goldilocks::Element getElement(uint64_t idx, uint64_t subIdx);
    void calculateRootFromProof(Goldilocks::Element (&value)[4], std::vector<std::vector<Goldilocks::Element>> &mp, uint64_t &idx, uint64_t offset);

public:
    MerkleTreeGL(){};
    MerkleTreeGL(uint64_t _arity, uint64_t _last_level_verification, bool custom, Goldilocks::Element *tree, uint64_t height, uint64_t width);
    MerkleTreeGL(uint64_t _arity, uint64_t _last_level_verification, bool custom, uint64_t _height, uint64_t _width, bool allocateSource = false, bool allocateNodes = false);
    ~MerkleTreeGL();

    uint64_t numNodes;
    uint64_t height;
    uint64_t width;

    Goldilocks::Element *source;
    Goldilocks::Element *nodes;

    uint64_t arity;
    uint64_t last_level_verification;
    bool custom;

    bool isSourceAllocated = false;
    bool isNodesAllocated = false;

    uint64_t nFieldElements = HASH_SIZE;

    uint64_t getNumSiblings(); 
    uint64_t getMerkleTreeWidth(); 
    uint64_t getMerkleProofSize(); 
    uint64_t getMerkleProofLength();
    void genMerkleProof(Goldilocks::Element *proof, uint64_t idx, uint64_t offset, uint64_t n);
    inline uint64_t getMerkleTreeNFieldElements()
    {
        return nFieldElements;
    }
    inline uint64_t getMerkleTreeHeight()
    {
        return height;
    }

    uint64_t getNumNodes(uint64_t height);
    void getLevel(Goldilocks::Element *level);
    void getRoot(Goldilocks::Element *root);
    void setSource(Goldilocks::Element *_source);
    void setNodes(Goldilocks::Element *_nodes);
    void initSource();
    void initNodes();

    void getGroupProof(Goldilocks::Element *proof, uint64_t idx);
    bool verifyGroupProof(Goldilocks::Element* root, Goldilocks::Element* level, std::vector<std::vector<Goldilocks::Element>> &mp, uint64_t idx, std::vector<Goldilocks::Element> &v);

    void merkelize();
    Goldilocks::Element *get_nodes_ptr()
    {
        return nodes;
    }

    void writeFile(std::string file);

    bool static verifyMerkleRoot(Goldilocks::Element *root, Goldilocks::Element *level, uint64_t height, uint64_t lastLevelVerification, uint64_t arity, uint64_t nFieldElements) {
        uint64_t numNodesLevel = height;
        while (numNodesLevel > std::pow(arity, lastLevelVerification)) {
            numNodesLevel = (numNodesLevel + (arity - 1)) / arity;
        }
        Goldilocks::Element computedRoot[nFieldElements];
        switch(arity) {
            case 2:
                Poseidon2Goldilocks<8>::partial_merkle_tree(computedRoot, (Goldilocks::Element *)level, numNodesLevel, arity);
                break;
            case 3:
                Poseidon2Goldilocks<12>::partial_merkle_tree(computedRoot, (Goldilocks::Element *)level, numNodesLevel, arity);
                break;
            case 4:
                Poseidon2Goldilocks<16>::partial_merkle_tree(computedRoot, (Goldilocks::Element *)level, numNodesLevel, arity);
                break;
            default:
                zklog.error("MerkleTreeGL::verifyMerkleRoot: Unsupported arity");
                exitProcess();
                exit(-1);
        }

        for (uint64_t i = 0; i < nFieldElements; ++i) {
            if (Goldilocks::toU64(computedRoot[i]) != Goldilocks::toU64(root[i])) {
                return false;
            }
        }

        return true;
    }
};

#endif