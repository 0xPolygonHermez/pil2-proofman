#ifndef MERKLETREEBN128
#define MERKLETREEBN128

#include <math.h>
#include "fr.hpp"
#include "goldilocks_base_field.hpp"
#include "goldilocks_cubic_extension.hpp"
#include "poseidon_opt.hpp"
#include "zklog.hpp"

class MerkleTreeBN128
{
private:
    void linearHash();
    void linearHash(RawFrP::Element* result, Goldilocks::Element* values);

    Goldilocks::Element getElement(uint64_t idx, uint64_t subIdx);
    void genMerkleProof(RawFrP::Element *proof, uint64_t idx, uint64_t offset, uint64_t n);
    void calculateRootFromProof(RawFrP::Element *value, std::vector<std::vector<RawFrP::Element>> &mp, uint64_t &idx, uint64_t offset);

public:
    MerkleTreeBN128(){};
    MerkleTreeBN128(uint64_t arity, uint64_t last_level_verification, bool custom, Goldilocks::Element *tree, uint64_t height, uint64_t width);
    MerkleTreeBN128(uint64_t arity, uint64_t last_level_verification, bool custom, uint64_t _height, uint64_t _width, bool allocateSource = false, bool allocateNodes = false);
    ~MerkleTreeBN128();

    uint64_t numNodes;
    uint64_t height;
    uint64_t width;

    Goldilocks::Element *source;
    RawFrP::Element *nodes;

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
    void getRoot(RawFrP::Element *root);
    void getLevel(RawFrP::Element *level);
    void setSource(Goldilocks::Element *source);
    void setNodes(RawFrP::Element *nodes);

    void getGroupProof(RawFrP::Element *proof, uint64_t idx);
    
    void merkelize();
    Goldilocks::Element* get_nodes_ptr() {
        assert(false); //This function cannot be used, is just for compatibility
        return source;
    }

    bool verifyGroupProof(RawFrP::Element* root, RawFrP::Element* level, std::vector<std::vector<RawFrP::Element>> &mp, uint64_t idx, std::vector<Goldilocks::Element> &v);

    void writeFile(std::string constTreeFile);

    bool static verifyMerkleRoot(RawFrP::Element *root, RawFrP::Element *level, uint64_t height, uint64_t lastLevelVerification, uint64_t arity, uint64_t nFieldElements) {
        // TODO: implement
        return true;
    }
};
#endif