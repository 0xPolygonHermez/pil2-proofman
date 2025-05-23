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
    void calculateRootFromProof(Goldilocks::Element (&value)[4], std::vector<std::vector<Goldilocks::Element>> &mp, uint64_t idx, uint64_t offset);

public:
    MerkleTreeGL(){};
    MerkleTreeGL(uint64_t _arity, bool custom, Goldilocks::Element *tree);
    MerkleTreeGL(uint64_t _arity, bool custom, uint64_t _height, uint64_t _width, bool allocateSource = false, bool allocateNodes = false);
    ~MerkleTreeGL();

    uint64_t numNodes;
    uint64_t height;
    uint64_t width;

    Goldilocks::Element *source;
    Goldilocks::Element *nodes;

    uint64_t arity;
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
    void getRoot(Goldilocks::Element *root);
    void setSource(Goldilocks::Element *_source);
    void setNodes(Goldilocks::Element *_nodes);
    void initSource();
    void initNodes();

    void getGroupProof(Goldilocks::Element *proof, uint64_t idx);
    bool verifyGroupProof(Goldilocks::Element* root, std::vector<std::vector<Goldilocks::Element>> &mp, uint64_t idx, std::vector<Goldilocks::Element> &v);

    void merkelize();
    Goldilocks::Element *get_nodes_ptr()
    {
        return nodes;
    }

    void writeFile(std::string file);
};

#endif