
#include "merkleTreeBN128.hpp"
#include <algorithm> // std::max
#include <cassert>

MerkleTreeBN128::MerkleTreeBN128(uint64_t _arity, uint64_t _last_level_verification, bool _custom, uint64_t _height, uint64_t _width, bool allocateSource, bool allocateNodes) : height(_height), width(_width)
{

    arity = _arity;
    last_level_verification = _last_level_verification;
    custom = _custom;
    numNodes = getNumNodes(height);
    
    if(allocateSource) {
        source = (Goldilocks::Element *)calloc(height * width, sizeof(Goldilocks::Element));
        isSourceAllocated = true;
    }

    if(allocateNodes) {
        nodes = (RawFr::Element *)calloc(numNodes, sizeof(RawFr::Element));
        isNodesAllocated = true;
    }
   
}

MerkleTreeBN128::MerkleTreeBN128(uint64_t _arity, uint64_t _last_level_verification, bool _custom, Goldilocks::Element *tree, uint64_t height_, uint64_t width_)
{
    width = width_;
    height = height_;
    source = tree;
    arity = _arity;
    last_level_verification = _last_level_verification;
    custom = _custom;
    numNodes = getNumNodes(height);
    nodes = (RawFr::Element *)&source[width * height];
}

MerkleTreeBN128::~MerkleTreeBN128()
{
    if(isSourceAllocated) {
        free(source);
    }

    if(isNodesAllocated) {
        free(nodes);
    }
}

uint64_t MerkleTreeBN128::getNumSiblings() 
{
    return arity * nFieldElements;
}

uint64_t MerkleTreeBN128::getMerkleTreeWidth()
{
    return width;
}

uint64_t MerkleTreeBN128::getMerkleProofLength()
{
    return ceil((double)log(height) / log(arity)) - last_level_verification;
}


uint64_t MerkleTreeBN128::getMerkleProofSize()
{
    return getMerkleProofLength() * arity * sizeof(RawFr::Element);
}

uint64_t MerkleTreeBN128::getNumNodes(uint64_t n)
{   
    uint n_tmp = n;
    uint64_t nextN = floor(((double)(n_tmp - 1) / arity) + 1);
    uint64_t acc = nextN * arity;
    while (n_tmp > 1)
    {
        // FIll with zeros if n nodes in the leve is not even
        n_tmp = nextN;
        nextN = floor((n_tmp - 1) / arity) + 1;
        if (n_tmp > 1)
        {
            acc += nextN * arity;
        }
        else
        {
            acc += 1;
        }
    }

    return acc;
}


void MerkleTreeBN128::getLevel(RawFr::Element *level)
{
    if (last_level_verification != 0) {
        uint64_t n = height;
        uint64_t offset = 0;
        while (n > std::pow(arity, last_level_verification)) {
            n = (std::floor((n - 1) / arity) + 1);
            offset += n * arity;
        }

        std::memcpy(level, &nodes[offset], n * sizeof(RawFr::Element));
        for (uint64_t i = n; i < std::pow(arity, last_level_verification); i++) {
            level[i] = RawFr::field.zero();
        }
    }
}


void MerkleTreeBN128::getRoot(RawFr::Element *root)
{
    std::memcpy(root, &nodes[numNodes - 1], sizeof(RawFr::Element));
}


void MerkleTreeBN128::setSource(Goldilocks::Element *_source)
{
    if(isSourceAllocated) {
        zklog.error("MerkleTreeBN128: Source was allocated when initializing");
        exitProcess();
        exit(-1);
    }
    source = _source;
}

void MerkleTreeBN128::setNodes(RawFr::Element *_nodes)
{
    if(isNodesAllocated) {
        zklog.error("MerkleTreeBN128: Nodes were allocated when initializing");
        exitProcess();
        exit(-1);
    }
    nodes = _nodes;
}

Goldilocks::Element MerkleTreeBN128::getElement(uint64_t idx, uint64_t subIdx)
{
    assert((idx > 0) || (idx < width));
    return source[width * idx + subIdx];
}

void MerkleTreeBN128::getGroupProof(RawFr::Element *proof, uint64_t idx)
{
    assert(idx < height);

    Goldilocks::Element v[width];
    for (uint64_t i = 0; i < width; i++)
    {
        v[i] = getElement(idx, i);
    }
    std::memcpy(proof, &v[0], width * sizeof(Goldilocks::Element));
    void *proofCursor = (uint8_t *)proof + width * sizeof(Goldilocks::Element);

    genMerkleProof((RawFr::Element *)proofCursor, idx, 0, height);
}

void MerkleTreeBN128::genMerkleProof(RawFr::Element *proof, uint64_t idx, uint64_t offset, uint64_t n)
{
    if ((last_level_verification == 0 && n == 1) || (last_level_verification > 0 && n <= std::pow(arity, last_level_verification))) return;

    uint64_t nBitsArity = std::ceil(std::log2(arity));

    uint64_t nextIdx = idx >> nBitsArity;
    uint64_t si = idx ^ (idx & (arity - 1));

    std::memcpy(proof, &nodes[offset + si], arity * sizeof(RawFr::Element));
    uint64_t nextN = (std::floor((n - 1) / arity) + 1);
    genMerkleProof(&proof[arity], nextIdx, offset + nextN * arity, nextN);
}

void MerkleTreeBN128::calculateRootFromProof(RawFr::Element *value, std::vector<std::vector<RawFr::Element>> &mp, uint64_t &idx, uint64_t offset) {
    if(offset == mp.size()) return;

    uint64_t nBitsArity = std::ceil(std::log2(arity));

    uint64_t currIdx = idx & (arity - 1);
    idx = idx >> nBitsArity;

    PoseidonBN128 p;
    std::vector<RawFr::Element> elements(arity + 1);
    std::memset(&elements[0], 0, (arity + 1) * sizeof(RawFr::Element));

    for(uint64_t i = 0; i < arity; ++i) {
        std::memcpy(&elements[1 + i], &mp[offset][i], sizeof(RawFr::Element));
    }

    std::memcpy(&elements[1 + currIdx], &value[0], sizeof(RawFr::Element));
    p.hash(elements, &value[0]);

    calculateRootFromProof(value, mp, idx, offset + 1);

}


bool MerkleTreeBN128::verifyGroupProof(RawFr::Element* root, RawFr::Element* level, std::vector<std::vector<RawFr::Element>> &mp, uint64_t idx, std::vector<Goldilocks::Element> &v) {
    RawFr::Element value[1];
    value[0] = RawFr::field.zero();
    PoseidonBN128 p;
    p.linearHash(value, v.data(), width, arity+1, custom);

    uint64_t queryIdx = idx;
    calculateRootFromProof(&value[0], mp, queryIdx, 0);

    if (last_level_verification == 0) {
        if (!RawFr::field.eq(root[0], value[0])) {
            return false;
        }
    } else {
        if (!RawFr::field.eq(level[queryIdx], value[0])) {
            return false;
        }
    }

    return true;
}

void MerkleTreeBN128::merkelize()
{
    PoseidonBN128 p;
    p.merkletree(nodes, source, height, width, arity, custom);
}

void MerkleTreeBN128::writeFile(std::string constTreeFile) {
    ofstream fw(constTreeFile.c_str(), std::fstream::out | std::fstream::binary);
    uint64_t nodesOffset = width * height * sizeof(Goldilocks::Element);
    fw.close();
    writeFileParallel(constTreeFile, (const char *)source, width * height * sizeof(Goldilocks::Element), 0);
    writeFileParallel(constTreeFile, (const char *)nodes, numNodes * sizeof(RawFr::Element), nodesOffset);
}