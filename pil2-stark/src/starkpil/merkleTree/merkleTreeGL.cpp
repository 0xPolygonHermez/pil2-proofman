#include "merkleTreeGL.hpp"
#include <cassert>
#include <algorithm> // std::max


MerkleTreeGL::MerkleTreeGL(uint64_t _arity, bool _custom, uint64_t _height, uint64_t _width, bool allocateSource, bool allocateNodes) : height(_height), width(_width)
{
    arity = _arity;
    numNodes = getNumNodes(height);
    custom = _custom;

    if(allocateSource) {
        source = (Goldilocks::Element *)calloc(height * width, sizeof(Goldilocks::Element));
        isSourceAllocated = true;
    }

    if(allocateNodes) {
        nodes = (Goldilocks::Element *)calloc(numNodes, sizeof(Goldilocks::Element));
        isNodesAllocated = true;
    }
};

MerkleTreeGL::MerkleTreeGL(uint64_t _arity, bool _custom, Goldilocks::Element *tree)
{
    width = Goldilocks::toU64(tree[0]);
    height = Goldilocks::toU64(tree[1]);
    source = &tree[2];
    arity = _arity;
    custom = _custom;
    numNodes = getNumNodes(height);
    nodes = &tree[2 + height * width];
};

MerkleTreeGL::~MerkleTreeGL()
{
    if(isSourceAllocated) {
        free(source);
    }
    
    if(isNodesAllocated) {
        free(nodes);
    }
}

uint64_t MerkleTreeGL::getNumSiblings() 
{
    return (arity - 1) * nFieldElements;
}

uint64_t MerkleTreeGL::getMerkleTreeWidth() 
{
    return width;
}

uint64_t MerkleTreeGL::getMerkleProofLength() {
    if(height > 1) {
        return (uint64_t)ceil(log10(height) / log10(arity));
    } 
    return 0;
}

uint64_t MerkleTreeGL::getMerkleProofSize() {
    return getMerkleProofLength() * (arity - 1) * nFieldElements;
}

uint64_t MerkleTreeGL::getNumNodes(uint64_t height)
{
    uint64_t numNodes = height;
    uint64_t nodesLevel = height;
    
    while (nodesLevel > 1) {
        uint64_t extraZeros = (arity - (nodesLevel % arity)) % arity;
        numNodes += extraZeros;
        uint64_t nextN = (nodesLevel + (arity - 1))/arity;        
        numNodes += nextN;
        nodesLevel = nextN;
    }


    return numNodes * nFieldElements;
}

void MerkleTreeGL::getRoot(Goldilocks::Element *root)
{
    std::memcpy(root, &nodes[numNodes - nFieldElements], nFieldElements * sizeof(Goldilocks::Element));
}


void MerkleTreeGL::setSource(Goldilocks::Element *_source)
{
    if(isSourceAllocated) {
        zklog.error("MerkleTreeGL: Source was allocated when initializing");
        exitProcess();
        exit(-1);
    }
    source = _source;
}

void MerkleTreeGL::setNodes(Goldilocks::Element *_nodes)
{
    if(isNodesAllocated) {
        if(isNodesAllocated) {
        zklog.error("MerkleTreeGL: Nodes were allocated when initializing");
        exitProcess();
        exit(-1);
    }
    }
    nodes = _nodes;
}


Goldilocks::Element MerkleTreeGL::getElement(uint64_t idx, uint64_t subIdx)
{
    assert((idx > 0) || (idx < width));
    return source[idx * width + subIdx];
};

void MerkleTreeGL::getGroupProof(Goldilocks::Element *proof, uint64_t idx) {
    assert(idx < height);

    for (uint64_t i = 0; i < width; i++)
    {
        proof[i] = getElement(idx, i);
    }

    genMerkleProof(&proof[width], idx, 0, height);
}

void MerkleTreeGL::getTraceProof(Goldilocks::Element *proof, uint64_t idx) {
    assert(idx < height);

    for (uint64_t i = 0; i < width; i++)
    {
        proof[i] = getElement(idx, i);
    }
}

void MerkleTreeGL::genMerkleProof(Goldilocks::Element *proof, uint64_t idx, uint64_t offset, uint64_t n)
{
    if (n == 1) return;
    
    uint64_t currIdx = idx % arity;
    uint64_t nextIdx = idx / arity;
    uint64_t si = idx - currIdx;

    Goldilocks::Element *proofPtr = proof;
    for (uint64_t i = 0; i < arity; i++)
    {
        if (i == currIdx) continue;  // Skip the current index
        std::memcpy(proofPtr, &nodes[(offset + (si + i)) * nFieldElements], nFieldElements * sizeof(Goldilocks::Element));
        proofPtr += nFieldElements;
    }
   
    // Compute new offset for parent level
    uint64_t nextN = (n + (arity - 1))/arity;
    genMerkleProof(&proof[(arity - 1) * nFieldElements], nextIdx, offset + nextN * arity, nextN);
}

bool MerkleTreeGL::verifyGroupProof(Goldilocks::Element* root, std::vector<std::vector<Goldilocks::Element>> &mp, uint64_t idx, std::vector<Goldilocks::Element> &v) {
    Goldilocks::Element value[4] = { Goldilocks::zero(), Goldilocks::zero(), Goldilocks::zero(), Goldilocks::zero() };
    
    Poseidon2Goldilocks::linear_hash_seq(value, v.data(), v.size());

    calculateRootFromProof(value, mp, idx, 0);
    for(uint64_t i = 0; i < 4; ++i) {
        if(Goldilocks::toU64(value[i]) != Goldilocks::toU64(root[i])) {
            return false;
        }
    }

    return true;
}

void MerkleTreeGL::calculateRootFromProof(Goldilocks::Element (&value)[4], std::vector<std::vector<Goldilocks::Element>> &mp, uint64_t idx, uint64_t offset) {
    if(offset == mp.size()) return;

    uint64_t currIdx = idx % arity;
    uint64_t nextIdx = idx / arity;

    Goldilocks::Element inputs[12];
    for(uint64_t i = 0; i < 12; ++i) {
        inputs[i] = Goldilocks::zero();
    }

    uint64_t p = 0;
    for(uint64_t i = 0; i < arity; ++i) {
        if (i == currIdx) continue;
        std::memcpy(&inputs[i*nFieldElements], &mp[offset][nFieldElements * (p++)], nFieldElements * sizeof(Goldilocks::Element));
    }

    std::memcpy(&inputs[currIdx*4], value, nFieldElements * sizeof(Goldilocks::Element));

    Poseidon2Goldilocks::hash_seq(value, inputs);

    calculateRootFromProof(value, mp, nextIdx, offset + 1);
}


void MerkleTreeGL::merkelize()
{
#ifdef __AVX512__
    // Poseidon2Goldilocks::merkletree_avx512(nodes, source, width, height, arity); // AVX512 is not supported yet
    Poseidon2Goldilocks::merkletree_avx(nodes, source, width, height, arity);
#elif defined(__AVX2__)
    Poseidon2Goldilocks::merkletree_avx(nodes, source, width, height, arity);
#else
    Poseidon2Goldilocks::merkletree_seq(nodes, source, width, height, arity);
#endif
}

void MerkleTreeGL::writeFile(std::string constTreeFile)
{
    ofstream fw(constTreeFile.c_str(), std::fstream::out | std::fstream::binary);
    fw.write((const char *)&(width), sizeof(uint64_t));
    fw.write((const char *)&(height), sizeof(uint64_t)); 
    // fw.write((const char *)source, width * height * sizeof(Goldilocks::Element));
    // fw.write((const char *)nodes, numNodes * sizeof(Goldilocks::Element));
    // fw.close();

    uint64_t sourceOffset = sizeof(uint64_t) * 2;
    uint64_t nodesOffset = sourceOffset + width * height * sizeof(Goldilocks::Element);
    fw.close();
    writeFileParallel(constTreeFile, source, width * height * sizeof(Goldilocks::Element), sourceOffset);
    writeFileParallel(constTreeFile, nodes, numNodes * sizeof(Goldilocks::Element), nodesOffset);
}