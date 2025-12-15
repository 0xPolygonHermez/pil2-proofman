#include "merkleTreeGL.hpp"
#include <cassert>
#include <algorithm> // std::max



MerkleTreeGL::MerkleTreeGL(uint64_t _arity, uint64_t _last_level_verification, bool _custom, uint64_t _height, uint64_t _width, bool allocateSource, bool allocateNodes) : height(_height), width(_width)
{
    arity = _arity;
    last_level_verification = _last_level_verification;
    numNodes = getNumNodes(height);
    custom = _custom;

    if(allocateSource) {
        initSource();
    }

    if(allocateNodes) {
        initNodes();
    }
};

MerkleTreeGL::MerkleTreeGL(uint64_t _arity, uint64_t _last_level_verification, bool _custom, Goldilocks::Element *tree, uint64_t height_, uint64_t width_)
{
    width = width_;
    height = height_;
    source = tree;
    arity = _arity;
    last_level_verification = _last_level_verification;
    custom = _custom;
    numNodes = getNumNodes(height);
    nodes = &tree[height * width];
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
        return (uint64_t)ceil(std::log2(height) / std::log2(arity)) - last_level_verification;
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

void MerkleTreeGL::getLevel(Goldilocks::Element *level)
{
    if (last_level_verification != 0) {
        uint64_t n = height;
        uint64_t offset = 0;
        while (n > std::pow(arity, last_level_verification)) {
            n = (n + (arity - 1))/arity;
            offset += n * arity * nFieldElements;
        }

        std::memcpy(level, &nodes[offset], n * nFieldElements * sizeof(Goldilocks::Element));
        for (uint64_t i = n; i < std::pow(arity, last_level_verification); i++) {
            for (uint64_t j = 0; j < nFieldElements; j++) {
                level[i * nFieldElements + j] = Goldilocks::zero();
            }
        }
    }
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
        zklog.error("MerkleTreeGL: Nodes were allocated when initializing");
        exitProcess();
        exit(-1);
    }
    nodes = _nodes;
}

void MerkleTreeGL::initSource() {
    source = (Goldilocks::Element *)malloc(height * width * sizeof(Goldilocks::Element));
    isSourceAllocated = true;
}

void MerkleTreeGL::initNodes() {
    nodes = (Goldilocks::Element *)malloc(numNodes * sizeof(Goldilocks::Element));
    isNodesAllocated = true;
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

void MerkleTreeGL::genMerkleProof(Goldilocks::Element *proof, uint64_t idx, uint64_t offset, uint64_t n)
{
    if ((last_level_verification == 0 && n == 1) || (last_level_verification > 0 && (n <= std::pow(arity, last_level_verification)))) return;
    
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

bool MerkleTreeGL::verifyGroupProof(Goldilocks::Element* root, Goldilocks::Element* level, std::vector<std::vector<Goldilocks::Element>> &mp, uint64_t idx, std::vector<Goldilocks::Element> &v) {
    Goldilocks::Element value[4] = { Goldilocks::zero(), Goldilocks::zero(), Goldilocks::zero(), Goldilocks::zero() };

    switch(arity) {
        case 2:
            Poseidon2Goldilocks<8>::linear_hash_seq(value, v.data(), v.size());
            break;
        case 3:
            Poseidon2Goldilocks<12>::linear_hash_seq(value, v.data(), v.size());
            break;
        case 4:
            Poseidon2Goldilocks<16>::linear_hash_seq(value, v.data(), v.size());
            break;
        default:
            zklog.error("MerkleTreeGL::verifyGroupProof: Unsupported arity");
            exitProcess();
            exit(-1);
    }
    

    uint64_t queryIdx = idx;
    calculateRootFromProof(value, mp, queryIdx, 0);

    if (last_level_verification == 0) {
        for(uint64_t i = 0; i < nFieldElements; ++i) {
            if(Goldilocks::toU64(value[i]) != Goldilocks::toU64(root[i])) {
                return false;
            }
        }
    } else {
        for(uint64_t i = 0; i < nFieldElements; ++i) {
            if(Goldilocks::toU64(value[i]) != Goldilocks::toU64(level[queryIdx * nFieldElements + i])) {
                return false;
            }
        }
    }

    return true;
}

void MerkleTreeGL::calculateRootFromProof(Goldilocks::Element (&value)[4], std::vector<std::vector<Goldilocks::Element>> &mp, uint64_t &idx, uint64_t offset) {
    if(offset == mp.size()) return;

    uint64_t currIdx = idx % arity;
    idx = idx / arity;

    
    switch(arity) {
        case 2: {
            Goldilocks::Element inputs[Poseidon2Goldilocks<8>::SPONGE_WIDTH];
            for(uint64_t i = 0; i < Poseidon2Goldilocks<8>::SPONGE_WIDTH; ++i) {
                inputs[i] = Goldilocks::zero();
            }
            uint64_t p = 0;
            for(uint64_t i = 0; i < arity; ++i) {
                if (i == currIdx) continue;
                std::memcpy(&inputs[i*nFieldElements], &mp[offset][nFieldElements * (p++)], nFieldElements * sizeof(Goldilocks::Element));
            }
            std::memcpy(&inputs[currIdx*nFieldElements], value, nFieldElements * sizeof(Goldilocks::Element));
            Poseidon2Goldilocks<8>::hash_seq(value, inputs);
            break;
        }
        case 3: {
            Goldilocks::Element inputs[Poseidon2Goldilocks<12>::SPONGE_WIDTH];
            for(uint64_t i = 0; i < Poseidon2Goldilocks<12>::SPONGE_WIDTH; ++i) {
                inputs[i] = Goldilocks::zero();
            }
            uint64_t p = 0;
            for(uint64_t i = 0; i < arity; ++i) {
                if (i == currIdx) continue;
                std::memcpy(&inputs[i*nFieldElements], &mp[offset][nFieldElements * (p++)], nFieldElements * sizeof(Goldilocks::Element));
            }
            std::memcpy(&inputs[currIdx*nFieldElements], value, nFieldElements * sizeof(Goldilocks::Element));
            Poseidon2Goldilocks<12>::hash_seq(value, inputs);
            break;
        }
        case 4: {
            Goldilocks::Element inputs[Poseidon2Goldilocks<16>::SPONGE_WIDTH];
            for(uint64_t i = 0; i < Poseidon2Goldilocks<16>::SPONGE_WIDTH; ++i) {
                inputs[i] = Goldilocks::zero();
            }
            uint64_t p = 0;
            for(uint64_t i = 0; i < arity; ++i) {
                if (i == currIdx) continue;
                std::memcpy(&inputs[i*nFieldElements], &mp[offset][nFieldElements * (p++)], nFieldElements * sizeof(Goldilocks::Element));
            }
            std::memcpy(&inputs[currIdx*nFieldElements], value, nFieldElements * sizeof(Goldilocks::Element));
            Poseidon2Goldilocks<16>::hash_seq(value, inputs);
            break;
        }
        default:
            zklog.error("MerkleTreeGL::calculateRootFromProof: Unsupported arity");
            exitProcess();
            exit(-1);
    }

    calculateRootFromProof(value, mp, idx, offset + 1);
}


void MerkleTreeGL::merkelize()
{
    switch(arity) {
        case 2:
            #ifdef __AVX512__
                Poseidon2Goldilocks<8>::merkletree_batch_avx512(nodes, source, width, height, arity);
            #elif defined(__AVX2__)
                Poseidon2Goldilocks<8>::merkletree_batch_avx(nodes, source, width, height, arity);
            #else
                Poseidon2Goldilocks<8>::merkletree_seq(nodes, source, width, height, arity);
            #endif
            break;
        case 3:
            #ifdef __AVX512__
                Poseidon2Goldilocks<12>::merkletree_batch_avx512(nodes, source, width, height, arity);
            #elif defined(__AVX2__)
                Poseidon2Goldilocks<12>::merkletree_batch_avx(nodes, source, width, height, arity);
            #else
                Poseidon2Goldilocks<12>::merkletree_seq(nodes, source, width, height, arity);
            #endif
            break;
        case 4:
            #ifdef __AVX512__
                Poseidon2Goldilocks<16>::merkletree_batch_avx512(nodes, source, width, height, arity);
            #elif defined(__AVX2__)
                Poseidon2Goldilocks<16>::merkletree_batch_avx(nodes, source, width, height, arity);
            #else
                Poseidon2Goldilocks<16>::merkletree_seq(nodes, source, width, height, arity);
            #endif
            break;
        default:
            zklog.error("MerkleTreeGL::merkelize: Unsupported arity");
            exitProcess();
            exit(-1);   
    }
}

void MerkleTreeGL::writeFile(std::string constTreeFile)
{
    Goldilocks::Element *root = &nodes[numNodes - 4];
    ofstream fw(constTreeFile.c_str(), std::fstream::out | std::fstream::binary);
    uint64_t nodesOffset = width * height * sizeof(Goldilocks::Element);
    fw.close();
    writeFileParallel(constTreeFile, source, width * height * sizeof(Goldilocks::Element), 0);
    writeFileParallel(constTreeFile, nodes, numNodes * sizeof(Goldilocks::Element), nodesOffset);
}