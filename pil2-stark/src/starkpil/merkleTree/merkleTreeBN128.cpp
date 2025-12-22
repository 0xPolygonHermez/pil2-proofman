
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
        nodes = (RawFrP::Element *)calloc(numNodes, sizeof(RawFrP::Element));
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
    nodes = (RawFrP::Element *)&source[width * height];
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
    return getMerkleProofLength() * arity * sizeof(RawFrP::Element);
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


void MerkleTreeBN128::getLevel(RawFrP::Element *level)
{
    if (last_level_verification != 0) {
        uint64_t n = height;
        uint64_t offset = 0;
        while (n > std::pow(arity, last_level_verification)) {
            n = (std::floor((n - 1) / arity) + 1);
            offset += n * arity;
        }

        std::memcpy(level, &nodes[offset], n * sizeof(RawFrP::Element));
        for (uint64_t i = n; i < std::pow(arity, last_level_verification); i++) {
            level[i] = RawFrP::field.zero();
        }
    }
}


void MerkleTreeBN128::getRoot(RawFrP::Element *root)
{
    std::memcpy(root, &nodes[numNodes - 1], sizeof(RawFrP::Element));
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

void MerkleTreeBN128::setNodes(RawFrP::Element *_nodes)
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

void MerkleTreeBN128::getGroupProof(RawFrP::Element *proof, uint64_t idx)
{
    assert(idx < height);

    Goldilocks::Element v[width];
    for (uint64_t i = 0; i < width; i++)
    {
        v[i] = getElement(idx, i);
    }
    std::memcpy(proof, &v[0], width * sizeof(Goldilocks::Element));
    void *proofCursor = (uint8_t *)proof + width * sizeof(Goldilocks::Element);

    genMerkleProof((RawFrP::Element *)proofCursor, idx, 0, height);
}

void MerkleTreeBN128::genMerkleProof(RawFrP::Element *proof, uint64_t idx, uint64_t offset, uint64_t n)
{
    if ((last_level_verification == 0 && n == 1) || (last_level_verification > 0 && n <= std::pow(arity, last_level_verification))) return;

    uint64_t nBitsArity = std::ceil(std::log2(arity));

    uint64_t nextIdx = idx >> nBitsArity;
    uint64_t si = idx ^ (idx & (arity - 1));

    std::memcpy(proof, &nodes[offset + si], arity * sizeof(RawFrP::Element));
    uint64_t nextN = (std::floor((n - 1) / arity) + 1);
    genMerkleProof(&proof[arity], nextIdx, offset + nextN * arity, nextN);
}

void MerkleTreeBN128::linearHash(RawFrP::Element* result, Goldilocks::Element* values)
{
    if (width > 4)
    {
        uint64_t widthRawFrElements = ceil((double)width / FIELD_EXTENSION);
        RawFrP::Element buff[widthRawFrElements]; 

        uint64_t nElementsGL = (width > FIELD_EXTENSION + 1) ? ceil((double)width / FIELD_EXTENSION) : 0;
        for (uint64_t j = 0; j < nElementsGL; j++)
        {
            uint64_t pending = width - j * FIELD_EXTENSION;
            uint64_t batch;
            (pending >= FIELD_EXTENSION) ? batch = FIELD_EXTENSION : batch = pending;
            for (uint64_t k = 0; k < batch; k++)
            {
                buff[j].v[k] = Goldilocks::toU64(values[j * FIELD_EXTENSION + k]);
            }
            RawFrP::field.toMontgomery(buff[j], buff[j]);
        }

        uint pending = nElementsGL;
        Poseidon_opt p;
        std::vector<RawFrP::Element> elements(arity + 1);
        while (pending > 0)
        {
            std::memset(&elements[0], 0, (arity + 1) * sizeof(RawFrP::Element));
            if (pending >= arity)
            {
                std::memcpy(&elements[1], &buff[nElementsGL - pending], arity * sizeof(RawFrP::Element));
                std::memcpy(&elements[0], &result[0], sizeof(RawFrP::Element));
                p.hash(elements, &result[0]);
                pending = pending - arity;
            }
            else if(custom) 
            {
                std::memcpy(&elements[1], &buff[nElementsGL - pending], pending * sizeof(RawFrP::Element));
                std::memcpy(&elements[0], &result[0], sizeof(RawFrP::Element));
                p.hash(elements, &result[0]);
                pending = 0;
            }
            else
            {
                std::vector<RawFrP::Element> elements_last(pending + 1);
                std::memcpy(&elements_last[1], &buff[nElementsGL - pending], pending * sizeof(RawFrP::Element));
                std::memcpy(&elements_last[0], &result[0], sizeof(RawFrP::Element));
                p.hash(elements_last, &result[0]);
                pending = 0;
            }
        } 
    } else {
        for (uint64_t k = 0; k < width; k++)
        {
            result[0].v[k] = Goldilocks::toU64(values[k]);
        }
        RawFrP::field.toMontgomery(result[0], result[0]);
    }
}

/*
 * LinearHash BN128
 */
void MerkleTreeBN128::linearHash()
{
    if (width > 4)
    {
        uint64_t widthRawFrElements = ceil((double)width / FIELD_EXTENSION);
        RawFrP::Element *buff = (RawFrP::Element *)calloc(height * widthRawFrElements, sizeof(RawFrP::Element));

    uint64_t nElementsGL = (width > FIELD_EXTENSION + 1) ? ceil((double)width / FIELD_EXTENSION) : 0;
#pragma omp parallel for
        for (uint64_t i = 0; i < height; i++)
        {
            for (uint64_t j = 0; j < nElementsGL; j++)
            {
                uint64_t pending = width - j * FIELD_EXTENSION;
                uint64_t batch;
                (pending >= FIELD_EXTENSION) ? batch = FIELD_EXTENSION : batch = pending;
                for (uint64_t k = 0; k < batch; k++)
                {
                    buff[i * nElementsGL + j].v[k] = Goldilocks::toU64(source[i * width + j * FIELD_EXTENSION + k]);
                }
                RawFrP::field.toMontgomery(buff[i * nElementsGL + j], buff[i * nElementsGL + j]);
            }
        }

#pragma omp parallel for
        for (uint64_t i = 0; i < height; i++)
        {
            uint pending = nElementsGL;
            Poseidon_opt p;
            std::vector<RawFrP::Element> elements(arity + 1);
            while (pending > 0)
            {
                std::memset(&elements[0], 0, (arity + 1) * sizeof(RawFrP::Element));
                if (pending >= arity)
                {
                    std::memcpy(&elements[1], &buff[i * nElementsGL + nElementsGL - pending], arity * sizeof(RawFrP::Element));
                    std::memcpy(&elements[0], &nodes[i], sizeof(RawFrP::Element));
                    p.hash(elements, &nodes[i]);
                    pending = pending - arity;
                }
                else if(custom) 
                {
                    std::memcpy(&elements[1], &buff[i * nElementsGL + nElementsGL - pending], pending * sizeof(RawFrP::Element));
                    std::memcpy(&elements[0], &nodes[i], sizeof(RawFrP::Element));
                    p.hash(elements, &nodes[i]);
                    pending = 0;
                }
                else
                {
                    std::vector<RawFrP::Element> elements_last(pending + 1);
                    assert(i * nElementsGL + nElementsGL - pending < height * nElementsGL); //to avoid out of bounds access compiler warning
                    std::memcpy(&elements_last[1], &buff[i * nElementsGL + nElementsGL - pending], pending * sizeof(RawFrP::Element));
                    std::memcpy(&elements_last[0], &nodes[i], sizeof(RawFrP::Element));
                    p.hash(elements_last, &nodes[i]);
                    pending = 0;
                }
            }
        }
        free(buff);
    }
    else
    {
#pragma omp parallel for
        for (uint64_t i = 0; i < height; i++)
        {
            for (uint64_t k = 0; k < width; k++)
            {
                nodes[i].v[k] = Goldilocks::toU64(source[i * width + k]);
            }
            RawFrP::field.toMontgomery(nodes[i], nodes[i]);
        }
    }
}

void MerkleTreeBN128::calculateRootFromProof(RawFrP::Element *value, std::vector<std::vector<RawFrP::Element>> &mp, uint64_t &idx, uint64_t offset) {
    if(offset == mp.size()) return;

    uint64_t nBitsArity = std::ceil(std::log2(arity));

    uint64_t currIdx = idx & (arity - 1);
    uint64_t nextIdx = idx >> nBitsArity;

    Poseidon_opt p;
    std::vector<RawFrP::Element> elements(arity + 1);
    std::memset(&elements[0], 0, (arity + 1) * sizeof(RawFrP::Element));

    for(uint64_t i = 0; i < arity; ++i) {
        std::memcpy(&elements[i], &mp[offset][i], sizeof(RawFrP::Element));
    }

    std::memcpy(&elements[currIdx], &value[0], sizeof(RawFrP::Element));
    p.hash(elements, &value[0]);

    calculateRootFromProof(value, mp, nextIdx, offset + 1);

}


bool MerkleTreeBN128::verifyGroupProof(RawFrP::Element* root, RawFrP::Element* level, std::vector<std::vector<RawFrP::Element>> &mp, uint64_t idx, std::vector<Goldilocks::Element> &v) {
    RawFrP::Element value[1];
    value[0] = RawFrP::field.zero();

    linearHash(value, v.data());

    uint64_t queryIdx = idx;
    calculateRootFromProof(&value[0], mp, queryIdx, 0);

    if (last_level_verification == 0) {
        if (!RawFrP::field.eq(root[0], value[0])) {
            return false;
        }
    } else {
        if (!RawFrP::field.eq(level[queryIdx], value[0])) {
            return false;
        }
    }

    return true;
}


void MerkleTreeBN128::merkelize()
{

    linearHash();

    RawFrP::Element *cursor = &nodes[0];
    uint64_t n256 = height;
    uint64_t nextN256 = floor((double)(n256 - 1) / arity) + 1;
    RawFrP::Element *cursorNext = &nodes[nextN256 * arity];
    while (n256 > 1)
    {
        uint64_t batches = ceil((double)n256 / arity);
#pragma omp parallel for
        for (uint64_t i = 0; i < batches; i++)
        {
            Poseidon_opt p;
            vector<RawFrP::Element> elements(arity + 1);
            std::memset(&elements[0], 0, (arity + 1) * sizeof(RawFrP::Element));
            uint numHashes = (i == batches - 1) ? n256 - i*arity : arity;
            std::memcpy(&elements[1], &cursor[i * arity], numHashes * sizeof(RawFrP::Element));
            p.hash(elements, &cursorNext[i]);
        }

        n256 = nextN256;
        nextN256 = floor((double)(n256 - 1) / arity) + 1;
        cursor = cursorNext;
        cursorNext = &cursor[nextN256 * arity];
    }
}

void MerkleTreeBN128::writeFile(std::string constTreeFile) {
    ofstream fw(constTreeFile.c_str(), std::fstream::out | std::fstream::binary);
    uint64_t nodesOffset = width * height * sizeof(Goldilocks::Element);
    fw.close();
    writeFileParallel(constTreeFile, (const char *)source, width * height * sizeof(Goldilocks::Element), 0);
    writeFileParallel(constTreeFile, (const char *)nodes, numNodes * sizeof(RawFrP::Element), nodesOffset);
}