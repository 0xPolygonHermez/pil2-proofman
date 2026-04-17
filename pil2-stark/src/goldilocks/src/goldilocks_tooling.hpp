#ifndef GOLDILOCKS_TOOLS_HPP
#define GOLDILOCKS_TOOLS_HPP

#include <cstdint>
#include "poseidon2_goldilocks.hpp"

// Total number of Goldilocks elements in a Merkle tree buffer (all levels
// including padding), given the number of leaves and the tree arity.
inline uint64_t getTreeNumElements(uint64_t degree, uint32_t arity = 2)
{
    uint64_t numNodes = degree;
    uint64_t nodesLevel = degree;
    while (nodesLevel > 1) {
        uint64_t extraZeros = (arity - (nodesLevel % arity)) % arity;
        numNodes += extraZeros;
        uint64_t nextN = (nodesLevel + (arity - 1)) / arity;
        numNodes += nextN;
        nodesLevel = nextN;
    }
    return numNodes * HASH_SIZE;
}

#endif
