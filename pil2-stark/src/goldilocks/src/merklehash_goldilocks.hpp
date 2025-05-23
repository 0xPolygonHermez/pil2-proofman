#ifndef MERKLEHASH_GOLDILOCKS
#define MERKLEHASH_GOLDILOCKS

#include <cassert>
#include <math.h> /* floor */
#include "goldilocks_base_field.hpp"
#include "goldilocks_cubic_extension.hpp"
#include "poseidon_goldilocks.hpp"

#define MERKLEHASHGOLDILOCKS_HEADER_SIZE 2
#define MERKLEHASHGOLDILOCKS_ARITY 3
class MerklehashGoldilocks
{
public:
    inline static void root(Goldilocks::Element *root, Goldilocks::Element *tree, uint64_t numElementsTree)
    {
        std::memcpy(root, &tree[numElementsTree - HASH_SIZE], HASH_SIZE * sizeof(Goldilocks::Element));
    }

    static void root(Goldilocks::Element (&root)[HASH_SIZE], Goldilocks::Element *tree, uint64_t numElementsTree)
    {
        std::memcpy(root, &tree[numElementsTree - HASH_SIZE], HASH_SIZE * sizeof(Goldilocks::Element));
    }

    static inline uint64_t getTreeNumElements(uint64_t degree, uint32_t arity=2)
    {
        uint64_t numNodes = degree;
        uint64_t nodesLevel = degree;
    
        while (nodesLevel > 1) {
            uint64_t extraZeros = (arity - (nodesLevel % arity)) % arity;
            numNodes += extraZeros;
            uint64_t nextN = (nodesLevel + (arity - 1))/arity;        
            numNodes += nextN;
            nodesLevel = nextN;
        }

        return numNodes * HASH_SIZE;
    };

    static inline uint64_t getTreeNumElementsArity(uint64_t degree)
    {
        uint64_t arity = MERKLEHASHGOLDILOCKS_ARITY;
        uint64_t numNodes = degree;
        uint64_t nodesLevel = degree;
        
        while (nodesLevel > 1) {
            uint64_t extraZeros = (arity - (nodesLevel % arity)) % arity;
            numNodes += extraZeros;
            uint64_t nextN = (nodesLevel + (arity - 1))/arity;        
            numNodes += nextN;
            nodesLevel = nextN;
        }


        return numNodes * HASH_SIZE;
    };
};

#endif