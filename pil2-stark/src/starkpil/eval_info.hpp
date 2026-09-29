#ifndef EVAL_INFO_HPP
#define EVAL_INFO_HPP

#include <algorithm>
#include <cstdint>

// One (polynomial, opening point) pair of the evaluation map. Split out of
// stark_info.hpp so the FRI/evaluation device code can be compiled (and unit
// tested) without dragging in the json-parsing StarkInfo translation unit.
struct EvalInfo
{
    uint64_t type; // 0: cm, 1: custom, 2: fixed
    uint64_t offset;
    uint64_t stagePos;
    uint64_t stageCols;
    uint64_t dim;
    uint64_t openingPos;
    uint64_t evalPos;
};

// One term of the FRI polynomial (fri_expression.cuh): its column's first element in the buffer `src`
// selects (0: cm, 1: custom, 2: fixed), ColMajor over the extended domain.
struct FriTerm
{
    uint64_t col;
    uint32_t evalPos;
    uint16_t src;
    uint16_t dim;
};

// Threads per block: a power of two dividing the domain. nrowsPack is a power of two <= 256 on the
// prover path, the only one that reaches the FRI polynomial.
inline uint32_t friThreads(uint64_t nrowsPack, uint64_t domainSize)
{
    return (uint32_t)std::min<uint64_t>(std::max<uint64_t>(nrowsPack, 256), domainSize);
}

// Rows of D a block of nThreads reads, or 0 when they do not fit the default 48 KiB of shared
// memory (cubic elements, 24 bytes each) and the FRI polynomial takes the per-row fallback.
inline uint64_t friShiftedWindow(int64_t oMin, int64_t oMax, uint64_t extendBits, uint64_t nThreads)
{
    const uint64_t window = nThreads + ((uint64_t)(oMax - oMin) << extendBits);
    return window * 3 * sizeof(uint64_t) <= 48 * 1024 ? window : 0;
}

#endif
