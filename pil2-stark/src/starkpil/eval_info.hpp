#ifndef EVAL_INFO_HPP
#define EVAL_INFO_HPP

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

#endif
