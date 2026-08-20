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

#endif
