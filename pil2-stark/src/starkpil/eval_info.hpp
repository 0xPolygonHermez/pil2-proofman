#ifndef EVAL_INFO_HPP
#define EVAL_INFO_HPP

#include <algorithm>
#include <cstdint>
#include <map>
#include <vector>

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
// selects (0: cm, 1: custom, 2: fixed), ColMajor over the extended domain. vf2Exp: the polynomial's
// first appearance in the eval map.
struct FriTerm
{
    uint64_t col;
    uint32_t evalPos;
    uint16_t src;
    uint16_t dim;
    uint32_t vf2Exp;
};

// The polynomials grouped by the set of openings they are in: group g is cols[colStart[g], colStart[g + 1]),
// and opening o is in the groups opGroups[opStart[o], opStart[o + 1]).
struct FriColGroups
{
    std::vector<FriTerm> cols;
    std::vector<uint32_t> colStart{0}, opStart, opGroups;
};

// From the opening-major terms of nPols polynomials.
inline FriColGroups buildFriColGroups(const std::vector<FriTerm> &terms, const std::vector<uint64_t> &termStart, uint64_t nPols)
{
    const uint64_t nOpenings = termStart.size() - 1;
    std::vector<FriTerm> polTerm(nPols);
    std::vector<std::vector<uint32_t>> polOps(nPols);
    for (uint32_t o = 0; o < nOpenings; o++) {
        for (uint64_t j = termStart[o]; j < termStart[o + 1]; j++) {
            polTerm[terms[j].vf2Exp] = terms[j];
            polOps[terms[j].vf2Exp].push_back(o);
        }
    }
    std::map<std::vector<uint32_t>, std::vector<uint32_t>> bySet;
    for (uint32_t c = 0; c < nPols; c++) bySet[polOps[c]].push_back(c);
    FriColGroups r;
    std::vector<std::vector<uint32_t>> byOpening(nOpenings);
    for (const auto &[ops, pols] : bySet) {
        for (uint32_t o : ops) byOpening[o].push_back(r.colStart.size() - 1);
        for (uint32_t c : pols) r.cols.push_back(polTerm[c]);
        r.colStart.push_back(r.cols.size());
    }
    r.opStart.push_back(0);
    for (const auto &g : byOpening) {
        r.opGroups.insert(r.opGroups.end(), g.begin(), g.end());
        r.opStart.push_back(r.opGroups.size());
    }
    return r;
}

#endif
