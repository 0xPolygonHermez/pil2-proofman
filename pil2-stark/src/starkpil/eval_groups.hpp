#ifndef EVAL_GROUPS_HPP
#define EVAL_GROUPS_HPP

// The per-air table of computeEvalsShifted (starks_gpu.cu), kept apart so AirInstanceInfo builds it
// without seeing the kernels.
#include <algorithm>
#include <map>
#include <stdexcept>
#include <tuple>
#include <vector>
#include "stark_info.hpp"
#include "goldilocks_trace_layout.cuh"

// Openings of one polynomial per block: 3-4 measured fastest at the zisk shapes, more costs registers.
#define EVALS_GROUP_OPENINGS 4

struct EvalGroup {
    uint64_t col;    // first element of the column in its buffer (ColMajor, extended domain)
    uint32_t src;    // 0: cm, 1: custom, 2: fixed
    uint32_t dim;
    uint32_t nOpen;
    uint32_t shift[EVALS_GROUP_OPENINGS];   // opening point mod N
    uint32_t evalPos[EVALS_GROUP_OPENINGS];
};

// A polynomial's openings in ascending order, so a group's shifted reads stay in neighbouring rows.
inline std::vector<EvalGroup> buildEvalGroups(StarkInfo &starkInfo)
{
    const uint64_t N = 1ULL << starkInfo.starkStruct.nBits;
    const uint64_t NExt = 1ULL << starkInfo.starkStruct.nBitsExt;
    std::map<std::tuple<uint32_t, uint64_t, uint32_t>, std::vector<std::pair<int64_t, uint32_t>>> byPol;
    for (uint64_t k = 0; k < starkInfo.evMap.size(); k++) {
        const EvMap &ev = starkInfo.evMap[k];
        const uint32_t src = ev.type == EvMap::eType::cm ? 0 : ev.type == EvMap::eType::custom ? 1 : 2;
        const PolMap &pol = src == 0 ? starkInfo.cmPolsMap[ev.id]
                          : src == 1 ? starkInfo.customCommitsMap[ev.commitId][ev.id]
                                     : starkInfo.constPolsMap[ev.id];
        const std::string stage = src == 0 ? "cm" + std::to_string(pol.stage)
                                : src == 1 ? starkInfo.customCommits[pol.commitId].name + "0"
                                           : "const";
        const Layout layout = src == 0 ? resolveLayout(starkInfo.starkStruct.nBits, starkInfo.mapSectionsN[stage])
                                       : fixedLayout();
        if (layout != Layout::ColMajor) throw std::runtime_error("buildEvalGroups: sections must be ColMajor");
        const uint64_t col = starkInfo.mapOffsets[std::make_pair(stage, true)] + pol.stagePos * NExt;
        byPol[{src, col, (uint32_t)pol.dim}].push_back({starkInfo.openingPoints[ev.openingPos], (uint32_t)k});
    }

    std::vector<EvalGroup> groups;
    for (auto &[key, opens] : byPol) {
        std::sort(opens.begin(), opens.end());
        // Even splits: 17 openings as 3+3+3+4+4, not 4+4+4+4+1.
        const size_t nGroups = (opens.size() + EVALS_GROUP_OPENINGS - 1) / EVALS_GROUP_OPENINGS;
        for (size_t gi = 0, at = 0; gi < nGroups; gi++) {
            EvalGroup g{};
            std::tie(g.src, g.col, g.dim) = key;
            g.nOpen = (uint32_t)((opens.size() - at) / (nGroups - gi));
            for (uint32_t o = 0; o < g.nOpen; o++, at++) {
                g.shift[o] = (uint32_t)((uint64_t)opens[at].first & (N - 1));
                g.evalPos[o] = opens[at].second;
            }
            groups.push_back(g);
        }
    }
    return groups;
}

#endif
