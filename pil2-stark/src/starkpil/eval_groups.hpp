#ifndef EVAL_GROUPS_HPP
#define EVAL_GROUPS_HPP

// The per-air table of computeEvalsPairs (starks_gpu.cu), kept apart so AirInstanceInfo builds it
// without seeing the kernels.
#include <algorithm>
#include <map>
#include <stdexcept>
#include <tuple>
#include <vector>
#include "stark_info.hpp"
#include "goldilocks_trace_layout.cuh"

// A slot is one evaluation (polynomial, opening). computeEvalsPairs runs EVALS_TILE_SLOTS slots of one dim
// per block, sharing each Lagrange load; 2 measured fastest (4: +2% kernel time on ZisK).
#define EVALS_TILE_SLOTS 2

struct EvalSlot {
    uint64_t col;      // first element of the column in its buffer (ColMajor, extended domain)
    uint32_t src;      // 0: cm, 1: custom, 2: fixed
    uint32_t dim;
    uint32_t shift;    // opening point mod N
    uint32_t evalPos;
};

struct EvalSlots {
    std::vector<EvalSlot> slots;      // by (dim, src, col, opening): a tile's shifted reads share rows
    std::vector<uint32_t> tileStart;  // nTiles + 1 bounds into slots; a tile never mixes dims
};

inline EvalSlots buildEvalSlots(StarkInfo &starkInfo)
{
    const uint64_t N = 1ULL << starkInfo.starkStruct.nBits;
    const uint64_t NExt = 1ULL << starkInfo.starkStruct.nBitsExt;
    std::map<std::tuple<uint32_t, uint32_t, uint64_t>, std::vector<std::pair<int64_t, uint32_t>>> byPol;
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
        if (layout != Layout::ColMajor) throw std::runtime_error("buildEvalSlots: sections must be ColMajor");
        const uint64_t col = starkInfo.mapOffsets[std::make_pair(stage, true)] + pol.stagePos * NExt;
        byPol[{(uint32_t)pol.dim, src, col}].push_back({starkInfo.openingPoints[ev.openingPos], (uint32_t)k});
    }

    EvalSlots out;
    for (auto &[key, opens] : byPol) {
        std::sort(opens.begin(), opens.end());
        for (auto &[point, evalPos] : opens)
            out.slots.push_back({std::get<2>(key), std::get<1>(key), std::get<0>(key), (uint32_t)((uint64_t)point & (N - 1)), evalPos});
    }
    for (uint32_t i = 0; i < out.slots.size();) {
        out.tileStart.push_back(i);
        const uint32_t dim = out.slots[i].dim;
        uint32_t j = i + 1;
        while (j < out.slots.size() && j - i < EVALS_TILE_SLOTS && out.slots[j].dim == dim) j++;
        i = j;
    }
    out.tileStart.push_back((uint32_t)out.slots.size());
    return out;
}

#endif
