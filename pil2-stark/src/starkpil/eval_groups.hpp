#ifndef EVAL_GROUPS_HPP
#define EVAL_GROUPS_HPP

// The per-air table of computeEvalsPairs (starks_gpu.cu), kept apart so AirInstanceInfo builds it
// without seeing the kernels.
#include <algorithm>
#include <map>
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

// Where a polynomial's ColMajor column starts in the device arena (extended domain) and which buffer it is in
// (src 0: cm, 1: custom, 2: fixed). `stage`, `stagePos` are the PolMap's; `commitId` only matters for custom.
struct DeviceColumn {
    uint16_t src;
    uint64_t col;
};

inline DeviceColumn deviceColumn(StarkInfo &si, EvMap::eType type, uint64_t commitId, uint64_t stage, uint64_t stagePos)
{
    const uint16_t src = type == EvMap::eType::cm ? 0 : type == EvMap::eType::custom ? 1 : 2;
    const std::string section = si.sectionName(type, stage, commitId);
    const Layout layout = src == 0 ? resolveLayout(si.starkStruct.nBits, si.mapSectionsN[section]) : fixedLayout();
    if (layout != Layout::ColMajor) { zklog.error("deviceColumn: sections must be ColMajor"); exitProcess(); }
    const uint64_t NExt = 1ULL << si.starkStruct.nBitsExt;
    return {src, si.mapOffsets[std::make_pair(section, true)] + stagePos * NExt};
}

struct EvalSlots {
    std::vector<EvalSlot> slots;      // by (dim, src, col, opening): a tile's shifted reads share rows
    std::vector<uint32_t> tileStart;  // nTiles + 1 bounds into slots; a tile never mixes dims
};

inline EvalSlots buildEvalSlots(StarkInfo &starkInfo)
{
    const uint64_t N = 1ULL << starkInfo.starkStruct.nBits;
    std::map<std::tuple<uint32_t, uint32_t, uint64_t>, std::vector<std::pair<int64_t, uint32_t>>> byPol;
    for (uint64_t k = 0; k < starkInfo.evMap.size(); k++) {
        const EvMap &ev = starkInfo.evMap[k];
        const PolMap &pol = starkInfo.evPol(ev);
        const DeviceColumn dc = deviceColumn(starkInfo, ev.type, ev.commitId, pol.stage, pol.stagePos);
        byPol[{(uint32_t)pol.dim, (uint32_t)dc.src, dc.col}].push_back({starkInfo.openingPoints[ev.openingPos], (uint32_t)k});
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
