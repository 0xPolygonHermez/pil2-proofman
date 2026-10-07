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

// The rows of D a FRI block reads: opening o reads D[r0 - o B + t], t < nThreads, for the block's rows r0 + t.
// Those intervals, merged where they meet, are the segments kept in shared memory: segment s holds the rows
// r0 + offset + i, i < len, at shared index base + i, and opening o reads at shared index opBase[o] + t.
struct FriSegment
{
    int64_t offset;
    uint32_t base, len;
    uint64_t w;   // wExt^offset, filled in by the caller
};

struct FriWindow
{
    uint32_t nThreads = 0;   // 0 if not even one row per block fits
    uint64_t size = 0;       // rows in shared memory
    std::vector<FriSegment> segments;
    std::vector<uint32_t> opBase;
};

// The largest power of two <= 256 threads per block whose segments fit 48 KiB of shared memory (24 bytes a row).
inline FriWindow friWindow(const std::vector<int64_t> &openings, uint64_t extendBits, uint64_t domainSize)
{
    std::vector<int64_t> offsets;
    for (int64_t op : openings) offsets.push_back(-(op * (int64_t)(1ULL << extendBits)));
    std::sort(offsets.begin(), offsets.end());
    offsets.erase(std::unique(offsets.begin(), offsets.end()), offsets.end());
    for (uint64_t t = std::min<uint64_t>(256, domainSize); t > 0; t /= 2) {
        FriWindow w;
        w.nThreads = (uint32_t)t;
        for (int64_t off : offsets) {
            if (!w.segments.empty() && off <= w.segments.back().offset + (int64_t)w.segments.back().len) {
                FriSegment &last = w.segments.back();
                last.len = (uint32_t)std::max<int64_t>(last.len, off + (int64_t)t - last.offset);
            } else {
                w.segments.push_back(FriSegment{off, 0, (uint32_t)t, 0});
            }
        }
        for (FriSegment &seg : w.segments) {
            seg.base = (uint32_t)w.size;
            w.size += seg.len;
        }
        if (w.size * 24 > 48 * 1024) continue;
        for (int64_t op : openings) {
            const int64_t off = -(op * (int64_t)(1ULL << extendBits));
            const FriSegment &seg = *std::prev(std::upper_bound(w.segments.begin(), w.segments.end(), off,
                                                                [](int64_t v, const FriSegment &s) { return v < s.offset; }));
            w.opBase.push_back(seg.base + (uint32_t)(off - seg.offset));
        }
        return w;
    }
    return FriWindow{};
}

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
