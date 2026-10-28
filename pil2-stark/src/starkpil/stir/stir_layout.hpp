#ifndef STIR_LAYOUT_HPP
#define STIR_LAYOUT_HPP
// Single source of truth for the GPU STIR buffers: setMapOffsets, the pinned proof size, the
// driver and the tests all size from here.
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <functional>
#include <vector>
#include "stir.hpp"

struct StirLayout {
    // Aux-trace regions, element offsets relative to `base` (the caller adds the base).
    uint64_t ntt;                       // B: |L_1|·3 (|L_0|·3/2), column-major work buffer
    std::vector<uint64_t> treeSrc;      // T_i leaves, |L_i|·3 each
    std::vector<uint64_t> treeNodes;    // T_i nodes, numNodes(|L_i| >> k_i) each
    std::vector<uint64_t> treeNumNodes; // exact node count of T_i (the region is aligned up)
    uint64_t state;                     // small device state, see below
    uint64_t proof;                     // proof sub-blocks, see below
    uint64_t queries;                   // coset indices (u64), reused by every round: sized for t_0 >= t_i
    uint64_t total;                     // end of the last region
    // state sub-blocks (relative to `state`), each an E3 unless noted
    uint64_t rFold, rFoldNext, rComb, rho, challenge, rOut /*s E3*/, gHat /*d_1 E3*/,
             gPoints /*maxG E3*/, gValues /*maxG E3*/, gTmp /*maxG E3*/, zCoeffs /*maxG+1 E3, Z(X) = Π(X - a)*/, interpT /*maxG E3*/,
             powerSums /*maxG E3*/, nG /*1 u64*/, stateSize;
    // proof sub-blocks (relative to `proof`), in proof2pointer order
    uint64_t betas, finalPol, nonces;
    std::vector<uint64_t> ans, shake;   // per quotient round i = 1..M-1, (s + t_{i-1})·3 each
    uint64_t oodFlag;                   // 1 word, never serialized
    uint64_t proofSize;                 // includes oodFlag
    uint64_t maxG;                      // max_i (s + t_{i-1})
};


inline uint64_t stirAlign(uint64_t n) { return (n + 31) & ~uint64_t(31); }

inline StirLayout makeStirLayout(const StirParams &p, uint64_t base, const std::function<uint64_t(uint64_t)> &numNodes)
{
    const uint64_t M = p.M();
    assert(M >= 1);
    // Every query buffer (this region, fri_queries, fri_queries_perm, proof_queries blocks) is sized for t_0.
    for (uint64_t i = 0; i < M; i++) assert(p.numQueries[i] <= p.numQueries[0]);
    const uint64_t s = p.numOodSamples;
    StirLayout l;
    l.maxG = 0;
    for (uint64_t i = 1; i <= M; i++) l.maxG = std::max(l.maxG, s + p.numQueries[i - 1]);

    uint64_t at = base;
    auto take = [&](uint64_t n) { uint64_t o = at; at += stirAlign(n); return o; };

    l.ntt = take((uint64_t(1) << p.logDomainSizes[std::min<uint64_t>(1, M)]) * FIELD_EXTENSION);
    for (uint64_t i = 0; i < M; i++) {
        l.treeSrc.push_back(take((uint64_t(1) << p.logDomainSizes[i]) * FIELD_EXTENSION));
        l.treeNumNodes.push_back(numNodes(uint64_t(1) << (p.logDomainSizes[i] - p.logFoldingFactors[i])));
        l.treeNodes.push_back(take(l.treeNumNodes.back()));
    }

    uint64_t o = 0;
    auto sub = [&](uint64_t n) { uint64_t r = o; o += n; return r; };
    const uint64_t d1 = uint64_t(1) << p.logDegrees[std::min<uint64_t>(1, M)];
    l.rFold = sub(FIELD_EXTENSION); l.rFoldNext = sub(FIELD_EXTENSION); l.rComb = sub(FIELD_EXTENSION);
    l.rho = sub(FIELD_EXTENSION); l.challenge = sub(HASH_SIZE);
    l.rOut = sub(s * FIELD_EXTENSION);
    l.gHat = sub(d1 * FIELD_EXTENSION);
    l.gPoints = sub(l.maxG * FIELD_EXTENSION); l.gValues = sub(l.maxG * FIELD_EXTENSION);
    l.gTmp = sub(l.maxG * FIELD_EXTENSION);
    l.zCoeffs = sub((l.maxG + 1) * FIELD_EXTENSION); l.interpT = sub(l.maxG * FIELD_EXTENSION);
    l.powerSums = sub(l.maxG * FIELD_EXTENSION);
    l.nG = sub(1);
    l.stateSize = o;
    l.state = take(l.stateSize);

    o = 0;
    l.betas = sub((M - 1) * s * FIELD_EXTENSION);
    l.finalPol = sub((uint64_t(1) << p.logDegrees[M]) * FIELD_EXTENSION);
    l.nonces = sub(M);
    for (uint64_t i = 1; i < M; i++) l.ans.push_back(sub((s + p.numQueries[i - 1]) * FIELD_EXTENSION));
    for (uint64_t i = 1; i < M; i++) l.shake.push_back(sub((s + p.numQueries[i - 1]) * FIELD_EXTENSION));
    l.oodFlag = sub(1);
    l.proofSize = o;
    l.proof = take(l.proofSize);

    l.queries = take(p.numQueries[0]);
    l.total = at;
    return l;
}
#endif
