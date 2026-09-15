#ifndef MULTIPLICITY_STREAM_CUH
#define MULTIPLICITY_STREAM_CUH

#include <cuda_runtime.h>
#include <map>
#include <mutex>
#include "multiplicity.cuh"
#include "multiplicity_kernel.cuh"
#include "multiplicity_plan.hpp"
#include "stream_commit.cuh"
#include "setup_ctx.hpp"

// Counting lookups during a STREAMING slot commit.
//
// The slot exists because gpu-mops has borrowed the first GPU's buffer, so there is no room to
// materialise cm1: the commit unpacks a few columns at a time and LDEs them in place. Nothing after
// the upload can read the witness, and nothing before it is unpacked -- which is why the scatter
// hooks in exactly between the two.
//
// Rather than teach the scatter to read bit-packed rows, this materialises cm1 a TILE of rows at a
// time and runs the ordinary kernel over the tile. Two facts from the plan make that sound:
//   * no job reads a later stage, so cm1 is all the scatter needs at commit time; and
//   * no job uses a row-shifted reference, so tiles are independent and need no halo.
// Both are asserted below rather than assumed -- an air that broke either would be miscounted
// silently, so it falls back to refusing the slot instead.
#define MUL_STREAM_TILE_ROWS (1u << 14)

struct MulStreamCtx {
    SetupCtx *setupCtx;
    uint64_t airgroupId, airId;
    uint64_t *acc;
    uint64_t *oob;
    const uint8_t *dColSource, *dColLane;
    const uint64_t *dTable;
    const uint64_t *constPols;          // device, this air's const pols
    // No publics / value pools / custom commits: they live in the aux trace, which a slot commit
    // does without. An air needing one is refused the slot (mulPlanStreamable) rather than counted
    // against a null base.

};

// One scratch buffer per device, grown to the widest air seen. Cached because a slot commit is on
// the critical path and cudaMalloc there would serialise against the copy engine.
inline std::map<int, std::pair<uint64_t*, size_t>>& mulStreamTiles() {
    static std::map<int, std::pair<uint64_t*, size_t>> m;
    return m;
}
inline std::mutex& mulStreamTilesMutex() { static std::mutex m; return m; }

inline uint64_t* mulStreamTile(int gpuId, size_t elems) {
    std::lock_guard<std::mutex> lk(mulStreamTilesMutex());
    auto& e = mulStreamTiles()[gpuId];
    if (e.second >= elems) return e.first;
    if (e.first != nullptr) cudaFree(e.first);
    e.first = nullptr;
    e.second = 0;
    if (cudaMalloc(&e.first, elems * sizeof(uint64_t)) != cudaSuccess) return nullptr;
    e.second = elems;
    return e.first;
}

// The hook handed to streamCommitPacked.
inline void mulStreamHook(const uint64_t *dPacked, const uint64_t *dWidths,
                          const StreamCommitDims &dims, cudaStream_t stream, void *user) {
    MulStreamCtx *c = (MulStreamCtx *)user;
    if (c == nullptr || c->acc == nullptr || mulDecoders().empty()) return;

    const MulPlan &plan = mulPlanFor(*c->setupCtx, c->airgroupId, c->airId);
    if (plan.jobs.empty()) return;     // the interpreter fallback cannot run here; see the caller
    // Asserted, not assumed: a later-stage read has nothing to read from here, and a row-shifted
    // one would cross a tile boundary. Both are false for every air today; an air that changed
    // must not be silently miscounted.
    if (!mulPlanStreamable(plan)) {
        zklog.error("multiplicity: air " + std::to_string(c->airgroupId) + "/"
                    + std::to_string(c->airId) + " reads " + mulSrcMaskNames(plan.srcMask)
                    + ", and a slot commit has only the const pols and the tile -- it must not take "
                    "the streaming path");
        exitProcess();
    }

    int gpuId = 0;
    CHECKCUDAERR(cudaGetDevice(&gpuId));
    const MulPlanDev dev = mulPlanDevice(plan, c->airgroupId, c->airId, gpuId);
    if (dev.jobs == nullptr) return;

    const uint64_t nRows = 1ull << dims.nBits;
    const uint64_t tileRows = std::min<uint64_t>(MUL_STREAM_TILE_ROWS, nRows);
    uint64_t *tile = mulStreamTile(gpuId, dims.nCols * tileRows);
    if (tile == nullptr) {
        zklog.error("multiplicity: no room for the streaming tile buffer on gpu "
                    + std::to_string(gpuId) + " -- this air's lookups would go uncounted");
        exitProcess();
    }

    // The jobs address cm1 as `col * nRows + row`; over a tile the stride is the tile's height, and
    // the kernel derives it from the domain size it is given.
    for (uint64_t begin = 0; begin < nRows; begin += tileRows) {
        const uint64_t rows = std::min(tileRows, nRows - begin);
        streamCommitUnpackTile(dPacked, dWidths, dims, begin, rows, 0, (uint32_t)dims.nCols,
                               tile, stream, c->dColSource, c->dColLane, c->dTable);
        // Only these two are resident on a slot; mulPlanStreamable above guarantees no job reads
        // any other, so the remaining bases are unreachable rather than merely unset.
        const uint64_t *bases[MUL_SRC_N] = { c->constPols, tile, nullptr, nullptr,
                                             nullptr, nullptr, nullptr, nullptr };
        // Degree-0 jobs contribute once per instance, not once per tile: only the first tile runs
        // them. Every other job is per-row and its rows are the tile's.
        mul_scatter_launch_tile(dev.jobs, (uint32_t)plan.jobs.size(), rows, begin, bases,
                                tileRows, nRows, c->acc, c->oob,
                                (c->airgroupId << 32) | c->airId, stream);
    }
    mul_note_scatter(gpuId, stream);
}

#endif
