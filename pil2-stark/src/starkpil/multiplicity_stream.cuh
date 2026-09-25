#ifndef MULTIPLICITY_STREAM_CUH
#define MULTIPLICITY_STREAM_CUH

#include <cuda_runtime.h>
#include <map>
#include <mutex>
#include "multiplicity.cuh"
#include "multiplicity_kernel.cuh"
#include "multiplicity_plan.hpp"
#include "stream_commit.cuh"
#include "witness_hints_slot.cuh"
#include "setup_ctx.hpp"
#include "gpu_timer.cuh"

// Counting lookups during a STREAMING slot commit.
//
// The slot exists because gpu-mops has borrowed the first GPU's buffer, so there is no room to
// materialise cm1: the commit unpacks a few columns at a time and LDEs them in place. Nothing after
// the upload can read the witness, and nothing before it is unpacked -- which is why the scatter
// hooks in exactly between the two.
//
// Rather than teach the scatter to read bit-packed rows, this materialises cm1 a TILE of rows at a
// time and runs the ordinary kernel over the tile. One fact from the plan makes that sound: no job
// reads a later stage, so cm1 is all the scatter needs at commit time. A `'`-shifted cm1 reference
// is served by giving the tile a halo, and only a reach wider than MUL_TILE_MAX_HALO is refused.
// Both are checked in mulPlanStreamable rather than assumed -- an air that broke either would be
// miscounted silently, so it refuses the slot instead.
#define MUL_STREAM_TILE_ROWS (1u << 14)

struct MulStreamCtx {
    SetupCtx *setupCtx;
    uint64_t airgroupId, airId;
    uint64_t *acc;
    uint64_t *oob;
    const uint8_t *dColSource, *dColLane;
    const uint64_t *dTable;
    // Device, this air's const pols UNPACKED (column-major, `col * nRows + row`), expanded into
    // mulStreamConst. d_constPols is bit-packed behind a header and must not be handed to the jobs.
    const uint64_t *constPols;
    uint64_t slotIdx;                   // which streaming-commit slot this call owns
    // Publics and the value pools, packed host-side into one contiguous window (consecutive from
    // `publics` in the aux trace), capped at PINNED_AUX_VALUES_MAX. The hook uploads it into its own
    // buffer. Null = no value reads, which mulPlanStreamable must have excluded.
    const uint64_t *hostVals;
    uint64_t nVals;                     // words in the window
    // Already on the device (the air's witness_calc hints read the same pools): used instead of uploading.
    const uint64_t *dVals;
    // Stage-1 columns the prover computes rather than the witness carrying them. The tile below
    // is unpacked from the packed rows, which predate them, so a lookup reading one would count
    // against a stale value -- they are patched in before the scatter. Null when the air has no
    // witness_calc hints. See witness_hints_slot.hpp.
    const uint64_t *hintSide;
    const uint32_t *hintDestCols, *hintDestSlots;
    uint32_t hintNDest;
    uint64_t offPublics, offAirValues, offProofValues, offAirgroupValues;  // words into it
    // Custom commits are still absent: they are trace-sized, not value-sized.
    // Slot scatter timer, the counterpart of MUL_SCATTER_KERNEL on the legacy path.
    TimerGPU *timer;
};

// One scratch buffer per (device, slot), grown to the widest air seen on that slot. Cached because
// a slot commit is on the critical path and cudaMalloc there would serialise against the copy
// engine. Keyed by slotIdx, not just gpuId: commit_witness_streaming_gpu is documented safe to call
// concurrently on distinct slots, and a per-GPU-only key handed the same pointer to two slots'
// concurrently-running async unpack/scatter kernels, plus a cudaFree-while-in-use on resize. Each
// slot's own commit is synchronous (streamCommitPacked syncs its stream before returning), so
// growing a given slot's tile between two calls on that same slot is safe -- the prior kernels
// using the old buffer have already completed by the time the next call could resize it.
using MulStreamBufs = std::map<std::pair<int, uint64_t>, std::pair<uint64_t*, size_t>>;

inline std::mutex& mulStreamBufsMutex() { static std::mutex m; return m; }

inline uint64_t* mulStreamBuf(MulStreamBufs& bufs, int gpuId, uint64_t slotIdx, size_t elems) {
    std::lock_guard<std::mutex> lk(mulStreamBufsMutex());
    auto& e = bufs[{gpuId, slotIdx}];
    if (e.second >= elems) return e.first;
    if (e.first != nullptr) cudaFree(e.first);
    e.first = nullptr;
    e.second = 0;
    if (cudaMalloc(&e.first, elems * sizeof(uint64_t)) != cudaSuccess) return nullptr;
    e.second = elems;
    return e.first;
}

inline uint64_t* mulStreamTile(int gpuId, uint64_t slotIdx, size_t elems) {
    static MulStreamBufs bufs;
    return mulStreamBuf(bufs, gpuId, slotIdx, elems);
}

// Where the caller expands this air's const pols before the commit; same caching rationale as the
// tile. Sized nConstants * N, which is small next to the trace (zisk Main: 2 columns, 64 MiB).
inline uint64_t* mulStreamConst(int gpuId, uint64_t slotIdx, size_t elems) {
    static MulStreamBufs bufs;
    return mulStreamBuf(bufs, gpuId, slotIdx, elems);
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
    // A `'`-shifted cm1 reference reads off its own row, so the tile carries that many extra rows
    // on each side. zisk Keccakf reaches 4, all of it inside compiled programs; most airs need 0.
    // The halo wraps at the domain edges, which is why the tile is filled in up to three passes
    // and the kernel's trace addressing is modular.
    const uint64_t halo = std::min<uint64_t>(plan.traceHalo, nRows / 2);
    const uint64_t tileH = tileRows + 2 * halo;
    // The window rides at the end of the tile buffer rather than in a buffer of its own: same
    // lifetime, same per-(device, slot) key, one fewer allocation.
    uint64_t *tile = mulStreamTile(gpuId, c->slotIdx, dims.nCols * tileH + c->nVals);
    if (tile == nullptr) {
        zklog.error("multiplicity: no room for the streaming tile buffer on gpu "
                    + std::to_string(gpuId) + " -- this air's lookups would go uncounted");
        exitProcess();
    }

    const uint64_t *vals = c->dVals;
    if (vals == nullptr && c->hostVals != nullptr && c->nVals != 0) {
        uint64_t *dst = tile + dims.nCols * tileH;
        CHECKCUDAERR(cudaMemcpyAsync(dst, c->hostVals, c->nVals * sizeof(uint64_t),
                                     cudaMemcpyHostToDevice, stream));
        vals = dst;
    }

    // The jobs address cm1 as `col * nRows + row`; over a tile the stride is the tile's height.
    // That is `rows`, not `tileRows`: the unpack lays its columns out at the height it was asked
    // for, so a short final tile would otherwise be read at the nominal stride. Every domain is a
    // power of two today and MUL_STREAM_TILE_ROWS divides them all, so no tile is short -- but the
    // two strides have to be the same variable, not two that happen to agree.
    for (uint64_t begin = 0; begin < nRows; begin += tileRows) {
        const uint64_t rows = std::min(tileRows, nRows - begin);
        // The window this tile serves: `rows` counted rows, plus the halo on each side. It is a
        // cyclic range, so it is filled in up to three contiguous passes -- all writing at the
        // whole tile's stride, not their own row count.
        const uint64_t winBegin = (begin + nRows - halo) & (nRows - 1);
        const uint64_t winRows = rows + 2 * halo;
        for (uint64_t done = 0; done < winRows; ) {
            const uint64_t from = (winBegin + done) & (nRows - 1);
            const uint64_t take = std::min(winRows - done, nRows - from);
            streamCommitUnpackTile(dPacked, dWidths, dims, from, take, 0, (uint32_t)dims.nCols,
                                   tile, stream, c->dColSource, c->dColLane, c->dTable,
                                   tileH, done);
            done += take;
        }
        // The prover's own stage-1 columns over the witness's, before anything counts them.
        if (c->hintNDest != 0)
            for (uint64_t done = 0; done < winRows; ) {
                const uint64_t from = (winBegin + done) & (nRows - 1);
                const uint64_t take = std::min(winRows - done, nRows - from);
                slotHintPatchLaunch(tile + done, 0, (uint32_t)dims.nCols, take, from, nRows,
                                    c->hintSide, c->hintDestCols, c->hintDestSlots, c->hintNDest,
                                    stream, tileH);
                done += take;
            }
        // Const pols, the tile, and the value window the caller staged. Aux and the custom
        // commits stay unreachable -- mulPlanStreamable guarantees no job reads them.
        const uint64_t *bases[MUL_SRC_N] = {
            c->constPols, tile, nullptr,
            vals ? vals + c->offPublics        : nullptr,
            vals ? vals + c->offAirValues      : nullptr,
            vals ? vals + c->offProofValues    : nullptr,
            vals ? vals + c->offAirgroupValues : nullptr,
            nullptr, nullptr, nullptr };   // no custom commits; the tile is already unpacked
        // Degree-0 jobs contribute once per instance, not once per tile: only the first tile runs
        // them. Every other job is per-row and its rows are the tile's.
        if (c->timer) c->timer->startCategory("MUL_SCATTER_TILE");
        mul_scatter_launch_tile(dev.jobs, (uint32_t)plan.jobs.size(), rows, winBegin, begin, bases,
                                tileH, nRows, c->acc, c->oob,
                                (c->airgroupId << 32) | c->airId, dev.prog, stream);
        if (c->timer) c->timer->stopCategory("MUL_SCATTER_TILE");
    }
    mul_note_scatter(gpuId, stream);
}

#endif
