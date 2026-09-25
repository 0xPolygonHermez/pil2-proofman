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
// The slot has no room to materialise cm1: the commit unpacks a few columns at a time and LDEs
// them in place, so the scatter must hook in between upload and unpack. Its program is rewritten
// onto the bit-packed rows (mulPackedProgramFor), so one launch covers the domain and `'`-shifted
// reads wrap. Sound only because no job reads a later stage; mulPlanStreamable checks that and
// refuses the slot otherwise.

struct MulStreamCtx {
    // Packed rows are column-major (word w of row r at packed[w * nRows + r]) when
    // StreamCommitDims::colMajorForHook is set, which keeps the +-row shifts in one cache line.
    uint32_t packedColMajor = 0;
    SetupCtx *setupCtx;
    uint64_t airgroupId, airId;
    uint64_t *acc;
    uint64_t *oob;
    const uint64_t *dTable;
    // Device, this air's const pols UNPACKED (column-major, `col * nRows + row`), expanded into
    // mulStreamConst. d_constPols is bit-packed behind a header and must not be handed to the jobs.
    const uint64_t *constPols;
    // Publics and the value pools, one contiguous device window (uploaded by the slot hook), capped
    // at PINNED_AUX_VALUES_MAX.
    const uint64_t *dVals;
    uint64_t offPublics, offAirValues, offProofValues, offAirgroupValues;  // words into it
    // Stage-1 columns the prover computes; the packed rows predate them, so the rewrite reads them
    // from this buffer. Null when the air has no witness_calc hints (see witness_hints_slot.hpp).
    const uint64_t *hintSide;
    // Host-side packing layout for rewriting the program onto the packed rows. For an INDEXED air the
    // column map says whether a value sits in the compact row or the instruction table, and which lane.
    const std::vector<uint64_t> *widths = nullptr;
    const std::vector<uint8_t> *colSource = nullptr, *colLane = nullptr;
    const SlotHintPlan *hintPlan = nullptr;
    uint64_t indexBits = 0, wordsPerEntry = 0, numEntries = 0, lanes = 0;
    // Slot scatter timer, the counterpart of MUL_SCATTER_KERNEL on the legacy path.
    TimerGPU *timer;
};

// Rewrite the scatter to read the packed rows directly.
//
// Column addressing must mirror unpackIndexedRow bit for bit. Runtime columns start past the
// index header, a table column at the start of its lane's entry; non-indexed is the degenerate
// case. On refusal the CALLER declines the slot; the hook itself cannot refuse.
struct MulPackedProg { const MulInsnDev* prog = nullptr; bool ok = false; };

inline MulPackedProg mulPackedProgramFor(const MulPlan& plan, const MulStreamCtx& c,
                                         uint64_t airgroupId, uint64_t airId, int gpuId) {
    static std::map<std::tuple<uint64_t,uint64_t,int>, MulPackedProg> cache;
    static std::mutex mtx;
    std::lock_guard<std::mutex> lock(mtx);
    auto key = std::make_tuple(airgroupId, airId, gpuId);
    auto it = cache.find(key);
    if (it != cache.end()) return it->second;

    MulPackedProg r;
    const std::vector<uint64_t>* w = c.widths;
    if (w == nullptr || w->empty() || plan.prog.empty()) { cache[key] = r; return r; }
    const bool indexed = c.colSource != nullptr && !c.colSource->empty();
    if (indexed && (c.colSource->size() < w->size() || c.numEntries == 0 || c.indexBits == 0)) {
        cache[key] = r; return r;
    }
    const uint64_t nLanes = c.lanes ? c.lanes : 1;
    auto laneOf = [&](size_t i) -> uint8_t {
        return (c.colLane != nullptr && i < c.colLane->size()) ? (*c.colLane)[i] : 0;
    };
    auto fromTable = [&](size_t i) { return indexed && (*c.colSource)[i] != 0; };

    // Bit offset of each column, in the stream it actually belongs to.
    std::vector<uint64_t> bitOf(w->size(), 0);
    { uint64_t cur = nLanes * c.indexBits;                       // runtime columns
      for (size_t i = 0; i < w->size(); ++i)
          if (!fromTable(i)) { bitOf[i] = cur; cur += (*w)[i]; } }
    for (uint64_t l = 0; l < nLanes; ++l) {                      // one entry stream per lane
        uint64_t cur = 0;
        for (size_t i = 0; i < w->size(); ++i)
            if (fromTable(i) && laneOf(i) == l) { bitOf[i] = cur; cur += (*w)[i]; }
    }

    std::map<uint32_t,uint32_t> hintSlotOf;
    if (c.hintPlan != nullptr)
        for (const auto& op : c.hintPlan->ops) hintSlotOf[op.destCol] = op.destSlot;

    std::vector<MulInsnDev> prog = plan.prog;
    for (auto& in : prog)
        for (MulOperandDev* o : {&in.a, &in.b}) {
            if (o->kind != MUL_OPND_COL) continue;
            MulTermDev& t = o->term;
            if (MUL_SRC_IS_UNIFORM(t.src) || t.src == MUL_SRC_CONST) continue;   // served as-is
            if (t.src != MUL_SRC_TRACE) { cache[key] = r; return r; }
            auto h = hintSlotOf.find(t.col);
            if (h != hintSlotOf.end()) {          // a column a hint produced, not the witness
                t.src = MUL_SRC_HINTCOL;
                t.sectionOffset = h->second;
                continue;
            }
            if (t.col >= w->size() || (*w)[t.col] == 0 || (*w)[t.col] > 64) { cache[key] = r; return r; }
            const uint32_t width = (uint32_t)(*w)[t.col];
            if (fromTable(t.col)) {
                t.src = MUL_SRC_PACKED_IDX;
                t.sectionOffset = bitOf[t.col];
                t.col = laneOf(t.col);            // the lane, from here on
                t.nCols = width;
            } else {
                t.src = MUL_SRC_PACKED;
                t.sectionOffset = bitOf[t.col];
                t.nCols = width;
            }
        }

    MulInsnDev* dp = nullptr;
    const size_t bytes = prog.size() * sizeof(MulInsnDev);
    if (cudaMalloc(&dp, bytes) != cudaSuccess) { cache[key] = r; return r; }
    // Never on the default stream: it would poison concurrent graph captures.
    mulCopySync(gpuId, dp, prog.data(), bytes, cudaMemcpyHostToDevice);
    r.prog = dp; r.ok = true;
    cache[key] = r;
    return r;
}

// One scratch buffer per (device, slot), grown to the widest air seen; cached to keep cudaMalloc
// off the critical path. Keyed by slot because distinct slots commit concurrently. Growing is
// safe because each slot's commit syncs its stream before returning.
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

// Where the caller expands this air's const pols before the commit. Sized nConstants * N.
inline uint64_t* mulStreamConst(int gpuId, uint64_t slotIdx, size_t elems) {
    static MulStreamBufs bufs;
    return mulStreamBuf(bufs, gpuId, slotIdx, elems);
}

// Min cm1 reads per row for the scatter to get a column-major copy (StreamCommitDims::colMajorForHook).
// The transpose costs one pass over the packed rows, so it pays only for read-heavy scatters
// (Keccakf, BinaryHuge, BinaryExtensionLarge), not e.g. Mem.
#define MUL_COLMAJOR_MIN_READS_PER_ROW 100

inline bool mulScatterWantsColMajor(SetupCtx &setupCtx, uint64_t airgroupId, uint64_t airId,
                                    const StreamCommitDims &dims) {
    // The indexed walk addresses the row and the table with one expression, so it is left alone.
    if (dims.indexBits != 0 || dims.wordsPerRow == 0) return false;
    const MulPlan &plan = mulPlanFor(setupCtx, airgroupId, airId);
    if (plan.jobs.empty() || !mulPlanStreamable(plan)) return false;
    uint64_t reads = 0;
    for (const MulInsnDev &in : plan.prog)
        for (const MulOperandDev *o : {&in.a, &in.b})
            if (o->kind == MUL_OPND_COL && o->term.src == MUL_SRC_TRACE) reads++;
    return reads >= MUL_COLMAJOR_MIN_READS_PER_ROW;
}

// The hook handed to streamCommitPacked.
inline void mulStreamHook(const uint64_t *dPacked, const StreamCommitDims &dims, cudaStream_t stream,
                          void *user) {
    MulStreamCtx *c = (MulStreamCtx *)user;
    if (c == nullptr || c->acc == nullptr || mulDecoders().empty()) return;

    const MulPlan &plan = mulPlanFor(*c->setupCtx, c->airgroupId, c->airId);
    if (plan.jobs.empty()) return;
    // Asserted, not assumed: a later-stage read has nothing to read here and would be miscounted.
    if (!mulPlanStreamable(plan)) {
        zklog.error("multiplicity: air " + std::to_string(c->airgroupId) + "/"
                    + std::to_string(c->airId) + " reads " + mulSrcMaskNames(plan.srcMask)
                    + ", and a slot commit has only the const pols and the packed rows -- it must not take "
                    "the streaming path");
        exitProcess();
    }

    int gpuId = 0;
    CHECKCUDAERR(cudaGetDevice(&gpuId));
    const MulPlanDev dev = mulPlanDevice(plan, c->airgroupId, c->airId, gpuId);

    const uint64_t nRows = 1ull << dims.nBits;
    const uint64_t *vals = c->dVals;

    // The rewrite was validated before the commit (the slot is refused otherwise), so it cannot fail
    // here. One launch over the whole domain; `'`-shifted reads wrap on rowMask.
    const MulPackedProg packedProg = mulPackedProgramFor(plan, *c, c->airgroupId, c->airId, gpuId);
    if (!packedProg.ok) {
        zklog.error("multiplicity: air " + std::to_string(c->airgroupId) + "/"
                    + std::to_string(c->airId) + " reached the slot scatter with no packed "
                    "program -- its lookups would go uncounted");
        exitProcess();
    }
    const uint64_t *bases[MUL_SRC_N] = {
        c->constPols, nullptr, nullptr,
        vals ? vals + c->offPublics        : nullptr,
        vals ? vals + c->offAirValues      : nullptr,
        vals ? vals + c->offProofValues    : nullptr,
        vals ? vals + c->offAirgroupValues : nullptr,
        nullptr, nullptr, nullptr, nullptr };
    if (c->timer) c->timer->startCategory("MUL_SCATTER_PACKED");
    mul_scatter_launch(dev.jobs, (uint32_t)plan.jobs.size(), nRows, nRows, bases, c->acc, c->oob,
                       (c->airgroupId << 32) | c->airId, packedProg.prog, stream,
                       dPacked, dims.wordsPerRow, c->hintSide,
                       c->dTable, c->wordsPerEntry, c->numEntries, c->indexBits,
                       c->packedColMajor);
    if (c->timer) c->timer->stopCategory("MUL_SCATTER_PACKED");
    mul_note_scatter(gpuId, stream);
}

#endif
