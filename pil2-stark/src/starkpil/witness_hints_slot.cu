// Device half of evaluating an air's witness_calc hints inside a streaming-commit slot.
// See witness_hints_slot.hpp for how the plan is built.

#include "witness_hints_slot.cuh"
#include "multiplicity_eval.cuh"
#include "multiplicity_stream.cuh"
#include <map>
#include <mutex>
#include <vector>

// One hint, one thread per row, through the scatter's evaluator (mulTermValue).
__global__ __launch_bounds__(SLOT_HINT_BLOCK)
void slotHintEvalKernel(const MulInsnDev *__restrict__ prog, uint32_t n, MulBases b,
                        uint64_t rowMask, uint32_t destSlot, uint64_t *__restrict__ side) {
    const uint64_t row = (uint64_t)blockIdx.x * SLOT_HINT_BLOCK + threadIdx.x;
    if (row > rowMask) return;
    side[(uint64_t)destSlot * (rowMask + 1) + row] = mulEvalProgram(prog, n, b, row, rowMask);
}

// Overwrite the hint columns of this chunk: the prover's value, not the unpacked one, is committed.
__global__ __launch_bounds__(SLOT_HINT_BLOCK)
void slotHintPatchKernel(uint64_t *__restrict__ dst, uint32_t c0, uint32_t cc, uint64_t dstRows,
                         uint64_t dstStride, uint64_t rowBegin, uint64_t fullRows,
                         const uint64_t *__restrict__ side, const uint32_t *__restrict__ destCols,
                         const uint32_t *__restrict__ destSlots, uint32_t nDest) {
    const uint64_t row = (uint64_t)blockIdx.x * SLOT_HINT_BLOCK + threadIdx.x;
    if (row >= dstRows) return;
    // `dst` may be a tile; the side buffer is full height, indexed by the global row.
    for (uint32_t i = 0; i < nDest; ++i) {
        const uint32_t c = destCols[i];
        if (c < c0 || c >= c0 + cc) continue;
        dst[(uint64_t)(c - c0) * dstStride + row] =
            side[(uint64_t)destSlots[i] * fullRows + rowBegin + row];
    }
}

void slotHintEvalLaunch(const MulInsnDev *dProg, const SlotHintOp *hOps, uint32_t nOps,
                        const uint64_t *dPacked, uint64_t wordsPerRow, bool packedColMajor,
                        const uint64_t *dConstPols,
                        const uint64_t *dVals, const SlotHintValOffsets &vo, uint64_t *dSide,
                        uint64_t nRows, cudaStream_t stream) {
    if (dProg == nullptr || nOps == 0 || dSide == nullptr || nRows == 0) return;
    // One base per value pool, as the scatter sets them.
    MulBases b{};
    b.constPols = dConstPols;
    b.publics        = dVals ? dVals + vo.publics        : nullptr;
    b.proofValues    = dVals ? dVals + vo.proofValues    : nullptr;
    b.airgroupValues = dVals ? dVals + vo.airgroupValues : nullptr;
    b.airValues      = dVals ? dVals + vo.airValues      : nullptr;
    b.traceRows = nRows;
    b.packed = dPacked;
    b.side = dSide;
    b.wordsPerRow = wordsPerRow;
    b.packedColMajor = packedColMajor ? 1u : 0u;
    const uint32_t blocks = (uint32_t)((nRows + SLOT_HINT_BLOCK - 1) / SLOT_HINT_BLOCK);
    // Declaration order, one launch each on one stream: a hint may read an earlier hint's column.
    for (uint32_t i = 0; i < nOps; ++i) {
        slotHintEvalKernel<<<blocks, SLOT_HINT_BLOCK, 0, stream>>>(
            dProg + hOps[i].progOff, hOps[i].progLen, b, nRows - 1, hOps[i].destSlot, dSide);
        CHECKCUDAERR(cudaGetLastError());
    }
}

void slotHintPatchLaunch(uint64_t *dst, uint32_t c0, uint32_t cc, uint64_t dstRows,
                         uint64_t rowBegin, uint64_t fullRows, const uint64_t *dSide,
                         const uint32_t *dDestCols, const uint32_t *dDestSlots, uint32_t nDest,
                         cudaStream_t stream, uint64_t dstStride) {
    if (dSide == nullptr || nDest == 0 || dstRows == 0 || cc == 0) return;
    // Write at the whole tile's height, not this pass's row count (as the unpack does).
    if (dstStride == 0) dstStride = dstRows;
    const uint32_t blocks = (uint32_t)((dstRows + SLOT_HINT_BLOCK - 1) / SLOT_HINT_BLOCK);
    slotHintPatchKernel<<<blocks, SLOT_HINT_BLOCK, 0, stream>>>(
        dst, c0, cc, dstRows, dstStride, rowBegin, fullRows, dSide, dDestCols, dDestSlots, nDest);
    CHECKCUDAERR(cudaGetLastError());
}

SlotHintPlanDev slotHintPlanDevice(const SlotHintPlan &plan, uint64_t airgroupId, uint64_t airId,
                                   int gpuId) {
    static std::map<std::tuple<uint64_t,uint64_t,int>, SlotHintPlanDev> bufs;
    static std::mutex mtx;
    std::lock_guard<std::mutex> lock(mtx);
    auto key = std::make_tuple(airgroupId, airId, gpuId);
    auto it = bufs.find(key);
    if (it != bufs.end()) return it->second;

    SlotHintPlanDev d;
    if (plan.ok && !plan.ops.empty()) {
        std::vector<uint32_t> cols, slots;
        for (uint32_t c : plan.destCols) {
            cols.push_back(c);
            uint32_t sl = 0;
            for (const auto &op : plan.ops) if (op.destCol == c) { sl = op.destSlot; break; }
            slots.push_back(sl);
        }
        MulInsnDev *dp = nullptr; uint32_t *dc = nullptr, *ds = nullptr;
        const size_t pb = plan.prog.size() * sizeof(MulInsnDev);
        const size_t cb = cols.size() * sizeof(uint32_t);
        if (cudaMalloc(&dp, pb) == cudaSuccess && cudaMalloc(&dc, cb) == cudaSuccess &&
            cudaMalloc(&ds, cb) == cudaSuccess) {
            CHECKCUDAERR(cudaMemcpy(dp, plan.prog.data(), pb, cudaMemcpyHostToDevice));
            CHECKCUDAERR(cudaMemcpy(dc, cols.data(), cb, cudaMemcpyHostToDevice));
            CHECKCUDAERR(cudaMemcpy(ds, slots.data(), cb, cudaMemcpyHostToDevice));
            d.prog = dp; d.destCols = dc; d.destSlots = ds;
            d.nDest = (uint32_t)cols.size();
            d.ready = true;
        } else {
            if (dp) cudaFree(dp);
            if (dc) cudaFree(dc);
            if (ds) cudaFree(ds);
        }
    } else if (plan.ok) {
        d.ready = true;   // nothing to evaluate is a valid, ready plan
    }
    bufs[key] = d;
    return d;
}

uint64_t *slotHintSideBuffer(int gpuId, uint64_t slotIdx, size_t elems) {
    // Reuses the scatter's per-(device, slot) cache.

    static MulStreamBufs bufs;
    return mulStreamBuf(bufs, gpuId, slotIdx, elems);
}
