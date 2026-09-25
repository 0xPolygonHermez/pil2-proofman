#ifndef WITNESS_HINTS_SLOT_CUH
#define WITNESS_HINTS_SLOT_CUH

#include <cuda_runtime.h>
#include <cstdint>
#include "witness_hints_slot.hpp"
#include "cuda_utils.cuh"

#define SLOT_HINT_BLOCK 256

// Start of each value pool in the staged window; a uniform term's index is pool-relative.
struct SlotHintValOffsets {
    uint64_t publics = 0, proofValues = 0, airgroupValues = 0, airValues = 0;
};

// This air's plan on one GPU, built once and shared by every instance (side buffer is per-slot).
struct SlotHintPlanDev {
    const MulInsnDev *prog = nullptr;
    const uint32_t   *destCols = nullptr;
    const uint32_t   *destSlots = nullptr;
    uint32_t          nDest = 0;
    bool              ready = false;
};

SlotHintPlanDev slotHintPlanDevice(const SlotHintPlan &plan, uint64_t airgroupId, uint64_t airId,
                                   int gpuId);

// Per-(device, slot) side buffer, grown to the widest air seen. Avoids a cudaMalloc on the
// slot-commit critical path.
uint64_t *slotHintSideBuffer(int gpuId, uint64_t slotIdx, size_t elems);

// Evaluate this air's witness_calc hints into `dSide`, one column per plan slot.
// `hOps` is a HOST array, one launch per hint in declaration order. `dConstPols` must be the
// UNPACKED const pols; it and `dVals` may be null only if the plan reads neither.
void slotHintEvalLaunch(const MulInsnDev *dProg, const SlotHintOp *hOps, uint32_t nOps,
                        const uint64_t *dPacked, uint64_t wordsPerRow, const uint64_t *dConstPols,
                        const uint64_t *dVals, const SlotHintValOffsets &vo, uint64_t *dSide,
                        uint64_t nRows, cudaStream_t stream);

// Overwrite the hint columns of an unpacked chunk [c0, c0 + cc) with what the hints produced.
void slotHintPatchLaunch(uint64_t *dst, uint32_t c0, uint32_t cc, uint64_t dstRows,
                         uint64_t rowBegin, uint64_t fullRows, const uint64_t *dSide,
                         const uint32_t *dDestCols, const uint32_t *dDestSlots, uint32_t nDest,
                         cudaStream_t stream, uint64_t dstStride = 0);

#endif
