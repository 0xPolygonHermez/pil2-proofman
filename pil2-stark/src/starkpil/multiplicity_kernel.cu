#include "multiplicity_kernel.cuh"
#include "multiplicity_eval.cuh"
#include "multiplicity_decoders.hpp"
#include <algorithm>
#include "multiplicity_combine.cuh"
// Dedicated range-check scatter: tuple, selector and bus id are compiled programs, so a thread
// evaluates all three from registers instead of running `computeExpressions_` once per hint.

#define MUL_SCATTER_BLOCK 256   // rows per block

// MulBases has named members, not an array: runtime indexing would spill it to local memory.
// The second launch bound matches the occupancy the shared combining cache allows (5 blocks/SM),
// so the compiler cannot spend more registers than that.
__global__ __launch_bounds__(MUL_SCATTER_BLOCK, 5)
void mul_scatter_kernel_rows(const MulJobDev* __restrict__ jobs, uint32_t nJobs, MulBases bases,
                             uint64_t rowMask, uint64_t rows, uint64_t* __restrict__ acc,
                             uint64_t* __restrict__ oob, uint64_t air,
                             const MulInsnDev* __restrict__ prog) {
    __shared__ uint64_t           sKey[MUL_COMBINE_SLOTS];
    __shared__ unsigned long long sVal[MUL_COMBINE_SLOTS];
    MulCombine comb{sKey, sVal};
    mulCombineInit(comb);
    unsigned long long* gacc = (unsigned long long*)acc;

    // One row per thread. No early return: every thread must reach the flush barrier.
    const uint64_t row = (uint64_t)blockIdx.x * MUL_SCATTER_BLOCK + threadIdx.x;
    if (row < rows) {
        // The cached selector is only valid for the row it was computed on.
        uint32_t lastSelOff = 0xFFFFFFFFu;
        uint64_t lastSelVal = 0;

        for (uint32_t k = 0; k < nJobs; ++k) {
            const MulJobDev& j = jobs[k];
            // A degree-0 job contributes once per instance, not once per row.
            if (row >= j.rows) continue;

            uint64_t sel = 1;
            if (!j.selConstOne) {
                // Jobs sharing a selector program share its offset and are ordered by it, so
                // reusing the last result evaluates each run once.
                if (j.selProgOff == lastSelOff) {
                    sel = lastSelVal;
                } else {
                    sel = mulEvalField(j.selProgOff, j.selProgLen, prog, bases, row, rowMask);
                    lastSelOff = j.selProgOff;
                    lastSelVal = sel;
                }
                if (sel == 0) continue;
            }

            // Dynamic opid (multi_range_check): the bus id selects the table per row.
            if (j.hasBus && mulEvalField(j.busProgOff, j.busProgLen, prog, bases, row,
                                         rowMask) != (uint64_t)j.tableId) continue;

            uint64_t key[MUL_MAX_TUPLE];
            if (j.mapSlots == 0) {
                key[0] = mulAddFE(mulEvalField(j.valProgOff, j.valProgLen, prog,
                                               bases, row, rowMask), j.biasFE);
            } else {
                for (uint32_t c = 0; c < j.nKey; ++c)
                    key[c] = mulEvalField(j.keyProgOff[c], j.keyProgLen[c], prog,
                                          bases, row, rowMask);
            }
            uint64_t idx;
            if (!mulResolveRow(key, j.nKey, j.mapSlots, j.mapKV, idx) || idx >= j.nTableRows) {
                // nKey is 0 for a range job (its value is in key[0]).
                mulRecordOob(oob, j.tableId, air, key, j.nKey ? j.nKey : 1u);
                continue;
            }
            const uint64_t slot = j.accBase + idx;
            // sel == 1 hits mulCombineAdd's all-equal (popc) fast path.
            mulCombineAdd(comb, __activemask(), slot, sel, gacc);
        }
    }
    mulCombineFlush(comb, gacc);
}

void mul_scatter_launch(const MulJobDev* d_jobs, uint32_t nJobs, uint64_t rows, uint64_t domainSize,
                        const uint64_t* const* bases_, uint64_t* acc, uint64_t* oob, uint64_t air,
                        const MulInsnDev* d_prog, cudaStream_t stream,
                        const uint64_t* packed, uint64_t wordsPerRow, const uint64_t* side,
                        const uint64_t* table, uint64_t wordsPerEntry, uint64_t numEntries,
                        uint64_t indexBits, uint32_t packedColMajor) {
    if (d_jobs == nullptr || nJobs == 0 || rows == 0) return;
    MulBases bases = { bases_[MUL_SRC_CONST],  bases_[MUL_SRC_TRACE],
                       bases_[MUL_SRC_AUX],    bases_[MUL_SRC_PUBLIC],
                       bases_[MUL_SRC_AIRVALUE], bases_[MUL_SRC_PROOFVALUE],
                       bases_[MUL_SRC_AIRGROUPVALUE], bases_[MUL_SRC_CUSTOM] };
    bases.packed = packed; bases.side = side; bases.wordsPerRow = wordsPerRow;
    bases.packedColMajor = packedColMajor;
    bases.table = table; bases.wordsPerEntry = wordsPerEntry;
    bases.numEntries = numEntries; bases.indexBits = indexBits;
    const uint32_t blocks = (uint32_t)((rows + MUL_SCATTER_BLOCK - 1) / MUL_SCATTER_BLOCK);
    mul_scatter_kernel_rows<<<blocks, MUL_SCATTER_BLOCK, 0, stream>>>(
        d_jobs, nJobs, bases, domainSize - 1, rows, acc, oob, air, d_prog);
    CHECKCUDAERR(cudaGetLastError());
}

// ---- Prover-owned table export: accumulator -> committed trace, on the device ----
//
// The accumulator is [column][row] (a warp's atomics hit the same cache lines); the committed trace
// is [row][column]. Transposed through shared memory so both sides coalesce; +1 pad avoids bank
// conflicts.
#define MUL_T_TILE 32
__global__ __launch_bounds__(MUL_T_TILE * 8)
void mul_transpose_acc_kernel(const uint64_t* __restrict__ acc, uint64_t* __restrict__ trace,
                              uint64_t numRows, uint64_t nCols) {
    __shared__ uint64_t tile[MUL_T_TILE][MUL_T_TILE + 1];

    const uint64_t col0 = (uint64_t)blockIdx.y * MUL_T_TILE;
    const uint64_t row0 = (uint64_t)blockIdx.x * MUL_T_TILE;

    // Read a tile: consecutive threadIdx.x walk consecutive rows of one column (coalesced in acc).
    for (uint32_t j = threadIdx.y; j < MUL_T_TILE; j += blockDim.y) {
        const uint64_t c = col0 + j;
        const uint64_t r = row0 + threadIdx.x;
        tile[j][threadIdx.x] = (c < nCols && r < numRows) ? acc[c * numRows + r] : 0ULL;
    }
    __syncthreads();

    // Write it back transposed: consecutive threadIdx.x now walk consecutive columns of one row
    // (coalesced in trace).
    for (uint32_t j = threadIdx.y; j < MUL_T_TILE; j += blockDim.y) {
        const uint64_t r = row0 + j;
        const uint64_t c = col0 + threadIdx.x;
        if (r < numRows && c < nCols) trace[r * nCols + c] = tile[threadIdx.x][j];
    }
}

void mul_transpose_acc_launch(const uint64_t* acc, uint64_t* trace, uint64_t numRows, uint64_t nCols,
                              cudaStream_t stream) {
    if (acc == nullptr || trace == nullptr || numRows == 0 || nCols == 0) return;
    dim3 block(MUL_T_TILE, 8);
    dim3 grid((uint32_t)((numRows + MUL_T_TILE - 1) / MUL_T_TILE),
              (uint32_t)((nCols + MUL_T_TILE - 1) / MUL_T_TILE));
    mul_transpose_acc_kernel<<<grid, block, 0, stream>>>(acc, trace, numRows, nCols);
}

// Elementwise add of a peer GPU's partial accumulator, staged on this device by the caller.
__global__ __launch_bounds__(256)
void mul_acc_add_kernel(uint64_t* __restrict__ dst, const uint64_t* __restrict__ src, uint64_t n) {
    const uint64_t stride = (uint64_t)gridDim.x * blockDim.x;
    for (uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride)
        if (src[i]) dst[i] += src[i];
}

void mul_acc_add_launch(uint64_t* dst, const uint64_t* src, uint64_t n, cudaStream_t stream) {
    if (dst == nullptr || src == nullptr || n == 0) return;
    const uint32_t blocks = (uint32_t)std::min<uint64_t>((n + 255) / 256, 4096);
    mul_acc_add_kernel<<<blocks, 256, 0, stream>>>(dst, src, n);
}
