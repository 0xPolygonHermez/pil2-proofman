// Widening the compact witness buffer into the device trace.
//
// Nothing to do with gate bands, but motivated by the same fact: `getCommitedPols` fills only the
// exec map's columns, leaving the rest for a gate-band expander to rebuild, so those columns need
// never cross PCIe. The host hands over a compact `N x mapCols` buffer, copied in one contiguous
// run, and this widens it on device.

#include <cstdint>
#include "cuda_utils.cuh"

namespace {

// The copy has to be contiguous. Going straight into the strided columns with a 2D copy is slower
// than shipping the full width, because the rows are only mapCols*8 bytes -- too small for the DMA
// engine. The caller zeroes the destination first: the expander does not write every row of the
// columns it owns (padding rows past the last band, the multiplicity columns past the table), and
// those cells still enter the commitment, so they must be a function of the witness rather than
// whatever the buffer last held.
__global__ void widenCompactWitnessKernel(uint64_t *trace, uint64_t nCols, uint64_t nRows,
                                          const uint64_t *compact, uint64_t mapCols) {
    const uint64_t t = blockIdx.x * (uint64_t)blockDim.x + threadIdx.x;
    if (t >= nRows * mapCols) return;
    const uint64_t row = t / mapCols, col = t - row * mapCols;
    trace[row * nCols + col] = compact[t];
}

}  // namespace

// Widens a compact `N x mapCols` device buffer into the trace's first `mapCols` columns. The
// destination must already be zeroed; see the kernel.
extern "C" void widenCompactWitnessGPU(uint64_t *d_trace, uint64_t nCols, uint64_t nRows,
                                       const uint64_t *d_compact, uint64_t mapCols, void *stream_) {
    if (mapCols == 0 || nRows == 0) return;
    const int tpb = 256;
    const uint64_t n = nRows * mapCols;
    widenCompactWitnessKernel<<<(unsigned)((n + tpb - 1) / tpb), tpb, 0, (cudaStream_t)stream_>>>(
        d_trace, nCols, nRows, d_compact, mapCols);
    CHECKCUDAERR(cudaGetLastError());
}
