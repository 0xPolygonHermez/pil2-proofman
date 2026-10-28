#ifndef MERKLE_NOLEAVES_CUH
#define MERKLE_NOLEAVES_CUH

// Merkle trees stored without their leaf level: level 1 at offset 0, every higher level as in the
// full tree (getTreeNumElementsNoLeaves elements). H is a GPU hash family class.

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include "cuda_utils.cuh"
#include "goldilocks_tooling.hpp"
#include "goldilocks_trace_layout.cuh"

// Leaf chunks are hashed into the not yet written levels >= 2, so the build needs no extra memory.
// ColMajor input only (rows are addressed by offset). maxChunkRows is for tests.
template <class H>
void merkletreeNoLeaves(uint32_t arity, uint64_t *d_tree, uint64_t *d_input, uint64_t nCols, uint64_t nRows,
                        Layout layout, cudaStream_t stream, uint64_t maxChunkRows = 0)
{
    if (nRows == 0) return;
    if (layout != Layout::ColMajor) {
        fprintf(stderr, "merkletreeNoLeaves: only ColMajor input can be hashed by row range\n");
        abort();
    }
    const uint64_t a = arity;
    const uint64_t L1 = (nRows + a - 1) / a;
    const uint64_t L1pad = L1 > 1 ? L1 + (a - L1 % a) % a : 1;  // the root is never padded
    const uint64_t scratchWords = getTreeNumElementsNoLeaves(nRows, arity) - L1pad * HASH_SIZE;
    // c leaves need c digests plus c/a parents.
    uint64_t chunk = (scratchWords / HASH_SIZE) * a / (a + 1);
    if (maxChunkRows != 0) chunk = std::min(chunk, maxChunkRows);
    chunk -= chunk % a;
    if (chunk == 0) {
        fprintf(stderr, "merkletreeNoLeaves: a %lu-row tree leaves no room for leaf scratch\n", (unsigned long)nRows);
        abort();
    }
    uint64_t *scratch = d_tree + L1pad * HASH_SIZE;
    for (uint64_t r0 = 0; r0 < nRows; r0 += chunk) {
        const uint64_t cnt = std::min(chunk, nRows - r0);
        const uint64_t groups = (cnt + a - 1) / a;
        const uint64_t cntPad = groups * a;
        H::linearHash(scratch, d_input + r0, nCols, cnt, layout, stream, nRows);
        if (cntPad > cnt) {
            CHECKCUDAERR(cudaMemsetAsync(scratch + cnt * HASH_SIZE, 0, (cntPad - cnt) * HASH_SIZE * sizeof(uint64_t), stream));
        }
        H::reduceOneLevel(arity, scratch, cntPad, stream);
        CHECKCUDAERR(cudaMemcpyAsync(d_tree + (r0 / a) * HASH_SIZE, scratch + cntPad * HASH_SIZE,
                                     groups * HASH_SIZE * sizeof(uint64_t), cudaMemcpyDeviceToDevice, stream));
    }
    H::reduceLevels(arity, d_tree, L1, stream);
}

// Queries rehash their level-0 siblings (the other arity-1 rows of the group) from the trace, in
// genMerkleProof's order: sibling j of query q skips the query's own slot.

__device__ __forceinline__ uint64_t leafSiblingRow(const uint64_t *queries, uint64_t s, uint32_t arity) {
    const uint64_t q = s / (arity - 1), j = s % (arity - 1);
    const uint64_t idx = queries[q], self = idx % arity;
    return idx - self + (j < self ? j : j + 1);
}

// Sibling s = q*(arity-1)+j's row, row-major at rowsOut[s*nCols].
__global__ void gatherLeafSiblingRows(const gl64_t *trace, uint64_t nCols, uint64_t nRows, Layout layout,
                                      const uint64_t *queries, uint64_t nSiblings, uint32_t arity, gl64_t *rowsOut) {
    const uint64_t col = blockIdx.x * (uint64_t)blockDim.x + threadIdx.x;
    const uint64_t s = blockIdx.y * (uint64_t)blockDim.y + threadIdx.y;
    if (col >= nCols || s >= nSiblings) return;
    const uint64_t row = leafSiblingRow(queries, s, arity);
    rowsOut[s * nCols + col] = row < nRows ? trace[getBufferOffset(row, col, nRows, nCols, layout)] : gl64_t(uint64_t(0));
}

// A sibling past the last row is the full tree's zero padding.
__global__ void scatterLeafSiblings(const uint64_t *digests, const uint64_t *queries, uint64_t nSiblings,
                                    uint64_t nRows, uint32_t arity, gl64_t *proofBuf, uint64_t bufferWidth,
                                    uint64_t maxTreeWidth) {
    const uint64_t s = blockIdx.x * (uint64_t)blockDim.x + threadIdx.x;
    if (s >= nSiblings) return;
    const uint64_t q = s / (arity - 1), j = s % (arity - 1);
    const bool pad = leafSiblingRow(queries, s, arity) >= nRows;
    gl64_t *out = proofBuf + q * bufferWidth + maxTreeWidth + j * HASH_SIZE;
    for (int w = 0; w < HASH_SIZE; w++) out[w] = pad ? gl64_t(uint64_t(0)) : gl64_t(digests[s * HASH_SIZE + w]);
}

// Scratch leafSiblingsNoLeaves needs: the gathered rows, then their digests.
inline uint64_t leafSiblingsScratchWords(uint64_t nQueries, uint32_t arity, uint64_t nCols) {
    return nQueries * (arity - 1) * (nCols + HASH_SIZE);
}

// Writes query q's level-0 siblings at proofBuf[q*bufferWidth + maxTreeWidth], as genMerkleProof would.
template <class H>
void leafSiblingsNoLeaves(const gl64_t *trace, uint64_t nCols, uint64_t nRows, Layout layout, const uint64_t *d_queries,
                          uint64_t nQueries, uint32_t arity, gl64_t *d_scratch, gl64_t *d_proofBuf,
                          uint64_t bufferWidth, uint64_t maxTreeWidth, cudaStream_t stream) {
    const uint64_t nSiblings = nQueries * (arity - 1);
    if (nSiblings == 0) return;
    gl64_t *rows = d_scratch;
    uint64_t *digests = (uint64_t *)(d_scratch + nSiblings * nCols);
    if (nCols > 0) {
        dim3 threads(32, 8);
        dim3 blocks((nCols + threads.x - 1) / threads.x, (nSiblings + threads.y - 1) / threads.y);
        gatherLeafSiblingRows<<<blocks, threads, 0, stream>>>(trace, nCols, nRows, layout, d_queries, nSiblings, arity, rows);
        CHECKCUDAERR(cudaGetLastError());
    }
    H::linearHash(digests, (uint64_t *)rows, nCols, nSiblings, Layout::RowMajor, stream);
    scatterLeafSiblings<<<(nSiblings + 127) / 128, 128, 0, stream>>>(digests, d_queries, nSiblings, nRows, arity,
                                                                    d_proofBuf, bufferWidth, maxTreeWidth);
    CHECKCUDAERR(cudaGetLastError());
}

#endif
