#ifndef __DATA_LAYOUT_CUH__
#define __DATA_LAYOUT_CUH__

// GPU polynomial memory layouts. getBufferOffset is the prover's storage accessor (ColMajor flat by
// default, ColMajorTiled for the native NTT path -- see resolveLayout); the RowMajor / RowMajorPacked
// variants are the within-tile orderings the native NTT butterfly/LDE kernels use internally.

#include <stdint.h>

#define TILE_HEIGHT_LOG2 8
#define TILE_HEIGHT (1 << TILE_HEIGHT_LOG2)
#define TILE_WIDTH  4

// Memory layout, chosen at RUNTIME (no build flag). One enum shared by getBufferOffset and the
// Merkle/linear-hash kernels (which pass it straight through to getBufferOffset):
//   - RowMajor      : element (row,col) at row*nCols + col (contiguous columns per row).
//   - ColMajor      : FLAT column-major, (row,col) at col*nRows + row (each column one contiguous run).
//                     sppark's native per-column format -> LDE/computeQ run in place, no transpose.
//   - ColMajorTiled : column-major WITHIN TILE_HEIGHT x TILE_WIDTH tiles; the layout the native
//                     (non-sppark) NTT path operates on.
// A committed section is stored ColMajor (flat, sppark) by default and ColMajorTiled (native NTT) only
// for small high-column AIRs where the native path is faster -- see resolveLayout. Const and custom
// commits are stored ColMajorTiled -- see fixedLayout(). getBufferOffset defaults to ColMajor so the
// flat call sites are unchanged.
enum class Layout : uint8_t { RowMajor, ColMajor, ColMajorTiled };

// The storage layout of a committed section, a pure function of its small-domain size (nBits) and
// column count. Writer (commit LDE + Merkle), readers (expression load__), NTT, the row-major->storage
// transpose, AND the commit-Merkle / FRI / opening-eval kernels in starks_gpu.cu all call this with the
// SAME (nBits, nCols), so they always agree without storing the choice. Native tiled wins for small
// domains with many columns; everything else uses flat (sppark).
// FORCE_TILED_LAYOUT: benchmark switch forcing every committed section through the native tiled
// NTT. Must match cm_layout()/store_qq() in setup/exps-codegen/src/emit.rs; rebuild the C++ AND
// regenerate the exps kernels (gen-exps) together, or readers/writers disagree -> wrong proofs.
__host__ __device__ __forceinline__ Layout resolveLayout(uint64_t nBits, uint64_t nCols) {
#ifdef FORCE_TILED_LAYOUT
    (void)nBits; (void)nCols;
    return Layout::ColMajorTiled;
#else
    return (nBits <= 17 && nCols > 500) ? Layout::ColMajorTiled : Layout::ColMajor;
#endif
}

// Single source of truth for CUDA-graph capturability of the NTT/LDE/computeQ path: only the native
// (ColMajorTiled) backend is pure kernels; the flat (ColMajor) backend delegates to sppark (own stream
// + host sync), which is illegal to capture. The LDE/computeQ dispatch and the capture guards in
// starks_gpu.cu both gate on THIS, so "runs the native path" and "is capturable" can never drift apart.
__host__ __device__ __forceinline__ bool isGraphCapturableLayout(uint64_t nBits, uint64_t nCols) {
    return resolveLayout(nBits, nCols) == Layout::ColMajorTiled;
}

// Storage layout of the fixed/preprocessed sections (const pols, const tree, custom commits). Single
// source of truth so every producer and consumer agrees. ColMajorTiled matches pre-1.0.0-beta behavior.
__host__ __device__ __forceinline__ Layout fixedLayout() {
    return Layout::ColMajorTiled;
}

// (row,col) offset for the given layout. Default ColMajor (flat): col*nRows + row.
__device__ __forceinline__ uint64_t getBufferOffset(uint64_t row, uint64_t col, uint64_t nRows, uint64_t nCols, Layout layout) {
    if (layout == Layout::ColMajor) {
        (void)nCols;
        return col * nRows + row;
    }
    if (layout == Layout::RowMajor) {
        return row * nCols + col;
    }
    // ColMajorTiled: column-major within tiles.
    uint64_t blockY = col / TILE_WIDTH;
    uint64_t blockX = row / TILE_HEIGHT;
    uint64_t nCols_block = (nCols - TILE_WIDTH * blockY < TILE_WIDTH) ? (nCols - TILE_WIDTH * blockY) : TILE_WIDTH;
    uint64_t col_block = col % TILE_WIDTH;
    uint64_t row_block = row % TILE_HEIGHT;
    return blockY * TILE_WIDTH * nRows + blockX * nCols_block * TILE_HEIGHT + col_block * TILE_HEIGHT + row_block;
}

// Specialization for row = chunkBase + threadIdx.x (chunkBase a multiple of TILE_HEIGHT). ColMajor/Tiled.
__device__ __forceinline__ uint64_t getBufferOffset_pack256(uint64_t chunkBase, uint64_t col, uint64_t nRows, uint64_t nCols, Layout layout) {
    if (layout == Layout::ColMajor) {
        (void)nCols;
        return col * nRows + chunkBase + threadIdx.x;
    }
    uint64_t blockY = col >> 2;
    uint64_t blockX = chunkBase >> TILE_HEIGHT_LOG2;
    uint64_t rem = nCols - (blockY << 2);
    uint64_t nCols_block = rem < TILE_WIDTH ? rem : TILE_WIDTH;
    uint64_t col_block = col & 3;
    return (blockY << 2) * nRows + blockX * nCols_block * TILE_HEIGHT + (col_block << TILE_HEIGHT_LOG2) + threadIdx.x;
}

// Row-major-within-tile accessors used internally by the native (ColMajorTiled) NTT path.
__device__ __forceinline__ uint64_t getBufferOffsetRowMajor(uint64_t row, uint64_t col, uint64_t nRows, uint64_t nCols) {
    uint64_t blockY = col / TILE_WIDTH;
    uint64_t nCols_block = (nCols - TILE_WIDTH * blockY < TILE_WIDTH) ? (nCols - TILE_WIDTH * blockY) : TILE_WIDTH;
    uint64_t col_block = col % TILE_WIDTH;
    return blockY * TILE_WIDTH * nRows + row * nCols_block + col_block;
}

// Packed row-major within tiles (native LDE): only the first TILE_HEIGHT/blowup rows per tile hold data.
__device__ __forceinline__ uint64_t getBufferOffsetRowMajorPacked(uint64_t row, uint64_t col, uint64_t nRows, uint64_t nCols, uint32_t blowup) {
    uint64_t tile_height_blown = TILE_HEIGHT / blowup;
    uint64_t blockY = col / TILE_WIDTH;
    uint64_t blockX = (row / tile_height_blown);
    uint64_t nCols_block = (nCols - TILE_WIDTH * blockY < TILE_WIDTH) ? (nCols - TILE_WIDTH * blockY) : TILE_WIDTH;
    uint64_t col_block = col % TILE_WIDTH;
    uint64_t row_block = row % tile_height_blown;
    return blockY * TILE_WIDTH * nRows + blockX * nCols_block * TILE_HEIGHT + row_block * nCols_block + col_block;
}

#endif