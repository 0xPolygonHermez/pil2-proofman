#ifndef __DATA_LAYOUT_CUH__
#define __DATA_LAYOUT_CUH__

/*
 * Tile-based memory layouts for GPU polynomials.
 *
 * All polynomial data is grouped in tiles of TILE_HEIGHT x TILE_WIDTH elements.
 * Three orderings within tiles are used:
 *
 * - Column-major tiles (getBufferOffset): within each tile, elements are stored
 *   column-by-column. This is the prover's storage format 
 *
 * - Row-major tiles (getBufferOffsetRowMajor): within each tile, elements are
 *   stored row-by-row. NTT butterfly kernels operate on this layout because each
 *   thread processes one row across all columns in a tile.
 *
 * - Packed row-major tiles (getBufferOffsetRowMajorPacked): row-major within tiles,
 *   but for LDE only. When extending domain N to N*B (blowup factor B), each tile
 *   has only TILE_HEIGHT/B rows of actual data, packed at the start.
 */

#include <stdint.h>

#define TILE_HEIGHT_LOG2 8
#define TILE_HEIGHT (1 << TILE_HEIGHT_LOG2)
#define TILE_WIDTH  4

// Column-major within tiles
__device__ __forceinline__ uint64_t getBufferOffset(
    uint64_t row,
    uint64_t col,
    uint64_t nRows,
    uint64_t nCols
) {
    uint64_t blockY = col / TILE_WIDTH;                  
    uint64_t blockX = row / TILE_HEIGHT;
    uint64_t nCols_block = (nCols - TILE_WIDTH * blockY < TILE_WIDTH) 
                           ? (nCols - TILE_WIDTH * blockY) : TILE_WIDTH;
    uint64_t col_block = col % TILE_WIDTH;
    uint64_t row_block = row % TILE_HEIGHT;

    return blockY * TILE_WIDTH * nRows + blockX * nCols_block * TILE_HEIGHT
           + col_block * TILE_HEIGHT + row_block;
}

// Specialized getBufferOffset when row = chunkBase + threadIdx.x and chunkBase is
// a multiple of TILE_HEIGHT. Saves the row / TILE_HEIGHT division and row % TILE_HEIGHT
// modulo since row_block == threadIdx.x and blockX == chunkBase >> TILE_HEIGHT_LOG2.
__device__ __forceinline__ uint64_t getBufferOffset_pack256(
    uint64_t chunkBase,
    uint64_t col,
    uint64_t nRows,
    uint64_t nCols
) {
    uint64_t blockY = col >> 2;
    uint64_t blockX = chunkBase >> TILE_HEIGHT_LOG2;
    uint64_t rem = nCols - (blockY << 2);
    uint64_t nCols_block = rem < TILE_WIDTH ? rem : TILE_WIDTH;
    uint64_t col_block = col & 3;

    return (blockY << 2) * nRows + blockX * nCols_block * TILE_HEIGHT
           + (col_block << TILE_HEIGHT_LOG2) + threadIdx.x;
}

// Row-major within tiles
__device__ __forceinline__ uint64_t getBufferOffsetRowMajor(
    uint64_t row,
    uint64_t col,
    uint64_t nRows,
    uint64_t nCols
) {
    uint64_t blockY = col / TILE_WIDTH;                  
    uint64_t nCols_block = (nCols - TILE_WIDTH * blockY < TILE_WIDTH) 
                           ? (nCols - TILE_WIDTH * blockY) : TILE_WIDTH;
    uint64_t col_block = col % TILE_WIDTH;

    return blockY * TILE_WIDTH * nRows + row * nCols_block + col_block;
}

// Packed row-major within tiles: only first TILE_HEIGHT/blowup rows per tile contain data.
__device__ __forceinline__ uint64_t getBufferOffsetRowMajorPacked(
    uint64_t row,
    uint64_t col,
    uint64_t nRows,
    uint64_t nCols,
    uint32_t blowup
) {

    uint64_t tile_height_blown = TILE_HEIGHT / blowup;
    uint64_t blockY = col / TILE_WIDTH;                  
    uint64_t blockX = (row / tile_height_blown);
    uint64_t nCols_block = (nCols - TILE_WIDTH * blockY < TILE_WIDTH) 
                           ? (nCols - TILE_WIDTH * blockY) : TILE_WIDTH;
    uint64_t col_block = col % TILE_WIDTH;
    uint64_t row_block = row % tile_height_blown;

    return blockY * TILE_WIDTH * nRows + blockX * nCols_block * TILE_HEIGHT
           + row_block * nCols_block + col_block;
}

#endif