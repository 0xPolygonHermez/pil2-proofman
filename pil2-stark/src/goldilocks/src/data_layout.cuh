#ifndef __DATA_LAYOUT_CUH__
#define __DATA_LAYOUT_CUH__

#include <stdint.h>

#define TILE_HEIGHT_LOG2 8
#define TILE_HEIGHT (1 << TILE_HEIGHT_LOG2)
#define TILE_WIDTH  4

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

__device__ __forceinline__ uint64_t getBufferOffsetRowMajor(
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
           + row_block * nCols_block + col_block;
}

#endif