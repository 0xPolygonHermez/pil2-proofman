#ifndef UNPACK_INDEXED_DEVICE_CUH
#define UNPACK_INDEXED_DEVICE_CUH

#include <cstdint>
#include "goldilocks_trace_layout.cuh"  // Layout, getBufferOffset

// The prover's device copy of the indexed unpack, in a header for the same reason the CPU
// walk lives in unpack_indexed_row.hpp: so a test can drive it. starks_gpu.cu includes this,
// so the kernel a test exercises is the kernel the prover runs.

// Read `nbits` from a packed stream at cursor (word,idx,off), advancing the cursor.
// Mirrors unpack()'s bit-walk exactly so indexed output is bit-identical.
__device__ __forceinline__ uint64_t idx_read_bits(
    const uint64_t* base, uint64_t words, uint64_t &word, uint64_t &idx, uint64_t &off, uint64_t nbits)
{
    uint64_t val;
    uint64_t bits_left = 64 - off;
    if (nbits <= bits_left) {
        uint64_t mask = (nbits == 64) ? ~0ULL : ((1ULL << nbits) - 1ULL);
        val = (word >> off) & mask;
        off += nbits;
        if (off == 64 && idx + 1 < words) { word = base[++idx]; off = 0; }
    } else {
        uint64_t low = word >> off;
        word = base[++idx];
        uint64_t high = word & ((1ULL << (nbits - bits_left)) - 1ULL);
        val = (high << bits_left) | low;
        off = nbits - bits_left;
    }
    return val;
}

// Indexed unpack: a compact row of `lanes` instruction indices plus its runtime columns,
// each tagged column read from the entry ITS LANE's index selects. Output is bit-identical
// to unpack(). The walk is unpackIndexedRow (unpack_indexed_row.hpp); this copy and
// scUnpackRangeIndexedKernel must agree with it or a slot root stops matching cm1.
__global__ void unpack_indexed(
    const uint64_t* src,             // compact rows: words_per_row each
    const uint64_t* table,           // instruction table: words_per_entry each
    uint64_t* dst,
    uint64_t nRows,
    uint64_t nCols,
    uint64_t words_per_row,
    uint64_t words_per_entry,
    const uint64_t* d_unpack_info,   // nbits per output column
    const uint8_t*  d_col_source,    // 0 = from row stream, 1 = from table stream
    const uint8_t*  d_col_lane,      // lane whose index selects the entry (null = lane 0)
    uint64_t index_bits,             // width of ONE index in the row's header
    uint64_t lanes,                  // indices per row; 0/1 is the single-lane shape
    uint64_t num_entries,            // instruction-table entry count (index bound)
    Layout layout
) {
    // One shared word per column: width | source<<32 | lane<<33 (nbits <= 64, lanes <= 256),
    // one shared read instead of dependent global loads. The footprint stays nCols * 8, so
    // unpack_trace's sharedMemSize covers both kernels. Null map = lane 0. DRAM-bound
    // (strided row reads dominate): hygiene, not a win.
    extern __shared__ uint64_t shared_unpack_info[];
    for (uint64_t i = threadIdx.x; i < nCols; i += blockDim.x) {
        shared_unpack_info[i] = d_unpack_info[i] | ((uint64_t)(d_col_source[i] != 0) << 32) |
                                ((uint64_t)(d_col_lane != nullptr ? d_col_lane[i] : 0) << 33);
    }
    __syncthreads();

    uint64_t row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= nRows) return;

    const uint64_t* rbase = src + row * words_per_row;
    const uint64_t nLanes = lanes ? lanes : 1;

    // Runtime pass: the untagged columns, from just past the header of `nLanes` indices.
    {
        const uint64_t hdr_bits = nLanes * index_bits;
        uint64_t ridx = hdr_bits / 64, roff = hdr_bits % 64;
        uint64_t rword = (ridx < words_per_row) ? rbase[ridx] : 0;
        for (uint64_t c = 0; c < nCols; c++) {
            const uint64_t info = shared_unpack_info[c];
            if ((info >> 32) & 1ull) continue;
            dst[getBufferOffset(row, c, nRows, nCols, layout)] =
                idx_read_bits(rbase, words_per_row, rword, ridx, roff, info & 0xFFFFFFFFull);
        }
    }

    // One pass per lane; each lane's index sits at a known header offset.
    for (uint64_t l = 0; l < nLanes; l++) {
        const uint64_t h_bits = l * index_bits;
        uint64_t hidx = h_bits / 64, hoff = h_bits % 64;
        uint64_t hword = rbase[hidx];
        uint64_t index = idx_read_bits(rbase, words_per_row, hword, hidx, hoff, index_bits);
        // A witness bug can land a stale index here. The CPU walk reports it; a kernel
        // cannot, so fall back to entry 0 -- a failing proof beats reading past the table.
        if (index >= num_entries) index = 0;

        const uint64_t* tbase = table + index * words_per_entry;
        uint64_t tword = tbase[0], tidx = 0, toff = 0;
        for (uint64_t c = 0; c < nCols; c++) {
            const uint64_t info = shared_unpack_info[c];
            // Warp-uniform: source and lane depend only on c, so this never diverges.
            if (!((info >> 32) & 1ull) || ((info >> 33) & 0xFFull) != l) continue;
            dst[getBufferOffset(row, c, nRows, nCols, layout)] =
                idx_read_bits(tbase, words_per_entry, tword, tidx, toff, info & 0xFFFFFFFFull);
        }
    }
}

#endif
