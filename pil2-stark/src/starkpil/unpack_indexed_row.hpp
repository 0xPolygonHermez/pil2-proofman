#ifndef UNPACK_INDEXED_ROW_HPP
#define UNPACK_INDEXED_ROW_HPP

#include <cstdint>

// Row-level unpack of an indexed (compact) trace, shared by the CPU witness path. Device
// code cannot call this, so unpack_indexed (starks_gpu.cu) and scUnpackRangeIndexedKernel
// (stream_commit.cu) keep their own copy of this walk; all three must agree or a slot root
// stops matching cm1.
//
// A row is `lanes` instruction indices (indexBits each) followed by its runtime columns.
// An instruction-derived column comes from the entry that ITS LANE's index selects, so one
// row mixes columns from up to `lanes` entries.
//
// One sequential pass per stream, not one interleaved pass: the row pass reads the untagged
// columns, then lane l's pass reads the columns tagged for lane l. One cursor per pass
// suffices because a lane's columns appear in the row in the field order of an entry --
// which is what lets one entry layout serve every lane.

/// Cursor over a bit-packed stream: `word` is the live word at `idx`, `off` the bit in it.
struct IndexedBitCursor {
    const uint64_t *base;
    uint64_t words;
    uint64_t word;
    uint64_t idx;
    uint64_t off;
};

/// Cursor positioned `bitOffset` bits into `base`. A cursor at the end of the stream reads
/// no word: an all-`@instr` row starts its runtime pass exactly past the header.
static inline IndexedBitCursor indexedCursorAt(const uint64_t *base, uint64_t words, uint64_t bitOffset)
{
    IndexedBitCursor c{base, words, 0, bitOffset / 64, bitOffset % 64};
    if (c.idx < words) c.word = base[c.idx];
    return c;
}

/// Read `nbits` at the cursor, advancing it. Mirrors the GPU idx_read_bits bit-walk exactly.
static inline uint64_t indexedReadBits(IndexedBitCursor &c, uint64_t nbits)
{
    uint64_t val;
    const uint64_t bits_left = 64 - c.off;
    if (nbits <= bits_left) {
        const uint64_t mask = (nbits == 64) ? ~0ULL : ((1ULL << nbits) - 1ULL);
        val = (c.word >> c.off) & mask;
        c.off += nbits;
        if (c.off == 64 && c.idx + 1 < c.words) {
            c.word = c.base[++c.idx];
            c.off = 0;
        }
    } else {
        const uint64_t low = c.word >> c.off;
        c.word = c.base[++c.idx];
        const uint64_t high = c.word & ((1ULL << (nbits - bits_left)) - 1ULL);
        val = (high << bits_left) | low;
        c.off = nbits - bits_left;
    }
    return val;
}

/// Unpack one indexed row into `out` (nCols values). Returns false when a lane's index is
/// past the table, reporting that lane and index; `out` is then incomplete. `lanes` 0 or 1
/// is the single-lane shape.
static inline bool unpackIndexedRow(const uint64_t *rbase, uint64_t wordsPerRow, const uint64_t *table,
                                    uint64_t wordsPerEntry, uint64_t numEntries, uint64_t indexBits,
                                    uint64_t lanes, const uint64_t *unpackInfo, const uint8_t *colSource,
                                    const uint8_t *colLane, uint64_t nCols, uint64_t *out,
                                    uint64_t *badLane = nullptr, uint64_t *badIndex = nullptr)
{
    const uint64_t nLanes = lanes ? lanes : 1;

    // Runtime pass: the untagged columns, in order, from just past the index header.
    IndexedBitCursor rc = indexedCursorAt(rbase, wordsPerRow, nLanes * indexBits);
    for (uint64_t c = 0; c < nCols; c++) {
        if (!colSource[c]) out[c] = indexedReadBits(rc, unpackInfo[c]);
    }

    // Each lane's index sits at a known header offset, so no per-lane state is carried.
    for (uint64_t l = 0; l < nLanes; l++) {
        IndexedBitCursor hc = indexedCursorAt(rbase, wordsPerRow, l * indexBits);
        const uint64_t index = indexedReadBits(hc, indexBits);
        if (index >= numEntries) {
            if (badLane != nullptr) *badLane = l;
            if (badIndex != nullptr) *badIndex = index;
            return false;
        }
        IndexedBitCursor tc = indexedCursorAt(&table[index * wordsPerEntry], wordsPerEntry, 0);
        for (uint64_t c = 0; c < nCols; c++) {
            if (colSource[c] && colLane[c] == l) out[c] = indexedReadBits(tc, unpackInfo[c]);
        }
    }
    return true;
}

#endif
