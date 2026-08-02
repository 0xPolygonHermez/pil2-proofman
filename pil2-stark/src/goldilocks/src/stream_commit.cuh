#ifndef STREAM_COMMIT_GPU_CUH
#define STREAM_COMMIT_GPU_CUH

#include <cuda_runtime.h>
#include <cstdint>

class gl64_t;

// Streaming sponge commit for AIRs with a bit-packed witness.
//
// Instead of materializing the full NExt x nCols extension and hashing rows
// (unpack -> LDE -> merkletree), it keeps a W=16-column working set
// (Poseidon1 arity 4: RATE=12 data columns + CAPACITY=4 state columns) and
// loops over column chunks of RATE:
//   1) unpack the chunk compactly (ColMajor, stride N) at the rate base
//   2) LDE it in place (ldeColMajor equal-base aliasing)
//   3) per extended row, absorb [rate | capacity] -> permute -> new capacity
// After the last chunk the capacity columns are the leaf digests; the node
// reduction runs in the then-dead rate region.
//
// All working memory lives inside one caller-provided slot (see layout in
// streamCommitSlotElems); concurrent calls on different slots/streams are 
// independent.

// Widest witness a slot commit accepts: the slot head reserves one element per
// column for the bit widths, and this bounds that area. Not an algorithmic
// limit -- the working set is 16 columns whatever nCols is -- just the size of
// the reserved header (a few hundred bytes against a multi-GB slot).
static constexpr uint64_t SC_MAX_COLS = 64;

struct StreamCommitDims {
    uint64_t nBits;        // log2 trace rows
    uint64_t nBitsExt;     // log2 extended rows
    uint64_t nCols;        // witness columns (<= SC_MAX_COLS)
    uint64_t wordsPerRow;  // packed 64-bit words per row
};

// Returns required slot size in gl64 elements for the given dims. Slot layout:
//   [0, SC_MAX_COLS)              column bit widths (nCols used)
//   [SC_MAX_COLS, +N*wordsPerRow) packed witness
//   [.., +16*NExt)             sponge state (rate 12 | capacity 4), ColMajor
//   [.., +N)                   LDE scratch
uint64_t streamCommitSlotElems(const StreamCommitDims &dims);

// Commit the bit-packed witness at hPacked (N*wordsPerRow u64, row-major) and
// write the 4-element root to hRoot. colWidths: per-column bit widths (nCols
// entries). The packed upload is issued as 32 MiB cudaMemcpyAsync blocks.
// Synchronous on return: the root is valid, and both the slot and the caller's
// packed-witness buffer are free for reuse -- callers need no event handling.
// Returns 0, or a negative value on invalid dims (nCols outside
// (0, SC_MAX_COLS], arity mismatch with the slot layout contract).
int64_t streamCommitPacked(gl64_t *slotBase, const StreamCommitDims &dims,
                           const uint64_t *colWidths, const void *hPacked,
                           uint64_t *hRoot, cudaStream_t stream);

#endif
