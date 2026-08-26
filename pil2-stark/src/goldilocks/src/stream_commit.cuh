#ifndef STREAM_COMMIT_GPU_CUH
#define STREAM_COMMIT_GPU_CUH

#include <cuda_runtime.h>
#include <cstdint>

class gl64_t;

// Streaming commit for AIRs with a bit-packed witness.
//
// Instead of materializing the full NExt x nCols extension and hashing rows
// (unpack -> LDE -> merkletree), it keeps a small fixed working set of data
// + state columns (Poseidon1: 12 + 4; blake3: 8 + 4) and loops over column
// chunks:
//   1) unpack the chunk compactly (ColMajor, stride N) at the data base
//   2) LDE it in place (ldeColMajor equal-base aliasing)
//   3) per extended row, fold the chunk into the carried hash state
// After the last chunk the state columns are the leaf digests; the node
// reduction runs in the then-dead data region.
//
// Two hash families, chunked identically to their row-hash reference so the
// slot root is bit-identical to the legacy commit path:
//   * Poseidon1 (arity 4): W=16 sponge, chunks of RATE=12, digest carried in
//     the 4 capacity columns (matches linearHashKernel_pos1).
//   * blake3 (arity 2): chunks of 8 (one 64-byte block), raw chaining value
//     carried in the 4 state columns, packed to the digest on the final
//     block (matches blake3core::compress_chunk; nCols <= SC_MAX_COLS < 128
//     keeps every row inside a single blake3 chunk, so the chunk counter is
//     always 0). The 12-column working set makes a blake3 slot smaller than
//     a Poseidon1 one for the same shape.
//
// All working memory lives inside one caller-provided slot (see layout in
// streamCommitSlotElems); concurrent calls on different slots/streams are
// independent.

// Widest witness a slot commit accepts: the slot head reserves one element per
// column for the bit widths, and this bounds that area. Not an algorithmic
// limit -- the working set stays 16/12 columns whatever nCols is -- just the
// size of the reserved header (a few hundred bytes against a multi-GB slot).
static constexpr uint64_t SC_MAX_COLS = 64;

// Hash family the slot commits with. Must match the proving key's family --
// the caller (commit_witness_streaming_gpu) derives it from get_hash_family().
enum class StreamCommitHash : uint32_t { Poseidon1 = 0, Blake3 = 1 };

struct StreamCommitDims {
    uint64_t nBits;        // log2 trace rows
    uint64_t nBitsExt;     // log2 extended rows
    uint64_t nCols;        // witness columns (<= SC_MAX_COLS)
    uint64_t wordsPerRow;  // packed 64-bit words per row

    // Indexed (compact) witness. Each row is a leading instruction-index header
    // (indexBits wide) followed by the runtime columns; the columns flagged in
    // dColSource are read instead from a shared instruction table of numEntries
    // entries, wordsPerEntry words each. Left zero for a plain packed witness --
    // dColSource == nullptr at the call is what actually selects the plain path.
    uint64_t indexBits = 0;
    uint64_t wordsPerEntry = 0;
    uint64_t numEntries = 0;
};

// Returns required slot size in gl64 elements for the given dims and family.
// Slot layout:
//   [0, SC_MAX_COLS)              column bit widths (nCols used)
//   [SC_MAX_COLS, +N*wordsPerRow) packed witness
//   [.., +W*NExt)              hash working set (data | 4 state), ColMajor;
//                              W = 16 (Poseidon1) or 12 (blake3)
//   [.., +N)                   LDE scratch
uint64_t streamCommitSlotElems(const StreamCommitDims &dims,
                               StreamCommitHash hash = StreamCommitHash::Poseidon1);

// Commit the bit-packed witness at hPacked (N*wordsPerRow u64, row-major) and
// write the 4-element root to hRoot. colWidths: per-column bit widths (nCols
// entries, host). The packed upload is issued as 32 MiB cudaMemcpyAsync blocks.
// Synchronous on return: the root is valid, and both the slot and the caller's
// packed-witness buffer are free for reuse -- callers need no event handling.
//
// dColSource / dTable are DEVICE pointers and select the indexed unpack: per
// column 0 = read from the row stream, 1 = read from the instruction table.
// Both must be non-null together, resident on the current device, and stay
// alive for the call; they are borrowed, never freed here. Pass nullptr for
// both (the default) to commit a plain packed witness.
//
// Returns 0, or a negative value on invalid dims (nCols outside
// (0, SC_MAX_COLS], arity mismatch with the slot layout contract, or an
// inconsistent indexed descriptor).
int64_t streamCommitPacked(gl64_t *slotBase, const StreamCommitDims &dims,
                           const uint64_t *colWidths, const void *hPacked,
                           uint64_t *hRoot, cudaStream_t stream,
                           const uint8_t *dColSource = nullptr,
                           const uint64_t *dTable = nullptr,
                           StreamCommitHash hash = StreamCommitHash::Poseidon1);

#endif
