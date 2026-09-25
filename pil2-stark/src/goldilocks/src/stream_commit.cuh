#ifndef STREAM_COMMIT_GPU_CUH
#define STREAM_COMMIT_GPU_CUH

#include <cuda_runtime.h>
#include <cstdint>

class gl64_t;
class TimerGPU;

// Streaming commit for AIRs with a bit-packed witness.
//
// Instead of materializing the full NExt x nCols extension and hashing rows
// (unpack -> LDE -> merkletree), it keeps a small fixed working set of data
// + state columns (Poseidon1: 12 + 4; blake3: 8 + 4, or 8 + 8 when a row is
// wider than 128 columns) and loops over column chunks:
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
//     block (matches b3_hash_row block for block). A row wider than one
//     blake3 chunk (128 words) spans two chunks: chunk 0's chaining value is
//     parked in 4 extra state columns and the leaf is the parent node of the
//     two chunk values, as b3_hash_row builds it. The 12-column (16 for two
//     chunks) working set makes a blake3 slot smaller than a Poseidon1 one
//     for the same shape.
//
// All working memory lives inside one caller-provided slot (see layout in
// streamCommitSlotElems); concurrent calls on different slots/streams are
// independent.

// Widest witness a slot commit accepts. Cost is only the slot head (one element per column for
// the bit widths). 512 covers zisk Keccakf (453 columns).
static constexpr uint64_t SC_MAX_COLS = 512;

// Widest row blake3's absorb can hash (two parked CVs, four 128-word chunks). Tied to SC_MAX_COLS
// by a static_assert in stream_commit.cu.
static constexpr uint64_t SC_B3_MAX_COLS = 512;

// A lane is named by a u8 (dColLane, and bits 33-40 of the kernel's metadata word).
static constexpr uint64_t SC_MAX_LANES = 256;

// Hash family the slot commits with. Must match the proving key's family --
// the caller (commit_witness_streaming_gpu) derives it from get_hash_family().
enum class StreamCommitHash : uint32_t { Poseidon1 = 0, Blake3 = 1 };

struct StreamCommitDims {
    uint64_t nBits;        // log2 trace rows
    uint64_t nBitsExt;     // log2 extended rows
    uint64_t nCols;        // witness columns (<= SC_MAX_COLS)
    uint64_t wordsPerRow;  // packed 64-bit words per row
    // Hand the hook a column-major copy of the packed rows, transposed into the slot's unused
    // tail (scTransposePackedKernel).
    bool colMajorForHook = false;

    // Indexed (compact) witness. Each row is a header of `lanes` instruction indices
    // (indexBits wide each) followed by the runtime columns; the columns flagged in
    // dColSource are read instead from a shared instruction table of numEntries
    // entries, wordsPerEntry words each. Left zero for a plain packed witness --
    // dColSource == nullptr at the call is what actually selects the plain path.
    uint64_t indexBits = 0;
    uint64_t wordsPerEntry = 0;
    uint64_t numEntries = 0;
    // Execution steps a row packs; 0 or 1 is the single-lane shape.
    uint64_t lanes = 0;
};

// Returns required slot size in gl64 elements for the given dims and family.
// Slot layout:
//   [0, SC_MAX_COLS)              column bit widths (nCols used)
//   [SC_MAX_COLS, +N*wordsPerRow) packed witness
//   [.., +W*NExt)              hash working set (data | state), ColMajor;
//                              W = 16 (Poseidon1: 12 + 4 state), 12 (blake3:
//                              8 + 4 state) or 16 (blake3, nCols > 128: 8 + 4
//                              state + 4 parked chunk-0 CV)
//   [.., +N)                   LDE scratch
uint64_t streamCommitSlotElems(const StreamCommitDims &dims,
                               StreamCommitHash hash = StreamCommitHash::Poseidon1);

// Commit the bit-packed witness at hPacked (N*wordsPerRow u64, row- or column-major) and
// write the 4-element root to hRoot. colWidths: per-column bit widths (nCols
// entries, host). The packed upload is issued as 32 MiB cudaMemcpyAsync blocks.
// Synchronous on return: the root is valid, and both the slot and the caller's
// packed-witness buffer are free for reuse -- callers need no event handling.
//
// dColSource / dColLane / dTable are DEVICE pointers and select the indexed
// unpack: dColSource is 0 = row stream, 1 = instruction table, and dColLane
// names the lane whose index selects that entry. Every dColLane entry must be
// below max(dims.lanes, 1) -- lanes 0 and 1 are both the single-lane shape, so
// lane 0 is the only valid entry there. The caller's contract, not checked here (it is device
// memory, and reading it back would stall the stream); a stray lane leaves that
// column written by no pass. indexedDescriptorError (unpack_indexed_row.hpp)
// checks it host-side at upload. dColSource and dTable must be non-null together,
// dColLane whenever dims.lanes > 1; all are borrowed and must stay resident on
// the current device for the call. Pass nullptr for all three for a plain witness.
//
// Returns 0, or a negative value on invalid dims (nCols outside
// (0, SC_MAX_COLS], lanes above SC_MAX_LANES, arity mismatch with the slot
// layout contract, or an inconsistent indexed descriptor).

// Called once, after the packed witness (`dPacked`, device) is uploaded and before the chunk loop.
// The only point where the whole witness exists: the loop LDEs columns IN PLACE, so nothing after
// it can read cm1.
typedef void (*StreamCommitHook)(const uint64_t *dPacked, const StreamCommitDims &dims,
                                 cudaStream_t stream, void *user);

// Called after each chunk is unpacked and BEFORE it is extended in place. `dst` holds `cc` columns
// from `c0`, ColMajor at the small-domain stride. Prover-computed stage-1 columns (witness_calc
// hints) are written here.
typedef void (*StreamCommitChunkHook)(uint64_t *dst, uint32_t c0, uint32_t cc, uint64_t nRows,
                                      cudaStream_t stream, void *user);

// Whether a slot of `slotElems` elements fits this layout AND a column-major packed copy at the tail.
inline bool streamCommitColMajorFits(const StreamCommitDims &dims, StreamCommitHash hash,
                                     uint64_t slotElems) {
    const uint64_t need = (1ull << dims.nBits) * dims.wordsPerRow;
    return need != 0 && slotElems >= need &&
           streamCommitSlotElems(dims, hash) + need <= slotElems;
}

int64_t streamCommitPacked(gl64_t *slotBase, const StreamCommitDims &dims,
                           const uint64_t *colWidths, const void *hPacked,
                           uint64_t *hRoot, cudaStream_t stream,
                           const uint8_t *dColSource = nullptr,
                           const uint8_t *dColLane = nullptr,
                           const uint64_t *dTable = nullptr,
                           StreamCommitHash hash = StreamCommitHash::Poseidon1,
                           StreamCommitHook hook = nullptr, void *hookUser = nullptr,
                           TimerGPU *timer = nullptr,
                           StreamCommitChunkHook chunkHook = nullptr, void *chunkUser = nullptr,
                           // Elements the slot owns; colMajorForHook transposes into the
                           // unused tail. 0 disables it.
                           uint64_t slotElems = 0);

#endif
