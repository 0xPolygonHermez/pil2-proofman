#ifndef BLAKE3_DEVICE_CUH
#define BLAKE3_DEVICE_CUH

// ---------------------------------------------------------------------------
// blake3_device.cuh -- self-contained GPU BLAKE3, for benchmarking ONLY.
//
// Direct port of the official BLAKE3 portable reference (public domain CC0 /
// Apache-2.0) to a scalar __device__ implementation: one CUDA thread hashes one
// input, mirroring the one-thread-per-row mapping of the Poseidon linearHash
// kernels so the comparison is apples-to-apples.
//
// SCOPE: single-chunk inputs only, i.e. <= 1024 bytes = <= 128 Goldilocks u64
// elements. This covers the linear/leaf-hash benchmark widths (<= 56 columns);
// the BLAKE3 chunk tree (inputs > 1024 bytes) is intentionally NOT implemented.
//
// Inputs are whole Goldilocks field elements (u64, little-endian), so every
// 64-byte BLAKE3 block holds an integer number (1..8) of u64 words -- no
// sub-word byte handling is needed.
//
// This header lives under benchs/ so the prover library never compiles it.
// ---------------------------------------------------------------------------

#include <cstdint>

namespace blake3gpu {

// BLAKE3 domain-separation flags.
static constexpr uint8_t FLAG_CHUNK_START = 1u << 0;
static constexpr uint8_t FLAG_CHUNK_END   = 1u << 1;
static constexpr uint8_t FLAG_PARENT      = 1u << 2;
static constexpr uint8_t FLAG_ROOT        = 1u << 3;

static constexpr uint32_t BLOCK_LEN = 64;   // bytes per compression block
static constexpr uint32_t MAX_U64   = 128;  // 1024 bytes = one chunk (single-chunk path)
static constexpr uint32_t CHUNK_U64 = 128;  // u64 words per BLAKE3 chunk
static constexpr int      CV_STACK  = 24;   // CV-stack depth (handles up to 2^24 chunks)

// constexpr so device code folds index expressions at compile time, keeping the
// message/state arrays in registers (a runtime index would spill them to local).
__device__ constexpr uint32_t IV[8] = {
    0x6A09E667u, 0xBB67AE85u, 0x3C6EF372u, 0xA54FF53Au,
    0x510E527Fu, 0x9B05688Cu, 0x1F83D9ABu, 0x5BE0CD19u};

// Per-round message-word permutation.
__device__ constexpr uint8_t MSG_SCHEDULE[7][16] = {
    {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15},
    {2, 6, 3, 10, 7, 0, 4, 13, 1, 11, 12, 5, 9, 14, 15, 8},
    {3, 4, 10, 12, 13, 2, 7, 14, 6, 5, 9, 0, 11, 15, 8, 1},
    {10, 7, 12, 9, 14, 3, 13, 15, 4, 0, 11, 2, 5, 8, 1, 6},
    {12, 13, 9, 11, 15, 10, 14, 8, 7, 2, 5, 3, 0, 1, 6, 4},
    {9, 14, 11, 5, 8, 12, 15, 1, 13, 3, 0, 10, 2, 6, 4, 7},
    {11, 15, 5, 0, 1, 9, 8, 6, 14, 10, 2, 12, 3, 4, 7, 13}};

__device__ __forceinline__ uint32_t rotr32(uint32_t x, uint32_t n)
{
    return (x >> n) | (x << (32u - n));
}

// The BLAKE3 quarter-round mixing function.
__device__ __forceinline__ void g(uint32_t *st, int a, int b, int c, int d,
                                   uint32_t mx, uint32_t my)
{
    st[a] = st[a] + st[b] + mx;
    st[d] = rotr32(st[d] ^ st[a], 16);
    st[c] = st[c] + st[d];
    st[b] = rotr32(st[b] ^ st[c], 12);
    st[a] = st[a] + st[b] + my;
    st[d] = rotr32(st[d] ^ st[a], 8);
    st[c] = st[c] + st[d];
    st[b] = rotr32(st[b] ^ st[c], 7);
}

// One BLAKE3 round. R is a template parameter so MSG_SCHEDULE[R][k] is a
// constant expression and m[] stays in registers.
template<int R>
__device__ __forceinline__ void round_fn(uint32_t st[16], const uint32_t m[16])
{
    g(st, 0, 4,  8, 12, m[MSG_SCHEDULE[R][0]],  m[MSG_SCHEDULE[R][1]]);
    g(st, 1, 5,  9, 13, m[MSG_SCHEDULE[R][2]],  m[MSG_SCHEDULE[R][3]]);
    g(st, 2, 6, 10, 14, m[MSG_SCHEDULE[R][4]],  m[MSG_SCHEDULE[R][5]]);
    g(st, 3, 7, 11, 15, m[MSG_SCHEDULE[R][6]],  m[MSG_SCHEDULE[R][7]]);
    g(st, 0, 5, 10, 15, m[MSG_SCHEDULE[R][8]],  m[MSG_SCHEDULE[R][9]]);
    g(st, 1, 6, 11, 12, m[MSG_SCHEDULE[R][10]], m[MSG_SCHEDULE[R][11]]);
    g(st, 2, 7,  8, 13, m[MSG_SCHEDULE[R][12]], m[MSG_SCHEDULE[R][13]]);
    g(st, 3, 4,  9, 14, m[MSG_SCHEDULE[R][14]], m[MSG_SCHEDULE[R][15]]);
}

// Compress one 64-byte block (given as 16 little-endian words) into the
// chaining value `cv`, in place. 7 rounds, then the feed-forward XOR.
__device__ __forceinline__ void compress_in_place(uint32_t cv[8],
                                                  const uint32_t block[16],
                                                  uint8_t block_len,
                                                  uint64_t counter,
                                                  uint8_t flags)
{
    uint32_t st[16];
#pragma unroll
    for (int i = 0; i < 8; ++i) st[i] = cv[i];
    st[8]  = IV[0];
    st[9]  = IV[1];
    st[10] = IV[2];
    st[11] = IV[3];
    st[12] = (uint32_t)counter;
    st[13] = (uint32_t)(counter >> 32);
    st[14] = (uint32_t)block_len;
    st[15] = (uint32_t)flags;

    round_fn<0>(st, block);
    round_fn<1>(st, block);
    round_fn<2>(st, block);
    round_fn<3>(st, block);
    round_fn<4>(st, block);
    round_fn<5>(st, block);
    round_fn<6>(st, block);

#pragma unroll
    for (int i = 0; i < 8; ++i) cv[i] = st[i] ^ st[i + 8];
}

// Compress one chunk (<= 128 u64 = <= 1024 bytes = <= 16 blocks) into its 8-word
// chaining value. `counter` is the chunk index (0 for a single chunk); `root`
// sets the ROOT flag on the final block (true only when the whole input is one
// chunk -- otherwise the root lives in a parent node).
__device__ __forceinline__ void compress_chunk(const uint64_t *in, uint32_t n_u64,
                                               uint64_t counter, bool root,
                                               uint32_t cv[8])
{
#pragma unroll
    for (int i = 0; i < 8; ++i) cv[i] = IV[i];

    uint32_t nblocks = (n_u64 + 7u) / 8u;
    if (nblocks == 0) nblocks = 1;  // empty input still compresses one zero block

    uint32_t idx = 0;
    uint32_t remaining = n_u64;
    for (uint32_t b = 0; b < nblocks; ++b)
    {
        uint32_t in_block = (remaining >= 8u) ? 8u : remaining;  // u64 words this block
        uint32_t block[16];
#pragma unroll
        for (int k = 0; k < 8; ++k)
        {
            uint64_t v = ((uint32_t)k < in_block) ? in[idx + k] : 0ull;
            block[2 * k]     = (uint32_t)v;
            block[2 * k + 1] = (uint32_t)(v >> 32);
        }
        uint8_t flags = 0;
        if (b == 0)           flags |= FLAG_CHUNK_START;
        if (b == nblocks - 1) { flags |= FLAG_CHUNK_END; if (root) flags |= FLAG_ROOT; }
        compress_in_place(cv, block, (uint8_t)(in_block * 8u), counter, flags);
        idx       += in_block;
        remaining -= in_block;
    }
}

// Parent node: combine left||right child CVs (16-word block) into one CV. `out`
// may alias `right` (both children are read into `block` before `out` is written).
__device__ __forceinline__ void parent_cv(const uint32_t left[8], const uint32_t right[8],
                                          uint32_t out[8], bool root)
{
    uint32_t block[16];
#pragma unroll
    for (int i = 0; i < 8; ++i) { block[i] = left[i]; block[8 + i] = right[i]; }
    uint32_t cv[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) cv[i] = IV[i];
    uint8_t flags = FLAG_PARENT | (root ? FLAG_ROOT : 0);
    compress_in_place(cv, block, (uint8_t)BLOCK_LEN, 0ull, flags);
#pragma unroll
    for (int i = 0; i < 8; ++i) out[i] = cv[i];
}

__device__ __forceinline__ void pack_out(const uint32_t cv[8], uint64_t out[4])
{
    out[0] = (uint64_t)cv[0] | ((uint64_t)cv[1] << 32);
    out[1] = (uint64_t)cv[2] | ((uint64_t)cv[3] << 32);
    out[2] = (uint64_t)cv[4] | ((uint64_t)cv[5] << 32);
    out[3] = (uint64_t)cv[6] | ((uint64_t)cv[7] << 32);
}

// Single-chunk hash (n_u64 <= 128): the chunk's final block carries ROOT, so its
// chaining value IS the 256-bit digest. Register-resident, no CV stack.
__device__ __forceinline__ void hash_le64(const uint64_t *in, uint32_t n_u64,
                                          uint64_t out[4])
{
    uint32_t cv[8];
    compress_chunk(in, n_u64, 0ull, true, cv);
    pack_out(cv, out);
}

// Multi-chunk hash (any length): split into 1024-byte chunks, hash each (chunk i
// uses counter i), and combine chunk CVs with a left-to-right CV-stack binary
// tree (the standard BLAKE3 construction). The outermost parent carries ROOT.
__device__ void hash_le64_multi(const uint64_t *in, uint32_t n_u64, uint64_t out[4])
{
    uint32_t nchunks = (n_u64 + CHUNK_U64 - 1) / CHUNK_U64;
    uint32_t cv[8];

    if (nchunks <= 1)
    {
        compress_chunk(in, n_u64, 0ull, true, cv);
        pack_out(cv, out);
        return;
    }

    uint32_t stack[CV_STACK * 8];   // CVs of completed left subtrees awaiting a sibling
    int      slen = 0;
    uint32_t node[8];
    uint32_t idx = 0, remaining = n_u64;

    for (uint32_t ci = 0; ci < nchunks; ++ci)
    {
        uint32_t cu = (remaining >= CHUNK_U64) ? CHUNK_U64 : remaining;
        compress_chunk(in + idx, cu, (uint64_t)ci, false, node);  // this chunk's CV
        idx += cu; remaining -= cu;

        if (ci != nchunks - 1)
        {
            // Merge as many complete right subtrees as the chunk index allows.
            uint64_t total = (uint64_t)ci + 1;
            while ((total & 1ull) == 0)
            {
                --slen;
                parent_cv(&stack[slen * 8], node, node, false);
                total >>= 1;
            }
#pragma unroll
            for (int i = 0; i < 8; ++i) stack[slen * 8 + i] = node[i];
            ++slen;
        }
        else
        {
            // Final chunk: fold it down the stack; the last merge is the root.
            while (slen > 0)
            {
                --slen;
                parent_cv(&stack[slen * 8], node, node, slen == 0);
            }
        }
    }
#pragma unroll
    for (int i = 0; i < 8; ++i) cv[i] = node[i];
    pack_out(cv, out);
}

// One thread per row. Row-major: row tid occupies in[tid*num_cols .. +num_cols).
// Output: 4 u64 digest words per row at out[tid*4 ..].
__global__ void linearHashKernel(uint64_t *__restrict__ out,
                                 const uint64_t *__restrict__ in,
                                 uint32_t num_cols, uint32_t num_rows)
{
    uint64_t tid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_rows) return;

    uint64_t digest[4];
    hash_le64(in + tid * (uint64_t)num_cols, num_cols, digest);

    uint64_t *o = out + tid * 4ull;
#pragma unroll
    for (int i = 0; i < 4; ++i) o[i] = digest[i];
}

// One thread per row, multi-chunk path (num_cols > 128). Kept separate from the
// single-chunk kernel so the common narrow case never pays for the CV stack.
__global__ void linearHashKernelMulti(uint64_t *__restrict__ out,
                                      const uint64_t *__restrict__ in,
                                      uint32_t num_cols, uint32_t num_rows)
{
    uint64_t tid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_rows) return;

    uint64_t digest[4];
    hash_le64_multi(in + tid * (uint64_t)num_cols, num_cols, digest);

    uint64_t *o = out + tid * 4ull;
#pragma unroll
    for (int i = 0; i < 4; ++i) o[i] = digest[i];
}

static constexpr uint32_t TPB = 128;  // threads/block, matches the Poseidon path
static constexpr uint32_t CAPACITY = 4;  // digest words (matches Poseidon CAPACITY)

// Host launcher: same shape as PoseidonGoldilocksGPU<W>::linearHash (RowMajor).
// Dispatches to the single-chunk kernel for num_cols <= 128 (fast, register-only)
// or the multi-chunk tree kernel for wider rows.
inline void blake3_linear_hash(uint64_t *d_out, const uint64_t *d_in,
                               uint64_t num_cols, uint64_t num_rows,
                               cudaStream_t stream)
{
    if (num_rows == 0) return;
    uint64_t blks = (num_rows + TPB - 1) / TPB;
    if (num_cols <= MAX_U64)
        linearHashKernel<<<(unsigned)blks, TPB, 0, stream>>>(
            d_out, d_in, (uint32_t)num_cols, (uint32_t)num_rows);
    else
        linearHashKernelMulti<<<(unsigned)blks, TPB, 0, stream>>>(
            d_out, d_in, (uint32_t)num_cols, (uint32_t)num_rows);
}

// ---------------------------------------------------------------------------
// Merkle tree reduction. One internal node hashes `arity` child digests
// (arity * 4 u64 = arity * 32 bytes) into a 4-u64 parent digest. arity<=4 means
// the node input is <= 16 u64 = 128 bytes = a single chunk (1-2 blocks). The
// tree layout (cursor offsets, zero padding) mirrors PoseidonGoldilocksGPU's
// merkletree so a Poseidon-sized tree buffer is reused unchanged.
// ---------------------------------------------------------------------------
__global__ void merkleNodeKernel(uint64_t *cursor, uint64_t nextN,
                                 uint64_t nextIndex, uint64_t pending, uint32_t arity)
{
    uint64_t tid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= nextN) return;

    const uint64_t base = nextIndex + tid * (uint64_t)arity * CAPACITY;  // arity children
    uint64_t digest[4];
    hash_le64(&cursor[base], arity * CAPACITY, digest);                  // <= 16 u64

    uint64_t *o = &cursor[nextIndex + (pending + tid) * CAPACITY];
#pragma unroll
    for (int i = 0; i < 4; ++i) o[i] = digest[i];
}

// Reduce N leaf digests already resident in d_tree[0 .. N*4) up to the root.
inline void blake3_merkletree_reduce(uint32_t arity, uint64_t *d_tree,
                                     uint64_t num_rows, cudaStream_t stream)
{
    uint64_t pending = num_rows;
    uint64_t nextN   = (pending + arity - 1) / arity;
    uint64_t nextIndex = 0;
    while (pending > 1)
    {
        uint64_t extraZeros = (arity - (pending % arity)) % arity;
        if (extraZeros > 0)
            cudaMemsetAsync(d_tree + nextIndex + pending * CAPACITY, 0,
                            extraZeros * CAPACITY * sizeof(uint64_t), stream);

        uint64_t tpb = TPB, blks = (nextN + TPB - 1) / TPB;
        if (nextN < TPB) { tpb = nextN; blks = 1; }
        merkleNodeKernel<<<(unsigned)blks, (unsigned)tpb, 0, stream>>>(
            d_tree, nextN, nextIndex, pending + extraZeros, arity);

        nextIndex += (pending + extraZeros) * CAPACITY;
        pending = (pending + arity - 1) / arity;
        nextN   = (pending + arity - 1) / arity;
    }
}

// Full Merkle tree: leaf linear hash (RowMajor) + arity-ary reduction.
inline void blake3_merkletree(uint32_t arity, uint64_t *d_tree, uint64_t *d_input,
                              uint64_t num_cols, uint64_t num_rows, cudaStream_t stream)
{
    if (num_rows == 0) return;
    blake3_linear_hash(d_tree, d_input, num_cols, num_rows, stream);  // leaves -> d_tree
    blake3_merkletree_reduce(arity, d_tree, num_rows, stream);
}

}  // namespace blake3gpu

#endif  // BLAKE3_DEVICE_CUH
