#ifndef BLAKE3_CORE_HPP
#define BLAKE3_CORE_HPP

// ---------------------------------------------------------------------------
// blake3_core.hpp -- shared scalar BLAKE3 core, used by BOTH the CPU
// (Blake3Goldilocks, g++) and GPU (Blake3GoldilocksGPU, nvcc) paths so the
// prover and verifier are bit-identical by construction.
// All outputs are reduced to canonical Goldilocks elements (in [0, p)).
//
// Conventions (match the bench-validated implementation):
//   * Inputs are whole Goldilocks u64 words, little-endian.
//   * A BLAKE3 block is 64 bytes = 8 u64 words; a chunk is 1024 bytes = 128 u64.
//   * Leaf/node digest = first 32 bytes (4 u64). Transcript "permute" = 64-byte
//     XOF output (8 u64), the drop-in for PoseidonGoldilocks<8>::permute.
// ---------------------------------------------------------------------------

#include <cstdint>

#if defined(__CUDACC__)
#define B3_HD __host__ __device__ __forceinline__
#else
#define B3_HD inline
#endif

namespace blake3core {

static constexpr uint64_t GL_P = 0xFFFFFFFF00000001ULL;

B3_HD uint64_t to_canonical(uint64_t x)
{
    // p = 2^64 - 2^32 + 1, so 2^64 - p = 2^32 - 1 < p: one conditional subtract
    // maps any u64 into [0, p).
    return (x >= GL_P) ? (x - GL_P) : x;
}

static constexpr uint8_t FLAG_CHUNK_START = 1u << 0;
static constexpr uint8_t FLAG_CHUNK_END   = 1u << 1;
static constexpr uint8_t FLAG_PARENT      = 1u << 2;
static constexpr uint8_t FLAG_ROOT        = 1u << 3;

static constexpr uint32_t BLOCK_U64 = 8;     // u64 words per block (64 bytes)
static constexpr uint32_t CHUNK_U64 = 128;   // u64 words per chunk (1024 bytes)
static constexpr int      CV_STACK  = 24;    // handles up to 2^24 chunks

// IV / message schedule. A plain namespace-scope constexpr array is host-only
// for nvcc when indexed at runtime in device code, so keep a host copy and a
// __device__ copy and pick per compilation pass via __CUDA_ARCH__ accessors.
#define B3_IV_INIT { \
    0x6A09E667u, 0xBB67AE85u, 0x3C6EF372u, 0xA54FF53Au, \
    0x510E527Fu, 0x9B05688Cu, 0x1F83D9ABu, 0x5BE0CD19u }
#define B3_SIGMA_INIT { \
    {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15}, \
    {2, 6, 3, 10, 7, 0, 4, 13, 1, 11, 12, 5, 9, 14, 15, 8}, \
    {3, 4, 10, 12, 13, 2, 7, 14, 6, 5, 9, 0, 11, 15, 8, 1}, \
    {10, 7, 12, 9, 14, 3, 13, 15, 4, 0, 11, 2, 5, 8, 1, 6}, \
    {12, 13, 9, 11, 15, 10, 14, 8, 7, 2, 5, 3, 0, 1, 6, 4}, \
    {9, 14, 11, 5, 8, 12, 15, 1, 13, 3, 0, 10, 2, 6, 4, 7}, \
    {11, 15, 5, 0, 1, 9, 8, 6, 14, 10, 2, 12, 3, 4, 7, 13} }

static const uint32_t IV_host[8] = B3_IV_INIT;
static const uint8_t  SIGMA_host[7][16] = B3_SIGMA_INIT;
#if defined(__CUDACC__)
__device__ static const uint32_t IV_dev[8] = B3_IV_INIT;
__device__ static const uint8_t  SIGMA_dev[7][16] = B3_SIGMA_INIT;
#endif

B3_HD uint32_t b3_iv(int i)
{
#ifdef __CUDA_ARCH__
    return IV_dev[i];
#else
    return IV_host[i];
#endif
}

B3_HD uint8_t b3_msg(int r, int k)
{
#ifdef __CUDA_ARCH__
    return SIGMA_dev[r][k];
#else
    return SIGMA_host[r][k];
#endif
}

B3_HD uint32_t rotr32(uint32_t x, uint32_t n) { return (x >> n) | (x << (32u - n)); }

B3_HD void g(uint32_t *st, int a, int b, int c, int d, uint32_t mx, uint32_t my)
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

template<int R>
B3_HD void round_fn(uint32_t st[16], const uint32_t m[16])
{
    g(st, 0, 4,  8, 12, m[b3_msg(R, 0)],  m[b3_msg(R, 1)]);
    g(st, 1, 5,  9, 13, m[b3_msg(R, 2)],  m[b3_msg(R, 3)]);
    g(st, 2, 6, 10, 14, m[b3_msg(R, 4)],  m[b3_msg(R, 5)]);
    g(st, 3, 7, 11, 15, m[b3_msg(R, 6)],  m[b3_msg(R, 7)]);
    g(st, 0, 5, 10, 15, m[b3_msg(R, 8)],  m[b3_msg(R, 9)]);
    g(st, 1, 6, 11, 12, m[b3_msg(R, 10)], m[b3_msg(R, 11)]);
    g(st, 2, 7,  8, 13, m[b3_msg(R, 12)], m[b3_msg(R, 13)]);
    g(st, 3, 4,  9, 14, m[b3_msg(R, 14)], m[b3_msg(R, 15)]);
}

// Fill state[16] from cv + block + counter/len/flags and run the 7 rounds.
B3_HD void compress_pre(uint32_t st[16], const uint32_t cv[8], const uint32_t block[16],
                        uint8_t block_len, uint64_t counter, uint8_t flags)
{
    for (int i = 0; i < 8; ++i) st[i] = cv[i];
    st[8] = b3_iv(0); st[9] = b3_iv(1); st[10] = b3_iv(2); st[11] = b3_iv(3);
    st[12] = (uint32_t)counter;
    st[13] = (uint32_t)(counter >> 32);
    st[14] = (uint32_t)block_len;
    st[15] = (uint32_t)flags;
    round_fn<0>(st, block); round_fn<1>(st, block); round_fn<2>(st, block);
    round_fn<3>(st, block); round_fn<4>(st, block); round_fn<5>(st, block);
    round_fn<6>(st, block);
}

// Standard chaining-value update (8-word output).
B3_HD void compress_in_place(uint32_t cv[8], const uint32_t block[16],
                             uint8_t block_len, uint64_t counter, uint8_t flags)
{
    uint32_t st[16];
    compress_pre(st, cv, block, block_len, counter, flags);
    for (int i = 0; i < 8; ++i) cv[i] = st[i] ^ st[i + 8];
}

// XOF output (16-word output) -- used by the transcript "permute".
B3_HD void compress_xof(const uint32_t cv[8], const uint32_t block[16],
                        uint8_t block_len, uint64_t counter, uint8_t flags,
                        uint32_t out[16])
{
    uint32_t st[16];
    compress_pre(st, cv, block, block_len, counter, flags);
    for (int i = 0; i < 8; ++i) out[i] = st[i] ^ st[i + 8];
    for (int i = 0; i < 8; ++i) out[8 + i] = st[8 + i] ^ cv[i];
}

// Compress one chunk (<= 128 u64) into its 8-word chaining value. `root` sets
// the ROOT flag on the final block (single-chunk inputs only).
B3_HD void compress_chunk(const uint64_t *in, uint32_t n_u64, uint64_t counter,
                          bool root, uint32_t cv[8])
{
    for (int i = 0; i < 8; ++i) cv[i] = b3_iv(i);
    uint32_t nblocks = (n_u64 + 7u) / 8u;
    if (nblocks == 0) nblocks = 1;
    uint32_t idx = 0, remaining = n_u64;
    for (uint32_t b = 0; b < nblocks; ++b)
    {
        uint32_t in_block = (remaining >= 8u) ? 8u : remaining;
        uint32_t block[16];
        for (int k = 0; k < 8; ++k)
        {
            // Field-element inputs must be canonicalized: the proof serializes
            // every value mod p (verifier reads via fromString, which reduces),
            // so the prover must hash the same canonical bytes or BLAKE3 (which,
            // unlike Poseidon, is not invariant under +p) diverges from verify.
            uint64_t v = ((uint32_t)k < in_block) ? to_canonical(in[idx + k]) : 0ull;
            block[2 * k]     = (uint32_t)v;
            block[2 * k + 1] = (uint32_t)(v >> 32);
        }
        uint8_t flags = 0;
        if (b == 0)           flags |= FLAG_CHUNK_START;
        if (b == nblocks - 1) { flags |= FLAG_CHUNK_END; if (root) flags |= FLAG_ROOT; }
        compress_in_place(cv, block, (uint8_t)(in_block * 8u), counter, flags);
        idx += in_block; remaining -= in_block;
    }
}

B3_HD void parent_cv(const uint32_t left[8], const uint32_t right[8], bool root, uint32_t out[8])
{
    uint32_t block[16];
    for (int i = 0; i < 8; ++i) { block[i] = left[i]; block[8 + i] = right[i]; }
    uint32_t cv[8];
    for (int i = 0; i < 8; ++i) cv[i] = b3_iv(i);
    uint8_t flags = FLAG_PARENT | (root ? FLAG_ROOT : 0);
    compress_in_place(cv, block, 64, 0ull, flags);
    for (int i = 0; i < 8; ++i) out[i] = cv[i];
}

B3_HD void pack4(const uint32_t cv[8], uint64_t out[4])
{
    out[0] = to_canonical((uint64_t)cv[0] | ((uint64_t)cv[1] << 32));
    out[1] = to_canonical((uint64_t)cv[2] | ((uint64_t)cv[3] << 32));
    out[2] = to_canonical((uint64_t)cv[4] | ((uint64_t)cv[5] << 32));
    out[3] = to_canonical((uint64_t)cv[6] | ((uint64_t)cv[7] << 32));
}

// BLAKE3-256 of n_u64 Goldilocks words -> 4 canonical u64 digest words.
// Single chunk for n_u64 <= 128; otherwise the standard chunk Merkle tree.
B3_HD void hash_le64(const uint64_t *in, uint32_t n_u64, uint64_t out[4])
{
    uint32_t nchunks = (n_u64 + CHUNK_U64 - 1) / CHUNK_U64;
    uint32_t cv[8];

    if (nchunks <= 1)
    {
        compress_chunk(in, n_u64, 0ull, true, cv);
        pack4(cv, out);
        return;
    }

    uint32_t stack[CV_STACK * 8];
    int slen = 0;
    uint32_t node[8];
    uint32_t idx = 0, remaining = n_u64;
    for (uint32_t ci = 0; ci < nchunks; ++ci)
    {
        uint32_t cu = (remaining >= CHUNK_U64) ? CHUNK_U64 : remaining;
        compress_chunk(in + idx, cu, (uint64_t)ci, false, node);
        idx += cu; remaining -= cu;
        if (ci != nchunks - 1)
        {
            uint64_t total = (uint64_t)ci + 1;
            while ((total & 1ull) == 0)
            {
                uint32_t merged[8];
                parent_cv(&stack[(slen - 1) * 8], node, false, merged);
                for (int i = 0; i < 8; ++i) node[i] = merged[i];
                --slen; total >>= 1;
            }
            for (int i = 0; i < 8; ++i) stack[slen * 8 + i] = node[i];
            ++slen;
        }
        else
        {
            while (slen > 0)
            {
                uint32_t merged[8];
                parent_cv(&stack[(slen - 1) * 8], node, slen == 1, merged);
                for (int i = 0; i < 8; ++i) node[i] = merged[i];
                --slen;
            }
        }
    }
    pack4(node, out);
}

// Transcript "permute": n_u64 in -> n_u64 canonical u64 out, a BLAKE3 XOF over
// the n_u64-word (single-chunk) input, squeezing n_u64 words back. Drop-in for
// the Poseidon sponge permutation of width n_u64 (8/12/16 for arity 2/3/4).
B3_HD void permute_xof(const uint64_t *in, uint32_t n_u64, uint64_t *out)
{
    uint32_t cv[8];
    for (int i = 0; i < 8; ++i) cv[i] = b3_iv(i);

    uint32_t nblocks = (n_u64 + 7u) / 8u;
    if (nblocks == 0) nblocks = 1;

    // Absorb all but the last block; keep the last (root) block for the XOF.
    uint32_t last_block[16];
    uint8_t  last_len = 0, last_flags = 0;
    uint32_t idx = 0, remaining = n_u64;
    for (uint32_t b = 0; b < nblocks; ++b)
    {
        uint32_t in_block = (remaining >= 8u) ? 8u : remaining;
        uint32_t block[16];
        for (int k = 0; k < 8; ++k)
        {
            // Canonicalize field inputs (see compress_chunk): the transcript /
            // calculateHash absorb raw .fe buffers, but the verifier re-hashes
            // the mod-p value carried in the proof.
            uint64_t v = ((uint32_t)k < in_block) ? to_canonical(in[idx + k]) : 0ull;
            block[2 * k]     = (uint32_t)v;
            block[2 * k + 1] = (uint32_t)(v >> 32);
        }
        uint8_t flags = 0;
        if (b == 0) flags |= FLAG_CHUNK_START;
        if (b == nblocks - 1)
        {
            flags |= (FLAG_CHUNK_END | FLAG_ROOT);
            for (int j = 0; j < 16; ++j) last_block[j] = block[j];
            last_len = (uint8_t)(in_block * 8u);
            last_flags = flags;
        }
        else
        {
            compress_in_place(cv, block, (uint8_t)(in_block * 8u), 0ull, flags);
        }
        idx += in_block; remaining -= in_block;
    }

    // Squeeze n_u64 words from the root node (XOF: output-block counter ob).
    uint32_t out_blocks = (n_u64 + 7u) / 8u;
    if (out_blocks == 0) out_blocks = 1;
    for (uint32_t ob = 0; ob < out_blocks; ++ob)
    {
        uint32_t xof[16];
        compress_xof(cv, last_block, last_len, (uint64_t)ob, last_flags, xof);
        for (int k = 0; k < 8; ++k)
        {
            uint32_t oi = ob * 8u + (uint32_t)k;
            if (oi < n_u64)
                out[oi] = to_canonical((uint64_t)xof[2 * k] | ((uint64_t)xof[2 * k + 1] << 32));
        }
    }
}

// Width-8 permutation (grinding + arity-2 transcript).
B3_HD void permute8(const uint64_t in[8], uint64_t out[8]) { permute_xof(in, 8, out); }

}  // namespace blake3core

#endif  // BLAKE3_CORE_HPP
