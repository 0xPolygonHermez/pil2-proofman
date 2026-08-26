#ifndef BLAKE2B_DEVICE_CUH
#define BLAKE2B_DEVICE_CUH

// ---------------------------------------------------------------------------
// blake2b_device.cuh -- self-contained GPU BLAKE2b-256, for benchmarking ONLY.
//
// Direct port of RFC 7693 BLAKE2b to a scalar __device__ implementation: one
// CUDA thread hashes one input, mirroring the Poseidon/BLAKE3 leaf kernels.
//
// BLAKE2b is sequential (no chunk tree): it streams the input through 128-byte
// blocks (16 u64 words), 12 rounds of 64-bit mixing per block, with a byte
// counter and a last-block flag. So a single hash_le64() handles ANY length --
// no single/multi-chunk split. Output is the first 32 bytes (256 bits = 4 u64),
// i.e. BLAKE2b configured with digest_size = 32 (unkeyed).
//
// Inputs are whole Goldilocks field elements (u64, little-endian), so every
// 128-byte block holds an integer number (1..16) of u64 words.
//
// This header lives under benchs/ so the prover library never compiles it.
// ---------------------------------------------------------------------------

#include <cstdint>

namespace blake2bgpu {

static constexpr uint32_t BLOCK_U64 = 16;   // 128 bytes per compression block
static constexpr uint32_t CAPACITY  = 4;    // digest words (matches Poseidon)

// BLAKE2b IV (= SHA-512 IV). constexpr so device code folds constant indices.
__device__ constexpr uint64_t IV[8] = {
    0x6a09e667f3bcc908ull, 0xbb67ae8584caa73bull,
    0x3c6ef372fe94f82bull, 0xa54ff53a5f1d36f1ull,
    0x510e527fade682d1ull, 0x9b05688c2b3e6c1full,
    0x1f83d9abfb41bd6bull, 0x5be0cd19137e2179ull};

// Message-word permutation. 12 rounds; rows 10,11 reuse rows 0,1.
__device__ constexpr uint8_t SIGMA[12][16] = {
    { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9,10,11,12,13,14,15},
    {14,10, 4, 8, 9,15,13, 6, 1,12, 0, 2,11, 7, 5, 3},
    {11, 8,12, 0, 5, 2,15,13,10,14, 3, 6, 7, 1, 9, 4},
    { 7, 9, 3, 1,13,12,11,14, 2, 6, 5,10, 4, 0,15, 8},
    { 9, 0, 5, 7, 2, 4,10,15,14, 1,11,12, 6, 8, 3,13},
    { 2,12, 6,10, 0,11, 8, 3, 4,13, 7, 5,15,14, 1, 9},
    {12, 5, 1,15,14,13, 4,10, 0, 7, 6, 3, 9, 2, 8,11},
    {13,11, 7,14,12, 1, 3, 9, 5, 0,15, 4, 8, 6, 2,10},
    { 6,15,14, 9,11, 3, 0, 8,12, 2,13, 7, 1, 4,10, 5},
    {10, 2, 8, 4, 7, 6, 1, 5,15,11, 9,14, 3,12,13, 0},
    { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9,10,11,12,13,14,15},
    {14,10, 4, 8, 9,15,13, 6, 1,12, 0, 2,11, 7, 5, 3}};

__device__ __forceinline__ uint64_t rotr64(uint64_t x, uint32_t n)
{
    return (x >> n) | (x << (64u - n));
}

// BLAKE2b mixing function.
__device__ __forceinline__ void G(uint64_t *v, int a, int b, int c, int d,
                                   uint64_t x, uint64_t y)
{
    v[a] = v[a] + v[b] + x;
    v[d] = rotr64(v[d] ^ v[a], 32);
    v[c] = v[c] + v[d];
    v[b] = rotr64(v[b] ^ v[c], 24);
    v[a] = v[a] + v[b] + y;
    v[d] = rotr64(v[d] ^ v[a], 16);
    v[c] = v[c] + v[d];
    v[b] = rotr64(v[b] ^ v[c], 63);
}

// One round. R is a template parameter so SIGMA[R][k] is a constant expression
// and the message array stays in registers.
template<int R>
__device__ __forceinline__ void round_fn(uint64_t v[16], const uint64_t m[16])
{
    G(v, 0, 4,  8, 12, m[SIGMA[R][0]],  m[SIGMA[R][1]]);
    G(v, 1, 5,  9, 13, m[SIGMA[R][2]],  m[SIGMA[R][3]]);
    G(v, 2, 6, 10, 14, m[SIGMA[R][4]],  m[SIGMA[R][5]]);
    G(v, 3, 7, 11, 15, m[SIGMA[R][6]],  m[SIGMA[R][7]]);
    G(v, 0, 5, 10, 15, m[SIGMA[R][8]],  m[SIGMA[R][9]]);
    G(v, 1, 6, 11, 12, m[SIGMA[R][10]], m[SIGMA[R][11]]);
    G(v, 2, 7,  8, 13, m[SIGMA[R][12]], m[SIGMA[R][13]]);
    G(v, 3, 4,  9, 14, m[SIGMA[R][14]], m[SIGMA[R][15]]);
}

// Compress one 128-byte block (16 LE u64 words) into the state h, in place.
// `t` = total bytes hashed through this block; `last` sets the final-block flag.
__device__ __forceinline__ void compress(uint64_t h[8], const uint64_t m[16],
                                          uint64_t t, bool last)
{
    uint64_t v[16];
#pragma unroll
    for (int i = 0; i < 8; ++i) v[i] = h[i];
#pragma unroll
    for (int i = 0; i < 8; ++i) v[8 + i] = IV[i];
    v[12] ^= t;                                  // low 64 bits of byte counter
    // v[13] ^= 0  (high 64 bits -- always 0 for our input sizes)
    if (last) v[14] ^= 0xFFFFFFFFFFFFFFFFull;

    round_fn<0>(v, m);  round_fn<1>(v, m);  round_fn<2>(v, m);  round_fn<3>(v, m);
    round_fn<4>(v, m);  round_fn<5>(v, m);  round_fn<6>(v, m);  round_fn<7>(v, m);
    round_fn<8>(v, m);  round_fn<9>(v, m);  round_fn<10>(v, m); round_fn<11>(v, m);

#pragma unroll
    for (int i = 0; i < 8; ++i) h[i] ^= v[i] ^ v[8 + i];
}

// Sequential BLAKE2b-256 of n_u64 Goldilocks words -> 4 u64 digest words.
// Handles any length (1 .. many blocks). digest_size = 32 -> parameter block
// XOR is 0x01010020 into h[0].
__device__ __forceinline__ void hash_le64(const uint64_t *in, uint32_t n_u64,
                                          uint64_t out[4])
{
    uint64_t h[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) h[i] = IV[i];
    h[0] ^= 0x0000000001010020ull;   // 0x01010000 ^ key_len(0)<<8 ^ digest_len(32)

    uint32_t nblocks = (n_u64 + BLOCK_U64 - 1) / BLOCK_U64;
    if (nblocks == 0) nblocks = 1;   // empty input still compresses one zero block

    uint32_t idx = 0;
    uint32_t remaining = n_u64;
    for (uint32_t b = 0; b < nblocks; ++b)
    {
        uint32_t in_block = (remaining >= BLOCK_U64) ? BLOCK_U64 : remaining;
        uint64_t m[16];
#pragma unroll
        for (int k = 0; k < 16; ++k)
            m[k] = ((uint32_t)k < in_block) ? in[idx + k] : 0ull;

        bool last = (b == nblocks - 1);
        uint64_t t = last ? (uint64_t)n_u64 * 8ull : (uint64_t)(b + 1) * 128ull;
        compress(h, m, t, last);
        idx       += in_block;
        remaining -= in_block;
    }

#pragma unroll
    for (int i = 0; i < 4; ++i) out[i] = h[i];   // BLAKE2b output is little-endian
}

// One thread per row, RowMajor input (matches the Poseidon/BLAKE3 leaf kernels).
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

// Internal Merkle node: hash arity child digests (arity*4 u64). For arity 4 the
// input is 16 u64 = 128 bytes = exactly one BLAKE2b block.
__global__ void merkleNodeKernel(uint64_t *cursor, uint64_t nextN,
                                 uint64_t nextIndex, uint64_t pending, uint32_t arity)
{
    uint64_t tid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= nextN) return;

    const uint64_t base = nextIndex + tid * (uint64_t)arity * CAPACITY;
    uint64_t digest[4];
    hash_le64(&cursor[base], arity * CAPACITY, digest);

    uint64_t *o = &cursor[nextIndex + (pending + tid) * CAPACITY];
#pragma unroll
    for (int i = 0; i < 4; ++i) o[i] = digest[i];
}

static constexpr uint32_t TPB = 128;

inline void blake2b_linear_hash(uint64_t *d_out, const uint64_t *d_in,
                                uint64_t num_cols, uint64_t num_rows,
                                cudaStream_t stream)
{
    if (num_rows == 0) return;
    uint64_t blks = (num_rows + TPB - 1) / TPB;
    linearHashKernel<<<(unsigned)blks, TPB, 0, stream>>>(
        d_out, d_in, (uint32_t)num_cols, (uint32_t)num_rows);
}

inline void blake2b_merkletree_reduce(uint32_t arity, uint64_t *d_tree,
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

inline void blake2b_merkletree(uint32_t arity, uint64_t *d_tree, uint64_t *d_input,
                               uint64_t num_cols, uint64_t num_rows, cudaStream_t stream)
{
    if (num_rows == 0) return;
    blake2b_linear_hash(d_tree, d_input, num_cols, num_rows, stream);
    blake2b_merkletree_reduce(arity, d_tree, num_rows, stream);
}

}  // namespace blake2bgpu

#endif  // BLAKE2B_DEVICE_CUH
