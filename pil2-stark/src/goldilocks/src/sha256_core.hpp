#ifndef SHA256_CORE_HPP
#define SHA256_CORE_HPP

// Shared scalar SHA-256 core, compiled by BOTH g++ (Sha256Goldilocks) and nvcc
// (Sha256GoldilocksGPU) so prover and verifier are bit-identical. Inputs are
// canonical little-endian Goldilocks u64; a block is 8 words, a digest 4.
//
// Three constructions, because SHA-256 has no parameter block: FIPS carries the
// length as message data, costing a whole extra compression whenever the byte
// length is 0 mod 64.
//   hash_le64   LEAVES  -- literal FIPS 180-4. Width varies, so the length binds.
//   node_hash   NODES   -- compression chain from IV_NODE, no padding. Length is
//                          fixed (arity * DIGEST_U64) and IV_NODE separates the
//                          domain, so nothing is ambiguous. Stricter than blake3,
//                          whose node hash IS its 8-element leaf hash.
//   grind_hash  PoW     -- one compression from IV_GRIND. Runs ~2^31 times.
// The two IVs are SHA-256 digests of ASCII domain strings, so anyone can rederive
// them.
//
// NOTE a digest is 4 u64, HALF of blake3's 8-u64 compress_xof output. Where the
// API asks for 8 (runGrindingPermute) only the low 4 carry a digest.
// ---------------------------------------------------------------------------

#include <cstdint>

#if defined(__CUDACC__)
#define SHA_HD __host__ __device__ __forceinline__
#else
#define SHA_HD inline
#endif

namespace sha256core {

static constexpr uint64_t GL_P = 0xFFFFFFFF00000001ULL;

SHA_HD uint64_t to_canonical(uint64_t x)
{
    // 2^64 - p = 2^32 - 1 < p, so one conditional subtract lands in [0, p).
    return (x >= GL_P) ? (x - GL_P) : x;
}

static constexpr uint32_t BLOCK_U64  = 8;   // u64 words per block (64 bytes)
static constexpr uint32_t DIGEST_U64 = 4;   // u64 words per digest (32 bytes)

// FIPS 180-4 initial hash value: the fractional parts of sqrt(2..19).
#define SHA_IV_INIT { \
    0x6A09E667u, 0xBB67AE85u, 0x3C6EF372u, 0xA54FF53Au, \
    0x510E527Fu, 0x9B05688Cu, 0x1F83D9ABu, 0x5BE0CD19u }

// IV_NODE = SHA256("pil2-stark/sha256/merkle-node/v1")
#define SHA_IV_NODE_INIT { \
    0x14FDA625u, 0x32FCCD27u, 0x853E32C5u, 0xED19966Du, \
    0x16699720u, 0x63E7CCADu, 0x8AE17E84u, 0xCA32C0F3u }

// IV_GRIND = SHA256("pil2-stark/sha256/grinding/v1")
#define SHA_IV_GRIND_INIT { \
    0xA9AFF67Fu, 0xED176A33u, 0xBF35D926u, 0xE35A0AF2u, \
    0x1A4F8C73u, 0x7E6AB8B1u, 0x3460F3DCu, 0x8829058Fu }

// FIPS 180-4 round constants: the fractional parts of cbrt(first 64 primes).
#define SHA_K_INIT { \
    0x428A2F98u, 0x71374491u, 0xB5C0FBCFu, 0xE9B5DBA5u, \
    0x3956C25Bu, 0x59F111F1u, 0x923F82A4u, 0xAB1C5ED5u, \
    0xD807AA98u, 0x12835B01u, 0x243185BEu, 0x550C7DC3u, \
    0x72BE5D74u, 0x80DEB1FEu, 0x9BDC06A7u, 0xC19BF174u, \
    0xE49B69C1u, 0xEFBE4786u, 0x0FC19DC6u, 0x240CA1CCu, \
    0x2DE92C6Fu, 0x4A7484AAu, 0x5CB0A9DCu, 0x76F988DAu, \
    0x983E5152u, 0xA831C66Du, 0xB00327C8u, 0xBF597FC7u, \
    0xC6E00BF3u, 0xD5A79147u, 0x06CA6351u, 0x14292967u, \
    0x27B70A85u, 0x2E1B2138u, 0x4D2C6DFCu, 0x53380D13u, \
    0x650A7354u, 0x766A0ABBu, 0x81C2C92Eu, 0x92722C85u, \
    0xA2BFE8A1u, 0xA81A664Bu, 0xC24B8B70u, 0xC76C51A3u, \
    0xD192E819u, 0xD6990624u, 0xF40E3585u, 0x106AA070u, \
    0x19A4C116u, 0x1E376C08u, 0x2748774Cu, 0x34B0BCB5u, \
    0x391C0CB3u, 0x4ED8AA4Au, 0x5B9CCA4Fu, 0x682E6FF3u, \
    0x748F82EEu, 0x78A5636Fu, 0x84C87814u, 0x8CC70208u, \
    0x90BEFFFAu, 0xA4506CEBu, 0xBEF9A3F7u, 0xC67178F2u }

// Host + __device__ copies picked per compilation pass, as blake3_core does: a
// namespace-scope constexpr array is host-only to nvcc when indexed at runtime.
static const uint32_t IV_host[8]       = SHA_IV_INIT;
static const uint32_t IV_NODE_host[8]  = SHA_IV_NODE_INIT;
static const uint32_t IV_GRIND_host[8] = SHA_IV_GRIND_INIT;
static const uint32_t K_host[64]       = SHA_K_INIT;
#if defined(__CUDACC__)
__device__ static const uint32_t IV_dev[8]       = SHA_IV_INIT;
__device__ static const uint32_t IV_NODE_dev[8]  = SHA_IV_NODE_INIT;
__device__ static const uint32_t IV_GRIND_dev[8] = SHA_IV_GRIND_INIT;
__device__ static const uint32_t K_dev[64]       = SHA_K_INIT;
#endif

SHA_HD uint32_t sha_iv(int i)
{
#ifdef __CUDA_ARCH__
    return IV_dev[i];
#else
    return IV_host[i];
#endif
}

SHA_HD uint32_t sha_iv_node(int i)
{
#ifdef __CUDA_ARCH__
    return IV_NODE_dev[i];
#else
    return IV_NODE_host[i];
#endif
}

SHA_HD uint32_t sha_iv_grind(int i)
{
#ifdef __CUDA_ARCH__
    return IV_GRIND_dev[i];
#else
    return IV_GRIND_host[i];
#endif
}

SHA_HD uint32_t sha_k(int i)
{
#ifdef __CUDA_ARCH__
    return K_dev[i];
#else
    return K_host[i];
#endif
}

SHA_HD uint32_t rotr32(uint32_t x, uint32_t n) { return (x >> n) | (x << (32u - n)); }

// Message bytes arrive little-endian; SHA-256 consumes big-endian words.
SHA_HD uint32_t bswap32(uint32_t x)
{
#ifdef __CUDA_ARCH__
    return __byte_perm(x, 0u, 0x0123u);   // single PRMT instruction
#else
    return ((x >> 24) & 0x000000FFu) | ((x >> 8) & 0x0000FF00u)
         | ((x << 8) & 0x00FF0000u) | ((x << 24) & 0xFF000000u);
#endif
}

SHA_HD uint32_t big_sigma0(uint32_t x) { return rotr32(x, 2) ^ rotr32(x, 13) ^ rotr32(x, 22); }
SHA_HD uint32_t big_sigma1(uint32_t x) { return rotr32(x, 6) ^ rotr32(x, 11) ^ rotr32(x, 25); }
SHA_HD uint32_t small_sigma0(uint32_t x) { return rotr32(x, 7) ^ rotr32(x, 18) ^ (x >> 3); }
SHA_HD uint32_t small_sigma1(uint32_t x) { return rotr32(x, 17) ^ rotr32(x, 19) ^ (x >> 10); }
// 3-input boolean functions, so nvcc folds each to one LOP3.
SHA_HD uint32_t ch(uint32_t x, uint32_t y, uint32_t z) { return z ^ (x & (y ^ z)); }
SHA_HD uint32_t maj(uint32_t x, uint32_t y, uint32_t z) { return (x & y) | (z & (x | y)); }

// One round. Roles rotate by one per round, so callers pass the state shifted.
SHA_HD void step(uint32_t a, uint32_t b, uint32_t c, uint32_t &d,
                 uint32_t e, uint32_t f, uint32_t g, uint32_t &h,
                 uint32_t k, uint32_t w)
{
    uint32_t t1 = h + big_sigma1(e) + ch(e, f, g) + k + w;
    uint32_t t2 = big_sigma0(a) + maj(a, b, c);
    d += t1;
    h  = t1 + t2;
}

// Rolling 16-word schedule window, 16 registers instead of 64. Must run j = 0..15
// in order: each iteration reads words the earlier ones just updated.
SHA_HD void expand(uint32_t W[16])
{
    for (int j = 0; j < 16; ++j)
        W[j] += small_sigma0(W[(j + 1) & 15]) + W[(j + 9) & 15] + small_sigma1(W[(j + 14) & 15]);
}

// 16 rounds from round 16*R. R is a template parameter so every K index folds.
// 16 is a multiple of 8, so roles return to (a..h) and each group looks the same.
template<int R>
SHA_HD void round16(uint32_t st[8], uint32_t W[16])
{
    if (R > 0) expand(W);
    step(st[0], st[1], st[2], st[3], st[4], st[5], st[6], st[7], sha_k(16 * R +  0), W[ 0]);
    step(st[7], st[0], st[1], st[2], st[3], st[4], st[5], st[6], sha_k(16 * R +  1), W[ 1]);
    step(st[6], st[7], st[0], st[1], st[2], st[3], st[4], st[5], sha_k(16 * R +  2), W[ 2]);
    step(st[5], st[6], st[7], st[0], st[1], st[2], st[3], st[4], sha_k(16 * R +  3), W[ 3]);
    step(st[4], st[5], st[6], st[7], st[0], st[1], st[2], st[3], sha_k(16 * R +  4), W[ 4]);
    step(st[3], st[4], st[5], st[6], st[7], st[0], st[1], st[2], sha_k(16 * R +  5), W[ 5]);
    step(st[2], st[3], st[4], st[5], st[6], st[7], st[0], st[1], sha_k(16 * R +  6), W[ 6]);
    step(st[1], st[2], st[3], st[4], st[5], st[6], st[7], st[0], sha_k(16 * R +  7), W[ 7]);
    step(st[0], st[1], st[2], st[3], st[4], st[5], st[6], st[7], sha_k(16 * R +  8), W[ 8]);
    step(st[7], st[0], st[1], st[2], st[3], st[4], st[5], st[6], sha_k(16 * R +  9), W[ 9]);
    step(st[6], st[7], st[0], st[1], st[2], st[3], st[4], st[5], sha_k(16 * R + 10), W[10]);
    step(st[5], st[6], st[7], st[0], st[1], st[2], st[3], st[4], sha_k(16 * R + 11), W[11]);
    step(st[4], st[5], st[6], st[7], st[0], st[1], st[2], st[3], sha_k(16 * R + 12), W[12]);
    step(st[3], st[4], st[5], st[6], st[7], st[0], st[1], st[2], sha_k(16 * R + 13), W[13]);
    step(st[2], st[3], st[4], st[5], st[6], st[7], st[0], st[1], sha_k(16 * R + 14), W[14]);
    step(st[1], st[2], st[3], st[4], st[5], st[6], st[7], st[0], sha_k(16 * R + 15), W[15]);
}

// Compress one block in place. W is CONSUMED by the rolling schedule.
SHA_HD void compress_in_place(uint32_t h[8], uint32_t W[16])
{
    uint32_t st[8];
    for (int i = 0; i < 8; ++i) st[i] = h[i];
    round16<0>(st, W);
    round16<1>(st, W);
    round16<2>(st, W);
    round16<3>(st, W);
    for (int i = 0; i < 8; ++i) h[i] += st[i];
}

// One block from `in_u64` <= 8 words, plus FIPS padding: 0x80 when `pad_start`,
// the big-endian bit length when `with_len`.
//
// Inputs are canonicalized because the proof carries values mod p and the verifier
// re-hashes those; SHA-256, unlike Poseidon, is not invariant under +p.
SHA_HD void fill_block_le64(uint32_t W[16], const uint64_t *in, uint32_t in_u64,
                            bool pad_start, bool with_len, uint64_t total_bits)
{
    for (int k = 0; k < 8; ++k)
    {
        uint32_t lo = 0, hi = 0;
        if ((uint32_t)k < in_u64)
        {
            const uint64_t v = to_canonical(in[k]);
            lo = bswap32((uint32_t)v);          // bytes 8k..8k+3, big-endian
            hi = bswap32((uint32_t)(v >> 32));  // bytes 8k+4..8k+7
        }
        else if ((uint32_t)k == in_u64 && pad_start)
        {
            lo = 0x80000000u;                   // the terminating 1 bit
        }
        W[2 * k]     = lo;
        W[2 * k + 1] = hi;
    }
    if (with_len)
    {
        W[14] = (uint32_t)(total_bits >> 32);
        W[15] = (uint32_t)total_bits;
    }
}

// Digest bytes are h[] big-endian, read back as canonical little-endian u64.
SHA_HD void pack4(const uint32_t h[8], uint64_t out[4])
{
    for (int i = 0; i < 4; ++i)
        out[i] = to_canonical((uint64_t)bswap32(h[2 * i]) | ((uint64_t)bswap32(h[2 * i + 1]) << 32));
}

// LEAVES: literal FIPS 180-4 over `n_u64` words, any length.
SHA_HD void hash_le64(const uint64_t *in, uint32_t n_u64, uint64_t out[4])
{
    uint32_t h[8];
    for (int i = 0; i < 8; ++i) h[i] = sha_iv(i);

    const uint64_t total_bits = (uint64_t)n_u64 * 64ull;
    const uint32_t full = n_u64 / 8u;   // blocks entirely filled with data
    const uint32_t tail = n_u64 % 8u;   // leftover words in the last block
    // 0x80 plus the 8-byte length need 9 spare bytes, so they fit iff tail <= 6.
    const bool tail_fits = (tail <= 6u);

    uint32_t W[16];
    for (uint32_t b = 0; b < full; ++b)
    {
        fill_block_le64(W, in + (uint64_t)b * 8ull, 8u, false, false, 0ull);
        compress_in_place(h, W);
    }
    // Tail/padding block. A 64-byte-aligned message pays a whole extra one.
    fill_block_le64(W, in + (uint64_t)full * 8ull, tail, true, tail_fits, total_bits);
    compress_in_place(h, W);
    if (!tail_fits)
    {
        // Length-only block; the 0x80 went in the block above.
        fill_block_le64(W, nullptr, 0u, false, true, total_bits);
        compress_in_place(h, W);
    }
    pack4(h, out);
}

// NODES: chain from IV_NODE, no padding. `n_u64` MUST be a positive multiple of 8
// (arity * DIGEST_U64); a partial block has no unambiguous encoding here.
SHA_HD void node_hash(const uint64_t *in, uint32_t n_u64, uint64_t out[4])
{
    uint32_t h[8];
    for (int i = 0; i < 8; ++i) h[i] = sha_iv_node(i);

    const uint32_t nblocks = n_u64 / 8u;
    uint32_t W[16];
    for (uint32_t b = 0; b < nblocks; ++b)
    {
        fill_block_le64(W, in + (uint64_t)b * 8ull, 8u, false, false, 0ull);
        compress_in_place(h, W);
    }
    pack4(h, out);
}

// GRINDING: one compression from IV_GRIND. Caller checks out[0] against the target.
SHA_HD void grind_hash(const uint64_t in[8], uint64_t out[4])
{
    uint32_t h[8];
    for (int i = 0; i < 8; ++i) h[i] = sha_iv_grind(i);
    uint32_t W[16];
    fill_block_le64(W, in, 8u, false, false, 0ull);
    compress_in_place(h, W);
    pack4(h, out);
}

// TRANSCRIPT: incremental SHA-256. No invented XOF -- squeezing is two nested
// literal FIPS hashes, SHA256(SHA256(absorbed) || LE64(counter)), a counter-mode
// expansion over a commitment to the transcript. `digest`/`squeeze` are const:
// they finalize a copy, because a transcript keeps absorbing after a challenge.
// A full block compresses as soon as it fills -- the padding is a separate block,
// so nothing must stay buffered for a terminal flag (unlike blake3).
struct Hasher
{
    uint32_t h[8];
    uint64_t buf[BLOCK_U64];   // words not yet forming a whole block
    uint32_t buf_len;
    uint64_t absorbed;         // total words, for the FIPS length field

    SHA_HD void init()
    {
        for (int i = 0; i < 8; ++i) h[i] = sha_iv(i);
        buf_len = 0;
        absorbed = 0;
    }

    SHA_HD void absorb(const uint64_t *in, uint32_t n)
    {
        for (uint32_t i = 0; i < n; ++i)
        {
            buf[buf_len++] = in[i];
            ++absorbed;
            if (buf_len == BLOCK_U64)
            {
                uint32_t W[16];
                fill_block_le64(W, buf, BLOCK_U64, false, false, 0ull);
                compress_in_place(h, W);
                buf_len = 0;
            }
        }
    }

    // SHA256 of everything absorbed so far. Does not consume.
    SHA_HD void digest(uint64_t out[4]) const
    {
        uint32_t hc[8];
        for (int i = 0; i < 8; ++i) hc[i] = h[i];

        const uint64_t total_bits = absorbed * 64ull;
        const bool tail_fits = (buf_len <= 6u);

        uint32_t W[16];
        fill_block_le64(W, buf, buf_len, true, tail_fits, total_bits);
        compress_in_place(hc, W);
        if (!tail_fits)
        {
            fill_block_le64(W, nullptr, 0u, false, true, total_bits);
            compress_in_place(hc, W);
        }
        pack4(hc, out);
    }

    // SHA256(digest || LE64(counter)): 4 words per call, 40 bytes = one block.
    SHA_HD void squeeze(uint64_t counter, uint64_t out[4]) const
    {
        uint64_t d[4];
        digest(d);
        const uint64_t msg[5] = {d[0], d[1], d[2], d[3], counter};
        hash_le64(msg, 5, out);
    }
};

}  // namespace sha256core

#endif  // SHA256_CORE_HPP
