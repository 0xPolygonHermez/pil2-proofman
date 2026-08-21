#ifndef BLAKE3_GOLDILOCKS_CIRCOM
#define BLAKE3_GOLDILOCKS_CIRCOM

// ---------------------------------------------------------------------------
// Witness implementation of the Blake3 circom custom gates.
//
// Companion to circuits.gl/hash/blake3/blake3.circom. The gate bodies there are
// dead code under `extern_c` (the compiler replaces them with a call into this
// file), so these functions are what actually produces the witness.
//
// One gate: Blake3Compress(in[16], blockLen, counterLo, flags, key, raw).
// Two input shapes over one row geometry, selected by `raw`:
//
//   raw = 0   `in[0..8]` is the chaining value, `in[8..16]` a block of eight
//             full-range Goldilocks words, ordered by `key` and split here.
//   raw = 1   `in` IS the sixteen u32 block words -- the parent shape, two
//             chaining values side by side -- and the compression's own
//             chaining value is the IV. `key` is a don't-care on this path.
//
// A cv plus a Goldilocks block is exactly 16 cells, the same as a raw block,
// which is why the parent shape needs no block input of its own and the whole
// design fits a single gate.
//
// counterHi is not passed: st[13] is identically zero across the design (chunk
// index capped at 2^24 by CV_STACK, XOF output block index 0 at width 8,
// parents at counter 0), so it is hardcoded rather than wired.
//
// That round loop is duplicated from blake3core because it has to emit the
// per-G intermediates the AIR constrains, which compress_pre does not expose.
// To keep the duplication from ever drifting, every compression is
// cross-checked against blake3core::compress_xof before returning. The prover
// hashes with that same core, so a silent divergence here would produce proofs
// the verifier rejects.
// ---------------------------------------------------------------------------

#include <cstdint>
#include <cstdio>
#ifdef __AVX2__
#include <immintrin.h>
#endif
#include <stdexcept>

#include "blake3_core.hpp"
#include "goldilocks_base_field.hpp"

namespace {

// im layout -- mirrors the offsets documented in blake3_core.circom.
constexpr uint32_t B3_ROW_CELLS = 53;
constexpr uint32_t B3_OFF_ST = 2968;    // st[4..8] only
constexpr uint32_t B3_OFF_CV = 2984;    // cv[0..4] only
constexpr uint32_t B3_OFF_OUT = 3000;
constexpr uint32_t B3_OFF_SPLIT = 3064;

// The four state words G function `g` of a round mixes: column mixes for
// g < 4, diagonal mixes for g >= 4.
constexpr uint8_t B3_G_IDX[8][4] = {
    {0, 4,  8, 12}, {1, 5,  9, 13}, {2, 6, 10, 14}, {3, 7, 11, 15},
    {0, 5, 10, 15}, {1, 6, 11, 12}, {2, 7,  8, 13}, {3, 4,  9, 14},
};

// A row spends most of its cells on byte and 16-bit-limb decompositions: 528
// put_bytes calls per compression, four u64 stores each. Writing im is ~63% of
// this gate's time (measured against a variant with the stores removed), so
// each of these becomes one 32-byte store rather than four 8-byte ones.
#ifdef __AVX2__
inline void put_bytes(uint64_t *dst, uint32_t w)
{
    const __m256i sh = _mm256_setr_epi64x(0, 8, 16, 24);
    __m256i v = _mm256_srlv_epi64(_mm256_set1_epi64x((long long)w), sh);
    _mm256_storeu_si256((__m256i *)dst, _mm256_and_si256(v, _mm256_set1_epi64x(0xFF)));
}

inline void put_limbs16(uint64_t *dst, uint32_t w)
{
    const __m128i v = _mm_srlv_epi64(_mm_set1_epi64x((long long)w), _mm_set_epi64x(16, 0));
    _mm_storeu_si128((__m128i *)dst, _mm_and_si128(v, _mm_set1_epi64x(0xFFFF)));
}
#else
inline void put_bytes(uint64_t *dst, uint32_t w)
{
    dst[0] = w & 0xFF;
    dst[1] = (w >> 8) & 0xFF;
    dst[2] = (w >> 16) & 0xFF;
    dst[3] = (w >> 24) & 0xFF;
}

inline void put_limbs16(uint64_t *dst, uint32_t w)
{
    dst[0] = w & 0xFFFF;
    dst[1] = (w >> 16) & 0xFFFF;
}
#endif

// The rot-12 nibble table: cell 28+j holds ((zb >> 4j) & 0xF), shifted up by 4
// on even j. Eight scalar stores per row -- the largest scalar block left --
// so it gets the same treatment as put_bytes.
#ifdef __AVX2__
inline void put_nibbles(uint64_t *dst, uint32_t zb)
{
    const __m256i mask = _mm256_set1_epi64x(0xF);
    const __m256i up = _mm256_setr_epi64x(4, 0, 4, 0);
    const __m256i z = _mm256_set1_epi64x((long long)zb);
    __m256i lo = _mm256_and_si256(_mm256_srlv_epi64(z, _mm256_setr_epi64x(0, 4, 8, 12)), mask);
    __m256i hi = _mm256_and_si256(_mm256_srlv_epi64(z, _mm256_setr_epi64x(16, 20, 24, 28)), mask);
    _mm256_storeu_si256((__m256i *)dst, _mm256_sllv_epi64(lo, up));
    _mm256_storeu_si256((__m256i *)(dst + 4), _mm256_sllv_epi64(hi, up));
}
#else
inline void put_nibbles(uint64_t *dst, uint32_t zb)
{
    for (int i = 0; i < 4; ++i)
    {
        const uint32_t zi = (zb >> (8 * i)) & 0xFF;
        dst[2 * i] = (zi & 0xF) << 4;
        dst[2 * i + 1] = zi >> 4;
    }
}
#endif

// Split eight Goldilocks words into the sixteen u32 block words, writing the
// canonicity witnesses into the split section of im.
//
// The ABI passes canonical Goldilocks values (see Poseidon2_16's fromU64/toU64
// round trip), so in[i] is already in [0, p). Since p - 1 = (2^32 - 1) * 2^32,
// a split is canonical iff hi != 2^32 - 1 or lo == 0; isMax flags the boundary
// and dInv witnesses that (2^32 - 1) - hi is invertible away from it.
//
// The eight dInv values are batch-inverted rather than inverted one by one:
// Goldilocks::inv is extended Euclid with a 64-bit division per iteration, and
// eight of them measured as roughly half of this gate's witness time.
// Montgomery's trick trades them for one inversion plus ~21 multiplications.
void split_block(uint64_t *im, uint32_t blk[16], const uint64_t *in)
{
    uint64_t d[8];
    for (int i = 0; i < 8; ++i)
    {
        const uint64_t x = in[i];
        const uint32_t lo = (uint32_t)x;
        const uint32_t hi = (uint32_t)(x >> 32);
        blk[2 * i] = lo;
        blk[2 * i + 1] = hi;

        d[i] = 0xFFFFFFFFull - (uint64_t)hi;
        im[B3_OFF_SPLIT + 2 * i + 1] = (d[i] == 0) ? 1 : 0;   // isMax
    }

    // Prefix products, skipping the boundary words so a zero never enters the
    // running product -- which is also why the inv below cannot see zero.
    Goldilocks::Element prefix[8];
    Goldilocks::Element run = Goldilocks::one();
    for (int i = 0; i < 8; ++i)
    {
        prefix[i] = run;
        if (d[i] != 0) run = Goldilocks::mul(run, Goldilocks::fromU64(d[i]));
    }

    Goldilocks::Element z = Goldilocks::inv(run);
    for (int i = 7; i >= 0; --i)
    {
        if (d[i] == 0)
        {
            im[B3_OFF_SPLIT + 2 * i] = 0;
            continue;
        }
        im[B3_OFF_SPLIT + 2 * i] = Goldilocks::toU64(Goldilocks::mul(z, prefix[i]));
        z = Goldilocks::mul(z, Goldilocks::fromU64(d[i]));
    }
}

// Run one compression, filling im[0, B3_OFF_SPLIT) and out[0, 16).
void compress_instrumented(uint64_t *im, uint64_t *out, const uint32_t cv_in[8],
                           const uint32_t blk[16], uint8_t block_len,
                           uint32_t counter_lo, uint32_t counter_hi, uint8_t flags)
{
    uint32_t st[16];
    for (int i = 0; i < 8; ++i) st[i] = cv_in[i];
    st[8] = blake3core::b3_iv(0);
    st[9] = blake3core::b3_iv(1);
    st[10] = blake3core::b3_iv(2);
    st[11] = blake3core::b3_iv(3);
    st[12] = counter_lo;
    st[13] = counter_hi;
    st[14] = (uint32_t)block_len;
    st[15] = (uint32_t)flags;

    for (uint32_t round = 0; round < 7; ++round)
    {
        for (uint32_t g = 0; g < 8; ++g)
        {
            uint64_t *row = &im[B3_ROW_CELLS * (8 * round + g)];

            const uint8_t ia = B3_G_IDX[g][0];
            const uint8_t ib = B3_G_IDX[g][1];
            const uint8_t ic = B3_G_IDX[g][2];
            const uint8_t id = B3_G_IDX[g][3];

            const uint32_t va = st[ia];
            const uint32_t vb = st[ib];
            const uint32_t vc = st[ic];
            const uint32_t vd = st[id];
            const uint32_t x = blk[blake3core::b3_msg(round, 2 * g)];
            const uint32_t y = blk[blake3core::b3_msg(round, 2 * g + 1)];

            put_limbs16(row + 0, va);
            put_bytes(row + 2, vb);
            put_limbs16(row + 6, vc);
            put_bytes(row + 8, vd);
            put_limbs16(row + 12, x);
            put_limbs16(row + 14, y);

            // va' = (va + vb + x) mod 2^32
            const uint32_t vaP = va + vb + x;
            put_bytes(row + 16, vaP);

            // vd' = (vd ^ va') >>> 16
            const uint32_t vdP = blake3core::rotr32(vd ^ vaP, 16);
            put_bytes(row + 20, vdP);

            // vc' = (vc + vd') mod 2^32
            const uint32_t vcP = vc + vdP;
            put_bytes(row + 24, vcP);

            // vb' = (vb ^ vc') >>> 12
            //
            // 12 is not byte aligned, so the XOR-rotr table splits each operand
            // byte z across two result bytes: (z & 0xF) << 4 and z >> 4.
            const uint32_t zb = vb ^ vcP;
            const uint32_t vbP = blake3core::rotr32(zb, 12);
            put_nibbles(row + 28, zb);

            // va'' = (va' + vb' + y) mod 2^32
            const uint32_t vaPP = vaP + vbP + y;
            put_bytes(row + 36, vaPP);

            // vd'' = (vd' ^ va'') >>> 8
            const uint32_t vdPP = blake3core::rotr32(vdP ^ vaPP, 8);
            put_bytes(row + 40, vdPP);

            // vc'' = (vc' + vd'') mod 2^32
            const uint32_t vcPP = vcP + vdPP;
            put_bytes(row + 44, vcPP);

            // vb'' = (vb' ^ vc'') >>> 7 = rotl1(rotr8(z)); the AIR spends one
            // boolean on the bit rotl1 carries out instead of a rot-7 table.
            const uint32_t zb2 = vbP ^ vcPP;
            const uint32_t vbPP = blake3core::rotr32(zb2, 7);
            put_bytes(row + 48, zb2);
            row[52] = (zb2 >> 7) & 1;

            st[ia] = vaPP;
            st[ib] = vbPP;
            st[ic] = vcPP;
            st[id] = vdPP;
        }
    }

    uint32_t xof[16];
    for (int i = 0; i < 8; ++i)
    {
        xof[i] = st[i] ^ st[8 + i];
        xof[8 + i] = st[8 + i] ^ cv_in[i];
    }

    // Cross-check against the shared core, so the duplicated round loop above
    // cannot drift from what the prover hashed.
    {
        uint32_t ref[16];
        const uint64_t counter = (uint64_t)counter_lo | ((uint64_t)counter_hi << 32);
        blake3core::compress_xof(cv_in, blk, block_len, counter, flags, ref);
        for (int i = 0; i < 16; ++i)
        {
            if (ref[i] != xof[i])
            {
                fprintf(stderr,
                        "Blake3Compress: diverged from blake3core at word %d "
                        "(got %08x, expected %08x)\n",
                        i, xof[i], ref[i]);
                throw std::runtime_error("Blake3Compress diverged from blake3core");
            }
        }
    }

    // The feed-forward sections carry only what the G rows do not already hold.
    // Rows 52-55 store va''/vc''/vd'' as bytes, covering st[0..4] and st[8..16];
    // the four vb'' words st[4..8] are the gap. cv[4..8] are rows 0-3's vb bytes
    // already, so only cv[0..4] need bytes here.
    for (int i = 4; i < 8; ++i)
    {
        put_bytes(&im[B3_OFF_ST + 4 * (i - 4)], st[i]);
    }
    for (int i = 0; i < 4; ++i)
    {
        put_bytes(&im[B3_OFF_CV + 4 * i], cv_in[i]);
    }
    for (int i = 0; i < 16; ++i)
    {
        put_bytes(&im[B3_OFF_OUT + 4 * i], xof[i]);
        out[i] = xof[i];
    }
}

}  // namespace

void Blake3Compress(uint64_t *im, uint *size_im, uint64_t *out, uint *size_out,
                    uint64_t *in, uint *size_in,
                    uint64_t *blockLen, uint *size_blockLen,
                    uint64_t *counterLo, uint *size_counterLo,
                    uint64_t *flags, uint *size_flags,
                    uint64_t *key, uint *size_key,
                    uint64_t *raw, uint *size_raw)
{
    uint32_t cv_in[8];
    uint32_t blk[16];

    if (raw[0] != 0)
    {
        // Parent shape: `in` is already the u32 block, cv is the IV, and
        // nothing was split so the shared split section stays zero.
        for (int i = 0; i < 8; ++i)
        {
            cv_in[i] = blake3core::b3_iv(i);
            blk[i] = (uint32_t)in[i];
            blk[8 + i] = (uint32_t)in[8 + i];
        }
        for (uint32_t i = 0; i < 16; ++i) im[B3_OFF_SPLIT + i] = 0;
    }
    else
    {
        for (int i = 0; i < 8; ++i) cv_in[i] = (uint32_t)in[i];

        // Order the two 4-word halves of the block by the key bit before
        // splitting. Only a Merkle level passes a non-zero key; every other
        // call site passes 0.
        uint64_t ordered[8];
        for (int i = 0; i < 4; ++i)
        {
            if (key[0] == 0)
            {
                ordered[i] = in[8 + i];
                ordered[4 + i] = in[12 + i];
            }
            else
            {
                ordered[i] = in[12 + i];
                ordered[4 + i] = in[8 + i];
            }
        }
        split_block(im, blk, ordered);
    }

    compress_instrumented(im, out, cv_in, blk, (uint8_t)blockLen[0],
                          (uint32_t)counterLo[0], 0, (uint8_t)flags[0]);
}

#endif  // BLAKE3_GOLDILOCKS_CIRCOM
