// Witness of the `extern_c` custom gates of the BN128 final circuits, whose circom bodies are dead
// under it: Num2Bytes (circuits.bn128/custom/rangecheck.circom) and, of a blake3 key, the Blake3
// gates (circuits.gl/hash/blake3/blake3.circom), as setup/circom/blake3_gate.cpp is for the
// Goldilocks recursion, over circom's FrElement.

#include <cstdint>
#include <sys/types.h>

#include "blake3_core.hpp"
#include "fr.hpp"

namespace {

constexpr uint64_t GOLDILOCKS_P = 0xFFFFFFFF00000001ull;

inline uint64_t canonical(uint64_t x) { return x >= GOLDILOCKS_P ? x - GOLDILOCKS_P : x; }

// The value of `a`, every one of the gates' below 2^64.
inline uint64_t to_u64(FrElement *a) {
    FrElement n;
    Fr_toNormal(&n, a);
    return (n.type & Fr_LONG) ? n.longVal[0] : (uint64_t)(int64_t)n.shortVal;
}

inline void set_u64(FrElement *r, uint64_t v) {
    if (v < 0x80000000ull) {
        r->type = Fr_SHORT;
        r->shortVal = (int32_t)v;
    } else {
        r->type = Fr_LONG;
        r->shortVal = 0;
        r->longVal[0] = v;
        r->longVal[1] = r->longVal[2] = r->longVal[3] = 0;
    }
}

// The block of eight Goldilocks words `w`, halves swapped by `key`, as sixteen u32 words.
inline void split_ordered(uint32_t block[16], const uint64_t w[8], uint64_t key) {
    for (int i = 0; i < 8; ++i) {
        const uint64_t x = canonical(w[key ? (i + 4) % 8 : i]);
        block[2 * i] = (uint32_t)x;
        block[2 * i + 1] = (uint32_t)(x >> 32);
    }
}

}  // namespace

void Blake3Node(FrElement *out, uint *size_out, FrElement *in, uint *size_in, FrElement *key, uint *size_key) {
    uint32_t cv[8];
    uint32_t block[16];
    uint64_t w[8];
    for (int i = 0; i < 8; ++i) {
        cv[i] = blake3core::b3_iv(i);
        w[i] = to_u64(&in[i]);
    }
    split_ordered(block, w, to_u64(key));
    uint32_t xof[16];
    const uint8_t flags = blake3core::FLAG_CHUNK_START | blake3core::FLAG_CHUNK_END | blake3core::FLAG_ROOT;
    blake3core::compress_xof(cv, block, 64, 0, flags, xof);
    for (int i = 0; i < 4; ++i) {
        set_u64(&out[i], canonical((uint64_t)xof[2 * i] + ((uint64_t)xof[2 * i + 1] << 32)));
    }
}

void Blake3Compress(FrElement *flags, FrElement *isParent, FrElement *out, uint *size_out, FrElement *in,
                    uint *size_in, FrElement *blockLen, uint *size_blockLen, FrElement *counterLo,
                    uint *size_counterLo) {
    uint32_t cv[8];
    uint32_t block[16];
    if (to_u64(isParent) != 0) {
        for (int i = 0; i < 8; ++i) {
            cv[i] = blake3core::b3_iv(i);
            block[i] = (uint32_t)to_u64(&in[i]);
            block[8 + i] = (uint32_t)to_u64(&in[8 + i]);
        }
    } else {
        uint64_t w[8];
        for (int i = 0; i < 8; ++i) {
            cv[i] = (uint32_t)to_u64(&in[i]);
            w[i] = to_u64(&in[8 + i]);
        }
        split_ordered(block, w, 0);
    }
    uint32_t xof[16];
    blake3core::compress_xof(cv, block, (uint8_t)to_u64(blockLen), to_u64(counterLo), (uint8_t)to_u64(flags), xof);
    for (int i = 0; i < 16; ++i) set_u64(&out[i], xof[i]);
}

// `in` in chunks of 16 bits, least significant first, ⌈nBits/16⌉ of them (nBits <= 80).
void Num2Bytes(FrElement *nBits, FrElement *out, uint *size_out, FrElement *in, uint *size_in) {
    FrElement n;
    Fr_toNormal(&n, in);
    uint64_t limbs[2] = {0, 0};
    if (n.type & Fr_LONG) {
        limbs[0] = n.longVal[0];
        limbs[1] = n.longVal[1];
    } else {
        limbs[0] = (uint64_t)(int64_t)n.shortVal;
    }
    const uint64_t n_chunks = (to_u64(nBits) + 15) / 16;
    for (uint64_t k = 0; k < n_chunks; ++k) {
        const uint64_t bit = 16 * k;
        const uint64_t chunk = bit < 64 ? limbs[0] >> bit | (bit ? limbs[1] << (64 - bit) : 0) : limbs[1] >> (bit - 64);
        set_u64(&out[k], chunk & 0xffff);
    }
}
