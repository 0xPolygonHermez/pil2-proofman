#ifndef PILFFLONK_NTT_HPP
#define PILFFLONK_NTT_HPP

#include <cstdint>
#include <memory>
#include <x86intrin.h>

#include "pilfflonk_fr.hpp"

namespace PilFflonk {

// r, little-endian limbs.
constexpr uint64_t FR_LIMBS[4] = {0x43e1f593f0000001ULL, 0x2833e84879b97091ULL, 0xb85045b68181585dULL,
                                  0x30644e72e131a029ULL};

// Inline a ± b mod r for a, b < r: the values of ffiasm's Fr_rawAdd and Fr_rawSub, branchless.

inline void frAdd(FrElement &out, const FrElement &a, const FrElement &b) {
    unsigned long long s[4], d[4];
    unsigned char c = 0;
    c = _addcarry_u64(c, a.v[0], b.v[0], &s[0]);
    c = _addcarry_u64(c, a.v[1], b.v[1], &s[1]);
    c = _addcarry_u64(c, a.v[2], b.v[2], &s[2]);
    _addcarry_u64(c, a.v[3], b.v[3], &s[3]); // r < 2^254: no carry out
    unsigned char br = 0;
    br = _subborrow_u64(br, s[0], FR_LIMBS[0], &d[0]);
    br = _subborrow_u64(br, s[1], FR_LIMBS[1], &d[1]);
    br = _subborrow_u64(br, s[2], FR_LIMBS[2], &d[2]);
    br = _subborrow_u64(br, s[3], FR_LIMBS[3], &d[3]);
    const uint64_t keep = -uint64_t(br);
    for (int i = 0; i < 4; ++i) {
        out.v[i] = (s[i] & keep) | (d[i] & ~keep);
    }
}

inline void frSub(FrElement &out, const FrElement &a, const FrElement &b) {
    unsigned long long d[4], e[4];
    unsigned char br = 0;
    br = _subborrow_u64(br, a.v[0], b.v[0], &d[0]);
    br = _subborrow_u64(br, a.v[1], b.v[1], &d[1]);
    br = _subborrow_u64(br, a.v[2], b.v[2], &d[2]);
    br = _subborrow_u64(br, a.v[3], b.v[3], &d[3]);
    const uint64_t mask = -uint64_t(br);
    unsigned char c = 0;
    c = _addcarry_u64(c, d[0], FR_LIMBS[0] & mask, &e[0]);
    c = _addcarry_u64(c, d[1], FR_LIMBS[1] & mask, &e[1]);
    c = _addcarry_u64(c, d[2], FR_LIMBS[2] & mask, &e[2]);
    _addcarry_u64(c, d[3], FR_LIMBS[3] & mask, &e[3]);
    for (int i = 0; i < 4; ++i) {
        out.v[i] = e[i];
    }
}

// A cache-blocked radix-2 NTT over BN128's Fr with ffiasm's roots (ω_n = 5^((r−1)/n), Montgomery
// form), for sizes up to 2^maxBits: the transforms of ffiasm's FFT, bit for bit. Its functions
// only read it, and may run concurrently.
class Ntt {
public:
    explicit Ntt(uint64_t maxBits);

    // a, 2^k <= 2^maxBits elements in bit-reversed order, into natural order: a'[i] = Σ_j
    // a[BR(j)]·ω^(±i·j), ω = ω_{2^k}, ω^−1 if `inverse` (without the 1/2^k). On all of OpenMP's
    // threads if `parallel`, on the calling one otherwise.
    void fromBitReversed(FrElement *a, uint64_t k, bool inverse, bool parallel) const;

    // a[i] <-> a[BR(i)] for the 2^k elements of a, in place.
    static void bitReverse(FrElement *a, uint64_t k, bool parallel);

    // i with its k low bits reversed (i < 2^k).
    static uint64_t reverse(uint64_t i, uint64_t k) { return k == 0 ? 0 : reverse64(i) >> (64 - k); }

private:
    static uint64_t reverse64(uint64_t x) {
        x = ((x >> 1) & 0x5555555555555555ULL) | ((x & 0x5555555555555555ULL) << 1);
        x = ((x >> 2) & 0x3333333333333333ULL) | ((x & 0x3333333333333333ULL) << 2);
        x = ((x >> 4) & 0x0F0F0F0F0F0F0F0FULL) | ((x & 0x0F0F0F0F0F0F0F0FULL) << 4);
        return __builtin_bswap64(x);
    }

    uint64_t bits;
    // twiddles[m + j] = ω_{2m}^j for each power of two m < 2^bits and j < m.
    std::unique_ptr<FrElement[]> twiddles;
};

} // namespace PilFflonk

#endif
