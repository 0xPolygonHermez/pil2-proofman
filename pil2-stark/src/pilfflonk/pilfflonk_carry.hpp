#ifndef PILFFLONK_CARRY_HPP
#define PILFFLONK_CARRY_HPP

// _addcarry_u64 / _subborrow_u64: the x86 intrinsics, or the same operations through
// unsigned __int128 elsewhere (Apple Silicon has no <x86intrin.h>).

#if defined(__x86_64__) || defined(_M_X64)
#include <x86intrin.h>
#else
inline unsigned char _addcarry_u64(unsigned char c, unsigned long long a, unsigned long long b,
                                   unsigned long long *out) {
    const unsigned __int128 s = (unsigned __int128)a + b + c;
    *out = (unsigned long long)s;
    return (unsigned char)(s >> 64);
}

inline unsigned char _subborrow_u64(unsigned char br, unsigned long long a, unsigned long long b,
                                    unsigned long long *out) {
    const unsigned __int128 d = (unsigned __int128)a - b - br;
    *out = (unsigned long long)d;
    return (unsigned char)((d >> 64) & 1);
}
#endif

#endif
