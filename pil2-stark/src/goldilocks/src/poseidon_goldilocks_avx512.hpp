#ifndef POSEIDON_GOLDILOCKS_AVX512
#define POSEIDON_GOLDILOCKS_AVX512
#ifdef __AVX512__

#include "poseidon_goldilocks.hpp"
#include "goldilocks_base_field.hpp"
#include <immintrin.h>

// AVX512 primitives for Poseidon v1 over Goldilocks.
// Math identical to pil2-stark 0.14.0 poseidon_goldilocks_avx512.hpp;
// templated per class template parameter W.

template<uint32_t W>
inline void PoseidonGoldilocks<W>::pow7_avx512(__m512i &st0, __m512i &st1, __m512i &st2)
{
    __m512i pw2_0, pw2_1, pw2_2;
    Goldilocks::square_avx512(pw2_0, st0);
    Goldilocks::square_avx512(pw2_1, st1);
    Goldilocks::square_avx512(pw2_2, st2);
    __m512i pw4_0, pw4_1, pw4_2;
    Goldilocks::square_avx512(pw4_0, pw2_0);
    Goldilocks::square_avx512(pw4_1, pw2_1);
    Goldilocks::square_avx512(pw4_2, pw2_2);
    __m512i pw3_0, pw3_1, pw3_2;
    Goldilocks::mult_avx512(pw3_0, pw2_0, st0);
    Goldilocks::mult_avx512(pw3_1, pw2_1, st1);
    Goldilocks::mult_avx512(pw3_2, pw2_2, st2);

    Goldilocks::mult_avx512(st0, pw3_0, pw4_0);
    Goldilocks::mult_avx512(st1, pw3_1, pw4_1);
    Goldilocks::mult_avx512(st2, pw3_2, pw4_2);
}

template<uint32_t W>
inline void PoseidonGoldilocks<W>::add_avx512(__m512i &st0, __m512i &st1, __m512i &st2, const Goldilocks::Element C_[SPONGE_WIDTH])
{
    __m512i c0 = _mm512_set4_epi64(C_[3].fe, C_[2].fe, C_[1].fe, C_[0].fe);
    __m512i c1 = _mm512_set4_epi64(C_[7].fe, C_[6].fe, C_[5].fe, C_[4].fe);
    __m512i c2 = _mm512_set4_epi64(C_[11].fe, C_[10].fe, C_[9].fe, C_[8].fe);
    Goldilocks::add_avx512(st0, st0, c0);
    Goldilocks::add_avx512(st1, st1, c1);
    Goldilocks::add_avx512(st2, st2, c2);
}

template<uint32_t W>
inline void PoseidonGoldilocks<W>::add_avx512_a(__m512i &st0, __m512i &st1, __m512i &st2, const Goldilocks::Element C_[SPONGE_WIDTH])
{
    // Aligned-load variant. The set4_epi64 path is already optimal for broadcast
    // of 12 scalars across 2 sponges (8 lanes), matching the non-aligned form.
    add_avx512(st0, st1, st2, C_);
}

template<uint32_t W>
inline void PoseidonGoldilocks<W>::add_avx512_small(__m512i &st0, __m512i &st1, __m512i &st2, const Goldilocks::Element C_small[SPONGE_WIDTH])
{
    __m512i c0 = _mm512_set4_epi64(C_small[3].fe, C_small[2].fe, C_small[1].fe, C_small[0].fe);
    __m512i c1 = _mm512_set4_epi64(C_small[7].fe, C_small[6].fe, C_small[5].fe, C_small[4].fe);
    __m512i c2 = _mm512_set4_epi64(C_small[11].fe, C_small[10].fe, C_small[9].fe, C_small[8].fe);

    Goldilocks::add_avx512_b_c(st0, st0, c0);
    Goldilocks::add_avx512_b_c(st1, st1, c1);
    Goldilocks::add_avx512_b_c(st2, st2, c2);
}

#endif // __AVX512__
#endif // POSEIDON_GOLDILOCKS_AVX512
