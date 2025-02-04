#ifndef POSEIDON2_GOLDILOCKS_AVX512
#define POSEIDON2_GOLDILOCKS_AVX512
#ifdef __AVX512__
#include "poseidon2_goldilocks.hpp"
#include "goldilocks_base_field.hpp"
#include <immintrin.h>

inline void Poseidon2Goldilocks::hash_avx512(Goldilocks::Element (&state)[2 * CAPACITY], Goldilocks::Element const (&input)[2 * SPONGE_WIDTH])
{
    Goldilocks::Element aux[2 * SPONGE_WIDTH];
    hash_full_result_avx512(aux, input);
    std::memcpy(state, aux, 2 * CAPACITY * sizeof(Goldilocks::Element));
}

inline void Poseidon2Goldilocks::matmul_external_avx512(__m512i &st0, __m512i &st1, __m512i &st2)
{
    __m512i indx1 = _mm512_set_epi64(13, 12, 5, 4, 9, 8, 1, 0);
    __m512i indx2 = _mm512_set_epi64(15, 14, 7, 6, 11, 10, 3, 2);

    __m512i t0 = _mm512_permutex2var_epi64(st0, indx1, st2);
    __m512i t1 = _mm512_permutex2var_epi64(st1, indx1, zero);
    __m512i t2 = _mm512_permutex2var_epi64(st0, indx2, st2);
    __m512i t3 = _mm512_permutex2var_epi64(st1, indx2, zero);

    __m512i c0 = _mm512_castpd_si512(_mm512_unpacklo_pd(_mm512_castsi512_pd(t0), _mm512_castsi512_pd(t1)));
    __m512i c1 = _mm512_castpd_si512(_mm512_unpackhi_pd(_mm512_castsi512_pd(t0), _mm512_castsi512_pd(t1)));
    __m512i c2 = _mm512_castpd_si512(_mm512_unpacklo_pd(_mm512_castsi512_pd(t2), _mm512_castsi512_pd(t3)));
    __m512i c3 = _mm512_castpd_si512(_mm512_unpackhi_pd(_mm512_castsi512_pd(t2), _mm512_castsi512_pd(t3)));

    __m512i t0, t0_2, t1, t1_2, t2, t3, t4, t5, t6, t7;
    Goldilocks::add_avx512(t0, c0, c1);
    Goldilocks::add_avx512(t1, c2, c3);
    Goldilocks::add_avx512(t2, c1, c1);
    Goldilocks::add_avx512(t2, t2, t1);
    Goldilocks::add_avx512(t3, c3, c3);
    Goldilocks::add_avx512(t3, t3, t0);
    Goldilocks::add_avx512(t1_2, t1, t1);
    Goldilocks::add_avx512(t0_2, t0, t0);
    Goldilocks::add_avx512(t4, t1_2, t1_2);
    Goldilocks::add_avx512(t4, t4, t3);
    Goldilocks::add_avx512(t5, t0_2, t0_2);
    Goldilocks::add_avx512(t5, t5, t2);
    Goldilocks::add_avx512(t6, t3, t5);
    Goldilocks::add_avx512(t7, t2, t4);

    // Step 1: Reverse unpacking
    t0_ = _mm512_castpd_si512(_mm512_unpacklo_pd(_mm512_castsi512_pd(t6), _mm512_castsi512_pd(t5)));
    t1_ = _mm512_castpd_si512(_mm512_unpackhi_pd(_mm512_castsi512_pd(t6), _mm512_castsi512_pd(t5)));
    t2_ = _mm512_castpd_si512(_mm512_unpacklo_pd(_mm512_castsi512_pd(t7), _mm512_castsi512_pd(t4)));
    t3_ = _mm512_castpd_si512(_mm512_unpackhi_pd(_mm512_castsi512_pd(t7), _mm512_castsi512_pd(t4)));

    // Step 2: Reverse _mm512_permutex2var_epi64
    
    
    __m512i stored;
    Goldilocks::add_avx512(stored, st0, st1);
    Goldilocks::add_avx512(stored, stored, st2);

    Goldilocks::add_avx512(st0, st0, stored);
    Goldilocks::add_avx512(st1, st1, stored);
    Goldilocks::add_avx512(st2, st2, stored);
};



inline void Poseidon2Goldilocks::pow7_avx512(__m512i &st0, __m512i &st1, __m512i &st2)
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
};

inline void Poseidon2Goldilocks::add_avx512(__m512i &st0, __m512i &st1, __m512i &st2, const Goldilocks::Element C_[SPONGE_WIDTH])
{
    __m512i c0 = _mm512_set4_epi64(C_[3].fe, C_[2].fe, C_[1].fe, C_[0].fe);
    __m512i c1 = _mm512_set4_epi64(C_[7].fe, C_[6].fe, C_[5].fe, C_[4].fe);
    __m512i c2 = _mm512_set4_epi64(C_[11].fe, C_[10].fe, C_[9].fe, C_[8].fe);
    Goldilocks::add_avx512(st0, st0, c0);
    Goldilocks::add_avx512(st1, st1, c1);
    Goldilocks::add_avx512(st2, st2, c2);
}

inline void Poseidon2Goldilocks::add_avx512_small(__m512i &st0, __m512i &st1, __m512i &st2, const Goldilocks::Element C_small[SPONGE_WIDTH])
{
    __m512i c0 = _mm512_set4_epi64(C_small[3].fe, C_small[2].fe, C_small[1].fe, C_small[0].fe);
    __m512i c1 = _mm512_set4_epi64(C_small[7].fe, C_small[6].fe, C_small[5].fe, C_small[4].fe);
    __m512i c2 = _mm512_set4_epi64(C_small[11].fe, C_small[10].fe, C_small[9].fe, C_small[8].fe);

    Goldilocks::add_avx512_b_c(st0, st0, c0);
    Goldilocks::add_avx512_b_c(st1, st1, c1);
    Goldilocks::add_avx512_b_c(st2, st2, c2);
}
#endif
#endif