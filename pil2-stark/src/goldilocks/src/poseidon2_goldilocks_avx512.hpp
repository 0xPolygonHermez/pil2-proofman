#ifndef POSEIDON2_GOLDILOCKS_AVX512
#define POSEIDON2_GOLDILOCKS_AVX512
#ifdef __AVX512__
#include "poseidon2_goldilocks.hpp"
#include "goldilocks_base_field.hpp"
#include <immintrin.h>


inline void Poseidon2Goldilocks::hash_batch_avx512(Goldilocks::Element (&state)[8 * CAPACITY], Goldilocks::Element const (&input)[8 * SPONGE_WIDTH])
{
    Goldilocks::Element aux[8 * SPONGE_WIDTH];
    hash_full_result_batch_avx512(aux, input);
    for(uint64_t i = 0; i < 8; ++i) {
        std::memcpy(&state[4*i], &aux[i * SPONGE_WIDTH], CAPACITY * sizeof(Goldilocks::Element));
    }
}


inline void Poseidon2Goldilocks::matmul_m4_batch_avx(__m512i &st0, __m512i &st1, __m512i &st2, __m512i &st3) {
    __m512i t0, t0_2, t1, t1_2, t2, t3, t4, t5, t6, t7;
    Goldilocks::add_avx512(t0, st0, st1);
    Goldilocks::add_avx512(t1, st2, st3);
    Goldilocks::add_avx512(t2, st1, st1);
    Goldilocks::add_avx512(t2, t2, t1);
    Goldilocks::add_avx512(t3, st3, st3);
    Goldilocks::add_avx512(t3, t3, t0);
    Goldilocks::add_avx512(t1_2, t1, t1);
    Goldilocks::add_avx512(t0_2, t0, t0);
    Goldilocks::add_avx512(t4, t1_2, t1_2);
    Goldilocks::add_avx512(t4, t4, t3);
    Goldilocks::add_avx512(t5, t0_2, t0_2);
    Goldilocks::add_avx512(t5, t5, t2);
    Goldilocks::add_avx512(t6, t3, t5);
    Goldilocks::add_avx512(t7, t2, t4);

    Goldilocks::copy_avx512(st0, t6);
    Goldilocks::copy_avx512(st1, t5);
    Goldilocks::copy_avx512(st2, t7);
    Goldilocks::copy_avx512(st3, t4);
}

inline void Poseidon2Goldilocks::matmul_external_batch_avx(__m512i *x) {
    matmul_m4_batch_avx512(x[0], x[1], x[2], x[3]);
    matmul_m4_batch_avx512(x[4], x[5], x[6], x[7]);
    matmul_m4_batch_avx512(x[8], x[9], x[10], x[11]);

    __m512i stored[4];
    Goldilocks::add_avx512(stored[0], x[0], x[4]);
    Goldilocks::add_avx512(stored[0], stored[0], x[8]);
    Goldilocks::add_avx512(stored[1], x[1], x[5]);
    Goldilocks::add_avx512(stored[1], stored[1], x[9]);
    Goldilocks::add_avx512(stored[2], x[2], x[6]);
    Goldilocks::add_avx512(stored[2], stored[2], x[10]);
    Goldilocks::add_avx512(stored[3], x[3], x[7]);
    Goldilocks::add_avx512(stored[3], stored[3], x[11]);

    for (int i = 0; i < SPONGE_WIDTH; ++i)
    {
        Goldilocks::add_avx512(x[i], x[i], stored[i % 4]);
    }
}

inline void Poseidon2Goldilocks::element_pow7_avx(__m512i &x) {
    __m512i x2, x3, x4;
    Goldilocks::square_avx512(x2, x);
    Goldilocks::mult_avx512(x3, x, x2);
    Goldilocks::square_avx512(x4, x2);
    Goldilocks::mult_avx512(x, x3, x4);
}

inline void Poseidon2Goldilocks::pow7add_avx512(__m512i *x, const Goldilocks::Element C_[SPONGE_WIDTH]) {
    __m512i x2[SPONGE_WIDTH], x3[SPONGE_WIDTH], x4[SPONGE_WIDTH];

    __m512i c[SPONGE_WIDTH];
    for (int i = 0; i < SPONGE_WIDTH; ++i)
    {
        c[i] = _mm512_set1_epi64(C_[i].fe);
        Goldilocks::add_avx(x[i], x[i], c[i]);
        Goldilocks::square_avx(x2[i], x[i]);
        Goldilocks::square_avx(x4[i], x2[i]);
        Goldilocks::mult_avx(x3[i], x[i], x2[i]);
        Goldilocks::mult_avx(x[i], x3[i], x4[i]);
    }
}

// inline void Poseidon2Goldilocks::hash_avx512(Goldilocks::Element (&state)[2 * CAPACITY], Goldilocks::Element const (&input)[2 * SPONGE_WIDTH])
// {
//     Goldilocks::Element aux[2 * SPONGE_WIDTH];
//     hash_full_result_avx512(aux, input);
//     std::memcpy(state, aux, 2 * CAPACITY * sizeof(Goldilocks::Element));
// }

// inline void Poseidon2Goldilocks::matmul_external_avx512(__m512i &st0, __m512i &st1, __m512i &st2)
// {
//     __m512i indx1 = _mm512_set_epi64(13, 12, 5, 4, 9, 8, 1, 0);
//     __m512i indx2 = _mm512_set_epi64(15, 14, 7, 6, 11, 10, 3, 2);

//     __m512i t0 = _mm512_permutex2var_epi64(st0, indx1, st2);
//     __m512i t1 = _mm512_permutex2var_epi64(st1, indx1, zero);
//     __m512i t2 = _mm512_permutex2var_epi64(st0, indx2, st2);
//     __m512i t3 = _mm512_permutex2var_epi64(st1, indx2, zero);

//     __m512i c0 = _mm512_castpd_si512(_mm512_unpacklo_pd(_mm512_castsi512_pd(t0), _mm512_castsi512_pd(t1)));
//     __m512i c1 = _mm512_castpd_si512(_mm512_unpackhi_pd(_mm512_castsi512_pd(t0), _mm512_castsi512_pd(t1)));
//     __m512i c2 = _mm512_castpd_si512(_mm512_unpacklo_pd(_mm512_castsi512_pd(t2), _mm512_castsi512_pd(t3)));
//     __m512i c3 = _mm512_castpd_si512(_mm512_unpackhi_pd(_mm512_castsi512_pd(t2), _mm512_castsi512_pd(t3)));

//     __m512i t0, t0_2, t1, t1_2, t2, t3, t4, t5, t6, t7;
//     Goldilocks::add_avx512(t0, c0, c1);
//     Goldilocks::add_avx512(t1, c2, c3);
//     Goldilocks::add_avx512(t2, c1, c1);
//     Goldilocks::add_avx512(t2, t2, t1);
//     Goldilocks::add_avx512(t3, c3, c3);
//     Goldilocks::add_avx512(t3, t3, t0);
//     Goldilocks::add_avx512(t1_2, t1, t1);
//     Goldilocks::add_avx512(t0_2, t0, t0);
//     Goldilocks::add_avx512(t4, t1_2, t1_2);
//     Goldilocks::add_avx512(t4, t4, t3);
//     Goldilocks::add_avx512(t5, t0_2, t0_2);
//     Goldilocks::add_avx512(t5, t5, t2);
//     Goldilocks::add_avx512(t6, t3, t5);
//     Goldilocks::add_avx512(t7, t2, t4);

//     // Step 1: Reverse unpacking
//     t0_ = _mm512_castpd_si512(_mm512_unpacklo_pd(_mm512_castsi512_pd(t6), _mm512_castsi512_pd(t5)));
//     t1_ = _mm512_castpd_si512(_mm512_unpackhi_pd(_mm512_castsi512_pd(t6), _mm512_castsi512_pd(t5)));
//     t2_ = _mm512_castpd_si512(_mm512_unpacklo_pd(_mm512_castsi512_pd(t7), _mm512_castsi512_pd(t4)));
//     t3_ = _mm512_castpd_si512(_mm512_unpackhi_pd(_mm512_castsi512_pd(t7), _mm512_castsi512_pd(t4)));

//     // Step 2: Reverse _mm512_permutex2var_epi64
    
    
//     __m512i stored;
//     Goldilocks::add_avx512(stored, st0, st1);
//     Goldilocks::add_avx512(stored, stored, st2);

//     Goldilocks::add_avx512(st0, st0, stored);
//     Goldilocks::add_avx512(st1, st1, stored);
//     Goldilocks::add_avx512(st2, st2, stored);
// };



// inline void Poseidon2Goldilocks::pow7_avx512(__m512i &st0, __m512i &st1, __m512i &st2)
// {
//     __m512i pw2_0, pw2_1, pw2_2;
//     Goldilocks::square_avx512(pw2_0, st0);
//     Goldilocks::square_avx512(pw2_1, st1);
//     Goldilocks::square_avx512(pw2_2, st2);
//     __m512i pw4_0, pw4_1, pw4_2;
//     Goldilocks::square_avx512(pw4_0, pw2_0);
//     Goldilocks::square_avx512(pw4_1, pw2_1);
//     Goldilocks::square_avx512(pw4_2, pw2_2);
//     __m512i pw3_0, pw3_1, pw3_2;
//     Goldilocks::mult_avx512(pw3_0, pw2_0, st0);
//     Goldilocks::mult_avx512(pw3_1, pw2_1, st1);
//     Goldilocks::mult_avx512(pw3_2, pw2_2, st2);

//     Goldilocks::mult_avx512(st0, pw3_0, pw4_0);
//     Goldilocks::mult_avx512(st1, pw3_1, pw4_1);
//     Goldilocks::mult_avx512(st2, pw3_2, pw4_2);
// };

// inline void Poseidon2Goldilocks::add_avx512(__m512i &st0, __m512i &st1, __m512i &st2, const Goldilocks::Element C_[SPONGE_WIDTH])
// {
//     __m512i c0 = _mm512_set4_epi64(C_[3].fe, C_[2].fe, C_[1].fe, C_[0].fe);
//     __m512i c1 = _mm512_set4_epi64(C_[7].fe, C_[6].fe, C_[5].fe, C_[4].fe);
//     __m512i c2 = _mm512_set4_epi64(C_[11].fe, C_[10].fe, C_[9].fe, C_[8].fe);
//     Goldilocks::add_avx512(st0, st0, c0);
//     Goldilocks::add_avx512(st1, st1, c1);
//     Goldilocks::add_avx512(st2, st2, c2);
// }

// inline void Poseidon2Goldilocks::add_avx512_small(__m512i &st0, __m512i &st1, __m512i &st2, const Goldilocks::Element C_small[SPONGE_WIDTH])
// {
//     __m512i c0 = _mm512_set4_epi64(C_small[3].fe, C_small[2].fe, C_small[1].fe, C_small[0].fe);
//     __m512i c1 = _mm512_set4_epi64(C_small[7].fe, C_small[6].fe, C_small[5].fe, C_small[4].fe);
//     __m512i c2 = _mm512_set4_epi64(C_small[11].fe, C_small[10].fe, C_small[9].fe, C_small[8].fe);

//     Goldilocks::add_avx512_b_c(st0, st0, c0);
//     Goldilocks::add_avx512_b_c(st1, st1, c1);
//     Goldilocks::add_avx512_b_c(st2, st2, c2);
// }
#endif
#endif