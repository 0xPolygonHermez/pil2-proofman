#ifndef POSEIDON2_GOLDILOCKS_AVX
#define POSEIDON2_GOLDILOCKS_AVX

#include "poseidon2_goldilocks.hpp"
#include "goldilocks_base_field.hpp"
#define __AVX2__
#ifdef __AVX2__
#include <immintrin.h>

const __m256i zero = _mm256_setzero_si256();

template<uint32_t SPONGE_WIDTH_T>
inline void Poseidon2Goldilocks<SPONGE_WIDTH_T>::hash_avx(Goldilocks::Element (&state)[CAPACITY], Goldilocks::Element const (&input)[SPONGE_WIDTH])
{
    Goldilocks::Element aux[SPONGE_WIDTH];
    hash_full_result_avx(aux, input);
    std::memcpy(state, aux, CAPACITY * sizeof(Goldilocks::Element));
}

template<uint32_t SPONGE_WIDTH_T>
inline void Poseidon2Goldilocks<SPONGE_WIDTH_T>::hash_batch_avx(Goldilocks::Element (&state)[4 * CAPACITY], Goldilocks::Element const (&input)[4 * SPONGE_WIDTH])
{
    Goldilocks::Element aux[4 * SPONGE_WIDTH];
    hash_full_result_batch_avx(aux, input);
    std::memcpy(state, aux, CAPACITY * sizeof(Goldilocks::Element));
    std::memcpy(&state[4], &aux[SPONGE_WIDTH], CAPACITY * sizeof(Goldilocks::Element));
    std::memcpy(&state[8], &aux[2*SPONGE_WIDTH], CAPACITY * sizeof(Goldilocks::Element));
    std::memcpy(&state[12], &aux[3*SPONGE_WIDTH], CAPACITY * sizeof(Goldilocks::Element));
}

template<uint32_t SPONGE_WIDTH_T>
inline void Poseidon2Goldilocks<SPONGE_WIDTH_T>::matmul_m4_batch_avx(__m256i &st0, __m256i &st1, __m256i &st2, __m256i &st3) {
    __m256i t0, t0_2, t1, t1_2, t2, t3, t4, t5, t6, t7;
    Goldilocks::add_avx(t0, st0, st1);
    Goldilocks::add_avx(t1, st2, st3);
    Goldilocks::add_avx(t2, st1, st1);
    Goldilocks::add_avx(t2, t2, t1);
    Goldilocks::add_avx(t3, st3, st3);
    Goldilocks::add_avx(t3, t3, t0);
    Goldilocks::add_avx(t1_2, t1, t1);
    Goldilocks::add_avx(t0_2, t0, t0);
    Goldilocks::add_avx(t4, t1_2, t1_2);
    Goldilocks::add_avx(t4, t4, t3);
    Goldilocks::add_avx(t5, t0_2, t0_2);
    Goldilocks::add_avx(t5, t5, t2);
    Goldilocks::add_avx(t6, t3, t5);
    Goldilocks::add_avx(t7, t2, t4);

    Goldilocks::copy_avx(st0, t6);
    Goldilocks::copy_avx(st1, t5);
    Goldilocks::copy_avx(st2, t7);
    Goldilocks::copy_avx(st3, t4);
}

template<uint32_t SPONGE_WIDTH_T>   
inline void Poseidon2Goldilocks<SPONGE_WIDTH_T>::matmul_external_batch_avx(__m256i *x) {
    matmul_m4_batch_avx(x[0], x[1], x[2], x[3]);
    matmul_m4_batch_avx(x[4], x[5], x[6], x[7]);
    matmul_m4_batch_avx(x[8], x[9], x[10], x[11]);

    __m256i stored[4];
    Goldilocks::add_avx(stored[0], x[0], x[4]);
    Goldilocks::add_avx(stored[0], stored[0], x[8]);
    Goldilocks::add_avx(stored[1], x[1], x[5]);
    Goldilocks::add_avx(stored[1], stored[1], x[9]);
    Goldilocks::add_avx(stored[2], x[2], x[6]);
    Goldilocks::add_avx(stored[2], stored[2], x[10]);
    Goldilocks::add_avx(stored[3], x[3], x[7]);
    Goldilocks::add_avx(stored[3], stored[3], x[11]);

    for (int i = 0; i < SPONGE_WIDTH; ++i)
    {
        Goldilocks::add_avx(x[i], x[i], stored[i % 4]);
    }
}

template<uint32_t SPONGE_WIDTH_T>
inline void Poseidon2Goldilocks<SPONGE_WIDTH_T>::element_pow7_avx(__m256i &x) {
    __m256i x2, x3, x4;
    Goldilocks::square_avx(x2, x);
    Goldilocks::mult_avx(x3, x, x2);
    Goldilocks::square_avx(x4, x2);
    Goldilocks::mult_avx(x, x3, x4);
}

template<uint32_t SPONGE_WIDTH_T>
inline void Poseidon2Goldilocks<SPONGE_WIDTH_T>::pow7add_avx(__m256i *x, const Goldilocks::Element C_[SPONGE_WIDTH]) {
    __m256i x2[SPONGE_WIDTH], x3[SPONGE_WIDTH], x4[SPONGE_WIDTH];

    __m256i c[SPONGE_WIDTH];
    for (int i = 0; i < SPONGE_WIDTH; ++i)
    {
        c[i] = _mm256_set1_epi64x(C_[i].fe);
        Goldilocks::add_avx(x[i], x[i], c[i]);
        Goldilocks::square_avx(x2[i], x[i]);
        Goldilocks::square_avx(x4[i], x2[i]);
        Goldilocks::mult_avx(x3[i], x[i], x2[i]);
        Goldilocks::mult_avx(x[i], x3[i], x4[i]);
    }
}

template<uint32_t SPONGE_WIDTH_T>   
inline void Poseidon2Goldilocks<SPONGE_WIDTH_T>::matmul_external_avx(__m256i st[(SPONGE_WIDTH >> 2)])
{

    assert(SPONGE_WIDTH == 12 || SPONGE_WIDTH == 16);
#if SPONGE_WIDTH == 12
    __m256i t0_ = _mm256_permute2f128_si256(st[0], st[2], 0b00100000);
    __m256i t1_ = _mm256_permute2f128_si256(st[1], zero, 0b00100000);
    __m256i t2_ = _mm256_permute2f128_si256(st[0], st[2], 0b00110001);
    __m256i t3_ = _mm256_permute2f128_si256(st[1], zero, 0b00110001);
    __m256i x0 = _mm256_castpd_si256(_mm256_unpacklo_pd(_mm256_castsi256_pd(t0_), _mm256_castsi256_pd(t1_)));
    __m256i x1 = _mm256_castpd_si256(_mm256_unpackhi_pd(_mm256_castsi256_pd(t0_), _mm256_castsi256_pd(t1_)));
    __m256i x2 = _mm256_castpd_si256(_mm256_unpacklo_pd(_mm256_castsi256_pd(t2_), _mm256_castsi256_pd(t3_)));
    __m256i x3 = _mm256_castpd_si256(_mm256_unpackhi_pd(_mm256_castsi256_pd(t2_), _mm256_castsi256_pd(t3_)));
#else
    __m256i t0_ = _mm256_permute2f128_si256(st[0], st[2], 0b00100000);
    __m256i t1_ = _mm256_permute2f128_si256(st[1], st[3], 0b00100000);
    __m256i t2_ = _mm256_permute2f128_si256(st[0], st[2], 0b00110001);
    __m256i t3_ = _mm256_permute2f128_si256(st[1], st[3], 0b00110001);
    __m256i x0 = _mm256_castpd_si256(_mm256_unpacklo_pd(_mm256_castsi256_pd(t0_), _mm256_castsi256_pd(t1_)));
    __m256i x1 = _mm256_castpd_si256(_mm256_unpackhi_pd(_mm256_castsi256_pd(t0_), _mm256_castsi256_pd(t1_)));
    __m256i x2 = _mm256_castpd_si256(_mm256_unpacklo_pd(_mm256_castsi256_pd(t2_), _mm256_castsi256_pd(t3_)));
    __m256i x3 = _mm256_castpd_si256(_mm256_unpackhi_pd(_mm256_castsi256_pd(t2_), _mm256_castsi256_pd(t3_)));
#endif
    __m256i t0, t0_2, t1, t1_2, t2, t3, t4, t5, t6, t7;
    Goldilocks::add_avx(t0, x0, x1);
    Goldilocks::add_avx(t1, x2, x3);
    Goldilocks::add_avx(t2, x1, x1);
    Goldilocks::add_avx(t2, t2, t1);
    Goldilocks::add_avx(t3, x3, x3);
    Goldilocks::add_avx(t3, t3, t0);
    Goldilocks::add_avx(t1_2, t1, t1);
    Goldilocks::add_avx(t0_2, t0, t0);
    Goldilocks::add_avx(t4, t1_2, t1_2);
    Goldilocks::add_avx(t4, t4, t3);
    Goldilocks::add_avx(t5, t0_2, t0_2);
    Goldilocks::add_avx(t5, t5, t2);
    Goldilocks::add_avx(t6, t3, t5);
    Goldilocks::add_avx(t7, t2, t4);

#if SPONGE_WIDTH == 12
    t0_ = _mm256_castpd_si256(_mm256_unpacklo_pd(_mm256_castsi256_pd(t6), _mm256_castsi256_pd(t5)));
    t1_ = _mm256_castpd_si256(_mm256_unpackhi_pd(_mm256_castsi256_pd(t6), _mm256_castsi256_pd(t5)));
    t2_ = _mm256_castpd_si256(_mm256_unpacklo_pd(_mm256_castsi256_pd(t7), _mm256_castsi256_pd(t4)));
    t3_ = _mm256_castpd_si256(_mm256_unpackhi_pd(_mm256_castsi256_pd(t7), _mm256_castsi256_pd(t4)));

    // Step 2: Reverse _mm256_permute2f128_si256
    st[0] = _mm256_permute2f128_si256(t0_, t2_, 0b00100000); // Combine low halves
    st[2] = _mm256_permute2f128_si256(t0_, t2_, 0b00110001); // Combine high halves
    st[1] = _mm256_permute2f128_si256(t1_, t3_, 0b00100000); // Combine low halves
#else
    t0_ = _mm256_castpd_si256(_mm256_unpacklo_pd(_mm256_castsi256_pd(t6), _mm256_castsi256_pd(t5)));
    t1_ = _mm256_castpd_si256(_mm256_unpackhi_pd(_mm256_castsi256_pd(t6), _mm256_castsi256_pd(t5)));
    t2_ = _mm256_castpd_si256(_mm256_unpacklo_pd(_mm256_castsi256_pd(t7), _mm256_castsi256_pd(t4)));
    t3_ = _mm256_castpd_si256(_mm256_unpackhi_pd(_mm256_castsi256_pd(t7), _mm256_castsi256_pd(t4)));

    // Step 2: Reverse _mm256_permute2f128_si256
    st[0] = _mm256_permute2f128_si256(t0_, t2_, 0b00100000); // Combine low halves
    st[2] = _mm256_permute2f128_si256(t0_, t2_, 0b00110001); // Combine high halves
    st[1] = _mm256_permute2f128_si256(t1_, t3_, 0b00100000); // Combine low halves
    st[3] = _mm256_permute2f128_si256(t1_, t3_, 0b00110001); // Combine high halves
#endif
    __m256i stored;
    if(SPONGE_WIDTH > 4) {
        Goldilocks::add_avx(stored, st[0], st[1]);
        for(int i = 2; i < (SPONGE_WIDTH >> 2); i++) {
            Goldilocks::add_avx(stored, stored, st[i]);            
        }
        for(int i = 0; i < (SPONGE_WIDTH >> 2); i++) {
            Goldilocks::add_avx(st[i], st[i], stored);
        }
    } else {
        exit(1);
    }
};

template<uint32_t SPONGE_WIDTH_T>
inline void Poseidon2Goldilocks<SPONGE_WIDTH_T>::pow7_avx(__m256i st[(SPONGE_WIDTH >> 2)])
{
    for(int i = 0; i < (SPONGE_WIDTH >> 2); i++) {
        __m256i pw2, pw3, pw4;
        Goldilocks::square_avx(pw2, st[i]);
        Goldilocks::square_avx(pw4, pw2);
        Goldilocks::mult_avx(pw3, pw2, st[i]);
        Goldilocks::mult_avx(st[i], pw3, pw4);
    }
};

template<uint32_t SPONGE_WIDTH_T>    
inline void Poseidon2Goldilocks<SPONGE_WIDTH_T>::add_avx(__m256i st[(SPONGE_WIDTH >> 2)], const Goldilocks::Element C_[SPONGE_WIDTH])
{
    for(int i = 0; i < (SPONGE_WIDTH >> 2); i++) {
        __m256i c;
        Goldilocks::load_avx(c, &(C_[i << 2]));
        Goldilocks::add_avx(st[i], st[i], c);
    }
}

template<uint32_t SPONGE_WIDTH_T>    
inline void Poseidon2Goldilocks<SPONGE_WIDTH_T>::add_avx_a(__m256i st[(SPONGE_WIDTH >> 2)], const Goldilocks::Element C_a[SPONGE_WIDTH])
{
    for(int i = 0; i < (SPONGE_WIDTH >> 2); i++) {
        __m256i c;
        Goldilocks::load_avx_a(c, &(C_a[i << 2]));
        Goldilocks::add_avx(st[i], st[i], c);
    }
}

template<uint32_t SPONGE_WIDTH_T>
inline void Poseidon2Goldilocks<SPONGE_WIDTH_T>::add_avx_small(__m256i st[(SPONGE_WIDTH >> 2)], const Goldilocks::Element C_small[SPONGE_WIDTH])
{
    for(int i = 0; i < (SPONGE_WIDTH >> 2); i++) {
        __m256i c;
        Goldilocks::load_avx(c, &(C_small[i << 2]));
        Goldilocks::add_avx_b_small(st[i], st[i], c);
    }
}
#endif
#endif