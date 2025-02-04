#ifndef POSEIDON2_GOLDILOCKS_AVX
#define POSEIDON2_GOLDILOCKS_AVX

#include "poseidon2_goldilocks.hpp"
#include "goldilocks_base_field.hpp"
#include <immintrin.h>

const __m256i zero = _mm256_setzero_si256();

inline void Poseidon2Goldilocks::hash(Goldilocks::Element (&state)[CAPACITY], Goldilocks::Element const (&input)[SPONGE_WIDTH])
{
    Goldilocks::Element aux[SPONGE_WIDTH];
    hash_full_result(aux, input);
    std::memcpy(state, aux, CAPACITY * sizeof(Goldilocks::Element));
}

inline void Poseidon2Goldilocks::matmul_external_avx(__m256i &st0, __m256i &st1, __m256i &st2)
{

    __m256i t0_ = _mm256_permute2f128_si256(st0, st2, 0b00100000);
    __m256i t1_ = _mm256_permute2f128_si256(st1, zero, 0b00100000);
    __m256i t2_ = _mm256_permute2f128_si256(st0, st2, 0b00110001);
    __m256i t3_ = _mm256_permute2f128_si256(st1, zero, 0b00110001);
    __m256i c0 = _mm256_castpd_si256(_mm256_unpacklo_pd(_mm256_castsi256_pd(t0_), _mm256_castsi256_pd(t1_)));
    __m256i c1 = _mm256_castpd_si256(_mm256_unpackhi_pd(_mm256_castsi256_pd(t0_), _mm256_castsi256_pd(t1_)));
    __m256i c2 = _mm256_castpd_si256(_mm256_unpacklo_pd(_mm256_castsi256_pd(t2_), _mm256_castsi256_pd(t3_)));
    __m256i c3 = _mm256_castpd_si256(_mm256_unpackhi_pd(_mm256_castsi256_pd(t2_), _mm256_castsi256_pd(t3_)));

    __m256i t0, t0_2, t1, t1_2, t2, t3, t4, t5, t6, t7;
    Goldilocks::add_avx(t0, c0, c1);
    Goldilocks::add_avx(t1, c2, c3);
    Goldilocks::add_avx(t2, c1, c1);
    Goldilocks::add_avx(t2, t2, t1);
    Goldilocks::add_avx(t3, c3, c3);
    Goldilocks::add_avx(t3, t3, t0);
    Goldilocks::add_avx(t1_2, t1, t1);
    Goldilocks::add_avx(t0_2, t0, t0);
    Goldilocks::add_avx(t4, t1_2, t1_2);
    Goldilocks::add_avx(t4, t4, t3);
    Goldilocks::add_avx(t5, t0_2, t0_2);
    Goldilocks::add_avx(t5, t5, t2);
    Goldilocks::add_avx(t6, t3, t5);
    Goldilocks::add_avx(t7, t2, t4);

    // Step 1: Reverse unpacking
    t0_ = _mm256_castpd_si256(_mm256_unpacklo_pd(_mm256_castsi256_pd(t6), _mm256_castsi256_pd(t5)));
    t1_ = _mm256_castpd_si256(_mm256_unpackhi_pd(_mm256_castsi256_pd(t6), _mm256_castsi256_pd(t5)));
    t2_ = _mm256_castpd_si256(_mm256_unpacklo_pd(_mm256_castsi256_pd(t7), _mm256_castsi256_pd(t4)));
    t3_ = _mm256_castpd_si256(_mm256_unpackhi_pd(_mm256_castsi256_pd(t7), _mm256_castsi256_pd(t4)));

    // Step 2: Reverse _mm256_permute2f128_si256
    st0 = _mm256_permute2f128_si256(t0_, t2_, 0b00100000); // Combine low halves
    st2 = _mm256_permute2f128_si256(t0_, t2_, 0b00110001); // Combine high halves
    st1 = _mm256_permute2f128_si256(t1_, t3_, 0b00100000); // Combine low halves
    
    __m256i stored;
    Goldilocks::add_avx(stored, st0, st1);
    Goldilocks::add_avx(stored, stored, st2);

    Goldilocks::add_avx(st0, st0, stored);
    Goldilocks::add_avx(st1, st1, stored);
    Goldilocks::add_avx(st2, st2, stored);
};

inline void Poseidon2Goldilocks::pow7_avx(__m256i &st0, __m256i &st1, __m256i &st2)
{
    __m256i pw2_0, pw2_1, pw2_2;
    Goldilocks::square_avx(pw2_0, st0);
    Goldilocks::square_avx(pw2_1, st1);
    Goldilocks::square_avx(pw2_2, st2);
    __m256i pw4_0, pw4_1, pw4_2;
    Goldilocks::square_avx(pw4_0, pw2_0);
    Goldilocks::square_avx(pw4_1, pw2_1);
    Goldilocks::square_avx(pw4_2, pw2_2);
    __m256i pw3_0, pw3_1, pw3_2;
    Goldilocks::mult_avx(pw3_0, pw2_0, st0);
    Goldilocks::mult_avx(pw3_1, pw2_1, st1);
    Goldilocks::mult_avx(pw3_2, pw2_2, st2);

    Goldilocks::mult_avx(st0, pw3_0, pw4_0);
    Goldilocks::mult_avx(st1, pw3_1, pw4_1);
    Goldilocks::mult_avx(st2, pw3_2, pw4_2);
};

inline void Poseidon2Goldilocks::add_avx(__m256i &st0, __m256i &st1, __m256i &st2, const Goldilocks::Element C_[SPONGE_WIDTH])
{
    __m256i c0, c1, c2;
    Goldilocks::load_avx(c0, &(C_[0]));
    Goldilocks::load_avx(c1, &(C_[4]));
    Goldilocks::load_avx(c2, &(C_[8]));
    Goldilocks::add_avx(st0, st0, c0);
    Goldilocks::add_avx(st1, st1, c1);
    Goldilocks::add_avx(st2, st2, c2);
}
// Assuming C_a is aligned
inline void Poseidon2Goldilocks::add_avx_a(__m256i &st0, __m256i &st1, __m256i &st2, const Goldilocks::Element C_a[SPONGE_WIDTH])
{
    __m256i c0, c1, c2;
    Goldilocks::load_avx_a(c0, &(C_a[0]));
    Goldilocks::load_avx_a(c1, &(C_a[4]));
    Goldilocks::load_avx_a(c2, &(C_a[8]));
    Goldilocks::add_avx(st0, st0, c0);
    Goldilocks::add_avx(st1, st1, c1);
    Goldilocks::add_avx(st2, st2, c2);
}
inline void Poseidon2Goldilocks::add_avx_small(__m256i &st0, __m256i &st1, __m256i &st2, const Goldilocks::Element C_small[SPONGE_WIDTH])
{
    __m256i c0, c1, c2;
    Goldilocks::load_avx(c0, &(C_small[0]));
    Goldilocks::load_avx(c1, &(C_small[4]));
    Goldilocks::load_avx(c2, &(C_small[8]));

    Goldilocks::add_avx_b_small(st0, st0, c0);
    Goldilocks::add_avx_b_small(st1, st1, c1);
    Goldilocks::add_avx_b_small(st2, st2, c2);
}

#endif