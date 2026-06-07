#ifndef POSEIDON2_GOLDILOCKS_AVX512
#define POSEIDON2_GOLDILOCKS_AVX512
#ifdef __AVX512__
#include "poseidon2_goldilocks.hpp"
#include "goldilocks_base_field.hpp"
#include <immintrin.h>


template<uint32_t SPONGE_WIDTH_T>
inline void Poseidon2Goldilocks<SPONGE_WIDTH_T>::permuteTrunc_batch_avx512(Goldilocks::Element (&state)[8 * CAPACITY], Goldilocks::Element const (&input)[8 * SPONGE_WIDTH])
{
    Goldilocks::Element aux[8 * SPONGE_WIDTH];
    permute_batch_avx512(aux, input);
    for(uint64_t i = 0; i < 8; ++i) {
        std::memcpy(&state[4*i], &aux[i * SPONGE_WIDTH], CAPACITY * sizeof(Goldilocks::Element));
    }
}

template<uint32_t SPONGE_WIDTH_T>
inline void Poseidon2Goldilocks<SPONGE_WIDTH_T>::matmul_m4_batch_avx512(__m512i &st0, __m512i &st1, __m512i &st2, __m512i &st3) {
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

template<uint32_t SPONGE_WIDTH_T>
inline void Poseidon2Goldilocks<SPONGE_WIDTH_T>::matmul_external_batch_avx512(__m512i *x) {
    for (uint32_t i = 0; i < SPONGE_WIDTH; i += 4) {
        matmul_m4_batch_avx512(x[i], x[i+1], x[i+2], x[i+3]);
    }
    if (SPONGE_WIDTH > 4) {
        __m512i stored[4];
        Goldilocks::add_avx512(stored[0], x[0], x[4]);
        Goldilocks::add_avx512(stored[1], x[1], x[5]);
        Goldilocks::add_avx512(stored[2], x[2], x[6]);
        Goldilocks::add_avx512(stored[3], x[3], x[7]);
        for (uint32_t i = 8; i < SPONGE_WIDTH; i += 4) {
            Goldilocks::add_avx512(stored[0], stored[0], x[i]);
            Goldilocks::add_avx512(stored[1], stored[1], x[i+1]);
            Goldilocks::add_avx512(stored[2], stored[2], x[i+2]);
            Goldilocks::add_avx512(stored[3], stored[3], x[i+3]);
        }
        for (uint32_t i = 0; i < SPONGE_WIDTH; ++i) {
            Goldilocks::add_avx512(x[i], x[i], stored[i % 4]);
        }
    }
}

template<uint32_t SPONGE_WIDTH_T>
inline void Poseidon2Goldilocks<SPONGE_WIDTH_T>::element_pow7_avx512(__m512i &x) {
    __m512i x2, x3, x4;
    Goldilocks::square_avx512(x2, x);
    Goldilocks::mult_avx512(x3, x, x2);
    Goldilocks::square_avx512(x4, x2);
    Goldilocks::mult_avx512(x, x3, x4);
}

template<uint32_t SPONGE_WIDTH_T>
inline void Poseidon2Goldilocks<SPONGE_WIDTH_T>::pow7add_avx512(__m512i *x, const Goldilocks::Element C_[SPONGE_WIDTH]) {
    __m512i x2[SPONGE_WIDTH], x3[SPONGE_WIDTH], x4[SPONGE_WIDTH];

    __m512i c[SPONGE_WIDTH];
    for (uint32_t i = 0; i < SPONGE_WIDTH; ++i)
    {
        c[i] = _mm512_set1_epi64(C_[i].fe);
        Goldilocks::add_avx512(x[i], x[i], c[i]);
        Goldilocks::square_avx512(x2[i], x[i]);
        Goldilocks::square_avx512(x4[i], x2[i]);
        Goldilocks::mult_avx512(x3[i], x[i], x2[i]);
        Goldilocks::mult_avx512(x[i], x3[i], x4[i]);
    }
}

#endif
#endif