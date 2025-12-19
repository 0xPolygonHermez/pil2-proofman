#ifndef POSEIDON2_GOLDILOCKS_AVX
#define POSEIDON2_GOLDILOCKS_AVX

#include "poseidon2_goldilocks.hpp"
#include "goldilocks_base_field.hpp"
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
    
    for(uint32_t i = 0; i < SPONGE_WIDTH; i +=4) {
        matmul_m4_batch_avx(x[i], x[i+1], x[i+2], x[i+3]);
    }
    if( SPONGE_WIDTH > 4){
        __m256i stored[4];
        Goldilocks::add_avx(stored[0], x[0], x[4]);
        Goldilocks::add_avx(stored[1], x[1], x[5]);
        Goldilocks::add_avx(stored[2], x[2], x[6]);
        Goldilocks::add_avx(stored[3], x[3], x[7]);
        for(uint32_t i = 8; i < SPONGE_WIDTH; i +=4) {
            Goldilocks::add_avx(stored[0], stored[0], x[i]);
            Goldilocks::add_avx(stored[1], stored[1], x[i+1]);
            Goldilocks::add_avx(stored[2], stored[2], x[i+2]);
            Goldilocks::add_avx(stored[3], stored[3], x[i+3]);
        }
        for(uint32_t i = 0; i < SPONGE_WIDTH; ++i)
        {
            Goldilocks::add_avx(x[i], x[i], stored[i % 4]);
        }
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
    for (uint32_t i = 0; i < SPONGE_WIDTH; ++i)
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

    if(SPONGE_WIDTH == 4){

        Goldilocks::Element x[4];
        Goldilocks::store_avx(&(x[0]), st[0]);
        Goldilocks::Element t0 = x[0] + x[1];
        Goldilocks::Element t1 = x[2] + x[3];
        Goldilocks::Element t2 = x[1] + x[1] + t1;
        Goldilocks::Element t3 = x[3] + x[3] + t0;
        Goldilocks::Element t1_2 = t1 + t1;
        Goldilocks::Element t0_2 = t0 + t0;
        Goldilocks::Element t4 = t1_2 + t1_2 + t3;
        Goldilocks::Element t5 = t0_2 + t0_2 + t2;
        Goldilocks::Element t6 = t3 + t5;
        Goldilocks::Element t7 = t2 + t4;
        
        x[0] = t6;
        x[1] = t5;
        x[2] = t7;
        x[3] = t4;
        Goldilocks::load_avx(st[0], &(x[0]));
    } else {

        const __m256i &st0 = st[0];
        const __m256i &st1 = st[1];
        const __m256i &st2 = SPONGE_WIDTH >= 12 ? st[2] : zero;
        const __m256i &st3 = SPONGE_WIDTH == 16 ? st[3] : zero;
        __m256i t0_ = _mm256_permute2f128_si256(st0, st2, 0b00100000);
        __m256i t1_ = _mm256_permute2f128_si256(st1, st3, 0b00100000);
        __m256i t2_ = _mm256_permute2f128_si256(st0, st2, 0b00110001);
        __m256i t3_ = _mm256_permute2f128_si256(st1, st3, 0b00110001);
        __m256i x0 = _mm256_castpd_si256(_mm256_unpacklo_pd(_mm256_castsi256_pd(t0_), _mm256_castsi256_pd(t1_)));
        __m256i x1 = _mm256_castpd_si256(_mm256_unpackhi_pd(_mm256_castsi256_pd(t0_), _mm256_castsi256_pd(t1_)));
        __m256i x2 = _mm256_castpd_si256(_mm256_unpacklo_pd(_mm256_castsi256_pd(t2_), _mm256_castsi256_pd(t3_)));
        __m256i x3 = _mm256_castpd_si256(_mm256_unpackhi_pd(_mm256_castsi256_pd(t2_), _mm256_castsi256_pd(t3_)));
    #
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

        t0_ = _mm256_castpd_si256(_mm256_unpacklo_pd(_mm256_castsi256_pd(t6), _mm256_castsi256_pd(t5)));
        t1_ = _mm256_castpd_si256(_mm256_unpackhi_pd(_mm256_castsi256_pd(t6), _mm256_castsi256_pd(t5)));
        t2_ = _mm256_castpd_si256(_mm256_unpacklo_pd(_mm256_castsi256_pd(t7), _mm256_castsi256_pd(t4)));
        t3_ = _mm256_castpd_si256(_mm256_unpackhi_pd(_mm256_castsi256_pd(t7), _mm256_castsi256_pd(t4)));

        // Step 2: Reverse _mm256_permute2f128_si256
        st[0] = _mm256_permute2f128_si256(t0_, t2_, 0b00100000); // Combine low halves
        st[1] = _mm256_permute2f128_si256(t1_, t3_, 0b00100000); // Combine low halves
        if(SPONGE_WIDTH >= 12) st[2] = _mm256_permute2f128_si256(t0_, t2_, 0b00110001); // Combine high halves
        if(SPONGE_WIDTH == 16) st[3] = _mm256_permute2f128_si256(t1_, t3_, 0b00110001); // Combine high halves

        __m256i stored;   
        Goldilocks::add_avx(stored, st[0], st[1]);
        for(uint32_t i = 2; i < (SPONGE_WIDTH >> 2); i++) {
            Goldilocks::add_avx(stored, stored, st[i]);            
        }
        for(uint32_t i = 0; i < (SPONGE_WIDTH >> 2); i++) {
            Goldilocks::add_avx(st[i], st[i], stored);
        }
    }
    
};

template<uint32_t SPONGE_WIDTH_T>
inline void Poseidon2Goldilocks<SPONGE_WIDTH_T>::pow7_avx(__m256i st[(SPONGE_WIDTH >> 2)])
{
    for(uint32_t i = 0; i < (SPONGE_WIDTH >> 2); i++) {
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
    for(uint32_t i = 0; i < (SPONGE_WIDTH >> 2); i++) {
        __m256i c;
        Goldilocks::load_avx(c, &(C_[i << 2]));
        Goldilocks::add_avx(st[i], st[i], c);
    }
}

template<uint32_t SPONGE_WIDTH_T>
inline void Poseidon2Goldilocks<SPONGE_WIDTH_T>::add_avx_small(__m256i st[(SPONGE_WIDTH >> 2)], const Goldilocks::Element C_small[SPONGE_WIDTH])
{
    for(uint32_t i = 0; i < (SPONGE_WIDTH >> 2); i++) {
        __m256i c;
        Goldilocks::load_avx(c, &(C_small[i << 2]));
        Goldilocks::add_avx_b_small(st[i], st[i], c);
    }
}
#endif
#endif