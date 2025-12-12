#ifndef POSEIDON2_GOLDILOCKS
#define POSEIDON2_GOLDILOCKS

#include "poseidon2_goldilocks_constants.hpp"
#include "goldilocks_base_field.hpp"
#define __AVX2__
#ifdef __AVX2__
#include <immintrin.h>
#endif

template<uint32_t SPONGE_WIDTH_T>
class Poseidon2Goldilocks
{
public:
    static_assert(SPONGE_WIDTH_T == 4 || SPONGE_WIDTH_T == 12 || SPONGE_WIDTH_T == 16, "SPONGE_WIDTH_T must be 4, 12, or 16");
    static constexpr uint32_t RATE = SPONGE_WIDTH_T-4;
    static constexpr uint32_t CAPACITY = 4;
    static constexpr uint32_t SPONGE_WIDTH = SPONGE_WIDTH_T;
    static constexpr uint32_t HASH_SIZE = 4;
    static constexpr uint32_t N_FULL_ROUNDS_TOTAL = 8;
    static constexpr uint32_t HALF_N_FULL_ROUNDS = N_FULL_ROUNDS_TOTAL / 2;
    static constexpr uint32_t N_PARTIAL_ROUNDS = SPONGE_WIDTH_T == 4 ? 21 : 22;
    static constexpr uint32_t N_ROUNDS = N_FULL_ROUNDS_TOTAL + N_PARTIAL_ROUNDS;

private:
    inline void static pow7(Goldilocks::Element &x);
    inline void static pow7_(Goldilocks::Element *x);
    inline void static add_(Goldilocks::Element &x, const Goldilocks::Element *st);
    inline void static pow7add_(Goldilocks::Element *x, const Goldilocks::Element C[SPONGE_WIDTH]);
    inline void static prodadd_(Goldilocks::Element *x, const Goldilocks::Element D[SPONGE_WIDTH], const Goldilocks::Element &sum);
    inline void static matmul_m4_(Goldilocks::Element *x);
    inline void static matmul_external_(Goldilocks::Element *x);
#ifdef __AVX2__
    inline void static add_avx(__m256i st[(SPONGE_WIDTH >> 2)], const Goldilocks::Element C[SPONGE_WIDTH]);
    inline void static pow7_avx(__m256i st[(SPONGE_WIDTH >> 2)]);
    inline void static add_avx_small(__m256i st[(SPONGE_WIDTH >> 2)], const Goldilocks::Element C[SPONGE_WIDTH]);
    inline void static matmul_external_avx(__m256i st[(SPONGE_WIDTH >> 2)]);
    inline void static matmul_external_batch_avx(__m256i *x);
    inline void static matmul_m4_batch_avx(__m256i &st0, __m256i &st1, __m256i &st2, __m256i &st3);
    inline void static pow7add_avx(__m256i *x, const Goldilocks::Element C_[SPONGE_WIDTH]);
    inline void static element_pow7_avx(__m256i &x);
#endif
#ifdef __AVX512__
    // inline void static pow7_avx512(__m512i &st0, __m512i &st1, __m512i &st2);
    // inline void static add_avx512(__m512i &st0, __m512i &st1, __m512i &st2, const Goldilocks::Element C[SPONGE_WIDTH]);
    // inline void static add_avx512_a(__m512i &st0, __m512i &st1, __m512i &st2, const Goldilocks::Element C[SPONGE_WIDTH]);
    // inline void static add_avx512_small(__m512i &st0, __m512i &st1, __m512i &st2, const Goldilocks::Element C[SPONGE_WIDTH]);
    inline void static matmul_external_batch_avx512(__m512i *x);
    inline void static matmul_m4_batch_avx512(__m512i &st0, __m512i &st1, __m512i &st2, __m512i &st3);
    inline void static pow7add_avx512(__m512i *x, const Goldilocks::Element C_[SPONGE_WIDTH]);
    inline void static element_pow7_avx512(__m512i &x);
#endif

public:
    
// Non-vectorized:
    void static hash_full_result_seq(Goldilocks::Element *, const Goldilocks::Element *);
    void static linear_hash_seq(Goldilocks::Element *output, Goldilocks::Element *input, uint64_t size);
    void static merkletree_seq(Goldilocks::Element *tree, Goldilocks::Element *input, uint64_t num_cols, uint64_t num_rows, uint64_t arity, int nThreads = 0, uint64_t dim = 1);
    void static hash_seq(Goldilocks::Element (&state)[CAPACITY], const Goldilocks::Element (&input)[SPONGE_WIDTH]);
    void static merkletree_batch_seq(Goldilocks::Element *tree, Goldilocks::Element *input, uint64_t num_cols, uint64_t num_rows, uint64_t arity, uint64_t batch_size, int nThreads = 0, uint64_t dim = 1);
    void static grinding(uint64_t& out_idx, const uint64_t* in, const uint32_t n_bits);


    // Vectorized AVX:
#ifdef __AVX2__
    // Note, the functions that do not have the _avx suffix are the default ones to
    // be used in the prover, they implement avx vectorixation though.
    void static hash_full_result_batch_avx(Goldilocks::Element *, const Goldilocks::Element *);
    void static hash_batch_avx(Goldilocks::Element (&state)[4 * CAPACITY], const Goldilocks::Element (&input)[4 * SPONGE_WIDTH]);
    void static linear_hash_batch_avx(Goldilocks::Element *output, Goldilocks::Element *input, uint64_t size);
    void static merkletree_batch_avx(Goldilocks::Element *tree, Goldilocks::Element *input, uint64_t num_cols, uint64_t num_rows, uint64_t arity, int nThreads = 0, uint64_t dim = 1);
    void static hash_full_result_avx(Goldilocks::Element *, const Goldilocks::Element *);
    void static hash_avx(Goldilocks::Element (&state)[CAPACITY], const Goldilocks::Element (&input)[SPONGE_WIDTH]);
    void static linear_hash_avx(Goldilocks::Element *output, Goldilocks::Element *input, uint64_t size);
    void static merkletree_avx(Goldilocks::Element *tree, Goldilocks::Element *input, uint64_t num_cols, uint64_t num_rows, uint64_t arity, int nThreads = 0, uint64_t dim = 1);
#endif
#ifdef __AVX512__
    // Vectorized AVX512:
    void static hash_full_result_batch_avx512(Goldilocks::Element *, const Goldilocks::Element *);
    void static hash_batch_avx512(Goldilocks::Element (&state)[8 * CAPACITY], const Goldilocks::Element (&input)[8 * SPONGE_WIDTH]);
    void static linear_hash_batch_avx512(Goldilocks::Element *output, Goldilocks::Element *input, uint64_t size);
    void static merkletree_batch_avx512(Goldilocks::Element *tree, Goldilocks::Element *input, uint64_t num_cols, uint64_t num_rows, uint64_t arity, int nThreads = 0, uint64_t dim = 1);
    // void static hash_full_result_avx512(Goldilocks::Element *, const Goldilocks::Element *);
    // void static hash_avx512(Goldilocks::Element (&state)[4 * CAPACITY], const Goldilocks::Element (&input)[4 * SPONGE_WIDTH]);
    // void static linear_hash_avx512(Goldilocks::Element *output, Goldilocks::Element *input, uint64_t size);
    // void static merkletree_avx512(Goldilocks::Element *tree, Goldilocks::Element *input, uint64_t num_cols, uint64_t num_rows, uint64_t arity, int nThreads = 0, uint64_t dim = 1);
#endif

    // Wrapper:
    inline void static merkletree(Goldilocks::Element *tree, Goldilocks::Element *input, uint64_t num_cols, uint64_t num_rows, uint64_t arity,int nThreads = 0, uint64_t dim = 1);
    inline void static merkletree_batch(Goldilocks::Element *tree, Goldilocks::Element *input, uint64_t num_cols, uint64_t num_rows, uint64_t batch_size, int nThreads = 0, uint64_t dim = 1);
};


// WRAPPERS - Template specializations for Poseidon2GoldilocksCommit

template<uint32_t SPONGE_WIDTH_T>
inline void Poseidon2Goldilocks<SPONGE_WIDTH_T>::merkletree(Goldilocks::Element *tree, Goldilocks::Element *input, uint64_t num_cols, uint64_t num_rows, uint64_t arity,int nThreads, uint64_t dim)
{
//#ifdef __AVX512__
    // needs to be tested
    // merkletree_avx512(tree, input, num_cols, num_rows, nThreads, dim);
#if defined(__AVX2__) || defined(__AVX512__)
    merkletree_avx(tree, input, num_cols, num_rows, arity, nThreads, dim);
#else
    merkletree_seq(tree, input, num_cols, num_rows, arity, nThreads, dim);
#endif
    
}
template<uint32_t SPONGE_WIDTH_T>
inline void Poseidon2Goldilocks<SPONGE_WIDTH_T>::merkletree_batch(Goldilocks::Element *tree, Goldilocks::Element *input, uint64_t num_cols, uint64_t num_rows, uint64_t batch_size, int nThreads, uint64_t dim)
{
//#ifdef __AVX512__
    // needs to be tested
    //merkletree_batch_avx512(tree, input, num_cols, num_rows, batch_size, nThreads, dim);
#if defined(__AVX2__) || defined(__AVX512__)
    merkletree_batch_avx(tree, input, num_cols, num_rows, batch_size, nThreads, dim);
#else
    merkletree_batch_seq(tree, input, num_cols, num_rows, batch_size, nThreads, dim);
#endif
}

template<uint32_t SPONGE_WIDTH_T>
inline void Poseidon2Goldilocks<SPONGE_WIDTH_T>::pow7(Goldilocks::Element &x)
{
    Goldilocks::Element x2 = x * x;
    Goldilocks::Element x3 = x * x2;
    Goldilocks::Element x4 = x2 * x2;
    x = x3 * x4;
};

template<uint32_t SPONGE_WIDTH_T>
inline void Poseidon2Goldilocks<SPONGE_WIDTH_T>::pow7_(Goldilocks::Element *x)
{
    Goldilocks::Element x2[SPONGE_WIDTH], x3[SPONGE_WIDTH], x4[SPONGE_WIDTH];
    for (uint32_t i = 0; i < SPONGE_WIDTH; ++i)
    {
        x2[i] = x[i] * x[i];
        x3[i] = x[i] * x2[i];
        x4[i] = x2[i] * x2[i];
        x[i] = x3[i] * x4[i];
    }
};

    
template<uint32_t SPONGE_WIDTH_T>
inline void Poseidon2Goldilocks<SPONGE_WIDTH_T>::add_(Goldilocks::Element &x, const Goldilocks::Element *st)
{
    for (uint32_t i = 0; i < SPONGE_WIDTH; ++i)
    {
        x = x + st[i];
    }
}
template<uint32_t SPONGE_WIDTH_T>
inline void Poseidon2Goldilocks<SPONGE_WIDTH_T>::prodadd_(Goldilocks::Element *x, const Goldilocks::Element D[SPONGE_WIDTH], const Goldilocks::Element &sum)
{
    for (uint32_t i = 0; i < SPONGE_WIDTH; ++i)
    {
        x[i] = x[i]*D[i] + sum;
    }
}

template<uint32_t SPONGE_WIDTH_T>
inline void Poseidon2Goldilocks<SPONGE_WIDTH_T>::pow7add_(Goldilocks::Element *x, const Goldilocks::Element C[SPONGE_WIDTH])
{
    Goldilocks::Element x2[SPONGE_WIDTH], x3[SPONGE_WIDTH], x4[SPONGE_WIDTH];
    
    for (uint32_t i = 0; i < SPONGE_WIDTH; ++i)
    {
        Goldilocks::Element xi = x[i] + C[i];
        x2[i] = xi * xi;
        x3[i] = xi * x2[i];
        x4[i] = x2[i] * x2[i];
        x[i] = x3[i] * x4[i];
    }
};

    
template<uint32_t SPONGE_WIDTH_T>
inline void Poseidon2Goldilocks<SPONGE_WIDTH_T>::matmul_m4_(Goldilocks::Element *x) {
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
}

template<uint32_t SPONGE_WIDTH_T>
inline void Poseidon2Goldilocks<SPONGE_WIDTH_T>::matmul_external_(Goldilocks::Element *x) {
    
    for(uint32_t i = 0; i < SPONGE_WIDTH; i +=4) {
        matmul_m4_(&x[i]);
    }
    if(SPONGE_WIDTH > 4){
        Goldilocks::Element stored[4] = {Goldilocks::zero(), Goldilocks::zero(), Goldilocks::zero(), Goldilocks::zero()};
        for (uint32_t i = 0; i < SPONGE_WIDTH; i+=4) {
            stored[0] = stored[0] + x[i];
            stored[1] = stored[1] + x[i+1];
            stored[2] = stored[2] + x[i+2];
            stored[3] = stored[3] + x[i+3];
        }
        
        for (uint32_t i = 0; i < SPONGE_WIDTH; ++i)
        {
            x[i] = x[i] + stored[i % 4];
        }
    }
}

template<uint32_t SPONGE_WIDTH_T>
inline void Poseidon2Goldilocks<SPONGE_WIDTH_T>::hash_seq(Goldilocks::Element (&state)[CAPACITY], Goldilocks::Element const (&input)[SPONGE_WIDTH])
{
    Goldilocks::Element aux[SPONGE_WIDTH];
    hash_full_result_seq(aux, input);
    std::memcpy(state, aux, CAPACITY * sizeof(Goldilocks::Element));
}

#include "poseidon2_goldilocks_avx.hpp"

#ifdef __AVX512__
 #include "poseidon2_goldilocks_avx512.hpp"
 #endif
#endif

using Poseidon2GoldilocksCommit = Poseidon2Goldilocks<16>;   // SPONGE_WIDTH = 16
using Poseidon2GoldilocksGrinding = Poseidon2Goldilocks<4>;  // SPONGE_WIDTH = 4