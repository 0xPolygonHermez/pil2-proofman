#ifndef POSEIDON2_GOLDILOCKS
#define POSEIDON2_GOLDILOCKS

#include "poseidon2_goldilocks_constants.hpp"
#include "goldilocks_base_field.hpp"
#include <immintrin.h>

#define RATE 8
#define CAPACITY 4
#define HASH_SIZE 4
#define SPONGE_WIDTH (RATE + CAPACITY)
#define HALF_N_FULL_ROUNDS 4
#define N_FULL_ROUNDS_TOTAL (2 * HALF_N_FULL_ROUNDS)
#define N_PARTIAL_ROUNDS 22
#define N_ROUNDS (N_FULL_ROUNDS_TOTAL + N_PARTIAL_ROUNDS)

class Poseidon2Goldilocks
{
private:
    inline void static pow7(Goldilocks::Element &x);
    inline void static pow7_(Goldilocks::Element *x);
    inline void static add_(Goldilocks::Element &x, const Goldilocks::Element *st);
    inline void static pow7add_(Goldilocks::Element *x, const Goldilocks::Element C[SPONGE_WIDTH]);
    inline void static prodadd_(Goldilocks::Element *x, const Goldilocks::Element D[SPONGE_WIDTH], const Goldilocks::Element &sum);
    inline void static matmul_m4_(Goldilocks::Element *x);
    inline void static matmul_external_(Goldilocks::Element *x);

    inline void static add_avx(__m256i &st0, __m256i &st1, __m256i &st2, const Goldilocks::Element C[SPONGE_WIDTH]);
    inline void static pow7_avx(__m256i &st0, __m256i &st1, __m256i &st2);
    inline void static add_avx_a(__m256i &st0, __m256i &st1, __m256i &st2, const Goldilocks::Element C[SPONGE_WIDTH]);
    inline void static add_avx_small(__m256i &st0, __m256i &st1, __m256i &st2, const Goldilocks::Element C[SPONGE_WIDTH]);
    inline void static matmul_external_avx(__m256i &st0, __m256i &st1, __m256i &st2);
// #ifdef __AVX512__
//     inline void static pow7_avx512(__m512i &st0, __m512i &st1, __m512i &st2);
//     inline void static add_avx512(__m512i &st0, __m512i &st1, __m512i &st2, const Goldilocks::Element C[SPONGE_WIDTH]);
//     inline void static add_avx512_a(__m512i &st0, __m512i &st1, __m512i &st2, const Goldilocks::Element C[SPONGE_WIDTH]);
//     inline void static add_avx512_small(__m512i &st0, __m512i &st1, __m512i &st2, const Goldilocks::Element C[SPONGE_WIDTH]);
// #endif

public:
    // Wrapper:
    void static merkletree(Goldilocks::Element *tree, Goldilocks::Element *input, uint64_t num_cols, uint64_t num_rows, int nThreads = 0, uint64_t dim = 1);
    void static merkletree_batch(Goldilocks::Element *tree, Goldilocks::Element *input, uint64_t num_cols, uint64_t num_rows, uint64_t batch_size, int nThreads = 0, uint64_t dim = 1);

    // Non-vectorized:
    void static hash_full_result_seq(Goldilocks::Element *, const Goldilocks::Element *);
    void static linear_hash_seq(Goldilocks::Element *output, Goldilocks::Element *input, uint64_t size);
    void static merkletree_seq(Goldilocks::Element *tree, Goldilocks::Element *input, uint64_t num_cols, uint64_t num_rows, uint64_t arity, int nThreads = 0, uint64_t dim = 1);
    void static hash_seq(Goldilocks::Element (&state)[CAPACITY], const Goldilocks::Element (&input)[SPONGE_WIDTH]);
    void static merkletree_batch_seq(Goldilocks::Element *tree, Goldilocks::Element *input, uint64_t num_cols, uint64_t num_rows, uint64_t arity, uint64_t batch_size, int nThreads = 0, uint64_t dim = 1);

    // Vectorized AVX:
    // Note, the functions that do not have the _avx suffix are the default ones to
    // be used in the prover, they implement avx vectorixation though.
    void static hash_full_result(Goldilocks::Element *, const Goldilocks::Element *);
    void static hash(Goldilocks::Element (&state)[CAPACITY], const Goldilocks::Element (&input)[SPONGE_WIDTH]);
    void static linear_hash(Goldilocks::Element *output, Goldilocks::Element *input, uint64_t size);
    void static merkletree_avx(Goldilocks::Element *tree, Goldilocks::Element *input, uint64_t num_cols, uint64_t num_rows, uint64_t arity, int nThreads = 0, uint64_t dim = 1);
    void static merkletree_batch_avx(Goldilocks::Element *tree, Goldilocks::Element *input, uint64_t num_cols, uint64_t num_rows, uint64_t arity, uint64_t batch_size, int nThreads = 0, uint64_t dim = 1);

// #ifdef __AVX512__
//     // Vectorized AVX512:
//     void static hash_full_result_avx512(Goldilocks::Element *, const Goldilocks::Element *);
//     void static hash_avx512(Goldilocks::Element (&state)[2 * CAPACITY], const Goldilocks::Element (&input)[2 * SPONGE_WIDTH]);
//     void static linear_hash_avx512(Goldilocks::Element *output, Goldilocks::Element *input, uint64_t size);
//     void static merkletree_avx512(Goldilocks::Element *tree, Goldilocks::Element *input, uint64_t num_cols, uint64_t num_rows, int nThreads = 0, uint64_t dim = 1);
//     void static merkletree_batch_avx512(Goldilocks::Element *tree, Goldilocks::Element *input, uint64_t num_cols, uint64_t num_rows, uint64_t batch_size, int nThreads = 0, uint64_t dim = 1);
// #endif

#ifdef __USE_CUDA__
    void static merkletree_cuda(Goldilocks::Element *tree, Goldilocks::Element *input, uint64_t num_cols, uint64_t num_rows, int nThreads = 0, uint64_t dim = 1);
    void static merkletree_cuda_gpudata(Goldilocks::Element *tree, uint64_t *gpu_input, uint64_t num_cols, uint64_t num_rows, int nThreads = 0, uint64_t dim = 1);
    void static merkletree_cuda_gpudata_inplace(uint64_t **d_tree, uint64_t *d_input, uint64_t num_cols, uint64_t num_rows, int nThreads = 0, uint64_t dim = 1);
    void static merkletree_cuda_streams(uint64_t **d_tree, uint64_t *d_input, uint64_t num_cols, uint64_t num_rows, int nThreads = 0, uint64_t dim = 1);
    void static partial_hash_init_gpu(uint64_t **state, uint32_t num_rows, uint32_t ngpus);
    void static partial_hash_gpu(uint64_t *input, uint32_t num_cols, uint32_t num_rows, uint64_t *state);
    void static merkletree_cuda_multi_gpu_full(Goldilocks::Element *tree, uint64_t** gpu_inputs, uint64_t** gpu_trees, void* gpu_streams, uint64_t num_cols, uint64_t num_rows, uint64_t num_rows_device, uint32_t const ngpu, uint64_t dim = 1);
    void static merkletree_cuda_multi_gpu_steps(uint64_t** gpu_inputs, uint64_t** gpu_trees, void* v_gpu_streams, uint64_t num_cols, uint64_t num_rows_device, uint32_t const ngpu, uint64_t dim = 1);
    void static merkletree_cuda_multi_gpu_final(Goldilocks::Element *tree, uint64_t* final_tree, void* v_gpu_streams, uint64_t num_rows);

    void static merkletree_cuda_async(Goldilocks::Element *tree, Goldilocks::Element *input, uint64_t num_cols, uint64_t num_rows);

    void static merkletree_cuda_coalesced(uint64_t **d_tree, uint64_t *d_input, uint64_t num_cols, uint64_t num_rows, int nThreads = 0, uint64_t dim = 1);

#endif
};

// WRAPPERS

inline void Poseidon2Goldilocks::merkletree(Goldilocks::Element *tree, Goldilocks::Element *input, uint64_t num_cols, uint64_t num_rows, int nThreads, uint64_t dim)
{
// #ifdef __AVX512__
//     merkletree_avx512(tree, input, num_cols, num_rows, nThreads, dim);
// #else
    merkletree_avx(tree, input, num_cols, num_rows, nThreads, dim);
// #endif
}
inline void Poseidon2Goldilocks::merkletree_batch(Goldilocks::Element *tree, Goldilocks::Element *input, uint64_t num_cols, uint64_t num_rows, uint64_t batch_size, int nThreads, uint64_t dim)
{
// #ifdef __AVX512__
//     merkletree_batch_avx512(tree, input, num_cols, num_rows, batch_size, nThreads, dim);
// #else
    merkletree_batch_avx(tree, input, num_cols, num_rows, batch_size, nThreads, dim);
// #endif
}

inline void Poseidon2Goldilocks::pow7(Goldilocks::Element &x)
{
    Goldilocks::Element x2 = x * x;
    Goldilocks::Element x3 = x * x2;
    Goldilocks::Element x4 = x2 * x2;
    x = x3 * x4;
};
inline void Poseidon2Goldilocks::pow7_(Goldilocks::Element *x)
{
    Goldilocks::Element x2[SPONGE_WIDTH], x3[SPONGE_WIDTH], x4[SPONGE_WIDTH];
    for (int i = 0; i < SPONGE_WIDTH; ++i)
    {
        x2[i] = x[i] * x[i];
        x3[i] = x[i] * x2[i];
        x4[i] = x2[i] * x2[i];
        x[i] = x3[i] * x4[i];
    }
};

inline void Poseidon2Goldilocks::add_(Goldilocks::Element &x, const Goldilocks::Element *st)
{
    for (int i = 0; i < SPONGE_WIDTH; ++i)
    {
        x = x + st[i];
    }
}
inline void Poseidon2Goldilocks::prodadd_(Goldilocks::Element *x, const Goldilocks::Element D[SPONGE_WIDTH], const Goldilocks::Element &sum)
{
    for (int i = 0; i < SPONGE_WIDTH; ++i)
    {
        x[i] = x[i]*D[i] + sum;
    }
}

inline void Poseidon2Goldilocks::pow7add_(Goldilocks::Element *x, const Goldilocks::Element C[SPONGE_WIDTH])
{
    Goldilocks::Element x2[SPONGE_WIDTH], x3[SPONGE_WIDTH], x4[SPONGE_WIDTH];
    
    for (int i = 0; i < SPONGE_WIDTH; ++i)
    {
        Goldilocks::Element xi = x[i] + C[i];
        x2[i] = xi * xi;
        x3[i] = xi * x2[i];
        x4[i] = x2[i] * x2[i];
        x[i] = x3[i] * x4[i];
    }
};

inline void Poseidon2Goldilocks::matmul_m4_(Goldilocks::Element *x) {
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

inline void Poseidon2Goldilocks::matmul_external_(Goldilocks::Element *x) {
    matmul_m4_(&x[0]);
    matmul_m4_(&x[4]);
    matmul_m4_(&x[8]);
    
    Goldilocks::Element stored[4] = {
        x[0] + x[4] + x[8],
        x[1] + x[5] + x[9],
        x[2] + x[6] + x[10],
        x[3] + x[7] + x[11],
    };
    
    for (int i = 0; i < SPONGE_WIDTH; ++i)
    {
        x[i] = x[i] + stored[i % 4];
    }
}

inline void Poseidon2Goldilocks::hash_seq(Goldilocks::Element (&state)[CAPACITY], Goldilocks::Element const (&input)[SPONGE_WIDTH])
{
    Goldilocks::Element aux[SPONGE_WIDTH];
    hash_full_result_seq(aux, input);
    std::memcpy(state, aux, CAPACITY * sizeof(Goldilocks::Element));
}

#include "poseidon2_goldilocks_avx.hpp"

// #ifdef __AVX512__
// #include "poseidon2_goldilocks_avx512.hpp"
// #endif
#endif