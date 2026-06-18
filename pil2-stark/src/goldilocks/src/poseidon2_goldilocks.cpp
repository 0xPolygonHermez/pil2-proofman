#include "poseidon2_goldilocks.hpp"
#include <math.h> /* floor */
#include <atomic>
#include <cmath>
#include <stdexcept>


    
template<uint32_t SPONGE_WIDTH_T>
void Poseidon2Goldilocks<SPONGE_WIDTH_T>::permute_seq(Goldilocks::Element (&state)[SPONGE_WIDTH], const Goldilocks::Element (&input)[SPONGE_WIDTH])
{
    const int length = SPONGE_WIDTH * sizeof(Goldilocks::Element);

    std::memcpy(state, input, length);
    const Goldilocks::Element* C = Poseidon2GoldilocksConstants::Poseidon2Tables<SPONGE_WIDTH>::C;
    const Goldilocks::Element* D = Poseidon2GoldilocksConstants::Poseidon2Tables<SPONGE_WIDTH>::D;

    matmul_external_(state);

    for (uint32_t r = 0; r < HALF_N_FULL_ROUNDS; r++)
    {
        pow7add_(state, &(C[r * SPONGE_WIDTH]));
        matmul_external_(state);
    }

    for( uint32_t r = 0; r < N_PARTIAL_ROUNDS; r++)
    {
        state[0] = state[0] + C[HALF_N_FULL_ROUNDS * SPONGE_WIDTH + r];
        pow7(state[0]);
        Goldilocks::Element sum_ = Goldilocks::zero();
        add_(sum_, state);
        prodadd_(state, D, sum_);
    }

    for( uint32_t r = 0; r < HALF_N_FULL_ROUNDS; r++)
    {
        pow7add_(state, &(C[HALF_N_FULL_ROUNDS * SPONGE_WIDTH + N_PARTIAL_ROUNDS + r * SPONGE_WIDTH]));
        matmul_external_(state);
    }

}
template<uint32_t SPONGE_WIDTH_T>
void Poseidon2Goldilocks<SPONGE_WIDTH_T>::linear_hash_seq(Goldilocks::Element *output, Goldilocks::Element *input, uint64_t size)
{
    uint64_t remaining = size;
    Goldilocks::Element state[SPONGE_WIDTH];

    while (remaining)
    {
        if (remaining == size)
        {
            memset(state + RATE, 0, CAPACITY * sizeof(Goldilocks::Element));
        }
        else
        {
// avoid -Wrestrict warning, there is not overlapping in practice            
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wrestrict"
            std::memcpy(state + RATE, state, CAPACITY * sizeof(Goldilocks::Element));
#pragma GCC diagnostic pop
        }

        uint64_t n = (remaining < RATE) ? remaining : RATE;
        memset(&state[n], 0, (RATE - n) * sizeof(Goldilocks::Element));
        std::memcpy(state, input + (size - remaining), n * sizeof(Goldilocks::Element));
        permute_seq(state, state);
        remaining -= n;
    }
    if (size > 0)
    {
        std::memcpy(output, state, CAPACITY * sizeof(Goldilocks::Element));
    }
    else
    {
        memset(output, 0, CAPACITY * sizeof(Goldilocks::Element));
    }
}

template<uint32_t SPONGE_WIDTH_T>
void Poseidon2Goldilocks<SPONGE_WIDTH_T>::merkletreeReduce(Goldilocks::Element *root,Goldilocks::Element *input, uint64_t num_elements, uint64_t arity)
{
    uint64_t numNodes = num_elements;
    uint64_t nodesLevel = num_elements;
    
    while (nodesLevel > 1) {
        uint64_t extraZeros = (arity - (nodesLevel % arity)) % arity;
        numNodes += extraZeros;
        uint64_t nextN = (nodesLevel + (arity - 1))/arity;        
        numNodes += nextN;
        nodesLevel = nextN;
    }

    
    Goldilocks::Element *cursor = new Goldilocks::Element[numNodes * CAPACITY];
    memcpy(cursor, input, num_elements * CAPACITY * sizeof(Goldilocks::Element));

    // Build the merkle tree
    uint64_t pending = num_elements;
    uint64_t nextN = (pending + (arity - 1)) / arity;
    uint64_t nextIndex = 0;

    while (pending > 1)
    {
        uint64_t extraZeros = (arity - (pending % arity)) % arity;
        if (extraZeros > 0) 
        {
            std::memset(&cursor[nextIndex + pending * CAPACITY], 0, extraZeros * CAPACITY * sizeof(Goldilocks::Element));
        }

        for (uint64_t i = 0; i < nextN; i++)
        {
            Goldilocks::Element pol_input[SPONGE_WIDTH];
            memset(pol_input, 0, SPONGE_WIDTH * sizeof(Goldilocks::Element));

            std::memcpy(pol_input, &cursor[nextIndex + i * SPONGE_WIDTH], SPONGE_WIDTH * sizeof(Goldilocks::Element));

            permuteTrunc_seq((Goldilocks::Element(&)[CAPACITY])cursor[nextIndex + (pending + extraZeros + i) * CAPACITY], pol_input);
        }

        nextIndex += (pending + extraZeros) * CAPACITY;
        pending = (pending + (arity - 1)) / arity;
        nextN = (pending + (arity - 1)) / arity;
    }

    std::memcpy(root, &cursor[nextIndex], CAPACITY * sizeof(Goldilocks::Element));
    delete[] cursor;
}

template<uint32_t SPONGE_WIDTH_T>
void Poseidon2Goldilocks<SPONGE_WIDTH_T>::merkletree_seq(Goldilocks::Element *tree, Goldilocks::Element *input, uint64_t num_cols, uint64_t num_rows, uint64_t arity, int num_threads, uint64_t dim)
{
    if (num_rows == 0)
    {
        return;
    }

    Goldilocks::Element *cursor = tree;
    if (num_threads == 0)
        num_threads = omp_get_max_threads();

#pragma omp parallel for num_threads(num_threads)
    for (uint64_t i = 0; i < num_rows; i++)
    {
        linear_hash_seq(&cursor[i * CAPACITY], &input[i * num_cols * dim], num_cols * dim);
    }

    // Build the merkle tree
    uint64_t pending = num_rows;
    uint64_t nextN = (pending + (arity - 1)) / arity;
    uint64_t nextIndex = 0;

    while (pending > 1)
    {
        uint64_t extraZeros = (arity - (pending % arity)) % arity;
        if (extraZeros > 0) 
        {
            std::memset(&cursor[nextIndex + pending * CAPACITY], 0, extraZeros * CAPACITY * sizeof(Goldilocks::Element));
        }

    #pragma omp parallel for num_threads(num_threads)
        for (uint64_t i = 0; i < nextN; i++)
        {
            Goldilocks::Element pol_input[SPONGE_WIDTH];
            memset(pol_input, 0, SPONGE_WIDTH * sizeof(Goldilocks::Element));

            std::memcpy(pol_input, &cursor[nextIndex + i * SPONGE_WIDTH], SPONGE_WIDTH * sizeof(Goldilocks::Element));

            permuteTrunc_seq((Goldilocks::Element(&)[CAPACITY])cursor[nextIndex + (pending + extraZeros + i) * CAPACITY], pol_input);
        }

        nextIndex += (pending + extraZeros) * CAPACITY;
        pending = (pending + (arity - 1)) / arity;
        nextN = (pending + (arity - 1)) / arity;
    }
}
template<uint32_t SPONGE_WIDTH_T>
void Poseidon2Goldilocks<SPONGE_WIDTH_T>::grinding(uint64_t &nonce, const uint64_t* in, const uint32_t n_bits)
{
    // Total hashes needed for S-bit security at n_bits target difficulty:
    //   N = -S · ln(2) / ln(1 - 2^-n_bits).  Same formula as the GPU path;
    // log1p/ldexp keep precision for small 2^-n_bits.
    constexpr uint64_t security = 128;
    const uint64_t level = uint64_t(1) << (64 - n_bits);
    const double eps   = std::ldexp(1.0, -int(n_bits));
    const double total = -double(security) * std::log(2.0) / std::log1p(-eps);
    const uint64_t N   = (uint64_t)std::ceil(total);

    // Smallest-found-nonce wins 
    std::atomic<uint64_t> found{UINT64_MAX};

    #pragma omp parallel
    {
        const int tid  = omp_get_thread_num();
        const int nthr = omp_get_num_threads();
        constexpr uint64_t POLL_MASK = 15;
        uint64_t local_found = UINT64_MAX;
        uint64_t poll = 0;


        // STARK grinding contract:
        //   state[0..2] = FIELD_EXTENSION challenge (in[0..2])
        //   state[3]    = nonce (set per iteration below)
        //   state[4..W-1] = 0 (zero-padding)
        Goldilocks::Element state[SPONGE_WIDTH] = {};
        Goldilocks::Element out[SPONGE_WIDTH];
        std::memcpy(state, in, 3 * sizeof(Goldilocks::Element));

        for (uint64_t i = tid; i < N; i += nthr)
        {
            if ((poll++ & POLL_MASK) == 0)
                local_found = found.load(std::memory_order_relaxed);
            if (i >= local_found) break;

            state[3] = Goldilocks::fromU64(i);
            permute_seq(out, state);
            if (out[0].fe < level) {
                #pragma omp critical(grinding_update)
                {
                    if (i < found.load(std::memory_order_relaxed))
                        found.store(i, std::memory_order_relaxed);
                }
                break;
            }
        }
    }

    nonce = found.load();
    if (nonce == UINT64_MAX)
    {
        throw std::runtime_error("Poseidon2Goldilocks::grinding: could not find a valid nonce");
    }
}

#ifdef __AVX2__

template<uint32_t SPONGE_WIDTH_T>
void Poseidon2Goldilocks<SPONGE_WIDTH_T>::permute_batch_avx(Goldilocks::Element *state, const Goldilocks::Element *input) {

     const Goldilocks::Element* C = Poseidon2GoldilocksConstants::Poseidon2Tables<SPONGE_WIDTH>::C;
    const Goldilocks::Element* D = Poseidon2GoldilocksConstants::Poseidon2Tables<SPONGE_WIDTH>::D;

    const int length = SPONGE_WIDTH * sizeof(Goldilocks::Element);

    std::memcpy(state, input, 4 * length);
    __m256i st[SPONGE_WIDTH];
    for(uint32_t i = 0; i < SPONGE_WIDTH; i++) {
        Goldilocks::load_avx(st[i], &(state[i]), SPONGE_WIDTH);
    }
    
    matmul_external_batch_avx(st);

    for( uint32_t r = 0; r < HALF_N_FULL_ROUNDS; r++)
    {
        pow7add_avx(st,  &(C[r * SPONGE_WIDTH]));
        matmul_external_batch_avx(st);
    }

    __m256i d[SPONGE_WIDTH];
    for( uint32_t i = 0; i < SPONGE_WIDTH; ++i) {
        d[i] = _mm256_set1_epi64x(D[i].fe);
    }

    for( uint32_t r = 0; r < N_PARTIAL_ROUNDS; r++)
    {
        __m256i c = _mm256_set1_epi64x(C[HALF_N_FULL_ROUNDS * SPONGE_WIDTH + r].fe);
        Goldilocks::add_avx(st[0], st[0], c);
        element_pow7_avx(st[0]);
        __m256i sum = _mm256_set1_epi64x(Goldilocks::zero().fe);
        for( uint32_t i = 0; i < SPONGE_WIDTH; ++i)
        {
            Goldilocks::add_avx(sum, sum, st[i]);
        }
        for( uint32_t i = 0; i < SPONGE_WIDTH; ++i)
        {
            Goldilocks::mult_avx(st[i], st[i], d[i]);
            Goldilocks::add_avx(st[i], st[i], sum);
        }
    }

    for( uint32_t r = 0; r < HALF_N_FULL_ROUNDS; r++)
    {
        pow7add_avx(st, &(C[HALF_N_FULL_ROUNDS * SPONGE_WIDTH + N_PARTIAL_ROUNDS + r * SPONGE_WIDTH]));
        matmul_external_batch_avx(st);
    }

    for(uint32_t i = 0; i < SPONGE_WIDTH; i++) {
        Goldilocks::store_avx(&(state[i]), SPONGE_WIDTH, st[i]);
    }

}

template<uint32_t SPONGE_WIDTH_T>
void Poseidon2Goldilocks<SPONGE_WIDTH_T>::permute_avx(Goldilocks::Element (&state)[SPONGE_WIDTH], const Goldilocks::Element (&input)[SPONGE_WIDTH])
{

     const Goldilocks::Element* C = Poseidon2GoldilocksConstants::Poseidon2Tables<SPONGE_WIDTH>::C;
    const Goldilocks::Element* D = Poseidon2GoldilocksConstants::Poseidon2Tables<SPONGE_WIDTH>::D;

    const int length = SPONGE_WIDTH * sizeof(Goldilocks::Element);

    std::memcpy(state, input, length);
    __m256i st[(SPONGE_WIDTH >> 2)];

    for(uint32_t i = 0; i < (SPONGE_WIDTH >> 2); i++) {
        Goldilocks::load_avx(st[i], &(state[i << 2]));
    }

    matmul_external_avx(st);
    
    for(uint32_t r = 0; r < HALF_N_FULL_ROUNDS; r++)
    {
        add_avx_small(st, &(C[r * SPONGE_WIDTH]));
        pow7_avx(st);
        matmul_external_avx(st);
    }
    
    Goldilocks::store_avx(&(state[0]), st[0]);
    Goldilocks::Element state0 = state[0];
    __m256i D_[(SPONGE_WIDTH >> 2)];
    for( uint32_t i = 0; i < (SPONGE_WIDTH >> 2); ++i) {
        Goldilocks::load_avx(D_[i], &(D[i << 2]));
    }

    __m256i partial_sum_;
    Goldilocks::Element partial_sum[4];
    Goldilocks::Element aux = state0;
    for( uint32_t r = 0; r < N_PARTIAL_ROUNDS; r++)
    {
        if( SPONGE_WIDTH > 4){
            Goldilocks::add_avx(partial_sum_, st[0], st[1]);
            for(uint32_t i = 2; i < (SPONGE_WIDTH >> 2); i++) {
                Goldilocks::add_avx(partial_sum_, partial_sum_, st[i]);            
            }
            Goldilocks::store_avx(partial_sum, partial_sum_);
        }else{
            Goldilocks::store_avx(partial_sum, st[0]);
        }       

        Goldilocks::Element sum = partial_sum[0] + partial_sum[1] + partial_sum[2] + partial_sum[3];
        sum = sum - aux;
        state0 = state0 + C[HALF_N_FULL_ROUNDS * SPONGE_WIDTH + r];
        pow7(state0);
        sum = sum + state0;    
            
        __m256i scalar = _mm256_set1_epi64x(sum.fe);
        for(uint32_t i = 0; i < (SPONGE_WIDTH >> 2); i++) {
            Goldilocks::mult_avx(st[i], st[i], D_[i]);
            Goldilocks::add_avx(st[i], st[i], scalar);
        }
        state0 = state0 * D[0] + sum;
        aux = aux * D[0] + sum;
    }

    Goldilocks::store_avx(&(state[0]), st[0]);
    state[0] = state0;
    Goldilocks::load_avx(st[0], &(state[0]));

    for( uint32_t r = 0; r < HALF_N_FULL_ROUNDS; r++)
    {
        add_avx_small(st, &(C[HALF_N_FULL_ROUNDS * SPONGE_WIDTH + N_PARTIAL_ROUNDS + r * SPONGE_WIDTH]));
        pow7_avx(st);        
        matmul_external_avx(st);
    }
    
    for(uint32_t i = 0; i < (SPONGE_WIDTH >> 2); i++) {
        Goldilocks::store_avx(&(state[i << 2]), st[i]);
    }

}

template<uint32_t SPONGE_WIDTH_T>
void Poseidon2Goldilocks<SPONGE_WIDTH_T>::linear_hash_avx(Goldilocks::Element *output, Goldilocks::Element *input, uint64_t size)
{
    uint64_t remaining = size;
    Goldilocks::Element state[SPONGE_WIDTH];

    while (remaining)
    {
        if (remaining == size)
        {
            memset(state + RATE, 0, CAPACITY * sizeof(Goldilocks::Element));
        }
        else
        {
// avoid -Wrestrict warning, there is not overlapping in practice            
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wrestrict"
            std::memcpy(state + RATE, state, CAPACITY * sizeof(Goldilocks::Element));
#pragma GCC diagnostic pop
        }

        uint64_t n = (remaining < RATE) ? remaining : RATE;
        memset(&state[n], 0, (RATE - n) * sizeof(Goldilocks::Element));
        std::memcpy(state, input + (size - remaining), n * sizeof(Goldilocks::Element));
        permute_avx(state, state);
        remaining -= n;
    }
    if (size > 0)
    {
        std::memcpy(output, state, CAPACITY * sizeof(Goldilocks::Element));
    }
    else
    {
        memset(output, 0, CAPACITY * sizeof(Goldilocks::Element));
    }
}

template<uint32_t SPONGE_WIDTH_T>
void Poseidon2Goldilocks<SPONGE_WIDTH_T>::linear_hash_batch_avx(Goldilocks::Element *output, Goldilocks::Element *input, uint64_t size)
{
    uint64_t remaining = size;
    Goldilocks::Element state[4*SPONGE_WIDTH];

    while (remaining)
    {
        if (remaining == size)
        {
            for(uint64_t i = 0; i < 4; ++i) {
                memset(&state[i*SPONGE_WIDTH + RATE], 0, CAPACITY * sizeof(Goldilocks::Element));
            }
        }
        else
        {
            for(uint64_t i = 0; i < 4; ++i) {
                memmove(&state[i*SPONGE_WIDTH + RATE], &state[i*SPONGE_WIDTH], CAPACITY * sizeof(Goldilocks::Element));
            }
        }

        uint64_t n = (remaining < RATE) ? remaining : RATE;
        for(uint64_t i = 0; i < 4; ++i) {
            memset(&state[i*SPONGE_WIDTH + n], 0, (RATE - n) * sizeof(Goldilocks::Element));
            std::memcpy(&state[i * SPONGE_WIDTH], &input[i*size + (size - remaining)], n * sizeof(Goldilocks::Element));
        }
        permute_batch_avx(state, state);
        remaining -= n;
    }
    if (size > 0)
    {
        for(uint64_t i = 0; i < 4; ++i) {
            std::memcpy(&output[i * CAPACITY], &state[i*SPONGE_WIDTH], CAPACITY * sizeof(Goldilocks::Element));
        }
    }
    else
    {
        memset(output, 0, 4 * CAPACITY * sizeof(Goldilocks::Element));
    }
}

template<uint32_t SPONGE_WIDTH_T>
void Poseidon2Goldilocks<SPONGE_WIDTH_T>::merkletree_avx(Goldilocks::Element *tree, Goldilocks::Element *input, uint64_t num_cols, uint64_t num_rows, uint64_t arity, int num_threads, uint64_t dim)
{
    if (num_rows == 0)
    {
        return;
    }
    Goldilocks::Element *cursor = tree;
    if (num_threads == 0)
        num_threads = omp_get_max_threads();

#pragma omp parallel for num_threads(num_threads)
    for (uint64_t i = 0; i < num_rows; i++)
    {
        linear_hash_avx(&cursor[i * CAPACITY], &input[i * num_cols * dim], num_cols * dim);
    }
    
    // Build the merkle tree
    uint64_t pending = num_rows;
    uint64_t nextN = (pending + (arity - 1)) / arity;
    uint64_t nextIndex = 0;

    while (pending > 1)
    {
        uint64_t extraZeros = (arity - (pending % arity)) % arity;
        if (extraZeros > 0) 
        {
            std::memset(&cursor[nextIndex + pending * CAPACITY], 0, extraZeros * CAPACITY * sizeof(Goldilocks::Element));
        }

    #pragma omp parallel for num_threads(num_threads)
        for (uint64_t i = 0; i < nextN; i++)
        {
            Goldilocks::Element pol_input[SPONGE_WIDTH];
            memset(pol_input, 0, SPONGE_WIDTH * sizeof(Goldilocks::Element));

            std::memcpy(pol_input, &cursor[nextIndex + i * SPONGE_WIDTH], SPONGE_WIDTH * sizeof(Goldilocks::Element));

            permuteTrunc_avx((Goldilocks::Element(&)[CAPACITY])cursor[nextIndex + (pending + extraZeros + i) * CAPACITY], pol_input);
        }

        nextIndex += (pending + extraZeros) * CAPACITY;
        pending = (pending + (arity - 1)) / arity;
        nextN = (pending + (arity - 1)) / arity;
    }
}

template<uint32_t SPONGE_WIDTH_T>
void Poseidon2Goldilocks<SPONGE_WIDTH_T>::merkletree_batch_avx(Goldilocks::Element *tree, Goldilocks::Element *input, uint64_t num_cols, uint64_t num_rows, uint64_t arity, int num_threads, uint64_t dim)
{
    if (num_rows == 0)
    {
        return;
    }
    Goldilocks::Element *cursor = tree;
    if (num_threads == 0)
        num_threads = omp_get_max_threads();

#pragma omp parallel for num_threads(num_threads)
    for (uint64_t i = 0; i < num_rows; i+=4)
    {
        linear_hash_batch_avx(&cursor[i * CAPACITY], &input[i * num_cols * dim], num_cols * dim);
    }
    
    // Build the merkle tree
    uint64_t pending = num_rows;
    uint64_t nextN = (pending + (arity - 1)) / arity;
    uint64_t nextIndex = 0;

    while (pending > 1)
    {
        uint64_t extraZeros = (arity - (pending % arity)) % arity;
        if (extraZeros > 0) 
        {
            std::memset(&cursor[nextIndex + pending * CAPACITY], 0, extraZeros * CAPACITY * sizeof(Goldilocks::Element));
        }

    #pragma omp parallel for num_threads(num_threads)
        for (uint64_t i = 0; i < nextN; i += 4)
        {

            if (nextN - i < 4) {
                Goldilocks::Element pol_input[SPONGE_WIDTH];
                memset(pol_input, 0, SPONGE_WIDTH * sizeof(Goldilocks::Element));
                for(int j = 0; j < int(nextN - i); j++) {
                    std::memcpy(pol_input, &cursor[nextIndex + (i+j) * SPONGE_WIDTH], SPONGE_WIDTH * sizeof(Goldilocks::Element));
                    permuteTrunc_avx((Goldilocks::Element(&)[CAPACITY])cursor[nextIndex + (pending + extraZeros + (i + j)) * CAPACITY], pol_input);
                }
            } else {
                Goldilocks::Element pol_input[4*SPONGE_WIDTH];
                memset(pol_input, 0, 4*SPONGE_WIDTH * sizeof(Goldilocks::Element));
                for( uint32_t j = 0; j < 4; j++)
                {
                    std::memcpy(pol_input + j*SPONGE_WIDTH, &cursor[nextIndex + (i+j) * SPONGE_WIDTH], SPONGE_WIDTH * sizeof(Goldilocks::Element));
                }
                permuteTrunc_batch_avx((Goldilocks::Element(&)[4 * CAPACITY])cursor[nextIndex + (pending + extraZeros + i) * CAPACITY], pol_input);
            }
        }

        nextIndex += (pending + extraZeros) * CAPACITY;
        pending = (pending + (arity - 1)) / arity;
        nextN = (pending + (arity - 1)) / arity;
    }
}
#endif

#ifdef __AVX512__


template<uint32_t SPONGE_WIDTH_T>
void Poseidon2Goldilocks<SPONGE_WIDTH_T>::permute_batch_avx512(Goldilocks::Element *state, const Goldilocks::Element *input) {
    const Goldilocks::Element* C = Poseidon2GoldilocksConstants::Poseidon2Tables<SPONGE_WIDTH>::C;
    const Goldilocks::Element* D = Poseidon2GoldilocksConstants::Poseidon2Tables<SPONGE_WIDTH>::D;

    const int length = SPONGE_WIDTH * sizeof(Goldilocks::Element);

    std::memcpy(state, input, 8 * length);
    __m512i st[SPONGE_WIDTH];
    for(uint32_t i = 0; i < SPONGE_WIDTH; i++) {
        Goldilocks::load_avx512(st[i], &(state[i]), SPONGE_WIDTH);
    }

    matmul_external_batch_avx512(st);

    for( uint32_t r = 0; r < HALF_N_FULL_ROUNDS; r++)
    {
        pow7add_avx512(st,  &(C[r * SPONGE_WIDTH]));
        matmul_external_batch_avx512(st);
    }

    __m512i d[SPONGE_WIDTH];
    for( uint32_t i = 0; i < SPONGE_WIDTH; ++i) {
        d[i] = _mm512_set1_epi64(D[i].fe);
    }

    for( uint32_t r = 0; r < N_PARTIAL_ROUNDS; r++)
    {
        __m512i c = _mm512_set1_epi64(C[HALF_N_FULL_ROUNDS * SPONGE_WIDTH + r].fe);
        Goldilocks::add_avx512(st[0], st[0], c);
        element_pow7_avx512(st[0]);
        __m512i sum = _mm512_set1_epi64(Goldilocks::zero().fe);
        for( uint32_t i = 0; i < SPONGE_WIDTH; ++i)
        {
            Goldilocks::add_avx512(sum, sum, st[i]);
        }
        for( uint32_t i = 0; i < SPONGE_WIDTH; ++i)
        {
            Goldilocks::mult_avx512(st[i], st[i], d[i]);
            Goldilocks::add_avx512(st[i], st[i], sum);
        }
    }

    for( uint32_t r = 0; r < HALF_N_FULL_ROUNDS; r++)
    {
        pow7add_avx512(st, &(C[HALF_N_FULL_ROUNDS * SPONGE_WIDTH + N_PARTIAL_ROUNDS + r * SPONGE_WIDTH]));
        matmul_external_batch_avx512(st);
    }

    for(uint32_t i = 0; i < SPONGE_WIDTH; i++) {
        Goldilocks::store_avx512(&(state[i]), SPONGE_WIDTH, st[i]);
    }

}


template<uint32_t SPONGE_WIDTH_T>
void Poseidon2Goldilocks<SPONGE_WIDTH_T>::linear_hash_batch_avx512(Goldilocks::Element *output, Goldilocks::Element *input, uint64_t size)
{
    uint64_t remaining = size;
    Goldilocks::Element state[8*SPONGE_WIDTH];

    while (remaining)
    {
        if (remaining == size)
        {
            for(uint64_t i = 0; i < 8; ++i) {
                memset(&state[i*SPONGE_WIDTH + RATE], 0, CAPACITY * sizeof(Goldilocks::Element));
            }
        }
        else
        {
            for(uint64_t i = 0; i < 8; ++i) {
                memmove(&state[i*SPONGE_WIDTH + RATE], &state[i*SPONGE_WIDTH], CAPACITY * sizeof(Goldilocks::Element));
            }
        }

        uint64_t n = (remaining < RATE) ? remaining : RATE;
        for(uint64_t i = 0; i < 8; ++i) {
            memset(&state[i*SPONGE_WIDTH + n], 0, (RATE - n) * sizeof(Goldilocks::Element));
            std::memcpy(&state[i * SPONGE_WIDTH], &input[i*size + (size - remaining)], n * sizeof(Goldilocks::Element));
        }
        permute_batch_avx512(state, state);
        remaining -= n;
    }
    if (size > 0)
    {
        for(uint64_t i = 0; i < 8; ++i) {
            std::memcpy(&output[i * CAPACITY], &state[i*SPONGE_WIDTH], CAPACITY * sizeof(Goldilocks::Element));
        }
    }
    else
    {
        memset(output, 0, 8 * CAPACITY * sizeof(Goldilocks::Element));
    }
}

template<uint32_t SPONGE_WIDTH_T>
void Poseidon2Goldilocks<SPONGE_WIDTH_T>::merkletree_batch_avx512(Goldilocks::Element *tree, Goldilocks::Element *input, uint64_t num_cols, uint64_t num_rows, uint64_t arity, int num_threads, uint64_t dim)
{
    if (num_rows == 0)
    {
        return;
    }
    Goldilocks::Element *cursor = tree;
    if (num_threads == 0)
        num_threads = omp_get_max_threads();

#pragma omp parallel for num_threads(num_threads)
    for (uint64_t i = 0; i < num_rows; i+=8)
    {
        linear_hash_batch_avx512(&cursor[i * CAPACITY], &input[i * num_cols * dim], num_cols * dim);
    }
    
    // Build the merkle tree
    uint64_t pending = num_rows;
    uint64_t nextN = (pending + (arity - 1)) / arity;
    uint64_t nextIndex = 0;

    while (pending > 1)
    {
        uint64_t extraZeros = (arity - (pending % arity)) % arity;
        if (extraZeros > 0) 
        {
            std::memset(&cursor[nextIndex + pending * CAPACITY], 0, extraZeros * CAPACITY * sizeof(Goldilocks::Element));
        }

    #pragma omp parallel for num_threads(num_threads)
        for (uint64_t i = 0; i < nextN; i += 8)
        {

            if (nextN - i < 8) {
                Goldilocks::Element pol_input[SPONGE_WIDTH];
                memset(pol_input, 0, SPONGE_WIDTH * sizeof(Goldilocks::Element));
                for( uint32_t j = 0; j < uint32_t(nextN - i); j++) {
                    std::memcpy(pol_input, &cursor[nextIndex + (i+j) * SPONGE_WIDTH], SPONGE_WIDTH * sizeof(Goldilocks::Element));
                    permuteTrunc_avx((Goldilocks::Element(&)[CAPACITY])cursor[nextIndex + (pending + extraZeros + (i + j)) * CAPACITY], pol_input);
                }
            } else {
                Goldilocks::Element pol_input[8*SPONGE_WIDTH];
                memset(pol_input, 0, 8*SPONGE_WIDTH * sizeof(Goldilocks::Element));
                for( uint32_t j = 0; j < 8; j++)
                {
                    std::memcpy(pol_input + j*SPONGE_WIDTH, &cursor[nextIndex + (i+j) * SPONGE_WIDTH], SPONGE_WIDTH * sizeof(Goldilocks::Element));
                }
                permuteTrunc_batch_avx512((Goldilocks::Element(&)[8 * CAPACITY])cursor[nextIndex + (pending + extraZeros + i) * CAPACITY], pol_input);
            }
        }

        nextIndex += (pending + extraZeros) * CAPACITY;
        pending = (pending + (arity - 1)) / arity;
        nextN = (pending + (arity - 1)) / arity;
    }
}

#endif

// Explicit template instantiations
// Instantiate both DM=false (tests, benchmarks) and DM=true (production
// default: merkleTree, transcript, grinding) for every width in use.
template class Poseidon2Goldilocks<4>;
template class Poseidon2Goldilocks<8>;
template class Poseidon2Goldilocks<12>;
template class Poseidon2Goldilocks<16>;
