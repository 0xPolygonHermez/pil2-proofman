#include "poseidon2_goldilocks.hpp"
#include <math.h> /* floor */


    
template<uint32_t SPONGE_WIDTH_T>
void Poseidon2Goldilocks<SPONGE_WIDTH_T>::permute_seq(Goldilocks::Element (&state)[SPONGE_WIDTH], const Goldilocks::Element (&input)[SPONGE_WIDTH])
{
    const int length = SPONGE_WIDTH * sizeof(Goldilocks::Element);
    std::memcpy(state, input, length);
    const Goldilocks::Element* C = SPONGE_WIDTH == 4 ? Poseidon2GoldilocksConstants::C4 : SPONGE_WIDTH == 8 ? Poseidon2GoldilocksConstants::C8 : SPONGE_WIDTH == 12 ? Poseidon2GoldilocksConstants::C12 : Poseidon2GoldilocksConstants::C16;
    const Goldilocks::Element* D = SPONGE_WIDTH == 4 ? Poseidon2GoldilocksConstants::D4 : SPONGE_WIDTH == 8 ? Poseidon2GoldilocksConstants::D8 : SPONGE_WIDTH == 12 ? Poseidon2GoldilocksConstants::D12 : Poseidon2GoldilocksConstants::D16;

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

            compress_seq((Goldilocks::Element(&)[CAPACITY])cursor[nextIndex + (pending + extraZeros + i) * CAPACITY], pol_input);
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

            compress_seq((Goldilocks::Element(&)[CAPACITY])cursor[nextIndex + (pending + extraZeros + i) * CAPACITY], pol_input);
        }

        nextIndex += (pending + extraZeros) * CAPACITY;
        pending = (pending + (arity - 1)) / arity;
        nextN = (pending + (arity - 1)) / arity;
    }
}
template<uint32_t SPONGE_WIDTH_T>
void Poseidon2Goldilocks<SPONGE_WIDTH_T>::grinding(uint64_t &nonce, const uint64_t* in, const uint32_t n_bits)
{
    uint64_t checkChunk = omp_get_max_threads() * 512;
    uint64_t level   = uint64_t(1) << (64 - n_bits);
    uint64_t* chunkIdxs = new uint64_t[omp_get_max_threads()];
    uint64_t offset = 0;
    nonce = UINT64_MAX;

    for(int i = 0; i < omp_get_max_threads(); ++i)
    {
        chunkIdxs[i] = UINT64_MAX;
    }

    //we are trying (1 << n_bits) * 512 * num_threads possibilities maximum
    for(int k = 0; k < (1 << n_bits); ++k)
    {

#if PIL2_HAS_NEON
        // 2-nonce-batched NEON path: each iter runs permute_batch_neon over
        // two consecutive nonces (2 sponges per NEON lane). Grinding's
        // existing semantic is "any matching nonce wins" (selection across
        // threads is via first-thread-order), so batching two nonces per
        // iter keeps correctness — if both lanes match, prefer the lower.
        // checkChunk = max_threads * 512 is always even, so the step-by-2
        // loop hits every nonce exactly once.
        #pragma omp parallel for
        for (uint64_t i = 0; i < checkChunk; i += 2) {
            if (chunkIdxs[omp_get_thread_num()] != UINT64_MAX)
                continue;

            Goldilocks::Element buf[2 * SPONGE_WIDTH];
            std::memcpy(&buf[0], in, (SPONGE_WIDTH - 1) * sizeof(Goldilocks::Element));
            buf[SPONGE_WIDTH - 1] = Goldilocks::fromU64(offset + i);
            std::memcpy(&buf[SPONGE_WIDTH], in, (SPONGE_WIDTH - 1) * sizeof(Goldilocks::Element));
            buf[2 * SPONGE_WIDTH - 1] = Goldilocks::fromU64(offset + i + 1);

            permute_batch_neon(buf, buf);

            if (buf[0].fe < level) {
                chunkIdxs[omp_get_thread_num()] = offset + i;
            } else if (buf[SPONGE_WIDTH].fe < level) {
                chunkIdxs[omp_get_thread_num()] = offset + i + 1;
            }
        }
#else
        #pragma omp parallel for
        for (uint64_t i = 0; i < checkChunk; i++) {
            if (chunkIdxs[omp_get_thread_num()] != UINT64_MAX)
                continue;

            Goldilocks::Element state[SPONGE_WIDTH];
            std::memcpy(state, in, (SPONGE_WIDTH - 1) * sizeof(Goldilocks::Element));
            state[SPONGE_WIDTH - 1] = Goldilocks::fromU64(offset + i);
            permute_seq(state, state);
            if (state[0].fe < level) {
                chunkIdxs[omp_get_thread_num()] = offset + i;
            }
        }
#endif

        for(int i = 0; i < omp_get_max_threads(); ++i)
        {
            if (chunkIdxs[i] != UINT64_MAX)
            {
                nonce = chunkIdxs[i];
                break;
            }
        }

        if (nonce != UINT64_MAX)
            break;

        offset += checkChunk;
    }
    if(nonce == UINT64_MAX)
    {
        throw std::runtime_error("Poseidon2Goldilocks::grinding: could not find a valid nonce");
    }
    delete[] chunkIdxs;
}

#ifdef __AVX2__

template<uint32_t SPONGE_WIDTH_T>
void Poseidon2Goldilocks<SPONGE_WIDTH_T>::permute_batch_avx(Goldilocks::Element *state, const Goldilocks::Element *input) {

     const Goldilocks::Element* C = SPONGE_WIDTH == 4 ? Poseidon2GoldilocksConstants::C4 : SPONGE_WIDTH == 8 ? Poseidon2GoldilocksConstants::C8 : SPONGE_WIDTH == 12 ? Poseidon2GoldilocksConstants::C12 : Poseidon2GoldilocksConstants::C16;
    const Goldilocks::Element* D = SPONGE_WIDTH == 4 ? Poseidon2GoldilocksConstants::D4 : SPONGE_WIDTH == 8 ? Poseidon2GoldilocksConstants::D8 : SPONGE_WIDTH == 12 ? Poseidon2GoldilocksConstants::D12 : Poseidon2GoldilocksConstants::D16;

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

     const Goldilocks::Element* C = SPONGE_WIDTH == 4 ? Poseidon2GoldilocksConstants::C4 : SPONGE_WIDTH == 8 ? Poseidon2GoldilocksConstants::C8 : SPONGE_WIDTH == 12 ? Poseidon2GoldilocksConstants::C12 : Poseidon2GoldilocksConstants::C16;
    const Goldilocks::Element* D = SPONGE_WIDTH == 4 ? Poseidon2GoldilocksConstants::D4 : SPONGE_WIDTH == 8 ? Poseidon2GoldilocksConstants::D8 : SPONGE_WIDTH == 12 ? Poseidon2GoldilocksConstants::D12 : Poseidon2GoldilocksConstants::D16;
 
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

            compress_avx((Goldilocks::Element(&)[CAPACITY])cursor[nextIndex + (pending + extraZeros + i) * CAPACITY], pol_input);
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
                    compress_avx((Goldilocks::Element(&)[CAPACITY])cursor[nextIndex + (pending + extraZeros + (i + j)) * CAPACITY], pol_input);
                }
            } else {
                Goldilocks::Element pol_input[4*SPONGE_WIDTH];
                memset(pol_input, 0, 4*SPONGE_WIDTH * sizeof(Goldilocks::Element));
                for( uint32_t j = 0; j < 4; j++)
                {
                    std::memcpy(pol_input + j*SPONGE_WIDTH, &cursor[nextIndex + (i+j) * SPONGE_WIDTH], SPONGE_WIDTH * sizeof(Goldilocks::Element));
                }
                compress_batch_avx((Goldilocks::Element(&)[4 * CAPACITY])cursor[nextIndex + (pending + extraZeros + i) * CAPACITY], pol_input);
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
    const Goldilocks::Element* C = SPONGE_WIDTH == 4 ? Poseidon2GoldilocksConstants::C4 : SPONGE_WIDTH == 8 ? Poseidon2GoldilocksConstants::C8 : SPONGE_WIDTH == 12 ? Poseidon2GoldilocksConstants::C12 : Poseidon2GoldilocksConstants::C16;
    const Goldilocks::Element* D = SPONGE_WIDTH == 4 ? Poseidon2GoldilocksConstants::D4 : SPONGE_WIDTH == 8 ? Poseidon2GoldilocksConstants::D8 : SPONGE_WIDTH == 12 ? Poseidon2GoldilocksConstants::D12 : Poseidon2GoldilocksConstants::D16;

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
                    compress_avx((Goldilocks::Element(&)[CAPACITY])cursor[nextIndex + (pending + extraZeros + (i + j)) * CAPACITY], pol_input);
                }
            } else {
                Goldilocks::Element pol_input[8*SPONGE_WIDTH];
                memset(pol_input, 0, 8*SPONGE_WIDTH * sizeof(Goldilocks::Element));
                for( uint32_t j = 0; j < 8; j++)
                {
                    std::memcpy(pol_input + j*SPONGE_WIDTH, &cursor[nextIndex + (i+j) * SPONGE_WIDTH], SPONGE_WIDTH * sizeof(Goldilocks::Element));
                }
                compress_batch_avx512((Goldilocks::Element(&)[8 * CAPACITY])cursor[nextIndex + (pending + extraZeros + i) * CAPACITY], pol_input);
            }
        }

        nextIndex += (pending + extraZeros) * CAPACITY;
        pending = (pending + (arity - 1)) / arity;
        nextN = (pending + (arity - 1)) / arity;
    }
}

#endif

// ===========================================================================
// NEON single-sponge linear_hash + merkletree (Part 5)
// Mirror linear_hash_seq / merkletree_seq exactly, but call permute_neon /
// compress_neon inside so the hot path actually uses the NEON code on Darwin.
// Batched (4-row) variant lands in Tasks 36-37 as a follow-up.
// ===========================================================================

#if PIL2_HAS_NEON
template<uint32_t SPONGE_WIDTH_T>
void Poseidon2Goldilocks<SPONGE_WIDTH_T>::linear_hash_neon(Goldilocks::Element *output, Goldilocks::Element *input, uint64_t size)
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
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wrestrict"
            std::memcpy(state + RATE, state, CAPACITY * sizeof(Goldilocks::Element));
#pragma GCC diagnostic pop
        }

        uint64_t n = (remaining < RATE) ? remaining : RATE;
        memset(&state[n], 0, (RATE - n) * sizeof(Goldilocks::Element));
        std::memcpy(state, input + (size - remaining), n * sizeof(Goldilocks::Element));
        permute_neon(state, state);
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
void Poseidon2Goldilocks<SPONGE_WIDTH_T>::merkletree_neon(Goldilocks::Element *tree, Goldilocks::Element *input, uint64_t num_cols, uint64_t num_rows, uint64_t arity, int num_threads, uint64_t dim)
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
        linear_hash_neon(&cursor[i * CAPACITY], &input[i * num_cols * dim], num_cols * dim);
    }

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
            compress_neon((Goldilocks::Element(&)[CAPACITY])cursor[nextIndex + (pending + extraZeros + i) * CAPACITY], pol_input);
        }

        nextIndex += (pending + extraZeros) * CAPACITY;
        pending = (pending + (arity - 1)) / arity;
        nextN = (pending + (arity - 1)) / arity;
    }
}
// ===========================================================================
// NEON 2-sponge BATCH linear_hash + merkletree (Part 5 Task 36/37).
// Mirrors merkletree_batch_avx / linear_hash_batch_avx exactly, but with
// batch size = 2 (NEON has 2 lanes per uint64x2_t vs AVX2's 4).
// ===========================================================================

template<uint32_t SPONGE_WIDTH_T>
void Poseidon2Goldilocks<SPONGE_WIDTH_T>::linear_hash_batch_neon(Goldilocks::Element *output, Goldilocks::Element *input, uint64_t size)
{
    uint64_t remaining = size;
    Goldilocks::Element state[2 * SPONGE_WIDTH];

    while (remaining)
    {
        if (remaining == size)
        {
            for (uint64_t i = 0; i < 2; ++i)
                memset(&state[i * SPONGE_WIDTH + RATE], 0, CAPACITY * sizeof(Goldilocks::Element));
        }
        else
        {
            for (uint64_t i = 0; i < 2; ++i)
                memmove(&state[i * SPONGE_WIDTH + RATE], &state[i * SPONGE_WIDTH], CAPACITY * sizeof(Goldilocks::Element));
        }

        uint64_t n = (remaining < RATE) ? remaining : RATE;
        for (uint64_t i = 0; i < 2; ++i) {
            memset(&state[i * SPONGE_WIDTH + n], 0, (RATE - n) * sizeof(Goldilocks::Element));
            std::memcpy(&state[i * SPONGE_WIDTH], &input[i * size + (size - remaining)], n * sizeof(Goldilocks::Element));
        }
        permute_batch_neon(state, state);
        remaining -= n;
    }
    if (size > 0)
    {
        for (uint64_t i = 0; i < 2; ++i)
            std::memcpy(&output[i * CAPACITY], &state[i * SPONGE_WIDTH], CAPACITY * sizeof(Goldilocks::Element));
    }
    else
    {
        memset(output, 0, 2 * CAPACITY * sizeof(Goldilocks::Element));
    }
}

template<uint32_t SPONGE_WIDTH_T>
void Poseidon2Goldilocks<SPONGE_WIDTH_T>::merkletree_batch_neon(Goldilocks::Element *tree, Goldilocks::Element *input, uint64_t num_cols, uint64_t num_rows, uint64_t arity, int nThreads, uint64_t dim)
{
    if (num_rows == 0)
        return;

    Goldilocks::Element *cursor = tree;
    if (nThreads == 0)
        nThreads = omp_get_max_threads();

    // Leaf layer: pairs of rows hashed together. Scalar tail for odd num_rows.
#pragma omp parallel for num_threads(nThreads)
    for (uint64_t i = 0; i < num_rows; i += 2)
    {
        if (num_rows - i < 2) {
            linear_hash_neon(&cursor[i * CAPACITY], &input[i * num_cols * dim], num_cols * dim);
        } else {
            linear_hash_batch_neon(&cursor[i * CAPACITY], &input[i * num_cols * dim], num_cols * dim);
        }
    }

    // Internal nodes: same arity-step but in pairs.
    uint64_t pending = num_rows;
    uint64_t nextN = (pending + (arity - 1)) / arity;
    uint64_t nextIndex = 0;

    while (pending > 1)
    {
        uint64_t extraZeros = (arity - (pending % arity)) % arity;
        if (extraZeros > 0)
            std::memset(&cursor[nextIndex + pending * CAPACITY], 0, extraZeros * CAPACITY * sizeof(Goldilocks::Element));

#pragma omp parallel for num_threads(nThreads)
        for (uint64_t i = 0; i < nextN; i += 2)
        {
            if (nextN - i < 2) {
                Goldilocks::Element pol_input[SPONGE_WIDTH];
                memset(pol_input, 0, SPONGE_WIDTH * sizeof(Goldilocks::Element));
                for (int j = 0; j < int(nextN - i); j++) {
                    std::memcpy(pol_input, &cursor[nextIndex + (i + j) * SPONGE_WIDTH], SPONGE_WIDTH * sizeof(Goldilocks::Element));
                    compress_neon((Goldilocks::Element(&)[CAPACITY])cursor[nextIndex + (pending + extraZeros + (i + j)) * CAPACITY], pol_input);
                }
            } else {
                Goldilocks::Element pol_input[2 * SPONGE_WIDTH];
                memset(pol_input, 0, 2 * SPONGE_WIDTH * sizeof(Goldilocks::Element));
                for (uint32_t j = 0; j < 2; j++)
                    std::memcpy(pol_input + j * SPONGE_WIDTH, &cursor[nextIndex + (i + j) * SPONGE_WIDTH], SPONGE_WIDTH * sizeof(Goldilocks::Element));
                compress_batch_neon((Goldilocks::Element(&)[2 * CAPACITY])cursor[nextIndex + (pending + extraZeros + i) * CAPACITY], pol_input);
            }
        }

        nextIndex += (pending + extraZeros) * CAPACITY;
        pending = (pending + (arity - 1)) / arity;
        nextN = (pending + (arity - 1)) / arity;
    }
}
#endif  // PIL2_HAS_NEON

// Explicit template instantiations
template class Poseidon2Goldilocks<4>;
template class Poseidon2Goldilocks<8>;
template class Poseidon2Goldilocks<12>;
template class Poseidon2Goldilocks<16>;
