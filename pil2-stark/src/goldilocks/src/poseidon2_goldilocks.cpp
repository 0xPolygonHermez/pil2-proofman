#include "poseidon2_goldilocks.hpp"
#include <math.h> /* floor */
#include "merklehash_goldilocks.hpp"


    
template<uint32_t SPONGE_WIDTH_T>
void Poseidon2Goldilocks<SPONGE_WIDTH_T>::hash_full_result_seq(Goldilocks::Element *state, const Goldilocks::Element *input)
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

    if (size <= CAPACITY)
    {
        std::memcpy(output, input, size * sizeof(Goldilocks::Element));
        std::memset(&output[size], 0, (CAPACITY - size) * sizeof(Goldilocks::Element));
        return; // no need to hash
    }
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
        hash_full_result_seq(state, state);
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
void Poseidon2Goldilocks<SPONGE_WIDTH_T>::merkletree_seq(Goldilocks::Element *tree, Goldilocks::Element *input, uint64_t num_cols, uint64_t num_rows, uint64_t arity, int nThreads, uint64_t dim)
{
    if (num_rows == 0)
    {
        return;
    }

    Goldilocks::Element *cursor = tree;
    // memset(cursor, 0, num_rows * CAPACITY * sizeof(Goldilocks::Element));
    if (nThreads == 0)
        nThreads = omp_get_max_threads();

#pragma omp parallel for num_threads(nThreads)
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

    #pragma omp parallel for num_threads(nThreads)
        for (uint64_t i = 0; i < nextN; i++)
        {
            Goldilocks::Element pol_input[SPONGE_WIDTH];
            memset(pol_input, 0, SPONGE_WIDTH * sizeof(Goldilocks::Element));

            std::memcpy(pol_input, &cursor[nextIndex + i * SPONGE_WIDTH], SPONGE_WIDTH * sizeof(Goldilocks::Element));

            hash_seq((Goldilocks::Element(&)[CAPACITY])cursor[nextIndex + (pending + extraZeros + i) * CAPACITY], pol_input);
        }

        nextIndex += (pending + extraZeros) * CAPACITY;
        pending = (pending + (arity - 1)) / arity;
        nextN = (pending + (arity - 1)) / arity;
    }
}
template<uint32_t SPONGE_WIDTH_T>
void Poseidon2Goldilocks<SPONGE_WIDTH_T>::merkletree_batch_seq(Goldilocks::Element *tree, Goldilocks::Element *input, uint64_t num_cols, uint64_t num_rows, uint64_t arity, uint64_t batch_size, int nThreads, uint64_t dim)
{
    if (num_rows == 0)
    {
        return;
    }

    Goldilocks::Element *cursor = tree;
    uint64_t nbatches = 1;
    if (num_cols > 0)
    {
        nbatches = (num_cols + batch_size - 1) / batch_size;
    }
    uint64_t nlastb = num_cols - (nbatches - 1) * batch_size;

    if (nThreads == 0)
        nThreads = omp_get_max_threads();
    Goldilocks::Element **buffers = new Goldilocks::Element*[nThreads];
    for( int i = 0; i < nThreads; ++i)
    {
        buffers[i] = new Goldilocks::Element[nbatches * CAPACITY];
    }

#pragma omp parallel for num_threads(nThreads)
    for (uint64_t i = 0; i < num_rows; i++)
    {
        Goldilocks::Element *buff0 = buffers[omp_get_thread_num()];
        for (uint64_t j = 0; j < nbatches; j++)
        {
            uint64_t nn = batch_size;
            if (j == nbatches - 1)
                nn = nlastb;
            linear_hash_seq(&buff0[j * CAPACITY], &input[i * num_cols * dim + j * batch_size * dim], nn * dim);
        }
        linear_hash_seq(&cursor[i * CAPACITY], buff0, nbatches * CAPACITY);
    }
    for(int i = 0; i < nThreads; ++i)
    {
        delete[] buffers[i];
    }
    delete[] buffers;

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

    #pragma omp parallel for num_threads(nThreads)
        for (uint64_t i = 0; i < nextN; i++)
        {
            Goldilocks::Element pol_input[SPONGE_WIDTH];
            memset(pol_input, 0, SPONGE_WIDTH * sizeof(Goldilocks::Element));

            std::memcpy(pol_input, &cursor[nextIndex + i * SPONGE_WIDTH], SPONGE_WIDTH * sizeof(Goldilocks::Element));

            hash_seq((Goldilocks::Element(&)[CAPACITY])cursor[nextIndex + (pending + extraZeros + i) * CAPACITY], pol_input);
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

        #pragma omp parallel for
        for (uint64_t i = 0; i < checkChunk; i++) {
            if (chunkIdxs[omp_get_thread_num()] != UINT64_MAX)
                continue;

            Goldilocks::Element state[SPONGE_WIDTH];
            std::memcpy(state, in, (SPONGE_WIDTH - 1) * sizeof(Goldilocks::Element));
            state[SPONGE_WIDTH - 1] = Goldilocks::fromU64(offset + i);
            hash_full_result_seq(state, state);
            if (state[0].fe < level) {
                chunkIdxs[omp_get_thread_num()] = offset + i;
            }
        }

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
void Poseidon2Goldilocks<SPONGE_WIDTH_T>::hash_full_result_batch_avx(Goldilocks::Element *state, const Goldilocks::Element *input) {

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
void Poseidon2Goldilocks<SPONGE_WIDTH_T>::hash_full_result_avx(Goldilocks::Element *state, const Goldilocks::Element *input)
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

    if (size <= CAPACITY)
    {
        std::memcpy(output, input, size * sizeof(Goldilocks::Element));
        std::memset(&output[size], 0, (CAPACITY - size) * sizeof(Goldilocks::Element));
        return; // no need to hash
    }
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
        hash_full_result_avx(state, state);
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

    if (size <= CAPACITY)
    {
        for(uint64_t i = 0; i < 4; ++i) {
            std::memcpy(&output[i * CAPACITY], &input[i*size], size * sizeof(Goldilocks::Element));
            std::memset(&output[i * CAPACITY + size], 0, (CAPACITY - size) * sizeof(Goldilocks::Element));
        }
        
        return; // no need to hash
    }
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
                memcpy(&state[i*SPONGE_WIDTH + RATE], &state[i*SPONGE_WIDTH], CAPACITY * sizeof(Goldilocks::Element));
            }
        }

        uint64_t n = (remaining < RATE) ? remaining : RATE;
        for(uint64_t i = 0; i < 4; ++i) {
            memset(&state[i*SPONGE_WIDTH + n], 0, (RATE - n) * sizeof(Goldilocks::Element));
            std::memcpy(&state[i * SPONGE_WIDTH], &input[i*size + (size - remaining)], n * sizeof(Goldilocks::Element));
        }
        hash_full_result_batch_avx(state, state);
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
void Poseidon2Goldilocks<SPONGE_WIDTH_T>::merkletree_avx(Goldilocks::Element *tree, Goldilocks::Element *input, uint64_t num_cols, uint64_t num_rows, uint64_t arity, int nThreads, uint64_t dim)
{
    if (num_rows == 0)
    {
        return;
    }
    Goldilocks::Element *cursor = tree;
    // memset(cursor, 0, num_rows * CAPACITY * sizeof(Goldilocks::Element));
    if (nThreads == 0)
        nThreads = omp_get_max_threads();

#pragma omp parallel for num_threads(nThreads)
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

    #pragma omp parallel for num_threads(nThreads)
        for (uint64_t i = 0; i < nextN; i++)
        {
            Goldilocks::Element pol_input[SPONGE_WIDTH];
            memset(pol_input, 0, SPONGE_WIDTH * sizeof(Goldilocks::Element));

            std::memcpy(pol_input, &cursor[nextIndex + i * SPONGE_WIDTH], SPONGE_WIDTH * sizeof(Goldilocks::Element));

            hash_avx((Goldilocks::Element(&)[CAPACITY])cursor[nextIndex + (pending + extraZeros + i) * CAPACITY], pol_input);
        }

        nextIndex += (pending + extraZeros) * CAPACITY;
        pending = (pending + (arity - 1)) / arity;
        nextN = (pending + (arity - 1)) / arity;
    }
}

template<uint32_t SPONGE_WIDTH_T>
void Poseidon2Goldilocks<SPONGE_WIDTH_T>::merkletree_batch_avx(Goldilocks::Element *tree, Goldilocks::Element *input, uint64_t num_cols, uint64_t num_rows, uint64_t arity, int nThreads, uint64_t dim)
{
    if (num_rows == 0)
    {
        return;
    }
    Goldilocks::Element *cursor = tree;
    // memset(cursor, 0, num_rows * CAPACITY * sizeof(Goldilocks::Element));
    if (nThreads == 0)
        nThreads = omp_get_max_threads();

#pragma omp parallel for num_threads(nThreads)
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

    #pragma omp parallel for num_threads(nThreads)
        for (uint64_t i = 0; i < nextN; i += 4)
        {

            if (nextN - i < 4) {
                Goldilocks::Element pol_input[SPONGE_WIDTH];
                memset(pol_input, 0, SPONGE_WIDTH * sizeof(Goldilocks::Element));
                for( uint32_t j = 0; j < int(nextN - i); j++) {
                    std::memcpy(pol_input, &cursor[nextIndex + (i+j) * SPONGE_WIDTH], SPONGE_WIDTH * sizeof(Goldilocks::Element));
                    hash_avx((Goldilocks::Element(&)[CAPACITY])cursor[nextIndex + (pending + extraZeros + (i + j)) * CAPACITY], pol_input);
                }
            } else {
                Goldilocks::Element pol_input[4*SPONGE_WIDTH];
                memset(pol_input, 0, 4*SPONGE_WIDTH * sizeof(Goldilocks::Element));
                for( uint32_t j = 0; j < 4; j++)
                {
                    std::memcpy(pol_input + j*SPONGE_WIDTH, &cursor[nextIndex + (i+j) * SPONGE_WIDTH], SPONGE_WIDTH * sizeof(Goldilocks::Element));
                }
                hash_batch_avx((Goldilocks::Element(&)[4 * CAPACITY])cursor[nextIndex + (pending + extraZeros + i) * CAPACITY], pol_input);
            }
        }

        nextIndex += (pending + extraZeros) * CAPACITY;
        pending = (pending + (arity - 1)) / arity;
        nextN = (pending + (arity - 1)) / arity;
    }
}
#endif

#ifdef __AVX512__


void Poseidon2Goldilocks::hash_full_result_batch_avx512(Goldilocks::Element *state, const Goldilocks::Element *input) {
    const int length = SPONGE_WIDTH * sizeof(Goldilocks::Element);
    std::memcpy(state, input, 8 * length);
    __m512i st[SPONGE_WIDTH];
    for(uint32_t i = 0; i < SPONGE_WIDTH; i++) {
        Goldilocks::load_avx512(st[i], &(state[i]), SPONGE_WIDTH);
    }
    
    matmul_external_batch_avx512(st);

    for( uint32_t r = 0; r < HALF_N_FULL_ROUNDS; r++)
    {
        pow7add_avx512(st,  &(Poseidon2GoldilocksConstants::C12[r * SPONGE_WIDTH]));
        matmul_external_batch_avx512(st);
    }

    __m512i d[SPONGE_WIDTH];
    for( uint32_t i = 0; i < SPONGE_WIDTH; ++i) {
        d[i] = _mm512_set1_epi64(Poseidon2GoldilocksConstants::D12[i].fe);
    }

    for( uint32_t r = 0; r < N_PARTIAL_ROUNDS; r++)
    {
        __m512i c = _mm512_set1_epi64(Poseidon2GoldilocksConstants::C12[HALF_N_FULL_ROUNDS * SPONGE_WIDTH + r].fe);
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
        pow7add_avx512(st, &(Poseidon2GoldilocksConstants::C12[HALF_N_FULL_ROUNDS * SPONGE_WIDTH + N_PARTIAL_ROUNDS + r * SPONGE_WIDTH]));
        matmul_external_batch_avx512(st);
    }

    for(uint32_t i = 0; i < SPONGE_WIDTH; i++) {
        Goldilocks::store_avx512(&(state[i]), SPONGE_WIDTH, st[i]);
    }
}


void Poseidon2Goldilocks::linear_hash_batch_avx512(Goldilocks::Element *output, Goldilocks::Element *input, uint64_t size)
{
    uint64_t remaining = size;
    Goldilocks::Element state[8*SPONGE_WIDTH];

    if (size <= CAPACITY)
    {
        for(uint64_t i = 0; i < 8; ++i) {
            std::memcpy(&output[i * CAPACITY], &input[i*size], size * sizeof(Goldilocks::Element));
            std::memset(&output[i * CAPACITY + size], 0, (CAPACITY - size) * sizeof(Goldilocks::Element));
        }
        
        return; // no need to hash
    }
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
                memcpy(&state[i*SPONGE_WIDTH + RATE], &state[i*SPONGE_WIDTH], CAPACITY * sizeof(Goldilocks::Element));
            }
        }

        uint64_t n = (remaining < RATE) ? remaining : RATE;
        for(uint64_t i = 0; i < 8; ++i) {
            memset(&state[i*SPONGE_WIDTH + n], 0, (RATE - n) * sizeof(Goldilocks::Element));
            std::memcpy(&state[i * SPONGE_WIDTH], &input[i*size + (size - remaining)], n * sizeof(Goldilocks::Element));
        }
        hash_full_result_batch_avx512(state, state);
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

void Poseidon2Goldilocks::merkletree_batch_avx512(Goldilocks::Element *tree, Goldilocks::Element *input, uint64_t num_cols, uint64_t num_rows, uint64_t arity, int nThreads, uint64_t dim)
{
    if (num_rows == 0)
    {
        return;
    }
    Goldilocks::Element *cursor = tree;
    // memset(cursor, 0, num_rows * CAPACITY * sizeof(Goldilocks::Element));
    if (nThreads == 0)
        nThreads = omp_get_max_threads();

#pragma omp parallel for num_threads(nThreads)
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

    #pragma omp parallel for num_threads(nThreads)
        for (uint64_t i = 0; i < nextN; i += 8)
        {

            if (nextN - i < 8) {
                Goldilocks::Element pol_input[SPONGE_WIDTH];
                memset(pol_input, 0, SPONGE_WIDTH * sizeof(Goldilocks::Element));
                for( uint32_t j = 0; j < int(nextN - i); j++) {
                    std::memcpy(pol_input, &cursor[nextIndex + (i+j) * SPONGE_WIDTH], SPONGE_WIDTH * sizeof(Goldilocks::Element));
                    hash_avx((Goldilocks::Element(&)[CAPACITY])cursor[nextIndex + (pending + extraZeros + (i + j)) * CAPACITY], pol_input);
                }
            } else {
                Goldilocks::Element pol_input[8*SPONGE_WIDTH];
                memset(pol_input, 0, 8*SPONGE_WIDTH * sizeof(Goldilocks::Element));
                for( uint32_t j = 0; j < 8; j++)
                {
                    std::memcpy(pol_input + j*SPONGE_WIDTH, &cursor[nextIndex + (i+j) * SPONGE_WIDTH], SPONGE_WIDTH * sizeof(Goldilocks::Element));
                }
                hash_batch_avx512((Goldilocks::Element(&)[8 * CAPACITY])cursor[nextIndex + (pending + extraZeros + i) * CAPACITY], pol_input);
            }
        }

        nextIndex += (pending + extraZeros) * CAPACITY;
        pending = (pending + (arity - 1)) / arity;
        nextN = (pending + (arity - 1)) / arity;
    }
}

// void Poseidon2Goldilocks::hash_full_result_avx512(Goldilocks::Element *state, const Goldilocks::Element *input)
// {

//     const int length = 2 * SPONGE_WIDTH * sizeof(Goldilocks::Element);
//     std::memcpy(state, input, length);
//     __m512i st0, st1, st2;
//     Goldilocks::load_avx512(st0, &(state[0]));
//     Goldilocks::load_avx512(st1, &(state[8]));
//     Goldilocks::load_avx512(st2, &(state[16]));
//     add_avx512_small(st0, st1, st2, &(Poseidon2GoldilocksConstants::C12[0]));

//     for( uint32_t r = 0; r < HALF_N_FULL_ROUNDS - 1; r++)
//     {
//         pow7_avx512(st0, st1, st2);
//         add_avx512_small(st0, st1, st2, &(Poseidon2GoldilocksConstants::C12[(r + 1) * SPONGE_WIDTH])); // rick
//         Goldilocks::mmult_avx512_8(st0, st1, st2, &(Poseidon2GoldilocksConstants::M_[0]));
//     }
//     pow7_avx512(st0, st1, st2);
//     add_avx512(st0, st1, st2, &(Poseidon2GoldilocksConstants::C12[(HALF_N_FULL_ROUNDS * SPONGE_WIDTH)]));
//     Goldilocks::mmult_avx512(st0, st1, st2, &(Poseidon2GoldilocksConstants::P_[0]));

//     Goldilocks::store_avx512(&(state[0]), st0);
//     Goldilocks::Element s04_[2] = {state[0], state[4]};
//     Goldilocks::Element s04[2];

//     __m512i mask = _mm512_set_epi64(0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF, 0, 0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF, 0); // rick, not better to define where u use it?
//     for( uint32_t r = 0; r < N_PARTIAL_ROUNDS; r++)
//     {
//         s04[0] = s04_[0];
//         s04[1] = s04_[1];
//         pow7(s04[0]);
//         pow7(s04[1]);
//         s04[0] = s04[0] + Poseidon2GoldilocksConstants::C12[(HALF_N_FULL_ROUNDS + 1) * SPONGE_WIDTH + r];
//         s04[1] = s04[1] + Poseidon2GoldilocksConstants::C12[(HALF_N_FULL_ROUNDS + 1) * SPONGE_WIDTH + r];
//         s04_[0] = s04[0] * Poseidon2GoldilocksConstants::S[(SPONGE_WIDTH * 2 - 1) * r];
//         s04_[1] = s04[1] * Poseidon2GoldilocksConstants::S[(SPONGE_WIDTH * 2 - 1) * r];
//         st0 = _mm512_and_si512(st0, mask); // rick, do we need a new one?
//         Goldilocks::Element aux[2];
//         Goldilocks::D12ot_avx512(aux, st0, st1, st2, &(Poseidon2GoldilocksConstants::S[(SPONGE_WIDTH * 2 - 1) * r]));
//         s04_[0] = s04_[0] + aux[0];
//         s04_[1] = s04_[1] + aux[1];
//         __m512i scalar1 = _mm512_set_epi64(s04[1].fe, s04[1].fe, s04[1].fe, s04[1].fe, s04[0].fe, s04[0].fe, s04[0].fe, s04[0].fe);
//         __m512i w0, w1, w2;

//         const Goldilocks::Element *auxS = &(Poseidon2GoldilocksConstants::S[(SPONGE_WIDTH * 2 - 1) * r + SPONGE_WIDTH - 1]);
//         __m512i s0 = _mm512_set4_epi64(auxS[3].fe, auxS[2].fe, auxS[1].fe, auxS[0].fe);
//         __m512i s1 = _mm512_set4_epi64(auxS[7].fe, auxS[6].fe, auxS[5].fe, auxS[4].fe);
//         __m512i s2 = _mm512_set4_epi64(auxS[11].fe, auxS[10].fe, auxS[9].fe, auxS[8].fe);

//         Goldilocks::mult_avx512(w0, scalar1, s0);
//         Goldilocks::mult_avx512(w1, scalar1, s1);
//         Goldilocks::mult_avx512(w2, scalar1, s2);
//         Goldilocks::add_avx512(st0, st0, w0);
//         Goldilocks::add_avx512(st1, st1, w1);
//         Goldilocks::add_avx512(st2, st2, w2);
//         s04[0] = s04[0] + Poseidon2GoldilocksConstants::S[(SPONGE_WIDTH * 2 - 1) * r + SPONGE_WIDTH - 1];
//         s04[1] = s04[1] + Poseidon2GoldilocksConstants::S[(SPONGE_WIDTH * 2 - 1) * r + SPONGE_WIDTH - 1];
//     }

//     Goldilocks::store_avx512(&(state[0]), st0);
//     state[0] = s04_[0];
//     state[4] = s04_[1];
//     Goldilocks::load_avx512(st0, &(state[0]));

//     for( uint32_t r = 0; r < HALF_N_FULL_ROUNDS - 1; r++)
//     {
//         pow7_avx512(st0, st1, st2);
//         add_avx512_small(st0, st1, st2, &(Poseidon2GoldilocksConstants::C12[(HALF_N_FULL_ROUNDS + 1) * SPONGE_WIDTH + N_PARTIAL_ROUNDS + r * SPONGE_WIDTH]));
//         Goldilocks::mmult_avx512_8(st0, st1, st2, &(Poseidon2GoldilocksConstants::M_[0]));
//     }
//     pow7_avx512(st0, st1, st2);
//     Goldilocks::mmult_avx512_8(st0, st1, st2, &(Poseidon2GoldilocksConstants::M_[0]));

//     Goldilocks::store_avx512(&(state[0]), st0);
//     Goldilocks::store_avx512(&(state[8]), st1);
//     Goldilocks::store_avx512(&(state[16]), st2);
// }
// void Poseidon2Goldilocks::linear_hash_avx512(Goldilocks::Element *output, Goldilocks::Element *input, uint64_t size)
// {
//     uint64_t remaining = size;
//     Goldilocks::Element state[2 * SPONGE_WIDTH];

//     if (size <= CAPACITY)
//     {
//         std::memcpy(output, input, size * sizeof(Goldilocks::Element));
//         std::memset(output + size, 0, (CAPACITY - size) * sizeof(Goldilocks::Element));
//         std::memcpy(output + CAPACITY, input + size, size * sizeof(Goldilocks::Element));
//         std::memset(output + CAPACITY + size, 0, (CAPACITY - size) * sizeof(Goldilocks::Element));
//         return; // no need to hash
//     }
//     while (remaining)
//     {
//         if (remaining == size)
//         {
//             memset(state + 2 * RATE, 0, 2 * CAPACITY * sizeof(Goldilocks::Element));
//         }
//         else
//         {
//             std::memcpy(state + 2 * RATE, state, 2 * CAPACITY * sizeof(Goldilocks::Element));
//         }

//         uint64_t n = (remaining < RATE) ? remaining : RATE;
//         memset(state, 0, 2 * RATE * sizeof(Goldilocks::Element));

//         if (n <= 4)
//         {
//             std::memcpy(state, input + (size - remaining), n * sizeof(Goldilocks::Element));
//             std::memcpy(state + 4, input + size + (size - remaining), n * sizeof(Goldilocks::Element));
//         }
//         else
//         {
//             std::memcpy(state, input + (size - remaining), 4 * sizeof(Goldilocks::Element));
//             std::memcpy(state + 4, input + size + (size - remaining), 4 * sizeof(Goldilocks::Element));
//             std::memcpy(state + 8, input + (size - remaining) + 4, (n - 4) * sizeof(Goldilocks::Element));
//             std::memcpy(state + 12, input + size + (size - remaining) + 4, (n - 4) * sizeof(Goldilocks::Element));
//         }

//         hash_full_result_avx512(state, state);
//         remaining -= n;
//     }
//     if (size > 0)
//     {
//         std::memcpy(output, state, 2 * CAPACITY * sizeof(Goldilocks::Element));
//     }
//     else
//     {
//         memset(output, 0, 2 * CAPACITY * sizeof(Goldilocks::Element));
//     }
// }
// void Poseidon2Goldilocks::merkletree_avx512(Goldilocks::Element *tree, Goldilocks::Element *input, uint64_t num_cols, uint64_t num_rows, int nThreads, uint64_t dim)
// {
//     if (num_rows == 0)
//     {
//         return;
//     }
//     Goldilocks::Element *cursor = tree;
//     // memset(cursor, 0, num_rows * CAPACITY * sizeof(Goldilocks::Element));
//     if (nThreads == 0)
//         nThreads = omp_get_max_threads();

// #pragma omp parallel for num_threads(nThreads)
//     for (uint64_t i = 0; i < num_rows; i += 2)
//     {
//         linear_hash_avx512(&cursor[i * CAPACITY], &input[i * num_cols * dim], num_cols * dim);
//     }

//     // Build the merkle tree
//     uint64_t pending = num_rows;
//     uint64_t nextN = floor((pending - 1) / 2) + 1;
//     uint64_t nextIndex = 0;

//     while (pending > 1)
//     {
// #pragma omp parallel for num_threads(nThreads)
//         for (uint64_t i = 0; i < nextN; i++)
//         {
//             Goldilocks::Element pol_input[SPONGE_WIDTH];
//             memset(pol_input, 0, SPONGE_WIDTH * sizeof(Goldilocks::Element));
//             std::memcpy(pol_input, &cursor[nextIndex + i * RATE], RATE * sizeof(Goldilocks::Element));
//             hash((Goldilocks::Element(&)[CAPACITY])cursor[nextIndex + (pending + i) * CAPACITY], pol_input);
//         }
//         nextIndex += pending * CAPACITY;
//         pending = pending / 2;
//         nextN = floor((pending - 1) / 2) + 1;
//     }
// }

#endif

// Explicit template instantiations
template class Poseidon2Goldilocks<4>;
template class Poseidon2Goldilocks<8>;
template class Poseidon2Goldilocks<12>;  
template class Poseidon2Goldilocks<16>;