#include "poseidon2_goldilocks.hpp"
#include <math.h> /* floor */
#include "merklehash_goldilocks.hpp"

void Poseidon2Goldilocks::hash_full_result_seq(Goldilocks::Element *state, const Goldilocks::Element *input)
{
    const int length = SPONGE_WIDTH * sizeof(Goldilocks::Element);
    std::memcpy(state, input, length);

    matmul_external_(state);
  
    for (int r = 0; r < HALF_N_FULL_ROUNDS; r++)
    {
        pow7add_(state, &(Poseidon2GoldilocksConstants::C[r * SPONGE_WIDTH]));
        matmul_external_(state);
    }

    for (int r = 0; r < N_PARTIAL_ROUNDS; r++)
    {
        state[0] = state[0] + Poseidon2GoldilocksConstants::C[HALF_N_FULL_ROUNDS * SPONGE_WIDTH + r];
        pow7(state[0]);
        Goldilocks::Element sum_ = Goldilocks::zero();
        add_(sum_, state);
        prodadd_(state, Poseidon2GoldilocksConstants::D, sum_);
    }

    for (int r = 0; r < HALF_N_FULL_ROUNDS; r++)
    {
        pow7add_(state, &(Poseidon2GoldilocksConstants::C[HALF_N_FULL_ROUNDS * SPONGE_WIDTH + N_PARTIAL_ROUNDS + r * SPONGE_WIDTH]));
        matmul_external_(state);
    }
}
void Poseidon2Goldilocks::linear_hash_seq(Goldilocks::Element *output, Goldilocks::Element *input, uint64_t size)
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
            std::memcpy(state + RATE, state, CAPACITY * sizeof(Goldilocks::Element));
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
void Poseidon2Goldilocks::merkletree_seq(Goldilocks::Element *tree, Goldilocks::Element *input, uint64_t num_cols, uint64_t num_rows, uint64_t arity, int nThreads, uint64_t dim)
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
void Poseidon2Goldilocks::merkletree_batch_seq(Goldilocks::Element *tree, Goldilocks::Element *input, uint64_t num_cols, uint64_t num_rows, uint64_t arity, uint64_t batch_size, int nThreads, uint64_t dim)
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
    for (int i = 0; i < nThreads; ++i)
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

#ifdef __AVX2__

void Poseidon2Goldilocks::hash_full_result_batch_avx(Goldilocks::Element *state, const Goldilocks::Element *input) {
    const int length = SPONGE_WIDTH * sizeof(Goldilocks::Element);
    std::memcpy(state, input, 4 * length);
    __m256i st[SPONGE_WIDTH];
    for(int i = 0; i < SPONGE_WIDTH; i++) {
        Goldilocks::load_avx(st[i], &(state[i]), SPONGE_WIDTH);
    }
    
    matmul_external_batch_avx(st);

    for (int r = 0; r < HALF_N_FULL_ROUNDS; r++)
    {
        pow7add_avx(st,  &(Poseidon2GoldilocksConstants::C[r * SPONGE_WIDTH]));
        matmul_external_batch_avx(st);
    }

    __m256i d[SPONGE_WIDTH];
    for (int i = 0; i < SPONGE_WIDTH; ++i) {
        d[i] = _mm256_set1_epi64x(Poseidon2GoldilocksConstants::D[i].fe);
    }

    for (int r = 0; r < N_PARTIAL_ROUNDS; r++)
    {
        __m256i c = _mm256_set1_epi64x(Poseidon2GoldilocksConstants::C[HALF_N_FULL_ROUNDS * SPONGE_WIDTH + r].fe);
        Goldilocks::add_avx(st[0], st[0], c);
        element_pow7_avx(st[0]);
        __m256i sum = _mm256_set1_epi64x(Goldilocks::zero().fe);
        for (int i = 0; i < SPONGE_WIDTH; ++i)
        {
            Goldilocks::add_avx(sum, sum, st[i]);
        }
        for (int i = 0; i < SPONGE_WIDTH; ++i)
        {
            Goldilocks::mult_avx(st[i], st[i], d[i]);
            Goldilocks::add_avx(st[i], st[i], sum);
        }
    }

    for (int r = 0; r < HALF_N_FULL_ROUNDS; r++)
    {
        pow7add_avx(st, &(Poseidon2GoldilocksConstants::C[HALF_N_FULL_ROUNDS * SPONGE_WIDTH + N_PARTIAL_ROUNDS + r * SPONGE_WIDTH]));
        matmul_external_batch_avx(st);
    }

    for(int i = 0; i < SPONGE_WIDTH; i++) {
        Goldilocks::store_avx(&(state[i]), SPONGE_WIDTH, st[i]);
    }
}

void Poseidon2Goldilocks::hash_full_result_avx(Goldilocks::Element *state, const Goldilocks::Element *input)
{
    const int length = SPONGE_WIDTH * sizeof(Goldilocks::Element);
    std::memcpy(state, input, length);
    __m256i st0, st1, st2;
    Goldilocks::load_avx(st0, &(state[0]));
    Goldilocks::load_avx(st1, &(state[4]));
    Goldilocks::load_avx(st2, &(state[8]));

    matmul_external_avx(st0, st1, st2);
    
    for (int r = 0; r < HALF_N_FULL_ROUNDS; r++)
    {
        add_avx_small(st0, st1, st2, &(Poseidon2GoldilocksConstants::C[r * SPONGE_WIDTH]));
        pow7_avx(st0, st1, st2);
        matmul_external_avx(st0, st1, st2);
    }
    
    Goldilocks::store_avx(&(state[0]), st0);
    Goldilocks::Element state0_ = state[0];

    __m256i d0, d1, d2;
    Goldilocks::load_avx(d0, &(Poseidon2GoldilocksConstants::D[0]));
    Goldilocks::load_avx(d1, &(Poseidon2GoldilocksConstants::D[4]));
    Goldilocks::load_avx(d2, &(Poseidon2GoldilocksConstants::D[8]));

    __m256i part_sum;
    Goldilocks::Element partial_sum[4];
    Goldilocks::Element aux = state0_;
    for (int r = 0; r < N_PARTIAL_ROUNDS; r++)
    {
        Goldilocks::add_avx(part_sum, st1, st2);
        Goldilocks::add_avx(part_sum, part_sum, st0);
        Goldilocks::store_avx(partial_sum, part_sum);
        Goldilocks::Element sum = partial_sum[0] + partial_sum[1] + partial_sum[2] + partial_sum[3];
        sum = sum - aux;

        state0_ = state0_ + Poseidon2GoldilocksConstants::C[HALF_N_FULL_ROUNDS * SPONGE_WIDTH + r];
        pow7(state0_);

        sum = sum + state0_;    
            
        __m256i scalar1 = _mm256_set1_epi64x(sum.fe);
        Goldilocks::mult_avx(st0, st0, d0);
        Goldilocks::mult_avx(st1, st1, d1);
        Goldilocks::mult_avx(st2, st2, d2);
        Goldilocks::add_avx(st0, st0, scalar1);
        Goldilocks::add_avx(st1, st1, scalar1);
        Goldilocks::add_avx(st2, st2, scalar1);
        state0_ = state0_ * Poseidon2GoldilocksConstants::D[0] + sum;
        aux = aux * Poseidon2GoldilocksConstants::D[0] + sum;
    }

    Goldilocks::store_avx(&(state[0]), st0);
    state[0] = state0_;
    Goldilocks::load_avx(st0, &(state[0]));

    for (int r = 0; r < HALF_N_FULL_ROUNDS; r++)
    {
        add_avx_small(st0, st1, st2, &(Poseidon2GoldilocksConstants::C[HALF_N_FULL_ROUNDS * SPONGE_WIDTH + N_PARTIAL_ROUNDS + r * SPONGE_WIDTH]));
        pow7_avx(st0, st1, st2);
        
        matmul_external_avx(st0, st1, st2);
    }
    
    Goldilocks::store_avx(&(state[0]), st0);
    Goldilocks::store_avx(&(state[4]), st1);
    Goldilocks::store_avx(&(state[8]), st2);
}

void Poseidon2Goldilocks::linear_hash_avx(Goldilocks::Element *output, Goldilocks::Element *input, uint64_t size)
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
            std::memcpy(state + RATE, state, CAPACITY * sizeof(Goldilocks::Element));
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

void Poseidon2Goldilocks::linear_hash_batch_avx(Goldilocks::Element *output, Goldilocks::Element *input, uint64_t size)
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

void Poseidon2Goldilocks::merkletree_avx(Goldilocks::Element *tree, Goldilocks::Element *input, uint64_t num_cols, uint64_t num_rows, uint64_t arity, int nThreads, uint64_t dim)
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
void Poseidon2Goldilocks::merkletree_batch_avx(Goldilocks::Element *tree, Goldilocks::Element *input, uint64_t num_cols, uint64_t num_rows, uint64_t arity, int nThreads, uint64_t dim)
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
                for (int j = 0; j < int(nextN - i); j++) {
                    std::memcpy(pol_input, &cursor[nextIndex + (i+j) * SPONGE_WIDTH], SPONGE_WIDTH * sizeof(Goldilocks::Element));
                    hash_avx((Goldilocks::Element(&)[CAPACITY])cursor[nextIndex + (pending + extraZeros + (i + j)) * CAPACITY], pol_input);
                }
            } else {
                Goldilocks::Element pol_input[4*SPONGE_WIDTH];
                memset(pol_input, 0, 4*SPONGE_WIDTH * sizeof(Goldilocks::Element));
                for (int j = 0; j < 4; j++)
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
    for(int i = 0; i < SPONGE_WIDTH; i++) {
        Goldilocks::load_avx512(st[i], &(state[i]), SPONGE_WIDTH);
    }
    
    matmul_external_batch_avx512(st);

    for (int r = 0; r < HALF_N_FULL_ROUNDS; r++)
    {
        pow7add_avx512(st,  &(Poseidon2GoldilocksConstants::C[r * SPONGE_WIDTH]));
        matmul_external_batch_avx512(st);
    }

    __m512i d[SPONGE_WIDTH];
    for (int i = 0; i < SPONGE_WIDTH; ++i) {
        d[i] = _mm512_set1_epi64(Poseidon2GoldilocksConstants::D[i].fe);
    }

    for (int r = 0; r < N_PARTIAL_ROUNDS; r++)
    {
        __m512i c = _mm512_set1_epi64(Poseidon2GoldilocksConstants::C[HALF_N_FULL_ROUNDS * SPONGE_WIDTH + r].fe);
        Goldilocks::add_avx512(st[0], st[0], c);
        element_pow7_avx512(st[0]);
        __m512i sum = _mm512_set1_epi64(Goldilocks::zero().fe);
        for (int i = 0; i < SPONGE_WIDTH; ++i)
        {
            Goldilocks::add_avx512(sum, sum, st[i]);
        }
        for (int i = 0; i < SPONGE_WIDTH; ++i)
        {
            Goldilocks::mult_avx512(st[i], st[i], d[i]);
            Goldilocks::add_avx512(st[i], st[i], sum);
        }
    }

    for (int r = 0; r < HALF_N_FULL_ROUNDS; r++)
    {
        pow7add_avx512(st, &(Poseidon2GoldilocksConstants::C[HALF_N_FULL_ROUNDS * SPONGE_WIDTH + N_PARTIAL_ROUNDS + r * SPONGE_WIDTH]));
        matmul_external_batch_avx512(st);
    }

    for(int i = 0; i < SPONGE_WIDTH; i++) {
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
                for (int j = 0; j < int(nextN - i); j++) {
                    std::memcpy(pol_input, &cursor[nextIndex + (i+j) * SPONGE_WIDTH], SPONGE_WIDTH * sizeof(Goldilocks::Element));
                    hash_avx((Goldilocks::Element(&)[CAPACITY])cursor[nextIndex + (pending + extraZeros + (i + j)) * CAPACITY], pol_input);
                }
            } else {
                Goldilocks::Element pol_input[8*SPONGE_WIDTH];
                memset(pol_input, 0, 8*SPONGE_WIDTH * sizeof(Goldilocks::Element));
                for (int j = 0; j < 8; j++)
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
//     add_avx512_small(st0, st1, st2, &(Poseidon2GoldilocksConstants::C[0]));

//     for (int r = 0; r < HALF_N_FULL_ROUNDS - 1; r++)
//     {
//         pow7_avx512(st0, st1, st2);
//         add_avx512_small(st0, st1, st2, &(Poseidon2GoldilocksConstants::C[(r + 1) * SPONGE_WIDTH])); // rick
//         Goldilocks::mmult_avx512_8(st0, st1, st2, &(Poseidon2GoldilocksConstants::M_[0]));
//     }
//     pow7_avx512(st0, st1, st2);
//     add_avx512(st0, st1, st2, &(Poseidon2GoldilocksConstants::C[(HALF_N_FULL_ROUNDS * SPONGE_WIDTH)]));
//     Goldilocks::mmult_avx512(st0, st1, st2, &(Poseidon2GoldilocksConstants::P_[0]));

//     Goldilocks::store_avx512(&(state[0]), st0);
//     Goldilocks::Element s04_[2] = {state[0], state[4]};
//     Goldilocks::Element s04[2];

//     __m512i mask = _mm512_set_epi64(0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF, 0, 0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF, 0); // rick, not better to define where u use it?
//     for (int r = 0; r < N_PARTIAL_ROUNDS; r++)
//     {
//         s04[0] = s04_[0];
//         s04[1] = s04_[1];
//         pow7(s04[0]);
//         pow7(s04[1]);
//         s04[0] = s04[0] + Poseidon2GoldilocksConstants::C[(HALF_N_FULL_ROUNDS + 1) * SPONGE_WIDTH + r];
//         s04[1] = s04[1] + Poseidon2GoldilocksConstants::C[(HALF_N_FULL_ROUNDS + 1) * SPONGE_WIDTH + r];
//         s04_[0] = s04[0] * Poseidon2GoldilocksConstants::S[(SPONGE_WIDTH * 2 - 1) * r];
//         s04_[1] = s04[1] * Poseidon2GoldilocksConstants::S[(SPONGE_WIDTH * 2 - 1) * r];
//         st0 = _mm512_and_si512(st0, mask); // rick, do we need a new one?
//         Goldilocks::Element aux[2];
//         Goldilocks::dot_avx512(aux, st0, st1, st2, &(Poseidon2GoldilocksConstants::S[(SPONGE_WIDTH * 2 - 1) * r]));
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

//     for (int r = 0; r < HALF_N_FULL_ROUNDS - 1; r++)
//     {
//         pow7_avx512(st0, st1, st2);
//         add_avx512_small(st0, st1, st2, &(Poseidon2GoldilocksConstants::C[(HALF_N_FULL_ROUNDS + 1) * SPONGE_WIDTH + N_PARTIAL_ROUNDS + r * SPONGE_WIDTH]));
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